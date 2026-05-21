# chatbot_agent.py  (v2 — hybrid retrieval, importance weighting, large-doc safe)
#
# Key improvements over v1:
#   1. Hybrid retriever: BM25 (keyword/sparse) + ChromaDB (semantic/dense)
#      combined with EnsembleRetriever.  Financial queries like "revenue FY24"
#      now hit both exact-keyword and semantic matches.
#   2. Importance-biased retrieval: chunks from flagged "important" pages
#      (financials, risk factors, objects of issue) are up-weighted.
#   3. k=10 with MMR (Maximal Marginal Relevance) to reduce redundant chunks.
#   4. Section-aware prompting: the LLM is told what section each chunk came from.
#   5. Table-aware response: if retrieved chunks contain TABLE chunks the LLM is
#      instructed to read them as markdown tables and extract numeric data.
#   6. Progress bar wired to the new document_processor callback.

import os
import shutil
from typing import Optional

# Vector store & embeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# BM25 sparse retriever
try:
    from langchain_community.retrievers import BM25Retriever
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    print("⚠️  langchain-community BM25Retriever not available — falling back to dense-only retrieval.")

# Ensemble (hybrid) retriever
try:
    from langchain.retrievers import EnsembleRetriever
    ENSEMBLE_AVAILABLE = True
except ImportError:
    ENSEMBLE_AVAILABLE = False

# LangChain chains & prompts
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq

# Our upgraded document processor
from document_processor import process_pdf_with_pdfplumber

# RHP downloader (unchanged)
from rhp_agent import find_and_download_all_pdfs

from langchain_pinecone import PineconeVectorStore
from cloud_utils import namespace_exists
PINECONE_INDEX_NAME = "ipo-rhp-index"
# ── configuration ────────────────────────────────────────────────────────────
LLM_MODEL             = "llama-3.3-70b-versatile"
EMBEDDING_MODEL       = "all-MiniLM-L6-v2"
VECTORSTORE_BASE_PATH = "ipo_vectorstores"

# Retrieval settings
DENSE_K  = 10   # top-k from ChromaDB (MMR)
SPARSE_K = 10   # top-k from BM25
FINAL_K  = 8    # docs passed to the LLM after ensemble merge
# ─────────────────────────────────────────────────────────────────────────────


def _safe_name(ipo_name: str) -> str:
    return "".join(c for c in ipo_name if c.isalnum() or c in " -_").rstrip()


# ── main processing function ─────────────────────────────────────────────────

def process_and_store_document(ipo_name: str, st_progress_bar=None):
    """
    Downloads ALL RHP/DRHP PDFs for the IPO (main prospectus + any supplementary
    docs above 2 MB), parses each with pdfplumber, and embeds everything into a
    single ChromaDB vectorstore.

    - Skips addenda / cover pages (< 2 MB).
    - Processes files from largest to smallest so the main RHP is indexed first.
    - Progress bar covers: download → parse (60%) → embed (40%).
    """
    # persist_directory = os.path.join(VECTORSTORE_BASE_PATH, _safe_name(ipo_name))
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # ── load from cache ───────────────────────────────────────────────────────
    if namespace_exists(ipo_name):
        print(f"✅ Vector store for '{ipo_name}' found in Pinecone. Loading…")
        return PineconeVectorStore(index_name="ipo-rhp-index", embedding=embeddings, namespace=ipo_name)

    print(f"🚧 No cache for '{ipo_name}'. Starting full pipeline.")

    # ── download all PDFs ─────────────────────────────────────────────────────
    if st_progress_bar:
        st_progress_bar.progress(0.02, text="Searching for RHP/DRHP documents…")

    pdf_paths = find_and_download_all_pdfs(ipo_name)
    if not pdf_paths:
        print(f"❌ No documents found for {ipo_name}.")
        return None

    print(f"📚 Processing {len(pdf_paths)} document(s).")

    # ── parse every PDF ───────────────────────────────────────────────────────
    all_docs = []
    for doc_idx, pdf_path in enumerate(pdf_paths):
        doc_label = os.path.basename(pdf_path)
        size_mb   = os.path.getsize(pdf_path) / 1_048_576

        print(f"\n[{doc_idx+1}/{len(pdf_paths)}] Parsing: {doc_label} ({size_mb:.1f} MB)")

        # Progress bar: parsing phase covers 0.05 → 0.60
        parse_start = 0.05 + doc_idx * (0.55 / len(pdf_paths))
        parse_end   = 0.05 + (doc_idx + 1) * (0.55 / len(pdf_paths))

        def _progress(fraction: float, msg: str, _s=parse_start, _e=parse_end):
            if st_progress_bar:
                overall = _s + fraction * (_e - _s)
                st_progress_bar.progress(overall, text=f"[Doc {doc_idx+1}/{len(pdf_paths)}] {msg}")
            print(f"   [{int(fraction*100):3d}%] {msg}")

        docs = process_pdf_with_pdfplumber(
            pdf_path,
            progress_callback=_progress,
            max_pages=900,
        )

        if docs:
            # Tag each chunk with which source file it came from
            for d in docs:
                d.metadata["source_file"] = doc_label
            all_docs.extend(docs)
            tables = sum(1 for d in docs if d.metadata.get("type") == "table")
            print(f"   → {len(docs)} chunks ({tables} tables) from {doc_label}")
        else:
            print(f"   ⚠️  No content extracted from {doc_label}.")

    if not all_docs:
        print(f"❌ No content extracted from any document for {ipo_name}.")
        return None

    total_chunks  = len(all_docs)
    total_tables  = sum(1 for d in all_docs if d.metadata.get("type") == "table")
    print(f"\n✅ Total: {total_chunks} chunks ({total_tables} tables) across {len(pdf_paths)} file(s).")

    if st_progress_bar:
        st_progress_bar.progress(0.60, text=f"Embedding {total_chunks} chunks into ChromaDB…")

    # ── embed in batches ──────────────────────────────────────────────────────
    embeddings    = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    batch_size    = 64
    vectorstore   = None

    for i in range(0, total_chunks, batch_size):
        batch = all_docs[i : i + batch_size]
        if vectorstore is None:
            # Pinecone handles batching internally via the LangChain wrapper
            vectorstore = PineconeVectorStore.from_documents(
                documents=all_docs,
                embedding=embeddings,
                index_name=PINECONE_INDEX_NAME,
                namespace=ipo_name
            )
        else:
            vectorstore.add_documents(documents=batch)

        if st_progress_bar:
            embed_frac = min((i + batch_size) / total_chunks, 1.0)
            overall    = 0.60 + embed_frac * 0.38
            st_progress_bar.progress(
                min(overall, 0.98),
                text=f"Embedding {min(i + batch_size, total_chunks)}/{total_chunks} chunks…",
            )

    print(f"✅ Embedding complete! Data synced to cloud.")
    return vectorstore


# ── RAG chain factory ────────────────────────────────────────────────────────

def create_rag_chain(vectorstore: PineconeVectorStore, llm_provider: str = "groq"):
    """
    Builds a conversational RAG chain with:
      • Hybrid BM25 + semantic dense retriever
      • MMR diversity in the dense leg
      • History-aware question contextualisation
      • Financial / table-aware system prompt
    """
    # ── LLM — Groq or HuggingFace ───────────────────────────────────────────
    _provider = llm_provider.lower()
    if _provider == "hf":
        try:
            from langchain_community.llms import HuggingFaceHub
            hf_model = os.environ.get("LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
            hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN", "")
            llm = HuggingFaceHub(
                repo_id=hf_model,
                huggingfacehub_api_token=hf_token,
                model_kwargs={"temperature": 0.1, "max_new_tokens": 1024},
            )
            print(f"Using HuggingFace model: {hf_model}")
        except Exception as exc:
            print(f"HuggingFace init failed ({exc}), falling back to Groq.")
            llm = ChatGroq(temperature=0, model_name=LLM_MODEL)
    else:
        llm = ChatGroq(temperature=0, model_name=LLM_MODEL)

    # ── dense retriever with MMR ─────────────────────────────────────────────
    dense_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": DENSE_K, "fetch_k": DENSE_K * 3, "lambda_mult": 0.6},
    )

    # ── BM25 sparse retriever ────────────────────────────────────────────────
    # We need all stored documents to initialise BM25.
    if BM25_AVAILABLE and ENSEMBLE_AVAILABLE:
        try:
            all_docs_result = vectorstore.get(include=["documents", "metadatas"])
            bm25_docs = [
                __import__("langchain_core.documents", fromlist=["Document"]).Document(
                    page_content=pc, metadata=md
                )
                for pc, md in zip(
                    all_docs_result.get("documents", []),
                    all_docs_result.get("metadatas", []),
                )
            ]
            bm25_retriever = BM25Retriever.from_documents(bm25_docs)
            bm25_retriever.k = SPARSE_K

            retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, dense_retriever],
                weights=[0.4, 0.6],   # lean toward semantic but keep keyword signal
            )
            print("✅ Using hybrid BM25 + semantic retriever.")
        except Exception as e:
            print(f"⚠️  Could not build BM25 retriever ({e}). Falling back to dense only.")
            retriever = dense_retriever
    else:
        retriever = dense_retriever
        print("ℹ️  Using dense-only retriever (BM25 libraries not installed).")

    # ── contextualise-question prompt (history-aware) ────────────────────────
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given the conversation history and a follow-up question, "
         "rewrite the question as a fully self-contained standalone question. "
         "Do NOT answer it. If the question is already standalone, return it unchanged."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # ── answer-generation prompt ─────────────────────────────────────────────
    qa_system_prompt = """You are a senior financial analyst assistant specialising in Indian IPOs.
You answer questions using ONLY the excerpts from the company's Red Herring Prospectus (RHP) or DRHP provided below.

Guidelines:
- If a chunk is labelled [TABLE — page N], treat it as a markdown table and extract exact numbers.
- Cite the page number when quoting a figure (e.g. "Revenue was ₹120 Cr (page 312)").
- If multiple chunks contain conflicting numbers, state the discrepancy and cite both pages.
- If the information is genuinely not present in the provided excerpts, say so clearly.
  Do NOT invent numbers or draw on external knowledge.
- For financial metrics (revenue, PAT, EBITDA, margins), always include the fiscal year / period.
- Respond in clear, structured prose. Use bullet points for lists of risks or objects of issue.

CONTEXT FROM RHP:
{context}"""

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain


# ── CLI entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("--- IPO Analysis Chatbot (v2) ---")
    ipo_to_analyze = input("Enter IPO name (e.g. 'Tata Technologies'): ").strip()

    vectorstore = process_and_store_document(ipo_to_analyze)
    if not vectorstore:
        print("Failed to load document. Exiting.")
        exit(1)

    qa_chain = create_rag_chain(vectorstore)
    chat_history = []

    print("\n✅ Chatbot ready. Type 'exit' to quit.\n")
    while True:
        question = input("Your Question: ").strip()
        if question.lower() == "exit":
            break
        result = qa_chain.invoke({"input": question, "chat_history": chat_history})
        answer = result["answer"]
        print(f"\n--- Answer ---\n{answer}\n--------------\n")
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=answer))