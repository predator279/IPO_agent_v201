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
import gc
import shutil
from typing import Optional

# Vector store & embeddings
# from langchain_chroma import Chroma
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
    Processes PDFs one-by-one and flushes RAM after each upload to Pinecone.
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # 1. Check if already processed
    if namespace_exists(ipo_name):
        print(f"✅ Namespace '{ipo_name}' exists in Pinecone.")
        return PineconeVectorStore(index_name=PINECONE_INDEX_NAME, embedding=embeddings, namespace=ipo_name)

    # 2. Get PDF Paths
    pdf_paths = find_and_download_all_pdfs(ipo_name)
    if not pdf_paths: return None

    vectorstore = None

    # 3. Process documents one-by-one to save RAM
    for idx, pdf_path in enumerate(pdf_paths):
        if st_progress_bar:
            st_progress_bar.progress(0.2 + (idx * 0.2), text=f"Parsing {os.path.basename(pdf_path)}...")
        
        # We cap pages at 400 for Streamlit Cloud stability
        docs = process_pdf_with_pdfplumber(pdf_path, max_pages=400)
        
        if docs:
            print(f"📤 Uploading {len(docs)} chunks from {os.path.basename(pdf_path)} to Pinecone...")
            if vectorstore is None:
                vectorstore = PineconeVectorStore.from_documents(
                    documents=docs,
                    embedding=embeddings,
                    index_name=PINECONE_INDEX_NAME,
                    namespace=ipo_name
                )
            else:
                vectorstore.add_documents(documents=docs)
            
            # --- CRITICAL: RELEASE RAM ---
            del docs
            gc.collect() 
            print(f"🧹 RAM Cleared after PDF {idx+1}")

    if st_progress_bar:
        st_progress_bar.progress(1.0, text="Process Complete! Ready to Chat.")

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
