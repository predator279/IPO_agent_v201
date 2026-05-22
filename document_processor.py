# document_processor.py  (v2 — pdfplumber + smart chunking for large RHP/DRHP PDFs)
#
# Key improvements over v1:
#   1. pdfplumber extracts tables as clean markdown — no more garbled text
#   2. Smart page-range filtering skips boilerplate (TOC, disclaimers, cover pages)
#   3. LLM table summarisation is LAZY — only called at query time for critical tables,
#      not for every table at ingest.  A lightweight text-based table repr is stored
#      for retrieval; full summarisation is a separate utility called by the RAG chain.
#   4. Streaming batch ingest — memory-safe for 2000-page docs.
#   5. Metadata tagging (page number, element type) so the retriever can surface
#      provenance to the user.

# document_processor.py (v2.1 — Memory Optimized for Cloud)
import os
import re
import pdfplumber
import fitz  # PyMuPDF
import gc    # Garbage Collector
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── tunables ────────────────────────────────────────────────────────────────
NARRATIVE_CHUNK_SIZE   = 1000   
NARRATIVE_CHUNK_OVERLAP = 120

IMPORTANT_SECTION_HINTS = [
    "financial statements", "revenue", "profit", "ebitda", "balance sheet",
    "cash flow", "objects of the issue", "risk factors", "business overview",
]

SKIP_PAGE_HINTS = ["table of contents", "contents", "index"]

def _is_skip_page(text: str) -> bool:
    snippet = text[:600].lower()
    return any(hint in snippet for hint in SKIP_PAGE_HINTS)

def _is_important_page(text: str) -> bool:
    snippet = text[:400].lower()
    return any(hint in snippet for hint in IMPORTANT_SECTION_HINTS)

def _table_to_markdown(table: list) -> str:
    if not table: return ""
    rows = []
    for i, row in enumerate(table):
        cleaned = [str(cell).strip() if cell is not None else "" for cell in row]
        rows.append("| " + " | ".join(cleaned) + " |")
        if i == 0: rows.append("|" + "|".join(["---"] * len(row)) + "|")
    return "\n".join(rows)

def process_pdf_with_pdfplumber(pdf_path: str, progress_callback=None, max_pages: int = 400) -> List[Document]:
    """
    Parses PDF page-by-page. Optimized to release RAM frequently.
    """
    print(f"📄 Opening PDF: {pdf_path}")
    all_chunks: List[Document] = []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=NARRATIVE_CHUNK_SIZE,
        chunk_overlap=NARRATIVE_CHUNK_OVERLAP,
    )

    try:
        # Use PyMuPDF for text (fast) and pdfplumber for tables (accurate)
        mupdf_doc = fitz.open(pdf_path)
        plumb_doc = pdfplumber.open(pdf_path)
        
        total_pages = len(mupdf_doc)
        effective_pages = min(max_pages, total_pages) if max_pages > 0 else total_pages

        for page_num in range(effective_pages):
            # 1. Extract Text
            try:
                page_text = mupdf_doc[page_num].get_text("text")
                if _is_skip_page(page_text): continue
                importance = _is_important_page(page_text)
            except: page_text = ""

            # 2. Extract Tables
            try:
                tables = plumb_doc.pages[page_num].extract_tables()
                for tbl in tables:
                    md = _table_to_markdown(tbl)
                    if len(md.strip()) > 20:
                        all_chunks.append(Document(
                            page_content=f"[TABLE — page {page_num + 1}]\n{md}",
                            metadata={"page": page_num+1, "type": "table", "source": os.path.basename(pdf_path)}
                        ))
            except: pass

            # 3. Create Narrative Chunks
            if len(page_text) > 80:
                chunks = text_splitter.create_documents(
                    [page_text],
                    metadatas=[{"page": page_num+1, "type": "text", "important": importance, "source": os.path.basename(pdf_path)}]
                )
                all_chunks.extend(chunks)

            # 4. Frequent RAM cleanup
            if page_num % 50 == 0:
                if progress_callback:
                    progress_callback(page_num / effective_pages, f"Parsed {page_num} pages...")
                gc.collect()

        mupdf_doc.close()
        plumb_doc.close()
        
    except Exception as e:
        print(f"❌ Error parsing {pdf_path}: {e}")

    return all_chunks

# ── Backwards-compatible alias so chatbot_agent.py import doesn't break ─────
def process_pdf_with_unstructured(pdf_path: str) -> List[Document]:
    """
    Drop-in replacement for the old unstructured-based processor.
    chatbot_agent.py calls this name; we just forward to the new implementation.
    """
    return process_pdf_with_pdfplumber(pdf_path)


# ── Optional: on-demand LLM summarisation of a single table chunk ───────────
def summarize_table_chunk(table_markdown: str, llm) -> str:
    """
    Call this at query time (not ingest time) when you want a richer
    interpretation of a specific table chunk retrieved by the RAG chain.
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    prompt = ChatPromptTemplate.from_template(
        "You are an expert financial analyst. "
        "Provide a concise, data-rich summary of the following table. "
        "Extract key figures, trends, and important labels. "
        "Do not merely describe the table — summarise its core insight.\n\n"
        "Table:\n{table_markdown}"
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"table_markdown": table_markdown})
