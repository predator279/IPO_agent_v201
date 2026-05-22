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

import os
import re
import pdfplumber
import fitz  # PyMuPDF — fast text extraction for narrative pages
from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ── tunables ────────────────────────────────────────────────────────────────
NARRATIVE_CHUNK_SIZE   = 1000   # characters per narrative chunk
NARRATIVE_CHUNK_OVERLAP = 120
TABLE_MAX_ROWS_INLINE  = 60     # tables larger than this get truncated in the chunk

# Sections we want to prioritise (case-insensitive substring match on page text).
# Pages whose first 400 chars contain ANY of these are flagged as "important".
IMPORTANT_SECTION_HINTS = [
    "financial statements", "revenue", "profit", "ebitda", "balance sheet",
    "cash flow", "objects of the issue", "risk factors", "business overview",
    "industry overview", "key performance", "restated", "consolidated",
    "standalone", "utilisation of proceeds", "basis of allotment",
    "promoter", "management discussion", "litigation",
]

# Pages whose content is almost entirely boilerplate are SKIPPED.
SKIP_PAGE_HINTS = [
    "table of contents", "contents", "index", "this page is intentionally left blank",
    "disclaimer clause", "definition of technical", "abbreviations",
]
# ────────────────────────────────────────────────────────────────────────────


def _is_skip_page(text: str) -> bool:
    snippet = text[:600].lower()
    return any(hint in snippet for hint in SKIP_PAGE_HINTS)


def _is_important_page(text: str) -> bool:
    snippet = text[:400].lower()
    return any(hint in snippet for hint in IMPORTANT_SECTION_HINTS)


def _table_to_markdown(table: list) -> str:
    """Convert pdfplumber table (list-of-lists) to a compact markdown string."""
    if not table:
        return ""
    rows = []
    for i, row in enumerate(table):
        cleaned = [str(cell).strip() if cell is not None else "" for cell in row]
        rows.append("| " + " | ".join(cleaned) + " |")
        if i == 0:
            rows.append("|" + "|".join(["---"] * len(row)) + "|")
    return "\n".join(rows)


def _truncate_table_markdown(md: str, max_rows: int = TABLE_MAX_ROWS_INLINE) -> str:
    lines = md.split("\n")
    # header row + separator row + data rows
    if len(lines) > max_rows + 2:
        kept = lines[: max_rows + 2]
        kept.append(f"... [{len(lines) - max_rows - 2} more rows truncated] ...")
        return "\n".join(kept)
    return md


def process_pdf_with_pdfplumber(
    pdf_path: str,
    progress_callback=None,   # optional callable(fraction, message)
    max_pages: int = 0,       # 0 = no limit; set e.g. 800 to cap processing
) -> List[Document]:
    """
    Main entry-point.  Reads the PDF page-by-page with pdfplumber:
      • Extracts tables → markdown chunks with metadata
      • Extracts narrative text  → split into overlapping chunks
    Returns a flat list of LangChain Document objects ready for embedding.
    """

    print(f"\n📄 Opening PDF: {pdf_path}")
    docs: List[Document] = []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=NARRATIVE_CHUNK_SIZE,
        chunk_overlap=NARRATIVE_CHUNK_OVERLAP,
    )

    # We use pdfplumber for table detection and PyMuPDF for fast raw text.
    # Open both handles once to avoid repeated file I/O.
    try:
        plumber_pdf = pdfplumber.open(pdf_path)
        mupdf_pdf   = fitz.open(pdf_path)
    except Exception as e:
        print(f"❌ Failed to open PDF: {e}")
        return []

    total_pages = len(plumber_pdf.pages)
    if max_pages and max_pages < total_pages:
        effective_pages = max_pages
        print(f"ℹ️  Capping at {max_pages} pages (document has {total_pages}).")
    else:
        effective_pages = total_pages

    print(f"✅ PDF loaded — {total_pages} pages total, processing {effective_pages}.")

    skipped = 0
    table_count = 0
    text_chunk_count = 0

    for page_num in range(effective_pages):
        # ── fast text via PyMuPDF ────────────────────────────────────────────
        try:
            mu_page  = mupdf_pdf[page_num]
            raw_text = mu_page.get_text("text")
        except Exception:
            raw_text = ""

        # Skip pure-boilerplate pages
        if _is_skip_page(raw_text):
            skipped += 1
            continue

        importance = _is_important_page(raw_text)

        # ── table extraction via pdfplumber ──────────────────────────────────
        try:
            pl_page = plumber_pdf.pages[page_num]
            tables  = pl_page.extract_tables()
        except Exception:
            tables = []

        table_bboxes = []   # track bounding boxes to remove table text from narrative
        if tables:
            for tbl in tables:
                md = _table_to_markdown(tbl)
                if len(md.strip()) < 20:   # skip near-empty tables
                    continue
                md_truncated = _truncate_table_markdown(md)
                doc = Document(
                    page_content=f"[TABLE — page {page_num + 1}]\n{md_truncated}",
                    metadata={
                        "page":        page_num + 1,
                        "type":        "table",
                        "important":   importance,
                        "source":      os.path.basename(pdf_path),
                    },
                )
                docs.append(doc)
                table_count += 1

            # Collect table bboxes so we can mask them from narrative text
            try:
                for tbl_finder in pl_page.find_tables():
                    table_bboxes.append(tbl_finder.bbox)
            except Exception:
                pass

        # ── narrative text (minus table regions) ────────────────────────────
        # Re-extract text excluding table bounding boxes for cleaner narrative.
        narrative_text = raw_text
        if table_bboxes:
            try:
                # Crop away table regions from pdfplumber page and re-extract
                filtered_words = []
                for word in (pl_page.extract_words() or []):
                    wx0, wy0, wx1, wy1 = word["x0"], word["top"], word["x1"], word["bottom"]
                    in_table = False
                    for bx0, by0, bx1, by1 in table_bboxes:
                        if wx0 >= bx0 and wy0 >= by0 and wx1 <= bx1 and wy1 <= by1:
                            in_table = True
                            break
                    if not in_table:
                        filtered_words.append(word["text"])
                narrative_text = " ".join(filtered_words)
            except Exception:
                pass   # fall back to full raw_text

        # Clean up whitespace
        narrative_text = re.sub(r"\n{3,}", "\n\n", narrative_text).strip()

        if len(narrative_text) > 80:   # skip nearly-empty pages
            chunks = text_splitter.create_documents(
                [narrative_text],
                metadatas=[{
                    "page":      page_num + 1,
                    "type":      "text",
                    "important": importance,
                    "source":    os.path.basename(pdf_path),
                }],
            )
            docs.extend(chunks)
            text_chunk_count += len(chunks)

        # ── progress reporting ───────────────────────────────────────────────
        if progress_callback and (page_num % 50 == 0 or page_num == effective_pages - 1):
            fraction = (page_num + 1) / effective_pages
            progress_callback(
                fraction,
                f"Parsing page {page_num + 1}/{effective_pages} "
                f"— {table_count} tables, {text_chunk_count} text chunks so far…",
            )

    plumber_pdf.close()
    mupdf_pdf.close()

    print(
        f"\n✅ Parsing complete:\n"
        f"   Pages skipped (boilerplate): {skipped}\n"
        f"   Table chunks:  {table_count}\n"
        f"   Text chunks:   {text_chunk_count}\n"
        f"   Total docs:    {len(docs)}"
    )
    return docs


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
