# ipo_extractor.py
#
# Runs at document-load time to extract ALL structured parameters from
# an RHP/DRHP vectorstore using targeted RAG queries.
#
# Architecture:
#   - Each "extractor" fires a focused retrieval query + LLM extraction prompt
#   - Results are merged into a single IPOProfile dataclass
#   - Extraction is parallelised with ThreadPoolExecutor for speed
#   - Every field degrades gracefully to None — no crashes on missing data
#   - Results are cached to disk so re-loading is instant

import os
import json
import re
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

from cloud_utils import get_cached_profile, save_profile_to_cache

# ── config ────────────────────────────────────────────────────────────────────
LLM_MODEL         = "llama-3.3-70b-versatile"
EXTRACTION_K      = 6      # chunks retrieved per extraction query
CACHE_DIR         = "ipo_analysis_cache"
MAX_WORKERS       = 6      # parallel extraction threads
# ─────────────────────────────────────────────────────────────────────────────


# ── data model ────────────────────────────────────────────────────────────────

@dataclass
class FinancialYear:
    year: str                    # e.g. "FY2024"
    revenue: Optional[str]       # e.g. "₹1,234 Cr"
    ebitda: Optional[str]
    ebitda_margin: Optional[str] # e.g. "18.2%"
    pat: Optional[str]
    pat_margin: Optional[str]
    eps: Optional[str]
    cash_flow_ops: Optional[str]


@dataclass
class BalanceSheet:
    total_assets: Optional[str]
    total_liabilities: Optional[str]
    net_worth: Optional[str]
    debt_to_equity: Optional[str]
    current_ratio: Optional[str]
    working_capital: Optional[str]


@dataclass
class ReturnRatios:
    roe: Optional[str]
    roce: Optional[str]
    roa: Optional[str]


@dataclass
class IPOProfile:
    # ── Basic Info ────────────────────────────────────────────────────────────
    company_name:         Optional[str] = None
    sector:               Optional[str] = None
    issue_size:           Optional[str] = None
    fresh_issue:          Optional[str] = None
    ofs:                  Optional[str] = None
    price_band:           Optional[str] = None
    lot_size:             Optional[str] = None
    listing_exchange:     Optional[str] = None
    face_value:           Optional[str] = None
    market_cap:           Optional[str] = None

    # ── Business ──────────────────────────────────────────────────────────────
    business_model:       Optional[str] = None
    revenue_streams:      List[str]     = field(default_factory=list)
    competitive_moat:     Optional[str] = None
    geographic_presence:  Optional[str] = None
    tam_sam:              Optional[str] = None
    customer_segments:    Optional[str] = None
    industry_growth:      Optional[str] = None

    # ── Financials ────────────────────────────────────────────────────────────
    financials:           List[FinancialYear] = field(default_factory=list)
    balance_sheet:        Optional[BalanceSheet] = None
    return_ratios:        Optional[ReturnRatios] = None

    # ── Valuation ─────────────────────────────────────────────────────────────
    pe_ratio:             Optional[str] = None
    pb_ratio:             Optional[str] = None
    ev_ebitda:            Optional[str] = None
    peer_comparison:      List[Dict]    = field(default_factory=list)

    # ── Shareholding ──────────────────────────────────────────────────────────
    promoter_holding_pre:  Optional[str] = None
    promoter_holding_post: Optional[str] = None
    anchor_investors:      List[str]     = field(default_factory=list)

    # ── Objects of Issue ─────────────────────────────────────────────────────
    objects_of_issue:      List[Dict]    = field(default_factory=list)

    # ── Risk Factors ─────────────────────────────────────────────────────────
    key_risks:             List[str]     = field(default_factory=list)

    # ── Management ───────────────────────────────────────────────────────────
    promoters:             List[str]     = field(default_factory=list)
    key_management:        List[str]     = field(default_factory=list)
    litigations_summary:   Optional[str] = None

    # ── Sector KPIs ──────────────────────────────────────────────────────────
    sector_kpis:           Dict[str, str] = field(default_factory=dict)

    # ── Meta ──────────────────────────────────────────────────────────────────
    extraction_warnings:   List[str]     = field(default_factory=list)


# ── helpers ───────────────────────────────────────────────────────────────────

def _rag_query(vectorstore: Any, query: str, k: int = EXTRACTION_K) -> str:
    """Retrieve top-k chunks and join their content."""
    docs = vectorstore.similarity_search(query, k=k)
    return "\n\n---\n\n".join(d.page_content for d in docs)


def _ask_llm(llm, system: str, user: str) -> str:
    """Single LLM call, returns raw text."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human",  "{user}"),
    ])
    chain = prompt | llm
    result = chain.invoke({"user": user})
    return result.content if hasattr(result, "content") else str(result)


def _parse_json(raw: str) -> Any:
    """Extract and parse the first JSON object/array from a string."""
    # Try direct parse first
    try:
        return json.loads(raw.strip())
    except Exception:
        pass
    # Strip markdown fences
    clean = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
    try:
        return json.loads(clean)
    except Exception:
        pass
    # Find first { } block
    m = re.search(r'\{[\s\S]+\}', clean)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    # Find first [ ] block
    m = re.search(r'\[[\s\S]+\]', clean)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return None


_JSON_SYSTEM = (
    "You are a financial data extraction AI. "
    "Extract the requested information from the RHP/DRHP excerpts provided. "
    "Return ONLY valid JSON — no markdown fences, no commentary, no apologies. "
    "If a value is not found in the text, use null for strings and [] for arrays."
)


# ── individual extractors ─────────────────────────────────────────────────────

def _extract_basic_info(vs: Any, llm) -> dict:
    ctx = _rag_query(vs, "IPO issue size price band lot size face value listing exchange market cap fresh issue OFS")
    raw = _ask_llm(llm, _JSON_SYSTEM, f"""
Extract these fields from the RHP excerpts. Return JSON only.
Schema:
{{
  "company_name": "<str>",
  "sector": "<str>",
  "issue_size": "<str with unit e.g. ₹1234 Cr>",
  "fresh_issue": "<str>",
  "ofs": "<str>",
  "price_band": "<str e.g. ₹100-120>",
  "lot_size": "<int as string>",
  "listing_exchange": "<NSE/BSE/Both>",
  "face_value": "<str>",
  "market_cap": "<str>"
}}

RHP Excerpts:
{ctx}
""")
    return _parse_json(raw) or {}


def _extract_business(vs: Any, llm) -> dict:
    ctx = _rag_query(vs, "business model revenue streams products services competitive advantage moat TAM SAM market opportunity industry growth customer segments")
    raw = _ask_llm(llm, _JSON_SYSTEM, f"""
Extract these fields. Return JSON only.
Schema:
{{
  "business_model": "<2-3 sentence description of how the company earns money>",
  "revenue_streams": ["<stream1>", "<stream2>"],
  "competitive_moat": "<key competitive advantages>",
  "geographic_presence": "<states/countries where present>",
  "tam_sam": "<total addressable market and serviceable market if mentioned>",
  "customer_segments": "<B2B/B2C/Government etc>",
  "industry_growth": "<industry CAGR or growth rate if mentioned>"
}}

RHP Excerpts:
{ctx}
""")
    return _parse_json(raw) or {}


def _extract_financials(vs: Any, llm) -> list:
    ctx = _rag_query(vs, "revenue total income EBITDA PAT profit after tax EPS earnings per share cash flow from operations FY2022 FY2023 FY2024 restated financials", k=8)
    raw = _ask_llm(llm, _JSON_SYSTEM, f"""
Extract 3 years of financial data. Return a JSON ARRAY of objects, one per fiscal year.
Schema for each item:
{{
  "year": "FY20XX",
  "revenue": "<value with unit>",
  "ebitda": "<value or null>",
  "ebitda_margin": "<% or null>",
  "pat": "<value with unit>",
  "pat_margin": "<% or null>",
  "eps": "<value or null>",
  "cash_flow_ops": "<value or null>"
}}

Include only years where at least revenue OR PAT is found.
If margins are not stated, calculate them if possible.

RHP Excerpts:
{ctx}
""")
    result = _parse_json(raw)
    if isinstance(result, list):
        return result
    if isinstance(result, dict):
        # Sometimes LLM wraps in {data: [...]}
        for v in result.values():
            if isinstance(v, list):
                return v
    return []


def _extract_balance_sheet(vs: Any, llm) -> dict:
    ctx = _rag_query(vs, "total assets total liabilities net worth shareholders equity debt borrowings current ratio working capital debt to equity")
    raw = _ask_llm(llm, _JSON_SYSTEM, f"""
Extract balance sheet data (most recent year). Return JSON only.
Schema:
{{
  "total_assets": "<value>",
  "total_liabilities": "<value>",
  "net_worth": "<value>",
  "debt_to_equity": "<ratio>",
  "current_ratio": "<ratio>",
  "working_capital": "<value>"
}}

RHP Excerpts:
{ctx}
""")
    return _parse_json(raw) or {}


def _extract_return_ratios(vs: Any, llm) -> dict:
    ctx = _rag_query(vs, "ROE return on equity ROCE return on capital employed ROA return on assets")
    raw = _ask_llm(llm, _JSON_SYSTEM, f"""
Extract return ratios (most recent year). Return JSON only.
Schema:
{{
  "roe": "<value%>",
  "roce": "<value%>",
  "roa": "<value%>"
}}

RHP Excerpts:
{ctx}
""")
    return _parse_json(raw) or {}


def _extract_valuation(vs: Any, llm) -> dict:
    ctx = _rag_query(vs, "P/E ratio price earnings EV EBITDA price book peer comparison listed peers valuation")
    raw = _ask_llm(llm, _JSON_SYSTEM, f"""
Extract valuation metrics. Return JSON only.
Schema:
{{
  "pe_ratio": "<value or range>",
  "pb_ratio": "<value>",
  "ev_ebitda": "<value>",
  "peer_comparison": [
    {{"name": "<peer company>", "pe": "<value>", "revenue": "<value>", "pat_margin": "<value>"}}
  ]
}}

RHP Excerpts:
{ctx}
""")
    return _parse_json(raw) or {}


def _extract_shareholding(vs: Any, llm) -> dict:
    ctx = _rag_query(vs, "promoter shareholding pre IPO post IPO anchor investors lock-in period institutional investors")
    raw = _ask_llm(llm, _JSON_SYSTEM, f"""
Extract shareholding info. Return JSON only.
Schema:
{{
  "promoter_holding_pre": "<% before IPO>",
  "promoter_holding_post": "<% after IPO>",
  "anchor_investors": ["<name1>", "<name2>"],
  "promoters": ["<name1>", "<name2>"]
}}

RHP Excerpts:
{ctx}
""")
    return _parse_json(raw) or {}


def _extract_objects_of_issue(vs: Any, llm) -> list:
    ctx = _rag_query(vs, "objects of the issue utilisation of proceeds debt repayment capex expansion working capital general corporate")
    raw = _ask_llm(llm, _JSON_SYSTEM, f"""
Extract how the IPO proceeds will be used. Return a JSON ARRAY.
Schema:
[
  {{"purpose": "<description>", "amount": "<value or % if mentioned>", "category": "debt_repayment|expansion|working_capital|acquisition|general_corporate|other"}}
]

RHP Excerpts:
{ctx}
""")
    result = _parse_json(raw)
    return result if isinstance(result, list) else []


def _extract_risks(vs: Any, llm) -> list:
    ctx = _rag_query(vs, "risk factors key risks litigation regulatory concentration dependency losses cash burn", k=8)
    raw = _ask_llm(llm, _JSON_SYSTEM, f"""
Extract the 8-10 most important risk factors from the RHP.
Return a JSON ARRAY of strings. Each string should be a concise 1-2 sentence risk.

RHP Excerpts:
{ctx}
""")
    result = _parse_json(raw)
    if isinstance(result, list):
        return [r if isinstance(r, str) else str(r) for r in result[:10]]
    return []


def _extract_management(vs: Any, llm) -> dict:
    ctx = _rag_query(vs, "managing director CEO chairman board of directors key management personnel promoter background experience litigation legal proceedings")
    raw = _ask_llm(llm, _JSON_SYSTEM, f"""
Extract management information. Return JSON only.
Schema:
{{
  "key_management": ["<Name — Role>", "<Name — Role>"],
  "litigations_summary": "<brief summary of outstanding litigations or null>"
}}

RHP Excerpts:
{ctx}
""")
    return _parse_json(raw) or {}


def _extract_sector_kpis(vs: Any, llm, sector: Optional[str]) -> dict:
    """
    Extract sector-specific operational KPIs based on detected sector.
    """
    sector_lower = (sector or "").lower()

    if any(w in sector_lower for w in ["tech", "saas", "software", "it"]):
        query = "ARR MRR customer retention CAC LTV net revenue retention"
        schema = '{"ARR": null, "MRR": null, "customer_retention": null, "CAC": null, "LTV": null}'
    elif any(w in sector_lower for w in ["bank", "nbfc", "finance", "lending"]):
        query = "NPA gross NPA net NPA CASA ratio capital adequacy AUM"
        schema = '{"gross_npa": null, "net_npa": null, "casa_ratio": null, "capital_adequacy": null, "aum": null}'
    elif any(w in sector_lower for w in ["manufactur", "industrial", "auto"]):
        query = "capacity utilization production volume order book plant"
        schema = '{"capacity_utilization": null, "production_volume": null, "order_book": null}'
    elif any(w in sector_lower for w in ["retail", "consumer", "ecommerce", "d2c"]):
        query = "same store sales growth number of stores GMV repeat customers active users"
        schema = '{"same_store_sales_growth": null, "store_count": null, "gmv": null, "repeat_customer_rate": null}'
    elif any(w in sector_lower for w in ["pharma", "health", "hospital"]):
        query = "ANDA filings regulated markets EBITDA per bed occupancy rate"
        schema = '{"anda_filings": null, "regulated_markets_revenue": null, "occupancy_rate": null}'
    else:
        # Generic operational metrics
        query = "key performance indicators operational metrics market share"
        schema = '{"market_share": null, "key_metric_1": null, "key_metric_2": null}'

    ctx = _rag_query(vs, query)
    raw = _ask_llm(llm, _JSON_SYSTEM, f"""
Extract these sector-specific KPIs from the RHP excerpts.
Return JSON with this schema (use null for missing values):
{schema}

RHP Excerpts:
{ctx}
""")
    result = _parse_json(raw)
    if isinstance(result, dict):
        # Remove null entries
        return {k: v for k, v in result.items() if v is not None}
    return {}


# ── main extraction orchestrator ──────────────────────────────────────────────

def extract_ipo_profile(
    vectorstore: Any,
    ipo_name: str,
    progress_callback=None,   # optional callable(message: str)
) -> IPOProfile:
    """
    Runs all extractors in parallel against the vectorstore and assembles
    a complete IPOProfile. Cached to disk — second call is instant.
    """
    # os.makedirs(CACHE_DIR, exist_ok=True)
    # safe = re.sub(r"[^a-zA-Z0-9_-]", "_", ipo_name)
    # cache_path = os.path.join(CACHE_DIR, f"{safe}_profile.json")

    # ── load from cache ───────────────────────────────────────────────────────
    cached_data = get_cached_profile(ipo_name)
    if cached_data:
        if progress_callback:
            progress_callback("✅ Loading profile from Supabase Cloud Cache…")
        return _dict_to_profile(cached_data)

    if progress_callback:
        progress_callback("Starting structured extraction from RHP…")

    llm = ChatGroq(temperature=0, model_name=LLM_MODEL)
    profile = IPOProfile()
    warnings = []

    # ── define extraction tasks ───────────────────────────────────────────────
    tasks = {
        "basic":        lambda: _extract_basic_info(vectorstore, llm),
        "business":     lambda: _extract_business(vectorstore, llm),
        "financials":   lambda: _extract_financials(vectorstore, llm),
        "balance":      lambda: _extract_balance_sheet(vectorstore, llm),
        "ratios":       lambda: _extract_return_ratios(vectorstore, llm),
        "valuation":    lambda: _extract_valuation(vectorstore, llm),
        "shareholding": lambda: _extract_shareholding(vectorstore, llm),
        "objects":      lambda: _extract_objects_of_issue(vectorstore, llm),
        "risks":        lambda: _extract_risks(vectorstore, llm),
        "management":   lambda: _extract_management(vectorstore, llm),
    }

    results = {}
    completed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_key = {executor.submit(fn): key for key, fn in tasks.items()}
        for future in as_completed(future_to_key):
            key = future_to_key[future]
            try:
                results[key] = future.result()
                completed += 1
                if progress_callback:
                    progress_callback(f"Extracted: {key} ({completed}/{len(tasks)})")
                print(f"  ✅ {key}")
            except Exception as exc:
                warnings.append(f"{key}: {exc}")
                results[key] = {}
                print(f"  ⚠️  {key} failed: {exc}")

    # ── assemble profile ──────────────────────────────────────────────────────
    b = results.get("basic", {}) or {}
    profile.company_name        = b.get("company_name") or ipo_name
    profile.sector              = b.get("sector")
    profile.issue_size          = b.get("issue_size")
    profile.fresh_issue         = b.get("fresh_issue")
    profile.ofs                 = b.get("ofs")
    profile.price_band          = b.get("price_band")
    profile.lot_size            = b.get("lot_size")
    profile.listing_exchange    = b.get("listing_exchange")
    profile.face_value          = b.get("face_value")
    profile.market_cap          = b.get("market_cap")

    biz = results.get("business", {}) or {}
    profile.business_model      = biz.get("business_model")
    profile.revenue_streams     = biz.get("revenue_streams") or []
    profile.competitive_moat    = biz.get("competitive_moat")
    profile.geographic_presence = biz.get("geographic_presence")
    profile.tam_sam             = biz.get("tam_sam")
    profile.customer_segments   = biz.get("customer_segments")
    profile.industry_growth     = biz.get("industry_growth")

    fin_list = results.get("financials", []) or []
    profile.financials = [
        FinancialYear(
            year          = f.get("year", ""),
            revenue       = f.get("revenue"),
            ebitda        = f.get("ebitda"),
            ebitda_margin = f.get("ebitda_margin"),
            pat           = f.get("pat"),
            pat_margin    = f.get("pat_margin"),
            eps           = f.get("eps"),
            cash_flow_ops = f.get("cash_flow_ops"),
        )
        for f in fin_list if isinstance(f, dict)
    ]

    bs = results.get("balance", {}) or {}
    profile.balance_sheet = BalanceSheet(
        total_assets    = bs.get("total_assets"),
        total_liabilities = bs.get("total_liabilities"),
        net_worth       = bs.get("net_worth"),
        debt_to_equity  = bs.get("debt_to_equity"),
        current_ratio   = bs.get("current_ratio"),
        working_capital = bs.get("working_capital"),
    )

    rr = results.get("ratios", {}) or {}
    profile.return_ratios = ReturnRatios(
        roe  = rr.get("roe"),
        roce = rr.get("roce"),
        roa  = rr.get("roa"),
    )

    val = results.get("valuation", {}) or {}
    profile.pe_ratio        = val.get("pe_ratio")
    profile.pb_ratio        = val.get("pb_ratio")
    profile.ev_ebitda       = val.get("ev_ebitda")
    profile.peer_comparison = val.get("peer_comparison") or []

    sh = results.get("shareholding", {}) or {}
    profile.promoter_holding_pre  = sh.get("promoter_holding_pre")
    profile.promoter_holding_post = sh.get("promoter_holding_post")
    profile.anchor_investors      = sh.get("anchor_investors") or []
    profile.promoters             = sh.get("promoters") or []

    profile.objects_of_issue = results.get("objects", []) or []
    profile.key_risks        = results.get("risks", []) or []

    mgmt = results.get("management", {}) or {}
    profile.key_management       = mgmt.get("key_management") or []
    profile.litigations_summary  = mgmt.get("litigations_summary")

    # Sector KPIs — run after we know the sector
    try:
        profile.sector_kpis = _extract_sector_kpis(vectorstore, llm, profile.sector)
        print("  ✅ sector_kpis")
    except Exception as exc:
        warnings.append(f"sector_kpis: {exc}")

    profile.extraction_warnings = warnings

    # ── 4. SAVE TO CLOUD CACHE ───────────────────────────────────────────────
    try:
        save_profile_to_cache(ipo_name, _profile_to_dict(profile))
        print(f"✅ Analysis for {ipo_name} saved to Supabase DB.")
    except Exception as e:
        print(f"⚠️ Cloud save failed: {e}")

    return profile


# ── serialisation helpers ─────────────────────────────────────────────────────

def _profile_to_dict(p: IPOProfile) -> dict:
    d = asdict(p)
    return d


def _dict_to_profile(d: dict) -> IPOProfile:
    """Reconstruct IPOProfile from a plain dict (from JSON cache)."""
    p = IPOProfile()
    for k, v in d.items():
        if k == "financials" and isinstance(v, list):
            setattr(p, k, [FinancialYear(**f) for f in v if isinstance(f, dict)])
        elif k == "balance_sheet" and isinstance(v, dict):
            setattr(p, k, BalanceSheet(**v))
        elif k == "return_ratios" and isinstance(v, dict):
            setattr(p, k, ReturnRatios(**v))
        else:
            setattr(p, k, v)
    return p