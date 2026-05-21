# sentiment_agent.py  (v2 — Tavily + Reddit + Google News + structured output)
#
# Sources (in priority order):
#   1. Tavily Search API  — best quality, real-time web results with full content
#   2. Reddit via PRAW    — retail investor community discussion
#   3. Google News RSS    — mainstream financial media headlines
#
# Output schema:
# {
#   "score": 3.8,                    # 0.0 (very negative) → 5.0 (very positive)
#   "sentiment_label": "Positive",   # Very Negative / Negative / Neutral / Positive / Very Positive
#   "gmp": "₹45 (~12%)",             # Grey Market Premium if found
#   "subscription_estimate": "...",  # QIB/HNI/Retail estimates from news
#   "summary": ["bullet1", ...],     # 4-6 key sentiment drivers
#   "positives": ["..."],            # Bull case points
#   "negatives": ["..."],            # Bear case / concern points
#   "sources_used": ["Tavily", "Reddit", "Google News"],
#   "articles": [                    # Top cited articles
#       {"title": "...", "url": "...", "source": "..."}
#   ]
# }

import os
import json
import re
import urllib.parse
from typing import List, Dict, Any, Optional

import feedparser
import praw
import requests
from bs4 import BeautifulSoup


# ── Reddit config (read from env or fallback to hardcoded for dev) ────────────
REDDIT_CLIENT_ID     = os.getenv("REDDIT_CLIENT_ID",     "KHbmxuI59KvPPjZOuPprvA")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "GWtaNscUFjCWPmGJtg0NIaWVCfXSLw")
REDDIT_USER_AGENT    = os.getenv("REDDIT_USER_AGENT",    "IPO_Analyzer/v2.0")

# ── Tavily config ─────────────────────────────────────────────────────────────
# Users can set TAVILY_API_KEY in environment or Streamlit secrets.
# Free tier: 1000 searches/month. https://app.tavily.com
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
TAVILY_SEARCH_URL = "https://api.tavily.com/search"

# ── sentiment label mapping ───────────────────────────────────────────────────
def _score_to_label(score: float) -> str:
    if score >= 4.2:   return "Very Positive 🚀"
    if score >= 3.4:   return "Positive 📈"
    if score >= 2.6:   return "Neutral ➡️"
    if score >= 1.8:   return "Negative 📉"
    return "Very Negative ⚠️"


# ==============================================================================
# SOURCE 1 — Tavily (best quality)
# ==============================================================================

def _fetch_tavily(ipo_name: str, max_results: int = 8) -> List[Dict]:
    """
    Uses Tavily Search API to find high-quality, recent articles about the IPO.
    Returns a list of {source, title, url, content} dicts.
    """
    api_key = TAVILY_API_KEY or os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        print("   ⚠️  No TAVILY_API_KEY set — skipping Tavily.")
        return []

    results = []
    queries = [
        f"{ipo_name} IPO review analysis 2025",
        f"{ipo_name} IPO GMP grey market premium subscription",
        f"{ipo_name} IPO allotment listing date investor opinion",
    ]

    for query in queries:
        try:
            resp = requests.post(
                TAVILY_SEARCH_URL,
                json={
                    "api_key":              api_key,
                    "query":                query,
                    "search_depth":         "advanced",
                    "include_answer":       False,
                    "include_raw_content":  False,
                    "max_results":          max_results // len(queries) + 1,
                    "include_domains":      [
                        "moneycontrol.com", "economictimes.indiatimes.com",
                        "livemint.com", "businessstandard.com", "zerodha.com",
                        "chittorgarh.com", "ipowatch.in", "investorgain.com",
                        "equitybulls.com", "reddit.com", "valuepickr.com",
                    ],
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            for item in data.get("results", []):
                results.append({
                    "source":  "Tavily/" + (item.get("url", "").split("/")[2] if item.get("url") else "web"),
                    "title":   item.get("title", ""),
                    "url":     item.get("url", ""),
                    "content": item.get("content", "")[:800],  # cap per article
                })
        except Exception as exc:
            print(f"   ⚠️  Tavily query failed: {exc}")

    # Deduplicate by URL
    seen_urls = set()
    unique = []
    for r in results:
        if r["url"] not in seen_urls:
            seen_urls.add(r["url"])
            unique.append(r)

    print(f"   ✅ Tavily: {len(unique)} articles found.")
    return unique[:max_results]


# ==============================================================================
# SOURCE 2 — Reddit
# ==============================================================================

def _fetch_reddit(ipo_name: str, limit_posts: int = 15, limit_comments: int = 3) -> List[Dict]:
    """Fetch recent Reddit discussions about the IPO."""
    try:
        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT,
        )
        records = []
        query = f"{ipo_name} IPO" if "ipo" not in ipo_name.lower() else ipo_name
        # Search focused subreddits first, then all
        subreddits = ["IndiaInvestments+DalalStreetTalks+IndianStockMarket+all"]
        for sub in subreddits:
            for submission in reddit.subreddit(sub).search(query, sort="new", limit=limit_posts):
                submission.comments.replace_more(limit=0)
                comments_text = " | ".join(
                    c.body for i, c in enumerate(submission.comments.list())
                    if i < limit_comments and hasattr(c, "body") and len(c.body) > 20
                )
                records.append({
                    "source":  f"Reddit/r/{submission.subreddit.display_name}",
                    "title":   submission.title,
                    "url":     f"https://reddit.com{submission.permalink}",
                    "content": f"{submission.title}. {getattr(submission, 'selftext', '')[:300]} | Comments: {comments_text}",
                })
        print(f"   ✅ Reddit: {len(records)} posts found.")
        return records[:limit_posts]
    except Exception as exc:
        print(f"   ⚠️  Reddit fetch failed: {exc}")
        return []


# ==============================================================================
# SOURCE 3 — Google News RSS
# ==============================================================================

def _fetch_google_news(ipo_name: str, limit: int = 10) -> List[Dict]:
    """Fetch Google News RSS headlines about the IPO."""
    try:
        query   = urllib.parse.quote(f"{ipo_name} IPO")
        rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
        feed    = feedparser.parse(rss_url)
        results = []
        for entry in feed.entries[:limit]:
            summary = BeautifulSoup(getattr(entry, "summary", ""), "html.parser").get_text(" ", strip=True)
            results.append({
                "source":  "Google News",
                "title":   getattr(entry, "title", ""),
                "url":     getattr(entry, "link", ""),
                "content": f"{getattr(entry, 'title', '')}. {summary}",
            })
        print(f"   ✅ Google News: {len(results)} articles found.")
        return results
    except Exception as exc:
        print(f"   ⚠️  Google News fetch failed: {exc}")
        return []


# ==============================================================================
# GMP extraction helper
# ==============================================================================

def _extract_gmp(all_text: str, ipo_name: str) -> Optional[str]:
    """
    Tries to pull Grey Market Premium from aggregated source text.
    Patterns: "GMP ₹45", "grey market premium of ₹45", "GMP: +45"
    """
    patterns = [
        r"GMP[:\s]+[₹+]?\s*(\d+[\.,]?\d*)\s*(?:rupees?|rs\.?)?(?:\s*\(([^)]+)\))?",
        r"grey\s+market\s+premium[:\s]+[₹+]?\s*(\d+[\.,]?\d*)",
        r"trading at[:\s]+[₹+]?\s*(\d+[\.,]?\d*)\s+(?:premium|above)",
    ]
    for pat in patterns:
        m = re.search(pat, all_text, re.IGNORECASE)
        if m:
            val = m.group(1).replace(",", "")
            pct = m.group(2) if m.lastindex and m.lastindex >= 2 else None
            return f"₹{val}" + (f" ({pct})" if pct else "")
    return None


# ==============================================================================
# LLM sentiment analysis
# ==============================================================================

def _run_llm_analysis(
    ipo_name: str,
    snippets: List[Dict],
    llm,
) -> Dict[str, Any]:
    """
    Sends all collected snippets to the LLM for structured sentiment analysis.
    """
    from agents.tools import invoke_model

    # Build context — weight Tavily content higher (more complete)
    tavily_items  = [s for s in snippets if "Tavily" in s.get("source", "")]
    other_items   = [s for s in snippets if "Tavily" not in s.get("source", "")]

    # Tavily gets up to 12 items, others up to 8
    selected = tavily_items[:12] + other_items[:8]

    all_text = "\n\n".join([
        f"[{s['source']}] {s['title']}\n{s['content']}"
        for s in selected
    ])

    gmp = _extract_gmp(all_text, ipo_name)

    system_prompt = """You are a precise Indian IPO sentiment analyst.
Analyze the provided news and forum snippets about an IPO.
Return ONLY a valid JSON object — no markdown, no commentary.

JSON Schema:
{
  "score": <float 0.0-5.0>,
  "sentiment_label": "<Very Negative|Negative|Neutral|Positive|Very Positive>",
  "gmp": "<GMP value if found in text, else null>",
  "subscription_estimate": "<QIB/HNI/Retail subscription estimates if mentioned, else null>",
  "summary": ["<key point 1>", "<key point 2>", "<3-5 total bullets>"],
  "positives": ["<bull case point>", "..."],
  "negatives": ["<concern>", "..."]
}

Scoring guide:
0.0-1.0: Very Negative (fraud alerts, massive losses, strong avoid)
1.0-2.0: Negative (overvalued, poor financials, red flags)
2.0-3.0: Neutral (mixed, wait and watch)
3.0-4.0: Positive (good fundamentals, reasonable valuation, recommend)
4.0-5.0: Very Positive (exceptional growth, strong demand, high GMP)"""

    user_prompt = f"Analyze sentiment for '{ipo_name}' IPO:\n\n{all_text[:4000]}"

    response = invoke_model(llm, [("system", system_prompt), ("user", user_prompt)])

    # Parse JSON robustly
    try:
        start = response.find("{")
        end   = response.rfind("}") + 1
        if start != -1 and end > start:
            result = json.loads(response[start:end])
        else:
            raise ValueError("No JSON found")
    except Exception:
        result = {
            "score": 2.5,
            "sentiment_label": "Neutral",
            "summary": [response[:500]],
            "positives": [],
            "negatives": [],
        }

    # Override GMP with regex-extracted value if LLM missed it
    if gmp and not result.get("gmp"):
        result["gmp"] = gmp

    # Add sentiment label if LLM returned score but no label
    if "score" in result and "sentiment_label" not in result:
        result["sentiment_label"] = _score_to_label(result["score"])

    return result


# ==============================================================================
# MAIN PUBLIC FUNCTION
# ==============================================================================

def analyze_sentiment(ipo_name: str) -> Dict[str, Any]:
    """
    Multi-source IPO sentiment analysis.
    Fetches from Tavily + Reddit + Google News, then uses LLM for structured output.

    Returns:
        {
            "score": float,
            "sentiment_label": str,
            "gmp": str | None,
            "subscription_estimate": str | None,
            "summary": [str, ...],
            "positives": [str, ...],
            "negatives": [str, ...],
            "sources_used": [str, ...],
            "articles": [{"title", "url", "source"}, ...]
        }
    """
    from agents.tools import get_llm

    print(f"\n🔍 [Sentiment Agent] Analyzing: {ipo_name}")
    all_snippets = []
    sources_used = []

    # ── fetch from all sources ────────────────────────────────────────────────
    print("→ Tavily…")
    tavily_data = _fetch_tavily(ipo_name)
    if tavily_data:
        all_snippets.extend(tavily_data)
        sources_used.append("Tavily")

    print("→ Reddit…")
    reddit_data = _fetch_reddit(ipo_name)
    if reddit_data:
        all_snippets.extend(reddit_data)
        sources_used.append("Reddit")

    print("→ Google News…")
    news_data = _fetch_google_news(ipo_name)
    if news_data:
        all_snippets.extend(news_data)
        sources_used.append("Google News")

    if not all_snippets:
        return {
            "score": 2.5,
            "sentiment_label": "Neutral ➡️",
            "gmp": None,
            "subscription_estimate": None,
            "summary": ["No market data found for this IPO yet."],
            "positives": [],
            "negatives": [],
            "sources_used": [],
            "articles": [],
        }

    # ── LLM analysis ─────────────────────────────────────────────────────────
    print("→ Running LLM sentiment analysis…")
    llm = get_llm(temperature=0, model_name="llama-3.3-70b-versatile")
    result = _run_llm_analysis(ipo_name, all_snippets, llm)

    # ── attach metadata ───────────────────────────────────────────────────────
    result["sources_used"] = sources_used
    result["articles"] = [
        {"title": s["title"], "url": s.get("url", ""), "source": s["source"]}
        for s in all_snippets
        if s.get("title") and s.get("url")
    ][:10]

    # Normalise label with emoji
    if "sentiment_label" in result:
        label = result["sentiment_label"].replace(" 🚀", "").replace(" 📈", "").replace(" ➡️", "").replace(" 📉", "").replace(" ⚠️", "")
        result["sentiment_label"] = _score_to_label(result.get("score", 2.5))

    print(f"✅ [Sentiment Agent] Score: {result.get('score')} | {result.get('sentiment_label')}")
    return result