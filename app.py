# app.py  (v3 — full analysis dashboard: extraction + sentiment + chat)
#
# Layout:
#   Sidebar  → API keys (Groq, Tavily, HuggingFace), IPO selector, Load button
#   Main     → Tab 1: Live IPO Market  |  Tab 2: IPO Analysis Dashboard  |  Tab 3: Chat
#
# On "Load":
#   1. Downloads + embeds RHP/DRHP into ChromaDB  (chatbot_agent)
#   2. Extracts structured IPO profile in parallel (ipo_extractor)
#   3. Runs multi-source sentiment analysis        (sentiment_agent)
#   4. Displays everything in Tab 2; Tab 3 opens for deep-dive chat

import os
import re
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from langchain_core.messages import AIMessage, HumanMessage

from ipo_fetcher     import fetch_all_ipo_data_separated
from chatbot_agent   import process_and_store_document, create_rag_chain
from ipo_extractor   import extract_ipo_profile, IPOProfile
from sentiment_agent import analyze_sentiment


# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="IPO Analysis Agent", page_icon="📈", layout="wide")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background:#1e2130; border:1px solid #2d3250;
    border-radius:10px; padding:16px 20px; margin-bottom:10px;
}
.metric-label { color:#8b95a6; font-size:11px; text-transform:uppercase; letter-spacing:.5px; }
.metric-value { color:#e8ecf0; font-size:19px; font-weight:700; margin-top:4px; }
.metric-sub   { color:#6b7685; font-size:11px; margin-top:2px; }
.section-header {
    color:#c7d0db; font-size:12px; font-weight:700; text-transform:uppercase;
    letter-spacing:1px; border-bottom:1px solid #2d3250;
    padding-bottom:6px; margin:22px 0 14px 0;
}
.risk-item     { background:#2a1f1f; border-left:3px solid #e05c5c; border-radius:4px; padding:8px 12px; margin:4px 0; font-size:13px; }
.object-item   { background:#1a2a1f; border-left:3px solid #4caf7d; border-radius:4px; padding:8px 12px; margin:4px 0; font-size:13px; }
.positive-item { background:#1a2a1f; border-left:3px solid #4caf7d; padding:8px 12px; border-radius:4px; margin:4px 0; font-size:13px; }
.negative-item { background:#2a1f1f; border-left:3px solid #e05c5c; padding:8px 12px; border-radius:4px; margin:4px 0; font-size:13px; }
.pill { display:inline-block; background:#2d3250; color:#a0b0c8; border-radius:20px; padding:3px 10px; font-size:12px; margin:2px; }
</style>
""", unsafe_allow_html=True)


# ── session state defaults ─────────────────────────────────────────────────────
for k, v in {
    "messages": [], "qa_chain": None, "ipo_name": "",
    "api_key_groq": "", "api_key_tavily": "",
    "api_key_valid": False, "llm_provider": "groq",
    "ipo_profile": None, "sentiment": None, "analysis_done": False,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── helper: metric card ────────────────────────────────────────────────────────
def _card(col, label, value, sub=None):
    col.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value or "—"}</div>'
        f'{"<div class=metric-sub>" + sub + "</div>" if sub else ""}'
        f'</div>',
        unsafe_allow_html=True,
    )

# ── helper: numeric parser for chart data ─────────────────────────────────────
def _parse_num(s):
    if not s:
        return None
    cleaned = re.sub(r"[₹,\s]", "", str(s))
    cleaned = re.sub(r"(?i)(cr|crore|lakh|mn|bn|million|billion|%)", "", cleaned)
    try:
        return float(cleaned)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration")

    provider = st.radio("LLM Provider", ["Groq (Recommended)", "HuggingFace"], index=0)
    st.session_state.llm_provider = "groq" if "Groq" in provider else "hf"

    with st.expander("🔑 API Keys", expanded=not st.session_state.api_key_valid):
        if st.session_state.llm_provider == "groq":
            groq_key = st.text_input("Groq API Key", type="password",
                                     value=st.session_state.api_key_groq, placeholder="gsk_…",
                                     help="Free at console.groq.com")
            if groq_key:
                st.session_state.api_key_groq = groq_key
                os.environ["GROQ_API_KEY"] = groq_key
                os.environ["LLM_PROVIDER"] = "groq"
                os.environ["LLM_MODEL"]    = "llama-3.3-70b-versatile"
                st.session_state.api_key_valid = True
        else:
            hf_key = st.text_input("HuggingFace Token", type="password", placeholder="hf_…")
            if hf_key:
                os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_key
                os.environ["LLM_PROVIDER"] = "hf"
                os.environ["LLM_MODEL"]    = "mistralai/Mistral-7B-Instruct-v0.2"
                st.session_state.api_key_valid = True

        tavily_key = st.text_input("Tavily API Key (optional, improves sentiment)",
                                   type="password", value=st.session_state.api_key_tavily,
                                   placeholder="tvly-…", help="Free 1000/mo at app.tavily.com")
        if tavily_key:
            st.session_state.api_key_tavily = tavily_key
            os.environ["TAVILY_API_KEY"] = tavily_key

        # Fallback from secrets.toml
        if not st.session_state.api_key_valid:
            for env, sec in [("GROQ_API_KEY","GROQ_API_KEY"),
                              ("HUGGINGFACEHUB_API_TOKEN","HUGGINGFACEHUB_API_TOKEN")]:
                try:
                    v = st.secrets.get(sec)
                    if v:
                        os.environ[env] = v
                        st.session_state.api_key_valid = True
                except Exception:
                    pass
            try:
                tv = st.secrets.get("TAVILY_API_KEY")
                if tv:
                    os.environ["TAVILY_API_KEY"] = tv
            except Exception:
                pass

        if st.session_state.api_key_valid:
            st.success("✅ API key active")
        else:
            st.warning("Enter an API key to enable AI analysis.")

    st.divider()
    st.header("📄 Select IPO")

    prioritized = []
    for cat in ["Current", "Upcoming", "Past"]:
        df = st.session_state.get(f"df_{cat}", pd.DataFrame())
        if not df.empty and "Company Name" in df.columns:
            prioritized.extend(df["Company Name"].tolist())

    options = ["--- Type manually ---"] + list(dict.fromkeys(prioritized))
    selected = st.selectbox("Choose an IPO:", options)
    ipo_to_process = (
        st.text_input("IPO name:", placeholder="e.g., Lenskart Solutions Limited")
        if selected == "--- Type manually ---"
        else selected
    )

    run_sentiment = st.checkbox("Run sentiment analysis", value=True,
                                help="Fetches market buzz from Tavily/Reddit/Google News (~30s extra).")

    if st.button("🚀 Load & Analyse", type="primary", use_container_width=True):
        if not st.session_state.api_key_valid:
            st.error("Enter an API key first.")
        elif not ipo_to_process:
            st.warning("Select or type an IPO name.")
        else:
            if st.session_state.ipo_name != ipo_to_process:
                st.session_state.update({
                    "messages": [], "qa_chain": None, "ipo_profile": None,
                    "sentiment": None, "analysis_done": False,
                    "ipo_name": ipo_to_process,
                })

            with st.status(f"Analysing {ipo_to_process}…", expanded=True) as status:
                st.write("📥 Downloading and embedding RHP/DRHP…")
                progress = st.progress(0, text="Starting…")
                vectorstore = process_and_store_document(ipo_to_process, progress)
                progress.empty()

                if not vectorstore:
                    st.error(f"Could not load documents for '{ipo_to_process}'.")
                    status.update(label="Failed ❌", state="error")
                    st.stop()

                st.write("🔗 Building RAG chain…")
                st.session_state.qa_chain = create_rag_chain(
                    vectorstore, llm_provider=st.session_state.llm_provider
                )

                st.write("🧠 Extracting structured IPO parameters (parallel)…")
                msg_slot = st.empty()
                profile = extract_ipo_profile(
                    vectorstore, ipo_to_process,
                    progress_callback=lambda m: msg_slot.caption(f"⏳ {m}")
                )
                msg_slot.empty()
                st.session_state.ipo_profile = profile

                if run_sentiment:
                    st.write("📡 Running sentiment analysis…")
                    st.session_state.sentiment = analyze_sentiment(ipo_to_process)

                st.session_state.analysis_done = True
                status.update(label=f"✅ Done — {ipo_to_process}", state="complete", expanded=False)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN AREA — tabs
# ─────────────────────────────────────────────────────────────────────────────
st.title("📈 IPO Analysis Agent")
st.caption("Live NSE data • AI-powered RHP/DRHP analysis • Multi-source market sentiment")

tab_market, tab_analysis, tab_chat = st.tabs([
    "🏦 Live IPO Market", "📊 Analysis Dashboard", "💬 Deep-Dive Chat"
])


# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — Live Market
# ──────────────────────────────────────────────────────────────────────────────
with tab_market:
    with st.spinner("Fetching NSE IPO data…"):
        ipo_data_dict = fetch_all_ipo_data_separated()
    for cat in ["Current", "Upcoming", "Past"]:
        st.session_state[f"df_{cat}"] = ipo_data_dict.get(cat, pd.DataFrame())

    t1, t2, t3 = st.tabs(["🔥 Current", "⏳ Upcoming", "✅ Past"])
    with t1:
        df = ipo_data_dict.get("Current", pd.DataFrame())
        if not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No current IPOs.")
    with t2:
        df = ipo_data_dict.get("Upcoming", pd.DataFrame())
        if not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No upcoming IPOs.")
    with t3:
        df = ipo_data_dict.get("Past", pd.DataFrame())
        if not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No past IPO data.")


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — Analysis Dashboard
# ──────────────────────────────────────────────────────────────────────────────
with tab_analysis:
    if not st.session_state.analysis_done:
        st.info("👈 Select an IPO from the sidebar and click **Load & Analyse** to see the dashboard.")
    else:
        profile: IPOProfile = st.session_state.ipo_profile
        sentiment: dict     = st.session_state.sentiment or {}

        # ── Hero ─────────────────────────────────────────────────────────────
        st.markdown(f"## {profile.company_name or st.session_state.ipo_name}")
        if profile.sector:
            st.markdown(f'`{profile.sector}`')

        # ── A: Basic Info ─────────────────────────────────────────────────────
        st.markdown('<div class="section-header">Basic IPO Information</div>', unsafe_allow_html=True)
        c = st.columns(5)
        _card(c[0], "Issue Size",   profile.issue_size)
        _card(c[1], "Price Band",   profile.price_band)
        _card(c[2], "Market Cap",   profile.market_cap)
        _card(c[3], "Lot Size",     profile.lot_size,    sub="shares/lot")
        _card(c[4], "Exchange",     profile.listing_exchange)
        c2 = st.columns(5)
        _card(c2[0], "Fresh Issue",    profile.fresh_issue)
        _card(c2[1], "OFS",            profile.ofs,         sub="Offer for Sale")
        _card(c2[2], "Face Value",     profile.face_value)
        _card(c2[3], "Promoter (Pre)", profile.promoter_holding_pre,  sub="before IPO")
        _card(c2[4], "Promoter (Post)",profile.promoter_holding_post, sub="after IPO")

        # ── B: Business ───────────────────────────────────────────────────────
        st.markdown('<div class="section-header">Business Overview</div>', unsafe_allow_html=True)
        biz1, biz2 = st.columns([3, 2])
        with biz1:
            if profile.business_model:
                st.markdown(f"**Business Model**\n\n{profile.business_model}")
            if profile.competitive_moat:
                st.markdown(f"\n**Competitive Moat**\n\n{profile.competitive_moat}")
            if profile.tam_sam:
                st.markdown(f"\n**Market Opportunity**\n\n{profile.tam_sam}")
        with biz2:
            if profile.revenue_streams:
                st.markdown("**Revenue Streams**")
                for rs in profile.revenue_streams:
                    st.markdown(f'<span class="pill">💰 {rs}</span>', unsafe_allow_html=True)
            if profile.customer_segments:
                st.markdown(f"\n**Customer Segments**\n\n{profile.customer_segments}")
            if profile.geographic_presence:
                st.markdown(f"\n**Geographic Presence**\n\n{profile.geographic_presence}")
            if profile.industry_growth:
                st.markdown(f"\n**Industry Growth**\n\n{profile.industry_growth}")

        # ── C: Financials ─────────────────────────────────────────────────────
        st.markdown('<div class="section-header">Financial Performance (3-Year Trend)</div>', unsafe_allow_html=True)
        if profile.financials:
            years    = [f.year          for f in profile.financials]
            revenues = [_parse_num(f.revenue)       for f in profile.financials]
            pats     = [_parse_num(f.pat)            for f in profile.financials]
            ebitda_m = [_parse_num(f.ebitda_margin)  for f in profile.financials]
            pat_m    = [_parse_num(f.pat_margin)     for f in profile.financials]

            fc1, fc2 = st.columns(2)
            with fc1:
                fig = go.Figure()
                if any(v is not None for v in revenues):
                    fig.add_bar(name="Revenue", x=years, y=revenues, marker_color="#4c8cf5",
                                text=[f"₹{v}Cr" if v else "" for v in revenues], textposition="outside")
                if any(v is not None for v in pats):
                    fig.add_bar(name="PAT", x=years, y=pats, marker_color="#4caf7d",
                                text=[f"₹{v}Cr" if v else "" for v in pats], textposition="outside")
                fig.update_layout(title="Revenue vs PAT (₹ Cr)", barmode="group",
                                  plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                                  font_color="#c7d0db", height=320,
                                  legend=dict(orientation="h", y=1.12))
                st.plotly_chart(fig, use_container_width=True)
            with fc2:
                fig2 = go.Figure()
                if any(v is not None for v in ebitda_m):
                    fig2.add_scatter(name="EBITDA Margin%", x=years, y=ebitda_m,
                                     mode="lines+markers+text",
                                     text=[f"{v}%" if v else "" for v in ebitda_m],
                                     textposition="top center", line_color="#f5a623")
                if any(v is not None for v in pat_m):
                    fig2.add_scatter(name="PAT Margin%", x=years, y=pat_m,
                                     mode="lines+markers+text",
                                     text=[f"{v}%" if v else "" for v in pat_m],
                                     textposition="bottom center", line_color="#4caf7d")
                fig2.update_layout(title="Margin Trends (%)", plot_bgcolor="#0e1117",
                                   paper_bgcolor="#0e1117", font_color="#c7d0db", height=320,
                                   legend=dict(orientation="h", y=1.12))
                st.plotly_chart(fig2, use_container_width=True)

            fin_rows = [{
                "Year": f.year, "Revenue": f.revenue or "—", "EBITDA": f.ebitda or "—",
                "EBITDA Margin": f.ebitda_margin or "—", "PAT": f.pat or "—",
                "PAT Margin": f.pat_margin or "—", "EPS": f.eps or "—", "CFO": f.cash_flow_ops or "—",
            } for f in profile.financials]
            st.dataframe(pd.DataFrame(fin_rows), use_container_width=True, hide_index=True)
        else:
            st.info("Financial data could not be extracted from this document.")

        # ── D: Key Metrics ────────────────────────────────────────────────────
        st.markdown('<div class="section-header">Key Metrics</div>', unsafe_allow_html=True)
        km1, km2, km3 = st.columns(3)
        with km1:
            st.markdown("**Balance Sheet (Latest Year)**")
            if profile.balance_sheet:
                bs = profile.balance_sheet
                for lbl, val in [("Total Assets", bs.total_assets), ("Total Liabilities", bs.total_liabilities),
                                  ("Net Worth", bs.net_worth), ("Debt/Equity", bs.debt_to_equity),
                                  ("Current Ratio", bs.current_ratio), ("Working Capital", bs.working_capital)]:
                    if val:
                        st.markdown(f"- **{lbl}:** {val}")
            else:
                st.caption("Not found.")
        with km2:
            st.markdown("**Return Ratios**")
            if profile.return_ratios:
                for lbl, val in [("ROE", profile.return_ratios.roe),
                                  ("ROCE", profile.return_ratios.roce),
                                  ("ROA", profile.return_ratios.roa)]:
                    if val:
                        st.markdown(f"- **{lbl}:** {val}")
            st.markdown("**Valuation Multiples**")
            for lbl, val in [("P/E", profile.pe_ratio), ("P/B", profile.pb_ratio),
                              ("EV/EBITDA", profile.ev_ebitda)]:
                if val:
                    st.markdown(f"- **{lbl}:** {val}")
        with km3:
            if profile.sector_kpis:
                st.markdown("**Sector-Specific KPIs**")
                for k, v in profile.sector_kpis.items():
                    if v:
                        st.markdown(f"- **{k.replace('_',' ').title()}:** {v}")

        # ── E: Peer Comparison ────────────────────────────────────────────────
        if profile.peer_comparison:
            st.markdown('<div class="section-header">Peer Comparison</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(profile.peer_comparison), use_container_width=True, hide_index=True)

        # ── F: Objects of Issue ───────────────────────────────────────────────
        if profile.objects_of_issue:
            st.markdown('<div class="section-header">Objects of the Issue</div>', unsafe_allow_html=True)
            obj1, obj2 = st.columns([1, 2])
            with obj1:
                amounts, labels = [], []
                for obj in profile.objects_of_issue:
                    m = re.search(r"[\d,]+\.?\d*", str(obj.get("amount", "") or "").replace(",", ""))
                    if m:
                        try:
                            amounts.append(float(m.group()))
                            labels.append((obj.get("purpose", "Other") or "Other")[:30])
                        except Exception:
                            pass
                if amounts:
                    fig3 = px.pie(values=amounts, names=labels, hole=0.45,
                                  color_discrete_sequence=px.colors.sequential.Blues_r)
                    fig3.update_layout(paper_bgcolor="#0e1117", font_color="#c7d0db",
                                       height=260, showlegend=True)
                    st.plotly_chart(fig3, use_container_width=True)
                else:
                    st.caption("Amount breakdown not specified.")
            with obj2:
                cat_icon = {"debt_repayment":"💳","expansion":"🏗️","working_capital":"⚙️",
                            "acquisition":"🤝","general_corporate":"📋"}
                for obj in profile.objects_of_issue:
                    icon = cat_icon.get(obj.get("category",""), "📌")
                    amt  = f" — **{obj['amount']}**" if obj.get("amount") else ""
                    st.markdown(
                        f'<div class="object-item">{icon} {obj.get("purpose","")}{amt}</div>',
                        unsafe_allow_html=True,
                    )

        # ── G: Management ─────────────────────────────────────────────────────
        if profile.key_management or profile.promoters:
            st.markdown('<div class="section-header">Management & Promoters</div>', unsafe_allow_html=True)
            mg1, mg2 = st.columns(2)
            with mg1:
                if profile.key_management:
                    st.markdown("**Key Management**")
                    for p in profile.key_management[:6]:
                        st.markdown(f'<span class="pill">👤 {p}</span>', unsafe_allow_html=True)
            with mg2:
                if profile.promoters:
                    st.markdown("**Promoters**")
                    for p in profile.promoters[:5]:
                        st.markdown(f'<span class="pill">🏢 {p}</span>', unsafe_allow_html=True)
            if profile.litigations_summary:
                st.markdown(f"\n**⚖️ Litigations:** {profile.litigations_summary}")

        # ── H: Risks ──────────────────────────────────────────────────────────
        if profile.key_risks:
            st.markdown('<div class="section-header">Key Risk Factors</div>', unsafe_allow_html=True)
            r1, r2 = st.columns(2)
            half = (len(profile.key_risks) + 1) // 2
            with r1:
                for risk in profile.key_risks[:half]:
                    st.markdown(f'<div class="risk-item">⚠️ {risk}</div>', unsafe_allow_html=True)
            with r2:
                for risk in profile.key_risks[half:]:
                    st.markdown(f'<div class="risk-item">⚠️ {risk}</div>', unsafe_allow_html=True)

        # ── I: Sentiment ──────────────────────────────────────────────────────
        if sentiment:
            st.markdown('<div class="section-header">Market Sentiment Analysis</div>', unsafe_allow_html=True)
            sg1, sg2, sg3 = st.columns([1, 1, 2])
            score = sentiment.get("score", 2.5)
            label = sentiment.get("sentiment_label", "Neutral")

            with sg1:
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number", value=score,
                    title={"text": "Sentiment Score", "font": {"color": "#c7d0db"}},
                    gauge={
                        "axis": {"range": [0, 5], "tickcolor": "#8b95a6"},
                        "bar":  {"color": "#4c8cf5"}, "bgcolor": "#1e2130",
                        "steps": [
                            {"range": [0,   1.8], "color": "#3d1515"},
                            {"range": [1.8, 2.6], "color": "#3d2e15"},
                            {"range": [2.6, 3.4], "color": "#1e2130"},
                            {"range": [3.4, 4.2], "color": "#15362a"},
                            {"range": [4.2, 5.0], "color": "#0f2a1e"},
                        ],
                    },
                    number={"font": {"color": "#e8ecf0"}},
                ))
                fig_g.update_layout(paper_bgcolor="#0e1117", font_color="#c7d0db",
                                    height=220, margin=dict(t=30, b=10))
                st.plotly_chart(fig_g, use_container_width=True)

            with sg2:
                st.metric("Overall Sentiment", label)
                if sentiment.get("gmp"):
                    st.metric("Grey Market Premium", sentiment["gmp"])
                if sentiment.get("subscription_estimate"):
                    st.markdown(f"**Subscription:** {sentiment['subscription_estimate']}")
                sources = sentiment.get("sources_used", [])
                if sources:
                    st.markdown("**Data from:** " + " · ".join(sources))

            with sg3:
                pos_col, neg_col = st.columns(2)
                with pos_col:
                    st.markdown("**✅ Positives**")
                    for p in (sentiment.get("positives") or [])[:5]:
                        st.markdown(f'<div class="positive-item">{p}</div>', unsafe_allow_html=True)
                with neg_col:
                    st.markdown("**⚠️ Concerns**")
                    for n in (sentiment.get("negatives") or [])[:5]:
                        st.markdown(f'<div class="negative-item">{n}</div>', unsafe_allow_html=True)

            if sentiment.get("summary"):
                st.markdown("**Analyst Summary**")
                for b in sentiment["summary"]:
                    st.markdown(f"- {b}")

            articles = sentiment.get("articles", [])
            if articles:
                with st.expander(f"📰 {len(articles)} source articles"):
                    for a in articles:
                        if a.get("url"):
                            st.markdown(f"- [{a['title']}]({a['url']}) — *{a.get('source','')}*")
                        else:
                            st.markdown(f"- {a.get('title','')} — *{a.get('source','')}*")

        if profile.extraction_warnings:
            with st.expander("⚠️ Extraction warnings"):
                for w in profile.extraction_warnings:
                    st.caption(w)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — Deep-Dive Chat
# ──────────────────────────────────────────────────────────────────────────────
with tab_chat:
    if not st.session_state.qa_chain:
        st.info("👈 Load an IPO from the sidebar to chat with its RHP document.")
    else:
        st.markdown(f"### Chat with **{st.session_state.ipo_name}** RHP")
        st.caption("Ask anything about the prospectus — financials, risks, management, use of funds…")

        # Quick-question buttons
        quick_qs = [
            "Summarise the key financials for the last 3 years",
            "What are the top 5 risk factors?",
            "How will the IPO proceeds be used?",
            "Who are the main competitors?",
            "What is the promoter background?",
        ]
        qcols = st.columns(len(quick_qs))
        for i, q in enumerate(quick_qs):
            if qcols[i].button(q[:34] + "…", key=f"q{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": q})

        st.divider()

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input(f"Ask about the {st.session_state.ipo_name} RHP…"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.spinner("Thinking…"):
                history = [
                    HumanMessage(content=m["content"]) if m["role"] == "user"
                    else AIMessage(content=m["content"])
                    for m in st.session_state.messages[:-1]
                ]
                result   = st.session_state.qa_chain.invoke({"input": prompt, "chat_history": history})
                response = result["answer"]

            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)