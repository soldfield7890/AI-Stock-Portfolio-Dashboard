import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


from src.scoring import score_portfolio
from src.loaders import load_csv
from src.ai_research import run_mini_trading_desk

# ----------------- Config -----------------
st.set_page_config(
    page_title="Oldfield AI Stock Dashboard",
    layout="wide",
)

LOGO_PATH = "assets/oldfield_logo.png"  # put your logo image here


# ----------------- Custom CSS – Liquid Glass Fintech -----------------
st.markdown(
    """
<style>

/* Global background & font */
.stApp {
    background: radial-gradient(circle at top left, #E0ECFF 0%, #F6F8FC 45%, #EAF1FF 100%) !important;
    color: #0F172A !important;
    font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

/* Main page container */
.block-container {
    padding-top: 1.0rem;
    padding-bottom: 2.0rem;
    max-width: 1400px;
}

/* Glass card base */
.glass-card {
    background: rgba(255, 255, 255, 0.96);
    border-radius: 20px;
    border: 1px solid rgba(148, 163, 184, 0.35);
    box-shadow: 0 18px 40px rgba(15, 23, 42, 0.10);
    backdrop-filter: blur(18px);
}

/* Header card – hero */
.header-card {
    padding: 18px 22px;
    margin-bottom: 10px;
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 40%, #0B1120 100%);
    color: #F9FAFB !important;
    border: 1px solid rgba(148,163,184,0.55);
}

/* Upload card */
.upload-card {
    padding: 12px 16px;
    margin-bottom: 16px;
}

/* KPI metric cards */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, rgba(255,255,255,0.98), rgba(239,246,255,0.96)) !important;
    border: 1px solid rgba(148, 163, 184, 0.40) !important;
    padding: 14px 16px !important;
    border-radius: 16px !important;
    box-shadow: 0px 10px 24px rgba(15,23,42,0.16);
}

/* Ensure metric text is dark and readable */
[data-testid="stMetricValue"],
[data-testid="stMetricLabel"],
[data-testid="stMetricDelta"] {
    color: #0F172A !important;
}

/* File uploader as soft glass */
section[data-testid="stFileUploader"] {
    background: rgba(15, 23, 42, 0.96) !important;
    border: 1px dashed rgba(148, 163, 184, 0.9) !important;
    border-radius: 14px !important;
    padding: 12px 14px !important;
    color: #E5E7EB !important;
}

/* Headings */
h1, h2, h3, h4 {
    color: #020617 !important;
    font-weight: 600 !important;
    letter-spacing: -0.3px;
}

/* Tabs – pill glass nav */
div[data-baseweb="tab-list"] {
    gap: 8px;
    padding: 6px 6px;
    border-radius: 999px;
    background: rgba(255,255,255,0.95);
    box-shadow: 0 14px 32px rgba(15,23,42,0.16);
    backdrop-filter: blur(18px);
    border: 1px solid rgba(148, 163, 184, 0.45);
    margin-bottom: 14px;
}

button[role="tab"] {
    border-radius: 999px !important;
    padding: 6px 16px !important;
    font-size: 0.9rem !important;
    color: #475569 !important;
    border: 1px solid transparent !important;
    background-color: transparent !important;
}

button[role="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #2563EB 0%, #38BDF8 100%) !important;
    color: #F9FAFB !important;
    border-color: transparent !Important;
}

/* Selectboxes & checkboxes – light glass controls */
.stSelectbox div[data-baseweb="select"] > div {
    background-color: rgba(255,255,255,0.96) !important;
    color: #0F172A !important;
    border-radius: 999px !important;
    border: 1px solid rgba(148,163,184,0.9) !important;
}
.stSelectbox svg {
    color: #64748B !important;
}

.stCheckbox label {
    color: #4B5563 !important;
    font-size: 0.8rem;
}

/* Small caption text */
.small-caption {
    font-size: 0.78rem;
    color: #6B7280;
}

/* Positive / negative helpers */
.positive-text {
    color: #16A34A;
    font-weight: 600;
}
.negative-text {
    color: #DC2626;
    font-weight: 600;
}

</style>
""",
    unsafe_allow_html=True,
)


# ----------------- Helper functions -----------------
def detect_symbol_column(df: pd.DataFrame):
    for col in df.columns:
        cl = col.lower()
        if "symbol" in cl or "ticker" in cl:
            return col
    return None


def fmt_percent(val, decimals=1):
    if pd.isna(val):
        return ""
    try:
        v = float(val)
    except Exception:
        return ""
    return f"{v:.{decimals}f}%"


def fmt_ratio_or_percent(val, decimals=1):
    if pd.isna(val):
        return ""
    try:
        v = float(val)
    except Exception:
        return ""
    if abs(v) <= 1:
        v = v * 100
    return f"{v:.{decimals}f}%"


def fmt_marketcap(val):
    if pd.isna(val):
        return ""
    try:
        v = float(val)
    except Exception:
        return ""
    abs_v = abs(v)
    if abs_v >= 1_000_000_000_000:
        return f"${v/1_000_000_000_000:.2f}T"
    if abs_v >= 1_000_000_000:
        return f"${v/1_000_000_000:.2f}B"
    if abs_v >= 1_000_000:
        return f"${v/1_000_000:.2f}M"
    return f"${v:,.0f}"


def fmt_currency(val, decimals=0):
    if pd.isna(val):
        return ""
    try:
        v = float(val)
    except Exception:
        return ""
    return f"${v:,.{decimals}f}"


def fmt_number(val, decimals=2):
    if pd.isna(val):
        return ""
    try:
        v = float(val)
    except Exception:
        return ""
    return f"{v:,.{decimals}f}"


def highlight_pl(val):
    if pd.isna(val):
        return ""
    try:
        v = float(val)
    except Exception:
        return ""
    if v > 0:
        return "color: #16A34A; font-weight:600;"
    else:
        return "color: #DC2626; font-weight:600;"


def highlight_score(val):
    if pd.isna(val):
        return ""
    try:
        v = float(val)
    except Exception:
        return ""
    if v >= 70:
        return "background-color: rgba(220, 252, 231, 0.95); color:#14532D;"
    elif v >= 50:
        return "background-color: rgba(254, 249, 195, 0.95); color:#78350F;"
    else:
        return "background-color: rgba(254, 226, 226, 0.95); color:#7F1D1D;"


def highlight_decision(val):
    if isinstance(val, str):
        v = val.lower()
    else:
        return ""
    if "strong buy" in v:
        return "background-color: rgba(22,163,74,0.12); color:#166534; font-weight:600;"
    if v == "buy":
        return "background-color: rgba(34,197,94,0.10); color:#15803D; font-weight:500;"
    if v == "hold":
        return "background-color: rgba(148,163,184,0.18); color:#0F172A;"
    if v == "trim":
        return "background-color: rgba(250,204,21,0.18); color:#854D0E; font-weight:500;"
    if "exit" in v or "avoid" in v:
        return "background-color: rgba(248,113,113,0.25); color:#7F1D1D; font-weight:600;"
    return ""


def style_table(df: pd.DataFrame, highlight_pl_cols=False, highlight_score_col=False, highlight_decision_col=False):
    format_dict = {}
    for col in df.columns:
        cl = col.lower()
        if cl in ("pe_ttm", "forwardpe"):
            format_dict[col] = lambda v, c=col: fmt_number(v, 2)
        elif "dividendyield" in cl:
            format_dict[col] = lambda v, c=col: fmt_ratio_or_percent(v, 2)
        elif "profitmargin" in cl:
            format_dict[col] = lambda v, c=col: fmt_ratio_or_percent(v, 1)
        elif "marketcap" in cl:
            format_dict[col] = fmt_marketcap
        elif "beta" in cl:
            format_dict[col] = lambda v, c=col: fmt_number(v, 2)
        elif "score" in cl:
            format_dict[col] = lambda v, c=col: fmt_number(v, 0)
        elif "portfolioweightpct" in cl:
            format_dict[col] = lambda v, c=col: fmt_percent(v, 2)
        elif "unrealizedplpct" in cl:
            format_dict[col] = lambda v, c=col: fmt_percent(v, 1)
        elif "unrealizedpl" in cl:
            format_dict[col] = lambda v, c=col: fmt_currency(v, 0)
        elif "currentvalue_clean" in cl or "costbasis_clean" in cl:
            format_dict[col] = lambda v, c=col: fmt_currency(v, 0)

    styler = df.style.format(format_dict, na_rep="")

    if highlight_pl_cols:
        pl_cols = [c for c in df.columns if "unrealizedpl" in c.lower()]
        if pl_cols:
            styler = styler.applymap(highlight_pl, subset=pl_cols)

    if highlight_score_col and "Score" in df.columns:
        styler = styler.applymap(highlight_score, subset=["Score"])

    if highlight_decision_col and "Decision" in df.columns:
        styler = styler.applymap(highlight_decision, subset=["Decision"])

    return styler


# ----------------- Overview Analytics -----------------
def compute_overview_analytics(scored_df: pd.DataFrame):
    symbol_col = detect_symbol_column(scored_df)
    metrics = {}

    if "CurrentValue_clean" in scored_df.columns:
        metrics["total_value"] = float(scored_df["CurrentValue_clean"].sum())
    else:
        metrics["total_value"] = None

    if "UnrealizedPL" in scored_df.columns:
        metrics["total_pl"] = float(scored_df["UnrealizedPL"].sum())
    else:
        metrics["total_pl"] = None

    if "Score" in scored_df.columns:
        metrics["avg_score"] = float(scored_df["Score"].mean())
        if "CurrentValue_clean" in scored_df.columns:
            w = scored_df["CurrentValue_clean"].fillna(0)
            if w.sum() > 0:
                metrics["weighted_score"] = float((scored_df["Score"] * w).sum() / w.sum())
            else:
                metrics["weighted_score"] = None
        else:
            metrics["weighted_score"] = None
    else:
        metrics["avg_score"] = None
        metrics["weighted_score"] = None

    if "Score" in scored_df.columns:
        buckets = {
            "Speculative (<40)": (scored_df["Score"] < 40).sum(),
            "Middle (40–59)": ((scored_df["Score"] >= 40) & (scored_df["Score"] <= 59)).sum(),
            "Core (60+)": (scored_df["Score"] >= 60).sum(),
        }
        metrics["score_buckets"] = buckets
    else:
        metrics["score_buckets"] = {}

    if symbol_col and "UnrealizedPL" in scored_df.columns:
        tmp = scored_df[[symbol_col, "UnrealizedPL"]].dropna()
        metrics["top_winners"] = tmp.sort_values("UnrealizedPL", ascending=False).head(3)
        metrics["top_losers"] = tmp.sort_values("UnrealizedPL", ascending=True).head(3)
    else:
        metrics["top_winners"] = pd.DataFrame()
        metrics["top_losers"] = pd.DataFrame()

    return metrics


# ----------------- AI Glance Card -----------------
def render_ai_glance_card(ai_view: Dict[str, Any]):
    """Pretty, color-coded AI Trading Desk card."""

    if not ai_view:
        st.info("Run the Mini Trading Desk to see AI output.")
        return

    # ---- Unpack data safely ----
    ticker = ai_view.get("ticker", "—")
    decision_raw = ai_view.get("final_decision") or "No decision"
    decision_clean = decision_raw.replace("_", " ").title()
    conviction = ai_view.get("conviction_score")
    horizon = ai_view.get("time_horizon") or "n/a"
    bucket = ai_view.get("bucket_view") or "Unknown"
    primary_action = ai_view.get("primary_action") or "—"

    fundamentals = ai_view.get("fundamental_view") or []
    technicals = ai_view.get("technical_view") or []
    sentiment = ai_view.get("sentiment_view") or []
    risks = ai_view.get("risk_factors") or []

    wl = ai_view.get("watchlist") or {}
    wl_flag = bool(wl.get("add_to_watchlist", False))
    wl_bucket = wl.get("watchlist_bucket", "None")
    wl_notes = wl.get("notes", "")

    next_actions = ai_view.get("next_actions") or {}
    add_on_dip = next_actions.get("add_on_dip_level")
    trim_above = next_actions.get("trim_above_level")
    hard_exit = next_actions.get("hard_exit_level")
    sizing_note = next_actions.get("position_sizing_note")

    # ---- Decision chip color map ----
    decision_palette = {
        "Strong Buy": ("#16a34a", "rgba(22,163,74,0.14)"),
        "Buy": ("#10b981", "rgba(16,185,129,0.14)"),
        "Hold": ("#6366f1", "rgba(99,102,241,0.14)"),
        "Trim": ("#facc15", "rgba(250,204,21,0.20)"),
        "Exit": ("#ef4444", "rgba(239,68,68,0.20)"),
        "Exit / Avoid": ("#ef4444", "rgba(239,68,68,0.20)"),
        "No_Api_Key": ("#6b7280", "rgba(148,163,184,0.30)"),
        "No Decision": ("#6b7280", "rgba(148,163,184,0.30)"),
    }

    if decision_raw.upper() == "NO_API_KEY":
        palette_key = "No_Api_Key"
    else:
        palette_key = decision_clean

    decision_color, decision_bg = decision_palette.get(
        palette_key, ("#0f172a", "rgba(15,23,42,0.06)")
    )

    # ---- Conviction badge ----
    conviction_label = "—"
    conviction_badge_html = ""

    if conviction is not None:
        conviction_label = f"{conviction:.0f}/100"
        if conviction >= 75:
            cv_color, cv_bg = "#16a34a", "rgba(22,163,74,0.14)"
        elif conviction >= 50:
            cv_color, cv_bg = "#eab308", "rgba(234,179,8,0.18)"
        else:
            cv_color, cv_bg = "#ef4444", "rgba(239,68,68,0.20)"

        conviction_badge_html = f"""
            <span style="
                border-radius:999px;
                padding:3px 10px;
                font-size:0.75rem;
                background:{cv_bg};
                color:{cv_color};
                margin-left:8px;
            ">
                Conviction {conviction_label}
            </span>
        """

    # ---- Card wrapper ----
    st.markdown(
        '<div class="glass-card" style="padding:18px 22px; margin-top:10px;">',
        unsafe_allow_html=True,
    )

    # =======================
    #   TOP: Header row
    # =======================
    top_l, top_r = st.columns([3, 1])

    with top_l:
        st.markdown(
            f"""
            <div style="display:flex; flex-direction:column; gap:4px;">
              <div style="font-size:0.9rem; color:#6B7280;">AI Trading Desk View</div>
              <div style="display:flex; align-items:center; gap:10px; flex-wrap:wrap;">
                <div style="font-size:1.1rem; font-weight:600; color:#0F172A;">
                  {ticker} – {decision_clean}
                </div>
                <span style="
                    border-radius:999px;
                    padding:4px 10px;
                    font-size:0.75rem;
                    background:{decision_bg};
                    color:{decision_color};
                    ">
                  {decision_clean}
                </span>
                {conviction_badge_html}
              </div>
              <div style="font-size:0.85rem; color:#6B7280;">
                Conviction: {conviction_label}
                · Horizon: {horizon}
                · Bucket: {bucket}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with top_r:
        st.markdown(
            f"""
            <div style="display:flex; justify-content:flex-end;">
              <span style="
                  border-radius:999px;
                  padding:4px 12px;
                  font-size:0.75rem;
                  background:rgba(15,23,42,0.04);
                  color:#4B5563;">
                Primary action:
                <span style="font-weight:600;">{primary_action}</span>
              </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # =======================
    #   MID: 3 analysis columns
    # =======================

    def render_bullets(title: str, items: list[str]):
        st.markdown(f"**{title}**")
        if items:
            for txt in items:
                st.markdown(f"- {txt}")
        else:
            st.caption("No insight yet.")

    mid1, mid2, mid3 = st.columns(3)
    with mid1:
        render_bullets("Fundamentals", fundamentals)
    with mid2:
        render_bullets("Trend & Technicals", technicals)
    with mid3:
        render_bullets("Sentiment & Story", sentiment)

    st.markdown(
        "<hr style='border:none; border-top:1px solid rgba(148,163,184,0.35); "
        "margin:12px 0 16px;'>",
        unsafe_allow_html=True,
    )

    # =======================
    #   BOTTOM: Risks + Watchlist
    # =======================
    bottom_l, bottom_r = st.columns([2, 1])

    with bottom_l:
        st.markdown("**Key Risks**")
        if risks:
            for r in risks:
                st.markdown(f"- {r}")
        else:
            st.caption("No specific risks highlighted yet.")

        st.markdown("**Next Actions**")
        # Keep these as plain text; no raw HTML.
        st.markdown(f"- Add on dip: {add_on_dip or '—'}")
        st.markdown(f"- Trim above: {trim_above or '—'}")
        st.markdown(f"- Hard exit: {hard_exit or '—'}")
        if sizing_note:
            st.markdown(f"- {sizing_note}")

    with bottom_r:
        st.markdown("**Watchlist**")
        st.markdown(f"- Add to watchlist: **{'Yes' if wl_flag else 'No'}**")
        st.markdown(f"- Bucket: **{wl_bucket}**")
        if wl_notes:
            st.markdown(f"- Notes: {wl_notes}")
        st.caption("AI view generated by your Mini Trading Desk engine.")

    st.markdown("</div>", unsafe_allow_html=True)


# ----------------- Tab renderers -----------------
def render_overview(scored_df: pd.DataFrame):
    metrics = compute_overview_analytics(scored_df)
    symbol_col = detect_symbol_column(scored_df)

    # ---------- TOP: Glass summary band (KPIs + Health gauge) ----------
    st.markdown(
        '<div class="glass-card" style="padding:18px 22px; margin-top:4px; margin-bottom:18px;">',
        unsafe_allow_html=True,
    )

    top_left, top_right = st.columns([2, 1.1])

    # Left side – KPI tiles
    with top_left:
        st.markdown(
            """
            <div style="display:flex; flex-direction:column; gap:10px;">
              <div style="font-size:0.9rem; color:#6B7280;">Portfolio Snapshot</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        k1, k2, k3, k4 = st.columns(4)

        k1.metric(
            "Total Portfolio Value",
            fmt_currency(metrics["total_value"], 0) if metrics["total_value"] is not None else "—",
        )
        k2.metric(
            "Total Unrealized P/L",
            fmt_currency(metrics["total_pl"], 0) if metrics["total_pl"] is not None else "—",
        )
        k3.metric("Positions", len(scored_df))
        if metrics["avg_score"] is not None:
            k4.metric("Avg Score", f"{metrics['avg_score']:.1f}")
        else:
            k4.metric("Avg Score", "—")

        # Second KPI row: weighted score + mix
        c1, c2, c3, c4 = st.columns(4)
        if metrics.get("weighted_score") is not None:
            c1.metric("Value-Weighted Score", f"{metrics['weighted_score']:.1f}")
        else:
            c1.metric("Value-Weighted Score", "—")

        buckets = metrics.get("score_buckets", {})
        c2.metric("Speculative (<40)", buckets.get("Speculative (<40)", 0))
        c3.metric("Middle (40–59)", buckets.get("Middle (40–59)", 0))
        c4.metric("Core (60+)", buckets.get("Core (60+)", 0))

    # Right side – Health gauge
    with top_right:
        base_score = metrics.get("weighted_score") or metrics.get("avg_score")
        if base_score is not None:
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=base_score,
                    number={"suffix": " / 100", "font": {"size": 22, "color": "#0F172A"}},
                    title={"text": "Portfolio Health", "font": {"size": 14, "color": "#4B5563"}},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#2563EB"},
                        "bgcolor": "rgba(241,245,249,0.9)",
                        "borderwidth": 0,
                        "steps": [
                            {"range": [0, 40], "color": "rgba(248,113,113,0.45)"},
                            {"range": [40, 60], "color": "rgba(234,179,8,0.45)"},
                            {"range": [60, 100], "color": "rgba(34,197,94,0.45)"},
                        ],
                    },
                )
            )
            fig_gauge.update_layout(
                margin=dict(l=10, r=10, t=40, b=0),
                paper_bgcolor="rgba(255,255,255,0.0)",
            )
            st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})
        else:
            st.caption("Portfolio health gauge will appear once scores are available.")

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- MIDDLE: Score mix + Action summary tiles ----------
    st.markdown(
        '<div class="glass-card" style="padding:18px 22px; margin-bottom:18px;">',
        unsafe_allow_html=True,
    )

    m1, m2 = st.columns(2)

    # Score mix as mini bar chart
    with m1:
        st.markdown("#### Score Mix")
        buckets = metrics.get("score_buckets", {})
        if buckets:
            mix_df = pd.DataFrame(
                {
                    "Bucket": list(buckets.keys()),
                    "Count": list(buckets.values()),
                }
            )
            fig_mix = px.bar(
                mix_df,
                x="Bucket",
                y="Count",
                text="Count",
                title="",
            )
            fig_mix.update_traces(textposition="outside")
            fig_mix.update_layout(
                margin=dict(l=0, r=0, t=10, b=0),
                plot_bgcolor="rgba(255,255,255,0.96)",
                paper_bgcolor="rgba(255,255,255,0.0)",
                font=dict(size=11, color="#0F172A"),
            )
            st.plotly_chart(fig_mix, use_container_width=True, config={"displayModeBar": False})
        else:
            st.caption("Scores not available yet.")

    # Action summary (Strong buy / buy / hold / trim / exit)
    with m2:
        st.markdown("#### Action Summary")
        if "Decision" in scored_df.columns:
            decision_counts = scored_df["Decision"].value_counts().to_dict()
            r1, r2, r3, r4, r5 = st.columns(5)
            r1.metric("Strong Buy", decision_counts.get("Strong Buy", 0))
            r2.metric("Buy", decision_counts.get("Buy", 0))
            r3.metric("Hold", decision_counts.get("Hold", 0))
            r4.metric("Trim", decision_counts.get("Trim", 0))
            r5.metric("Exit / Avoid", decision_counts.get("Exit / Avoid", 0))
        else:
            st.caption("No decision column in data – scoring step may need adjustment.")

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- LOWER: Key charts row ----------
    st.markdown(
        '<div class="glass-card" style="padding:18px 22px; margin-bottom:18px;">',
        unsafe_allow_html=True,
    )
    st.markdown("#### Key Charts")

    col_a, col_b = st.columns(2)

    # Score distribution
    with col_a:
        if "Score" in scored_df.columns:
            score_series = scored_df["Score"].dropna()
            if not score_series.empty:
                fig_score = px.histogram(score_series, nbins=15, title="Score Distribution")
                fig_score.update_traces(marker=dict(line=dict(width=0)))
                fig_score.update_layout(
                    margin=dict(l=0, r=0, t=30, b=0),
                    plot_bgcolor="rgba(255,255,255,0.96)",
                    paper_bgcolor="rgba(255,255,255,0.0)",
                    font=dict(size=11, color="#0F172A"),
                )
                st.plotly_chart(fig_score, use_container_width=True, config={"displayModeBar": False})
            else:
                st.caption("No scores available.")
        else:
            st.caption("No scores available.")

    # Allocation by decision
    with col_b:
        if "PortfolioWeightPct" in scored_df.columns and "Decision" in scored_df.columns:
            alloc = (
                scored_df.dropna(subset=["PortfolioWeightPct"])
                .groupby("Decision")["PortfolioWeightPct"]
                .sum()
                .reset_index()
            )
            if not alloc.empty:
                fig_alloc = px.pie(
                    alloc,
                    values="PortfolioWeightPct",
                    names="Decision",
                    title="Allocation by Decision",
                )
                fig_alloc.update_layout(
                    margin=dict(l=0, r=0, t=30, b=0),
                    paper_bgcolor="rgba(255,255,255,0.0)",
                    font=dict(size=11, color="#0F172A"),
                )
                st.plotly_chart(fig_alloc, use_container_width=True, config={"displayModeBar": False})
            else:
                st.caption("No allocation data.")
        else:
            st.caption("No allocation data.")

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- BOTTOM: Unrealized P/L focus & Top movers ----------
    st.markdown(
        '<div class="glass-card" style="padding:18px 22px; margin-bottom:8px;">',
        unsafe_allow_html=True,
    )

    st.markdown("#### Unrealized P/L & Today’s Focus")

    u1, u2 = st.columns([2, 1])

    # Unrealized P/L by ticker
    with u1:
        if symbol_col and "UnrealizedPL" in scored_df.columns:
            pl_df = scored_df[[symbol_col, "UnrealizedPL"]].copy().dropna(subset=["UnrealizedPL"])
            if not pl_df.empty:
                fig_pl = px.bar(pl_df, x=symbol_col, y="UnrealizedPL", title="Unrealized P/L by Ticker")
                fig_pl.update_layout(
                    margin=dict(l=0, r=0, t=30, b=0),
                    plot_bgcolor="rgba(255,255,255,0.96)",
                    paper_bgcolor="rgba(255,255,255,0.0)",
                    xaxis_title="",
                    yaxis_title="Unrealized P/L",
                    font=dict(size=11, color="#0F172A"),
                )
                st.plotly_chart(fig_pl, use_container_width=True, config={"displayModeBar": False})
            else:
                st.caption("No unrealized P/L data.")
        else:
            st.caption("No unrealized P/L data.")

    # Today’s Focus – top winners/losers bullets
    with u2:
        winners = metrics["top_winners"]
        losers = metrics["top_losers"]
        if not winners.empty:
            st.markdown("**Top Winners**")
            for _, r in winners.iterrows():
                st.markdown(f"- **{r[symbol_col]}** up {fmt_currency(r['UnrealizedPL'],0)} unrealized")
        else:
            st.caption("No winners yet.")

        st.markdown("---")

        if not losers.empty:
            st.markdown("**Top Losers**")
            for _, r in losers.iterrows():
                st.markdown(f"- **{r[symbol_col]}** down {fmt_currency(r['UnrealizedPL'],0)} unrealized")
        else:
            st.caption("No losers yet.")

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- AI Desk quick glance ----------
    st.markdown("### AI Desk – Quick Glance")

    if symbol_col is None:
        st.info("No ticker/symbol column found for AI desk.")
        return

    tickers = sorted(scored_df[symbol_col].dropna().unique())
    col_left, col_right = st.columns([1, 2])

    with col_left:
        selected_ticker = st.selectbox(
            "Select ticker for AI check",
            options=tickers,
            index=0,
            key="ai_glance_ticker",
        )
        mode = st.radio(
            "Depth",
            options=["lite", "full"],
            index=0,
            horizontal=True,
            key="ai_glance_mode",
        )
        run_btn = st.button("Run Mini Trading Desk", key="ai_glance_run")

    with col_right:
        if run_btn and selected_ticker:
            row = scored_df[scored_df[symbol_col] == selected_ticker].iloc[0]
            with st.spinner("Thinking like a 4-role trading desk..."):
                ai_view = run_mini_trading_desk(
                    selected_ticker,
                    row,
                    mode=mode,
                )
            render_ai_glance_card(ai_view)



def render_positions(scored_df: pd.DataFrame):
    st.subheader("Positions (raw view)")
    styled = style_table(scored_df, highlight_pl_cols=True, highlight_score_col=True, highlight_decision_col=True)
    st.dataframe(styled, use_container_width=True, height=560)


def render_fundamentals(scored_df: pd.DataFrame):
    st.subheader("Fundamentals (raw snapshot)")
    cols = []
    symbol_col = detect_symbol_column(scored_df)
    if symbol_col:
        cols.append(symbol_col)
    for c in ["PE_TTM", "ForwardPE", "DividendYield", "ProfitMargin", "MarketCap", "Beta", "Score", "Decision"]:
        if c in scored_df.columns and c not in cols:
            cols.append(c)
    if not cols:
        st.info("No fundamentals columns detected.")
        return
    df = scored_df[cols].copy()
    styled = style_table(df, highlight_score_col=True, highlight_decision_col=True)
    st.dataframe(styled, use_container_width=True, height=560)


def render_signals(scored_df: pd.DataFrame):
    st.subheader("Signals (basic)")
    if "Decision" not in scored_df.columns:
        st.info("No Decision column found.")
        return
    decision_counts = scored_df["Decision"].value_counts().to_dict()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Strong Buy", decision_counts.get("Strong Buy", 0))
    c2.metric("Buy", decision_counts.get("Buy", 0))
    c3.metric("Hold", decision_counts.get("Hold", 0))
    c4.metric("Trim", decision_counts.get("Trim", 0))
    c5.metric("Exit / Avoid", decision_counts.get("Exit / Avoid", 0))

    symbol_col = detect_symbol_column(scored_df)

    def subset(decisions):
        return scored_df[scored_df["Decision"].isin(decisions)].copy()

    strong_buy = subset(["Strong Buy"])
    buy = subset(["Buy"])
    trim = subset(["Trim"])
    exit_df = subset(["Exit / Avoid"])

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Strong Buy**")
        if strong_buy.empty:
            st.caption("None.")
        else:
            cols = [symbol_col, "Score", "PortfolioWeightPct", "UnrealizedPLPct"] if symbol_col else ["Score"]
            cols = [c for c in cols if c in strong_buy.columns]
            styled = style_table(strong_buy[cols], highlight_pl_cols=True, highlight_score_col=True)
            st.dataframe(styled, use_container_width=True, height=260)
    with col2:
        st.markdown("**Buy**")
        if buy.empty:
            st.caption("None.")
        else:
            cols = [symbol_col, "Score", "PortfolioWeightPct", "UnrealizedPLPct"] if symbol_col else ["Score"]
            cols = [c for c in cols if c in buy.columns]
            styled = style_table(buy[cols], highlight_pl_cols=True, highlight_score_col=True)
            st.dataframe(styled, use_container_width=True, height=260)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Trim**")
        if trim.empty:
            st.caption("None.")
        else:
            cols = [symbol_col, "Score", "PortfolioWeightPct", "UnrealizedPLPct"] if symbol_col else ["Score"]
            cols = [c for c in cols if c in trim.columns]
            styled = style_table(trim[cols], highlight_pl_cols=True, highlight_score_col=True)
            st.dataframe(styled, use_container_width=True, height=260)
    with col4:
        st.markdown("**Exit / Avoid**")
        if exit_df.empty:
            st.caption("None.")
        else:
            cols = [symbol_col, "Score", "PortfolioWeightPct", "UnrealizedPLPct"] if symbol_col else ["Score"]
            cols = [c for c in cols if c in exit_df.columns]
            styled = style_table(exit_df[cols], highlight_pl_cols=True, highlight_score_col=True)
            st.dataframe(styled, use_container_width=True, height=260)


# ----------------- Header with logo -----------------
st.markdown('<div class="glass-card header-card">', unsafe_allow_html=True)
hcol1, hcol2 = st.columns([4, 1])
with hcol1:
    st.markdown(
        """
        <div>
          <h1 style="margin-bottom:4px; color:#F9FAFB;">Oldfield AI Stock Dashboard</h1>
          <p style="margin:0; color:#CBD5F5; font-size:0.9rem;">
            Liquid-glass cockpit for your AI-scored portfolio — overview, signals, and AI desk in one view.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with hcol2:
    try:
        st.image(LOGO_PATH, width=80)
    except Exception:
        pass
    st.markdown(
        """
        <div style="text-align:right; margin-top:4px;">
          <span style="font-size:0.8rem; padding:6px 10px; border-radius:999px;
                       background:rgba(37,99,235,0.25); color:#E0EAFF;
                       border:1px solid rgba(129,140,248,0.9);">
            v1.2 • Overview Focus
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
st.markdown("</div>", unsafe_allow_html=True)


# ----------------- Upload + Main -----------------
st.markdown('<div class="glass-card upload-card">', unsafe_allow_html=True)
upload_col, right_col = st.columns([2, 3])
with upload_col:
    uploaded_file = st.file_uploader("Upload portfolio CSV", type=["csv"])
with right_col:
    st.markdown(
        """
        <p style="margin-top:4px; font-size:0.8rem; color:#4B5563;">
        Keep a single positions CSV in your drive, overwrite it, and reload this app to re-score
        the whole portfolio in one click.
        </p>
        """,
        unsafe_allow_html=True,
    )
st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is None:
    st.info("Upload a CSV to unlock the dashboard views.")
else:
    raw_df = load_csv(uploaded_file)
    scored_df = score_portfolio(raw_df)

    scored_csv = scored_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download scored portfolio as CSV",
        data=scored_csv,
        file_name="scored_portfolio.csv",
        mime="text/csv",
        key="download_scored_csv",
    )

    tabs = st.tabs(["Overview", "Positions", "Fundamentals", "Signals"])

    with tabs[0]:
        render_overview(scored_df)
    with tabs[1]:
        render_positions(scored_df)
    with tabs[2]:
        render_fundamentals(scored_df)
    with tabs[3]:
        render_signals(scored_df)
