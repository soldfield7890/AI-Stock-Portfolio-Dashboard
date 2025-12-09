import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from src.scoring import score_portfolio
from src.loaders import load_csv

# ----------------- Config -----------------
st.set_page_config(
    page_title="Oldfield AI Stock Dashboard",
    layout="wide",
)

LOGO_PATH = "assets/oldfield_logo.png"  # put your clover logo here


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


# ----------------- Analytics helpers -----------------
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


# ----------------- Page renderers -----------------
def render_overview(scored_df: pd.DataFrame):
    st.markdown('<div class="glass-card" style="padding:18px 20px;">', unsafe_allow_html=True)
    st.subheader("Portfolio Overview")

    metrics = compute_overview_analytics(scored_df)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Portfolio Value", fmt_currency(metrics["total_value"], 0) if metrics["total_value"] is not None else "—")
    c2.metric("Total Unrealized P/L", fmt_currency(metrics["total_pl"], 0) if metrics["total_pl"] is not None else "—")
    c3.metric("Positions", len(scored_df))

    if metrics["avg_score"] is not None:
        c4, c5, c6 = st.columns(3)
        c4.metric("Average Score", f"{metrics['avg_score']:.1f}")
        if metrics["weighted_score"] is not None:
            c5.metric("Value-Weighted Score", f"{metrics['weighted_score']:.1f}")
        if "PortfolioWeightPct" in scored_df.columns:
            top_weight = scored_df.sort_values("PortfolioWeightPct", ascending=False).head(1)
            symbol_col = detect_symbol_column(scored_df)
            if not top_weight.empty and symbol_col:
                row = top_weight.iloc[0]
                c6.metric(
                    "Largest Position",
                    f"{row[symbol_col]}",
                    help=f"{fmt_percent(row['PortfolioWeightPct'], 2)} of portfolio",
                )

    buckets = metrics.get("score_buckets", {})
    if buckets:
        st.markdown("#### Score Mix")
        b1, b2, b3 = st.columns(3)
        b1.metric("Speculative (<40)", buckets.get("Speculative (<40)", 0))
        b2.metric("Middle (40–59)", buckets.get("Middle (40–59)", 0))
        b3.metric("Core (60+)", buckets.get("Core (60+)", 0))

    if "Decision" in scored_df.columns:
        st.markdown("#### Action Summary")
        decision_counts = scored_df["Decision"].value_counts().to_dict()
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Strong Buy", decision_counts.get("Strong Buy", 0))
        c2.metric("Buy", decision_counts.get("Buy", 0))
        c3.metric("Hold", decision_counts.get("Hold", 0))
        c4.metric("Trim", decision_counts.get("Trim", 0))
        c5.metric("Exit / Avoid", decision_counts.get("Exit / Avoid", 0))

    st.markdown("---")

    st.markdown("#### Key Charts")
    col_a, col_b = st.columns(2)

    if "Score" in scored_df.columns:
        score_series = scored_df["Score"].dropna()
        if not score_series.empty:
            fig_score = px.histogram(score_series, nbins=15, title="Score Distribution")
            fig_score.update_traces(marker=dict(line=dict(width=0)))
            fig_score.update_layout(
                margin=dict(l=10, r=10, t=40, b=10),
                plot_bgcolor="rgba(255,255,255,0.96)",
                paper_bgcolor="rgba(255,255,255,0.0)",
                font=dict(size=12, color="#0F172A"),
            )
            col_a.plotly_chart(fig_score, use_container_width=True)

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
                margin=dict(l=10, r=10, t=40, b=10),
                paper_bgcolor="rgba(255,255,255,0.0)",
                font=dict(size=12, color="#0F172A"),
            )
            col_b.plotly_chart(fig_alloc, use_container_width=True)

    st.markdown("#### Unrealized P/L by Position")
    symbol_col = detect_symbol_column(scored_df)
    if symbol_col and "UnrealizedPL" in scored_df.columns:
        pl_df = scored_df[[symbol_col, "UnrealizedPL"]].copy().dropna(subset=["UnrealizedPL"])
        if not pl_df.empty:
            fig_pl = px.bar(pl_df, x=symbol_col, y="UnrealizedPL", title="Unrealized P/L by Ticker")
            fig_pl.update_layout(
                margin=dict(l=10, r=10, t=40, b=10),
                plot_bgcolor="rgba(255,255,255,0.96)",
                paper_bgcolor="rgba(255,255,255,0.0)",
                xaxis_title="Ticker",
                yaxis_title="Unrealized P/L",
                font=dict(size=12, color="#0F172A"),
            )
            st.plotly_chart(fig_pl, use_container_width=True)

    st.markdown("#### Today’s Focus")
    winners = metrics["top_winners"]
    losers = metrics["top_losers"]
    if not winners.empty or not losers.empty:
        c1, c2 = st.columns(2)
        if not winners.empty:
            c1.markdown("**Top Winners**")
            for _, r in winners.iterrows():
                c1.markdown(f"- **{r[symbol_col]}** up {fmt_currency(r['UnrealizedPL'],0)} unrealized")
        if not losers.empty:
            c2.markdown("**Top Losers**")
            for _, r in losers.iterrows():
                c2.markdown(f"- **{r[symbol_col]}** down {fmt_currency(r['UnrealizedPL'],0)} unrealized")

    st.markdown("</div>", unsafe_allow_html=True)


def render_positions(scored_df: pd.DataFrame, raw_df: pd.DataFrame):
    st.markdown('<div class="glass-card" style="padding:18px 20px;">', unsafe_allow_html=True)
    st.subheader("Positions & Metrics")

    symbol_col = detect_symbol_column(scored_df)

    col1, col2, col3, col4 = st.columns([1.1, 1, 1, 1.2])

    decision_options = sorted(scored_df["Decision"].dropna().unique()) if "Decision" in scored_df.columns else []
    default_decisions = decision_options
    selected_decisions = col1.multiselect(
        "Decision filter",
        options=decision_options,
        default=default_decisions,
        key="positions_decisions_ms",
    )

    if "Score" in scored_df.columns and not scored_df["Score"].dropna().empty:
        min_score = int(scored_df["Score"].min())
        max_score = int(scored_df["Score"].max())
        score_range = col2.slider(
            "Score range",
            min_value=min_score,
            max_value=max_score,
            value=(min_score, max_score),
            key="positions_score_range",
        )
    else:
        score_range = None

    if "PortfolioWeightPct" in scored_df.columns:
        max_w = float(scored_df["PortfolioWeightPct"].fillna(0).max())
        weight_min = col3.slider(
            "Min weight (%)",
            min_value=0.0,
            max_value=float(round(max_w, 2)) if max_w > 0 else 10.0,
            value=0.0,
            step=0.25,
            key="positions_weight_min",
        )
    else:
        weight_min = 0.0

    search_text = col4.text_input(
        "Search ticker/description",
        value="",
        key="positions_search_text",
    ).strip().lower()

    df = scored_df.copy()

    if "Decision" in df.columns and selected_decisions:
        df = df[df["Decision"].isin(selected_decisions)]

    if score_range and "Score" in df.columns:
        df = df[(df["Score"] >= score_range[0]) & (df["Score"] <= score_range[1])]

    if "PortfolioWeightPct" in df.columns:
        df = df[df["PortfolioWeightPct"].fillna(0) >= weight_min]

    if search_text:
        mask = pd.Series([False] * len(df))
        if symbol_col:
            mask = mask | df[symbol_col].astype(str).str.lower().str.contains(search_text)
        if "Description" in df.columns:
            mask = mask | df["Description"].astype(str).str.lower().str.contains(search_text)
        df = df[mask]

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        numeric_cols = list(df.columns)

    default_sort_index = 0
    if "Score" in numeric_cols:
        default_sort_index = numeric_cols.index("Score")

    sort_col = st.selectbox(
        "Sort column",
        options=numeric_cols,
        index=min(default_sort_index, len(numeric_cols) - 1),
        key="positions_sort_col",
    )
    sort_ascending = st.checkbox(
        "Sort ascending?",
        value=False,
        key="positions_sort_ascending",
    )

    df_sorted = df.sort_values(by=sort_col, ascending=sort_ascending)

    styled = style_table(
        df_sorted,
        highlight_pl_cols=True,
        highlight_score_col=True,
        highlight_decision_col=True,
    )
    st.dataframe(styled, use_container_width=True, height=520)

    st.markdown(
        '<p class="small-caption">Filter by decision, score, weight, and search. Click headers to sort interactively.</p>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def render_fundamentals(scored_df: pd.DataFrame):
    st.markdown('<div class="glass-card" style="padding:18px 20px;">', unsafe_allow_html=True)
    st.subheader("Fundamentals Snapshot")

    symbol_col = detect_symbol_column(scored_df)

    cols = []
    if symbol_col:
        cols.append(symbol_col)
    for c in ["PE_TTM", "ForwardPE", "DividendYield", "ProfitMargin", "MarketCap", "Beta", "Score", "Decision"]:
        if c in scored_df.columns and c not in cols:
            cols.append(c)

    if not cols:
        st.info("No fundamentals columns detected.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    df = scored_df[cols].copy()

    fb1, fb2, fb3, fb4 = st.columns([1.1, 1, 1, 1.2])

    decision_options = sorted(df["Decision"].dropna().unique()) if "Decision" in df.columns else []
    selected_decisions = fb1.multiselect(
        "Decision filter",
        options=decision_options,
        default=decision_options,
        key="fund_decisions_ms",
    )
    if "Decision" in df.columns and selected_decisions:
        df = df[df["Decision"].isin(selected_decisions)]

    if "PE_TTM" in df.columns and not df["PE_TTM"].replace([np.inf, -np.inf], np.nan).dropna().empty:
        max_pe = float(df["PE_TTM"].replace([np.inf, -np.inf], np.nan).dropna().max())
        pe_cap = fb2.slider(
            "Max P/E (TTM)",
            min_value=0.0,
            max_value=float(round(max_pe, 1)) if max_pe > 0 else 50.0,
            value=float(round(max_pe, 1)) if max_pe > 0 else 50.0,
            step=1.0,
            key="fund_pe_cap",
        )
        df = df[(df["PE_TTM"].replace([np.inf, -np.inf], np.nan) <= pe_cap) | df["PE_TTM"].isna()]

    if "MarketCap" in df.columns and not df["MarketCap"].dropna().empty:
        max_mc = float(df["MarketCap"].dropna().max())
        mc_min = fb3.slider(
            "Min market cap ($B)",
            min_value=0.0,
            max_value=float(round(max_mc / 1_000_000_000, 2)),
            value=0.0,
            step=1.0,
            key="fund_mc_min",
        )
        df = df[df["MarketCap"].fillna(0) >= mc_min * 1_000_000_000]

    search_text = fb4.text_input(
        "Search ticker",
        value="",
        key="fund_search_text",
    ).strip().lower()
    if search_text and symbol_col:
        df = df[df[symbol_col].astype(str).str.lower().str.contains(search_text)]

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    sort_col = st.selectbox(
        "Sort by",
        options=numeric_cols if numeric_cols else df.columns,
        index=0,
        key="fundamentals_sort_col",
    )
    sort_ascending = st.checkbox(
        "Sort ascending?",
        value=False,
        key="fundamentals_sort_ascending",
    )

    df_sorted = df.sort_values(by=sort_col, ascending=sort_ascending)

    styled = style_table(
        df_sorted,
        highlight_pl_cols=False,
        highlight_score_col=True,
        highlight_decision_col=True,
    )
    st.dataframe(styled, use_container_width=True, height=520)

    st.markdown(
        '<p class="small-caption">Combine decision, P/E, size, and ticker filters to surface fundamental opportunities.</p>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def render_signals(scored_df: pd.DataFrame):
    st.markdown('<div class="glass-card" style="padding:18px 20px;">', unsafe_allow_html=True)
    st.subheader("Signals & Action List")

    if "Decision" not in scored_df.columns:
        st.info("No Decision column found. Check scoring logic.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    symbol_col = detect_symbol_column(scored_df)

    def subset(decisions):
        return scored_df[scored_df["Decision"].isin(decisions)].copy()

    strong_buy = subset(["Strong Buy"])
    buy = subset(["Buy"])
    trim = subset(["Trim"])
    exit_df = subset(["Exit / Avoid"])

    st.markdown("#### Snapshot")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Strong Buy", len(strong_buy))
    s2.metric("Buy", len(buy))
    s3.metric("Trim", len(trim))
    s4.metric("Exit / Avoid", len(exit_df))

    st.markdown("#### Buy / Add Radar")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Strong Buy**")
        if strong_buy.empty:
            st.caption("None at the moment.")
        else:
            cols = [
                symbol_col,
                "Score",
                "PortfolioWeightPct",
                "UnrealizedPLPct",
                "PE_TTM",
                "ProfitMargin",
            ]
            cols = [c for c in cols if c in strong_buy.columns]
            styled = style_table(strong_buy[cols], highlight_pl_cols=True, highlight_score_col=True)
            st.dataframe(styled, use_container_width=True, height=260)

    with col2:
        st.markdown("**Buy**")
        if buy.empty:
            st.caption("None at the moment.")
        else:
            cols = [
                symbol_col,
                "Score",
                "PortfolioWeightPct",
                "UnrealizedPLPct",
                "PE_TTM",
                "ProfitMargin",
            ]
            cols = [c for c in cols if c in buy.columns]
            styled = style_table(buy[cols], highlight_pl_cols=True, highlight_score_col=True)
            st.dataframe(styled, use_container_width=True, height=260)

    st.markdown("#### De-Risk Radar")
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Trim**")
        if trim.empty:
            st.caption("None at the moment.")
        else:
            cols = [symbol_col, "Score", "PortfolioWeightPct", "UnrealizedPLPct"]
            cols = [c for c in cols if c in trim.columns]
            styled = style_table(trim[cols], highlight_pl_cols=True, highlight_score_col=True)
            st.dataframe(styled, use_container_width=True, height=260)

    with col4:
        st.markdown("**Exit / Avoid**")
        if exit_df.empty:
            st.caption("None at the moment.")
        else:
            cols = [symbol_col, "Score", "PortfolioWeightPct", "UnrealizedPLPct"]
            cols = [c for c in cols if c in exit_df.columns]
            styled = style_table(exit_df[cols], highlight_pl_cols=True, highlight_score_col=True)
            st.dataframe(styled, use_container_width=True, height=260)

    st.markdown("#### Action Playbook")
    bullets = []
    if len(strong_buy) > 0:
        tickers = ", ".join(strong_buy.sort_values("Score", ascending=False)[symbol_col].head(3))
        bullets.append(f"- **Review adds to**: {tickers} (Strong Buy)")
    if len(trim) > 0:
        tickers = ", ".join(trim.sort_values("PortfolioWeightPct", ascending=False)[symbol_col].head(3))
        bullets.append(f"- **Consider trims in**: {tickers}")
    if len(exit_df) > 0:
        tickers = ", ".join(exit_df.sort_values("Score")[symbol_col].head(3))
        bullets.append(f"- **Evaluate exiting**: {tickers}")

    if bullets:
        for b in bullets:
            st.markdown(b)
    else:
        st.caption("No immediate actions surfaced by the model.")

    st.markdown("</div>", unsafe_allow_html=True)


# ----------------- Header with logo -----------------
st.markdown('<div class="glass-card header-card">', unsafe_allow_html=True)
hcol1, hcol2 = st.columns([4, 1])
with hcol1:
    st.markdown(
        """
        <div>
          <h1 style="margin-bottom:4px; color:#F9FAFB;">Oldfield AI Stock Dashboard</h1>
          <p style="margin:0; color:#CBD5F5; font-size:0.9rem;">
            Liquid-glass cockpit for your AI-scored portfolio — positions, fundamentals, and actions in one view.
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
            v1.1 • Experimental
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
st.markdown("</div>", unsafe_allow_html=True)


# ----------------- Upload + main -----------------
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
        render_positions(scored_df, raw_df)
    with tabs[2]:
        render_fundamentals(scored_df)
    with tabs[3]:
        render_signals(scored_df)
