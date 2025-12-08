import streamlit as st
import pandas as pd
import plotly.express as px

from src.scoring import score_portfolio
from src.loaders import load_csv

# ----------------- Page config -----------------
st.set_page_config(
    page_title="Oldfield AI Stock Dashboard",
    layout="wide",
)

# ----------------- Custom CSS – Findeshy-style, grey + blue -----------------
st.markdown(
    """
<style>

/* Global background & font */
html, body, [class*="css"]  {
    background-color: #F2F3F7 !important;
    color: #111827 !important;
    font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

/* Main page container */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* Top header bar */
.header-card {
    background: linear-gradient(135deg, #FFFFFF 0%, #F7FAFF 100%);
    border-radius: 18px;
    padding: 20px 24px;
    border: 1px solid #E0E7F1;
    box-shadow: 0 10px 30px rgba(15, 23, 42, 0.04);
    margin-bottom: 16px;
}

/* KPI metric cards */
div[data-testid="metric-container"] {
    background: #FFFFFF !important;
    border: 1px solid #E0E7F1 !important;
    padding: 18px 18px !important;
    border-radius: 14px !important;
    box-shadow: 0px 4px 16px rgba(15,23,42,0.04);
}

/* File uploader as card */
section[data-testid="stFileUploader"] {
    background-color: #FFFFFF !important;
    border: 1px dashed #CBD5E1 !important;
    border-radius: 12px !important;
    padding: 14px 14px !important;
}

/* Headings */
h1, h2, h3, h4 {
    color: #0F172A !important;
    font-weight: 600 !important;
    letter-spacing: -0.4px;
}

/* Tabs styling */
div[data-baseweb="tab-list"] {
    gap: 6px;
}

button[role="tab"] {
    border-radius: 999px !important;
    padding: 6px 14px !important;
    font-size: 0.9rem !important;
    color: #4B5563 !important;
    border: 1px solid transparent !important;
    background-color: transparent !important;
}

button[role="tab"][aria-selected="true"] {
    background-color: #2563EB1A !important;
    color: #1D4ED8 !important;
    border-color: #BFDBFE !important;
}

/* Dataframes */
.dataframe table, .stDataFrame table {
    color: #111827 !important;
    background-color: #FFFFFF !important;
    border-radius: 12px !important;
}

thead tr th {
    background-color: #EEF2FF !important;
    color: #111827 !important;
    font-weight: 600 !important;
}

tbody tr:hover td {
    background-color: #F3F4FF !important;
}

/* Small captions */
.small-caption {
    font-size: 0.8rem;
    color: #6B7280;
}

/* Helper classes */
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

# ----------------- Helpers -----------------
def highlight_pl(val):
    """Color gains green and losses red."""
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
    """Background color for Score 0–100."""
    if pd.isna(val):
        return ""
    try:
        v = float(val)
    except Exception:
        return ""
    if v >= 70:
        return "background-color: #DCFCE7; color:#14532D;"
    elif v >= 50:
        return "background-color: #FEF9C3; color:#78350F;"
    else:
        return "background-color: #FEE2E2; color:#7F1D1D;"


def detect_symbol_column(df: pd.DataFrame):
    for col in df.columns:
        cl = col.lower()
        if "symbol" in cl or "ticker" in cl:
            return col
    return None


# ----------------- Page renderers -----------------
def render_overview(scored_df: pd.DataFrame):
    st.subheader("Portfolio Overview")

    # KPIs row
    if "CurrentValue_clean" in scored_df.columns and "UnrealizedPL" in scored_df.columns:
        current_val = scored_df["CurrentValue_clean"]
        total_value = float(current_val.sum()) if len(current_val) else 0.0
        total_pl = float(scored_df["UnrealizedPL"].sum())
        num_positions = len(scored_df)

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Portfolio Value", f"${total_value:,.0f}")
        c2.metric("Total Unrealized P/L", f"${total_pl:,.0f}")
        c3.metric("Positions", int(num_positions))
    else:
        st.info("Could not compute summary metrics. Check your CSV headers.")

    # Action summary
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

    # Charts row
    st.markdown("#### Key Charts")
    col_a, col_b = st.columns(2)

    # Score distribution
    if "Score" in scored_df.columns:
        score_series = scored_df["Score"].dropna()
        if not score_series.empty:
            fig_score = px.histogram(
                score_series,
                nbins=20,
                title="Score Distribution",
            )
            fig_score.update_layout(
                margin=dict(l=10, r=10, t=40, b=10),
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
            )
            col_a.plotly_chart(fig_score, use_container_width=True)

    # Allocation by Decision
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
                paper_bgcolor="#FFFFFF",
            )
            col_b.plotly_chart(fig_alloc, use_container_width=True)

    # P/L by ticker
    st.markdown("#### Unrealized P/L by Position")
    symbol_col = detect_symbol_column(scored_df)
    if symbol_col and "UnrealizedPL" in scored_df.columns:
        pl_df = scored_df[[symbol_col, "UnrealizedPL"]].copy().dropna(subset=["UnrealizedPL"])
        if not pl_df.empty:
            fig_pl = px.bar(
                pl_df,
                x=symbol_col,
                y="UnrealizedPL",
                title="Unrealized P/L by Ticker",
            )
            fig_pl.update_layout(
                margin=dict(l=10, r=10, t=40, b=10),
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                xaxis_title="Ticker",
                yaxis_title="Unrealized P/L",
            )
            st.plotly_chart(fig_pl, use_container_width=True)


def render_positions(scored_df: pd.DataFrame, raw_df: pd.DataFrame):
    st.subheader("Positions & Metrics")

    # Decision filter
    decision_options = ["All"]
    if "Decision" in scored_df.columns:
        decision_options += list(sorted(scored_df["Decision"].dropna().unique()))
    selected_decision = st.selectbox("Filter by decision", decision_options)

    filtered_df = scored_df.copy()
    if selected_decision != "All" and "Decision" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Decision"] == selected_decision]

    # Numeric columns for sorting
    numeric_cols = [
        c for c in filtered_df.columns if pd.api.types.is_numeric_dtype(filtered_df[c])
    ]
    if not numeric_cols:
        numeric_cols = list(filtered_df.columns)

    default_sort_index = 0
    if "Score" in numeric_cols:
        default_sort_index = numeric_cols.index("Score")

    sort_col = st.selectbox(
        "Sort by column",
        options=numeric_cols,
        index=default_sort_index,
    )

    sort_ascending = st.checkbox("Sort ascending?", value=False)

    scored_sorted = filtered_df.sort_values(by=sort_col, ascending=sort_ascending)

    # Styled table
    pl_cols = [c for c in ["UnrealizedPL", "UnrealizedPLPct"] if c in scored_sorted.columns]

    styled = scored_sorted.style
    if pl_cols:
        styled = styled.applymap(highlight_pl, subset=pl_cols)
    if "Score" in scored_sorted.columns:
        styled = styled.applymap(highlight_score, subset=["Score"])

    st.dataframe(styled, use_container_width=True)
    st.markdown('<p class="small-caption">Full positions view with metrics. Use filters and sorting to focus.</p>', unsafe_allow_html=True)


def render_fundamentals(scored_df: pd.DataFrame):
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
        return

    fundamentals_view = scored_df[cols].copy()

    numeric_cols = [
        c for c in fundamentals_view.columns
        if pd.api.types.is_numeric_dtype(fundamentals_view[c])
    ]
    sort_col = st.selectbox(
        "Sort by",
        options=numeric_cols if numeric_cols else fundamentals_view.columns,
        index=0,
    )
    sort_ascending = st.checkbox("Sort ascending?", value=False)

    fundamentals_view = fundamentals_view.sort_values(by=sort_col, ascending=sort_ascending)

    styled = fundamentals_view.style
    if "Score" in fundamentals_view.columns:
        styled = styled.applymap(highlight_score, subset=["Score"])

    st.dataframe(styled, use_container_width=True)
    st.markdown('<p class="small-caption">Quick fundamentals view for screening & comparison.</p>', unsafe_allow_html=True)


def render_signals(scored_df: pd.DataFrame):
    st.subheader("Signals & Action List")

    if "Decision" not in scored_df.columns:
        st.info("No Decision column found. Check scoring logic.")
        return

    symbol_col = detect_symbol_column(scored_df)

    def subset(decisions):
        df_sub = scored_df[scored_df["Decision"].isin(decisions)].copy()
        return df_sub

    strong_buy = subset(["Strong Buy"])
    buy = subset(["Buy"])
    trim = subset(["Trim"])
    exit_df = subset(["Exit / Avoid"])

    st.markdown("#### Buy / Add Radar")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Strong Buy**")
        if strong_buy.empty:
            st.caption("None at the moment.")
        else:
            cols = [symbol_col, "Score", "PortfolioWeightPct", "UnrealizedPLPct", "PE_TTM", "ProfitMargin"]
            cols = [c for c in cols if c in strong_buy.columns]
            st.dataframe(strong_buy[cols], use_container_width=True)

    with col2:
        st.markdown("**Buy**")
        if buy.empty:
            st.caption("None at the moment.")
        else:
            cols = [symbol_col, "Score", "PortfolioWeightPct", "UnrealizedPLPct", "PE_TTM", "ProfitMargin"]
            cols = [c for c in cols if c in buy.columns]
            st.dataframe(buy[cols], use_container_width=True)

    st.markdown("#### De-Risk Radar")
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Trim**")
        if trim.empty:
            st.caption("None at the moment.")
        else:
            cols = [symbol_col, "Score", "PortfolioWeightPct", "UnrealizedPLPct"]
            cols = [c for c in cols if c in trim.columns]
            st.dataframe(trim[cols], use_container_width=True)

    with col4:
        st.markdown("**Exit / Avoid**")
        if exit_df.empty:
            st.caption("None at the moment.")
        else:
            cols = [symbol_col, "Score", "PortfolioWeightPct", "UnrealizedPLPct"]
            cols = [c for c in cols if c in exit_df.columns]
            st.dataframe(exit_df[cols], use_container_width=True)


# ----------------- Top header + upload -----------------
st.markdown(
    """
<div class="header-card">
  <div style="display:flex; justify-content:space-between; align-items:center; gap:16px; flex-wrap:wrap;">
    <div>
      <h1 style="margin-bottom:4px;">Oldfield AI Stock Dashboard</h1>
      <p style="margin:0; color:#6B7280; font-size:0.9rem;">
        Clean, organized view of your positions, fundamentals, and signals — styled like a modern fintech console.
      </p>
    </div>
    <div style="min-width:260px;">
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

upload_col, _ = st.columns([2, 3])
with upload_col:
    uploaded_file = st.file_uploader("Upload portfolio CSV", type=["csv"])

# ----------------- Main content -----------------
if uploaded_file is None:
    st.info("Upload a CSV to unlock the dashboard views.")
else:
    raw_df = load_csv(uploaded_file)
    scored_df = score_portfolio(raw_df)

    tabs = st.tabs(["Overview", "Positions", "Fundamentals", "Signals"])

    with tabs[0]:
        render_overview(scored_df)

    with tabs[1]:
        render_positions(scored_df, raw_df)

    with tabs[2]:
        render_fundamentals(scored_df)

    with tabs[3]:
        render_signals(scored_df)
