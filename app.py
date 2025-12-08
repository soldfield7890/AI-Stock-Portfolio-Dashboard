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

# ----------------- Custom CSS (grey-blue DataAI-style) -----------------
st.markdown(
    """
<style>

/* Global background & font */
html, body, [class*="css"]  {
    background-color: #F3F6FA !important;
    color: #1C2733 !important;
    font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

/* Center content container */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

/* Headings */
h1, h2, h3, h4 {
    color: #1C2733 !important;
    font-weight: 600 !important;
    letter-spacing: -0.5px;
}

/* KPI metric cards */
div[data-testid="metric-container"] {
    background: #FFFFFF !important;
    border: 1px solid #DCE3ED !important;
    padding: 24px 20px !important;
    border-radius: 16px !important;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.04);
}

/* File upload styling */
section[data-testid="stFileUploader"] {
    background-color: #FFFFFF !important;
    border: 1px solid #DCE3ED !important;
    border-radius: 14px !important;
    padding: 18px !important;
}

/* Dataframe base styles */
.dataframe table, .stDataFrame table {
    color: #1C2733 !important;
    background-color: #FFFFFF !important;
    border-radius: 12px !important;
}

thead tr th {
    background-color: #EAF0F7 !important;
    color: #1C2733 !important;
}

/* Hover row */
tbody tr:hover td {
    background-color: #F0F4FA !important;
}

</style>
""",
    unsafe_allow_html=True,
)

# ----------------- Helper styling functions -----------------
def highlight_pl(val):
    """Color gains green and losses red."""
    if pd.isna(val):
        return ""
    try:
        v = float(val)
    except Exception:
        return ""
    if v > 0:
        return "color: #2ECC71; font-weight:600;"
    else:
        return "color: #E74C3C; font-weight:600;"


def highlight_score(val):
    """Background color for Score 0–100."""
    if pd.isna(val):
        return ""
    try:
        v = float(val)
    except Exception:
        return ""
    if v >= 70:
        return "background-color: #D9F7E8; color:#1C2733;"
    elif v >= 50:
        return "background-color: #FFF4CC; color:#1C2733;"
    else:
        return "background-color: #FFE0E0; color:#1C2733;"


def detect_symbol_column(df: pd.DataFrame):
    for col in df.columns:
        cl = col.lower()
        if "symbol" in cl or "ticker" in cl:
            return col
    return None


# ----------------- Page renderers -----------------
def render_dashboard(scored_df: pd.DataFrame):
    st.header("📊 Portfolio Overview")

    # ----- KPIs -----
    if "CurrentValue_clean" in scored_df.columns and "UnrealizedPL" in scored_df.columns:
        current_val = scored_df["CurrentValue_clean"]
        total_value = float(current_val.sum()) if len(current_val) else 0.0
        total_pl = float(scored_df["UnrealizedPL"].sum())
        num_positions = len(scored_df)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Portfolio Value", f"${total_value:,.0f}")
        col2.metric("Total Unrealized P/L", f"${total_pl:,.0f}")
        col3.metric("Number of Positions", int(num_positions))
    else:
        st.info("Could not compute summary metrics. Check your CSV headers.")

    # ----- Decision breakdown -----
    if "Decision" in scored_df.columns:
        st.subheader("🧭 Action Summary")
        decision_counts = scored_df["Decision"].value_counts().to_dict()
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Strong Buy", decision_counts.get("Strong Buy", 0))
        c2.metric("Buy", decision_counts.get("Buy", 0))
        c3.metric("Hold", decision_counts.get("Hold", 0))
        c4.metric("Trim", decision_counts.get("Trim", 0))
        c5.metric("Exit / Avoid", decision_counts.get("Exit / Avoid", 0))

    # ----- Charts row -----
    st.subheader("📈 Charts")

    col_a, col_b = st.columns(2)

    # Chart 1: Score distribution
    if "Score" in scored_df.columns:
        score_series = scored_df["Score"].dropna()
        if not score_series.empty:
            fig_score = px.histogram(
                score_series,
                nbins=20,
                title="Score Distribution",
            )
            fig_score.update_layout(margin=dict(l=10, r=10, t=40, b=10))
            col_a.plotly_chart(fig_score, use_container_width=True)

    # Chart 2: Allocation by Decision
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
            fig_alloc.update_layout(margin=dict(l=10, r=10, t=40, b=10))
            col_b.plotly_chart(fig_alloc, use_container_width=True)

    # ----- P/L bar chart -----
    st.subheader("💹 Unrealized P/L by Position")

    symbol_col = detect_symbol_column(scored_df)
    if symbol_col and "UnrealizedPL" in scored_df.columns:
        pl_df = scored_df[[symbol_col, "UnrealizedPL"]].copy()
        pl_df = pl_df.dropna(subset=["UnrealizedPL"])
        if not pl_df.empty:
            fig_pl = px.bar(
                pl_df,
                x=symbol_col,
                y="UnrealizedPL",
                title="Unrealized P/L by Ticker",
            )
            fig_pl.update_layout(margin=dict(l=10, r=10, t=40, b=10), xaxis_title="Ticker")
            st.plotly_chart(fig_pl, use_container_width=True)


def render_positions(scored_df: pd.DataFrame, raw_df: pd.DataFrame):
    st.header("📋 Positions & Metrics")

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

    st.subheader("Raw Positions with Metrics")
    st.dataframe(styled, use_container_width=True)

    st.markdown("### Underlying CSV (raw) – for debugging")
    st.dataframe(raw_df, use_container_width=True)


def render_fundamentals(scored_df: pd.DataFrame):
    st.header("📚 Fundamentals Snapshot")

    symbol_col = detect_symbol_column(scored_df)

    cols = []
    if symbol_col:
        cols.append(symbol_col)
    for c in ["PE_TTM", "ForwardPE", "DividendYield", "ProfitMargin", "MarketCap", "Beta", "Score", "Decision"]:
        if c in scored_df.columns and c not in cols:
            cols.append(c)

    if not cols:
        st.info("No fundamentals columns detected. Check scoring/fundamentals.")
        return

    fundamentals_view = scored_df[cols].copy()

    # Sort options
    numeric_cols = [
        c for c in fundamentals_view.columns
        if pd.api.types.is_numeric_dtype(fundamentals_view[c])
    ]
    sort_col = st.selectbox(
        "Sort fundamentals by",
        options=numeric_cols if numeric_cols else fundamentals_view.columns,
        index=0,
    )
    sort_ascending = st.checkbox("Sort ascending?", value=False)

    fundamentals_view = fundamentals_view.sort_values(by=sort_col, ascending=sort_ascending)

    styled = fundamentals_view.style
    if "Score" in fundamentals_view.columns:
        styled = styled.applymap(highlight_score, subset=["Score"])

    st.dataframe(styled, use_container_width=True)


def render_signals(scored_df: pd.DataFrame):
    st.header("🧠 Signals & Shopping List")

    if "Decision" not in scored_df.columns:
        st.info("No Decision column found. Check scoring logic.")
        return

    symbol_col = detect_symbol_column(scored_df)

    def subset(decisions):
        df_sub = scored_df[scored_df["Decision"].isin(decisions)].copy()
        if symbol_col and symbol_col not in df_sub.columns:
            return df_sub
        return df_sub

    strong_buy = subset(["Strong Buy"])
    buy = subset(["Buy"])
    trim = subset(["Trim"])
    exit_df = subset(["Exit / Avoid"])

    st.subheader("✅ Buy / Add Radar")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Strong Buy Candidates**")
        if strong_buy.empty:
            st.caption("None right now.")
        else:
            st.dataframe(
                strong_buy[[c for c in strong_buy.columns if c in [symbol_col, "Score", "PortfolioWeightPct", "UnrealizedPLPct", "PE_TTM", "ProfitMargin"]]],
                use_container_width=True,
            )

    with col2:
        st.markdown("**Buy Candidates**")
        if buy.empty:
            st.caption("None right now.")
        else:
            st.dataframe(
                buy[[c for c in buy.columns if c in [symbol_col, "Score", "PortfolioWeightPct", "UnrealizedPLPct", "PE_TTM", "ProfitMargin"]]],
                use_container_width=True,
            )

    st.subheader("⚠️ Risk / De-risk Radar")
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Trim Candidates**")
        if trim.empty:
            st.caption("None right now.")
        else:
            st.dataframe(
                trim[[c for c in trim.columns if c in [symbol_col, "Score", "PortfolioWeightPct", "UnrealizedPLPct", "PE_TTM", "ProfitMargin"]]],
                use_container_width=True,
            )

    with col4:
        st.markdown("**Exit / Avoid**")
        if exit_df.empty:
            st.caption("None right now.")
        else:
            st.dataframe(
                exit_df[[c for c in exit_df.columns if c in [symbol_col, "Score", "PortfolioWeightPct", "UnrealizedPLPct", "PE_TTM", "ProfitMargin"]]],
                use_container_width=True,
            )


# ----------------- Sidebar layout (navigation + upload) -----------------
with st.sidebar:
    st.markdown("### Oldfield AI Cockpit")
    st.caption("Upload a CSV and choose a view.")

    uploaded_file = st.file_uploader("Upload portfolio CSV", type=["csv"], key="sidebar_uploader")

    page = st.radio(
        "Navigation",
        ["Dashboard", "Positions", "Fundamentals", "Signals"],
    )

# ----------------- Main routing -----------------
if uploaded_file is None:
    st.info("Upload a CSV from the sidebar to get started.")
else:
    raw_df = load_csv(uploaded_file)
    scored_df = score_portfolio(raw_df)

    if page == "Dashboard":
        render_dashboard(scored_df)
    elif page == "Positions":
        render_positions(scored_df, raw_df)
    elif page == "Fundamentals":
        render_fundamentals(scored_df)
    elif page == "Signals":
        render_signals(scored_df)
