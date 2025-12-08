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

# ----------------- Custom CSS – FinDeshY-style, light grey + blue -----------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* Global background & font */
html, body, [class*="css"]  {
    background-color: #F3F4F6 !important;  /* soft grey */
    color: #111827 !important;             /* slate-900 */
    font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

/* Main container width & padding */
.block-container {
    padding-top: 1.75rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

/* Headings */
h1, h2, h3, h4 {
    color: #111827 !important;
    font-weight: 600 !important;
    letter-spacing: -0.4px;
}

/* Subtle section label */
.section-label {
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #6B7280;
}

/* KPI cards */
div[data-testid="metric-container"] {
    background: #FFFFFF !important;
    border: 1px solid #E5E7EB !important;
    padding: 18px 18px !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);
}

/* File uploader */
section[data-testid="stFileUploader"] {
    background-color: #FFFFFF !important;
    border-radius: 16px !important;
    padding: 16px !important;
    border: 1px solid #E5E7EB !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
}
.stTabs [data-baseweb="tab"] {
    background-color: #E5E7EB;
    color: #374151;
    border-radius: 999px;
    padding: 8px 18px;
    font-weight: 500;
    font-size: 0.9rem;
}
.stTabs [aria-selected="true"] {
    background-color: #2563EB !important;
    color: #F9FAFB !important;
}

/* Dataframe base styles */
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

/* Hover effect on rows */
tbody tr:hover td {
    background-color: #F3F4FF !important;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #111827 !important;  /* dark navy sidebar */
    color: #E5E7EB !important;
}
[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3, 
[data-testid="stSidebar"] h4, 
[data-testid="stSidebar"] p {
    color: #E5E7EB !important;
}

/* Sidebar headers */
.sidebar-title {
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 0.25rem;
    color: #F9FAFB;
}
.sidebar-subtitle {
    font-size: 0.85rem;
    color: #9CA3AF;
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
        return "color: #16A34A; font-weight:600;"  # green-600
    else:
        return "color: #DC2626; font-weight:600;"  # red-600


def highlight_score(val):
    """Background color for Score 0–100."""
    if pd.isna(val):
        return ""
    try:
        v = float(val)
    except Exception:
        return ""
    if v >= 70:
        return "background-color: #DCFCE7; color:#111827;"   # green-100
    elif v >= 50:
        return "background-color: #FEF9C3; color:#111827;"   # yellow-100
    else:
        return "background-color: #FEE2E2; color:#111827;"   # red-100


def detect_symbol_column(df: pd.DataFrame):
    for col in df.columns:
        cl = col.lower()
        if "symbol" in cl or "ticker" in cl:
            return col
    return None


def style_fig(fig):
    """Apply a consistent light theme to Plotly charts."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#F9FAFB",
        font=dict(family="Inter", size=12, color="#111827"),
        xaxis=dict(gridcolor="#E5E7EB"),
        yaxis=dict(gridcolor="#E5E7EB"),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


# ----------------- Page sections -----------------
def render_dashboard(scored_df: pd.DataFrame):
    st.markdown('<div class="section-label">Overview</div>', unsafe_allow_html=True)
    st.header("Portfolio Dashboard")

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
        st.markdown('<div class="section-label" style="margin-top:1.5rem;">Allocation & Decisions</div>', unsafe_allow_html=True)
        st.subheader("Action Summary")

        decision_counts = scored_df["Decision"].value_counts().to_dict()
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Strong Buy", decision_counts.get("Strong Buy", 0))
        c2.metric("Buy", decision_counts.get("Buy", 0))
        c3.metric("Hold", decision_counts.get("Hold", 0))
        c4.metric("Trim", decision_counts.get("Trim", 0))
        c5.metric("Exit / Avoid", decision_counts.get("Exit / Avoid", 0))

    # ----- Charts row -----
    st.markdown('<div class="section-label" style="margin-top:1.75rem;">Key Charts</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    # Chart 1: Score distribution
    if "Score" in scored_df.columns:
        score_series = scored_df["Score"].dropna()
        if not score_series.empty:
            fig_score = px.histogram(
                score_series,
                nbins=15,
                title="Score Distribution",
                color_discrete_sequence=["#2563EB"],  # blue
            )
            style_fig(fig_score)
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
                color="Decision",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            style_fig(fig_alloc)
            col_b.plotly_chart(fig_alloc, use_container_width=True)

    # ----- P/L bar chart -----
    st.markdown('<div class="section-label" style="margin-top:1.75rem;">Performance</div>', unsafe_allow_html=True)
    st.subheader("Unrealized P/L by Position")

    symbol_col = detect_symbol_column(scored_df)
    if symbol_col and "UnrealizedPL" in scored_df.columns:
        pl_df = scored_df[[symbol_col, "UnrealizedPL"]].copy().dropna(subset=["UnrealizedPL"])
        if not pl_df.empty:
            fig_pl = px.bar(
                pl_df,
                x=symbol_col,
                y="UnrealizedPL",
                title="Unrealized P/L by Ticker",
                color="UnrealizedPL",
                color_continuous_scale=["#DC2626", "#F97316", "#16A34A"],  # red → orange → green
            )
            style_fig(fig_pl)
            st.plotly_chart(fig_pl, use_container_width=True)


def render_positions(scored_df: pd.DataFrame, raw_df: pd.DataFrame):
    st.markdown('<div class="section-label">Holdings</div>', unsafe_allow_html=True)
    st.header("Positions & Metrics")

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

    st.subheader("Positions Table")
    st.dataframe(styled, use_container_width=True)

    with st.expander("View raw CSV data (debugging / verification)"):
        st.dataframe(raw_df, use_container_width=True)


def render_fundamentals(scored_df: pd.DataFrame):
    st.markdown('<div class="section-label">Valuation</div>', unsafe_allow_html=True)
    st.header("Fundamentals Snapshot")

    symbol_col = detect_symbol_column(scored_df)

    cols = []
    if symbol_col:
        cols.append(symbol_col)
    for c in [
        "PE_TTM",
        "ForwardPE",
        "DividendYield",
        "ProfitMargin",
        "MarketCap",
        "Beta",
        "Score",
        "Decision",
    ]:
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
    st.markdown('<div class="section-label">Signals</div>', unsafe_allow_html=True)
    st.header("Buy / Trim / Exit Radar")

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

    st.subheader("Upside Radar")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Strong Buy Candidates**")
        if strong_buy.empty:
            st.caption("None currently flagged.")
        else:
            cols = [symbol_col, "Score", "PortfolioWeightPct", "UnrealizedPLPct", "PE_TTM", "ProfitMargin"]
            cols = [c for c in cols if c in strong_buy.columns]
            st.dataframe(strong_buy[cols], use_container_width=True)

    with col2:
        st.markdown("**Buy Candidates**")
        if buy.empty:
            st.caption("None currently flagged.")
        else:
            cols = [symbol_col, "Score", "PortfolioWeightPct", "UnrealizedPLPct", "PE_TTM", "ProfitMargin"]
            cols = [c for c in cols if c in buy.columns]
            st.dataframe(buy[cols], use_container_width=True)

    st.subheader("Risk / De-Risk Radar")
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Trim Candidates**")
        if trim.empty:
            st.caption("None currently flagged.")
        else:
            cols = [symbol_col, "Score", "PortfolioWeightPct", "UnrealizedPLPct", "PE_TTM", "ProfitMargin"]
            cols = [c for c in cols if c in trim.columns]
            st.dataframe(trim[cols], use_container_width=True)

    with col4:
        st.markdown("**Exit / Avoid**")
        if exit_df.empty:
            st.caption("None currently flagged.")
        else:
            cols = [symbol_col, "Score", "PortfolioWeightPct", "UnrealizedPLPct", "PE_TTM", "ProfitMargin"]
            cols = [c for c in cols if c in exit_df.columns]
            st.dataframe(exit_df[cols], use_container_width=True)


# ----------------- Sidebar: brand + upload -----------------
with st.sidebar:
    st.markdown('<div class="sidebar-title">Oldfield AI Cockpit</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-subtitle">Upload a portfolio CSV to view dashboards and signals.</div>',
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader("Portfolio CSV", type=["csv"], key="sidebar_uploader")

# ----------------- Main layout -----------------
st.title("Oldfield AI Stock Dashboard")

st.write(
    "A clean, Figma-inspired financial cockpit for scoring, visualizing, and managing your positions."
)

if uploaded_file is None:
    st.info("Upload a CSV from the left sidebar to get started.")
else:
    raw_df = load_csv(uploaded_file)
    scored_df = score_portfolio(raw_df)

    tabs = st.tabs(["Dashboard", "Positions", "Fundamentals", "Signals"])

    with tabs[0]:
        render_dashboard(scored_df)
    with tabs[1]:
        render_positions(scored_df, raw_df)
    with tabs[2]:
        render_fundamentals(scored_df)
    with tabs[3]:
        render_signals(scored_df)
