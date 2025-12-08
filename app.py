import streamlit as st
import pandas as pd

from src.scoring import score_portfolio
from src.loaders import load_csv

# ----------------- Page config -----------------
st.set_page_config(
    page_title="Oldfield AI Stock Dashboard",
    layout="wide",
)

# ----------------- Custom CSS (Figma-style dark UI) -----------------
st.markdown(
    """
<style>
.stApp {
    background-color: #0d1117;
    color: #e6edf3;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

h1, h2, h3 {
    letter-spacing: -0.5px;
    font-weight: 600;
}

/* KPI metric cards */
div[data-testid="metric-container"] {
    background-color: #161b22;
    border: 1px solid #30363d;
    padding: 18px 16px;
    border-radius: 12px;
}

section[data-testid="stFileUploader"] {
    background-color: #161b22;
    border-radius: 12px;
    padding: 12px 16px;
    border: 1px solid #30363d;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

.dataframe table,
.stDataFrame table {
    background-color: #0d1117 !important;
    color: #e6edf3 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ----------------- Title & intro -----------------
st.title("Oldfield Investing – AI Stock Dashboard (v1)")

st.write(
    """
Upload your portfolio CSV to see position-level metrics, portfolio P/L, and a
first-pass fundamentals-based score and action label for each holding.
"""
)

# ----------------- File upload -----------------
uploaded_file = st.file_uploader("Upload your portfolio CSV", type=["csv"])

if uploaded_file is not None:
    # Load raw data
    df = load_csv(uploaded_file)

    # Run scoring / metrics (adds cleaned value columns + fundamentals + Score + Decision)
    scored_df = score_portfolio(df)

    # ----------------- Portfolio Summary KPIs -----------------
    st.subheader("📈 Portfolio Summary")

    if "CurrentValue_clean" in scored_df.columns and "UnrealizedPL" in scored_df.columns:
        current_val = scored_df["CurrentValue_clean"]
        total_value = float(current_val.sum()) if len(current_val) else 0.0
        total_pl = float(scored_df["UnrealizedPL"].sum())
        num_positions = len(scored_df)

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Portfolio Value", f"${total_value:,.0f}")

        pl_label = f"${total_pl:,.0f}"
        col2.metric("Total Unrealized P/L", pl_label)

        col3.metric("Number of Positions", int(num_positions))
    else:
        st.info(
            "Could not detect cleaned value columns (CurrentValue_clean / UnrealizedPL). "
            "Check scoring.py or your CSV headers."
        )

    # ----------------- Decision breakdown -----------------
    if "Decision" in scored_df.columns:
        st.subheader("🧭 Action Summary (Based on Score)")

        decision_counts = scored_df["Decision"].value_counts().to_dict()

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Strong Buy", decision_counts.get("Strong Buy", 0))
        c2.metric("Buy", decision_counts.get("Buy", 0))
        c3.metric("Hold", decision_counts.get("Hold", 0))
        c4.metric("Trim", decision_counts.get("Trim", 0))
        c5.metric("Exit / Avoid", decision_counts.get("Exit / Avoid", 0))

    # ----------------- Raw Data -----------------
    st.subheader("📌 Raw Portfolio Data")
    st.dataframe(df, use_container_width=True)

    # ----------------- Portfolio Metrics & Score -----------------
    st.subheader("📊 Portfolio Metrics, Score & Action")

    # Decision filter
    decision_options = ["All"]
    if "Decision" in scored_df.columns:
        decision_options += list(sorted(scored_df["Decision"].dropna().unique()))
    selected_decision = st.selectbox("Filter by decision", decision_options)

    filtered_df = scored_df.copy()
    if selected_decision != "All" and "Decision" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Decision"] == selected_decision]

    # Helper: choose numeric columns for sorting
    numeric_cols = [
        c
        for c in filtered_df.columns
        if pd.api.types.is_numeric_dtype(filtered_df[c])
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

    # ----------------- Styling functions -----------------
    def highlight_pl(val):
        """Color gains green and losses red."""
        if pd.isna(val):
            return ""
        try:
            v = float(val)
        except Exception:
            return ""
        color = "#00c853" if v > 0 else "#d50000"
        return f"color: {color};"

    def highlight_score(val):
        """Background gradient for Score 0–100."""
        if pd.isna(val):
            return ""
        try:
            v = float(val)
        except Exception:
            return ""
        if v < 40:
            color = "#ff5252"   # red
        elif v < 70:
            color = "#ffd600"   # yellow
        else:
            color = "#00e676"   # green
        return f"background-color: {color}; color: #000000;"

    # Build Styler with conditional formatting
    pl_cols = [c for c in ["UnrealizedPL", "UnrealizedPLPct"] if c in scored_sorted.columns]

    styled = scored_sorted.style

    if pl_cols:
        styled = styled.applymap(highlight_pl, subset=pl_cols)

    if "Score" in scored_sorted.columns:
        styled = styled.applymap(highlight_score, subset=["Score"])

    st.dataframe(styled, use_container_width=True)

else:
    st.info("Upload a CSV to get started.")
