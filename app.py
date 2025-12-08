import streamlit as st
import pandas as pd
from src.scoring import score_portfolio
from src.loaders import load_csv

st.set_page_config(page_title="Oldfield AI Stock Dashboard", layout="wide")

st.title("Oldfield Investing – AI Stock Dashboard (v1)")

st.write(
    """
Upload your portfolio CSV to begin. This version loads the file, shows the raw data,
and calculates basic portfolio metrics (P/L, weights, and a placeholder score).
"""
)

uploaded_file = st.file_uploader("Upload your portfolio CSV", type=["csv"])

if uploaded_file is not None:
    df = load_csv(uploaded_file)

    # Run scoring / metrics (this creates CurrentValue_clean, CostBasis_clean, etc.)
    scored_df = score_portfolio(df)

    # ----- High-level portfolio metrics -----
    st.subheader("📈 Portfolio Summary")

    if "CurrentValue_clean" in scored_df.columns and "UnrealizedPL" in scored_df.columns:
        current_val = scored_df["CurrentValue_clean"]
        total_value = float(current_val.sum()) if len(current_val) else 0.0
        total_pl = float(scored_df["UnrealizedPL"].sum())
        num_positions = len(scored_df)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Portfolio Value", f"${total_value:,.0f}")
        col2.metric("Total Unrealized P/L", f"${total_pl:,.0f}")
        col3.metric("Number of Positions", num_positions)
    else:
        st.info(
            "Could not detect cleaned value columns (CurrentValue_clean / UnrealizedPL). "
            "Check scoring.py or your CSV headers."
        )

    # ----- Raw data -----
    st.subheader("📌 Raw Portfolio Data")
    st.dataframe(df, use_container_width=True)

    # ----- Scored / metrics view -----
    st.subheader("📊 Portfolio Metrics & Score (v1 placeholder)")

    # Choose a column to sort by – prefer numeric ones
    numeric_cols = [
        c for c in scored_df.columns
        if pd.api.types.is_numeric_dtype(scored_df[c])
    ]
    if not numeric_cols:
        numeric_cols = list(scored_df.columns)

    sort_col = st.selectbox(
        "Sort by column",
        options=numeric_cols,
        index=numeric_cols.index("Score") if "Score" in numeric_cols else 0,
    )

    sort_ascending = st.checkbox("Sort ascending?", value=False)

    scored_sorted = scored_df.sort_values(by=sort_col, ascending=sort_ascending)
    st.dataframe(scored_sorted, use_container_width=True)

else:
    st.info("Upload a CSV to get started.")
