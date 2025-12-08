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

    # Run scoring / metrics
    scored_df = score_portfolio(df)

    # ----- High-level portfolio metrics -----
    st.subheader("📈 Portfolio Summary")

    total_value = None
    total_pl = None

    # Try to infer current value column
    col_current_value = None
    for col in scored_df.columns:
        if "current value" in col.lower() or "market value" in col.lower():
            col_current_value = col
            break

    if col_current_value is not None and "UnrealizedPL" in scored_df.columns:
        current_val = scored_df[col_current_value].astype(float)
        total_value = float(current_val.sum())
        total_pl = float(scored_df["UnrealizedPL"].astype(float).sum())
        num_positions = len(scored_df)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Portfolio Value", f"${total_value:,.0f}")
        col2.metric("Total Unrealized P/L", f"${total_pl:,.0f}")
        col3.metric("Number of Positions", num_positions)
    else:
        st.info("Could not detect value columns to compute summary metrics. Check your CSV headers.")

    # ----- Raw data -----
    st.subheader("📌 Raw Portfolio Data")
    st.dataframe(df, use_container_width=True)

    # ----- Scored / metrics view -----
    st.subheader("📊 Portfolio Metrics & Score (v1 placeholder)")

    # Let you sort by Score, P/L %, etc.
    sort_col = st.selectbox(
        "Sort by column",
        options=[c for c in scored_df.columns if scored_df[c].dtype != "object"],
        index=len(scored_df.columns) - 1,  # default to last, usually Score
    )

    sort_ascending = st.checkbox("Sort ascending?", value=False)

    scored_sorted = scored_df.sort_values(by=sort_col, ascending=sort_ascending)
    st.dataframe(scored_sorted, use_container_width=True)

else:
    st.info("Upload a CSV to get started.")
