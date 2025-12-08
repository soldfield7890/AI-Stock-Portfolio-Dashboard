import streamlit as st
import pandas as pd
from src.scoring import score_portfolio
from src.loaders import load_csv

st.set_page_config(page_title="Oldfield AI Stock Dashboard", layout="wide")

st.title("Oldfield Investing – AI Stock Dashboard (v1)")

st.write("""Upload your portfolio CSV to begin. This starter version simply loads the file,
shows the raw data, and runs placeholder scoring logic.
""")

uploaded_file = st.file_uploader("Upload your portfolio CSV", type=["csv"])

if uploaded_file is not None:
    df = load_csv(uploaded_file)

    st.subheader("📌 Raw Portfolio Data")
    st.dataframe(df, use_container_width=True)

    # Placeholder scoring function — replace with your real logic later
    scored_df = score_portfolio(df)

    st.subheader("📊 Portfolio Scoring Preview (Placeholder)")
    st.dataframe(scored_df, use_container_width=True)

else:
    st.info("Upload a CSV to get started.")
