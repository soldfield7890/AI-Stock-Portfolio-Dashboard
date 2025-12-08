# Oldfield AI Stock Dashboard (v1)

This is the starter structure for Stephen Oldfield’s AI-powered stock evaluation dashboard.

## How to use

1. Upload your portfolio CSV in the Streamlit interface.
2. The app displays raw data.
3. The app runs a placeholder scoring engine (to be replaced with your EM-style model).

## Structure

```text
oldfield-ai-stock-dashboard/
│
├─ app.py
├─ requirements.txt
├─ README.md
│
├─ src/
│   ├─ loaders.py
│   └─ scoring.py
│
└─ data/
    └─ sample_portfolio.csv
```
