import pandas as pd
import yfinance as yf

def fetch_fundamentals(symbols):
    """
    Fetch basic fundamentals for a list of ticker symbols using yfinance.

    Returns a DataFrame with one row per symbol and columns like:
    - PE_TTM
    - ForwardPE
    - DividendYield
    - MarketCap
    - Beta
    - ProfitMargin
    """
    symbols = [s for s in pd.Series(symbols).dropna().unique()]

    rows = []
    for sym in symbols:
        sym = str(sym).strip().upper()
        if not sym:
            continue

        try:
            t = yf.Ticker(sym)
            info = t.info  # yfinance metadata dict
        except Exception:
            info = {}

        rows.append(
            {
                "Symbol": sym,
                "PE_TTM": info.get("trailingPE"),
                "ForwardPE": info.get("forwardPE"),
                "DividendYield": info.get("dividendYield"),
                "MarketCap": info.get("marketCap"),
                "Beta": info.get("beta"),
                "ProfitMargin": info.get("profitMargins"),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["Symbol", "PE_TTM", "ForwardPE", "DividendYield", "MarketCap", "Beta", "ProfitMargin"])

    return pd.DataFrame(rows)
