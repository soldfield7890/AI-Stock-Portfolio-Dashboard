import pandas as pd
import numpy as np
import yfinance as yf

def safe_get(d: dict, key: str):
    """Return a value from dict or NaN if missing."""
    val = d.get(key)
    return np.nan if val is None else val

def fetch_fundamentals(symbols):
    """
    Fetch fundamentals from multiple yfinance sources:
    - fast_info (best for market cap, beta, last price)
    - info / get_info (good fallback for PE, dividends, profit margins)
    """

    symbols = [str(s).strip().upper() for s in pd.Series(symbols).dropna().unique()]
    rows = []

    for sym in symbols:
        try:
            t = yf.Ticker(sym)

            # Pull data from multiple APIs
            fast = t.fast_info if hasattr(t, "fast_info") else {}
            info = t.info if hasattr(t, "info") else {}
            try:
                full_info = t.get_info()
            except Exception:
                full_info = {}

            # Merge all sources
            merged = {}
            merged.update(fast or {})
            merged.update(info or {})
            merged.update(full_info or {})

            row = {
                "Symbol": sym,
                "MarketCap": safe_get(merged, "marketCap") or safe_get(merged, "market_cap"),
                "PE_TTM": safe_get(merged, "trailingPE"),
                "ForwardPE": safe_get(merged, "forwardPE"),
                "DividendYield": safe_get(merged, "dividendYield"),
                "Beta": safe_get(merged, "beta"),
                "ProfitMargin": safe_get(merged, "profitMargins"),
            }

            rows.append(row)

        except Exception:
            rows.append({
                "Symbol": sym,
                "MarketCap": np.nan,
                "PE_TTM": np.nan,
                "ForwardPE": np.nan,
                "DividendYield": np.nan,
                "Beta": np.nan,
                "ProfitMargin": np.nan,
            })

    return pd.DataFrame(rows)
