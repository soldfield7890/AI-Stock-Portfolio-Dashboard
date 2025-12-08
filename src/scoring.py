import pandas as pd
import numpy as np
from .fundamentals import fetch_fundamentals

def to_float(series):
    """
    Safely convert a pandas Series to float by removing $, commas,
    percent signs, and handling blanks.
    """
    return (
        series.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan, None: np.nan})
        .astype(float)
    )

def detect_symbol_column(df: pd.DataFrame) -> str | None:
    """
    Try to find the ticker/symbol column by name.
    """
    for col in df.columns:
        cl = col.lower()
        if "symbol" in cl or "ticker" in cl:
            return col
    return None

def basic_fundamental_score(row) -> float:
    """
    Very simple first-pass scoring model (0–100).

    - Start at 50
    - +15 if PE_TTM between 10 and 30
    - +10 if ProfitMargin > 0
    - +10 if DividendYield between 1% and 5%
    - -10 if PE_TTM > 60
    - Clip between 0 and 100
    """
    score = 50

    pe = row.get("PE_TTM")
    pm = row.get("ProfitMargin")
    dy = row.get("DividendYield")

    try:
        if pe is not None and not np.isnan(pe):
            if 10 <= pe <= 30:
                score += 15
            elif pe > 60:
                score -= 10
        if pm is not None and not np.isnan(pm):
            if pm > 0:
                score += 10
        if dy is not None and not np.isnan(dy):
            # yfinance dividendYield is usually in decimal (0.02 = 2%)
            dy_pct = dy * 100 if dy < 1 else dy
            if 1 <= dy_pct <= 5:
                score += 10
    except Exception:
        pass

    return float(np.clip(score, 0, 100))

def score_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    scored = df.copy()

    # ----- 1) Compute basic portfolio metrics -----
    col_current_value = None
    col_cost_basis = None

    for col in scored.columns:
        cl = col.lower()
        if ("current value" in cl or "market value" in cl or cl == "value") and col_current_value is None:
            col_current_value = col
        if "cost basis" in cl or "total cost" in cl or cl == "cost":
            col_cost_basis = col

    if col_current_value and col_cost_basis:
        current_val = to_float(scored[col_current_value])
        cost_basis = to_float(scored[col_cost_basis])

        scored["CurrentValue_clean"] = current_val
        scored["CostBasis_clean"] = cost_basis

        scored["UnrealizedPL"] = current_val - cost_basis
        scored["UnrealizedPLPct"] = scored["UnrealizedPL"] / cost_basis.replace(0, np.nan)

        total_value = current_val.sum()
        scored["PortfolioWeightPct"] = (current_val / total_value) * 100 if total_value else np.nan
    else:
        scored["CurrentValue_clean"] = np.nan
        scored["CostBasis_clean"] = np.nan
        scored["UnrealizedPL"] = np.nan
        scored["UnrealizedPLPct"] = np.nan
        scored["PortfolioWeightPct"] = np.nan

    # ----- 2) Attach fundamentals -----
    symbol_col = detect_symbol_column(scored)
    if symbol_col:
        fundamentals = fetch_fundamentals(scored[symbol_col])
        fundamentals["Symbol"] = fundamentals["Symbol"].astype(str).str.upper()

        # Normalize symbol column for merge
        scored["Symbol_for_merge"] = scored[symbol_col].astype(str).str.upper()
        scored = scored.merge(
            fundamentals,
            left_on="Symbol_for_merge",
            right_on="Symbol",
            how="left",
            suffixes=("", "_fund"),
        )
    else:
        # Ensure columns exist even if we couldn't fetch fundamentals
        for col in ["PE_TTM", "ForwardPE", "DividendYield", "MarketCap", "Beta", "ProfitMargin"]:
            if col not in scored.columns:
                scored[col] = np.nan

    # ----- 3) Compute simple score (v0.1) -----
    scored["Score"] = scored.apply(basic_fundamental_score, axis=1)

    return scored
