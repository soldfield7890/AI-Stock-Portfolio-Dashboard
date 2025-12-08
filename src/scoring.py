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


def detect_symbol_column(df: pd.DataFrame):
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
    Simple, explainable first-pass scoring model (0–100).

    Buckets:
    - Valuation (PE_TTM)
    - Profitability (ProfitMargin)
    - Dividend profile (DividendYield)
    - Risk profile (Beta)
    - Size (MarketCap)
    """

    score = 50.0

    pe = row.get("PE_TTM")
    pm = row.get("ProfitMargin")
    dy = row.get("DividendYield")
    beta = row.get("Beta")
    mc = row.get("MarketCap")

    # ---- Valuation: PE_TTM ----
    try:
        if pe is not None and not np.isnan(pe):
            if 0 < pe <= 15:
                score += 15
            elif 15 < pe <= 30:
                score += 10
            elif 30 < pe <= 40:
                score += 0
            elif 40 < pe <= 60:
                score -= 10
            elif pe > 60:
                score -= 20
    except Exception:
        pass

    # ---- Profitability: ProfitMargin ----
    try:
        if pm is not None and not np.isnan(pm):
            pm_pct = pm * 100 if pm < 1 else pm
            if pm_pct > 20:
                score += 10
            elif 5 < pm_pct <= 20:
                score += 5
            elif 0 < pm_pct <= 5:
                score += 0
            else:
                score -= 10
    except Exception:
        pass

    # ---- Dividend: DividendYield ----
    try:
        if dy is not None and not np.isnan(dy):
            dy_pct = dy * 100 if dy < 1 else dy
            if 1 <= dy_pct <= 5:
                score += 5
            elif dy_pct > 8:
                score -= 5
    except Exception:
        pass

    # ---- Risk profile: Beta ----
    try:
        if beta is not None and not np.isnan(beta):
            if 0.7 <= beta <= 1.3:
                score += 5
            elif beta < 0.7:
                score += 0
            else:
                score -= 5
    except Exception:
        pass

    # ---- Size: MarketCap ----
    try:
        if mc is not None and not np.isnan(mc):
            if mc >= 10_000_000_000:
                score += 5
            elif mc <= 1_000_000_000:
                score -= 5
    except Exception:
        pass

    return float(np.clip(score, 0, 100))


def decision_from_score(score: float) -> str:
    """
    Map 0–100 score to a simple action label.
    """
    if pd.isna(score):
        return "Watch / Review"

    if score >= 85:
        return "Strong Buy"
    if score >= 70:
        return "Buy"
    if score >= 55:
        return "Hold"
    if score >= 40:
        return "Trim"
    return "Exit / Avoid"


def score_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    scored = df.copy()

    # ----- 1) Compute basic portfolio metrics -----
    col_current_value = None
    col_cost_basis = None

    for col in scored.columns:
        cl = col.lower()
        if (
            ("current value" in cl or "market value" in cl or cl == "value")
            and col_current_value is None
        ):
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

        scored["Symbol_for_merge"] = scored[symbol_col].astype(str).str.upper()

        scored = scored.merge(
            fundamentals,
            left_on="Symbol_for_merge",
            right_on="Symbol",
            how="left",
            suffixes=("", "_fund"),
        )
    else:
        for col in ["PE_TTM", "ForwardPE", "DividendYield", "MarketCap", "Beta", "ProfitMargin"]:
            if col not in scored.columns:
                scored[col] = np.nan

    # ----- 3) Compute score & decision -----
    scored["Score"] = scored.apply(basic_fundamental_score, axis=1)
    scored["Decision"] = scored["Score"].apply(decision_from_score)

    return scored
