import pandas as pd
import numpy as np

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

def score_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    scored = df.copy()

    # Detect columns
    col_current_value = None
    col_cost_basis = None

    for col in scored.columns:
        cl = col.lower()
        if "current value" in cl or "market value" in cl or "value" in cl:
            col_current_value = col
        if "cost basis" in cl or "total cost" in cl or "cost" in cl:
            col_cost_basis = col

    # If we have both value + cost, compute metrics
    if col_current_value and col_cost_basis:

        # Clean numeric columns
        current_val = to_float(scored[col_current_value])
        cost_basis = to_float(scored[col_cost_basis])

        scored["CurrentValue_clean"] = current_val
        scored["CostBasis_clean"] = cost_basis

        scored["UnrealizedPL"] = current_val - cost_basis
        scored["UnrealizedPLPct"] = scored["UnrealizedPL"] / cost_basis.replace(0, np.nan)

        # Portfolio weight
        total_value = current_val.sum()
        scored["PortfolioWeightPct"] = (current_val / total_value) * 100 if total_value else np.nan

    else:
        scored["UnrealizedPL"] = np.nan
        scored["UnrealizedPLPct"] = np.nan
        scored["PortfolioWeightPct"] = np.nan

    # Placeholder score
    scored["Score"] = 50

    return scored
