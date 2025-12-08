import pandas as pd

def score_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic portfolio metrics.

    Adds:
    - UnrealizedPL: Current Value - Cost Basis Total
    - UnrealizedPLPct: UnrealizedPL / Cost Basis Total
    - PortfolioWeightPct: Current Value / portfolio total value
    - Score: placeholder for future Everything Money / AI model
    """
    scored = df.copy()

    # Try to find the relevant columns by name
    # Adjust these strings if your CSV uses slightly different headers
    col_current_value = None
    col_cost_basis = None

    for col in scored.columns:
        col_lower = col.lower()
        if "current value" in col_lower or "market value" in col_lower:
            col_current_value = col
        if "cost basis total" in col_lower or "total cost" in col_lower:
            col_cost_basis = col

    # Only add metrics if we found the needed columns
    if col_current_value is not None and col_cost_basis is not None:
        current_val = scored[col_current_value].astype(float)
        cost_basis = scored[col_cost_basis].astype(float)

        scored["UnrealizedPL"] = current_val - cost_basis
        scored["UnrealizedPLPct"] = scored["UnrealizedPL"] / cost_basis.replace(0, pd.NA)

        total_value = current_val.sum()
        if total_value != 0:
            scored["PortfolioWeightPct"] = current_val / total_value * 100
        else:
            scored["PortfolioWeightPct"] = 0.0
    else:
        # If we can't find the columns, just fill defaults so the app doesn't crash
        scored["UnrealizedPL"] = pd.NA
        scored["UnrealizedPLPct"] = pd.NA
        scored["PortfolioWeightPct"] = pd.NA

    # Placeholder score for now – we will replace this with the real model
    scored["Score"] = 50

    return scored
