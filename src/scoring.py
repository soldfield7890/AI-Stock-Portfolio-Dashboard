import pandas as pd

def score_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    """    Placeholder scoring model.

    This simply copies the DataFrame and adds a dummy score column
    so the dashboard can run. Replace this with your real scoring logic.
    """
    scored = df.copy()
    scored["Score"] = 50  # neutral placeholder score
    return scored
