import pandas as pd

def load_csv(uploaded_file):
    """Load user-uploaded CSV into a clean DataFrame."""
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        raise ValueError(f"Error loading CSV: {e}")
