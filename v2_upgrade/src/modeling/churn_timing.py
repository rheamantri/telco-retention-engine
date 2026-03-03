import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

WINDOWS = [30, 60, 90]

def simulate_days_to_churn(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Creates a synthetic 'days_to_churn' column.
    - For churned customers, draw a plausible number of days until churn (0..180) influenced by tenure.
    - For non-churned customers, set NaN.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()

    if "Churn_Target" not in out.columns:
        raise ValueError("Churn_Target missing. Run basic_clean first.")

    # base distribution: most churn within 0-120 days, some later
    base = rng.triangular(left=0, mode=45, right=180, size=len(out))

    # tenure influence: longer tenure -> slightly longer time-to-churn on average
    tenure = out.get("tenure", pd.Series([0]*len(out))).fillna(0).to_numpy()
    tenure_factor = np.clip(tenure / 72.0, 0, 1)  # 0..1
    adj = base + (tenure_factor * 30)  # add up to +30 days

    days_to_churn = np.where(out["Churn_Target"].to_numpy() == 1, adj, np.nan)
    out["days_to_churn"] = np.round(np.clip(days_to_churn, 0, 365)).astype("float")

    return out

def make_window_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds y_30d, y_60d, y_90d columns where 1 means churn within that window.
    Non-churned customers get 0.
    """
    out = df.copy()
    if "days_to_churn" not in out.columns:
        raise ValueError("days_to_churn missing. Run simulate_days_to_churn first.")

    for w in WINDOWS:
        out[f"y_{w}d"] = ((out["days_to_churn"].notna()) & (out["days_to_churn"] <= w)).astype(int)

    return out

def predict_window_risks(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """
    Returns probabilities for class 1.
    """
    X_model = X.drop(columns=["customerID"], errors="ignore")
    return pipe.predict_proba(X_model)[:, 1]
