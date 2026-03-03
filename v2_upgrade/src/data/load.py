import pandas as pd
from v2_upgrade.src.config import DATA_RAW

def load_raw(filename: str = "telco_customer_churn.csv") -> pd.DataFrame:
    return pd.read_csv(DATA_RAW / filename)

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    if "Churn" in df.columns:
        df["Churn_Target"] = (df["Churn"].astype(str).str.lower() == "yes").astype(int)

    return df
