import pandas as pd
from v2_upgrade.src.config import DATA_RAW

import pandas as pd
from v2_upgrade.src.config import DATA_RAW

def load_raw(filename: str | None = None) -> pd.DataFrame:
    """
    Load the raw telco churn dataset.

    Cloud ships: Churn_Telco.csv
    Local dev may also have: telco_customer_churn.csv
    """
    candidates = []
    if filename:
        candidates.append(filename)

    # Preferred order
    candidates += ["Churn_Telco.csv", "telco_customer_churn.csv"]

    for fn in candidates:
        p = DATA_RAW / fn
        if p.exists():
            return pd.read_csv(p)

    raise FileNotFoundError(f"Missing raw dataset in {DATA_RAW}. Tried: {candidates}")

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    if "Churn" in df.columns:
        df["Churn_Target"] = (df["Churn"].astype(str).str.lower() == "yes").astype(int)

    return df
