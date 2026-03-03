import numpy as np
import pandas as pd

INTERNET_COLS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies"
]

def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "InternetService" in df.columns:
        mask_no_internet = df["InternetService"] == "No"
        for c in INTERNET_COLS:
            if c in df.columns:
                df.loc[mask_no_internet, c] = "No internet service"

    if "PhoneService" in df.columns and "MultipleLines" in df.columns:
        mask_no_phone = df["PhoneService"] == "No"
        df.loc[mask_no_phone, "MultipleLines"] = "No phone service"

    if "tenure" in df.columns:
        df.loc[(df["tenure"] < 0) | (df["tenure"] > 120), "tenure"] = np.nan
        df["Tenure_Bucket"] = pd.cut(
            df["tenure"],
            bins=[-1, 12, 24, 48, 80, 120],
            labels=["0-1y", "1-2y", "2-4y", "4-6y", "6-10y"]
        )

    if "MonthlyCharges" in df.columns:
        df.loc[df["MonthlyCharges"] < 0, "MonthlyCharges"] = np.nan

    cols_to_count = [c for c in (["PhoneService", "MultipleLines", "Partner", "Dependents"] + INTERNET_COLS) if c in df.columns]
    if cols_to_count:
        df["Service_Count"] = (df[cols_to_count] == "Yes").sum(axis=1)

    if "TotalCharges" in df.columns and "tenure" in df.columns:
        df["Avg_Historical_Charge"] = df["TotalCharges"] / (df["tenure"].fillna(0) + 1)

    if "PaymentMethod" in df.columns:
        df["Payment_Simple"] = df["PaymentMethod"].astype(str).str.lower().apply(
            lambda x: "Automatic" if "automatic" in x else "Manual"
        )

    return df
