import pandas as pd

BUCKET_RULES = [
    ("Contract", "Contract & Commitment"),
    ("PaymentMethod", "Billing & Payments"),
    ("MonthlyCharges", "Pricing"),
    ("TotalCharges", "Pricing"),
    ("Avg_Historical_Charge", "Pricing"),
    ("TechSupport", "Support Experience"),
    ("OnlineSecurity", "Product Value"),
    ("DeviceProtection", "Product Value"),
    ("InternetService", "Product Value"),
    ("Streaming", "Product Usage"),
    ("tenure", "Tenure & Lifecycle"),
    ("Tenure", "Tenure & Lifecycle"),
    ("MultipleLines", "Plan Structure"),
]

ACTION_RULES = [
    ("Contract = Month-to-month", "Offer 12-month contract incentive"),
    ("PaymentMethod = Electronic check", "Nudge to autopay + small credit"),
    ("InternetService = Fiber optic", "Proactive service quality check"),
    ("TechSupport = No", "Offer support onboarding / free month support"),
    ("OnlineSecurity = No", "Bundle security add-on trial"),
    ("MonthlyCharges", "Offer targeted discount / plan optimization"),
    ("TotalCharges", "Offer plan reprice / discount review"),
    ("tenure", "New customer onboarding campaign"),
]

def bucket_reason(reason: str) -> str:
    if not isinstance(reason, str) or reason.strip() == "":
        return ""
    for key, bucket in BUCKET_RULES:
        if key in reason:
            return bucket
    return "Other"

def recommend_action(reasons: list[str]) -> str:
    for r in reasons:
        for key, action in ACTION_RULES:
            if key in r:
                return action
    return "Review account for retention offer"

def add_buckets_and_actions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for k in [1, 2, 3]:
        df[f"bucket_{k}"] = df[f"reason_{k}"].apply(bucket_reason)

    df["recommended_action"] = df.apply(
        lambda row: recommend_action([row.get("reason_1",""), row.get("reason_2",""), row.get("reason_3","")]),
        axis=1
    )

    return df
