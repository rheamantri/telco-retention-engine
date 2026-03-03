import pandas as pd
from v2_upgrade.src.config import MODELS_DIR, REPORTS_DIR
from v2_upgrade.src.modeling.churn_model import load_pipeline
from v2_upgrade.src.explain.reason_codes import compute_reason_codes
from v2_upgrade.src.explain.reason_policy import add_buckets_and_actions

def main():
    pipe = load_pipeline(str(MODELS_DIR / "churn_pipeline.joblib"))
    X_test = pd.read_csv(MODELS_DIR / "X_test.csv")

    rc = compute_reason_codes(pipe, X_test, top_k=3)

    if "customerID" in X_test.columns:
        rc.insert(0, "customerID", X_test["customerID"].values)

    rc = add_buckets_and_actions(rc)

    out = REPORTS_DIR / "reason_codes_test.csv"
    rc.to_csv(out, index=False)

    print(f"[OK] saved reason codes: {out}")
    print(rc.head(5)[[
        "customerID","churn_prob",
        "reason_1","bucket_1","impact_1",
        "reason_2","bucket_2","impact_2",
        "reason_3","bucket_3","impact_3",
        "recommended_action"
    ]].to_string(index=False))

if __name__ == "__main__":
    main()
