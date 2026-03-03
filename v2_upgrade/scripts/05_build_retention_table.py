import joblib
import pandas as pd

from v2_upgrade.src.config import MODELS_DIR, REPORTS_DIR
from v2_upgrade.src.modeling.churn_model import load_pipeline
from v2_upgrade.src.explain.reason_codes import compute_reason_codes
from v2_upgrade.src.explain.reason_policy import add_buckets_and_actions

def main():
    # base churn model
    churn_pipe = load_pipeline(str(MODELS_DIR / "churn_pipeline.joblib"))
    X_test = pd.read_csv(MODELS_DIR / "X_test.csv")

    # reason codes + actions
    rc = compute_reason_codes(churn_pipe, X_test, top_k=3)
    if "customerID" in X_test.columns:
        rc.insert(0, "customerID", X_test["customerID"].values)
    rc = add_buckets_and_actions(rc)

    # timing models
    for w in [30, 60, 90]:
        tm = joblib.load(MODELS_DIR / f"timing_model_{w}d.joblib")
        probs = tm.predict_proba(X_test.drop(columns=["customerID"], errors="ignore"))[:, 1]
        rc[f"risk_{w}d"] = probs

    # sort by near-term risk
    rc = rc.sort_values("risk_30d", ascending=False)

    out = REPORTS_DIR / "retention_table_test.csv"
    rc.to_csv(out, index=False)

    print(f"[OK] saved: {out}")
    print(rc.head(10)[["customerID","risk_30d","risk_60d","risk_90d","churn_prob","reason_1","recommended_action"]].to_string(index=False))

if __name__ == "__main__":
    main()
