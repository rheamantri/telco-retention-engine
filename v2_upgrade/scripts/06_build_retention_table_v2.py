import json
import joblib
import pandas as pd
from pathlib import Path

from v2_upgrade.src.config import MODELS_DIR, REPORTS_DIR
from v2_upgrade.src.modeling.churn_model import load_pipeline
from v2_upgrade.src.explain.reason_codes import compute_reason_codes
from v2_upgrade.src.explain.reason_policy import add_buckets_and_actions

from v2_upgrade.src.retention.clv import add_clv, add_value_segment
from v2_upgrade.src.retention.optimizer import (
    compute_expected_loss, compute_priority_score, add_intervention_and_roi
)

CONFIG_PATH = Path("v2_upgrade/config/retention_config.json")

def main():
    cfg = json.loads(CONFIG_PATH.read_text())

    # Base churn model
    churn_pipe = load_pipeline(str(MODELS_DIR / "churn_pipeline.joblib"))
    X_test = pd.read_csv(MODELS_DIR / "X_test.csv")

    # Reason codes + actions
    rc = compute_reason_codes(churn_pipe, X_test, top_k=3)
    if "customerID" in X_test.columns:
        rc.insert(0, "customerID", X_test["customerID"].values)
    rc = add_buckets_and_actions(rc)

    # Timing risks
    for w in [30, 60, 90]:
        tm = joblib.load(MODELS_DIR / f"timing_model_{w}d.joblib")
        rc[f"risk_{w}d"] = tm.predict_proba(X_test.drop(columns=["customerID"], errors="ignore"))[:, 1]

    # Attach basic numeric fields for CLV (tenure, MonthlyCharges) from X_test
    for c in ["tenure", "MonthlyCharges", "TotalCharges", "Contract", "PaymentMethod", "InternetService"]:
        if c in X_test.columns and c not in rc.columns:
            rc[c] = X_test[c].values

    # CLV
    rc = add_clv(
        rc,
        margin_rate=cfg["margin_rate"],
        expected_future_months=cfg["expected_future_months"],
    )
    rc = add_value_segment(
        rc,
        high_value_clv=cfg["value_segments"]["high_value_clv"],
        mid_value_clv=cfg["value_segments"]["mid_value_clv"],
    )

    # Optimizer outputs
    rc = compute_expected_loss(rc)
    rc = compute_priority_score(
        rc,
        w30=cfg["risk_weights"]["w30"],
        w60=cfg["risk_weights"]["w60"],
        w90=cfg["risk_weights"]["w90"],
    )
    rc = add_intervention_and_roi(rc, costs=cfg["costs"])

    # Sort for ops
    rc = rc.sort_values(["should_intervene", "priority_score", "expected_loss_30d"], ascending=[False, False, False])

    out = REPORTS_DIR / "retention_table_v2.csv"
    rc.to_csv(out, index=False)

    print(f"[OK] saved: {out}")
    print(rc.head(10)[[
        "customerID", "risk_30d", "risk_60d", "risk_90d",
        "clv", "expected_loss_30d", "offer_cost", "roi_proxy",
        "should_intervene", "intervention_type",
        "reason_1", "bucket_1", "recommended_action"
    ]].to_string(index=False))

if __name__ == "__main__":
    main()
