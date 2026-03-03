import json
import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

from v2_upgrade.src.config import MODELS_DIR, REPORTS_DIR
from v2_upgrade.src.modeling.churn_model import load_pipeline

def metrics(y_true, probs, thr=0.5):
    y_pred = (probs >= thr).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, probs)) if len(set(y_true)) > 1 else None,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "thr": float(thr),
        "positives": int(sum(y_true)),
        "n": int(len(y_true)),
    }

def main():
    out = {}

    X_test = pd.read_csv(MODELS_DIR / "X_test.csv")
    y_churn = pd.read_csv(MODELS_DIR / "y_test.csv").values.ravel()

    churn_pipe = load_pipeline(str(MODELS_DIR / "churn_pipeline.joblib"))
    churn_probs = churn_pipe.predict_proba(X_test.drop(columns=["customerID"], errors="ignore"))[:, 1]
    out["churn_model"] = metrics(y_churn, churn_probs, thr=0.5)

    for w in [30, 60, 90]:
        df = pd.read_csv(REPORTS_DIR / f"timing_test_{w}d.csv")
        out[f"timing_{w}d"] = metrics(df["label"].values, df[f"risk_{w}d"].values, thr=0.5)

    report_path = REPORTS_DIR / "model_metrics.json"
    report_path.write_text(json.dumps(out, indent=2))
    print(f"[OK] saved metrics: {report_path}")
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
