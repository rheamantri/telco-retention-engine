import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

NO_SPLIT_PREFIXES = {"Avg_Historical_Charge", "Service_Count", "TotalCharges", "MonthlyCharges"}

def _clean_feature_name(name: str) -> str:
    n = name.replace("num__", "").replace("cat__", "").strip()

    if n in NO_SPLIT_PREFIXES:
        return n

    if "_" in n:
        left, right = n.split("_", 1)
        return f"{left} = {right}".strip()

    return n

def compute_reason_codes(
    pipe: Pipeline,
    X: pd.DataFrame,
    top_k: int = 3,
) -> pd.DataFrame:
    """
    Returns df with columns:
      - reason_1, reason_2, reason_3
      - impact_1, impact_2, impact_3 (SHAP values)
      - churn_prob
    """
    X_model = X.drop(columns=["customerID"], errors="ignore")

    pre = pipe.named_steps["preprocessor"]
    clf = pipe.named_steps["classifier"]

    X_enc = pre.transform(X_model)
    feature_names = pre.get_feature_names_out()

    explainer = shap.TreeExplainer(clf)
    shap_vals = explainer.shap_values(X_enc)

    # churn probability
    churn_prob = pipe.predict_proba(X_model)[:, 1]

    out_rows = []
    for i in range(X_enc.shape[0]):
        sv = shap_vals[i]
        # positive contributions only (push toward churn)
        pos_idx = np.where(sv > 0)[0]
        if len(pos_idx) == 0:
            # fallback: take biggest absolute
            ranked = np.argsort(np.abs(sv))[::-1][:top_k]
        else:
            ranked = pos_idx[np.argsort(sv[pos_idx])[::-1]][:top_k]

        reasons = [_clean_feature_name(feature_names[j]) for j in ranked]
        impacts = [float(sv[j]) for j in ranked]

        row = {
            "churn_prob": float(churn_prob[i]),
        }
        for k in range(top_k):
            row[f"reason_{k+1}"] = reasons[k] if k < len(reasons) else ""
            row[f"impact_{k+1}"] = impacts[k] if k < len(impacts) else 0.0

        out_rows.append(row)

    return pd.DataFrame(out_rows)
