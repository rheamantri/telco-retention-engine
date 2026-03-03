import pandas as pd
import numpy as np
import shap

from v2_upgrade.src.config import MODELS_DIR, REPORTS_DIR
from v2_upgrade.src.modeling.churn_model import load_pipeline

def main():
    pipe = load_pipeline(str(MODELS_DIR / "churn_pipeline.joblib"))
    X_test = pd.read_csv(MODELS_DIR / "X_test.csv")
    X_model = X_test.drop(columns=["customerID"], errors="ignore")

    pre = pipe.named_steps["preprocessor"]
    clf = pipe.named_steps["classifier"]

    X_enc = pre.transform(X_model)
    feature_names = pre.get_feature_names_out()

    explainer = shap.TreeExplainer(clf)
    shap_vals = explainer.shap_values(X_enc)

    mean_abs = np.abs(shap_vals).mean(axis=0)
    df_imp = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs
    }).sort_values("mean_abs_shap", ascending=False)

    out = REPORTS_DIR / "global_shap_importance.csv"
    df_imp.to_csv(out, index=False)

    print(f"[OK] saved: {out}")
    print(df_imp.head(20).to_string(index=False))

if __name__ == "__main__":
    main()
