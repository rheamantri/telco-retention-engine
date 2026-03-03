import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from v2_upgrade.src.config import DATA_PROCESSED, MODELS_DIR, REPORTS_DIR
from v2_upgrade.src.modeling.churn_model import build_churn_pipeline
from v2_upgrade.src.modeling.churn_timing import simulate_days_to_churn, make_window_labels, WINDOWS

def train_one(df: pd.DataFrame, y_col: str):
    y = df[y_col]
    X = df.drop(columns=["Churn", "Churn_Target", "days_to_churn"] + [c for c in df.columns if c.startswith("y_")], errors="ignore")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipe = build_churn_pipeline(X_train, y_train)
    pipe.fit(X_train.drop(columns=["customerID"], errors="ignore"), y_train)

    return pipe, X_test, y_test

def main():
    df = pd.read_csv(DATA_PROCESSED / "telco_engineered.csv")

    # simulate timing + window labels
    df = simulate_days_to_churn(df, seed=42)
    df = make_window_labels(df)

    results = []
    for w in WINDOWS:
        y_col = f"y_{w}d"
        pipe, X_test, y_test = train_one(df, y_col)

        model_path = MODELS_DIR / f"timing_model_{w}d.joblib"
        joblib.dump(pipe, model_path)

        # save a small test prediction file for inspection
        probs = pipe.predict_proba(X_test.drop(columns=["customerID"], errors="ignore"))[:, 1]
        tmp = pd.DataFrame({
            "customerID": X_test["customerID"].values if "customerID" in X_test.columns else range(len(X_test)),
            f"risk_{w}d": probs,
            "label": y_test.values
        })
        out_path = REPORTS_DIR / f"timing_test_{w}d.csv"
        tmp.to_csv(out_path, index=False)

        results.append((w, float(tmp[f"risk_{w}d"].mean()), int(tmp["label"].sum())))
        print(f"[OK] trained timing model {w}d -> {model_path}")
        print(f"[OK] saved: {out_path}")

    print("\n[SUMMARY] window / avg_pred_risk / positives_in_test")
    for w, avg_risk, pos in results:
        print(f"{w}d  |  {avg_risk:.4f}  |  {pos}")

if __name__ == "__main__":
    main()
