import pandas as pd
from sklearn.model_selection import train_test_split
from v2_upgrade.src.config import DATA_PROCESSED, MODELS_DIR
from v2_upgrade.src.modeling.churn_model import build_churn_pipeline, save_pipeline

def main():
    df = pd.read_csv(DATA_PROCESSED / "telco_engineered.csv")

    y = df["Churn_Target"]
    X = df.drop(columns=["Churn", "Churn_Target"], errors="ignore")

    # keep id if present for downstream joins, but model will ignore it
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipe = build_churn_pipeline(X_train, y_train)
    pipe.fit(X_train.drop(columns=["customerID"], errors="ignore"), y_train)

    save_pipeline(pipe, str(MODELS_DIR / "churn_pipeline.joblib"))
    X_test.to_csv(MODELS_DIR / "X_test.csv", index=False)
    y_test.to_csv(MODELS_DIR / "y_test.csv", index=False)

    print("[OK] trained churn pipeline")
    print(f"[OK] saved model: {MODELS_DIR / 'churn_pipeline.joblib'}")
    print(f"[OK] saved X_test/y_test: {MODELS_DIR / 'X_test.csv'} / {MODELS_DIR / 'y_test.csv'}")

if __name__ == "__main__":
    main()
