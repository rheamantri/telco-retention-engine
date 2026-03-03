import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    RocCurveDisplay,
    classification_report
)

from v2_upgrade.src.config import DATA_RAW, MODELS_DIR, REPORTS_DIR, FIGURES_DIR


# ----------------------------
def save_plot(filename: str):
    path = FIGURES_DIR / filename
    plt.tight_layout()
    plt.savefig(path)
    print(f"[OK] saved plot: {path}")
    plt.close()


def feature_engineer_robust(df: pd.DataFrame) -> pd.DataFrame:
    df_feat = df.copy()

    internet_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    mask_no_internet = df_feat["InternetService"] == "No"
    df_feat.loc[mask_no_internet, internet_cols] = "No internet service"

    mask_no_phone = df_feat["PhoneService"] == "No"
    df_feat.loc[mask_no_phone, "MultipleLines"] = "No phone service"

    df_feat.loc[(df_feat["tenure"] < 0) | (df_feat["tenure"] > 120), "tenure"] = np.nan
    df_feat.loc[df_feat["MonthlyCharges"] < 0, "MonthlyCharges"] = np.nan

    df_feat["Tenure_Bucket"] = pd.cut(
        df_feat["tenure"],
        bins=[-1, 12, 24, 48, 80],
        labels=["0-1y", "1-2y", "2-4y", "4-6y+"]
    )

    df_feat["Service_Count"] = (
        (df_feat[["PhoneService", "MultipleLines", "Partner", "Dependents"] + internet_cols] == "Yes")
        .sum(axis=1)
    )

    df_feat["Avg_Historical_Charge"] = df_feat["TotalCharges"] / (df_feat["tenure"] + 1)

    if "PaymentMethod" in df_feat.columns:
        df_feat["Payment_Simple"] = df_feat["PaymentMethod"].apply(
            lambda x: "Automatic" if "automatic" in str(x).lower() else "Manual"
        )

    return df_feat


def generate_eda_plots(df: pd.DataFrame):
    print("[INFO] generating EDA plots...")

    num_cols = df.select_dtypes(include=np.number).columns
    if len(num_cols) > 0:
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Feature Correlation Heatmap")
        save_plot("heatmap_correlation.png")

    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    ignore = ["customerID", "Churn", "Churn_Target"]
    for col in cat_cols:
        if col in ignore:
            continue
        if col not in df.columns:
            continue
        try:
            plt.figure(figsize=(10, 5))
            rate = (
                df.groupby(col)["Churn_Target"]
                .mean()
                .reset_index()
                .sort_values("Churn_Target", ascending=False)
            )
            sns.barplot(x=col, y="Churn_Target", data=rate)
            plt.xticks(rotation=25, ha="right")
            plt.title(f"Churn Rate by {col}")
            save_plot(f"churn_rate_by_{col}.png")
        except Exception:
            pass

    plot_df = df.copy()
    plot_df["Churn_Label"] = plot_df["Churn_Target"].apply(lambda x: "Yes" if x == 1 else "No")
    numerical_cols_to_plot = ["MonthlyCharges", "TotalCharges"]
    if "tenure" in plot_df.columns:
        plot_df["Tenure_Years"] = plot_df["tenure"] / 12
        numerical_cols_to_plot.insert(0, "Tenure_Years")

    for col in numerical_cols_to_plot:
        if col not in plot_df.columns:
            continue
        plt.figure(figsize=(8, 5))
        sns.violinplot(x="Churn_Label", y=col, data=plot_df)
        plt.title("Distribution of Tenure (Years) by Churn" if col == "Tenure_Years" else f"Distribution of {col} by Churn")
        save_plot(f"violin_plot_{col}_vs_churn.png")


def evaluate_and_save_metrics(model, X_test, y_test, cost_fn=200, cost_fp=50):
    print("[INFO] calculating metrics & cost trade-offs...")
    y_probs = model.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)

    report_lines = []
    def log(s):
        print(s)
        report_lines.append(s)

    log("============================================================")
    log(f"COST COMPARISON (FN=${cost_fn}, FP=${cost_fp})")
    log("============================================================")
    log(f"{'Method':<15} | {'Threshold':<10} | {'Total Cost ($)':<15} | {'Recall':<10} | {'Precision':<10}")
    log("-" * 75)

    betas = [1.0, 1.5, 2.0]
    for b in betas:
        num = (1 + b**2) * precisions * recalls
        den = (b**2 * precisions + recalls)
        f_scores = np.divide(num, den, out=np.zeros_like(num), where=(den != 0))
        idx = np.argmax(f_scores[:-1]) if len(f_scores) > 1 else 0
        thr = thresholds[idx]

        y_pred = (y_probs >= thr).astype(int)
        TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()
        cost = (FN * cost_fn) + (FP * cost_fp)

        log(f"Beta {b:<10} | {thr:.4f}     | ${cost:,.2f}      | {recalls[idx]:.4f}     | {precisions[idx]:.4f}")

    scan_thresholds = np.linspace(0, 1, 101)
    min_cost = float("inf")
    best_thr = 0.5
    for t in scan_thresholds:
        y_pred_t = (y_probs >= t).astype(int)
        TN, FP, FN, TP = confusion_matrix(y_test, y_pred_t).ravel()
        total_cost = (FN * cost_fn) + (FP * cost_fp)
        if total_cost < min_cost:
            min_cost = total_cost
            best_thr = t

    y_pred_opt = (y_probs >= best_thr).astype(int)
    TN, FP, FN, TP = confusion_matrix(y_test, y_pred_opt).ravel()
    rec_opt = TP / (TP + FN) if (TP + FN) > 0 else 0
    prec_opt = TP / (TP + FP) if (TP + FP) > 0 else 0
    spec_opt = TN / (TN + FP) if (TN + FP) > 0 else 0

    log(f"{'MIN COST':<15} | {best_thr:.4f}     | ${min_cost:,.2f}      | {rec_opt:.4f}     | {prec_opt:.4f}")
    log("-" * 75)
    log("\n--- Classification Report at Cost-Optimal Threshold ---")
    log(classification_report(y_test, y_pred_opt))
    log("\n--- Key Classification Metrics ---")
    log(f"Sensitivity (Recall): {rec_opt:.4f}")
    log(f"Precision: {prec_opt:.4f}")
    log(f"Specificity: {spec_opt:.4f}")

    report_path = REPORTS_DIR / "model_performance_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"[OK] saved report: {report_path}")

    return float(best_thr)


def generate_model_plots(model, X_test, y_test, optimal_thr=0.5):
    print("[INFO] generating model performance plots...")
    y_probs = model.predict_proba(X_test)[:, 1]

    # ROC
    plt.figure(figsize=(8, 6))
    RocCurveDisplay.from_predictions(y_test, y_probs, name="XGBoost")
    plt.plot([0, 1], [0, 1], "k--")
    plt.title("ROC Curve")
    save_plot("roc_curve.png")

    # Calibration
    plt.figure(figsize=(8, 8))
    CalibrationDisplay.from_predictions(y_test, y_probs, n_bins=10, name="XGBoost")
    plt.title("Calibration Curve")
    save_plot("calibration_curve.png")

    # Probability distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(x=y_probs, hue=y_test, bins=30, kde=True, element="step")
    plt.axvline(optimal_thr, color="black", linestyle="--", label=f"Optimal Thr: {optimal_thr:.2f}")
    plt.legend()
    plt.title("Predicted Probability Distribution")
    save_plot("probability_distribution.png")

    # Decile plot
    df_res = pd.DataFrame({"actual": y_test, "prob": y_probs})
    df_res["decile"] = pd.qcut(df_res["prob"], 10, labels=False, duplicates="drop")
    decile_stats = df_res.groupby("decile").agg(actual=("actual", "mean"), pred=("prob", "mean")).reset_index()

    plt.figure(figsize=(10, 6))
    x = np.arange(len(decile_stats))
    width = 0.35
    plt.bar(x - width / 2, decile_stats["actual"], width, label="Actual Churn Rate")
    plt.bar(x + width / 2, decile_stats["pred"], width, label="Predicted Probability")
    plt.xlabel("Risk Decile (0=Low, 9=High)")
    plt.title("Calibration: Actual vs Predicted by Decile")
    plt.xticks(x, decile_stats["decile"])
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    save_plot("actual_vs_predicted_decile_plot.png")

    # Confusion matrix at optimal threshold
    y_pred = (y_probs >= optimal_thr).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Stay", "Churn"])
    plt.figure(figsize=(6, 6))
    disp.plot(ax=plt.gca(), values_format="d")
    plt.title(f"Confusion Matrix (Thr={optimal_thr:.2f})")
    save_plot("confusion_matrix_optimized.png")


def main():
    print("[INFO] loading data...")
    csv_path = DATA_RAW / "Churn_Telco.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing dataset at: {csv_path}")

    df = pd.read_csv(csv_path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["Churn_Target"] = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)

    df_final = feature_engineer_robust(df)

    # EDA
    generate_eda_plots(df_final)

    # split
    X = df_final.drop(columns=["Churn", "Churn_Target"])
    y = df_final["Churn_Target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # save test sets
    (MODELS_DIR / "X_test.csv").write_text(X_test.to_csv(index=False))
    (MODELS_DIR / "y_test.csv").write_text(pd.DataFrame({"y": y_test}).to_csv(index=False))
    print(f"[OK] saved: {MODELS_DIR / 'X_test.csv'}")
    print(f"[OK] saved: {MODELS_DIR / 'y_test.csv'}")

    # pipeline
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if "customerID" in cat_cols:
        cat_cols.remove("customerID")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )

    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", XGBClassifier(
                objective="binary:logistic",
                eval_metric="auc",
                scale_pos_weight=pos_weight,
                random_state=42,
                n_jobs=-1
            )),
        ]
    )

    print("[INFO] training XGBoost pipeline...")
    model.fit(X_train.drop(columns=["customerID"], errors="ignore"), y_train)

    out_model = MODELS_DIR / "churn_pipeline.joblib"
    joblib.dump(model, out_model)
    print(f"[OK] saved model: {out_model}")

    # evaluate + plots
    X_test_eval = X_test.drop(columns=["customerID"], errors="ignore")
    optimal_thr = evaluate_and_save_metrics(model, X_test_eval, y_test, cost_fn=200, cost_fp=50)
    generate_model_plots(model, X_test_eval, y_test, optimal_thr)

    print("[DONE] v1-style training completed inside v2.")


if __name__ == "__main__":
    main()