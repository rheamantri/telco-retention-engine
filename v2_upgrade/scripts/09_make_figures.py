# v2_upgrade/scripts/09_make_figures.py
import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from v2_upgrade.src.config import DATA_PROCESSED, FIGURES_DIR


def save_fig(name: str):
    out = Path(FIGURES_DIR) / name
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"[OK] saved figure: {out}")


def main():
    # load engineered data
    data_path = DATA_PROCESSED / "telco_engineered.csv"
    df = pd.read_csv(data_path)
    print(f"[INFO] loaded: {data_path} shape={df.shape}")

    # ensure churn target exists
    if "Churn_Target" not in df.columns:
        if "Churn" in df.columns:
            df["Churn_Target"] = (df["Churn"].astype(str).str.lower() == "yes").astype(int)
        else:
            raise ValueError("Need Churn_Target or Churn column in processed dataset.")

    # -----------------------------
    # 1) Correlation heatmap
    # -----------------------------
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in ["Churn_Target"]]

    if len(num_cols) >= 2:
        plt.figure(figsize=(10, 8))
        corr = df[num_cols + ["Churn_Target"]].corr(numeric_only=True)
        sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
        plt.title("Correlation Heatmap (Numeric + Churn_Target)")
        save_fig("01_corr_heatmap.png")

    # -----------------------------
    # 2) Churn rate by top categorical features
    # -----------------------------
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    ignore = {"customerID", "Churn"}
    cat_cols = [c for c in cat_cols if c not in ignore]

    # pick up to 8 categorical columns to plot
    for col in cat_cols[:8]:
        try:
            # churn rate by category
            rate = (
                df.groupby(col)["Churn_Target"]
                .mean()
                .sort_values(ascending=False)
                .reset_index()
            )

            # limit to top 12 categories for readability
            if len(rate) > 12:
                rate = rate.head(12)

            plt.figure(figsize=(9, 4.5))
            sns.barplot(data=rate, x=col, y="Churn_Target")
            plt.xticks(rotation=25, ha="right")
            plt.ylabel("Churn Rate")
            plt.title(f"Churn Rate by {col}")
            save_fig(f"02_churn_rate_by_{col}.png")
        except Exception as e:
            print(f"[WARN] skipped {col}: {e}")

    # -----------------------------
    # 3) Distribution plots vs churn
    # -----------------------------
    df_plot = df.copy()
    df_plot["Churn_Label"] = df_plot["Churn_Target"].map({0: "No", 1: "Yes"})

    candidates = []
    for c in ["tenure", "MonthlyCharges", "TotalCharges", "Avg_Historical_Charge"]:
        if c in df_plot.columns:
            candidates.append(c)

    for col in candidates:
        try:
            plt.figure(figsize=(8, 4.5))
            sns.violinplot(data=df_plot, x="Churn_Label", y=col)
            plt.title(f"Distribution of {col} by Churn")
            save_fig(f"03_violin_{col}_by_churn.png")
        except Exception as e:
            print(f"[WARN] violin failed for {col}: {e}")

    # -----------------------------
    # 4) Tenure bucket churn rates (if present)
    # -----------------------------
    if "Tenure_Bucket" in df.columns:
        try:
            rate = (
                df.groupby("Tenure_Bucket")["Churn_Target"]
                .mean()
                .reset_index()
            )
            plt.figure(figsize=(7.5, 4.2))
            sns.barplot(data=rate, x="Tenure_Bucket", y="Churn_Target")
            plt.ylabel("Churn Rate")
            plt.title("Churn Rate by Tenure Bucket")
            save_fig("04_churn_by_tenure_bucket.png")
        except Exception as e:
            print(f"[WARN] tenure bucket plot failed: {e}")

    print("[DONE] figures generated.")


if __name__ == "__main__":
    main()
