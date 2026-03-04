import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

import joblib
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve

def reason_card(title, bucket, impact, body):
    st.markdown(
        f"""
        <div style="padding:16px;border-radius:14px;background:rgba(255,255,255,0.04);
                    border:1px solid rgba(255,255,255,0.08);">
            <div style="font-size:12px;opacity:0.8;margin-bottom:6px;">{bucket}</div>
            <div style="font-size:18px;font-weight:700;margin-bottom:8px;">{title}</div>
            <div style="font-size:13px;opacity:0.9;margin-bottom:10px;">{body}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
def kpi_badge(text, color="#22c55e", icon="↑"):
    return f"""
    <span style="
        display:inline-block;
        padding:6px 10px;
        border-radius:999px;
        background:rgba(34,197,94,0.15);
        color:{color};
        font-weight:700;
        font-size:12px;
        line-height:1;
    ">{icon} {text}</span>
    """
    # impact bar
    st.progress(min(max(float(abs(impact))/2.0, 0.0), 1.0))
    st.caption(f"Impact score: {impact:.3f}")

# -----------------------------
# Paths
# -----------------------------
APP_DIR = Path(__file__).resolve().parent
V2_ROOT = APP_DIR.parent

MODELS_DIR = V2_ROOT / "models"
REPORTS_DIR = V2_ROOT / "artifacts" / "reports"
FIGURES_DIR = V2_ROOT / "artifacts" / "figures"

PIPE_PATH = MODELS_DIR / "churn_pipeline.joblib"
XTEST_PATH = MODELS_DIR / "X_test.csv"
YTEST_PATH = MODELS_DIR / "y_test.csv"

RET_TABLE_V2_PATH = REPORTS_DIR / "retention_table_v2.csv"
REASON_CODES_PATH = REPORTS_DIR / "reason_codes_test.csv"
GLOBAL_SHAP_PATH = REPORTS_DIR / "global_shap_importance.csv"
MODEL_METRICS_PATH = REPORTS_DIR / "model_metrics.json"
MODEL_REPORT_PATH = REPORTS_DIR / "model_performance_report.txt"

# -----------------------------
# Page
# -----------------------------
st.set_page_config(page_title="Telco Retention Command Center (Merged)", layout="wide")
st.title("📡 Telco Retention Command Center (Merged v1 + v2)")

# -----------------------------
# Load resources
# -----------------------------
@st.cache_resource
def load_pipeline():
    if PIPE_PATH.exists():
        return joblib.load(PIPE_PATH)

    st.warning("Model not found. Training churn model on server (one-time)...")

    # train via your existing v1-style training script inside v2
    import subprocess, sys

    cmd = [sys.executable, "-m", "v2_upgrade.scripts.02_train_churn_model_v1style"]
    res = subprocess.run(cmd, capture_output=True, text=True)

    if res.returncode != 0:
        st.error("Training failed on server.")
        st.code(res.stdout)
        st.code(res.stderr)
        st.stop()

    if not PIPE_PATH.exists():
        st.error("Training ran but model still not found.")
        st.code(res.stdout)
        st.stop()

    return joblib.load(PIPE_PATH)

@st.cache_data
def load_test_sets():
    if not XTEST_PATH.exists() or not YTEST_PATH.exists():
        st.error("Missing X_test.csv or y_test.csv. Run training first.")
        st.stop()
    X = pd.read_csv(XTEST_PATH)
    y = pd.read_csv(YTEST_PATH).values.ravel()
    return X, y

@st.cache_data
def load_retention_table():
    if RET_TABLE_V2_PATH.exists():
        return pd.read_csv(RET_TABLE_V2_PATH)

    st.warning("retention_table_v2.csv not found. Building v2 retention table on server (one-time)...")

    import subprocess, sys

    cmd = [sys.executable, "-m", "v2_upgrade.scripts.06_build_retention_table_v2"]
    res = subprocess.run(cmd, capture_output=True, text=True)

    if res.returncode != 0:
        st.error("Failed to build retention_table_v2.csv on server.")
        st.code(res.stdout)
        st.code(res.stderr)
        st.stop()

    if not RET_TABLE_V2_PATH.exists():
        st.error("Build script ran but retention_table_v2.csv still not found.")
        st.code(res.stdout)
        st.stop()

    return pd.read_csv(RET_TABLE_V2_PATH)

@st.cache_data
def load_reason_codes_if_any():
    if REASON_CODES_PATH.exists():
        return pd.read_csv(REASON_CODES_PATH)
    return None

@st.cache_data
def load_global_shap_if_any():
    if GLOBAL_SHAP_PATH.exists():
        return pd.read_csv(GLOBAL_SHAP_PATH)
    return None

pipe = load_pipeline()
X_test, y_test = load_test_sets()
ret_table = load_retention_table()
rc_df = load_reason_codes_if_any()
global_shap_df = load_global_shap_if_any()

# -----------------------------
# Sidebar business constraints (v1 style)
# -----------------------------
st.sidebar.header("⚙️ Business Constraints (v1 style)")
COST_FN = st.sidebar.number_input("Cost of Lost Customer (LTV $)", min_value=50, max_value=5000, value=200, step=50)
COST_FP = st.sidebar.number_input("Cost of Retention Offer ($)", min_value=0, max_value=500, value=50, step=5)
USE_FIXED_LTV = st.sidebar.checkbox("Use fixed LTV (override CLV)", value=False)
CALLS_PER_WEEK = st.sidebar.number_input("High-touch capacity (calls/week)", min_value=0, max_value=5000, value=200, step=25)
USE_RISK_WINDOW = st.sidebar.selectbox("Routing risk window", ["risk_30d", "risk_60d", "risk_90d"], index=0)
st.sidebar.divider()

min_value_segment = st.sidebar.selectbox("Minimum value segment", ["all", "mid", "high"], index=1)
only_intervene = st.sidebar.checkbox("Only show customers to intervene", value=True)

st.sidebar.info("These controls drive the v1-style threshold optimization + the v2 intervention list.")

# -----------------------------
# Global predictions
# -----------------------------
X_test_noid = X_test.drop(columns=["customerID"], errors="ignore")

# Cache only the DATA (X), not the model object (pipe).
@st.cache_data
def predict_probs(_pipeline, X):
    return _pipeline.predict_proba(X)[:, 1]

y_probs = predict_probs(pipe, X_test_noid)

# -----------------------------
# v1-style threshold optimization (cost curve)
# -----------------------------
thresholds = np.linspace(0, 1, 101)
costs = []

for t in thresholds:
    y_pred_t = (y_probs >= t).astype(int)
    cm = confusion_matrix(y_test, y_pred_t)
    if cm.shape == (2, 2):
        TN, FP, FN, TP = cm.ravel()
    else:
        TN, FP, FN, TP = 0, 0, 0, 0
    total_cost = (FN * COST_FN) + (FP * COST_FP)
    costs.append(total_cost)

min_cost_idx = int(np.argmin(costs))
best_thr = float(thresholds[min_cost_idx])
min_cost = float(costs[min_cost_idx])

# -----------------------------
# Build v1-style results_df (segmentation + actions)
# -----------------------------
results_df = X_test.copy()
results_df["Churn_Prob"] = y_probs
# -----------------------------
# Option 1 routing: capacity-based Critical, cost-optimal Target Zone
# -----------------------------
# Normalize IDs (prevents subtle mismatches)
if "customerID" in results_df.columns:
    results_df["customerID"] = results_df["customerID"].astype(str).str.strip()

# Economic cutoff: intervene vs no action
eligible = results_df["Churn_Prob"] >= best_thr

# Defaults
results_df["Segment"] = "🟢 Safe"
results_df["Action"] = "No Action"

# Target Zone = above economic cutoff
results_df.loc[eligible, "Segment"] = "🟡 Target Zone"
results_df.loc[eligible, "Action"] = "Send Discount"

# Critical = top N among eligible (capacity-based)
n_crit = int(CALLS_PER_WEEK)
if n_crit > 0 and "customerID" in results_df.columns:
    crit_ids = (
        results_df.loc[eligible]
        .sort_values("Churn_Prob", ascending=False)
        .head(n_crit)["customerID"]
        .tolist()
    )
    mask_crit = results_df["customerID"].isin(crit_ids)
    results_df.loc[mask_crit, "Segment"] = "🔴 Critical Risk"
    results_df.loc[mask_crit, "Action"] = "Call / Win-Back"

results_df["Actual_Churn"] = y_test

'''
def assign_segment(p):
    if p >= 0.85:
        return "🔴 Critical Risk", "Call / Win-Back"
    elif p >= best_thr:
        return "🟡 Target Zone", "Send Discount"
    else:
        return "🟢 Safe", "No Action"

results_df["Segment"], results_df["Action"] = zip(*results_df["Churn_Prob"].apply(assign_segment))
'''
def assign_segment(row):
    # we will set Segment later after we compute critical set
    return "🟢 Safe"
# -----------------------------
# Merge v2 retention table onto test customers
# -----------------------------
# retention_table_v2 has customerID + risk_30d/60d/90d + clv + expected loss etc.
# We'll left join so UI doesn't break if any mismatch
if "customerID" in results_df.columns and "customerID" in ret_table.columns:
    merged = results_df.merge(ret_table, on="customerID", how="left", suffixes=("", "_v2"))
else:
    merged = results_df.copy()

# -----------------------------
# LIVE policy recompute (v2) based on sidebar business constraints
# value_base = (fixed LTV) OR (customer CLV)
# expected_loss_30d_live = risk_30d * value_base
# should_intervene_live = expected_loss_30d_live > COST_FP
# roi_proxy_live = expected_loss_30d_live - COST_FP
# -----------------------------
if "risk_30d" in merged.columns:
    risk30 = merged["risk_30d"].fillna(0)

    if USE_FIXED_LTV:
        value_base = float(COST_FN)
        merged["value_basis_live"] = "Fixed LTV"
        merged["value_base_live"] = value_base
        merged["expected_loss_30d_live"] = risk30 * value_base
    else:
        clv = merged["clv"].fillna(0) if "clv" in merged.columns else 0
        merged["value_basis_live"] = "CLV proxy"
        merged["value_base_live"] = clv
        merged["expected_loss_30d_live"] = risk30 * clv

    merged["offer_cost_live"] = float(COST_FP)
    merged["should_intervene_live"] = (merged["expected_loss_30d_live"] > merged["offer_cost_live"]).astype(int)
    merged["roi_proxy_live"] = merged["expected_loss_30d_live"] - merged["offer_cost_live"]
else:
    merged["value_basis_live"] = "n/a"
    merged["value_base_live"] = np.nan
    merged["expected_loss_30d_live"] = np.nan
    merged["offer_cost_live"] = float(COST_FP)
    merged["should_intervene_live"] = 0
    merged["roi_proxy_live"] = np.nan

# Value segment filter (based on CLV tertiles inside retention table)
if "clv" in merged.columns:
    q1 = merged["clv"].quantile(0.33)
    q2 = merged["clv"].quantile(0.66)

    def value_bucket(x):
        if pd.isna(x):
            return "unknown"
        if x >= q2:
            return "high"
        if x >= q1:
            return "mid"
        return "low"

    merged["value_segment"] = merged["clv"].apply(value_bucket)
else:
    merged["value_segment"] = "unknown"

# Apply sidebar filters
if min_value_segment != "all":
    merged = merged[merged["value_segment"].isin([min_value_segment, "high"] if min_value_segment == "mid" else ["high"])]

if only_intervene and "should_intervene_live" in merged.columns:
    merged = merged[merged["should_intervene_live"] == 1]

# -----------------------------
# Tabs (keep v1 feel, add v2 power)
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🚀 Strategy & Operations (v1)",
    "🧠 Intervention Engine (v2)",
    "🔍 Drivers & Explainability",
    "🤖 Model Lab & EDA"
])

# =========================================================
# TAB 1: Strategy & Operations (v1)
# =========================================================
with tab1:
    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Optimal Risk Threshold", f"{best_thr:.1%}")
        st.markdown(kpi_badge("Dynamic Cutoff"), unsafe_allow_html=True)

    with col2:
        st.metric("Projected Total Cost", f"${min_cost:,.0f}")
        st.markdown(kpi_badge("Minimized"), unsafe_allow_html=True)

    with col3:
        churn_count = int(np.sum(y_test))
        do_nothing_cost = churn_count * COST_FN
        savings = float(do_nothing_cost - min_cost)
        st.metric("Estimated Savings", f"${savings:,.0f}")
        st.markdown(kpi_badge("vs. No Action"), unsafe_allow_html=True)
        
    with col4:
        if "MonthlyCharges" in results_df.columns:
            at_risk_revenue = float(results_df[results_df["Churn_Prob"] >= best_thr]["MonthlyCharges"].sum())
            st.metric("Monthly Revenue at Risk", f"${at_risk_revenue:,.0f}")
            st.markdown(kpi_badge("Targeted Group"), unsafe_allow_html=True)
        else:
            st.metric("Monthly Revenue at Risk", "N/A")

    st.divider()

    # Cost curve + segmentation pie
    st.subheader("📉 Financial Optimization (v1)")
    col_chart, col_seg = st.columns([2, 1])

    with col_chart:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(thresholds, costs, linewidth=2)
        ax.axvline(best_thr, linestyle="--", label=f"Optimal: {best_thr:.2f}")
        ax.set_xlabel("Decision Threshold")
        ax.set_ylabel("Cost ($)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    with col_seg:
        st.write("**Customer Segmentation**")
        seg_counts = results_df["Segment"].value_counts()
        seg_df = seg_counts.reset_index()
        seg_df.columns = ["Segment", "Count"]

        color_map = {
            "🟢 Safe": "#22c55e",          # green
            "🟡 Target Zone": "#facc15",   # yellow
            "🔴 Critical Risk": "#ef4444", # red
        }

        fig_pie = px.pie(
            seg_df,
            names="Segment",
            values="Count",
            hole=0.4,
            color="Segment",
            color_discrete_map=color_map
        )
        fig_pie.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0), height=300)
        st.plotly_chart(fig_pie, use_container_width=True)
        st.caption(f"Critical = top {CALLS_PER_WEEK} customers above cutoff (capacity-based). Target Zone = above economic cutoff, excluding Critical.")

    st.divider()

    # Actionable list (v1)
    st.subheader("📋 Actionable Customer List (v1 view)")
    c1, c2 = st.columns(2)
    with c1:
        filter_segment = st.multiselect(
            "Filter by Segment:",
            ["🔴 Critical Risk", "🟡 Target Zone", "🟢 Safe"],
            default=["🔴 Critical Risk", "🟡 Target Zone", "🟢 Safe"]
        )
    with c2:
        if "Contract" in results_df.columns:
            filter_contract = st.multiselect(
                "Filter by Contract:",
                results_df["Contract"].unique().tolist(),
                default=results_df["Contract"].unique().tolist()
            )
        else:
            filter_contract = []

    filtered_df = results_df[results_df["Segment"].isin(filter_segment)].copy()
    if filter_contract and "Contract" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Contract"].isin(filter_contract)]

    filtered_df = filtered_df.sort_values("Churn_Prob", ascending=False)

    show_cols = [c for c in ["customerID", "Segment", "Churn_Prob", "Action", "tenure", "MonthlyCharges", "Contract", "PaymentMethod"] if c in filtered_df.columns]
    st.dataframe(filtered_df[show_cols], use_container_width=True)

    st.subheader("📥 Export (v1)")
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            "Download Filtered List (CSV)",
            data=filtered_df.to_csv(index=False).encode("utf-8"),
            file_name="filtered_churn_risk_v1.csv",
            mime="text/csv",
        )
    with col_dl2:
        st.download_button(
            "Download Full Scored Test Set (CSV)",
            data=results_df.to_csv(index=False).encode("utf-8"),
            file_name="full_scored_testset_v1.csv",
            mime="text/csv",
        )

# =========================================================
# TAB 2: Intervention Engine (v2)
# =========================================================
with tab2:
    st.subheader("🧠 Retention Prioritization (v2)")
    st.caption("This is the upgrade: churn timing + CLV + expected loss + ROI proxy + recommended interventions + reason codes.")
        # -----------------------------
    # Capacity routing (Option 1) for v2:
    # Critical = top CALLS_PER_WEEK among should_intervene_live by selected risk window
    # Target Zone = should_intervene_live but not in Critical
    # Safe = everyone else
    # -----------------------------
    df_v2 = merged.copy()

    if "customerID" in df_v2.columns:
        df_v2["customerID"] = df_v2["customerID"].astype(str).str.strip()

    # pick routing score from selected window
    if USE_RISK_WINDOW in df_v2.columns:
        df_v2["routing_score_live"] = df_v2[USE_RISK_WINDOW].fillna(0)
    else:
        # fallback if something missing
        df_v2["routing_score_live"] = df_v2.get("risk_30d", 0).fillna(0) if "risk_30d" in df_v2.columns else 0

    # initialize tiers
    df_v2["routing_tier_live"] = "🟢 Safe"

    if "should_intervene_live" in df_v2.columns:
        eligible = df_v2["should_intervene_live"] == 1
        df_v2.loc[eligible, "routing_tier_live"] = "🟡 Target Zone"

        n_crit = int(CALLS_PER_WEEK)
        if n_crit > 0 and eligible.any() and "customerID" in df_v2.columns:
            crit_ids = (
                df_v2.loc[eligible]
                .sort_values("routing_score_live", ascending=False)
                .head(n_crit)["customerID"]
                .tolist()
            )
            df_v2.loc[df_v2["customerID"].isin(crit_ids), "routing_tier_live"] = "🔴 Critical Risk"

    if "should_intervene_live" in df_v2.columns:
        # KPIs based on merged view (respecting sidebar filters)
        k1, k2, k3, k4 = st.columns(4)

        target_n = int(df_v2[df_v2["should_intervene_live"] == 1].shape[0])
        expected_loss = float(df_v2["expected_loss_30d_live"].sum())
        offer_cost = float(df_v2["offer_cost_live"].sum())
        roi_proxy = float(df_v2["roi_proxy_live"].sum())

        crit_n = int((df_v2["routing_tier_live"] == "🔴 Critical Risk").sum())
        tz_n = int((df_v2["routing_tier_live"] == "🟡 Target Zone").sum())
        st.caption(f"Critical queue: {crit_n} | Target campaigns: {tz_n} | Window: {USE_RISK_WINDOW}")

        k1.metric("Target Customers", f"{target_n}")
        k2.metric("Expected Loss (30d)", f"${expected_loss:,.0f}")
        k3.metric("Offer Cost", f"${offer_cost:,.0f}")
        k4.metric("ROI Proxy", f"${roi_proxy:,.0f}")

    st.divider()

    st.write("### Prioritized Action List (v2)")
    if "roi_proxy_live" in merged.columns:
        sort_col = "roi_proxy_live"
    elif "roi_proxy" in merged.columns:
        sort_col = "roi_proxy"
    else:
        sort_col = "Churn_Prob"

    merged_sorted = df_v2.sort_values(sort_col, ascending=False)

    cols_v2 = [
    "routing_tier_live","customerID", "value_segment",
    "risk_30d", "risk_60d", "risk_90d",
    "Churn_Prob", "churn_prob",
    "clv",
    "value_basis_live", "value_base_live",
    "expected_loss_30d_live", "offer_cost_live", "roi_proxy_live", "should_intervene_live",
    "intervention_type", "recommended_action",
    "reason_1", "bucket_1", "impact_1",
    "reason_2", "bucket_2", "impact_2",
    "reason_3", "bucket_3", "impact_3"
]
    cols_v2 = [c for c in cols_v2 if c in merged_sorted.columns]

    st.dataframe(merged_sorted[cols_v2].head(500), use_container_width=True)
    st.divider()
    st.write("### Queues")

    q1, q2 = st.columns(2)

    with q1:
        st.write("**🔴 Critical Call Queue**")
        crit_df = df_v2[df_v2["routing_tier_live"] == "🔴 Critical Risk"].sort_values("routing_score_live", ascending=False)
        st.dataframe(crit_df[cols_v2].head(200), use_container_width=True)

    with q2:
        st.write("**🟡 Target Campaign Queue**")
        tz_df = df_v2[df_v2["routing_tier_live"] == "🟡 Target Zone"].sort_values("routing_score_live", ascending=False)
        st.dataframe(tz_df[cols_v2].head(200), use_container_width=True)
    st.write("### Customer Drilldown (Case File)")

    # Always drilldown from the same table we’re displaying
    df = df_v2.copy()

    if "customerID" not in df.columns:
        st.info("No customerID in retention table; drilldown disabled.")
        st.stop()

    # Pick from the currently filtered table so it never mismatches
    selected_id = st.selectbox("Pick a customerID", df["customerID"].head(500).tolist())
    selected_id = str(selected_id).strip()
    df["customerID"] = df["customerID"].astype(str).str.strip()

    match = df[df["customerID"] == selected_id]
    if match.empty:
        st.error(f"No row found for customerID={selected_id}.")
        st.write("Debug:")
        st.write("Rows in df:", len(df))
        st.write("Sample customerIDs:", df["customerID"].head(10).tolist())
        st.stop()

    row = match.iloc[0]

    st.subheader(f"Case File: {row['customerID']}")

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    prob = row.get("Churn_Prob", row.get("churn_prob", 0))
    k1.metric("Churn Probability", f"{float(prob):.1%}")
    k2.metric("30d Risk", f"{float(row.get('risk_30d',0)):.1%}")
    k3.metric("60d Risk", f"{float(row.get('risk_60d',0)):.1%}")
    k4.metric("90d Risk", f"{float(row.get('risk_90d',0)):.1%}")

    k5, k6, k7 = st.columns(3)
    k5.metric("CLV (proxy)", f"${float(row.get('clv',0)):,.0f}")
    #k6.metric("Expected Loss (30d)", f"${float(row.get('expected_loss_30d_live', row.get('expected_loss_30d',0))):,.0f}")
    k6.metric("Expected Loss (30d)", f"${float(row.get('expected_loss_30d_live',0)):,.0f}")
    #k7.metric("Offer Cost", f"${float(row.get('offer_cost_live', row.get('offer_cost',0))):,.0f}")
    k7.metric("Offer Cost", f"${float(row.get('offer_cost_live',0)):,.0f}")
    st.caption(f"Value basis: {row.get('value_basis_live','')}  |  Value used: ${float(row.get('value_base_live',0)):,.0f}")

    st.divider()

    # Recommendation banner
    st.markdown("### Recommended Intervention")
    st.info(
    f"**{row.get('intervention_type','')}**\n"
    f"**Action:** {row.get('recommended_action','')}\n\n"
    f"**ROI proxy (live):** ${float(row.get('roi_proxy_live', row.get('roi_proxy',0))):,.0f}  \n"
    f"**Should intervene (live):** {'Yes' if int(row.get('should_intervene_live', row.get('should_intervene',0)))==1 else 'No'}"
)

    st.divider()

    # Reason codes -> cards
    st.markdown("### Why this customer is at risk")
    c1, c2, c3 = st.columns(3)

    r1 = str(row.get("reason_1",""))
    b1 = str(row.get("bucket_1",""))
    i1 = float(row.get("impact_1",0))

    r2 = str(row.get("reason_2",""))
    b2 = str(row.get("bucket_2",""))
    i2 = float(row.get("impact_2",0))

    r3 = str(row.get("reason_3",""))
    b3 = str(row.get("bucket_3",""))
    i3 = float(row.get("impact_3",0))

    with c1:
        reason_card(r1, b1, i1, "Primary driver contributing to churn risk.")
    with c2:
        reason_card(r2, b2, i2, "Secondary driver contributing to churn risk.")
    with c3:
        reason_card(r3, b3, i3, "Third driver contributing to churn risk.")
    
    st.divider() 

    st.write("### Bucket mix (top drivers)")
    if "bucket_1" in merged.columns:
        buckets = pd.concat([
            merged["bucket_1"].dropna(),
            merged["bucket_2"].dropna() if "bucket_2" in merged.columns else pd.Series([], dtype=str),
            merged["bucket_3"].dropna() if "bucket_3" in merged.columns else pd.Series([], dtype=str)
        ])
        bucket_counts = buckets.value_counts().reset_index()
        bucket_counts.columns = ["bucket", "count"]
        fig_b = px.bar(bucket_counts, x="bucket", y="count")
        st.plotly_chart(fig_b, use_container_width=True)

    st.write("### Top reason codes (frequency)")
    if "reason_1" in merged.columns:
        reasons = pd.concat([
            merged["reason_1"].dropna(),
            merged["reason_2"].dropna() if "reason_2" in merged.columns else pd.Series([], dtype=str),
            merged["reason_3"].dropna() if "reason_3" in merged.columns else pd.Series([], dtype=str)
        ])
        reason_counts = reasons.value_counts().head(15).reset_index()
        reason_counts.columns = ["reason", "count"]
        fig_r = px.bar(reason_counts, x="count", y="reason", orientation="h")
        st.plotly_chart(fig_r, use_container_width=True)

# optional: keep the raw table behind an expander
    with st.expander("Show raw row (debug)"):
        st.dataframe(row.to_frame().T, width="stretch")

        st.download_button(
            "Download retention_table_v2 (CSV)",
            data=ret_table.to_csv(index=False).encode("utf-8"),
            file_name="retention_table_v2.csv",
            mime="text/csv",
        )

# =========================================================
# TAB 3: Drivers & Explainability
# =========================================================
with tab3:
    st.subheader("🔍 Drivers & Explainability")

    c1, c2 = st.columns(2)
    with c1:
        st.write("### Global Drivers (from SHAP importances)")
        if global_shap_df is not None and global_shap_df.shape[0] > 0:
            topn = st.slider("Top N features", 5, 30, 15)
            df_top = global_shap_df.head(topn).copy()
            fig = px.bar(df_top.iloc[::-1], x="mean_abs_shap", y="feature", orientation="h")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("global_shap_importance.csv not found. Run: python -m v2_upgrade.scripts.08_global_shap")

    with c2:
        st.write("### Reason Codes Preview (test set)")
        if rc_df is not None:
            st.dataframe(rc_df.head(50), use_container_width=True)
        else:
            st.info("reason_codes_test.csv not found. Run: python -m v2_upgrade.scripts.03_reason_codes")

    st.divider()

    
    

# =========================================================
# TAB 4: Model Lab & EDA (v1 feel)
# =========================================================
with tab4:
    st.subheader("🤖 Model Lab & EDA (v1 feel)")

    # Live metrics at current best_thr
    y_pred_current = (y_probs >= best_thr).astype(int)
    report = classification_report(y_test, y_pred_current, output_dict=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{report['accuracy']:.2%}")
    m2.metric("Precision (Churn)", f"{report['1']['precision']:.2%}")
    m3.metric("Recall (Churn)", f"{report['1']['recall']:.2%}")
    m4.metric("F1-Score", f"{report['1']['f1-score']:.2f}")

    st.divider()

    st.write("### Dynamic Diagnostics (live)")
    col_dyn1, col_dyn2 = st.columns(2)

    with col_dyn1:
        st.write("**Confusion Matrix (live)**")
        cm = confusion_matrix(y_test, y_pred_current)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm, cbar=False)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        ax_cm.set_title(f"Threshold = {best_thr:.2f}")
        st.pyplot(fig_cm)

    with col_dyn2:
        st.write("**Precision vs Recall (live)**")
        precisions, recalls, thr_curve = precision_recall_curve(y_test, y_probs)
        fig_thr, ax_thr = plt.subplots()
        ax_thr.plot(thr_curve, precisions[:-1], label="Precision")
        ax_thr.plot(thr_curve, recalls[:-1], label="Recall")
        ax_thr.axvline(best_thr, linestyle="--", label="Current Thr")
        ax_thr.set_xlabel("Threshold")
        ax_thr.legend()
        ax_thr.grid(True, alpha=0.3)
        st.pyplot(fig_thr)

    st.divider()

    st.write("### Static Artifacts (from training)")
    if MODEL_REPORT_PATH.exists():
        with open(MODEL_REPORT_PATH, "r") as f:
            st.code(f.read(), language="text")

    # show key plots if they exist
    key_plots = [
        "roc_curve.png",
        "calibration_curve.png",
        "probability_distribution.png",
        "actual_vs_predicted_decile_plot.png",
        "confusion_matrix_optimized.png",
        "heatmap_correlation.png",
    ]

    cols = st.columns(2)
    idx = 0
    for kp in key_plots:
        p = FIGURES_DIR / kp
        if p.exists():
            cols[idx % 2].image(str(p), caption=kp, use_container_width=True)
            idx += 1

    st.divider()

    st.write("### Full EDA Gallery")
    if FIGURES_DIR.exists():
        imgs = sorted([x for x in FIGURES_DIR.iterdir() if x.suffix.lower() == ".png"])
        if len(imgs) == 0:
            st.info("No figures found.")
        else:
            gcols = st.columns(2)
            for i, img in enumerate(imgs):
                gcols[i % 2].image(str(img), caption=img.name, use_container_width=True)
    else:
        st.info(f"Figures directory not found: {FIGURES_DIR}")
