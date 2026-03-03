import json
import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path

REPORTS_DIR = Path("v2_upgrade/artifacts/reports")
CONFIG_PATH = Path("v2_upgrade/config/retention_config.json")

st.set_page_config(page_title="Retention Command Center (v2)", layout="wide")

@st.cache_data
def load_retention_table():
    return pd.read_csv(REPORTS_DIR / "retention_table_v2.csv")

@st.cache_data
def load_global_shap():
    return pd.read_csv(REPORTS_DIR / "global_shap_importance.csv")

def load_config():
    return json.loads(CONFIG_PATH.read_text())

st.title("Telco Retention Command Center (v2)")

cfg = load_config()

st.sidebar.header("Controls")
min_value_seg = st.sidebar.selectbox("Minimum value segment", ["low", "mid", "high"], index=1)
show_only_intervene = st.sidebar.checkbox("Only show customers to intervene", value=True)

df = load_retention_table()
shap_imp = load_global_shap()

seg_rank = {"low": 0, "mid": 1, "high": 2}
df = df[df["value_segment"].map(seg_rank) >= seg_rank[min_value_seg]]

if show_only_intervene:
    df = df[df["should_intervene"] == 1]

st.subheader("Executive KPIs")
c1, c2, c3, c4 = st.columns(4)

total_customers = len(df)
total_value_at_risk = float(df["expected_loss_30d"].sum()) if total_customers else 0.0
total_offer_cost = float(df["offer_cost"].sum()) if total_customers else 0.0
net_roi = float(df["roi_proxy"].sum()) if total_customers else 0.0

c1.metric("Target Customers", f"{total_customers:,}")
c2.metric("Expected Loss (30d)", f"${total_value_at_risk:,.0f}")
c3.metric("Offer Cost", f"${total_offer_cost:,.0f}")
c4.metric("ROI Proxy", f"${net_roi:,.0f}")

st.divider()

tab1, tab2, tab3 = st.tabs(["Prioritization", "Drivers", "Customer Drilldown"])

with tab1:
    st.subheader("Prioritized Action List")
    cols = [
        "customerID", "value_segment",
        "risk_30d", "risk_60d", "risk_90d",
        "clv", "expected_loss_30d", "offer_cost", "roi_proxy",
        "intervention_type",
        "reason_1", "bucket_1", "reason_2", "bucket_2",
        "recommended_action"
    ]

    st.dataframe(
        df[cols].head(300),
        use_container_width=True
    )

    st.subheader("Risk vs Value Scatter")
    fig = px.scatter(
        df.head(2000),
        x="risk_30d",
        y="clv",
        size="expected_loss_30d",
        hover_data=["customerID", "intervention_type", "reason_1", "recommended_action"],
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Global churn drivers (mean |SHAP|)")
    topn = st.slider("Top N features", 10, 50, 20)
    top = shap_imp.head(topn).copy()
    fig2 = px.bar(top[::-1], x="mean_abs_shap", y="feature")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Intervention mix")
    mix = df["intervention_type"].value_counts().reset_index()
    mix.columns = ["intervention_type", "count"]
    fig3 = px.pie(mix, names="intervention_type", values="count", hole=0.4)
    st.plotly_chart(fig3, use_container_width=True)

with tab3:
    st.subheader("Customer drilldown")
    if len(df) == 0:
        st.info("No customers in the current filter.")
    else:
        cid = st.selectbox("Select customerID", df["customerID"].head(500).tolist())
        row = df[df["customerID"] == cid].iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("30d Risk", f"{row['risk_30d']:.2%}")
        c2.metric("CLV", f"${row['clv']:.0f}")
        c3.metric("Expected Loss", f"${row['expected_loss_30d']:.0f}")
        c4.metric("Offer Cost", f"${row['offer_cost']:.0f}")

        st.write("Reasons")
        st.write({
            "reason_1": row["reason_1"],
            "reason_2": row["reason_2"],
            "reason_3": row["reason_3"],
        })

        st.write("Recommended action")
        st.success(str(row["recommended_action"]))

        st.write("Intervention type")
        st.info(str(row["intervention_type"]))
