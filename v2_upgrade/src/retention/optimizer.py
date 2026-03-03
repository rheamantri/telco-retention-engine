import pandas as pd

def compute_expected_loss(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for w in [30, 60, 90]:
        out[f"expected_loss_{w}d"] = out[f"risk_{w}d"] * out["clv"]
    return out

def compute_priority_score(df: pd.DataFrame, w30: float, w60: float, w90: float) -> pd.DataFrame:
    out = df.copy()
    out["priority_score"] = (
        w30 * out["risk_30d"] +
        w60 * out["risk_60d"] +
        w90 * out["risk_90d"]
    )
    return out

def choose_intervention(row: pd.Series, costs: dict) -> str:
    """
    Chooses an intervention based on recommended_action and value segment.
    This is rule-based now, later you can swap to uplift model.
    """
    action = str(row.get("recommended_action", "")).lower()
    seg = str(row.get("value_segment", "mid"))

    # Map recommended_action -> internal offer type
    if "contract" in action:
        offer = "contract_incentive"
    elif "autopay" in action:
        offer = "autopay_nudge"
    elif "support" in action:
        offer = "support_intervention"
    elif "service" in action or "quality" in action:
        offer = "service_quality_check"
    else:
        offer = "discount_offer"

    # Value segment gating
    if seg == "low" and offer in ["contract_incentive", "discount_offer"]:
        offer = "support_intervention"

    return offer

def add_intervention_and_roi(df: pd.DataFrame, costs: dict) -> pd.DataFrame:
    out = df.copy()
    out["intervention_type"] = out.apply(lambda r: choose_intervention(r, costs), axis=1)
    out["offer_cost"] = out["intervention_type"].apply(lambda k: float(costs.get(k, 0)))

    # Basic decision: intervene if 30d expected loss > cost
    out["should_intervene"] = (out["expected_loss_30d"] > out["offer_cost"]).astype(int)

    # ROI proxy: saved value minus cost (assuming perfect save of expected loss, conservative would be * save_rate)
    out["roi_proxy"] = out["expected_loss_30d"] - out["offer_cost"]

    return out
