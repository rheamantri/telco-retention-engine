import json
import pandas as pd
from pathlib import Path

def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def add_clv(df: pd.DataFrame, margin_rate: float, expected_future_months: int) -> pd.DataFrame:
    """
    Simple CLV proxy:
      monthly_margin = MonthlyCharges * margin_rate
      remaining_months = tenure + expected_future_months
      clv = monthly_margin * remaining_months
    """
    out = df.copy()

    # safety
    out["MonthlyCharges"] = pd.to_numeric(out.get("MonthlyCharges", 0), errors="coerce").fillna(0)
    out["tenure"] = pd.to_numeric(out.get("tenure", 0), errors="coerce").fillna(0)

    out["monthly_margin"] = out["MonthlyCharges"] * float(margin_rate)
    out["remaining_months_proxy"] = out["tenure"] + int(expected_future_months)
    out["clv"] = out["monthly_margin"] * out["remaining_months_proxy"]

    return out

def add_value_segment(df: pd.DataFrame, high_value_clv: float, mid_value_clv: float) -> pd.DataFrame:
    out = df.copy()

    def seg(x: float) -> str:
        if x >= high_value_clv:
            return "high"
        if x >= mid_value_clv:
            return "mid"
        return "low"

    out["value_segment"] = out["clv"].apply(lambda x: seg(float(x)))
    return out
