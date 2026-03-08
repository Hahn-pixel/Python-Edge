from __future__ import annotations

import pandas as pd


CONDITIONAL_FEATURE_COLS = [
    "cond_breadth_trend_intraday_rs",
    "cond_breadth_trend_mom_compression",
    "cond_breadth_range_str",
    "cond_breadth_weak_overnight",
    "cond_ivol_lowliq_penalty",
]



def add_conditional_factors(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    required = [
        "momentum_20d",
        "str_3d",
        "overnight_drift_20d",
        "ivol_20d",
        "vol_compression",
        "market_breadth",
        "liq_rank",
        "intraday_rs",
    ]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise RuntimeError(f"add_conditional_factors: missing columns: {missing}")

    breadth_trend = (out["market_breadth"] >= 0.55).astype("float64")
    breadth_range = ((out["market_breadth"] > 0.45) & (out["market_breadth"] < 0.55)).astype("float64")
    breadth_weak = (out["market_breadth"] <= 0.45).astype("float64")
    liq_low = (out["liq_rank"] <= 0.35).astype("float64")

    out["cond_breadth_trend_intraday_rs"] = out["intraday_rs"] * breadth_trend
    out["cond_breadth_trend_mom_compression"] = out["momentum_20d"] * out["vol_compression"] * breadth_trend
    out["cond_breadth_range_str"] = out["str_3d"] * breadth_range
    out["cond_breadth_weak_overnight"] = out["overnight_drift_20d"] * breadth_weak
    out["cond_ivol_lowliq_penalty"] = (-1.0) * out["ivol_20d"] * liq_low

    return out