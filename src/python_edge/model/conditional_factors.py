from __future__ import annotations

import pandas as pd


CONDITIONAL_FEATURE_COLS = [
    "cond_momentum_trend",
    "cond_str_range",
    "cond_overnight_breadth_weak",
    "cond_ivol_lowliq_penalty",
    "cond_vol_compression_trend",
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

    breadth_high = (out["market_breadth"] >= 0.55).astype("float64")
    breadth_low = (out["market_breadth"] <= 0.45).astype("float64")
    intraday_trend = (out["intraday_rs"] > 0).astype("float64")
    liq_low = (out["liq_rank"] <= 0.35).astype("float64")

    out["cond_momentum_trend"] = out["momentum_20d"] * breadth_high * intraday_trend
    out["cond_str_range"] = out["str_3d"] * (1.0 - breadth_high)
    out["cond_overnight_breadth_weak"] = out["overnight_drift_20d"] * breadth_low
    out["cond_ivol_lowliq_penalty"] = (-1.0) * out["ivol_20d"] * liq_low
    out["cond_vol_compression_trend"] = out["vol_compression"] * breadth_high * intraday_trend

    return out