from __future__ import annotations

import pandas as pd


CONDITIONAL_FEATURE_COLS = [
    "cond_breadth_trend_intraday_rs",
    "cond_breadth_trend_mom_compression",
    "cond_breadth_range_str",
    "cond_breadth_weak_overnight",
    "cond_ivol_lowliq_penalty",
    "cond_momentum_liq_trend",
    "cond_str_weak_breadth",
    "cond_overnight_trend_follow",
    "cond_vol_compression_liq_breakout",
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
        "intraday_pressure",
        "volume_shock",
    ]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise RuntimeError(f"add_conditional_factors: missing columns: {missing}")

    breadth_trend = (out["market_breadth"] >= 0.55).astype("float64")
    breadth_range = ((out["market_breadth"] > 0.45) & (out["market_breadth"] < 0.55)).astype("float64")
    breadth_weak = (out["market_breadth"] <= 0.45).astype("float64")
    liq_low = (out["liq_rank"] <= 0.35).astype("float64")
    liq_high = (out["liq_rank"] >= 0.65).astype("float64")
    intraday_trend = (out["intraday_rs"] > 0).astype("float64")
    pressure_pos = (out["intraday_pressure"] > 0.55).astype("float64")
    vol_expansion = (out["volume_shock"] >= 1.05).astype("float64")

    out["cond_breadth_trend_intraday_rs"] = out["intraday_rs"] * breadth_trend
    out["cond_breadth_trend_mom_compression"] = out["momentum_20d"] * out["vol_compression"] * breadth_trend
    out["cond_breadth_range_str"] = out["str_3d"] * breadth_range
    out["cond_breadth_weak_overnight"] = out["overnight_drift_20d"] * breadth_weak
    out["cond_ivol_lowliq_penalty"] = (-1.0) * out["ivol_20d"] * liq_low

    out["cond_momentum_liq_trend"] = out["momentum_20d"] * liq_high * breadth_trend * intraday_trend
    out["cond_str_weak_breadth"] = out["str_3d"] * breadth_weak * (1.0 - intraday_trend)
    out["cond_overnight_trend_follow"] = out["overnight_drift_20d"] * breadth_trend * pressure_pos
    out["cond_vol_compression_liq_breakout"] = out["vol_compression"] * liq_high * vol_expansion * intraday_trend

    return out