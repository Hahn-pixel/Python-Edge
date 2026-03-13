from __future__ import annotations

import pandas as pd

SIZING_PRESETS: dict[str, dict[str, float]] = {
    "baseline": {
        "long_max": 1.40,
        "long_high": 1.15,
        "long_mid": 0.85,
        "short_max": 1.40,
        "short_high": 1.15,
        "short_mid": 0.85,
    },
    "aggressive": {
        "long_max": 1.60,
        "long_high": 1.20,
        "long_mid": 0.70,
        "short_max": 1.60,
        "short_high": 1.20,
        "short_mid": 0.70,
    },
    "conservative": {
        "long_max": 1.25,
        "long_high": 1.10,
        "long_mid": 0.90,
        "short_max": 1.25,
        "short_high": 1.10,
        "short_mid": 0.90,
    },
    "barbell": {
        "long_max": 1.90,
        "long_high": 1.10,
        "long_mid": 0.55,
        "short_max": 1.90,
        "short_high": 1.10,
        "short_mid": 0.55,
    },
    "residual_stat_arb": {
        "long_max": 2.10,
        "long_high": 1.20,
        "long_mid": 0.45,
        "short_max": 2.10,
        "short_high": 1.20,
        "short_mid": 0.45,
    },
}


def attach_conviction_bucket(
    df: pd.DataFrame,
    score_col: str = "score",
    bucket_col: str = "conviction_bucket",
    strong_abs_rank_col: str = "score_abs_rank_pct",
    fresh_flag_col: str = "fresh_dislocation_flag",
) -> pd.DataFrame:
    out = df.copy()
    required = ["date", score_col]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise RuntimeError(f"attach_conviction_bucket: missing columns: {missing}")

    out["rank_pct_for_sizing"] = out.groupby("date")[score_col].rank(method="average", pct=True)
    abs_rank = pd.to_numeric(out.get(strong_abs_rank_col, 0.5), errors="coerce").fillna(0.5)
    fresh_flag = pd.to_numeric(out.get(fresh_flag_col, 0), errors="coerce").fillna(0).astype(int)

    bucket = pd.Series("flat", index=out.index, dtype="object")

    bucket.loc[out["rank_pct_for_sizing"] >= 0.98] = "long_max"
    bucket.loc[(out["rank_pct_for_sizing"] >= 0.94) & (out["rank_pct_for_sizing"] < 0.98)] = "long_high"
    bucket.loc[(out["rank_pct_for_sizing"] >= 0.88) & (out["rank_pct_for_sizing"] < 0.94)] = "long_mid"

    bucket.loc[out["rank_pct_for_sizing"] <= 0.02] = "short_max"
    bucket.loc[(out["rank_pct_for_sizing"] > 0.02) & (out["rank_pct_for_sizing"] <= 0.06)] = "short_high"
    bucket.loc[(out["rank_pct_for_sizing"] > 0.06) & (out["rank_pct_for_sizing"] <= 0.12)] = "short_mid"

    bucket.loc[(bucket == "long_mid") & ((abs_rank < 0.80) | (fresh_flag == 0))] = "flat"
    bucket.loc[(bucket == "short_mid") & ((abs_rank < 0.80) | (fresh_flag == 0))] = "flat"
    bucket.loc[(bucket == "long_high") & (abs_rank < 0.88)] = "long_mid"
    bucket.loc[(bucket == "short_high") & (abs_rank < 0.88)] = "short_mid"
    bucket.loc[(bucket == "long_max") & (abs_rank < 0.94)] = "long_high"
    bucket.loc[(bucket == "short_max") & (abs_rank < 0.94)] = "short_high"

    out[bucket_col] = bucket
    out["fresh_for_sizing_flag"] = fresh_flag
    out["abs_rank_for_sizing"] = abs_rank
    return out


def apply_signal_strength_sizing(
    df: pd.DataFrame,
    side_col: str = "side",
    score_col: str = "score",
    out_col: str = "side_sized",
    preset_name: str = "residual_stat_arb",
) -> pd.DataFrame:
    out = attach_conviction_bucket(df, score_col=score_col)
    if side_col not in out.columns:
        raise RuntimeError(f"apply_signal_strength_sizing: missing side_col={side_col}")
    if preset_name not in SIZING_PRESETS:
        raise RuntimeError(f"apply_signal_strength_sizing: unknown preset_name={preset_name!r}")

    preset = SIZING_PRESETS[preset_name]
    side = pd.to_numeric(out[side_col], errors="coerce").fillna(0.0)
    mult = pd.Series(0.0, index=out.index, dtype="float64")

    mult.loc[out["conviction_bucket"] == "long_max"] = preset["long_max"]
    mult.loc[out["conviction_bucket"] == "long_high"] = preset["long_high"]
    mult.loc[out["conviction_bucket"] == "long_mid"] = preset["long_mid"]
    mult.loc[out["conviction_bucket"] == "short_max"] = preset["short_max"]
    mult.loc[out["conviction_bucket"] == "short_high"] = preset["short_high"]
    mult.loc[out["conviction_bucket"] == "short_mid"] = preset["short_mid"]

    out[out_col] = side * mult
    out["conviction_mult"] = mult
    out["sizing_preset"] = preset_name
    return out