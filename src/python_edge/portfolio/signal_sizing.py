from __future__ import annotations

import pandas as pd



def attach_conviction_bucket(
    df: pd.DataFrame,
    score_col: str = "score",
    bucket_col: str = "conviction_bucket",
) -> pd.DataFrame:
    out = df.copy()
    required = ["date", score_col]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise RuntimeError(f"attach_conviction_bucket: missing columns: {missing}")

    out["rank_pct_for_sizing"] = out.groupby("date")[score_col].rank(method="average", pct=True)
    bucket = pd.Series("flat", index=out.index, dtype="object")

    bucket.loc[out["rank_pct_for_sizing"] >= 0.98] = "long_max"
    bucket.loc[(out["rank_pct_for_sizing"] >= 0.94) & (out["rank_pct_for_sizing"] < 0.98)] = "long_high"
    bucket.loc[(out["rank_pct_for_sizing"] >= 0.88) & (out["rank_pct_for_sizing"] < 0.94)] = "long_mid"

    bucket.loc[out["rank_pct_for_sizing"] <= 0.02] = "short_max"
    bucket.loc[(out["rank_pct_for_sizing"] > 0.02) & (out["rank_pct_for_sizing"] <= 0.06)] = "short_high"
    bucket.loc[(out["rank_pct_for_sizing"] > 0.06) & (out["rank_pct_for_sizing"] <= 0.12)] = "short_mid"

    out[bucket_col] = bucket
    return out



def apply_signal_strength_sizing(
    df: pd.DataFrame,
    side_col: str = "side",
    score_col: str = "score",
    out_col: str = "side_sized",
) -> pd.DataFrame:
    out = attach_conviction_bucket(df, score_col=score_col)

    if side_col not in out.columns:
        raise RuntimeError(f"apply_signal_strength_sizing: missing side_col={side_col}")

    side = pd.to_numeric(out[side_col], errors="coerce").fillna(0.0)
    mult = pd.Series(0.0, index=out.index, dtype="float64")

    mult.loc[out["conviction_bucket"] == "long_max"] = 1.40
    mult.loc[out["conviction_bucket"] == "long_high"] = 1.15
    mult.loc[out["conviction_bucket"] == "long_mid"] = 0.85

    mult.loc[out["conviction_bucket"] == "short_max"] = 1.40
    mult.loc[out["conviction_bucket"] == "short_high"] = 1.15
    mult.loc[out["conviction_bucket"] == "short_mid"] = 0.85

    out[out_col] = side * mult
    out["conviction_mult"] = mult
    return out