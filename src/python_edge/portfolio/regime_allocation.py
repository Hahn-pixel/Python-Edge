from __future__ import annotations

import pandas as pd



def attach_market_regime(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "market_breadth" not in out.columns:
        raise RuntimeError("attach_market_regime: missing market_breadth")

    regime = pd.Series("neutral", index=out.index, dtype="object")
    regime.loc[out["market_breadth"] >= 0.60] = "trend"
    regime.loc[out["market_breadth"] <= 0.45] = "weak"
    out["market_regime"] = regime
    return out



def attach_regime_multipliers(df: pd.DataFrame) -> pd.DataFrame:
    out = attach_market_regime(df)

    long_mult = pd.Series(1.0, index=out.index, dtype="float64")
    short_mult = pd.Series(1.0, index=out.index, dtype="float64")
    top_pct = pd.Series(0.10, index=out.index, dtype="float64")

    trend_mask = out["market_regime"] == "trend"
    weak_mask = out["market_regime"] == "weak"

    long_mult.loc[trend_mask] = 1.20
    short_mult.loc[trend_mask] = 0.80
    top_pct.loc[trend_mask] = 0.12

    long_mult.loc[weak_mask] = 0.85
    short_mult.loc[weak_mask] = 1.15
    top_pct.loc[weak_mask] = 0.08

    out["regime_long_mult"] = long_mult
    out["regime_short_mult"] = short_mult
    out["regime_top_pct"] = top_pct
    return out



def build_regime_aware_long_short_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    out = attach_regime_multipliers(df)

    required = ["date", "score", "regime_top_pct", "regime_long_mult", "regime_short_mult"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise RuntimeError(f"build_regime_aware_long_short_portfolio: missing columns: {missing}")

    out["rank"] = out.groupby("date")["score"].rank(method="average", pct=True)
    out["side"] = 0.0

    for dt, idx in out.groupby("date").groups.items():
        top_pct = float(out.loc[idx, "regime_top_pct"].iloc[0])
        long_mult = float(out.loc[idx, "regime_long_mult"].iloc[0])
        short_mult = float(out.loc[idx, "regime_short_mult"].iloc[0])

        long_mask = out.loc[idx, "rank"] >= (1.0 - top_pct)
        short_mask = out.loc[idx, "rank"] <= top_pct

        long_index = out.loc[idx].index[long_mask]
        short_index = out.loc[idx].index[short_mask]

        out.loc[long_index, "side"] = long_mult
        out.loc[short_index, "side"] = -1.0 * short_mult

    return out