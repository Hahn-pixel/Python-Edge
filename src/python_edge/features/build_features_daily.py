from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureConfig:
    # Percentile features are computed cross-sectionally per date.
    # These are intentionally simple and robust for MVP.
    mom_lookbacks: List[int] = (1, 3, 5)
    rv_lookback: int = 10
    atr_lookback: int = 14
    ema_fast: int = 10
    ema_slow: int = 30


def _safe_pct_rank(s: pd.Series) -> pd.Series:
    # pct=True returns [0,1]; we map to [-1,1] for symmetric scoring convenience.
    r = s.rank(pct=True, method="average")
    return (r * 2.0 - 1.0).astype(float)


def _ema(x: pd.Series, span: int) -> pd.Series:
    return x.ewm(span=span, adjust=False).mean()


def build_features_daily(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Input df requires at least: date, symbol, close.

    Produces:
      - mom_{N}d: N-day return
      - rv_{rv} : realized vol proxy (std of daily returns)
      - atr_pct : ATR proxy using abs returns, scaled
      - ema_slow_slope: slope proxy of slow EMA (1d diff)
      - compression: (rv / atr_pct) style proxy

    And percentile versions (cross-section per date):
      - mom_{N}d__pct, rv_10__pct, atr_pct__pct, ema_slow_slope__pct, compression__pct

    Also adds a small set of derived boolean-ish / interaction features:
      - is_high_mom_5d, is_high_rv10, mom_5d_minus_mom_1d, rel_rv_to_atr

    Note: derived features are numeric; they still participate in mining via quantiles.
    """
    required = {"date", "symbol", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"build_features_daily missing columns: {sorted(missing)}")

    out = df.copy()
    out = out.sort_values(["symbol", "date"]).reset_index(drop=True)

    # daily returns
    out["ret_1d"] = out.groupby("symbol", sort=False)["close"].pct_change()

    # momentum returns
    for n in cfg.mom_lookbacks:
        out[f"mom_{n}d"] = out.groupby("symbol", sort=False)["close"].pct_change(n)

    # realized vol (std of ret_1d)
    out[f"rv_{cfg.rv_lookback}"] = (
        out.groupby("symbol", sort=False)["ret_1d"].rolling(cfg.rv_lookback).std().reset_index(level=0, drop=True)
    )

    # ATR proxy: rolling mean of abs returns; scaled to percent
    out["atr_pct"] = (
        out.groupby("symbol", sort=False)["ret_1d"].apply(lambda s: s.abs().rolling(cfg.atr_lookback).mean())
    ).reset_index(level=0, drop=True)

    # EMA slope (slow EMA diff)
    ema_s = out.groupby("symbol", sort=False)["close"].apply(lambda s: _ema(s, cfg.ema_slow))
    ema_s = ema_s.reset_index(level=0, drop=True)
    out["ema_slow"] = ema_s
    out["ema_slow_slope"] = out.groupby("symbol", sort=False)["ema_slow"].diff()

    # Compression proxy: low rv relative to atr_pct
    rv_col = f"rv_{cfg.rv_lookback}"
    out["compression"] = out[rv_col] / (out["atr_pct"].replace(0.0, np.nan))

    # Cross-sectional percentile features per date
    pct_cols = []
    raw_cols = [
        *[f"mom_{n}d" for n in cfg.mom_lookbacks],
        rv_col,
        "atr_pct",
        "ema_slow_slope",
        "compression",
    ]

    # Use groupby.transform for speed + no apply warnings
    for c in raw_cols:
        pc = f"{c}__pct"
        out[pc] = out.groupby("date", sort=False)[c].transform(_safe_pct_rank)
        pct_cols.append(pc)

    # Derived / interaction features (still numeric)
    # 1) binary-ish flags based on pct ranks
    out["is_high_mom_5d"] = (out.get("mom_5d__pct", np.nan) > 0.5).astype(float)
    out["is_high_rv10"] = (out.get(f"{rv_col}__pct", np.nan) > 0.5).astype(float)

    # 2) momentum shape
    if "mom_5d" in out.columns and "mom_1d" in out.columns:
        out["mom_5d_minus_mom_1d"] = out["mom_5d"] - out["mom_1d"]
        out["mom_5d_minus_mom_1d__pct"] = out.groupby("date", sort=False)["mom_5d_minus_mom_1d"].transform(_safe_pct_rank)
        pct_cols.append("mom_5d_minus_mom_1d__pct")

    # 3) relative rv vs atr
    out["rel_rv_to_atr"] = out[rv_col] / (out["atr_pct"].replace(0.0, np.nan))
    out["rel_rv_to_atr__pct"] = out.groupby("date", sort=False)["rel_rv_to_atr"].transform(_safe_pct_rank)
    pct_cols.append("rel_rv_to_atr__pct")

    # Keep NaNs as-is; downstream mining handles NaNs via masks.
    return out
