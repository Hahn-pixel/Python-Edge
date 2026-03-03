from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DailyFeatureConfig:
    ema_fast: int = 20
    ema_slow: int = 50
    atr_n: int = 14
    rv_n: int = 10
    mom_days: Tuple[int, ...] = (1, 3, 5, 10, 20)
    fwd_days: int = 5  # forward horizon for labels


def _ema(x: pd.Series, span: int) -> pd.Series:
    return x.ewm(span=span, adjust=False).mean()


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_c = df["c"].shift(1)
    tr1 = (df["h"] - df["l"]).abs()
    tr2 = (df["h"] - prev_c).abs()
    tr3 = (df["l"] - prev_c).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def build_daily_features(df: pd.DataFrame, cfg: DailyFeatureConfig) -> pd.DataFrame:
    """
    Input df columns: date(str), o,h,l,c,v (daily bars).
    Output: features + forward label columns.
    """
    if df.empty:
        return df.copy()

    out = df.copy().sort_values("date").reset_index(drop=True)

    # returns
    out["ret_1d"] = out["c"].pct_change(1)

    # momentum
    for n in cfg.mom_days:
        out[f"mom_{n}d"] = out["c"].pct_change(n)

    # EMAs and slope proxies
    out[f"ema_{cfg.ema_fast}"] = _ema(out["c"], cfg.ema_fast)
    out[f"ema_{cfg.ema_slow}"] = _ema(out["c"], cfg.ema_slow)
    out["ema_dist"] = (out[f"ema_{cfg.ema_fast}"] / out[f"ema_{cfg.ema_slow}"] - 1.0)

    # EMA slope as normalized delta
    out["ema_fast_slope"] = out[f"ema_{cfg.ema_fast}"].pct_change(5)
    out["ema_slow_slope"] = out[f"ema_{cfg.ema_slow}"].pct_change(10)

    # ATR%
    tr = _true_range(out)
    out[f"atr_{cfg.atr_n}"] = tr.rolling(cfg.atr_n).mean()
    out["atr_pct"] = out[f"atr_{cfg.atr_n}"] / out["c"]

    # realized vol (std of daily returns)
    out[f"rv_{cfg.rv_n}"] = out["ret_1d"].rolling(cfg.rv_n).std()

    # compression proxy: current range vs rolling median range
    out["range"] = (out["h"] - out["l"]) / out["c"]
    out["range_med_20"] = out["range"].rolling(20).median()
    out["compression"] = out["range"] / out["range_med_20"]

    # forward label: forward return over cfg.fwd_days
    out[f"fwd_{cfg.fwd_days}d_ret"] = out["c"].shift(-cfg.fwd_days) / out["c"] - 1.0

    # "event" labels (move-centric): impulse up/down relative to rolling sigma
    sig = out["ret_1d"].rolling(60).std()
    out["impulse_up"] = (out[f"fwd_{cfg.fwd_days}d_ret"] > (1.5 * sig * np.sqrt(cfg.fwd_days))).astype("Int64")
    out["impulse_dn"] = (out[f"fwd_{cfg.fwd_days}d_ret"] < (-1.5 * sig * np.sqrt(cfg.fwd_days))).astype("Int64")

    return out