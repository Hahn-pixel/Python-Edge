from __future__ import annotations

import pandas as pd


def add_target_fwd_ret_1d(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["target_fwd_ret_1d"] = out["close"].shift(-1) / out["close"] - 1.0
    return out


def add_target_fwd_ret_3d(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["target_fwd_ret_3d"] = out["close"].shift(-3) / out["close"] - 1.0
    return out


def add_target_fwd_ret_5d(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["target_fwd_ret_5d"] = out["close"].shift(-5) / out["close"] - 1.0
    return out


def add_all_forward_return_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = add_target_fwd_ret_1d(out)
    out = add_target_fwd_ret_3d(out)
    out = add_target_fwd_ret_5d(out)
    return out