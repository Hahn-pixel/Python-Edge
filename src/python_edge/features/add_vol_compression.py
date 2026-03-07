from __future__ import annotations

import pandas as pd


def add_vol_compression(df1d: pd.DataFrame) -> pd.DataFrame:
    out = df1d.copy()
    ret = out["close"].pct_change()
    vol_5 = ret.rolling(5, min_periods=5).std()
    vol_20 = ret.rolling(20, min_periods=20).std()
    out["vol_5d"] = vol_5
    out["vol_20d"] = vol_20
    out["vol_compression"] = vol_5 / vol_20
    return out