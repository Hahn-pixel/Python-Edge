from __future__ import annotations

import pandas as pd



def add_overnight_drift_20d(df1d: pd.DataFrame) -> pd.DataFrame:
    out = df1d.copy()
    prev_close = out["close"].shift(1)
    overnight_ret = out["open"] / prev_close - 1.0
    out["overnight_ret_1d"] = overnight_ret
    out["overnight_drift_20d"] = overnight_ret.rolling(20, min_periods=20).mean()
    return out