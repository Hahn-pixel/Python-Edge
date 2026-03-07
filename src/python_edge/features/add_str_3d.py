from __future__ import annotations

import pandas as pd



def add_str_3d(df1d: pd.DataFrame) -> pd.DataFrame:
    out = df1d.copy()
    out["str_3d"] = -(out["close"] / out["close"].shift(3) - 1.0)
    return out