from __future__ import annotations

import pandas as pd



def add_momentum_20d(df1d: pd.DataFrame) -> pd.DataFrame:
    out = df1d.copy()
    out["momentum_20d"] = out["close"] / out["close"].shift(20) - 1.0
    return out