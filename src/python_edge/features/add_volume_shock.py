from __future__ import annotations

import pandas as pd



def add_volume_shock(df1d: pd.DataFrame) -> pd.DataFrame:
    out = df1d.copy()
    vol = pd.to_numeric(out["volume"], errors="coerce")
    vol_ma20 = vol.rolling(20, min_periods=20).mean()
    out["volume_ma20"] = vol_ma20
    out["volume_shock"] = vol / vol_ma20
    return out