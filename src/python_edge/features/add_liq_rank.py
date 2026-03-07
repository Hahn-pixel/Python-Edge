from __future__ import annotations

import pandas as pd


def add_liq_rank(panel_df: pd.DataFrame) -> pd.DataFrame:
    out = panel_df.copy()
    if "date" not in out.columns:
        raise RuntimeError("add_liq_rank: missing 'date' column")
    if "meta_dollar_volume" not in out.columns:
        raise RuntimeError("add_liq_rank: missing 'meta_dollar_volume' column")
    out["liq_rank"] = out.groupby("date")["meta_dollar_volume"].rank(method="average", pct=True)
    return out