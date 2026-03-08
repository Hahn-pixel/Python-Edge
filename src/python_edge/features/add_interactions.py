from __future__ import annotations

import pandas as pd


INTERACTION_COLS = [
    "mom_x_volume_shock",
    "intraday_rs_x_volume_shock",
    "mom_x_vol_compression",
    "mom_x_market_breadth",
    "intraday_pressure_x_volume_shock",
    "liq_rank_x_intraday_rs",
]



def add_interactions(panel_df: pd.DataFrame) -> pd.DataFrame:
    out = panel_df.copy()

    required = [
        "momentum_20d",
        "volume_shock",
        "intraday_rs",
        "vol_compression",
        "market_breadth",
        "intraday_pressure",
        "liq_rank",
    ]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise RuntimeError(f"add_interactions: missing columns: {missing}")

    out["mom_x_volume_shock"] = out["momentum_20d"] * out["volume_shock"]
    out["intraday_rs_x_volume_shock"] = out["intraday_rs"] * out["volume_shock"]
    out["mom_x_vol_compression"] = out["momentum_20d"] * out["vol_compression"]
    out["mom_x_market_breadth"] = out["momentum_20d"] * out["market_breadth"]
    out["intraday_pressure_x_volume_shock"] = out["intraday_pressure"] * out["volume_shock"]
    out["liq_rank_x_intraday_rs"] = out["liq_rank"] * out["intraday_rs"]

    return out