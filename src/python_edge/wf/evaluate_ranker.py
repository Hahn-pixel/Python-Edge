from __future__ import annotations

import pandas as pd


def evaluate_long_short(df: pd.DataFrame) -> pd.DataFrame:

    if "side" not in df.columns:
        raise RuntimeError("evaluate: missing side")

    if "target_fwd_ret_1d" not in df.columns:
        raise RuntimeError("evaluate: missing target")

    out = df.copy()

    out["pnl"] = out["side"] * out["target_fwd_ret_1d"]

    res = (
        out.groupby("date")["pnl"]
        .mean()
        .reset_index(name="portfolio_ret")
    )

    return res