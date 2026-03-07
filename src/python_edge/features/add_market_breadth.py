from __future__ import annotations

import pandas as pd


def add_market_breadth(panel_df: pd.DataFrame) -> pd.DataFrame:
    out = panel_df.copy()
    required = ["date", "close", "symbol"]
    for col in required:
        if col not in out.columns:
            raise RuntimeError(f"add_market_breadth: missing {col!r}")

    out = out.sort_values(["symbol", "date"], ascending=[True, True]).reset_index(drop=True)
    out["ret_1d_tmp"] = out.groupby("symbol")["close"].pct_change()
    breadth = out.groupby("date", as_index=False).agg(
        market_breadth=("ret_1d_tmp", lambda s: float((s > 0).mean()) if len(s.dropna()) > 0 else float("nan")),
        market_ret_mean=("ret_1d_tmp", "mean"),
    )
    out = out.merge(breadth[["date", "market_breadth", "market_ret_mean"]], on="date", how="left")
    out = out.drop(columns=["ret_1d_tmp"])
    return out