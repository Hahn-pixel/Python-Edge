from __future__ import annotations

import pandas as pd


def add_intraday_rs(df15: pd.DataFrame, df1d: pd.DataFrame) -> pd.DataFrame:
    if "session_date" not in df15.columns:
        raise RuntimeError("add_intraday_rs: df15 must contain 'session_date'")
    if "open" not in df15.columns or "close" not in df15.columns:
        raise RuntimeError("add_intraday_rs: df15 must contain open/close")
    if "session_date" not in df1d.columns:
        raise RuntimeError("add_intraday_rs: df1d must contain 'session_date'")

    x = df15.copy()
    x["bar_ret"] = x["close"] / x["open"] - 1.0
    daily = x.groupby("session_date", as_index=False).agg(
        intraday_ret_sum=("bar_ret", "sum"),
        intraday_ret_mean=("bar_ret", "mean"),
        intraday_ret_std=("bar_ret", "std"),
    )
    daily["intraday_rs"] = daily["intraday_ret_sum"]

    out = df1d.copy()
    out = out.merge(daily[["session_date", "intraday_rs"]], on="session_date", how="left")
    return out