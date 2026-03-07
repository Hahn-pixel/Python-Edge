from __future__ import annotations

import pandas as pd


def add_intraday_pressure(df15: pd.DataFrame, df1d: pd.DataFrame) -> pd.DataFrame:
    if "session_date" not in df15.columns:
        raise RuntimeError("add_intraday_pressure: df15 must contain 'session_date'")
    required = ["open", "high", "low", "close"]
    for col in required:
        if col not in df15.columns:
            raise RuntimeError(f"add_intraday_pressure: df15 missing {col!r}")

    x = df15.copy()
    rng = (x["high"] - x["low"]).replace(0, pd.NA)
    x["close_pos_in_bar"] = (x["close"] - x["low"]) / rng
    daily = x.groupby("session_date", as_index=False).agg(
        intraday_pressure_mean=("close_pos_in_bar", "mean"),
        intraday_pressure_last=("close_pos_in_bar", "last"),
    )
    daily["intraday_pressure"] = daily["intraday_pressure_last"]

    out = df1d.copy()
    out = out.merge(daily[["session_date", "intraday_pressure"]], on="session_date", how="left")
    return out