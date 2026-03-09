from __future__ import annotations

import pandas as pd


def evaluate_long_short(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    out = df.copy()

    out["gross_ret"] = out["weight"] * out[target_col]
    out["net_ret"] = out["gross_ret"] - out.get("trading_cost", 0.0) - out.get("borrow_cost", 0.0)

    daily = out.groupby("date", as_index=False).agg(
        gross_ret=("gross_ret", "sum"),
        net_ret=("net_ret", "sum"),
        raw_turnover=("raw_turnover", "mean"),
        capped_turnover=("capped_turnover", "mean"),
        cap_hit_rate=("cap_hit", "mean"),
    )

    return daily


def summarize_daily_returns(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}

    out = {}

    out["days"] = int(len(df))
    out["avg_daily_ret"] = float(df["net_ret"].mean())
    out["std_daily_ret"] = float(df["net_ret"].std())
    out["win_rate_days"] = float((df["net_ret"] > 0).mean())
    out["cum_ret"] = float((1 + df["net_ret"]).prod() - 1)

    out["avg_raw_turnover"] = float(df.get("raw_turnover", pd.Series()).mean())
    out["avg_turnover"] = float(df.get("capped_turnover", pd.Series()).mean())
    out["cap_hit_rate"] = float(df.get("cap_hit_rate", pd.Series()).mean())

    return out


def print_summary(prefix: str, summary: dict) -> None:
    if not summary:
        print(prefix, "EMPTY")
        return

    print(prefix, summary)