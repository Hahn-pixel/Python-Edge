from __future__ import annotations

import pandas as pd



def attach_dynamic_side_budgets(
    df: pd.DataFrame,
    min_long_budget: float = 0.35,
    max_long_budget: float = 0.75,
    input_lag_days: int = 1,
) -> pd.DataFrame:
    out = df.copy()

    required = ["date", "market_breadth", "intraday_rs", "volume_shock", "intraday_pressure"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise RuntimeError(f"attach_dynamic_side_budgets: missing columns: {missing}")
    if input_lag_days < 0:
        raise ValueError("input_lag_days must be >= 0")

    by_date = out.groupby("date", as_index=False).agg(
        market_breadth=("market_breadth", "mean"),
        market_intraday_rs=("intraday_rs", "mean"),
        market_volume_shock=("volume_shock", "mean"),
        market_intraday_pressure=("intraday_pressure", "mean"),
    )
    by_date = by_date.sort_values("date").reset_index(drop=True)

    if input_lag_days > 0:
        for col in ["market_breadth", "market_intraday_rs", "market_volume_shock", "market_intraday_pressure"]:
            by_date[col] = by_date[col].shift(input_lag_days)

    breadth_component = (pd.to_numeric(by_date["market_breadth"], errors="coerce") - 0.50) * 0.90
    flow_component = pd.to_numeric(by_date["market_intraday_rs"], errors="coerce").fillna(0.0) * 2.50
    pressure_component = (pd.to_numeric(by_date["market_intraday_pressure"], errors="coerce") - 0.50) * 0.40
    volume_component = (pd.to_numeric(by_date["market_volume_shock"], errors="coerce") - 1.00) * 0.08

    long_budget = 0.50 + breadth_component + flow_component + pressure_component + volume_component
    long_budget = long_budget.fillna(0.50).clip(lower=min_long_budget, upper=max_long_budget)
    short_budget = 1.0 - long_budget

    by_date["long_budget"] = long_budget
    by_date["short_budget"] = short_budget
    by_date["budget_signal_lag_days"] = int(input_lag_days)

    out = out.merge(by_date[["date", "long_budget", "short_budget", "budget_signal_lag_days"]], on="date", how="left")
    out["long_budget"] = pd.to_numeric(out["long_budget"], errors="coerce").fillna(0.50)
    out["short_budget"] = pd.to_numeric(out["short_budget"], errors="coerce").fillna(0.50)
    out["budget_signal_lag_days"] = pd.to_numeric(out["budget_signal_lag_days"], errors="coerce").fillna(input_lag_days).astype("int64")
    return out



def apply_side_budgets(
    df: pd.DataFrame,
    weight_col: str = "weight",
) -> pd.DataFrame:
    out = df.copy()
    required = ["date", weight_col, "long_budget", "short_budget"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise RuntimeError(f"apply_side_budgets: missing columns: {missing}")

    out[weight_col] = pd.to_numeric(out[weight_col], errors="coerce").fillna(0.0)

    for _, idx in out.groupby("date").groups.items():
        long_budget = float(out.loc[idx, "long_budget"].iloc[0])
        short_budget = float(out.loc[idx, "short_budget"].iloc[0])

        w = pd.to_numeric(out.loc[idx, weight_col], errors="coerce").fillna(0.0)
        long_mask = w > 0.0
        short_mask = w < 0.0

        long_gross = float(w.loc[long_mask].sum())
        short_gross = float((-w.loc[short_mask]).sum())

        if long_gross > 0.0:
            out.loc[idx[long_mask], weight_col] = w.loc[long_mask] * (long_budget / long_gross)
        if short_gross > 0.0:
            out.loc[idx[short_mask], weight_col] = w.loc[short_mask] * (short_budget / short_gross)

    return out