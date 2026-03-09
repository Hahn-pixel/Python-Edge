from __future__ import annotations

import pandas as pd



def attach_execution_costs(
    df: pd.DataFrame,
    weight_col: str = "weight",
    fee_bps: float = 1.0,
    slippage_bps: float = 2.0,
    borrow_bps_daily: float = 1.0,
) -> pd.DataFrame:
    out = df.copy()

    required = [weight_col, "meta_dollar_volume", "meta_price", "trade_abs_after"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise RuntimeError(f"attach_execution_costs: missing columns: {missing}")

    weight = pd.to_numeric(out[weight_col], errors="coerce").fillna(0.0)
    trade_abs = pd.to_numeric(out["trade_abs_after"], errors="coerce").fillna(0.0)
    px = pd.to_numeric(out["meta_price"], errors="coerce")
    dv = pd.to_numeric(out["meta_dollar_volume"], errors="coerce")

    fee_cost = trade_abs * (fee_bps / 10000.0)

    slip_rate = pd.Series(slippage_bps / 10000.0, index=out.index, dtype="float64")
    slip_rate.loc[dv < 20_000_000.0] += 1.0 / 10000.0
    slip_rate.loc[dv < 10_000_000.0] += 2.0 / 10000.0
    slip_rate.loc[dv < 5_000_000.0] += 4.0 / 10000.0
    slip_rate.loc[dv < 2_000_000.0] += 8.0 / 10000.0
    slip_rate.loc[px < 20.0] += 1.0 / 10000.0
    slip_rate.loc[px < 10.0] += 3.0 / 10000.0

    slippage_cost = trade_abs * slip_rate

    borrow_cost = pd.Series(0.0, index=out.index, dtype="float64")
    short_mask = weight < 0.0
    borrow_cost.loc[short_mask] = weight.loc[short_mask].abs() * (borrow_bps_daily / 10000.0)

    out["cost_fee"] = fee_cost
    out["cost_slippage"] = slippage_cost
    out["cost_borrow"] = borrow_cost
    out["cost_trading"] = out["cost_fee"] + out["cost_slippage"]
    out["cost_total"] = out["cost_trading"] + out["cost_borrow"]
    return out