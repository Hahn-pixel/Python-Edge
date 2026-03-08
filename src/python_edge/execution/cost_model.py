from __future__ import annotations

import pandas as pd



def attach_execution_costs(
    df: pd.DataFrame,
    fee_bps: float = 1.0,
    slippage_bps: float = 2.0,
    borrow_bps: float = 1.0,
) -> pd.DataFrame:
    out = df.copy()

    required = ["side", "target_fwd_ret_1d", "meta_dollar_volume", "meta_price"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise RuntimeError(f"attach_execution_costs: missing columns: {missing}")

    abs_side = pd.to_numeric(out["side"], errors="coerce").abs().fillna(0.0)
    px = pd.to_numeric(out["meta_price"], errors="coerce")
    dv = pd.to_numeric(out["meta_dollar_volume"], errors="coerce")

    fee_cost = abs_side * (fee_bps / 10000.0)
    base_slip = abs_side * (slippage_bps / 10000.0)

    liquidity_penalty = pd.Series(0.0, index=out.index, dtype="float64")
    low_liq = dv < 2_000_000.0
    mid_liq = (dv >= 2_000_000.0) & (dv < 10_000_000.0)
    liquidity_penalty.loc[mid_liq] = abs_side.loc[mid_liq] * (1.0 / 10000.0)
    liquidity_penalty.loc[low_liq] = abs_side.loc[low_liq] * (3.0 / 10000.0)

    price_penalty = pd.Series(0.0, index=out.index, dtype="float64")
    cheap = px < 10.0
    price_penalty.loc[cheap] = abs_side.loc[cheap] * (2.0 / 10000.0)

    short_borrow = pd.Series(0.0, index=out.index, dtype="float64")
    short_mask = pd.to_numeric(out["side"], errors="coerce") < 0
    short_borrow.loc[short_mask] = abs_side.loc[short_mask] * (borrow_bps / 10000.0)

    out["cost_fee"] = fee_cost
    out["cost_slippage"] = base_slip + liquidity_penalty + price_penalty
    out["cost_borrow"] = short_borrow
    out["cost_total"] = out["cost_fee"] + out["cost_slippage"] + out["cost_borrow"]
    return out