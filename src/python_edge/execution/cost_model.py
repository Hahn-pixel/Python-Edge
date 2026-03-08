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

    side = pd.to_numeric(out["side"], errors="coerce").fillna(0.0)
    abs_side = side.abs()
    px = pd.to_numeric(out["meta_price"], errors="coerce")
    dv = pd.to_numeric(out["meta_dollar_volume"], errors="coerce")

    fee_cost = abs_side * (fee_bps / 10000.0)

    base_slip = pd.Series(slippage_bps / 10000.0, index=out.index, dtype="float64")
    liq_slip = pd.Series(0.0, index=out.index, dtype="float64")
    liq_slip.loc[dv < 20_000_000.0] = 1.0 / 10000.0
    liq_slip.loc[dv < 10_000_000.0] = 2.0 / 10000.0
    liq_slip.loc[dv < 5_000_000.0] = 4.0 / 10000.0
    liq_slip.loc[dv < 2_000_000.0] = 8.0 / 10000.0

    price_slip = pd.Series(0.0, index=out.index, dtype="float64")
    price_slip.loc[px < 20.0] = 1.0 / 10000.0
    price_slip.loc[px < 10.0] = 3.0 / 10000.0

    short_borrow = pd.Series(0.0, index=out.index, dtype="float64")
    short_mask = side < 0.0
    short_borrow.loc[short_mask] = abs_side.loc[short_mask] * (borrow_bps / 10000.0)

    total_slip_rate = base_slip + liq_slip + price_slip

    out["cost_fee"] = fee_cost
    out["cost_slippage"] = abs_side * total_slip_rate
    out["cost_borrow"] = short_borrow
    out["cost_total"] = out["cost_fee"] + out["cost_slippage"] + out["cost_borrow"]
    return out