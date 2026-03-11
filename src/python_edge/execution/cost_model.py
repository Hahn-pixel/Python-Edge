from __future__ import annotations

import pandas as pd


def attach_execution_costs(
    df: pd.DataFrame,
    weight_col: str = "weight",
    fee_bps: float = 1.0,
    slippage_bps: float = 2.0,
    borrow_bps_daily: float = 1.0,
    spread_bps: float = 3.0,
    impact_bps: float = 8.0,
    low_price_penalty_bps: float = 4.0,
    htb_borrow_bps_daily: float = 8.0,
    max_participation: float = 0.05,
    portfolio_notional: float = 1.0,
) -> pd.DataFrame:
    out = df.copy()
    required = [weight_col, "meta_dollar_volume", "meta_price", "trade_abs_after"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise RuntimeError(f"attach_execution_costs: missing columns: {missing}")

    weight = pd.to_numeric(out[weight_col], errors="coerce").fillna(0.0)
    trade_abs = pd.to_numeric(out["trade_abs_after"], errors="coerce").fillna(0.0)
    px = pd.to_numeric(out["meta_price"], errors="coerce").fillna(0.0)
    dv = pd.to_numeric(out["meta_dollar_volume"], errors="coerce").fillna(0.0)

    fee_cost = trade_abs * (fee_bps / 10000.0)

    spread_rate = pd.Series(spread_bps / 10000.0, index=out.index, dtype="float64")
    spread_rate.loc[dv < 20_000_000.0] += 1.0 / 10000.0
    spread_rate.loc[dv < 10_000_000.0] += 2.0 / 10000.0
    spread_rate.loc[dv < 5_000_000.0] += 4.0 / 10000.0
    spread_rate.loc[dv < 2_000_000.0] += 8.0 / 10000.0
    spread_rate.loc[px < 20.0] += low_price_penalty_bps / 10000.0
    spread_rate.loc[px < 10.0] += low_price_penalty_bps / 10000.0

    participation = pd.Series(0.0, index=out.index, dtype="float64")
    if portfolio_notional > 0.0:
        participation = (trade_abs * portfolio_notional) / dv.replace(0.0, pd.NA)
        participation = pd.to_numeric(participation, errors="coerce").fillna(max_participation * 4.0)
    participation_clipped = participation.clip(lower=0.0)
    impact_rate = (impact_bps / 10000.0) * (participation_clipped / max(max_participation, 1e-9)) ** 0.5
    impact_rate = impact_rate.clip(lower=0.0)

    slippage_rate = spread_rate + impact_rate
    slippage_cost = trade_abs * slippage_rate

    borrow_rate = pd.Series(0.0, index=out.index, dtype="float64")
    short_mask = weight < 0.0
    borrow_rate.loc[short_mask] = borrow_bps_daily / 10000.0
    borrow_rate.loc[short_mask & (dv < 5_000_000.0)] += 2.0 / 10000.0
    borrow_rate.loc[short_mask & (dv < 2_000_000.0)] += 4.0 / 10000.0
    borrow_rate.loc[short_mask & (px < 10.0)] += htb_borrow_bps_daily / 10000.0

    if "borrow_bucket" in out.columns:
        bucket = out["borrow_bucket"].astype(str).str.upper()
        borrow_rate.loc[short_mask & (bucket == "HTB")] += htb_borrow_bps_daily / 10000.0
        borrow_rate.loc[short_mask & (bucket == "VHTB")] += (htb_borrow_bps_daily * 2.0) / 10000.0

    borrow_cost = weight.abs() * borrow_rate

    out["cost_fee"] = fee_cost
    out["cost_slippage"] = slippage_cost
    out["cost_borrow"] = borrow_cost
    out["cost_trading"] = out["cost_fee"] + out["cost_slippage"]
    out["cost_total"] = out["cost_trading"] + out["cost_borrow"]
    out["execution_spread_rate"] = spread_rate
    out["execution_impact_rate"] = impact_rate
    out["execution_slippage_rate"] = slippage_rate
    out["execution_participation"] = participation_clipped
    out["execution_borrow_rate"] = borrow_rate
    out["execution_participation_flag"] = (participation_clipped > max_participation).astype("int64")
    return out