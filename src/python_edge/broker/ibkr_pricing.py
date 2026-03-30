from __future__ import annotations

import math
from typing import Tuple

from python_edge.broker.ibkr_models import PreparedOrder


def to_float(value) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def normalize_min_tick(value: float, fallback_min_abs: float) -> float:
    tick = to_float(value)
    if tick <= 0.0 or not math.isfinite(tick):
        return float(fallback_min_abs)
    return tick


def round_to_valid_tick(price: float, min_tick: float, side: str, min_abs: float) -> float:
    tick = normalize_min_tick(min_tick, min_abs)
    px = max(float(min_abs), float(price))
    steps = px / tick
    side_up = str(side).upper()
    if side_up == "BUY":
        rounded_steps = math.ceil(steps - 1e-12)
    elif side_up == "SELL":
        rounded_steps = math.floor(steps + 1e-12)
    else:
        rounded_steps = round(steps)
    rounded = rounded_steps * tick
    decimals = max(0, int(round(-math.log10(tick))) + 2) if tick < 1.0 else 4
    return round(max(float(min_abs), rounded), min(decimals, 8))


def compute_limit_price(
    prepared: PreparedOrder,
    buy_bps: float,
    sell_bps: float,
    min_abs: float,
) -> Tuple[float, float]:
    price_hint = float(prepared.price)
    if price_hint <= 0.0:
        price_hint = abs(float(prepared.order_notional) / max(abs(float(prepared.qty)), 1e-12))
    if price_hint <= 0.0:
        price_hint = 100.0
    side_bps = float(buy_bps) if prepared.order_side == "BUY" else float(sell_bps)
    bump = max(float(min_abs), price_hint * side_bps / 10000.0)
    if prepared.order_side == "BUY":
        raw_limit_price = price_hint + bump
    elif prepared.order_side == "SELL":
        raw_limit_price = max(float(min_abs), price_hint - bump)
    else:
        raw_limit_price = price_hint
    rounded_limit_price = round_to_valid_tick(raw_limit_price, prepared.min_tick, prepared.order_side, min_abs)
    return float(raw_limit_price), float(rounded_limit_price)


def limit_price_debug_payload(prepared: PreparedOrder, raw_limit_price: float, rounded_limit_price: float) -> dict:
    return {
        "price_hint": float(prepared.price),
        "order_notional": float(prepared.order_notional),
        "qty": float(prepared.qty),
        "min_tick": float(prepared.min_tick),
        "raw_limit_price": float(raw_limit_price),
        "rounded_limit_price": float(rounded_limit_price),
    }