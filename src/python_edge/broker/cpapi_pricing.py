from __future__ import annotations

import math
from typing import Optional

from python_edge.broker.cpapi_models import ExecutionIntent, OrderSide

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_PRICE_ABS: float = 0.0001   # floor for any computed price
_DEFAULT_MIN_TICK: float = 0.01  # fallback tick when conid metadata absent


# ---------------------------------------------------------------------------
# Tick helpers
# ---------------------------------------------------------------------------

def _safe_float(value: object) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return 0.0


def _normalize_tick(tick: float, fallback: float = _DEFAULT_MIN_TICK) -> float:
    t = _safe_float(tick)
    if t <= 0.0 or not math.isfinite(t):
        return float(fallback)
    return t


def round_to_tick(price: float, tick: float, side: OrderSide) -> float:
    """
    Round *price* to the nearest valid tick boundary.
    BUY  → round UP   (never pay less than full tick)
    SELL → round DOWN (never accept less than full tick)
    """
    t = _normalize_tick(tick)
    px = max(_MIN_PRICE_ABS, float(price))
    steps = px / t
    if side is OrderSide.BUY:
        rounded_steps = math.ceil(steps - 1e-12)
    else:
        rounded_steps = math.floor(steps + 1e-12)
    rounded = rounded_steps * t
    if t < 1.0:
        decimals = min(max(0, int(round(-math.log10(t))) + 2), 8)
    else:
        decimals = 4
    return round(max(_MIN_PRICE_ABS, rounded), decimals)


# ---------------------------------------------------------------------------
# Parent price guard
# ---------------------------------------------------------------------------

class PriceGuardViolation(Exception):
    """Raised when a proposed execution price violates the parent intent."""


def check_parent_guard(
    intent: ExecutionIntent,
    proposed_price: float,
    label: str = "",
) -> None:
    """
    Validate *proposed_price* against the parent intent constraints.

    BUY  → price must be <= parent_cap   (if cap is set)
    SELL → price must be >= parent_floor (if floor is set)

    Raises PriceGuardViolation explicitly — never silently passes.
    """
    px = _safe_float(proposed_price)
    tag = f"[{label}] " if label else ""

    if intent.side is OrderSide.BUY and intent.parent_cap is not None:
        cap = _safe_float(intent.parent_cap)
        if px > cap + 1e-9:
            intent.debug_guard_rejected += 1
            raise PriceGuardViolation(
                f"{tag}BUY guard violated for {intent.symbol}: "
                f"proposed_price={px:.6f} > parent_cap={cap:.6f}"
            )

    if intent.side is OrderSide.SELL and intent.parent_floor is not None:
        floor_ = _safe_float(intent.parent_floor)
        if px < floor_ - 1e-9:
            intent.debug_guard_rejected += 1
            raise PriceGuardViolation(
                f"{tag}SELL guard violated for {intent.symbol}: "
                f"proposed_price={px:.6f} < parent_floor={floor_:.6f}"
            )


def clamp_to_guard(
    intent: ExecutionIntent,
    price: float,
    tick: float,
) -> float:
    """
    Return *price* clamped so it never violates the parent guard.
    Rounds to valid tick after clamping.
    Does NOT raise — caller decides whether to proceed.
    """
    px = _safe_float(price)

    if intent.side is OrderSide.BUY and intent.parent_cap is not None:
        cap = _safe_float(intent.parent_cap)
        px = min(px, cap)

    if intent.side is OrderSide.SELL and intent.parent_floor is not None:
        floor_ = _safe_float(intent.parent_floor)
        px = max(px, floor_)

    return round_to_tick(px, tick, intent.side)


# ---------------------------------------------------------------------------
# Fractional split
# ---------------------------------------------------------------------------

def split_qty(target_qty: float) -> tuple[float, float]:
    """
    Split target_qty into (whole_qty, frac_qty).

    whole_qty = floor(target_qty)   → sent as LMT at mid price
    frac_qty  = target_qty - whole  → sent as LMT with slippage after whole fills

    Returns (0.0, target_qty) when target_qty < 1.0 (pure fractional order).
    """
    qty = max(0.0, float(target_qty))
    whole = math.floor(qty)
    frac = round(qty - whole, 8)
    return float(whole), frac


# ---------------------------------------------------------------------------
# Whole leg: LMT at mid (replaces MIDPRICE)
# ---------------------------------------------------------------------------

# Whole leg slippage: невеликий буфер від mid щоб підвищити ймовірність fill.
# Менший ніж frac slippage — whole leg велика кількість акцій,
# зайвий bps коштує дорожче. Default: 1 bps від mid.
_WHOLE_SLIPPAGE_BPS_DEFAULT: float = 1.0


def whole_limit_price(
    intent: ExecutionIntent,
    tick: float,
    reference_price: float,
    whole_slippage_bps: float = _WHOLE_SLIPPAGE_BPS_DEFAULT,
) -> float:
    """
    Compute an explicit LMT price for the whole-qty leg.

    Замінює MIDPRICE — дає явну ціну щоб:
      1. Gateway не потребував market data streaming
      2. tif=DAY працював як очікується (MIDPRICE → CLOSE поза RTH)
      3. Ціна контрольована і логована

    Логіка:
      reference_price = mid з orders.csv (BBO midpoint від handoff)
      BUY:  limit = mid * (1 + whole_slippage_bps/10000)  → трохи вище mid
      SELL: limit = mid * (1 - whole_slippage_bps/10000)  → трохи нижче mid

    whole_slippage_bps default=1bps — мінімальний буфер для fill probability.
    Для whole leg це важливо: 66 акцій по $154 = $10164, 1bps = $1.02 переплата.

    Raises PriceGuardViolation якщо ціна виходить за parent guard.
    """
    ref = _safe_float(reference_price)
    if ref <= 0.0:
        raise ValueError(
            f"reference_price must be > 0 for whole_limit_price "
            f"(symbol={intent.symbol}, got {ref})"
        )

    bps_bump = abs(float(whole_slippage_bps)) / 10_000.0

    if intent.side is OrderSide.BUY:
        raw = ref * (1.0 + bps_bump)
    else:
        raw = ref * max(0.0, 1.0 - bps_bump)

    clamped = clamp_to_guard(intent, raw, tick)
    check_parent_guard(intent, clamped, label="whole_limit_price")

    print(
        f"[PRICING][{intent.symbol}] whole LMT: "
        f"ref={ref:.4f} slippage={whole_slippage_bps:.1f}bps "
        f"raw={raw:.4f} clamped={clamped:.4f} side={intent.side.value}"
    )

    return clamped


# ---------------------------------------------------------------------------
# Legacy alias — зберігаємо для зворотної сумісності якщо є інші імпорти
# ---------------------------------------------------------------------------

def midprice_price(
    intent: ExecutionIntent,
    tick: float,
    reference_price: float,
) -> Optional[float]:
    """
    Deprecated: використовувався для MIDPRICE order_type.
    Залишено для зворотної сумісності — викликає whole_limit_price.
    Повертає явну LMT ціну замість None.
    """
    return whole_limit_price(intent, tick, reference_price)


# ---------------------------------------------------------------------------
# Fractional leg: LMT with adaptive slippage
# ---------------------------------------------------------------------------

def frac_limit_price(
    intent: ExecutionIntent,
    tick: float,
    reference_price: float,
    slippage_bps: float = 5.0,
) -> float:
    """
    Compute a limit price for the fractional remainder leg.

    slippage_bps тут адаптивний (передається з _build_intent):
      = max(5, min(30, spread_bps/2 + 2))

    Raises PriceGuardViolation if even the clamped price violates the guard.
    """
    ref = _safe_float(reference_price)
    if ref <= 0.0:
        raise ValueError(
            f"reference_price must be > 0 for frac_limit_price "
            f"(symbol={intent.symbol}, got {ref})"
        )

    bps_bump = abs(float(slippage_bps)) / 10_000.0

    if intent.side is OrderSide.BUY:
        raw = ref * (1.0 + bps_bump)
    else:
        raw = ref * max(0.0, 1.0 - bps_bump)

    clamped = clamp_to_guard(intent, raw, tick)
    check_parent_guard(intent, clamped, label="frac_limit_price")
    return clamped
