from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Execution state machine states
# ---------------------------------------------------------------------------

class ExecState(str, Enum):
    NEW           = "NEW"
    PRECHECK      = "PRECHECK"
    SPLIT         = "SPLIT"
    WHOLE_SUBMIT  = "WHOLE_SUBMIT"
    WHOLE_WORKING = "WHOLE_WORKING"
    WHOLE_FILLED  = "WHOLE_FILLED"
    WHOLE_PARTIAL = "WHOLE_PARTIAL"
    WHOLE_TIMEOUT = "WHOLE_TIMEOUT"
    FRAC_SUBMIT   = "FRAC_SUBMIT"
    FRAC_WORKING  = "FRAC_WORKING"
    DONE          = "DONE"
    FAILED        = "FAILED"


# ---------------------------------------------------------------------------
# Order side
# ---------------------------------------------------------------------------

class OrderSide(str, Enum):
    BUY  = "BUY"
    SELL = "SELL"


# ---------------------------------------------------------------------------
# Execution intent — the parent unit that drives the state machine
# ---------------------------------------------------------------------------

@dataclass
class ExecutionIntent:
    """Represents a single symbol's desired execution from orders.csv."""
    symbol:           str
    conid:            str          # IBKR contract ID (required by CPAPI)
    side:             OrderSide
    target_qty:       float        # total qty including fractional part
    parent_cap:       Optional[float]   # BUY: price must be <= this
    parent_floor:     Optional[float]   # SELL: price must be >= this
    client_tag:       str          # idempotency key
    account_id:       str

    # Derived at SPLIT stage
    whole_qty:        float = 0.0
    frac_qty:         float = 0.0

    # Runtime state
    state:            ExecState = ExecState.NEW
    whole_order_id:   Optional[str] = None
    frac_order_id:    Optional[str] = None
    whole_filled_qty: float = 0.0
    whole_avg_price:  float = 0.0
    frac_filled_qty:  float = 0.0
    frac_avg_price:   float = 0.0

    # Audit trail — every state transition is appended here
    transitions:      List[Dict[str, Any]] = field(default_factory=list)
    errors:           List[Dict[str, Any]] = field(default_factory=list)

    # Debug counters (no silent fail-open)
    debug_precheck_ok:        int = 0
    debug_precheck_fail:      int = 0
    debug_split_ok:           int = 0
    debug_whole_submitted:    int = 0
    debug_whole_filled:       int = 0
    debug_whole_partial:      int = 0
    debug_whole_timeout:      int = 0
    debug_whole_cancel_sent:  int = 0   # cancel request sent on WHOLE_TIMEOUT
    debug_whole_cancel_fail:  int = 0   # cancel request failed (order may have filled)
    debug_frac_submitted:     int = 0
    debug_frac_filled:        int = 0
    debug_guard_rejected:     int = 0
    debug_failed:             int = 0


# ---------------------------------------------------------------------------
# CPAPI order request/response containers
# ---------------------------------------------------------------------------

@dataclass
class CpapiOrderRequest:
    """Payload sent to /iserver/account/{account}/orders."""
    conid:      str
    side:       str          # "BUY" or "SELL"
    quantity:   float
    order_type: str          # "MKT", "LMT", "MIDPRICE"
    price:      Optional[float]  # required for LMT; None for MKT/MIDPRICE
    tif:        str          # "DAY", "GTC", etc.
    account_id: str
    client_tag: str          # cOID / order reference for idempotency


@dataclass
class CpapiOrderResponse:
    """Parsed response from order submission endpoint."""
    order_id:   str
    local_order_id: Optional[str]
    message:    Optional[str]       # may contain a confirmation question
    needs_reply: bool               # True when /iserver/reply/{id} is needed
    raw:        Dict[str, Any] = field(default_factory=dict)


@dataclass
class CpapiOrderStatus:
    """Parsed result from polling /iserver/account/trades or order status."""
    order_id:     str
    status:       str           # "Filled", "Submitted", "Cancelled", etc.
    filled_qty:   float
    remaining_qty: float
    avg_price:    float
    last_price:   float
    raw:          Dict[str, Any] = field(default_factory=dict)


@dataclass
class FillResult:
    """Final execution result attached to an ExecutionIntent after DONE."""
    symbol:         str
    side:           str
    total_filled:   float
    avg_price:      float
    whole_filled:   float
    whole_avg_price: float
    frac_filled:    float
    frac_avg_price: float
    final_state:    ExecState
    client_tag:     str


# ---------------------------------------------------------------------------
# Auth / session status
# ---------------------------------------------------------------------------

@dataclass
class AuthStatus:
    authenticated: bool
    competing:     bool
    connected:     bool
    message:       str
    raw:           Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Position snapshot (from /portfolio/{account}/positions/0)
# ---------------------------------------------------------------------------

@dataclass
class CpapiPosition:
    conid:          str
    symbol:         str
    position:       float
    avg_cost:       float
    market_value:   float
    account_id:     str
    raw:            Dict[str, Any] = field(default_factory=dict)
