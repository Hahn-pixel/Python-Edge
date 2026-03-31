from __future__ import annotations

import re
import time
from typing import Any, Callable, Dict, List, Sequence, Tuple

from ibapi.contract import Contract

from python_edge.broker.ibkr_models import BrokerErrorInfo, PreparedOrder

RE_202_AGGR = re.compile(r"at or more aggressive than\s+([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
RE_202_MARKET = re.compile(r"current market price of\s+([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)


def normalize_status(status: str) -> str:
    return str(status or "").strip().lower().replace(" ", "")


def classify_outcome(status: str, filled_qty: float, remaining_qty: float) -> str:
    status_norm = normalize_status(status)
    if status_norm == "filled" or (filled_qty > 0.0 and remaining_qty <= 1e-9):
        return "filled_now"
    if status_norm == "partiallyfilled" or (filled_qty > 0.0 and remaining_qty > 1e-9):
        return "partial"
    if status_norm in {"submitted", "presubmitted", "pendingsubmit", "api_pending", "pending_submit"}:
        return "working"
    if status_norm in {"cancelled", "inactive", "apicancelled"}:
        return "failed"
    return "unknown"


def parse_202_limits(error_string: str) -> Tuple[float | None, float | None]:
    s = str(error_string or "")
    m1 = RE_202_AGGR.search(s)
    m2 = RE_202_MARKET.search(s)
    aggressive_cutoff = float(m1.group(1)) if m1 else None
    market_price = float(m2.group(1)) if m2 else None
    return aggressive_cutoff, market_price


def collect_app_errors(app, start_idx: int, req_ids: Sequence[int], utc_now_iso: Callable[[], str]) -> List[BrokerErrorInfo]:
    req_id_set = {int(x) for x in req_ids}
    out: List[BrokerErrorInfo] = []
    for payload in app._errors[start_idx:]:
        try:
            req_id = int(payload.get("reqId", 0) or 0)
        except Exception:
            req_id = 0
        if req_id_set and req_id not in req_id_set:
            continue
        out.append(
            BrokerErrorInfo(
                req_id=req_id,
                error_code=int(payload.get("errorCode", 0) or 0),
                error_string=str(payload.get("errorString", "") or ""),
                advanced_order_reject_json=str(payload.get("advancedOrderRejectJson", "") or ""),
                ts_utc=str(payload.get("ts_utc", "") or utc_now_iso()),
            )
        )
    return out


def find_error_code(broker_errors: Sequence[BrokerErrorInfo], code: int) -> BrokerErrorInfo | None:
    for err in broker_errors:
        if int(err.error_code) == int(code):
            return err
    return None


def query_contract_details(app, contract: Contract, req_id: int, timeout_sec: float) -> dict:
    app.done_contract_details[int(req_id)] = False
    app.reqContractDetails(int(req_id), contract)
    return app.wait_for_contract_details(int(req_id), timeout_sec=timeout_sec)


def resolve_contract_metadata(
    *,
    app,
    prepared: PreparedOrder,
    req_id_seed: int,
    timeout_sec: float,
    retries: int,
    retry_sleep_sec: float,
    allow_primary_exchange_fallback: bool,
    allow_submit_without_contract_details: bool,
    build_contract: Callable[[PreparedOrder], Contract],
    normalize_min_tick: Callable[[float, float], float],
    min_abs: float,
    to_float: Callable[[Any], float],
) -> Tuple[PreparedOrder, Contract, int, Dict[str, Any]]:
    req_cursor = int(req_id_seed)
    attempts_total = max(1, int(retries))
    last_exc: Exception | None = None

    for attempt_idx in range(attempts_total):
        req_cursor += 1
        contract = build_contract(prepared)
        try:
            print(
                f"[IB][CONTRACT_DETAILS] symbol={prepared.symbol} broker_symbol={prepared.broker_symbol} "
                f"attempt={attempt_idx + 1}/{attempts_total} use_primary_exchange=1 req_id={req_cursor}"
            )
            meta = query_contract_details(app, contract, req_cursor, timeout_sec)
            prepared2 = PreparedOrder(**{**prepared.__dict__, "min_tick": normalize_min_tick(to_float(meta.get("minTick", 0.0)), min_abs)})
            contract2 = build_contract(prepared2)
            if meta.get("primaryExchange") and not getattr(contract2, "primaryExchange", ""):
                contract2.primaryExchange = str(meta.get("primaryExchange", ""))
            return prepared2, contract2, req_cursor, {
                "contract_details_mode": "full",
                "contract_details_attempts": attempt_idx + 1,
                "contract_details_fallback_used": 0,
            }
        except Exception as exc:
            last_exc = exc
            print(
                f"[IB][CONTRACT_DETAILS][WARN] symbol={prepared.symbol} broker_symbol={prepared.broker_symbol} "
                f"attempt={attempt_idx + 1}/{attempts_total} use_primary_exchange=1 error={exc}"
            )
            if attempt_idx + 1 < attempts_total:
                time.sleep(max(0.0, retry_sleep_sec))

    if allow_primary_exchange_fallback:
        req_cursor += 1
        contract = build_contract(prepared)
        try:
            contract.primaryExchange = ""
        except Exception:
            pass
        try:
            print(
                f"[IB][CONTRACT_DETAILS] symbol={prepared.symbol} broker_symbol={prepared.broker_symbol} "
                f"attempt=fallback_no_primary use_primary_exchange=0 req_id={req_cursor}"
            )
            meta = query_contract_details(app, contract, req_cursor, timeout_sec)
            prepared2 = PreparedOrder(**{**prepared.__dict__, "min_tick": normalize_min_tick(to_float(meta.get("minTick", 0.0)), min_abs)})
            contract2 = build_contract(prepared2)
            try:
                contract2.primaryExchange = ""
            except Exception:
                pass
            return prepared2, contract2, req_cursor, {
                "contract_details_mode": "no_primary_exchange",
                "contract_details_attempts": attempts_total + 1,
                "contract_details_fallback_used": 1,
            }
        except Exception as exc:
            last_exc = exc
            print(
                f"[IB][CONTRACT_DETAILS][WARN] symbol={prepared.symbol} broker_symbol={prepared.broker_symbol} "
                f"attempt=fallback_no_primary use_primary_exchange=0 error={exc}"
            )

    if allow_submit_without_contract_details:
        prepared2 = PreparedOrder(**{**prepared.__dict__, "min_tick": normalize_min_tick(min_abs, min_abs)})
        contract2 = build_contract(prepared2)
        if allow_primary_exchange_fallback:
            try:
                contract2.primaryExchange = ""
            except Exception:
                pass
        print(
            f"[IB][CONTRACT_DETAILS][FALLBACK_SUBMIT] symbol={prepared.symbol} broker_symbol={prepared.broker_symbol} "
            f"reason={last_exc} min_tick={prepared2.min_tick:.6f}"
        )
        return prepared2, contract2, req_cursor, {
            "contract_details_mode": "submit_without_contract_details",
            "contract_details_attempts": attempts_total + (1 if allow_primary_exchange_fallback else 0),
            "contract_details_fallback_used": 1,
        }

    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Failed to resolve contract details for symbol={prepared.symbol}")


def latest_ib_entry(app, ib_order_id: int, default_entry: dict) -> dict:
    latest = app.orders_by_ib_id.get(int(ib_order_id), default_entry)
    return latest if isinstance(latest, dict) else default_entry


def poll_order_status(
    *,
    app,
    ib_order_id: int,
    initial_entry: dict,
    attempts: int,
    sleep_sec: float,
    label: str,
    refresh_open_orders: Callable[[Any], None],
    print_mode: str,
    print_every: int,
    to_float: Callable[[Any], float],
) -> dict:
    latest = dict(initial_entry or {})
    last_signature = None
    total_attempts = max(0, int(attempts))
    every_n = max(1, int(print_every))
    mode = str(print_mode or "changes").strip().lower()
    for poll_idx in range(total_attempts):
        time.sleep(max(0.0, float(sleep_sec)))
        refresh_open_orders(app)
        latest = latest_ib_entry(app, ib_order_id, latest)
        status = normalize_status(str(latest.get("status", "")))
        filled_qty = to_float(latest.get("filled_qty", 0.0))
        remaining_qty = to_float(latest.get("remaining_qty", 0.0))
        outcome = classify_outcome(status, filled_qty, remaining_qty)
        signature = (status, round(filled_qty, 8), round(remaining_qty, 8), outcome)
        should_print = False
        if mode == "all":
            should_print = True
        elif mode == "none":
            should_print = poll_idx == total_attempts - 1 or outcome in {"filled_now", "partial", "failed"}
        else:
            should_print = (
                poll_idx == 0
                or poll_idx == total_attempts - 1
                or signature != last_signature
                or ((poll_idx + 1) % every_n == 0)
                or outcome in {"filled_now", "partial", "failed"}
            )
        if should_print:
            print(
                f"[BROKER][POLL][{label}] ib_order_id={ib_order_id} poll={poll_idx + 1}/{total_attempts} "
                f"status={status or 'unknown'} outcome={outcome} filled_qty={filled_qty:.8f} remaining_qty={remaining_qty:.8f}"
            )
        last_signature = signature
        if outcome in {"filled_now", "partial", "failed"}:
            break
    return latest


def cancel_and_poll_order(
    *,
    app,
    ib_order_id: int,
    initial_entry: dict,
    cancel_attempts: int,
    cancel_sleep_sec: float,
    refresh_open_orders: Callable[[Any], None],
    print_mode: str,
    print_every: int,
    to_float: Callable[[Any], float],
) -> dict:
    latest = dict(initial_entry or {})
    print(f"[BROKER][CANCEL] ib_order_id={ib_order_id} reason=working_ttl_expired")
    app.cancelOrder(int(ib_order_id), "")
    return poll_order_status(
        app=app,
        ib_order_id=ib_order_id,
        initial_entry=latest,
        attempts=cancel_attempts,
        sleep_sec=cancel_sleep_sec,
        label="cancel_after_ttl",
        refresh_open_orders=refresh_open_orders,
        print_mode=print_mode,
        print_every=print_every,
        to_float=to_float,
    )


def apply_working_order_policy(
    *,
    app,
    ib_order_id: int,
    current_entry: dict,
    working_policy: str,
    ttl_sec: float,
    cancel_attempts: int,
    cancel_sleep_sec: float,
    refresh_open_orders: Callable[[Any], None],
    print_mode: str,
    print_every: int,
    to_float: Callable[[Any], float],
) -> Tuple[dict, str, bool]:
    latest = dict(current_entry or {})
    status = normalize_status(str(latest.get("status", "")))
    filled_qty = to_float(latest.get("filled_qty", 0.0))
    remaining_qty = to_float(latest.get("remaining_qty", 0.0))
    outcome = classify_outcome(status, filled_qty, remaining_qty)
    if outcome != "working":
        return latest, outcome, False

    if ttl_sec > 0.0:
        print(f"[BROKER][WORKING_POLICY] ib_order_id={ib_order_id} policy={working_policy} ttl_sec={ttl_sec:.1f} waiting_before_decision=1")
        time.sleep(ttl_sec)
        latest = poll_order_status(
            app=app,
            ib_order_id=ib_order_id,
            initial_entry=latest,
            attempts=1,
            sleep_sec=0.0,
            label="ttl_checkpoint",
            refresh_open_orders=refresh_open_orders,
            print_mode=print_mode,
            print_every=print_every,
            to_float=to_float,
        )
        status = normalize_status(str(latest.get("status", "")))
        filled_qty = to_float(latest.get("filled_qty", 0.0))
        remaining_qty = to_float(latest.get("remaining_qty", 0.0))
        outcome = classify_outcome(status, filled_qty, remaining_qty)
        if outcome != "working":
            return latest, outcome, False

    if str(working_policy).strip().lower() == "carry":
        print(f"[BROKER][WORKING_POLICY] ib_order_id={ib_order_id} policy=carry decision=keep_working")
        return latest, "working_carry", False

    latest = cancel_and_poll_order(
        app=app,
        ib_order_id=ib_order_id,
        initial_entry=latest,
        cancel_attempts=cancel_attempts,
        cancel_sleep_sec=cancel_sleep_sec,
        refresh_open_orders=refresh_open_orders,
        print_mode=print_mode,
        print_every=print_every,
        to_float=to_float,
    )
    status = normalize_status(str(latest.get("status", "")))
    filled_qty = to_float(latest.get("filled_qty", 0.0))
    remaining_qty = to_float(latest.get("remaining_qty", 0.0))
    outcome = classify_outcome(status, filled_qty, remaining_qty)
    if outcome == "filled_now":
        return latest, "filled_now", True
    if outcome == "partial":
        return latest, "partial", True
    return latest, "cancelled_after_ttl", True


def clip_limit_from_202(
    *,
    prepared: PreparedOrder,
    current_raw: float | None,
    current_rounded: float | None,
    err202: BrokerErrorInfo | None,
    order_type: str,
    min_abs: float,
    clip_ticks: int,
    compute_limit_price: Callable[[PreparedOrder, float, float, float], Tuple[float | None, float | None]],
) -> Tuple[float | None, float | None]:
    if current_raw is None or current_rounded is None or err202 is None or str(order_type).upper() != "LMT":
        return None, None
    cutoff, market_px = parse_202_limits(err202.error_string)
    if cutoff is None:
        return None, None
    tick = max(float(prepared.min_tick), float(min_abs))
    pad = max(1, int(clip_ticks)) * tick
    side = prepared.order_side.upper()
    if side == "BUY":
        clipped = max(float(min_abs), cutoff - pad)
        if market_px is not None:
            clipped = min(clipped, market_px + pad)
        clipped = min(clipped, float(current_rounded))
    elif side == "SELL":
        clipped = cutoff + pad
        if market_px is not None:
            clipped = max(clipped, market_px - pad)
        clipped = max(clipped, float(current_rounded))
    else:
        return None, None
    prepared_clip = PreparedOrder(**{**prepared.__dict__, "price": float(clipped)})
    return compute_limit_price(prepared_clip, 0.0, 0.0, min_abs)
