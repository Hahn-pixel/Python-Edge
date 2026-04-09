from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd
from ibapi.contract import Contract
from ibapi.order import Order

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
for p in [ROOT, SRC_DIR]:
    ps = str(p)
    if ps not in sys.path:
        sys.path.insert(0, ps)

from python_edge.broker.ibkr_client import IBKRApp
from python_edge.broker.ibkr_models import BrokerErrorInfo, ConfigPaths, OrderIssue, PreparedOrder
from python_edge.broker.ibkr_pricing import normalize_min_tick
from python_edge.broker.ibkr_storage import append_or_replace_fills, duplicate_fill_entry, existing_duplicate_status, load_broker_log, save_broker_log, upsert_broker_log_entry

PAUSE_ON_EXIT = str(os.getenv("PAUSE_ON_EXIT", "auto")).strip().lower()

EXECUTION_ROOT = Path(os.getenv("EXECUTION_ROOT", "artifacts/execution_loop"))
CONFIG_NAMES = [x.strip() for x in str(os.getenv("CONFIG_NAMES", "optimal|aggressive")).split("|") if x.strip()]
BROKER_NAME = str(os.getenv("BROKER_NAME", "MEXEM")).strip() or "MEXEM"
BROKER_PLATFORM = str(os.getenv("BROKER_PLATFORM", "IBKR")).strip() or "IBKR"
BROKER_ACCOUNT_ID = str(os.getenv("BROKER_ACCOUNT_ID", "")).strip()

IB_HOST = str(os.getenv("IB_HOST", "127.0.0.1")).strip()
IB_PORT = int(os.getenv("IB_PORT", "4002"))
IB_CLIENT_ID = int(os.getenv("IB_CLIENT_ID", "41"))
IB_TIMEOUT_SEC = float(os.getenv("IB_TIMEOUT_SEC", "20.0"))
IB_ACCOUNT_CODE = str(os.getenv("IB_ACCOUNT_CODE", BROKER_ACCOUNT_ID)).strip()
IB_TIME_IN_FORCE = str(os.getenv("IB_TIME_IN_FORCE", "DAY")).strip().upper()
IB_OUTSIDE_RTH = str(os.getenv("IB_OUTSIDE_RTH", "1")).strip().lower() not in {"0", "false", "no", "off"}
IB_ALLOW_FRACTIONAL = str(os.getenv("IB_ALLOW_FRACTIONAL", "0")).strip().lower() not in {"0", "false", "no", "off"}
IB_EXCHANGE = str(os.getenv("IB_EXCHANGE", "SMART")).strip().upper() or "SMART"
IB_PRIMARY_EXCHANGE = str(os.getenv("IB_PRIMARY_EXCHANGE", "")).strip().upper()
IB_CURRENCY = str(os.getenv("IB_CURRENCY", "USD")).strip().upper() or "USD"
IB_SECURITY_TYPE = str(os.getenv("IB_SECURITY_TYPE", "STK")).strip().upper() or "STK"
IB_OPEN_ORDERS_TIMEOUT_SEC = float(os.getenv("IB_OPEN_ORDERS_TIMEOUT_SEC", str(IB_TIMEOUT_SEC)))
IB_POSITIONS_TIMEOUT_SEC = float(os.getenv("IB_POSITIONS_TIMEOUT_SEC", str(IB_TIMEOUT_SEC)))
IB_REFRESH_POSITIONS_ON_CONNECT = str(os.getenv("IB_REFRESH_POSITIONS_ON_CONNECT", "0")).strip().lower() not in {"0", "false", "no", "off"}
IB_REQUIRE_POSITIONS_ON_CONNECT = str(os.getenv("IB_REQUIRE_POSITIONS_ON_CONNECT", "0")).strip().lower() not in {"0", "false", "no", "off"}
RESET_BROKER_LOG = str(os.getenv("RESET_BROKER_LOG", "0")).strip().lower() not in {"0", "false", "no", "off"}
SYMBOL_MAP_FILE = str(os.getenv("BROKER_SYMBOL_MAP_FILE", "")).strip()
SYMBOL_MAP_JSON = str(os.getenv("BROKER_SYMBOL_MAP_JSON", "")).strip()
IB_FRACTIONAL_REJECT_CODES = [int(x.strip()) for x in str(os.getenv("IB_FRACTIONAL_REJECT_CODES", "10243|10247|10248|10249|10250")).split("|") if x.strip()]
IB_LMT_PRICE_MIN_ABS = float(os.getenv("IB_LMT_PRICE_MIN_ABS", "0.01"))
IB_POLL_VERBOSE = str(os.getenv("IB_POLL_VERBOSE", "changes")).strip().lower() or "changes"
IB_POLL_PRINT_EVERY = int(os.getenv("IB_POLL_PRINT_EVERY", "5"))
IB_CONTRACT_DETAILS_RETRIES = int(os.getenv("IB_CONTRACT_DETAILS_RETRIES", "3"))
IB_CONTRACT_DETAILS_RETRY_SLEEP_SEC = float(os.getenv("IB_CONTRACT_DETAILS_RETRY_SLEEP_SEC", "1.0"))
IB_ALLOW_SUBMIT_WITHOUT_CONTRACT_DETAILS = str(os.getenv("IB_ALLOW_SUBMIT_WITHOUT_CONTRACT_DETAILS", "1")).strip().lower() not in {"0", "false", "no", "off"}
IB_ALLOW_PRIMARY_EXCHANGE_FALLBACK = str(os.getenv("IB_ALLOW_PRIMARY_EXCHANGE_FALLBACK", "1")).strip().lower() not in {"0", "false", "no", "off"}
IB_RETRY_202_ENABLED = str(os.getenv("IB_RETRY_202_ENABLED", "1")).strip().lower() not in {"0", "false", "no", "off"}
IB_RETRY_202_CLIP_TICKS = int(os.getenv("IB_RETRY_202_CLIP_TICKS", "2"))
IB_REPRICE_ENABLED = str(os.getenv("IB_REPRICE_ENABLED", "1")).strip().lower() not in {"0", "false", "no", "off"}
IB_REPRICE_WAIT_SEC = float(os.getenv("IB_REPRICE_WAIT_SEC", "8.0"))
IB_REPRICE_STEPS_BPS = [float(x.strip()) for x in str(os.getenv("IB_REPRICE_STEPS_BPS", "15|10|5|0")).split("|") if x.strip()]
IB_REPRICE_FINAL_MODE = str(os.getenv("IB_REPRICE_FINAL_MODE", "marketable_lmt")).strip().lower() or "marketable_lmt"
IB_REPRICE_FINAL_MARKETABLE_BPS = float(os.getenv("IB_REPRICE_FINAL_MARKETABLE_BPS", "12.0"))
IB_REPRICE_MAX_DEVIATION_PCT = float(os.getenv("IB_REPRICE_MAX_DEVIATION_PCT", "1.25"))
IB_REPRICE_CANCEL_POLL_ATTEMPTS = int(os.getenv("IB_REPRICE_CANCEL_POLL_ATTEMPTS", "6"))
IB_REPRICE_CANCEL_POLL_SLEEP_SEC = float(os.getenv("IB_REPRICE_CANCEL_POLL_SLEEP_SEC", "1.0"))

REQUIRE_PRICE_HINT_SOURCE = str(os.getenv("REQUIRE_PRICE_HINT_SOURCE", "1")).strip().lower() not in {"0", "false", "no", "off"}
REQUIRE_QUOTE_TS = str(os.getenv("REQUIRE_QUOTE_TS", "1")).strip().lower() not in {"0", "false", "no", "off"}

REQUIRED_ORDER_COLUMNS = ["symbol", "order_side", "delta_shares"]

RE_202_AGGR = re.compile(r"at or more aggressive than\s+([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
RE_202_MARKET = re.compile(r"current market price of\s+([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)


def enable_line_buffering() -> None:
    for stream_name in ["stdout", "stderr"]:
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(line_buffering=True)
            except Exception:
                pass


def should_pause() -> bool:
    if PAUSE_ON_EXIT in {"0", "false", "no", "off"}:
        return False
    if PAUSE_ON_EXIT in {"1", "true", "yes", "on"}:
        return True
    stdin_obj = getattr(sys, "stdin", None)
    stdout_obj = getattr(sys, "stdout", None)
    return bool(stdin_obj and stdout_obj and hasattr(stdin_obj, "isatty") and hasattr(stdout_obj, "isatty") and stdin_obj.isatty() and stdout_obj.isatty())


def safe_exit(code: int) -> None:
    if should_pause():
        try:
            print("")
            print(f"[EXIT] code={code}")
            input("Press Enter to exit...")
        except Exception:
            pass
    raise SystemExit(code)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def must_exist(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def to_float(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def normalize_symbol(symbol: str) -> str:
    return str(symbol or "").strip().upper()


def normalize_order_side(side: str) -> str:
    out = str(side or "").strip().upper()
    if out not in {"BUY", "SELL", "HOLD"}:
        raise RuntimeError(f"Unsupported order_side: {side!r}")
    return out


def build_price_context_from_row(row_dict: Dict[str, Any], prepared_price: float) -> Dict[str, Any]:
    quote_ts = str(row_dict.get("quote_ts", row_dict.get("quote_ts_utc", "")) or "").strip()
    quote_provider = str(row_dict.get("quote_provider", "") or "").strip()
    quote_timeframe = str(row_dict.get("quote_timeframe", "") or "").strip()
    price_hint_source = str(row_dict.get("price_hint_source", "") or "").strip()
    if not price_hint_source and quote_provider:
        source_suffix = ""
        for key in ["ask", "bid", "mid", "last", "close_price"]:
            if to_float(row_dict.get(key, 0.0)) > 0.0 and abs(prepared_price - to_float(row_dict.get(key, 0.0))) <= 1e-9:
                source_suffix = key.replace("_price", "")
                break
        price_hint_source = f"{quote_provider}_{source_suffix}" if source_suffix else quote_provider
    model_price_reference = to_float(row_dict.get("model_price_reference", 0.0))
    deviation_pct = to_float(row_dict.get("price_deviation_vs_model", row_dict.get("price_deviation_pct", 0.0)))
    if deviation_pct <= 0.0 and model_price_reference > 0.0 and prepared_price > 0.0:
        deviation_pct = abs(prepared_price - model_price_reference) / model_price_reference * 100.0
    return {
        "price_hint_source": price_hint_source,
        "quote_ts": quote_ts,
        "quote_provider": quote_provider,
        "quote_timeframe": quote_timeframe,
        "quote_market_data_type": int(to_float(row_dict.get("quote_market_data_type", 0.0))),
        "model_price_reference": float(model_price_reference),
        "bid": float(to_float(row_dict.get("bid", row_dict.get("bid_price", 0.0)))),
        "ask": float(to_float(row_dict.get("ask", row_dict.get("ask_price", 0.0)))),
        "mid": float(to_float(row_dict.get("mid", row_dict.get("mid_price", 0.0)))),
        "last": float(to_float(row_dict.get("last", row_dict.get("last_price", 0.0)))),
        "close_price": float(to_float(row_dict.get("close_price", 0.0))),
        "spread_bps": float(to_float(row_dict.get("spread_bps", 0.0))),
        "price_deviation_vs_model": float(deviation_pct),
        "fallback_reason": str(row_dict.get("fallback_reason", "") or "").strip(),
    }


def validate_price_context(symbol: str, side: str, prepared_price: float, price_context: Dict[str, Any]) -> None:
    issues: List[str] = []
    if prepared_price <= 0.0:
        issues.append("price<=0")
    if REQUIRE_PRICE_HINT_SOURCE and not str(price_context.get("price_hint_source", "")).strip():
        issues.append("missing_price_hint_source")
    if REQUIRE_QUOTE_TS and not str(price_context.get("quote_ts", "")).strip():
        issues.append("missing_quote_ts")
    if issues:
        raise RuntimeError(
            f"Execution-ready price diagnostics missing for symbol={symbol} side={side}: {issues}. "
            f"price={prepared_price} price_context={json.dumps(price_context, ensure_ascii=False, sort_keys=True)}"
        )


def config_paths(config_name: str) -> ConfigPaths:
    execution_dir = EXECUTION_ROOT / config_name
    return ConfigPaths(
        name=config_name,
        execution_dir=execution_dir,
        orders_csv=execution_dir / "orders.csv",
        fills_csv=execution_dir / "fills.csv",
        broker_log_json=execution_dir / "broker_log.json",
    )


def load_symbol_map() -> Dict[str, str]:
    out: Dict[str, str] = {}
    if SYMBOL_MAP_JSON:
        parsed = json.loads(SYMBOL_MAP_JSON)
        if isinstance(parsed, dict):
            out.update({normalize_symbol(str(k)): normalize_symbol(str(v)) for k, v in parsed.items()})
    if SYMBOL_MAP_FILE:
        path = Path(SYMBOL_MAP_FILE)
        must_exist(path, "BROKER_SYMBOL_MAP_FILE")
        parsed2 = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(parsed2, dict):
            out.update({normalize_symbol(str(k)): normalize_symbol(str(v)) for k, v in parsed2.items()})
    return out


def load_orders_csv(orders_csv: Path) -> pd.DataFrame:
    must_exist(orders_csv, "orders.csv")
    df = pd.read_csv(orders_csv)
    missing = [c for c in REQUIRED_ORDER_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(f"Orders CSV missing required columns {missing}: {orders_csv}")
    df["symbol"] = df["symbol"].astype(str).map(normalize_symbol)
    df["order_side"] = df["order_side"].astype(str).map(normalize_order_side)
    df["delta_shares"] = pd.to_numeric(df["delta_shares"], errors="coerce").fillna(0.0)
    return df.copy()


def round_to_tick(price: float, tick: float, side: str) -> float:
    tick_safe = max(float(tick), float(IB_LMT_PRICE_MIN_ABS))
    px = max(float(IB_LMT_PRICE_MIN_ABS), float(price))
    if normalize_order_side(side) == "BUY":
        units = math.floor((px / tick_safe) + 1e-12)
    else:
        units = math.ceil((px / tick_safe) - 1e-12)
    out = float(units * tick_safe)
    return max(float(IB_LMT_PRICE_MIN_ABS), out)


def build_cap_price(anchor_price: float, side: str, max_deviation_pct: float, tick: float) -> float:
    side_norm = normalize_order_side(side)
    bump = abs(float(max_deviation_pct)) / 100.0
    if side_norm == "BUY":
        raw = anchor_price * (1.0 + bump)
    else:
        raw = anchor_price * max(0.0, 1.0 - bump)
    return round_to_tick(raw, tick, side_norm)


def ladder_prices(anchor_price: float, side: str, tick: float) -> List[float]:
    side_norm = normalize_order_side(side)
    out: List[float] = []
    for bps in IB_REPRICE_STEPS_BPS:
        step = abs(float(bps)) / 10000.0
        if side_norm == "BUY":
            raw = anchor_price * (1.0 + step)
        else:
            raw = anchor_price * max(0.0, 1.0 - step)
        out.append(round_to_tick(raw, tick, side_norm))
    deduped: List[float] = []
    for price in out:
        if not deduped or abs(price - deduped[-1]) > 1e-9:
            deduped.append(price)
    return deduped or [round_to_tick(anchor_price, tick, side_norm)]


def load_existing_open_orders_if_needed(app: IBKRApp) -> None:
    if not ENFORCE_OPEN_ORDER_DUP_GUARD:
        return
    try:
        app.reqOpenOrders()
        app.wait_for_open_orders(timeout_sec=IB_OPEN_ORDERS_TIMEOUT_SEC)
    except Exception as exc:
        print(f"[IB][OPEN_ORDERS][WARN] preload failed: {exc}")
        if ENFORCE_OPEN_ORDER_DUP_GUARD_STRICT:
            raise


def normalize_status(status: str) -> str:
    s = str(status or "").strip().lower()
    if not s:
        return "unknown"
    if s in {"submitted", "presubmitted", "pendingSubmit", "pendingsubmit", "pending_submit"}:
        return "submitted"
    if s in {"filled"}:
        return "filled"
    if s in {"partiallyfilled", "partial", "partial_fill"}:
        return "partial"
    if s in {"cancelled", "canceled", "api cancelled", "apicancelled"}:
        return "cancelled"
    return s


def find_matching_open_order(app: IBKRApp, prepared: PreparedOrder) -> dict | None:
    if not ENFORCE_OPEN_ORDER_DUP_GUARD:
        return None
    target_symbol = normalize_symbol(prepared.broker_symbol)
    target_side = normalize_order_side(prepared.order_side)
    target_qty = float(prepared.qty)
    statuses_live = {"submitted", "presubmitted", "pending_submit", "submitted", "partial", "pendingcancel", "api_pending"}
    for entry in reversed(list(app.orders_by_ib_id.values())):
        symbol = normalize_symbol(str(entry.get("symbol", "")))
        side = normalize_order_side(str(entry.get("action", entry.get("side", "")) or "HOLD"))
        status = normalize_status(str(entry.get("status", "")))
        remaining_qty = abs(to_float(entry.get("remaining_qty", entry.get("remaining", 0.0))))
        total_qty = abs(to_float(entry.get("total_qty", entry.get("qty", 0.0))))
        qty_ref = remaining_qty if remaining_qty > 0.0 else total_qty
        if symbol != target_symbol:
            continue
        if side != target_side:
            continue
        if status not in statuses_live:
            continue
        if qty_ref <= 0.0:
            continue
        if abs(qty_ref - target_qty) > ENFORCE_OPEN_ORDER_DUP_GUARD_QTY_TOL:
            continue
        return {
            "ib_order_id": int(entry.get("order_id", 0) or 0),
            "status": status,
            "symbol": symbol,
            "side": side,
            "qty": float(qty_ref),
            "filled_qty": float(to_float(entry.get("filled_qty", entry.get("filled", 0.0)))),
            "remaining_qty": float(remaining_qty if remaining_qty > 0.0 else max(0.0, qty_ref - to_float(entry.get("filled_qty", entry.get("filled", 0.0))))),
            "avg_fill_price": float(to_float(entry.get("avg_fill_price", entry.get("avgFillPrice", 0.0)))),
            "limit_price": float(to_float(entry.get("limit_price", entry.get("lmtPrice", 0.0)))),
            "perm_id": int(entry.get("perm_id", entry.get("permId", 0)) or 0),
            "client_order_ref": str(entry.get("order_ref", entry.get("orderRef", "")) or ""),
            "opened_ts_utc": str(entry.get("ts_utc", utc_now_iso()) or utc_now_iso()),
        }
    return None


def classify_outcome(status: str, filled_qty: float, remaining_qty: float) -> str:
    s = normalize_status(status)
    if filled_qty > 0.0 and remaining_qty <= 1e-12:
        return "filled_now"
    if filled_qty > 0.0 and remaining_qty > 1e-12:
        return "partial"
    if s in {"submitted", "pending_submit", "partial", "presubmitted"}:
        return "working"
    if s in {"cancelled"}:
        return "cancelled"
    return "unknown"


def detect_fractional_reject(errors: Sequence[dict]) -> BrokerErrorInfo | None:
    for err in reversed(list(errors)):
        try:
            code = int(err.get("errorCode", 0) or 0)
        except Exception:
            code = 0
        if code in IB_FRACTIONAL_REJECT_CODES:
            return BrokerErrorInfo(
                req_id=int(err.get("reqId", 0) or 0),
                error_code=code,
                error_string=str(err.get("errorString", "") or ""),
                advanced_order_reject_json=str(err.get("advancedOrderRejectJson", "") or ""),
                ts_utc=str(err.get("ts_utc", utc_now_iso()) or utc_now_iso()),
            )
    return None


def detect_submit_reject(errors: Sequence[dict]) -> BrokerErrorInfo | None:
    for err in reversed(list(errors)):
        try:
            code = int(err.get("errorCode", 0) or 0)
        except Exception:
            code = 0
        if code <= 0:
            continue
        return BrokerErrorInfo(
            req_id=int(err.get("reqId", 0) or 0),
            error_code=code,
            error_string=str(err.get("errorString", "") or ""),
            advanced_order_reject_json=str(err.get("advancedOrderRejectJson", "") or ""),
            ts_utc=str(err.get("ts_utc", utc_now_iso()) or utc_now_iso()),
        )
    return None


def build_submit_issue(kind: str, status: str, err: BrokerErrorInfo | None, message: str) -> OrderIssue:
    return OrderIssue(kind=kind, status=status, broker_error=err, message=message)


def build_order_for_price(prepared: PreparedOrder, limit_price: float, is_market: bool) -> Order:
    order = Order()
    order.action = normalize_order_side(prepared.order_side)
    order.totalQuantity = float(prepared.qty)
    order.tif = IB_TIME_IN_FORCE
    order.outsideRth = bool(IB_OUTSIDE_RTH)
    order.account = IB_ACCOUNT_CODE
    order.orderRef = prepared.client_tag

    if is_market:
        order.orderType = "MKT"
    else:
        order.orderType = "LMT"
        order.lmtPrice = float(limit_price)

    try:
        order.eTradeOnly = False
    except Exception:
        pass

    try:
        order.firmQuoteOnly = False
    except Exception:
        pass

    return order


def contract_for_symbol(symbol: str) -> Contract:
    c = Contract()
    c.symbol = normalize_symbol(symbol)
    c.secType = IB_SECURITY_TYPE
    c.exchange = IB_EXCHANGE
    c.currency = IB_CURRENCY
    if IB_PRIMARY_EXCHANGE:
        c.primaryExchange = IB_PRIMARY_EXCHANGE
    return c


def merged_request_debug(prepared: PreparedOrder, request_debug: Dict[str, Any]) -> Dict[str, Any]:
    base = {
        "broker_symbol": prepared.broker_symbol,
        "order_side": prepared.order_side,
        "qty": float(prepared.qty),
        "price_hint": float(prepared.price),
        "client_tag": prepared.client_tag,
    }
    return {**base, **(request_debug or {})}


def current_market_price_from_prepared(prepared: PreparedOrder) -> float:
    row = prepared.source_row if isinstance(prepared.source_row, dict) else {}
    candidates = [
        to_float(row.get("bid", row.get("bid_price", 0.0))),
        to_float(row.get("ask", row.get("ask_price", 0.0))),
        to_float(row.get("mid", row.get("mid_price", 0.0))),
        to_float(row.get("last", row.get("last_price", 0.0))),
        to_float(row.get("close_price", 0.0)),
        to_float(prepared.price),
    ]
    positives = [float(x) for x in candidates if float(x) > 0.0]
    if not positives:
        return 0.0
    if normalize_order_side(prepared.order_side) == "SELL":
        return max(positives)
    return min(positives)


def clip_limit_from_202(prepared: PreparedOrder, previous_limit: float, err: BrokerErrorInfo) -> float | None:
    msg = str(err.error_string or "")
    side = normalize_order_side(prepared.order_side)
    boundary = None
    m1 = RE_202_AGGR.search(msg)
    if m1:
        boundary = to_float(m1.group(1))
    market_from_202 = 0.0
    m2 = RE_202_MARKET.search(msg)
    if m2:
        market_from_202 = to_float(m2.group(1))
        if boundary is None:
            boundary = market_from_202
    if boundary is None or boundary <= 0.0:
        return None
    tick = max(float(prepared.min_tick), float(IB_LMT_PRICE_MIN_ABS))
    clip_ticks = max(1, int(IB_RETRY_202_CLIP_TICKS))
    current_market_price = max(market_from_202, current_market_price_from_prepared(prepared))
    if side == "BUY":
        clipped = boundary + tick * clip_ticks
        clipped = max(float(IB_LMT_PRICE_MIN_ABS), round_to_tick(clipped, tick, side))
        if clipped <= previous_limit:
            clipped = max(float(IB_LMT_PRICE_MIN_ABS), round_to_tick(previous_limit + tick, tick, side))
    else:
        boundary_clip = boundary - tick * clip_ticks
        market_clip = current_market_price - tick * clip_ticks if current_market_price > 0.0 else 0.0
        clipped_raw = max(boundary_clip, market_clip) if market_clip > 0.0 else boundary_clip
        clipped = max(float(IB_LMT_PRICE_MIN_ABS), round_to_tick(clipped_raw, tick, side))
    return clipped if clipped > 0.0 else None


def detect_deferred_399(errors: Sequence[dict]) -> BrokerErrorInfo | None:
    for err in reversed(list(errors)):
        if int(err.get("errorCode", 0) or 0) != 399:
            continue
        return BrokerErrorInfo(
            req_id=int(err.get("reqId", 0) or 0),
            error_code=int(err.get("errorCode", 0) or 0),
            error_string=str(err.get("errorString", "") or ""),
            advanced_order_reject_json=str(err.get("advancedOrderRejectJson", "") or ""),
            ts_utc=str(err.get("ts_utc", utc_now_iso()) or utc_now_iso()),
        )
    return None


def deferred_until_from_399(err: BrokerErrorInfo) -> str:
    msg = str(err.error_string or "")
    m = re.search(r"will be placed at\s+([0-9: ]+[AP]M [A-Z]{2,5})", msg, re.IGNORECASE)
    return str(m.group(1)).strip() if m else ""


def cancel_and_poll_order(app: IBKRApp, ib_order_id: int, last_entry: dict | None, reason: str) -> dict:
    app.cancelOrder(int(ib_order_id))
    entry = last_entry or {}
    for _ in range(max(1, IB_REPRICE_CANCEL_POLL_ATTEMPTS)):
        time.sleep(max(0.1, IB_REPRICE_CANCEL_POLL_SLEEP_SEC))
        entry = app.orders_by_ib_id.get(int(ib_order_id), {}) or entry or {}
        status = normalize_status(str(entry.get("status", "")))
        if status in {"cancelled", "filled"}:
            break
    entry = dict(entry or {})
    entry["cancel_reason"] = reason
    return entry


def wait_for_fill_progress(app: IBKRApp, prepared: PreparedOrder, ib_order_id: int, ib_entry: dict | None) -> dict:
    deadline = time.time() + max(0.0, IB_REPRICE_WAIT_SEC)
    last_printed = None
    poll_count = 0
    entry = dict(ib_entry or {})
    while time.time() < deadline:
        poll_count += 1
        time.sleep(1.0)
        latest = app.orders_by_ib_id.get(int(ib_order_id), {}) or entry
        if latest:
            entry = dict(latest)
        filled_qty = to_float(entry.get("filled_qty", 0.0))
        remaining_qty = to_float(entry.get("remaining_qty", max(0.0, prepared.qty - filled_qty)))
        status = normalize_status(str(entry.get("status", "")))
        should_print = False
        if IB_POLL_VERBOSE == "all":
            should_print = True
        elif IB_POLL_VERBOSE == "changes":
            current_key = (status, round(filled_qty, 8), round(remaining_qty, 8))
            if current_key != last_printed:
                should_print = True
                last_printed = current_key
        elif poll_count % max(1, IB_POLL_PRINT_EVERY) == 0:
            should_print = True
        if should_print:
            print(
                f"[IB][POLL] symbol={prepared.symbol} order_id={ib_order_id} poll={poll_count} status={status} filled_qty={filled_qty:.8f} remaining_qty={remaining_qty:.8f}"
            )
        outcome = classify_outcome(status, filled_qty, remaining_qty)
        if outcome in {"filled_now", "partial", "cancelled"}:
            break
    return entry


def wait_until_session_open_or_state_change(app: IBKRApp, prepared: PreparedOrder, ib_order_id: int, ib_entry: dict | None, err_399: BrokerErrorInfo) -> dict:
    _ = err_399
    deadline = time.time() + max(0.0, IB_TIMEOUT_SEC)
    entry = dict(ib_entry or {})
    while time.time() < deadline:
        time.sleep(1.0)
        latest = app.orders_by_ib_id.get(int(ib_order_id), {}) or entry
        if latest:
            entry = dict(latest)
        status = normalize_status(str(entry.get("status", "")))
        if status not in {"submitted", "presubmitted", "pending_submit"}:
            break
    return entry


def request_contract_details(app: IBKRApp, req_id: int, contract: Contract) -> Dict[str, Any]:
    app.done_contract_details[int(req_id)] = False
    app.contract_details.pop(int(req_id), None)
    app.reqContractDetails(int(req_id), contract)
    result = app.wait_for_contract_details(int(req_id), timeout_sec=IB_TIMEOUT_SEC)
    return result if isinstance(result, dict) else {}


def resolve_contract_metadata(app: IBKRApp, prepared: PreparedOrder, req_id_seed: int) -> Tuple[PreparedOrder, Contract, int, Dict[str, Any]]:
    contract = contract_for_symbol(prepared.broker_symbol)
    req_cursor = int(req_id_seed)
    contract_meta_debug: Dict[str, Any] = {"contract_details_mode": "", "contract_details_retries": 0, "contract_details_req_id": 0}
    last_exc: Exception | None = None

    for attempt in range(1, max(1, IB_CONTRACT_DETAILS_RETRIES) + 1):
        req_cursor += 1
        req_id = req_cursor
        contract_meta_debug["contract_details_retries"] = attempt
        contract_meta_debug["contract_details_req_id"] = req_id
        try:
            best = request_contract_details(app, req_id, contract)
            if best:
                raw_min_tick = to_float(best.get("minTick", 0.0))
                min_tick = normalize_min_tick(raw_min_tick, IB_LMT_PRICE_MIN_ABS)
                prepared2 = PreparedOrder(**{**prepared.__dict__, "min_tick": min_tick})
                contract_meta_debug.update({
                    "contract_details_mode": "resolved",
                    "contract_exchange": str(best.get("exchange", "")),
                    "contract_primary_exchange": str(best.get("primaryExchange", "")),
                    "contract_currency": str(best.get("currency", "")),
                    "contract_min_tick": float(min_tick),
                    "contract_raw_min_tick": float(raw_min_tick),
                    "contract_local_symbol": str(best.get("localSymbol", "")),
                })
                return prepared2, contract, req_cursor, contract_meta_debug
            raise RuntimeError(f"No contract details returned for symbol={prepared.symbol}")
        except Exception as exc:
            last_exc = exc
            print(f"[IB][CONTRACT_DETAILS][WARN] symbol={prepared.symbol} attempt={attempt}/{IB_CONTRACT_DETAILS_RETRIES} error={exc}")
            time.sleep(max(0.1, IB_CONTRACT_DETAILS_RETRY_SLEEP_SEC))

    if IB_ALLOW_PRIMARY_EXCHANGE_FALLBACK and not IB_PRIMARY_EXCHANGE:
        contract2 = contract_for_symbol(prepared.broker_symbol)
        contract2.primaryExchange = "NASDAQ"
        for attempt in range(1, max(1, IB_CONTRACT_DETAILS_RETRIES) + 1):
            req_cursor += 1
            req_id = req_cursor
            contract_meta_debug["contract_details_retries"] = attempt
            contract_meta_debug["contract_details_req_id"] = req_id
            try:
                best = request_contract_details(app, req_id, contract2)
                if best:
                    raw_min_tick = to_float(best.get("minTick", 0.0))
                    min_tick = normalize_min_tick(raw_min_tick, IB_LMT_PRICE_MIN_ABS)
                    prepared2 = PreparedOrder(**{**prepared.__dict__, "min_tick": min_tick})
                    contract_meta_debug.update({
                        "contract_details_mode": "primary_exchange_fallback",
                        "contract_exchange": str(best.get("exchange", "")),
                        "contract_primary_exchange": str(best.get("primaryExchange", "")),
                        "contract_currency": str(best.get("currency", "")),
                        "contract_min_tick": float(min_tick),
                        "contract_local_symbol": str(best.get("localSymbol", "")),
                    })
                    return prepared2, contract2, req_cursor, contract_meta_debug
                raise RuntimeError(f"No contract details returned for symbol={prepared.symbol} via fallback")
            except Exception as exc:
                last_exc = exc
                print(f"[IB][CONTRACT_DETAILS][FALLBACK_WARN] symbol={prepared.symbol} attempt={attempt}/{IB_CONTRACT_DETAILS_RETRIES} error={exc}")
                time.sleep(max(0.1, IB_CONTRACT_DETAILS_RETRY_SLEEP_SEC))

    if IB_ALLOW_SUBMIT_WITHOUT_CONTRACT_DETAILS:
        contract_meta_debug.update({"contract_details_mode": "submit_without_details", "contract_details_error": str(last_exc or "")})
        return prepared, contract, req_cursor, contract_meta_debug
    raise RuntimeError(f"Contract details resolution failed for symbol={prepared.symbol}: {last_exc}")


def select_live_orders(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.loc[df["order_side"].astype(str).str.upper() != "HOLD"].copy()
    out = out.loc[out["delta_shares"].abs() > 1e-12].copy()
    return out.reset_index(drop=True)


def prepare_orders(config_name: str, df: pd.DataFrame, symbol_map: Dict[str, str]) -> List[PreparedOrder]:
    prepared: List[PreparedOrder] = []
    for _, row in df.iterrows():
        row_dict = {str(k): row[k] for k in df.columns}
        symbol = normalize_symbol(str(row_dict.get("symbol", "")))
        order_side = normalize_order_side(str(row_dict.get("order_side", "HOLD")))
        raw_qty = abs(to_float(row_dict.get("delta_shares", 0.0)))
        if raw_qty <= 0.0:
            continue
        qty = float(raw_qty) if IB_ALLOW_FRACTIONAL else float(int(round(raw_qty)))
        if qty <= 0.0:
            continue
        broker_symbol = symbol_map.get(symbol, symbol)
        prepared_price = to_float(row_dict.get("price", 0.0))
        price_context = build_price_context_from_row(row_dict, prepared_price)
        validate_price_context(symbol, order_side, prepared_price, price_context)
        idempotency_key = hashlib.sha256(
            json.dumps(
                {
                    "config": config_name,
                    "date": str(row_dict.get("date", "")).strip(),
                    "symbol": symbol,
                    "broker_symbol": broker_symbol,
                    "order_side": order_side,
                    "delta_shares": round(to_float(row_dict.get("delta_shares", 0.0)), 8),
                    "order_notional": round(to_float(row_dict.get("order_notional", 0.0)), 8),
                    "target_weight": round(to_float(row_dict.get("target_weight", 0.0)), 8),
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
        prepared.append(
            PreparedOrder(
                config=config_name,
                order_date=str(row_dict.get("date", "")).strip(),
                symbol=symbol,
                broker_symbol=broker_symbol,
                order_side=order_side,
                qty=qty,
                price=prepared_price,
                order_notional=abs(to_float(row_dict.get("order_notional", 0.0))),
                target_weight=to_float(row_dict.get("target_weight", 0.0)),
                current_shares=to_float(row_dict.get("current_shares", 0.0)),
                target_shares=to_float(row_dict.get("target_shares", 0.0)),
                delta_shares=to_float(row_dict.get("delta_shares", 0.0)),
                source_row=row_dict,
                idempotency_key=idempotency_key,
                client_tag=f"pe-{idempotency_key[:24]}",
            )
        )
    return prepared


def normalize_broker_entry_status(entry: dict) -> str:
    return normalize_status(str(entry.get("status", "")))


def format_issue_message(issue: OrderIssue | None) -> str:
    if issue is None:
        return ""
    if issue.broker_error is None:
        return issue.message
    return f"{issue.message} | code={issue.broker_error.error_code} msg={issue.broker_error.error_string}"


def submit_one_order(app: IBKRApp, prepared: PreparedOrder, req_id_seed: int) -> Tuple[dict, OrderIssue | None, int, Dict[str, Any]]:
    prepared2, contract, req_id_seed2, contract_meta_debug = resolve_contract_metadata(app, prepared, req_id_seed)
    prepared = prepared2
    tick = max(float(prepared.min_tick), float(IB_LMT_PRICE_MIN_ABS))
    anchor_price = round_to_tick(float(prepared.price), tick, prepared.order_side)
    cap_price = build_cap_price(anchor_price, prepared.order_side, IB_REPRICE_MAX_DEVIATION_PCT, tick)
    prices = ladder_prices(anchor_price, prepared.order_side, tick) if IB_REPRICE_ENABLED else [anchor_price]
    if prices:
        if normalize_order_side(prepared.order_side) == "BUY":
            prices = [min(px, cap_price) for px in prices]
        else:
            prices = [max(px, cap_price) for px in prices]
    request_debug: Dict[str, Any] = {
        **contract_meta_debug,
        "anchor_price": float(anchor_price),
        "reprice_enabled": int(IB_REPRICE_ENABLED),
        "reprice_wait_sec": float(IB_REPRICE_WAIT_SEC),
        "reprice_steps_bps": [float(x) for x in IB_REPRICE_STEPS_BPS],
        "reprice_final_mode": IB_REPRICE_FINAL_MODE,
        "reprice_final_marketable_bps": float(IB_REPRICE_FINAL_MARKETABLE_BPS),
        "reprice_max_deviation_pct": float(IB_REPRICE_MAX_DEVIATION_PCT),
        "cap_price": float(cap_price),
        "open_order_dup_guard": int(ENFORCE_OPEN_ORDER_DUP_GUARD),
    }
    existing_open = find_matching_open_order(app, prepared)
    if existing_open is not None:
        request_debug.update({
            "open_order_dup_guard_triggered": 1,
            "existing_ib_order_id": existing_open.get("ib_order_id", ""),
            "existing_status": str(existing_open.get("status", "")),
        })
        issue = OrderIssue(kind="duplicate_open_order", status="working", broker_error=None, message="Matching live broker order already exists; submit skipped")
        return existing_open, issue, 0, request_debug
    request_debug["open_order_dup_guard_triggered"] = 0
    reprices_used = 0
    last_entry: dict | None = None
    price_idx = 0
    while price_idx < len(prices):
        limit_price = float(prices[price_idx])
        is_last_step = price_idx == len(prices) - 1
        step_mode = "marketable_lmt" if is_last_step and IB_REPRICE_FINAL_MODE == "marketable_lmt" else "passive_lmt"
        order = build_order_for_price(prepared, limit_price=limit_price, is_market=False)
        error_cursor_before_submit = len(app._errors)
        ib_order_id = app.allocate_order_id()
        order_payload = {
            "orderType": str(getattr(order, "orderType", "") or ""),
            "outsideRth": int(bool(getattr(order, "outsideRth", False))),
            "tif": str(getattr(order, "tif", "") or ""),
            "exchange": str(getattr(contract, "exchange", "") or ""),
            "primaryExchange": str(getattr(contract, "primaryExchange", "") or ""),
            "lmtPrice": float(getattr(order, "lmtPrice", 0.0) or 0.0),
            "totalQuantity": float(getattr(order, "totalQuantity", 0.0) or 0.0),
            "account": str(getattr(order, "account", "") or ""),
            "orderRef": str(getattr(order, "orderRef", "") or ""),
        }
        request_debug["submitted_order"] = order_payload
        print(
            f"[BROKER][LADDER][SUBMIT] symbol={prepared.symbol} step={price_idx + 1}/{len(prices)} mode={step_mode} limit_price={limit_price:.4f} cap_price={cap_price:.4f} order={json.dumps(order_payload, ensure_ascii=False, sort_keys=True)}"
        )
        app.placeOrder(ib_order_id, contract, order)
        ib_entry = app.wait_for_order_terminalish(ib_order_id, timeout_sec=IB_TIMEOUT_SEC)
        recent_errors = list(app._errors[error_cursor_before_submit:]) if len(app._errors) > error_cursor_before_submit else []
        deferred_399 = detect_deferred_399(recent_errors)
        if deferred_399 is not None:
            deferred_until = deferred_until_from_399(deferred_399)
            request_debug.update({
                "reprices_used": reprices_used,
                "final_step": price_idx + 1,
                "final_limit_price": float(limit_price),
                "final_mode": step_mode,
                "deferred_until": deferred_until,
                "ib_399_message": str(deferred_399.error_string),
            })
            final_entry = wait_until_session_open_or_state_change(app, prepared, ib_order_id, ib_entry, deferred_399)
            final_status = normalize_status(str(final_entry.get("status", "")))
            final_filled_qty = to_float(final_entry.get("filled_qty", 0.0))
            final_remaining_qty = to_float(final_entry.get("remaining_qty", max(0.0, prepared.qty - final_filled_qty)))
            final_outcome = classify_outcome(final_status, final_filled_qty, final_remaining_qty)
            if final_outcome in {"filled_now", "partial"}:
                return final_entry, None, reprices_used, request_debug
            issue = OrderIssue(
                kind="session_deferred_399",
                status="working_carry",
                broker_error=deferred_399,
                message=f"IB deferred order until {deferred_until}; leaving live at broker" if deferred_until else "IB deferred order until later session; leaving live at broker",
            )
            return final_entry, issue, reprices_used, request_debug
        ib_entry = wait_for_fill_progress(app, prepared, ib_order_id, ib_entry)
        filled_qty = to_float(ib_entry.get("filled_qty", 0.0))
        remaining_qty = to_float(ib_entry.get("remaining_qty", max(0.0, prepared.qty - filled_qty)))
        outcome = classify_outcome(ib_entry.get("status", ""), filled_qty, remaining_qty)
        retryable_202 = None
        if IB_RETRY_202_ENABLED and recent_errors:
            for err in reversed(recent_errors):
                if int(err.get("errorCode", 0) or 0) == 202:
                    retryable_202 = BrokerErrorInfo(
                        req_id=int(err.get("reqId", 0) or 0),
                        error_code=int(err.get("errorCode", 0) or 0),
                        error_string=str(err.get("errorString", "") or ""),
                        advanced_order_reject_json=str(err.get("advancedOrderRejectJson", "") or ""),
                        ts_utc=str(err.get("ts_utc", utc_now_iso()) or utc_now_iso()),
                    )
                    break
        if outcome in {"filled_now", "partial"}:
            request_debug.update({"reprices_used": reprices_used, "final_step": price_idx + 1, "final_limit_price": float(limit_price), "final_mode": step_mode})
            return ib_entry, None, reprices_used, request_debug
        if retryable_202 is not None:
            clipped = clip_limit_from_202(prepared, limit_price, retryable_202)
            if clipped is not None and abs(clipped - limit_price) > 1e-9:
                print(f"[BROKER][LADDER][202_OVERRIDE] symbol={prepared.symbol} step={price_idx + 1} next_limit_price={clipped:.4f} (broker guided)")
                prices = [float(clipped)]
                reprices_used += 1
                price_idx = 0
                continue
        if not is_last_step:
            cancel_and_poll_order(app, ib_order_id, ib_entry, f"reprice_step_{price_idx + 1}")
            reprices_used += 1
            price_idx += 1
            continue
        if IB_REPRICE_FINAL_MODE == "marketable_lmt":
            last_entry = cancel_and_poll_order(app, ib_order_id, ib_entry, "final_unfilled_cap")
            request_debug.update({"reprices_used": reprices_used, "final_step": price_idx + 1, "final_limit_price": float(limit_price), "final_mode": step_mode})
            issue = OrderIssue(kind="ladder_cap_cancelled", status="cancelled_unfilled_cap", broker_error=None, message="Final capped price was not filled; order cancelled")
            return last_entry, issue, reprices_used, request_debug
        request_debug.update({"reprices_used": reprices_used, "final_mode": "mkt_blocked_by_cap"})
        return last_entry, OrderIssue(kind="mkt_disallowed_by_cap", status="cancelled_unfilled_cap", broker_error=None, message="MKT final step disabled because hard deviation cap must be preserved"), reprices_used, request_debug
    request_debug.update({"reprices_used": reprices_used, "final_mode": "none"})
    return last_entry, OrderIssue(kind="ladder_failed", status="cancelled_unfilled_cap", broker_error=None, message="Order ladder exhausted without fill"), reprices_used, request_debug


def build_broker_log_entry(
    paths: ConfigPaths,
    prepared: PreparedOrder,
    broker_entry: dict,
    issue: OrderIssue | None,
    request_debug: Dict[str, Any],
    ib_order_id_fallback: int = 0,
) -> dict:
    filled_qty = to_float(broker_entry.get("filled_qty", 0.0))
    remaining_qty = to_float(broker_entry.get("remaining_qty", max(0.0, prepared.qty - filled_qty)))
    avg_fill_price = to_float(
        broker_entry.get("avg_fill_price", broker_entry.get("last_fill_price", 0.0))
    )
    status_raw = str(broker_entry.get("status", issue.status if issue else "submitted"))
    status_norm = normalize_status(status_raw)
    ts_now = utc_now_iso()

    broker_order_id_int = int(broker_entry.get("ib_order_id", 0) or 0)
    if broker_order_id_int <= 0:
        broker_order_id_int = int(ib_order_id_fallback or 0)

    perm_id_int = int(broker_entry.get("perm_id", 0) or 0)

    fill_notional = 0.0
    if filled_qty > 0.0 and avg_fill_price > 0.0:
        fill_notional = float(filled_qty * avg_fill_price)

    if status_norm == "filled":
        outcome = "filled_now"
    elif status_norm == "partial":
        outcome = "partial"
    elif status_norm in {"submitted", "presubmitted", "working", "working_carry"}:
        outcome = "working_carry" if issue and issue.status == "working_carry" else "working"
    elif status_norm == "cancelled":
        outcome = "cancelled"
    else:
        outcome = status_norm or "unknown"

    submitted_at = str(broker_entry.get("submitted_at", ts_now) or ts_now)
    filled_at = str(broker_entry.get("filled_at", ts_now if filled_qty > 0.0 else "") or "")

    response_obj = {
        "ib_order_id": broker_order_id_int,
        "perm_id": perm_id_int,
        "status": status_raw,
        "outcome": outcome,
        "fills": broker_entry.get("fills", []),
        "avg_fill_price": float(avg_fill_price),
        "last_fill_price": float(to_float(broker_entry.get("last_fill_price", avg_fill_price))),
    }

    if issue is not None:
        if issue.status == "working_carry":
            response_obj["deferred_message"] = format_issue_message(issue)
        else:
            response_obj["issue_kind"] = issue.kind
            response_obj["issue_message"] = format_issue_message(issue)
            if issue.broker_error is not None:
                response_obj["broker_error_code"] = int(issue.broker_error.error_code)
                response_obj["broker_error_string"] = str(issue.broker_error.error_string)

    entry = {
        "idempotency_key": str(prepared.idempotency_key),
        "client_order_id": str(prepared.client_tag),
        "broker_order_id": str(broker_order_id_int) if broker_order_id_int > 0 else "",
        "perm_id": perm_id_int,
        "config": str(prepared.config),
        "date": str(prepared.order_date),
        "symbol": str(prepared.symbol),
        "broker_symbol": str(prepared.broker_symbol),
        "side": str(prepared.order_side),
        "qty": float(prepared.qty),
        "filled_qty": float(filled_qty),
        "remaining_qty": float(remaining_qty),
        "price_hint": float(prepared.price),
        "filled_avg_price": float(avg_fill_price),
        "order_notional": float(prepared.order_notional),
        "fill_notional": float(fill_notional),
        "status": status_norm,
        "submitted_at": submitted_at,
        "filled_at": filled_at,
        "mode": "ibkr_gateway",
        "source_order_path": str(paths.orders_csv),
        "request": merged_request_debug(prepared, request_debug),
        "response": response_obj,
    }
    return entry


def fill_entry_from_broker_log_entry(log_entry: dict) -> dict:
    return {
        "idempotency_key": str(log_entry.get("idempotency_key", "")),
        "client_order_id": str(log_entry.get("client_order_id", "")),
        "broker_order_id": str(log_entry.get("broker_order_id", "")),
        "perm_id": int(log_entry.get("perm_id", 0) or 0),
        "config": str(log_entry.get("config", "")),
        "source_order_path": str(log_entry.get("source_order_path", "")),
        "date": str(log_entry.get("date", "")),
        "symbol": str(log_entry.get("symbol", "")),
        "broker_symbol": str(log_entry.get("broker_symbol", "")),
        "side": str(log_entry.get("side", "")),
        "qty": float(to_float(log_entry.get("qty", 0.0))),
        "filled_qty": float(to_float(log_entry.get("filled_qty", 0.0))),
        "remaining_qty": float(to_float(log_entry.get("remaining_qty", 0.0))),
        "price_hint": float(to_float(log_entry.get("price_hint", 0.0))),
        "filled_avg_price": float(to_float(log_entry.get("filled_avg_price", 0.0))),
        "order_notional": float(to_float(log_entry.get("order_notional", 0.0))),
        "fill_notional": float(to_float(log_entry.get("fill_notional", 0.0))),
        "status": str(log_entry.get("status", "")),
        "submitted_at": str(log_entry.get("submitted_at", "")),
        "filled_at": str(log_entry.get("filled_at", "")),
        "mode": str(log_entry.get("mode", "ibkr_gateway")),
    }


def submit_config(app: IBKRApp, config_name: str, symbol_map: Dict[str, str]) -> None:
    paths = config_paths(config_name)
    must_exist(paths.orders_csv, f"orders.csv for config={config_name}")
    if RESET_BROKER_LOG and paths.broker_log_json.exists():
        save_broker_log(paths.broker_log_json, [])
    broker_log = load_broker_log(
        paths.broker_log_json,
        config_name=config_name,
        broker_name=BROKER_NAME,
        broker_platform=BROKER_PLATFORM,
        broker_account_id=BROKER_ACCOUNT_ID,
        utc_now_iso=utc_now_iso,
        reset=RESET_BROKER_LOG,
    )
    df = load_orders_csv(paths.orders_csv)
    live_df = select_live_orders(df)
    prepared_orders = prepare_orders(config_name, live_df, symbol_map)
    print(f"[CFG] config={config_name} orders_total={len(df)} live_orders={len(live_df)} prepared_orders={len(prepared_orders)} orders_csv={paths.orders_csv}")
    req_id_seed = 1000
    sent = 0
    duplicate_skipped = 0
    errors = 0
    for prepared in prepared_orders:
        dup_status = existing_duplicate_status(broker_log, prepared.idempotency_key)
        if dup_status is not None:
            duplicate_skipped += 1
            duplicate_entry = duplicate_fill_entry(prepared, dup_status, paths.orders_csv, broker_log)
            append_or_replace_fills(paths.fills_csv, [duplicate_entry])
            print(f"[BROKER][{config_name}][DUPLICATE] symbol={prepared.symbol} side={prepared.order_side} qty={prepared.qty:.8f} status={dup_status}")
            continue
        try:
            broker_entry, issue, reprices_used, request_debug = submit_one_order(app, prepared, req_id_seed)
            req_id_seed += 100
            log_entry = build_broker_log_entry(
                paths,
                prepared,
                broker_entry,
                issue,
                request_debug,
                ib_order_id_fallback=int(broker_entry.get("ib_order_id", 0) or 0),
            )
            upsert_broker_log_entry(broker_log, log_entry, utc_now_iso)
            save_broker_log(paths.broker_log_json, broker_log, utc_now_iso)
            filled_qty = to_float(log_entry.get("filled_qty", 0.0))
            if filled_qty > 0.0:
                append_or_replace_fills(paths.fills_csv, [fill_entry_from_broker_log_entry(log_entry)])
            final_status = str(log_entry.get("status", issue.status if issue else "submitted"))
            print(
                f"[BROKER][{config_name}][SEND] symbol={prepared.symbol} broker_symbol={prepared.broker_symbol} side={prepared.order_side} qty={prepared.qty:.8f} status={final_status} ib_order_id={int(log_entry.get('ib_order_id', 0) or 0)}"
            )
            if issue is None or issue.status in {"filled", "partial", "working", "working_carry", "submitted"}:
                sent += 1
            else:
                errors += 1
        except Exception as exc:
            errors += 1
            traceback.print_exc()
            print(f"[BROKER][{config_name}][ERR] symbol={prepared.symbol} error={exc}")
    print(
        f"[BROKER][{config_name}][SUMMARY] sent={sent} duplicate_skipped={duplicate_skipped} errors={errors} fills_csv={paths.fills_csv} broker_log_json={paths.broker_log_json}"
    )


def connect_and_wait(app: IBKRApp) -> None:
    app.start_network_loop()
    next_id = app.wait_for_next_valid_id(timeout_sec=IB_TIMEOUT_SEC)
    managed_accounts = app.wait_for_managed_accounts(timeout_sec=IB_TIMEOUT_SEC)
    if IB_ACCOUNT_CODE:
        account_list = [x.strip() for x in managed_accounts.split(",") if x.strip()]
        if IB_ACCOUNT_CODE not in account_list:
            raise RuntimeError(f"Configured IB_ACCOUNT_CODE={IB_ACCOUNT_CODE} is not present in managed accounts: {managed_accounts}")
    print(f"[IB] ready next_valid_id={next_id} managed_accounts={managed_accounts}")


def refresh_open_orders(app: IBKRApp) -> None:
    app.done_open_orders = False
    app.reqOpenOrders()
    app.wait_until_open_orders_end(timeout_sec=IB_OPEN_ORDERS_TIMEOUT_SEC)


def refresh_positions(app: IBKRApp, required: bool) -> bool:
    try:
        app.position_rows = []
        app.done_positions = False
        app.reqPositions()
        app.wait_until_positions_end(timeout_sec=IB_POSITIONS_TIMEOUT_SEC)
        try:
            app.cancelPositions()
        except Exception:
            pass
        print(f"[IB] positions refresh complete rows={len(app.position_rows)}")
        return True
    except TimeoutError:
        try:
            app.cancelPositions()
        except Exception:
            pass
        if required:
            raise
        print(f"[IB][WARN] Timed out waiting for positionEnd after {IB_POSITIONS_TIMEOUT_SEC:.1f}s; continuing without initial position snapshot")
        return False


def bootstrap_connection() -> IBKRApp:
    app = IBKRApp(utc_now_iso=utc_now_iso, to_float=to_float)
    print(f"[IB] connecting host={IB_HOST} port={IB_PORT} client_id={IB_CLIENT_ID}")
    app.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID)
    connect_and_wait(app)
    refresh_open_orders(app)
    if IB_REFRESH_POSITIONS_ON_CONNECT or IB_REQUIRE_POSITIONS_ON_CONNECT:
        refresh_positions(app, required=IB_REQUIRE_POSITIONS_ON_CONNECT)
    return app


def teardown_connection(app: IBKRApp) -> None:
    try:
        if app.isConnected():
            app.disconnect()
    except Exception:
        pass
    time.sleep(0.25)


def main() -> int:
    enable_line_buffering()
    print(
        f"[ENV] broker={BROKER_NAME} platform={BROKER_PLATFORM} host={IB_HOST} port={IB_PORT} client_id={IB_CLIENT_ID} account={IB_ACCOUNT_CODE or BROKER_ACCOUNT_ID} execution_root={EXECUTION_ROOT} configs={'|'.join(CONFIG_NAMES)}"
    )
    symbol_map = load_symbol_map()
    app = None
    try:
        app = bootstrap_connection()
        for config_name in CONFIG_NAMES:
            submit_config(app, config_name, symbol_map)
        print("[FINAL] IBKR broker adapter complete")
        return 0
    finally:
        if app is not None:
            teardown_connection(app)


ENFORCE_OPEN_ORDER_DUP_GUARD = str(os.getenv("ENFORCE_OPEN_ORDER_DUP_GUARD", "1")).strip().lower() not in {"0", "false", "no", "off"}
ENFORCE_OPEN_ORDER_DUP_GUARD_STRICT = str(os.getenv("ENFORCE_OPEN_ORDER_DUP_GUARD_STRICT", "0")).strip().lower() not in {"0", "false", "no", "off"}
ENFORCE_OPEN_ORDER_DUP_GUARD_QTY_TOL = float(os.getenv("ENFORCE_OPEN_ORDER_DUP_GUARD_QTY_TOL", "0.000001"))


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 1
    safe_exit(int(rc))