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
from zoneinfo import ZoneInfo

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
from python_edge.broker.ibkr_storage import (
    append_or_replace_fills,
    duplicate_fill_entry,
    existing_duplicate_status,
    load_broker_log,
    save_broker_log,
    upsert_broker_log_entry,
)

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
IB_DEFERRED_WAIT_HEARTBEAT_SEC = float(os.getenv("IB_DEFERRED_WAIT_HEARTBEAT_SEC", "30.0"))
IB_DEFERRED_POST_OPEN_WAIT_SEC = float(os.getenv("IB_DEFERRED_POST_OPEN_WAIT_SEC", "30.0"))
IB_DEFERRED_WAIT_MODE = str(os.getenv("IB_DEFERRED_WAIT_MODE", "carry")).strip().lower() or "carry"
REQUIRE_PRICE_HINT_SOURCE = str(os.getenv("REQUIRE_PRICE_HINT_SOURCE", "1")).strip().lower() not in {"0", "false", "no", "off"}
REQUIRE_QUOTE_TS = str(os.getenv("REQUIRE_QUOTE_TS", "1")).strip().lower() not in {"0", "false", "no", "off"}
ENFORCE_OPEN_ORDER_DUP_GUARD = str(os.getenv("ENFORCE_OPEN_ORDER_DUP_GUARD", "1")).strip().lower() not in {"0", "false", "no", "off"}
REQUIRED_ORDER_COLUMNS = ["symbol", "order_side", "delta_shares"]
RE_202_AGGR = re.compile(r"at or more aggressive than\s+([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
RE_202_MARKET = re.compile(r"current market price of\s+([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
RE_399_DEFERRED = re.compile(r"will not be placed at the exchange until\s+(.+?)\.", re.IGNORECASE)


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
    return bool(
        stdin_obj
        and stdout_obj
        and hasattr(stdin_obj, "isatty")
        and hasattr(stdout_obj, "isatty")
        and stdin_obj.isatty()
        and stdout_obj.isatty()
    )


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
        raise RuntimeError(f"Execution-ready price diagnostics missing for symbol={symbol} side={side}: {issues}")


def load_symbol_map() -> Dict[str, str]:
    merged: Dict[str, str] = {}
    if SYMBOL_MAP_FILE:
        path = Path(SYMBOL_MAP_FILE)
        must_exist(path, "BROKER_SYMBOL_MAP_FILE")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError(f"BROKER_SYMBOL_MAP_FILE must contain a JSON object: {path}")
        for k, v in payload.items():
            merged[normalize_symbol(str(k))] = normalize_symbol(str(v))
    if SYMBOL_MAP_JSON:
        payload = json.loads(SYMBOL_MAP_JSON)
        if not isinstance(payload, dict):
            raise RuntimeError("BROKER_SYMBOL_MAP_JSON must be a JSON object")
        for k, v in payload.items():
            merged[normalize_symbol(str(k))] = normalize_symbol(str(v))
    return merged


def load_orders_df(orders_csv: Path) -> pd.DataFrame:
    must_exist(orders_csv, "Orders CSV")
    df = pd.read_csv(orders_csv)
    if df.empty:
        return df.copy()
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
        if not deduped or abs(price - deduped[-1]) > 1e-12:
            deduped.append(price)
    return deduped or [round_to_tick(anchor_price, tick, side_norm)]


def normalize_status(status: str) -> str:
    return str(status or "").strip().lower()


def classify_outcome(status: str, filled_qty: float, remaining_qty: float) -> str:
    st = normalize_status(status)
    if filled_qty > 0.0 and remaining_qty <= 1e-12:
        return "filled_now"
    if filled_qty > 0.0 and remaining_qty > 1e-12:
        return "partial"
    if st in {"presubmitted", "submitted", "pendingsubmit", "pendingcancel", "api_pending"}:
        return "working"
    if st in {"cancelled", "apicancelled"}:
        return "cancelled_after_ttl"
    if st in {"inactive"}:
        return "failed"
    return "unknown"


def merged_request_debug(prepared: PreparedOrder, request_debug: Dict[str, Any]) -> Dict[str, Any]:
    base = {
        "broker_symbol": prepared.broker_symbol,
        "order_side": prepared.order_side,
        "qty": float(prepared.qty),
        "price_hint": float(prepared.price),
        "client_tag": prepared.client_tag,
    }
    return {**base, **(request_debug or {})}


def clip_limit_from_202(prepared: PreparedOrder, previous_limit: float, err: BrokerErrorInfo) -> float | None:
    msg = str(err.error_string or "")
    side = normalize_order_side(prepared.order_side)
    boundary = None
    m1 = RE_202_AGGR.search(msg)
    if m1:
        boundary = to_float(m1.group(1))
    if boundary is None:
        m2 = RE_202_MARKET.search(msg)
        if m2:
            boundary = to_float(m2.group(1))
    if boundary is None or boundary <= 0.0:
        return None
    tick = max(float(prepared.min_tick), float(IB_LMT_PRICE_MIN_ABS))
    if side == "BUY":
        clipped = boundary + tick * max(1, int(IB_RETRY_202_CLIP_TICKS))
        clipped = max(float(IB_LMT_PRICE_MIN_ABS), round_to_tick(clipped, tick, side))
        if clipped <= previous_limit:
            clipped = max(float(IB_LMT_PRICE_MIN_ABS), round_to_tick(previous_limit + tick, tick, side))
    else:
        clipped = boundary - tick * max(1, int(IB_RETRY_202_CLIP_TICKS))
        clipped = max(float(IB_LMT_PRICE_MIN_ABS), round_to_tick(clipped, tick, side))
        if clipped >= previous_limit:
            clipped = max(float(IB_LMT_PRICE_MIN_ABS), round_to_tick(previous_limit - tick, tick, side))
    return clipped if clipped > 0.0 else None


def detect_deferred_399(errors: Sequence[dict]) -> BrokerErrorInfo | None:
    for err in reversed(list(errors)):
        if int(err.get("errorCode", 0) or 0) != 399:
            continue
        msg = str(err.get("errorString", "") or "")
        if RE_399_DEFERRED.search(msg):
            return BrokerErrorInfo(
                req_id=int(err.get("reqId", 0) or 0),
                error_code=int(err.get("errorCode", 0) or 0),
                error_string=msg,
                advanced_order_reject_json=str(err.get("advancedOrderRejectJson", "") or ""),
                ts_utc=str(err.get("ts_utc", utc_now_iso()) or utc_now_iso()),
            )
    return None


def deferred_until_from_399(err: BrokerErrorInfo | None) -> str:
    if err is None:
        return ""
    m = RE_399_DEFERRED.search(str(err.error_string or ""))
    return str(m.group(1)).strip() if m else ""


def parse_ib_deferred_until(err: BrokerErrorInfo | None) -> datetime | None:
    raw = deferred_until_from_399(err)
    if not raw:
        return None
    parts = raw.rsplit(" ", 1)
    if len(parts) != 2:
        return None
    dt_part, tz_part = parts[0].strip(), parts[1].strip()
    try:
        naive = datetime.strptime(dt_part, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None
    tz_map = {
        "US/Eastern": "America/New_York",
        "America/New_York": "America/New_York",
        "EST": "America/New_York",
        "EDT": "America/New_York",
        "UTC": "UTC",
    }
    tz_name = tz_map.get(tz_part, tz_part)
    try:
        tzinfo = ZoneInfo(tz_name)
    except Exception:
        tzinfo = timezone.utc
    return naive.replace(tzinfo=tzinfo)


def wait_until_session_open_or_state_change(app: IBKRApp, prepared: PreparedOrder, ib_order_id: int, ib_entry: dict, err: BrokerErrorInfo | None) -> dict:
    deferred_dt = parse_ib_deferred_until(err)
    if deferred_dt is None:
        return ib_entry

    if IB_DEFERRED_WAIT_MODE != "block":
        print(f"[BROKER][DEFERRED_CARRY] ib_order_id={ib_order_id} deferred_until={deferred_dt.isoformat()} mode={IB_DEFERRED_WAIT_MODE}")
        return latest_ib_entry(app, ib_order_id, ib_entry)

    heartbeat = max(1.0, float(IB_DEFERRED_WAIT_HEARTBEAT_SEC))
    while True:
        latest = latest_ib_entry(app, ib_order_id, ib_entry)
        status = normalize_status(str(latest.get("status", "")))
        filled_qty = to_float(latest.get("filled_qty", 0.0))
        remaining_qty = to_float(latest.get("remaining_qty", max(0.0, prepared.qty - filled_qty)))
        outcome = classify_outcome(status, filled_qty, remaining_qty)
        if outcome in {"filled_now", "partial", "failed", "cancelled_after_ttl"}:
            return latest
        now_utc = datetime.now(timezone.utc)
        deferred_utc = deferred_dt.astimezone(timezone.utc)
        remaining_sec = (deferred_utc - now_utc).total_seconds()
        if remaining_sec <= 0:
            break
        sleep_sec = min(heartbeat, max(0.5, remaining_sec))
        print(
            f"[BROKER][DEFERRED_WAIT] ib_order_id={ib_order_id} status={status} deferred_until={deferred_dt.isoformat()} remaining_sec={remaining_sec:.1f}"
        )
        time.sleep(sleep_sec)
        ib_entry = latest

    end_time = time.time() + max(1.0, float(IB_DEFERRED_POST_OPEN_WAIT_SEC))
    latest = ib_entry
    while time.time() < end_time:
        time.sleep(max(0.25, IB_REPRICE_CANCEL_POLL_SLEEP_SEC))
        latest = latest_ib_entry(app, ib_order_id, latest)
        status = normalize_status(str(latest.get("status", "")))
        filled_qty = to_float(latest.get("filled_qty", 0.0))
        remaining_qty = to_float(latest.get("remaining_qty", max(0.0, prepared.qty - filled_qty)))
        outcome = classify_outcome(status, filled_qty, remaining_qty)
        print(
            f"[BROKER][DEFERRED_POST_OPEN] ib_order_id={ib_order_id} status={status} filled_qty={filled_qty:.8f} remaining_qty={remaining_qty:.8f} outcome={outcome}"
        )
        if outcome in {"filled_now", "partial", "failed", "cancelled_after_ttl"}:
            return latest
    return latest


def connect_and_wait(app: IBKRApp) -> None:
    app.start_network_loop()
    next_id = app.wait_for_next_valid_id(timeout_sec=IB_TIMEOUT_SEC)
    managed_accounts = app.wait_for_managed_accounts(timeout_sec=IB_TIMEOUT_SEC)
    if IB_ACCOUNT_CODE:
        account_list = [x.strip() for x in managed_accounts.split(",") if x.strip()]
        if IB_ACCOUNT_CODE not in account_list:
            raise RuntimeError(f"IB_ACCOUNT_CODE={IB_ACCOUNT_CODE!r} not present in managed accounts: {managed_accounts!r}")
    print(f"[IB] connected next_valid_id={next_id} managed_accounts={managed_accounts}")


def contract_for_symbol(symbol: str) -> Contract:
    contract = Contract()
    contract.symbol = normalize_symbol(symbol)
    contract.secType = IB_SECURITY_TYPE
    contract.exchange = IB_EXCHANGE
    contract.currency = IB_CURRENCY
    if IB_PRIMARY_EXCHANGE:
        contract.primaryExchange = IB_PRIMARY_EXCHANGE
    return contract


def ib_cancel_order(app: IBKRApp, ib_order_id: int) -> None:
    last_exc: Exception | None = None
    for args in [(int(ib_order_id),), (int(ib_order_id), "")]:
        try:
            app.cancelOrder(*args)
            return
        except TypeError as exc:
            last_exc = exc
            continue
        except Exception:
            raise
    if last_exc is not None:
        raise last_exc


def cancel_and_poll_order(app: IBKRApp, ib_order_id: int, ib_entry: dict, cancel_reason: str) -> dict:
    ib_cancel_order(app, int(ib_order_id))
    final_entry = ib_entry
    for _ in range(max(1, IB_REPRICE_CANCEL_POLL_ATTEMPTS)):
        time.sleep(max(0.1, IB_REPRICE_CANCEL_POLL_SLEEP_SEC))
        final_entry = latest_ib_entry(app, ib_order_id, final_entry)
        status = normalize_status(str(final_entry.get("status", "")))
        if status in {"cancelled", "apicancelled", "filled", "inactive"}:
            final_entry = {**final_entry, "cancel_reason": cancel_reason}
            return final_entry
    return {**final_entry, "cancel_reason": cancel_reason}


def wait_for_fill_progress(app: IBKRApp, prepared: PreparedOrder, ib_order_id: int, initial_entry: dict) -> dict:
    total_attempts = max(1, int(math.ceil(IB_REPRICE_WAIT_SEC / max(0.25, IB_REPRICE_CANCEL_POLL_SLEEP_SEC))))
    last_signature = None
    entry = initial_entry
    mode = IB_POLL_VERBOSE
    every_n = max(1, IB_POLL_PRINT_EVERY)
    for poll_idx in range(total_attempts):
        time.sleep(max(0.25, IB_REPRICE_CANCEL_POLL_SLEEP_SEC))
        entry = latest_ib_entry(app, ib_order_id, entry)
        filled_qty = to_float(entry.get("filled_qty", 0.0))
        remaining_qty = to_float(entry.get("remaining_qty", max(0.0, prepared.qty - filled_qty)))
        status = normalize_status(str(entry.get("status", "")))
        outcome = classify_outcome(status, filled_qty, remaining_qty)
        signature = (status, round(filled_qty, 8), round(remaining_qty, 8), round(to_float(entry.get("avg_fill_price", 0.0)), 8))
        should_print = (
            mode == "all"
            or (mode == "none" and (poll_idx == total_attempts - 1 or outcome in {"filled_now", "partial", "failed"}))
            or (
                mode != "none"
                and (
                    poll_idx == 0
                    or poll_idx == total_attempts - 1
                    or signature != last_signature
                    or ((poll_idx + 1) % every_n == 0)
                    or outcome in {"filled_now", "partial", "failed"}
                )
            )
        )
        if should_print:
            print(
                f"[BROKER][POLL] ib_order_id={ib_order_id} step={poll_idx + 1}/{total_attempts} status={status} filled_qty={filled_qty:.8f} remaining_qty={remaining_qty:.8f} avg_fill_price={to_float(entry.get('avg_fill_price', 0.0)):.4f} outcome={outcome}"
            )
        last_signature = signature
        if outcome in {"filled_now", "partial", "failed"}:
            return entry
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


def build_order_for_price(prepared: PreparedOrder, limit_price: float | None, is_market: bool) -> Order:
    order = Order()
    order.action = prepared.order_side
    order.orderType = "MKT" if is_market else "LMT"
    order.totalQuantity = float(prepared.qty)
    order.tif = IB_TIME_IN_FORCE
    order.outsideRth = bool(IB_OUTSIDE_RTH)
    if IB_ACCOUNT_CODE:
        order.account = IB_ACCOUNT_CODE
    order.orderRef = prepared.client_tag
    order.transmit = True
    order.eTradeOnly = False
    order.firmQuoteOnly = False
    if not is_market:
        if limit_price is None or limit_price <= 0.0:
            raise RuntimeError("LMT order requires positive limit price")
        order.lmtPrice = float(limit_price)
    return order


def refresh_open_orders(app: IBKRApp) -> None:
    app.done_open_orders = False
    app.reqOpenOrders()
    app.wait_until_open_orders_end(timeout_sec=IB_OPEN_ORDERS_TIMEOUT_SEC)


def latest_ib_entry(app: IBKRApp, ib_order_id: int, default_entry: dict) -> dict:
    latest = app.orders_by_ib_id.get(int(ib_order_id), default_entry)
    return latest if isinstance(latest, dict) else default_entry


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


def entry_from_ib(prepared: PreparedOrder, ib_entry: dict, source_path: Path, request_debug: Dict[str, Any], outcome_override: str) -> dict:
    status = normalize_status(str(ib_entry.get("status", ""))) or "unknown"
    filled_qty = to_float(ib_entry.get("filled_qty", 0.0))
    remaining_qty = to_float(ib_entry.get("remaining_qty", max(0.0, prepared.qty - filled_qty)))
    avg_fill_price = to_float(ib_entry.get("avg_fill_price", prepared.price)) or float(prepared.price)
    outcome = str(outcome_override or classify_outcome(status, filled_qty, remaining_qty))
    price_context = build_price_context_from_row(prepared.source_row, prepared.price)
    request_debug_full = merged_request_debug(prepared, request_debug)
    return {
        "idempotency_key": prepared.idempotency_key,
        "client_order_id": prepared.client_tag,
        "broker_order_id": str(ib_entry.get("ib_order_id", "")),
        "perm_id": int(ib_entry.get("perm_id", 0) or 0),
        "config": prepared.config,
        "source_order_path": str(source_path),
        "date": prepared.order_date,
        "symbol": prepared.symbol,
        "broker_symbol": prepared.broker_symbol,
        "side": prepared.order_side,
        "qty": float(prepared.qty),
        "filled_qty": float(filled_qty),
        "remaining_qty": float(remaining_qty),
        "price_hint": float(prepared.price),
        "price_hint_source": price_context.get("price_hint_source", ""),
        "quote_ts": price_context.get("quote_ts", ""),
        "quote_provider": price_context.get("quote_provider", ""),
        "quote_timeframe": price_context.get("quote_timeframe", ""),
        "model_price_reference": float(price_context.get("model_price_reference", 0.0)),
        "bid": float(price_context.get("bid", 0.0)),
        "ask": float(price_context.get("ask", 0.0)),
        "mid": float(price_context.get("mid", 0.0)),
        "last": float(price_context.get("last", 0.0)),
        "close_price": float(price_context.get("close_price", 0.0)),
        "spread_bps": float(price_context.get("spread_bps", 0.0)),
        "price_deviation_vs_model": float(price_context.get("price_deviation_vs_model", 0.0)),
        "fallback_reason": price_context.get("fallback_reason", ""),
        "filled_avg_price": float(avg_fill_price),
        "order_notional": float(prepared.order_notional),
        "fill_notional": abs(float(filled_qty * avg_fill_price)),
        "status": status,
        "submitted_at": str(ib_entry.get("submitted_at_utc", utc_now_iso())),
        "filled_at": utc_now_iso() if outcome in {"filled_now", "partial"} else "",
        "mode": "ibkr_gateway",
        "request": request_debug_full,
        "response": {
            "ib_order_id": ib_entry.get("ib_order_id", ""),
            "perm_id": ib_entry.get("perm_id", 0),
            "status": ib_entry.get("status", ""),
            "avg_fill_price": ib_entry.get("avg_fill_price", 0.0),
            "last_fill_price": ib_entry.get("last_fill_price", 0.0),
            "fills": ib_entry.get("fills", []),
            "outcome": outcome,
        },
        "outcome": outcome,
    }


def build_error_entry(prepared: PreparedOrder, paths: ConfigPaths, issue: OrderIssue, request_debug: Dict[str, Any]) -> dict:
    price_context = build_price_context_from_row(prepared.source_row, prepared.price)
    request_debug_full = merged_request_debug(prepared, request_debug)
    return {
        "idempotency_key": prepared.idempotency_key,
        "client_order_id": prepared.client_tag,
        "broker_order_id": "",
        "perm_id": 0,
        "config": prepared.config,
        "source_order_path": str(paths.orders_csv),
        "date": prepared.order_date,
        "symbol": prepared.symbol,
        "broker_symbol": prepared.broker_symbol,
        "side": prepared.order_side,
        "qty": float(prepared.qty),
        "filled_qty": 0.0,
        "remaining_qty": float(prepared.qty),
        "price_hint": float(prepared.price),
        "price_hint_source": price_context.get("price_hint_source", ""),
        "quote_ts": price_context.get("quote_ts", ""),
        "quote_provider": price_context.get("quote_provider", ""),
        "quote_timeframe": price_context.get("quote_timeframe", ""),
        "model_price_reference": float(price_context.get("model_price_reference", 0.0)),
        "bid": float(price_context.get("bid", 0.0)),
        "ask": float(price_context.get("ask", 0.0)),
        "mid": float(price_context.get("mid", 0.0)),
        "last": float(price_context.get("last", 0.0)),
        "close_price": float(price_context.get("close_price", 0.0)),
        "spread_bps": float(price_context.get("spread_bps", 0.0)),
        "price_deviation_vs_model": float(price_context.get("price_deviation_vs_model", 0.0)),
        "fallback_reason": price_context.get("fallback_reason", ""),
        "filled_avg_price": 0.0,
        "order_notional": float(prepared.order_notional),
        "fill_notional": 0.0,
        "status": issue.status,
        "submitted_at": utc_now_iso(),
        "filled_at": "",
        "mode": "ibkr_gateway",
        "request": request_debug_full,
        "response": {
            "error": issue.message,
            "error_code": int(issue.broker_error.error_code) if issue.broker_error else 0,
            "error_string": str(issue.broker_error.error_string) if issue.broker_error else "",
            "outcome": "failed",
        },
        "outcome": "failed",
    }


def find_matching_open_order(app: IBKRApp, prepared: PreparedOrder) -> dict | None:
    if not ENFORCE_OPEN_ORDER_DUP_GUARD:
        return None
    refresh_open_orders(app)
    expected_symbol = normalize_symbol(prepared.broker_symbol)
    expected_side = normalize_order_side(prepared.order_side)
    expected_qty = float(prepared.qty)
    expected_ref = str(prepared.client_tag)
    for entry in list(app.orders_by_ib_id.values()):
        if not isinstance(entry, dict):
            continue
        status = normalize_status(str(entry.get("status", "")))
        if status not in {"presubmitted", "submitted", "pendingsubmit", "pendingcancel", "api_pending"}:
            continue
        order_ref = str(entry.get("orderRef", entry.get("order_ref", "")) or "")
        symbol = normalize_symbol(str(entry.get("symbol", entry.get("local_symbol", "")) or ""))
        side = normalize_order_side(str(entry.get("side", entry.get("action", prepared.order_side)) or prepared.order_side))
        qty = abs(to_float(entry.get("qty", entry.get("total_quantity", prepared.qty))))
        same_ref = bool(order_ref and order_ref == expected_ref)
        same_shape = symbol == expected_symbol and side == expected_side and abs(qty - expected_qty) <= 1e-8
        if same_ref or same_shape:
            return entry
    return None


def run_ladder_order(app: IBKRApp, prepared: PreparedOrder, contract: Contract, contract_meta_debug: Dict[str, Any]) -> Tuple[dict | None, OrderIssue | None, int, Dict[str, Any]]:
    anchor_price = max(float(IB_LMT_PRICE_MIN_ABS), float(prepared.price))
    tick = max(float(prepared.min_tick), float(IB_LMT_PRICE_MIN_ABS))
    prices = ladder_prices(anchor_price, prepared.order_side, tick)
    cap_price = build_cap_price(anchor_price, prepared.order_side, IB_REPRICE_MAX_DEVIATION_PCT, tick)
    request_debug: Dict[str, Any] = {
        "account": IB_ACCOUNT_CODE,
        "host": IB_HOST,
        "port": IB_PORT,
        "client_id": IB_CLIENT_ID,
        "tif": IB_TIME_IN_FORCE,
        "outside_rth": int(IB_OUTSIDE_RTH),
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
    for step_idx, limit_price in enumerate(prices):
        is_last_step = step_idx == len(prices) - 1
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
            f"[BROKER][LADDER][SUBMIT] symbol={prepared.symbol} step={step_idx + 1}/{len(prices)} mode={step_mode} limit_price={limit_price:.4f} cap_price={cap_price:.4f} order={json.dumps(order_payload, ensure_ascii=False, sort_keys=True)}"
        )
        app.placeOrder(ib_order_id, contract, order)
        ib_entry = app.wait_for_order_terminalish(ib_order_id, timeout_sec=IB_TIMEOUT_SEC)
        recent_errors = list(app._errors[error_cursor_before_submit:]) if len(app._errors) > error_cursor_before_submit else []
        deferred_399 = detect_deferred_399(recent_errors)
        if deferred_399 is not None:
            deferred_until = deferred_until_from_399(deferred_399)
            request_debug.update({
                "reprices_used": reprices_used,
                "final_step": step_idx + 1,
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
            request_debug.update({"reprices_used": reprices_used, "final_step": step_idx + 1, "final_limit_price": float(limit_price), "final_mode": step_mode})
            return ib_entry, None, reprices_used, request_debug
        if retryable_202 is not None:
            clipped = clip_limit_from_202(prepared, limit_price, retryable_202)
            if clipped is not None and abs(clipped - limit_price) > 1e-9:
                print(f"[BROKER][LADDER][202_OVERRIDE] symbol={prepared.symbol} step={step_idx + 1} next_limit_price={clipped:.4f} (broker guided)")
                prices = prices[: step_idx + 1] + [clipped]
                reprices_used += 1
                continue
        if not is_last_step:
            cancel_and_poll_order(app, ib_order_id, ib_entry, f"reprice_step_{step_idx + 1}")
            reprices_used += 1
            continue
        if IB_REPRICE_FINAL_MODE == "marketable_lmt":
            last_entry = cancel_and_poll_order(app, ib_order_id, ib_entry, "final_unfilled_cap")
            request_debug.update({"reprices_used": reprices_used, "final_step": step_idx + 1, "final_limit_price": float(limit_price), "final_mode": step_mode})
            issue = OrderIssue(kind="ladder_cap_cancelled", status="cancelled_unfilled_cap", broker_error=None, message="Final capped price was not filled; order cancelled")
            return last_entry, issue, reprices_used, request_debug
        request_debug.update({"reprices_used": reprices_used, "final_mode": "mkt_blocked_by_cap"})
        return last_entry, OrderIssue(kind="mkt_disallowed_by_cap", status="cancelled_unfilled_cap", broker_error=None, message="MKT final step disabled because hard deviation cap must be preserved"), reprices_used, request_debug
    request_debug.update({"reprices_used": reprices_used, "final_mode": "none"})
    return last_entry, OrderIssue(kind="ladder_failed", status="cancelled_unfilled_cap", broker_error=None, message="Order ladder exhausted without fill"), reprices_used, request_debug


def run_one_config(app: IBKRApp, paths: ConfigPaths, symbol_map: Dict[str, str], req_id_seed: int) -> int:
    print(f"[BROKER][{paths.name}] orders={paths.orders_csv}")
    paths.execution_dir.mkdir(parents=True, exist_ok=True)
    broker_log = load_broker_log(paths.broker_log_json, paths.name, BROKER_NAME, BROKER_PLATFORM, BROKER_ACCOUNT_ID, utc_now_iso, RESET_BROKER_LOG)
    orders_df = load_orders_df(paths.orders_csv)
    prepared_orders = prepare_orders(paths.name, select_live_orders(orders_df), symbol_map)
    fill_entries: List[dict] = []
    sent_count = dup_count = err_count = 0
    outcome_counters: Dict[str, int] = {
        "filled_now": 0,
        "partial": 0,
        "working": 0,
        "working_carry": 0,
        "cancelled_after_ttl": 0,
        "cancelled_unfilled_cap": 0,
        "deferred_session_closed": 0,
        "failed": 0,
        "unknown": 0,
    }
    req_cursor = int(req_id_seed)

    for prepared0 in prepared_orders:
        duplicate_status = existing_duplicate_status(broker_log, prepared0.idempotency_key)
        if duplicate_status is not None:
            dup_count += 1
            price_context = build_price_context_from_row(prepared0.source_row, prepared0.price)
            duplicate_entry = duplicate_fill_entry(prepared0, duplicate_status, paths.orders_csv, broker_log)
            duplicate_entry = {
                **duplicate_entry,
                **price_context,
                "price_hint": float(prepared0.price),
                "price_hint_source": price_context.get("price_hint_source", ""),
                "quote_ts": price_context.get("quote_ts", ""),
            }
            fill_entries.append(duplicate_entry)
            print(f"[BROKER][{paths.name}][DUPLICATE] symbol={prepared0.symbol} side={prepared0.order_side} qty={prepared0.qty:.8f} price_hint_source={duplicate_entry.get('price_hint_source', '')} quote_ts={duplicate_entry.get('quote_ts', '')}")
            continue
        try:
            prepared, contract, req_cursor, contract_meta_debug = resolve_contract_metadata(app, prepared0, req_cursor)
            request_debug: Dict[str, Any] = {}
            ib_entry, issue, reprices_used, request_debug = run_ladder_order(app, prepared, contract, contract_meta_debug)
            if ib_entry is not None and (issue is None or issue.status in {"working_carry", "working"}):
                outcome_override = issue.status if issue is not None and issue.status in {"working_carry", "working"} else ""
                final_entry = entry_from_ib(prepared, ib_entry, paths.orders_csv, request_debug, outcome_override)
                if issue is not None:
                    final_entry["response"]["deferred_message"] = issue.message
                upsert_broker_log_entry(broker_log, final_entry, utc_now_iso)
                fill_entries.append(final_entry)
                sent_count += 1
                outcome = str(final_entry.get("outcome", "unknown"))
                outcome_counters[outcome] = int(outcome_counters.get(outcome, 0)) + 1
                print(
                    f"[BROKER][{paths.name}][SEND] symbol={prepared.symbol} broker_symbol={prepared.broker_symbol} side={prepared.order_side} qty={prepared.qty:.8f} status={final_entry['status']} outcome={outcome} ib_order_id={final_entry['broker_order_id']} reprices_used={reprices_used} contract_details_mode={contract_meta_debug.get('contract_details_mode', '')} price_hint={final_entry['price_hint']:.4f} price_hint_source={final_entry.get('price_hint_source', '')} quote_ts={final_entry.get('quote_ts', '')} model_ref={final_entry.get('model_price_reference', 0.0):.4f} dev_pct={final_entry.get('price_deviation_vs_model', 0.0):.2f}"
                )
                continue
            if issue is None:
                issue = OrderIssue(kind="ladder_failed", status="cancelled_unfilled_cap", broker_error=None, message="Order was cancelled after capped ladder exhausted")
            err_count += 1
            key = issue.status if issue.status in outcome_counters else "failed"
            outcome_counters[key] = int(outcome_counters.get(key, 0)) + 1
            error_entry = build_error_entry(prepared, paths, issue, {**contract_meta_debug, **request_debug})
            upsert_broker_log_entry(broker_log, error_entry, utc_now_iso)
            fill_entries.append(error_entry)
            print(f"[BROKER][{paths.name}][ERROR] symbol={prepared.symbol} side={prepared.order_side} status={issue.status} message={issue.message}")
        except Exception as exc:
            err_count += 1
            outcome_counters["failed"] = int(outcome_counters.get("failed", 0)) + 1
            print(f"[BROKER][{paths.name}][ERROR] {exc}")

    append_or_replace_fills(paths.fills_csv, fill_entries)
    save_broker_log(paths.broker_log_json, broker_log, utc_now_iso)
    print(f"[BROKER][{paths.name}][SUMMARY] sent={sent_count} duplicate_skipped={dup_count} errors={err_count} fills_csv={paths.fills_csv} broker_log_json={paths.broker_log_json} outcomes={json.dumps(outcome_counters, ensure_ascii=False, sort_keys=True)}")
    return req_cursor


def cfg_paths(config_name: str) -> ConfigPaths:
    base = EXECUTION_ROOT / config_name
    return ConfigPaths(name=config_name, execution_dir=base, orders_csv=base / "orders.csv", fills_csv=base / "fills.csv", broker_log_json=base / "broker_log.json")


def main() -> int:
    enable_line_buffering()
    print(f"[CFG] execution_root={EXECUTION_ROOT}")
    print(f"[CFG] configs={CONFIG_NAMES}")
    print(f"[CFG] broker_name={BROKER_NAME} broker_platform={BROKER_PLATFORM} broker_account_id={BROKER_ACCOUNT_ID}")
    print(f"[CFG] ib_host={IB_HOST} ib_port={IB_PORT} ib_client_id={IB_CLIENT_ID} ib_account_code={IB_ACCOUNT_CODE}")
    print(f"[CFG] exchange={IB_EXCHANGE} primary_exchange={IB_PRIMARY_EXCHANGE} currency={IB_CURRENCY} sec_type={IB_SECURITY_TYPE}")
    print(f"[CFG] outside_rth={int(IB_OUTSIDE_RTH)} reprice_enabled={int(IB_REPRICE_ENABLED)} reprice_wait_sec={IB_REPRICE_WAIT_SEC}")
    print(f"[CFG] reprice_steps_bps={IB_REPRICE_STEPS_BPS} reprice_final_mode={IB_REPRICE_FINAL_MODE} reprice_final_marketable_bps={IB_REPRICE_FINAL_MARKETABLE_BPS} reprice_max_deviation_pct={IB_REPRICE_MAX_DEVIATION_PCT}")
    print(f"[CFG] deferred_wait_mode={IB_DEFERRED_WAIT_MODE} deferred_wait_heartbeat_sec={IB_DEFERRED_WAIT_HEARTBEAT_SEC} deferred_post_open_wait_sec={IB_DEFERRED_POST_OPEN_WAIT_SEC}")
    print(f"[CFG] require_price_hint_source={int(REQUIRE_PRICE_HINT_SOURCE)} require_quote_ts={int(REQUIRE_QUOTE_TS)} enforce_open_order_dup_guard={int(ENFORCE_OPEN_ORDER_DUP_GUARD)}")
    symbol_map = load_symbol_map()
    app = bootstrap_connection()
    try:
        req_cursor = 800000
        for config_name in CONFIG_NAMES:
            req_cursor = run_one_config(app, cfg_paths(config_name), symbol_map, req_cursor)
    finally:
        teardown_connection(app)
    print("[FINAL] IBKR broker adapter complete")
    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 1
    safe_exit(rc)
