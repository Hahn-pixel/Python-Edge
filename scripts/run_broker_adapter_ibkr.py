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


def round_to_tick(price: float, tick: float, side: str) -> float:
    px = float(price)
    tk = max(float(tick), float(IB_LMT_PRICE_MIN_ABS))
    if normalize_order_side(side) == "BUY":
        return round(math.floor(px / tk) * tk, 10)
    return round(math.ceil(px / tk) * tk, 10)


def build_passive_price(anchor_price: float, side: str, offset_bps: float, tick: float) -> float:
    side_up = normalize_order_side(side)
    raw = anchor_price * (1.0 - offset_bps / 10000.0) if side_up == "BUY" else anchor_price * (1.0 + offset_bps / 10000.0)
    return max(float(IB_LMT_PRICE_MIN_ABS), round_to_tick(raw, tick, side_up))


def build_cap_price(anchor_price: float, side: str, max_dev_pct: float, tick: float) -> float:
    side_up = normalize_order_side(side)
    raw = anchor_price * (1.0 + max_dev_pct / 100.0) if side_up == "BUY" else anchor_price * (1.0 - max_dev_pct / 100.0)
    return max(float(IB_LMT_PRICE_MIN_ABS), round_to_tick(raw, tick, side_up))


def build_final_price(anchor_price: float, side: str, tick: float, cap_price: float) -> float:
    side_up = normalize_order_side(side)
    raw = anchor_price * (1.0 + IB_REPRICE_FINAL_MARKETABLE_BPS / 10000.0) if side_up == "BUY" else anchor_price * (1.0 - IB_REPRICE_FINAL_MARKETABLE_BPS / 10000.0)
    px = round_to_tick(raw, tick, side_up)
    return min(px, cap_price) if side_up == "BUY" else max(px, cap_price)


def ladder_prices(anchor_price: float, side: str, tick: float) -> List[float]:
    prices: List[float] = []
    for bps in IB_REPRICE_STEPS_BPS:
        px = build_passive_price(anchor_price, side, float(bps), tick)
        if not prices or abs(px - prices[-1]) > 1e-9:
            prices.append(px)
    cap_px = build_cap_price(anchor_price, side, IB_REPRICE_MAX_DEVIATION_PCT, tick)
    if IB_REPRICE_FINAL_MODE == "marketable_lmt":
        final_px = build_final_price(anchor_price, side, tick, cap_px)
        if not prices or abs(final_px - prices[-1]) > 1e-9:
            prices.append(final_px)
    return prices


def collect_app_errors(app: IBKRApp, start_idx: int, req_ids: Sequence[int]) -> List[BrokerErrorInfo]:
    req_id_set = {int(x) for x in req_ids}
    out: List[BrokerErrorInfo] = []
    for payload in app._errors[start_idx:]:
        try:
            req_id = int(payload.get("reqId", 0) or 0)
        except Exception:
            req_id = 0
        if req_id_set and req_id not in req_id_set:
            continue
        out.append(BrokerErrorInfo(req_id=req_id, error_code=int(payload.get("errorCode", 0) or 0), error_string=str(payload.get("errorString", "") or ""), advanced_order_reject_json=str(payload.get("advancedOrderRejectJson", "") or ""), ts_utc=str(payload.get("ts_utc", "") or utc_now_iso())))
    return out


def find_error_code(broker_errors: Sequence[BrokerErrorInfo], code: int) -> BrokerErrorInfo | None:
    for err in broker_errors:
        if int(err.error_code) == int(code):
            return err
    return None


def parse_202_limits(error_string: str) -> Tuple[float | None, float | None]:
    s = str(error_string or "")
    m1 = RE_202_AGGR.search(s)
    m2 = RE_202_MARKET.search(s)
    return (float(m1.group(1)) if m1 else None, float(m2.group(1)) if m2 else None)


def clip_limit_from_202(prepared: PreparedOrder, current_rounded: float | None, err202: BrokerErrorInfo | None) -> float | None:
    if current_rounded is None or err202 is None:
        return None
    cutoff, market_px = parse_202_limits(err202.error_string)
    if cutoff is None:
        return None
    tick = max(float(prepared.min_tick), float(IB_LMT_PRICE_MIN_ABS))
    pad = max(1, int(IB_RETRY_202_CLIP_TICKS)) * tick
    side = prepared.order_side.upper()
    if side == "BUY":
        clipped = max(float(IB_LMT_PRICE_MIN_ABS), cutoff - pad)
        if market_px is not None:
            clipped = min(clipped, market_px + pad)
        clipped = min(clipped, float(current_rounded))
    else:
        clipped = cutoff + pad
        if market_px is not None:
            clipped = max(clipped, market_px - pad)
        clipped = max(clipped, float(current_rounded))
    return round_to_tick(clipped, tick, side)


def build_contract(prepared: PreparedOrder) -> Contract:
    contract = Contract()
    contract.symbol = prepared.broker_symbol
    contract.secType = IB_SECURITY_TYPE
    contract.exchange = IB_EXCHANGE
    contract.currency = IB_CURRENCY
    if IB_PRIMARY_EXCHANGE:
        contract.primaryExchange = IB_PRIMARY_EXCHANGE
    return contract


def query_contract_details(app: IBKRApp, contract: Contract, req_id: int) -> dict:
    app.done_contract_details[int(req_id)] = False
    app.reqContractDetails(int(req_id), contract)
    return app.wait_for_contract_details(int(req_id), timeout_sec=IB_TIMEOUT_SEC)


def resolve_contract_metadata(app: IBKRApp, prepared: PreparedOrder, req_id_seed: int) -> Tuple[PreparedOrder, Contract, int, Dict[str, Any]]:
    req_cursor = int(req_id_seed)
    attempts_total = max(1, int(IB_CONTRACT_DETAILS_RETRIES))
    last_exc: Exception | None = None
    for attempt_idx in range(attempts_total):
        req_cursor += 1
        contract = build_contract(prepared)
        try:
            print(f"[IB][CONTRACT_DETAILS] symbol={prepared.symbol} broker_symbol={prepared.broker_symbol} attempt={attempt_idx + 1}/{attempts_total} use_primary_exchange=1 req_id={req_cursor}")
            meta = query_contract_details(app, contract, req_cursor)
            prepared2 = PreparedOrder(**{**prepared.__dict__, "min_tick": normalize_min_tick(to_float(meta.get('minTick', 0.0)), IB_LMT_PRICE_MIN_ABS)})
            contract2 = build_contract(prepared2)
            if meta.get("primaryExchange") and not IB_PRIMARY_EXCHANGE:
                contract2.primaryExchange = str(meta.get("primaryExchange", ""))
            return prepared2, contract2, req_cursor, {"contract_details_mode": "full", "contract_details_attempts": attempt_idx + 1, "contract_details_fallback_used": 0}
        except Exception as exc:
            last_exc = exc
            print(f"[IB][CONTRACT_DETAILS][WARN] symbol={prepared.symbol} broker_symbol={prepared.broker_symbol} attempt={attempt_idx + 1}/{attempts_total} use_primary_exchange=1 error={exc}")
            if attempt_idx + 1 < attempts_total:
                time.sleep(max(0.0, IB_CONTRACT_DETAILS_RETRY_SLEEP_SEC))
    if IB_ALLOW_PRIMARY_EXCHANGE_FALLBACK:
        req_cursor += 1
        contract = build_contract(prepared)
        try:
            contract.primaryExchange = ""
        except Exception:
            pass
        try:
            print(f"[IB][CONTRACT_DETAILS] symbol={prepared.symbol} broker_symbol={prepared.broker_symbol} attempt=fallback_no_primary use_primary_exchange=0 req_id={req_cursor}")
            meta = query_contract_details(app, contract, req_cursor)
            prepared2 = PreparedOrder(**{**prepared.__dict__, "min_tick": normalize_min_tick(to_float(meta.get('minTick', 0.0)), IB_LMT_PRICE_MIN_ABS)})
            contract2 = build_contract(prepared2)
            try:
                contract2.primaryExchange = ""
            except Exception:
                pass
            return prepared2, contract2, req_cursor, {"contract_details_mode": "no_primary_exchange", "contract_details_attempts": attempts_total + 1, "contract_details_fallback_used": 1}
        except Exception as exc:
            last_exc = exc
            print(f"[IB][CONTRACT_DETAILS][WARN] symbol={prepared.symbol} broker_symbol={prepared.broker_symbol} attempt=fallback_no_primary use_primary_exchange=0 error={exc}")
    if IB_ALLOW_SUBMIT_WITHOUT_CONTRACT_DETAILS:
        prepared2 = PreparedOrder(**{**prepared.__dict__, "min_tick": normalize_min_tick(float(IB_LMT_PRICE_MIN_ABS), IB_LMT_PRICE_MIN_ABS)})
        contract2 = build_contract(prepared2)
        if IB_ALLOW_PRIMARY_EXCHANGE_FALLBACK:
            try:
                contract2.primaryExchange = ""
            except Exception:
                pass
        print(f"[IB][CONTRACT_DETAILS][FALLBACK_SUBMIT] symbol={prepared.symbol} broker_symbol={prepared.broker_symbol} reason={last_exc} min_tick={prepared2.min_tick:.6f}")
        return prepared2, contract2, req_cursor, {"contract_details_mode": "submit_without_contract_details", "contract_details_attempts": attempts_total + (1 if IB_ALLOW_PRIMARY_EXCHANGE_FALLBACK else 0), "contract_details_fallback_used": 1}
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Failed to resolve contract details for symbol={prepared.symbol}")


def load_symbol_map() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if SYMBOL_MAP_FILE:
        path = Path(SYMBOL_MAP_FILE)
        must_exist(path, "BROKER_SYMBOL_MAP_FILE")
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if not isinstance(payload, dict):
            raise RuntimeError("BROKER_SYMBOL_MAP_FILE must contain a JSON object")
        for k, v in payload.items():
            mapping[normalize_symbol(str(k))] = normalize_symbol(str(v))
    if SYMBOL_MAP_JSON:
        payload = json.loads(SYMBOL_MAP_JSON)
        if not isinstance(payload, dict):
            raise RuntimeError("BROKER_SYMBOL_MAP_JSON must decode to a JSON object")
        for k, v in payload.items():
            mapping[normalize_symbol(str(k))] = normalize_symbol(str(v))
    return mapping


def load_orders_df(path: Path) -> pd.DataFrame:
    must_exist(path, "orders.csv")
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=REQUIRED_ORDER_COLUMNS)
    missing = [c for c in REQUIRED_ORDER_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(f"orders.csv missing required columns: {missing}")
    return df.copy()


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
        idempotency_key = hashlib.sha256(json.dumps({"config": config_name, "date": str(row_dict.get("date", "")).strip(), "symbol": symbol, "broker_symbol": broker_symbol, "order_side": order_side, "delta_shares": round(to_float(row_dict.get("delta_shares", 0.0)), 8), "order_notional": round(to_float(row_dict.get("order_notional", 0.0)), 8), "target_weight": round(to_float(row_dict.get("target_weight", 0.0)), 8)}, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
        prepared.append(PreparedOrder(config=config_name, order_date=str(row_dict.get("date", "")).strip(), symbol=symbol, broker_symbol=broker_symbol, order_side=order_side, qty=qty, price=to_float(row_dict.get("price", 0.0)), order_notional=abs(to_float(row_dict.get("order_notional", 0.0))), target_weight=to_float(row_dict.get("target_weight", 0.0)), current_shares=to_float(row_dict.get("current_shares", 0.0)), target_shares=to_float(row_dict.get("target_shares", 0.0)), delta_shares=to_float(row_dict.get("delta_shares", 0.0)), source_row=row_dict, idempotency_key=idempotency_key, client_tag=f"pe-{idempotency_key[:24]}"))
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


def poll_order_status(app: IBKRApp, ib_order_id: int, initial_entry: dict, attempts: int, sleep_sec: float, label: str) -> dict:
    latest = dict(initial_entry or {})
    last_signature = None
    total_attempts = max(0, int(attempts))
    every_n = max(1, int(IB_POLL_PRINT_EVERY))
    mode = str(IB_POLL_VERBOSE or "changes").strip().lower()
    for poll_idx in range(total_attempts):
        time.sleep(max(0.0, float(sleep_sec)))
        refresh_open_orders(app)
        latest = latest_ib_entry(app, ib_order_id, latest)
        status = normalize_status(str(latest.get("status", "")))
        filled_qty = to_float(latest.get("filled_qty", 0.0))
        remaining_qty = to_float(latest.get("remaining_qty", 0.0))
        outcome = classify_outcome(status, filled_qty, remaining_qty)
        signature = (status, round(filled_qty, 8), round(remaining_qty, 8), outcome)
        should_print = mode == "all" or (mode == "none" and (poll_idx == total_attempts - 1 or outcome in {"filled_now", "partial", "failed"})) or (mode != "none" and (poll_idx == 0 or poll_idx == total_attempts - 1 or signature != last_signature or ((poll_idx + 1) % every_n == 0) or outcome in {"filled_now", "partial", "failed"}))
        if should_print:
            print(f"[BROKER][POLL][{label}] ib_order_id={ib_order_id} poll={poll_idx + 1}/{total_attempts} status={status or 'unknown'} outcome={outcome} filled_qty={filled_qty:.8f} remaining_qty={remaining_qty:.8f}")
        last_signature = signature
        if outcome in {"filled_now", "partial", "failed"}:
            break
    return latest


def cancel_and_poll_order(app: IBKRApp, ib_order_id: int, initial_entry: dict, label: str) -> dict:
    latest = dict(initial_entry or {})
    print(f"[BROKER][CANCEL] ib_order_id={ib_order_id} reason={label}")
    app.cancelOrder(int(ib_order_id), "")
    return poll_order_status(app, ib_order_id, latest, IB_REPRICE_CANCEL_POLL_ATTEMPTS, IB_REPRICE_CANCEL_POLL_SLEEP_SEC, f"cancel_{label}")


def refresh_positions(app: IBKRApp, required: bool) -> bool:
    app.done_positions = False
    app.position_rows = []
    app.reqPositions()
    try:
        app.wait_until_positions_end(timeout_sec=IB_POSITIONS_TIMEOUT_SEC)
        app.cancelPositions()
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
    app.start_network_loop()
    next_id = app.wait_for_next_valid_id(timeout_sec=IB_TIMEOUT_SEC)
    managed_accounts = app.wait_for_managed_accounts(timeout_sec=IB_TIMEOUT_SEC)
    if IB_ACCOUNT_CODE:
        account_list = [x.strip() for x in managed_accounts.split(",") if x.strip()]
        if IB_ACCOUNT_CODE not in account_list:
            raise RuntimeError(f"IB_ACCOUNT_CODE={IB_ACCOUNT_CODE!r} not present in managed accounts: {managed_accounts!r}")
    print(f"[IB] connected next_valid_id={next_id} managed_accounts={managed_accounts}")
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
    return {"idempotency_key": prepared.idempotency_key, "client_order_id": prepared.client_tag, "broker_order_id": str(ib_entry.get("ib_order_id", "")), "perm_id": int(ib_entry.get("perm_id", 0) or 0), "config": prepared.config, "source_order_path": str(source_path), "date": prepared.order_date, "symbol": prepared.symbol, "broker_symbol": prepared.broker_symbol, "side": prepared.order_side, "qty": float(prepared.qty), "filled_qty": float(filled_qty), "remaining_qty": float(remaining_qty), "price_hint": float(prepared.price), "filled_avg_price": float(avg_fill_price), "order_notional": float(prepared.order_notional), "fill_notional": abs(float(filled_qty * avg_fill_price)), "status": status, "submitted_at": str(ib_entry.get("submitted_at_utc", utc_now_iso())), "filled_at": utc_now_iso() if outcome in {"filled_now", "partial"} else "", "mode": "ibkr_gateway", "request": request_debug, "response": {"ib_order_id": ib_entry.get("ib_order_id", ""), "perm_id": ib_entry.get("perm_id", 0), "status": ib_entry.get("status", ""), "avg_fill_price": ib_entry.get("avg_fill_price", 0.0), "last_fill_price": ib_entry.get("last_fill_price", 0.0), "fills": ib_entry.get("fills", []), "outcome": outcome}, "outcome": outcome}


def build_error_entry(prepared: PreparedOrder, paths: ConfigPaths, issue: OrderIssue, request_debug: Dict[str, Any]) -> dict:
    return {"idempotency_key": prepared.idempotency_key, "client_order_id": prepared.client_tag, "broker_order_id": "", "perm_id": 0, "config": prepared.config, "source_order_path": str(paths.orders_csv), "date": prepared.order_date, "symbol": prepared.symbol, "broker_symbol": prepared.broker_symbol, "side": prepared.order_side, "qty": float(prepared.qty), "filled_qty": 0.0, "remaining_qty": float(prepared.qty), "price_hint": float(prepared.price), "filled_avg_price": 0.0, "order_notional": float(prepared.order_notional), "fill_notional": 0.0, "status": issue.status, "submitted_at": utc_now_iso(), "filled_at": "", "mode": "ibkr_gateway", "request": request_debug, "response": {"error": issue.message, "error_code": int(issue.broker_error.error_code) if issue.broker_error else 0, "error_string": str(issue.broker_error.error_string) if issue.broker_error else "", "outcome": "failed"}, "outcome": "failed"}


def run_ladder_order(app: IBKRApp, prepared: PreparedOrder, contract: Contract, contract_meta_debug: Dict[str, Any]) -> Tuple[dict | None, OrderIssue | None, int, Dict[str, Any]]:
    anchor_price = max(float(IB_LMT_PRICE_MIN_ABS), float(prepared.price))
    tick = max(float(prepared.min_tick), float(IB_LMT_PRICE_MIN_ABS))
    prices = ladder_prices(anchor_price, prepared.order_side, tick)
    cap_price = build_cap_price(anchor_price, prepared.order_side, IB_REPRICE_MAX_DEVIATION_PCT, tick)
    request_debug: Dict[str, Any] = {"account": IB_ACCOUNT_CODE, "host": IB_HOST, "port": IB_PORT, "client_id": IB_CLIENT_ID, "tif": IB_TIME_IN_FORCE, "outside_rth": int(IB_OUTSIDE_RTH), **contract_meta_debug, "anchor_price": float(anchor_price), "reprice_enabled": int(IB_REPRICE_ENABLED), "reprice_wait_sec": float(IB_REPRICE_WAIT_SEC), "reprice_steps_bps": [float(x) for x in IB_REPRICE_STEPS_BPS], "reprice_final_mode": IB_REPRICE_FINAL_MODE, "reprice_final_marketable_bps": float(IB_REPRICE_FINAL_MARKETABLE_BPS), "reprice_max_deviation_pct": float(IB_REPRICE_MAX_DEVIATION_PCT), "cap_price": float(cap_price)}
    reprices_used = 0
    last_entry: dict | None = None
    for step_idx, limit_price in enumerate(prices):
        is_last_step = step_idx == len(prices) - 1
        step_mode = "marketable_lmt" if is_last_step and IB_REPRICE_FINAL_MODE == "marketable_lmt" else "passive_lmt"
        order = build_order_for_price(prepared, limit_price=limit_price, is_market=False)
        error_cursor_before_submit = len(app._errors)
        ib_order_id = app.allocate_order_id()
        print(f"[BROKER][LADDER][SUBMIT] symbol={prepared.symbol} step={step_idx + 1}/{len(prices)} mode={step_mode} limit_price={limit_price:.4f} cap_price={cap_price:.4f}")
        app.placeOrder(ib_order_id, contract, order)
        ib_entry = app.wait_for_order_terminalish(ib_order_id, timeout_sec=IB_TIMEOUT_SEC)
        poll_attempts = max(1, int(math.ceil(IB_REPRICE_WAIT_SEC / max(0.25, IB_REPRICE_CANCEL_POLL_SLEEP_SEC))))
        ib_entry = poll_order_status(app, ib_order_id, ib_entry, poll_attempts, max(0.25, IB_REPRICE_CANCEL_POLL_SLEEP_SEC), f"ladder_step_{step_idx + 1}")
        broker_errors = collect_app_errors(app, error_cursor_before_submit, [int(ib_order_id)])
        issue = None
        if IB_ALLOW_FRACTIONAL and abs(prepared.qty - round(prepared.qty)) > 1e-9:
            for err in broker_errors:
                if int(err.error_code) in set(IB_FRACTIONAL_REJECT_CODES):
                    issue = OrderIssue(kind="fractional_api_reject", status="rejected_fractional_api", broker_error=err, message=f"IBKR API rejected fractional order code={err.error_code}: {err.error_string}")
                    break
        if issue is None and broker_errors:
            err = broker_errors[-1]
            issue = OrderIssue(kind="broker_api_error", status=f"error_api_{int(err.error_code)}", broker_error=err, message=f"IBKR API error code={err.error_code}: {err.error_string}")
        retryable_202 = find_error_code(broker_errors, 202)
        if retryable_202 is not None and IB_RETRY_202_ENABLED:
            clipped = clip_limit_from_202(prepared, limit_price, retryable_202)
            if clipped is not None and abs(clipped - limit_price) > 1e-9:
                print(f"[BROKER][LADDER][202] symbol={prepared.symbol} step={step_idx + 1} next_limit_price={clipped:.4f}")
        outcome = classify_outcome(str(ib_entry.get("status", "")), to_float(ib_entry.get("filled_qty", 0.0)), to_float(ib_entry.get("remaining_qty", 0.0)))
        if issue is None and outcome in {"filled_now", "partial"}:
            request_debug.update({"reprices_used": reprices_used, "final_step": step_idx + 1, "final_limit_price": float(limit_price), "final_mode": step_mode})
            return ib_entry, None, reprices_used, request_debug
        if not is_last_step:
            cancel_and_poll_order(app, ib_order_id, ib_entry, f"reprice_step_{step_idx + 1}")
            reprices_used += 1
            continue
        last_entry = cancel_and_poll_order(app, ib_order_id, ib_entry, "final_unfilled_cap")
        request_debug.update({"reprices_used": reprices_used, "final_step": step_idx + 1, "final_limit_price": float(limit_price), "final_mode": step_mode})
        if issue is None:
            issue = OrderIssue(kind="ladder_cap_cancelled", status="cancelled_unfilled_cap", broker_error=None, message="Final capped price was not filled; order cancelled")
        return last_entry, issue, reprices_used, request_debug
    if IB_REPRICE_FINAL_MODE == "mkt":
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
    sent_count = dup_count = err_count = retry_202_count = 0
    outcome_counters: Dict[str, int] = {"filled_now": 0, "partial": 0, "working": 0, "working_carry": 0, "cancelled_after_ttl": 0, "cancelled_unfilled_cap": 0, "failed": 0, "unknown": 0}
    req_cursor = int(req_id_seed)

    for prepared0 in prepared_orders:
        duplicate_status = existing_duplicate_status(broker_log, prepared0.idempotency_key)
        if duplicate_status is not None:
            dup_count += 1
            fill_entries.append(duplicate_fill_entry(prepared0, duplicate_status, paths.orders_csv, broker_log))
            continue
        try:
            prepared, contract, req_cursor, contract_meta_debug = resolve_contract_metadata(app, prepared0, req_cursor)
            request_debug: Dict[str, Any] = {}
            ib_entry, issue, reprices_used, request_debug = run_ladder_order(app, prepared, contract, contract_meta_debug)
            if ib_entry is not None and issue is None:
                final_entry = entry_from_ib(prepared, ib_entry, paths.orders_csv, request_debug, "")
                upsert_broker_log_entry(broker_log, final_entry, utc_now_iso)
                fill_entries.append(final_entry)
                sent_count += 1
                outcome = str(final_entry.get("outcome", "unknown"))
                outcome_counters[outcome] = int(outcome_counters.get(outcome, 0)) + 1
                print(f"[BROKER][{paths.name}][SEND] symbol={prepared.symbol} broker_symbol={prepared.broker_symbol} side={prepared.order_side} qty={prepared.qty:.8f} status={final_entry['status']} outcome={outcome} ib_order_id={final_entry['broker_order_id']} reprices_used={reprices_used} contract_details_mode={contract_meta_debug.get('contract_details_mode', '')}")
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
    print(f"[BROKER][{paths.name}][SUMMARY] sent={sent_count} duplicate_skipped={dup_count} errors={err_count} retry_202={retry_202_count} filled_now={outcome_counters.get('filled_now', 0)} partial={outcome_counters.get('partial', 0)} working={outcome_counters.get('working', 0)} working_carry={outcome_counters.get('working_carry', 0)} cancelled_after_ttl={outcome_counters.get('cancelled_after_ttl', 0)} cancelled_unfilled_cap={outcome_counters.get('cancelled_unfilled_cap', 0)} failed={outcome_counters.get('failed', 0)} unknown={outcome_counters.get('unknown', 0)} fills_csv={paths.fills_csv} broker_log_json={paths.broker_log_json}")
    return req_cursor


def cfg_paths(config_name: str) -> ConfigPaths:
    base = EXECUTION_ROOT / config_name
    return ConfigPaths(name=config_name, execution_dir=base, orders_csv=base / "orders.csv", fills_csv=base / "fills.csv", broker_log_json=base / "broker_log.json")


def main() -> int:
    enable_line_buffering()
    symbol_map = load_symbol_map()
    print(f"[CFG] ib_host={IB_HOST} ib_port={IB_PORT} ib_client_id={IB_CLIENT_ID}")
    print(f"[CFG] outside_rth={int(IB_OUTSIDE_RTH)} reprice_enabled={int(IB_REPRICE_ENABLED)} reprice_wait_sec={IB_REPRICE_WAIT_SEC}")
    print(f"[CFG] reprice_steps_bps={IB_REPRICE_STEPS_BPS} reprice_final_mode={IB_REPRICE_FINAL_MODE} reprice_final_marketable_bps={IB_REPRICE_FINAL_MARKETABLE_BPS} reprice_max_deviation_pct={IB_REPRICE_MAX_DEVIATION_PCT}")
    print(f"[CFG] contract_details_retries={IB_CONTRACT_DETAILS_RETRIES} contract_details_retry_sleep_sec={IB_CONTRACT_DETAILS_RETRY_SLEEP_SEC} allow_primary_exchange_fallback={int(IB_ALLOW_PRIMARY_EXCHANGE_FALLBACK)} allow_submit_without_contract_details={int(IB_ALLOW_SUBMIT_WITHOUT_CONTRACT_DETAILS)}")
    app = bootstrap_connection()
    req_cursor = 100000
    try:
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
