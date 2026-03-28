from __future__ import annotations

import hashlib
import json
import os
import queue
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
from ibapi.client import EClient
from ibapi.common import OrderId
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.wrapper import EWrapper

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
for p in [ROOT, SRC_DIR]:
    ps = str(p)
    if ps not in sys.path:
        sys.path.insert(0, ps)

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
IB_ORDER_TYPE = str(os.getenv("IB_ORDER_TYPE", "MKT")).strip().upper()
IB_TIME_IN_FORCE = str(os.getenv("IB_TIME_IN_FORCE", "DAY")).strip().upper()
IB_OUTSIDE_RTH = str(os.getenv("IB_OUTSIDE_RTH", "0")).strip().lower() not in {"0", "false", "no", "off"}
IB_ALLOW_FRACTIONAL = str(os.getenv("IB_ALLOW_FRACTIONAL", "0")).strip().lower() not in {"0", "false", "no", "off"}
IB_POLL_AFTER_SUBMIT = str(os.getenv("IB_POLL_AFTER_SUBMIT", "1")).strip().lower() not in {"0", "false", "no", "off"}
IB_POLL_ATTEMPTS = int(os.getenv("IB_POLL_ATTEMPTS", "6"))
IB_POLL_SLEEP_SEC = float(os.getenv("IB_POLL_SLEEP_SEC", "1.5"))
IB_EXCHANGE = str(os.getenv("IB_EXCHANGE", "SMART")).strip().upper() or "SMART"
IB_PRIMARY_EXCHANGE = str(os.getenv("IB_PRIMARY_EXCHANGE", "")).strip().upper()
IB_CURRENCY = str(os.getenv("IB_CURRENCY", "USD")).strip().upper() or "USD"
IB_SECURITY_TYPE = str(os.getenv("IB_SECURITY_TYPE", "STK")).strip().upper() or "STK"
RESET_BROKER_LOG = str(os.getenv("RESET_BROKER_LOG", "0")).strip().lower() not in {"0", "false", "no", "off"}
SYMBOL_MAP_FILE = str(os.getenv("BROKER_SYMBOL_MAP_FILE", "")).strip()
SYMBOL_MAP_JSON = str(os.getenv("BROKER_SYMBOL_MAP_JSON", "")).strip()
TOPK_PRINT = int(os.getenv("TOPK_PRINT", "50"))
IB_FRACTIONAL_REJECT_CODES = [
    int(x.strip())
    for x in str(os.getenv("IB_FRACTIONAL_REJECT_CODES", "10243|10247|10248|10249|10250")).split("|")
    if x.strip()
]
IB_FRACTIONAL_FAILURE_MODE = str(os.getenv("IB_FRACTIONAL_FAILURE_MODE", "review")).strip().lower() or "review"
IB_REFRESH_POSITIONS_ON_CONNECT = str(os.getenv("IB_REFRESH_POSITIONS_ON_CONNECT", "0")).strip().lower() not in {"0", "false", "no", "off"}
IB_REQUIRE_POSITIONS_ON_CONNECT = str(os.getenv("IB_REQUIRE_POSITIONS_ON_CONNECT", "0")).strip().lower() not in {"0", "false", "no", "off"}
IB_POSITIONS_TIMEOUT_SEC = float(os.getenv("IB_POSITIONS_TIMEOUT_SEC", str(IB_TIMEOUT_SEC)))
IB_OPEN_ORDERS_TIMEOUT_SEC = float(os.getenv("IB_OPEN_ORDERS_TIMEOUT_SEC", str(IB_TIMEOUT_SEC)))

REQUIRED_ORDER_COLUMNS = ["symbol", "order_side", "delta_shares"]
FINAL_DUPLICATE_STATUSES = {
    "presubmitted",
    "submitted",
    "pending_submit",
    "pending_cancel",
    "api_pending",
    "api_cancelled",
    "partially_filled",
    "filled",
}


@dataclass(frozen=True)
class BrokerErrorInfo:
    req_id: int
    error_code: int
    error_string: str
    advanced_order_reject_json: str
    ts_utc: str


@dataclass(frozen=True)
class OrderIssue:
    kind: str
    status: str
    broker_error: BrokerErrorInfo | None
    message: str


@dataclass(frozen=True)
class ConfigPaths:
    name: str
    execution_dir: Path
    orders_csv: Path
    fills_csv: Path
    broker_log_json: Path


@dataclass(frozen=True)
class PreparedOrder:
    config: str
    order_date: str
    symbol: str
    broker_symbol: str
    order_side: str
    qty: float
    price: float
    order_notional: float
    target_weight: float
    current_shares: float
    target_shares: float
    delta_shares: float
    source_row: Dict[str, Any]
    idempotency_key: str
    client_tag: str


class IBKRApp(EWrapper, EClient):
    def __init__(self) -> None:
        EClient.__init__(self, self)
        self._next_valid_id_queue: queue.Queue[int] = queue.Queue()
        self._managed_accounts_queue: queue.Queue[str] = queue.Queue()
        self._submit_lock = threading.Lock()
        self._network_thread: Optional[threading.Thread] = None
        self._connected_flag = False
        self._errors: List[dict] = []
        self.orders_by_ib_id: Dict[int, dict] = {}
        self.ibid_by_client_tag: Dict[str, int] = {}
        self.done_contract_details: Dict[int, bool] = {}
        self.contract_details: Dict[int, dict] = {}
        self.done_open_orders = False
        self.done_positions = False
        self.position_rows: List[dict] = []
        self.order_id_cursor: Optional[int] = None

    def error(self, reqId: int, errorCode: int, errorString: str, advancedOrderRejectJson: str = "") -> None:
        payload = {
            "ts_utc": _utc_now_iso(),
            "reqId": int(reqId),
            "errorCode": int(errorCode),
            "errorString": str(errorString),
            "advancedOrderRejectJson": str(advancedOrderRejectJson or ""),
        }
        self._errors.append(payload)
        print(f"[IB][ERROR] reqId={reqId} code={errorCode} msg={errorString}")

    def connectAck(self) -> None:
        self._connected_flag = True
        print("[IB] connectAck received")

    def nextValidId(self, orderId: int) -> None:
        self.order_id_cursor = int(orderId)
        try:
            self._next_valid_id_queue.put_nowait(int(orderId))
        except Exception:
            pass
        print(f"[IB] nextValidId={orderId}")

    def managedAccounts(self, accountsList: str) -> None:
        try:
            self._managed_accounts_queue.put_nowait(str(accountsList))
        except Exception:
            pass
        print(f"[IB] managedAccounts={accountsList}")

    def openOrder(self, orderId: OrderId, contract: Contract, order: Order, orderState) -> None:
        client_tag = str(getattr(order, "orderRef", "") or "")
        entry = self.orders_by_ib_id.setdefault(
            int(orderId),
            {
                "ib_order_id": int(orderId),
                "symbol": str(getattr(contract, "symbol", "") or ""),
                "broker_symbol": str(getattr(contract, "localSymbol", "") or getattr(contract, "symbol", "") or ""),
                "status": str(getattr(orderState, "status", "") or ""),
                "client_tag": client_tag,
                "perm_id": int(getattr(order, "permId", 0) or 0),
                "action": str(getattr(order, "action", "") or ""),
                "total_qty": _to_float(getattr(order, "totalQuantity", 0.0)),
                "filled_qty": 0.0,
                "remaining_qty": _to_float(getattr(order, "totalQuantity", 0.0)),
                "avg_fill_price": 0.0,
                "last_fill_price": 0.0,
                "submitted_at_utc": _utc_now_iso(),
                "fills": [],
            },
        )
        entry["symbol"] = str(getattr(contract, "symbol", "") or entry.get("symbol", ""))
        entry["broker_symbol"] = str(getattr(contract, "localSymbol", "") or getattr(contract, "symbol", "") or entry.get("broker_symbol", ""))
        entry["status"] = str(getattr(orderState, "status", "") or entry.get("status", ""))
        entry["client_tag"] = client_tag or str(entry.get("client_tag", ""))
        entry["perm_id"] = int(getattr(order, "permId", 0) or entry.get("perm_id", 0) or 0)
        entry["action"] = str(getattr(order, "action", "") or entry.get("action", ""))
        entry["total_qty"] = _to_float(getattr(order, "totalQuantity", entry.get("total_qty", 0.0)))
        if client_tag:
            self.ibid_by_client_tag[client_tag] = int(orderId)

    def openOrderEnd(self) -> None:
        self.done_open_orders = True
        print("[IB] openOrderEnd")

    def orderStatus(
        self,
        orderId: OrderId,
        status: str,
        filled: float,
        remaining: float,
        avgFillPrice: float,
        permId: int,
        parentId: int,
        lastFillPrice: float,
        clientId: int,
        whyHeld: str,
        mktCapPrice: float,
    ) -> None:
        entry = self.orders_by_ib_id.setdefault(
            int(orderId),
            {
                "ib_order_id": int(orderId),
                "symbol": "",
                "broker_symbol": "",
                "status": str(status or ""),
                "client_tag": "",
                "perm_id": int(permId or 0),
                "action": "",
                "total_qty": float(filled or 0.0) + float(remaining or 0.0),
                "filled_qty": float(filled or 0.0),
                "remaining_qty": float(remaining or 0.0),
                "avg_fill_price": float(avgFillPrice or 0.0),
                "last_fill_price": float(lastFillPrice or 0.0),
                "submitted_at_utc": _utc_now_iso(),
                "fills": [],
            },
        )
        entry["status"] = str(status or entry.get("status", ""))
        entry["filled_qty"] = float(filled or 0.0)
        entry["remaining_qty"] = float(remaining or 0.0)
        entry["avg_fill_price"] = float(avgFillPrice or 0.0)
        entry["last_fill_price"] = float(lastFillPrice or 0.0)
        entry["perm_id"] = int(permId or entry.get("perm_id", 0) or 0)

    def execDetails(self, reqId, contract: Contract, execution) -> None:
        order_id = int(getattr(execution, "orderId", 0) or 0)
        entry = self.orders_by_ib_id.setdefault(
            order_id,
            {
                "ib_order_id": order_id,
                "symbol": str(getattr(contract, "symbol", "") or ""),
                "broker_symbol": str(getattr(contract, "localSymbol", "") or getattr(contract, "symbol", "") or ""),
                "status": "",
                "client_tag": "",
                "perm_id": int(getattr(execution, "permId", 0) or 0),
                "action": str(getattr(execution, "side", "") or ""),
                "total_qty": 0.0,
                "filled_qty": 0.0,
                "remaining_qty": 0.0,
                "avg_fill_price": 0.0,
                "last_fill_price": 0.0,
                "submitted_at_utc": _utc_now_iso(),
                "fills": [],
            },
        )
        fill_row = {
            "exec_id": str(getattr(execution, "execId", "") or ""),
            "time": str(getattr(execution, "time", "") or ""),
            "shares": _to_float(getattr(execution, "shares", 0.0)),
            "price": _to_float(getattr(execution, "price", 0.0)),
            "cumQty": _to_float(getattr(execution, "cumQty", 0.0)),
            "avgPrice": _to_float(getattr(execution, "avgPrice", 0.0)),
            "side": str(getattr(execution, "side", "") or ""),
        }
        entry.setdefault("fills", []).append(fill_row)
        entry["filled_qty"] = max(float(entry.get("filled_qty", 0.0)), float(fill_row["cumQty"]))
        entry["avg_fill_price"] = float(fill_row["avgPrice"])
        entry["last_fill_price"] = float(fill_row["price"])
        entry["perm_id"] = int(getattr(execution, "permId", 0) or entry.get("perm_id", 0) or 0)

    def position(self, account: str, contract: Contract, position: float, avgCost: float) -> None:
        self.position_rows.append(
            {
                "account": str(account),
                "symbol": str(getattr(contract, "symbol", "") or ""),
                "localSymbol": str(getattr(contract, "localSymbol", "") or ""),
                "secType": str(getattr(contract, "secType", "") or ""),
                "currency": str(getattr(contract, "currency", "") or ""),
                "exchange": str(getattr(contract, "exchange", "") or ""),
                "position": _to_float(position),
                "avgCost": _to_float(avgCost),
            }
        )

    def positionEnd(self) -> None:
        self.done_positions = True
        print("[IB] positionEnd")

    def contractDetails(self, reqId: int, contractDetails) -> None:
        contract = getattr(contractDetails, "contract", None)
        self.contract_details[int(reqId)] = {
            "symbol": str(getattr(contract, "symbol", "") or ""),
            "localSymbol": str(getattr(contract, "localSymbol", "") or ""),
            "primaryExchange": str(getattr(contract, "primaryExchange", "") or ""),
            "exchange": str(getattr(contract, "exchange", "") or ""),
            "currency": str(getattr(contract, "currency", "") or ""),
            "secType": str(getattr(contract, "secType", "") or ""),
            "minTick": _to_float(getattr(contractDetails, "minTick", 0.0)),
            "validExchanges": str(getattr(contractDetails, "validExchanges", "") or ""),
            "longName": str(getattr(contractDetails, "longName", "") or ""),
        }

    def contractDetailsEnd(self, reqId: int) -> None:
        self.done_contract_details[int(reqId)] = True

    def start_network_loop(self) -> None:
        if self._network_thread is not None and self._network_thread.is_alive():
            return
        self._network_thread = threading.Thread(target=self.run, name="ibapi-network", daemon=True)
        self._network_thread.start()

    def wait_for_next_valid_id(self, timeout_sec: float) -> int:
        return int(self._next_valid_id_queue.get(timeout=timeout_sec))

    def wait_for_managed_accounts(self, timeout_sec: float) -> str:
        return str(self._managed_accounts_queue.get(timeout=timeout_sec))

    def allocate_order_id(self) -> int:
        with self._submit_lock:
            if self.order_id_cursor is None:
                raise RuntimeError("IB nextValidId was not received")
            oid = int(self.order_id_cursor)
            self.order_id_cursor += 1
            return oid

    def wait_until_open_orders_end(self, timeout_sec: float) -> None:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            if self.done_open_orders:
                return
            time.sleep(0.1)
        raise TimeoutError("Timed out waiting for openOrderEnd")

    def wait_until_positions_end(self, timeout_sec: float) -> None:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            if self.done_positions:
                return
            time.sleep(0.1)
        raise TimeoutError("Timed out waiting for positionEnd")

    def wait_for_contract_details(self, req_id: int, timeout_sec: float) -> dict:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            if self.done_contract_details.get(int(req_id), False):
                return self.contract_details.get(int(req_id), {})
            time.sleep(0.1)
        raise TimeoutError(f"Timed out waiting for contract details req_id={req_id}")

    def wait_for_order_terminalish(self, order_id: int, timeout_sec: float) -> dict:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            entry = self.orders_by_ib_id.get(int(order_id), {})
            status = str(entry.get("status", "")).strip().lower()
            if status in {
                "presubmitted",
                "submitted",
                "partiallyfilled",
                "filled",
                "cancelled",
                "inactive",
                "apicancelled",
                "pendingcancel",
                "pendingsubmit",
                "presubmitted",
                "api_pending",
            }:
                return entry
            time.sleep(0.1)
        return self.orders_by_ib_id.get(int(order_id), {})


def _is_fractional_qty(qty: float) -> bool:
    return abs(float(qty) - float(round(float(qty)))) > 1e-9


def _fractional_error_code_set() -> set[int]:
    return {int(x) for x in IB_FRACTIONAL_REJECT_CODES}


def _collect_app_errors(app: IBKRApp, start_idx: int, req_ids: Sequence[int]) -> List[BrokerErrorInfo]:
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
                ts_utc=str(payload.get("ts_utc", "") or _utc_now_iso()),
            )
        )
    return out


def _first_fractional_issue(prepared: PreparedOrder, broker_errors: Sequence[BrokerErrorInfo]) -> OrderIssue | None:
    if not _is_fractional_qty(prepared.qty):
        return None
    code_set = _fractional_error_code_set()
    for err in broker_errors:
        if int(err.error_code) in code_set:
            return OrderIssue(
                kind="fractional_api_reject",
                status="rejected_fractional_api",
                broker_error=err,
                message=f"IBKR API rejected fractional order code={err.error_code}: {err.error_string}",
            )
    return None


def _first_generic_issue(broker_errors: Sequence[BrokerErrorInfo]) -> OrderIssue | None:
    if not broker_errors:
        return None
    err = broker_errors[-1]
    return OrderIssue(
        kind="broker_api_error",
        status=f"error_api_{int(err.error_code)}",
        broker_error=err,
        message=f"IBKR API error code={err.error_code}: {err.error_string}",
    )


def _issue_entry(prepared: PreparedOrder, issue: OrderIssue, source_path: Path) -> dict:
    broker_error = issue.broker_error
    response_payload = {"error": issue.message}
    if broker_error is not None:
        response_payload.update(
            {
                "error_code": int(broker_error.error_code),
                "error_string": str(broker_error.error_string),
                "advanced_order_reject_json": str(broker_error.advanced_order_reject_json),
                "error_ts_utc": str(broker_error.ts_utc),
            }
        )
    return {
        "idempotency_key": prepared.idempotency_key,
        "client_order_id": prepared.client_tag,
        "broker_order_id": "",
        "perm_id": 0,
        "config": prepared.config,
        "source_order_path": str(source_path),
        "date": prepared.order_date,
        "symbol": prepared.symbol,
        "broker_symbol": prepared.broker_symbol,
        "side": prepared.order_side,
        "qty": float(prepared.qty),
        "filled_qty": 0.0,
        "remaining_qty": float(prepared.qty),
        "price_hint": float(prepared.price),
        "filled_avg_price": 0.0,
        "order_notional": float(prepared.order_notional),
        "fill_notional": 0.0,
        "status": str(issue.status),
        "submitted_at": _utc_now_iso(),
        "filled_at": "",
        "mode": "ibkr_gateway",
        "request": {
            "account": IB_ACCOUNT_CODE,
            "host": IB_HOST,
            "port": IB_PORT,
            "client_id": IB_CLIENT_ID,
            "order_type": IB_ORDER_TYPE,
            "tif": IB_TIME_IN_FORCE,
            "outside_rth": int(IB_OUTSIDE_RTH),
            "exchange": IB_EXCHANGE,
            "primary_exchange": IB_PRIMARY_EXCHANGE,
            "allow_fractional": int(IB_ALLOW_FRACTIONAL),
        },
        "response": response_payload,
    }


def _manual_review_entry(prepared: PreparedOrder, issue: OrderIssue, source_path: Path) -> dict:
    broker_error = issue.broker_error
    return {
        "date": prepared.order_date,
        "config": prepared.config,
        "symbol": prepared.symbol,
        "broker_symbol": prepared.broker_symbol,
        "side": prepared.order_side,
        "qty": float(prepared.qty),
        "price_hint": float(prepared.price),
        "order_notional": float(prepared.order_notional),
        "target_weight": float(prepared.target_weight),
        "current_shares": float(prepared.current_shares),
        "target_shares": float(prepared.target_shares),
        "delta_shares": float(prepared.delta_shares),
        "idempotency_key": prepared.idempotency_key,
        "client_order_id": prepared.client_tag,
        "issue_kind": issue.kind,
        "issue_status": issue.status,
        "issue_message": issue.message,
        "broker_error_code": int(broker_error.error_code) if broker_error is not None else 0,
        "broker_error_string": str(broker_error.error_string) if broker_error is not None else "",
        "advanced_order_reject_json": str(broker_error.advanced_order_reject_json) if broker_error is not None else "",
        "source_order_path": str(source_path),
        "ib_exchange": IB_EXCHANGE,
        "ib_primary_exchange": IB_PRIMARY_EXCHANGE,
        "ib_order_type": IB_ORDER_TYPE,
        "ib_time_in_force": IB_TIME_IN_FORCE,
        "ib_outside_rth": int(IB_OUTSIDE_RTH),
        "ib_allow_fractional": int(IB_ALLOW_FRACTIONAL),
    }


def _append_manual_review_csv(path: Path, entries: List[dict]) -> None:
    cols = [
        "date", "config", "symbol", "broker_symbol", "side", "qty", "price_hint", "order_notional",
        "target_weight", "current_shares", "target_shares", "delta_shares", "idempotency_key",
        "client_order_id", "issue_kind", "issue_status", "issue_message", "broker_error_code",
        "broker_error_string", "advanced_order_reject_json", "source_order_path", "ib_exchange",
        "ib_primary_exchange", "ib_order_type", "ib_time_in_force", "ib_outside_rth", "ib_allow_fractional",
    ]
    new_df = pd.DataFrame(entries)
    if new_df.empty:
        if not path.exists():
            pd.DataFrame(columns=cols).to_csv(path, index=False)
        return
    for col in cols:
        if col not in new_df.columns:
            new_df[col] = ""
    new_df = new_df[cols].copy()
    if path.exists():
        prev = pd.read_csv(path)
        for col in cols:
            if col not in prev.columns:
                prev[col] = ""
        prev = prev[cols].copy()
        out = pd.concat([prev, new_df], ignore_index=True)
        out = out.drop_duplicates(subset=["idempotency_key"], keep="last")
    else:
        out = new_df
    out = out.sort_values(["date", "config", "symbol", "side", "idempotency_key"]).reset_index(drop=True)
    out.to_csv(path, index=False)


def _enable_line_buffering() -> None:
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


def _should_pause() -> bool:
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


def _safe_exit(code: int) -> None:
    if _should_pause():
        try:
            print("")
            print(f"[EXIT] code={code}")
            input("Press Enter to exit...")
        except Exception:
            pass
    raise SystemExit(code)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _must_exist(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def _to_float(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def _normalize_symbol(symbol: str) -> str:
    return str(symbol or "").strip().upper()


def _normalize_order_side(side: str) -> str:
    out = str(side or "").strip().upper()
    if out not in {"BUY", "SELL", "HOLD"}:
        raise RuntimeError(f"Unsupported order_side: {side!r}")
    return out


def _cfg_paths(name: str) -> ConfigPaths:
    execution_dir = EXECUTION_ROOT / name
    return ConfigPaths(
        name=name,
        execution_dir=execution_dir,
        orders_csv=execution_dir / "orders.csv",
        fills_csv=execution_dir / "fills.csv",
        broker_log_json=execution_dir / "broker_log.json",
    )


def _load_symbol_map() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if SYMBOL_MAP_FILE:
        path = Path(SYMBOL_MAP_FILE)
        _must_exist(path, "BROKER_SYMBOL_MAP_FILE")
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if not isinstance(payload, dict):
            raise RuntimeError("BROKER_SYMBOL_MAP_FILE must contain a JSON object")
        for k, v in payload.items():
            mapping[_normalize_symbol(str(k))] = _normalize_symbol(str(v))
    if SYMBOL_MAP_JSON:
        payload = json.loads(SYMBOL_MAP_JSON)
        if not isinstance(payload, dict):
            raise RuntimeError("BROKER_SYMBOL_MAP_JSON must decode to a JSON object")
        for k, v in payload.items():
            mapping[_normalize_symbol(str(k))] = _normalize_symbol(str(v))
    return mapping


def _load_broker_log(path: Path, config_name: str) -> dict:
    if RESET_BROKER_LOG or not path.exists():
        return {
            "config": config_name,
            "broker_name": BROKER_NAME,
            "broker_platform": BROKER_PLATFORM,
            "broker_account_id": BROKER_ACCOUNT_ID,
            "created_at_utc": _utc_now_iso(),
            "updated_at_utc": _utc_now_iso(),
            "orders": {},
        }
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise RuntimeError(f"broker_log.json must contain a JSON object: {path}")
    payload.setdefault("config", config_name)
    payload.setdefault("broker_name", BROKER_NAME)
    payload.setdefault("broker_platform", BROKER_PLATFORM)
    payload.setdefault("broker_account_id", BROKER_ACCOUNT_ID)
    payload.setdefault("created_at_utc", _utc_now_iso())
    payload.setdefault("updated_at_utc", _utc_now_iso())
    payload.setdefault("orders", {})
    if not isinstance(payload["orders"], dict):
        raise RuntimeError(f"broker_log.json orders must be an object: {path}")
    return payload


def _save_broker_log(path: Path, broker_log: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    broker_log["updated_at_utc"] = _utc_now_iso()
    path.write_text(json.dumps(broker_log, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _load_orders_df(path: Path) -> pd.DataFrame:
    _must_exist(path, "orders.csv")
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=REQUIRED_ORDER_COLUMNS)
    missing = [c for c in REQUIRED_ORDER_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(f"orders.csv missing required columns: {missing}")
    return df.copy()


def _select_live_orders(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.loc[df["order_side"].astype(str).str.upper() != "HOLD"].copy()
    out = out.loc[out["delta_shares"].abs() > 1e-12].copy()
    out = out.reset_index(drop=True)
    return out


def _resolve_broker_symbol(symbol: str, symbol_map: Dict[str, str]) -> str:
    return symbol_map.get(_normalize_symbol(symbol), _normalize_symbol(symbol))


def _make_idempotency_key(config: str, row: Dict[str, Any], broker_symbol: str) -> str:
    raw = {
        "config": config,
        "date": str(row.get("date", "")).strip(),
        "symbol": _normalize_symbol(str(row.get("symbol", ""))),
        "broker_symbol": broker_symbol,
        "order_side": _normalize_order_side(str(row.get("order_side", "HOLD"))),
        "delta_shares": round(_to_float(row.get("delta_shares", 0.0)), 8),
        "order_notional": round(_to_float(row.get("order_notional", 0.0)), 8),
        "target_weight": round(_to_float(row.get("target_weight", 0.0)), 8),
    }
    payload = json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _make_client_tag(idempotency_key: str) -> str:
    return f"pe-{idempotency_key[:24]}"


def _prepare_orders(config_name: str, df: pd.DataFrame, symbol_map: Dict[str, str]) -> List[PreparedOrder]:
    prepared: List[PreparedOrder] = []
    for _, row in df.iterrows():
        row_dict = {str(k): row[k] for k in df.columns}
        symbol = _normalize_symbol(str(row_dict.get("symbol", "")))
        order_side = _normalize_order_side(str(row_dict.get("order_side", "HOLD")))
        raw_qty = abs(_to_float(row_dict.get("delta_shares", 0.0)))
        if raw_qty <= 1e-12:
            continue

        qty = raw_qty
        if not IB_ALLOW_FRACTIONAL:
            rounded_qty = float(int(raw_qty))
            if abs(raw_qty - rounded_qty) > 1e-9:
                print(
                    f"[FRACTIONAL_BLOCKED] config={config_name} symbol={symbol} side={order_side} raw_qty={raw_qty:.8f} rounded_qty={rounded_qty:.8f}"
                )
            qty = rounded_qty

        if qty <= 1e-12:
            print(f"[SKIP_ZERO_QTY] config={config_name} symbol={symbol} side={order_side} raw_qty={raw_qty:.8f}")
            continue

        broker_symbol = _resolve_broker_symbol(symbol, symbol_map)
        row_for_key = dict(row_dict)
        row_for_key["delta_shares"] = qty if order_side == "BUY" else -qty
        idempotency_key = _make_idempotency_key(config_name, row_for_key, broker_symbol)
        prepared.append(
            PreparedOrder(
                config=config_name,
                order_date=str(row_dict.get("date", "")).strip(),
                symbol=symbol,
                broker_symbol=broker_symbol,
                order_side=order_side,
                qty=qty,
                price=_to_float(row_dict.get("price", 0.0)),
                order_notional=abs(_to_float(row_dict.get("order_notional", 0.0))),
                target_weight=_to_float(row_dict.get("target_weight", 0.0)),
                current_shares=_to_float(row_dict.get("current_shares", 0.0)),
                target_shares=_to_float(row_dict.get("target_shares", 0.0)),
                delta_shares=_to_float(row_dict.get("delta_shares", 0.0)),
                source_row=row_dict,
                idempotency_key=idempotency_key,
                client_tag=_make_client_tag(idempotency_key),
            )
        )
    return prepared


def _existing_duplicate_status(broker_log: dict, idempotency_key: str) -> Optional[str]:
    entry = broker_log.get("orders", {}).get(idempotency_key)
    if not isinstance(entry, dict):
        return None
    status = str(entry.get("status", "")).strip().lower()
    if status in FINAL_DUPLICATE_STATUSES:
        return status
    return None


def _build_contract(prepared: PreparedOrder) -> Contract:
    contract = Contract()
    contract.symbol = prepared.broker_symbol
    contract.secType = IB_SECURITY_TYPE
    contract.exchange = IB_EXCHANGE
    contract.currency = IB_CURRENCY
    if IB_PRIMARY_EXCHANGE:
        contract.primaryExchange = IB_PRIMARY_EXCHANGE
    return contract


def _build_order(prepared: PreparedOrder) -> Order:
    order = Order()
    order.action = prepared.order_side
    order.orderType = IB_ORDER_TYPE
    order.totalQuantity = float(prepared.qty)
    order.tif = IB_TIME_IN_FORCE
    order.outsideRth = bool(IB_OUTSIDE_RTH)
    if IB_ACCOUNT_CODE:
        order.account = IB_ACCOUNT_CODE
    order.orderRef = prepared.client_tag
    order.transmit = True
    order.eTradeOnly = False
    order.firmQuoteOnly = False
    if order.orderType != "MKT":
        raise RuntimeError("Minimal IBKR adapter currently supports only IB_ORDER_TYPE=MKT")
    return order


def _refresh_open_orders(app: IBKRApp) -> None:
    app.done_open_orders = False
    app.reqOpenOrders()
    app.wait_until_open_orders_end(timeout_sec=IB_OPEN_ORDERS_TIMEOUT_SEC)


def _refresh_positions(app: IBKRApp, required: bool) -> bool:
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
        msg = f"Timed out waiting for positionEnd after {IB_POSITIONS_TIMEOUT_SEC:.1f}s"
        if required:
            raise TimeoutError(msg)
        print(f"[IB][WARN] {msg}; continuing without initial position snapshot")
        return False


def _query_contract_details(app: IBKRApp, prepared: PreparedOrder, req_id: int) -> dict:
    app.done_contract_details[int(req_id)] = False
    app.reqContractDetails(int(req_id), _build_contract(prepared))
    return app.wait_for_contract_details(int(req_id), timeout_sec=IB_TIMEOUT_SEC)


def _entry_from_ib(prepared: PreparedOrder, ib_entry: dict, source_path: Path) -> dict:
    status = str(ib_entry.get("status", "")).strip().lower() or "unknown"
    avg_fill_price = _to_float(ib_entry.get("avg_fill_price", prepared.price))
    if avg_fill_price <= 1e-12:
        avg_fill_price = float(prepared.price)
    filled_qty = _to_float(ib_entry.get("filled_qty", 0.0))
    remaining_qty = _to_float(ib_entry.get("remaining_qty", max(0.0, prepared.qty - filled_qty)))
    fill_notional = abs(filled_qty * avg_fill_price)
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
        "filled_avg_price": float(avg_fill_price),
        "order_notional": float(prepared.order_notional),
        "fill_notional": float(fill_notional),
        "status": status,
        "submitted_at": str(ib_entry.get("submitted_at_utc", _utc_now_iso())),
        "filled_at": _utc_now_iso() if status in {"filled", "partiallyfilled"} else "",
        "mode": "ibkr_gateway",
        "request": {
            "account": IB_ACCOUNT_CODE,
            "host": IB_HOST,
            "port": IB_PORT,
            "client_id": IB_CLIENT_ID,
            "order_type": IB_ORDER_TYPE,
            "tif": IB_TIME_IN_FORCE,
            "outside_rth": int(IB_OUTSIDE_RTH),
        },
        "response": {
            "ib_order_id": ib_entry.get("ib_order_id", ""),
            "perm_id": ib_entry.get("perm_id", 0),
            "status": ib_entry.get("status", ""),
            "avg_fill_price": ib_entry.get("avg_fill_price", 0.0),
            "last_fill_price": ib_entry.get("last_fill_price", 0.0),
            "fills": ib_entry.get("fills", []),
        },
    }


def _upsert_broker_log_entry(broker_log: dict, entry: dict) -> None:
    broker_log.setdefault("orders", {})[entry["idempotency_key"]] = {
        "status": entry["status"],
        "client_order_id": entry["client_order_id"],
        "broker_order_id": entry["broker_order_id"],
        "perm_id": entry.get("perm_id", 0),
        "config": entry["config"],
        "date": entry["date"],
        "symbol": entry["symbol"],
        "broker_symbol": entry["broker_symbol"],
        "side": entry["side"],
        "qty": entry["qty"],
        "filled_qty": entry["filled_qty"],
        "remaining_qty": entry.get("remaining_qty", 0.0),
        "filled_avg_price": entry["filled_avg_price"],
        "order_notional": entry["order_notional"],
        "fill_notional": entry["fill_notional"],
        "submitted_at": entry["submitted_at"],
        "filled_at": entry["filled_at"],
        "mode": entry["mode"],
        "source_order_path": entry["source_order_path"],
        "request": entry.get("request", {}),
        "response": entry.get("response", {}),
        "updated_at_utc": _utc_now_iso(),
    }


def _append_or_replace_fills(fills_csv: Path, entries: List[dict]) -> None:
    cols = [
        "idempotency_key", "client_order_id", "broker_order_id", "perm_id", "config", "source_order_path",
        "date", "symbol", "broker_symbol", "side", "qty", "filled_qty", "remaining_qty", "price_hint",
        "filled_avg_price", "order_notional", "fill_notional", "status", "submitted_at", "filled_at", "mode",
    ]
    new_df = pd.DataFrame(entries)
    if new_df.empty:
        if not fills_csv.exists():
            pd.DataFrame(columns=cols).to_csv(fills_csv, index=False)
        return
    new_df = new_df[cols].copy()
    if fills_csv.exists():
        prev = pd.read_csv(fills_csv)
        for col in cols:
            if col not in prev.columns:
                prev[col] = ""
        prev = prev[cols].copy()
        out = pd.concat([prev, new_df], ignore_index=True)
        out = out.drop_duplicates(subset=["idempotency_key"], keep="last")
    else:
        out = new_df
    out = out.sort_values(["date", "symbol", "side", "idempotency_key"]).reset_index(drop=True)
    out.to_csv(fills_csv, index=False)


def _duplicate_fill_entry(prepared: PreparedOrder, duplicate_status: str, source_path: Path, broker_log: dict) -> dict:
    existing = broker_log.get("orders", {}).get(prepared.idempotency_key, {})
    return {
        "idempotency_key": prepared.idempotency_key,
        "client_order_id": str(existing.get("client_order_id", prepared.client_tag)),
        "broker_order_id": str(existing.get("broker_order_id", "")),
        "perm_id": int(existing.get("perm_id", 0) or 0),
        "config": prepared.config,
        "source_order_path": str(source_path),
        "date": prepared.order_date,
        "symbol": prepared.symbol,
        "broker_symbol": prepared.broker_symbol,
        "side": prepared.order_side,
        "qty": float(prepared.qty),
        "filled_qty": _to_float(existing.get("filled_qty", 0.0)),
        "remaining_qty": _to_float(existing.get("remaining_qty", 0.0)),
        "price_hint": float(prepared.price),
        "filled_avg_price": _to_float(existing.get("filled_avg_price", prepared.price)),
        "order_notional": float(prepared.order_notional),
        "fill_notional": _to_float(existing.get("fill_notional", 0.0)),
        "status": f"duplicate_skipped:{duplicate_status}",
        "submitted_at": str(existing.get("submitted_at", "")),
        "filled_at": str(existing.get("filled_at", "")),
        "mode": "ibkr_gateway",
    }


def _bootstrap_connection() -> IBKRApp:
    app = IBKRApp()
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
    _refresh_open_orders(app)
    if IB_REFRESH_POSITIONS_ON_CONNECT or IB_REQUIRE_POSITIONS_ON_CONNECT:
        got_positions = _refresh_positions(app, required=IB_REQUIRE_POSITIONS_ON_CONNECT)
        print(
            f"[IB] initial positions refresh enabled={int(IB_REFRESH_POSITIONS_ON_CONNECT or IB_REQUIRE_POSITIONS_ON_CONNECT)} success={int(got_positions)}"
        )
    else:
        print("[IB] initial positions refresh skipped by config")
    return app


def _teardown_connection(app: IBKRApp) -> None:
    try:
        if app.isConnected():
            app.disconnect()
    except Exception:
        pass
    time.sleep(0.25)


def _run_one_config(app: IBKRApp, paths: ConfigPaths, symbol_map: Dict[str, str], req_id_seed: int) -> int:
    print(f"[BROKER][{paths.name}] orders={paths.orders_csv}")
    paths.execution_dir.mkdir(parents=True, exist_ok=True)
    broker_log = _load_broker_log(paths.broker_log_json, paths.name)
    orders_df = _load_orders_df(paths.orders_csv)
    live_orders_df = _select_live_orders(orders_df)
    prepared_orders = _prepare_orders(paths.name, live_orders_df, symbol_map)

    print(f"[BROKER][{paths.name}] total_rows={len(orders_df)} live_orders={len(live_orders_df)} prepared_orders={len(prepared_orders)}")
    if len(prepared_orders):
        preview_df = pd.DataFrame([
            {
                "symbol": x.symbol,
                "broker_symbol": x.broker_symbol,
                "side": x.order_side,
                "qty": x.qty,
                "order_notional": x.order_notional,
                "client_tag": x.client_tag,
            }
            for x in prepared_orders
        ])
        print(f"[BROKER][{paths.name}][PREVIEW]")
        print(preview_df.head(min(TOPK_PRINT, len(preview_df))).to_string(index=False))

    fill_entries: List[dict] = []
    manual_review_entries: List[dict] = []
    sent_count = 0
    dup_count = 0
    err_count = 0
    fractional_reject_count = 0
    req_cursor = int(req_id_seed)
    manual_review_csv = paths.execution_dir / "broker_fractional_manual_review.csv"

    for prepared in prepared_orders:
        duplicate_status = _existing_duplicate_status(broker_log, prepared.idempotency_key)
        if duplicate_status is not None:
            dup_count += 1
            fill_entries.append(_duplicate_fill_entry(prepared, duplicate_status, paths.orders_csv, broker_log))
            print(f"[BROKER][{paths.name}][SKIP] symbol={prepared.symbol} side={prepared.order_side} qty={prepared.qty:.8f} duplicate_status={duplicate_status}")
            continue

        try:
            req_cursor += 1
            contract_meta = _query_contract_details(app, prepared, req_cursor)
            if not contract_meta:
                raise RuntimeError(f"No contract details returned for symbol={prepared.broker_symbol}")
            contract = _build_contract(prepared)
            if contract_meta.get("primaryExchange") and not IB_PRIMARY_EXCHANGE:
                contract.primaryExchange = str(contract_meta.get("primaryExchange", ""))
            order = _build_order(prepared)
            ib_order_id = app.allocate_order_id()
            error_cursor_before_submit = len(app._errors)
            app.placeOrder(ib_order_id, contract, order)
            ib_entry = app.wait_for_order_terminalish(ib_order_id, timeout_sec=IB_TIMEOUT_SEC)

            if IB_POLL_AFTER_SUBMIT:
                for _ in range(max(0, IB_POLL_ATTEMPTS)):
                    time.sleep(IB_POLL_SLEEP_SEC)
                    _refresh_open_orders(app)
                    latest = app.orders_by_ib_id.get(int(ib_order_id), ib_entry)
                    status = str(latest.get("status", "")).strip().lower()
                    ib_entry = latest
                    if status in {"filled", "submitted", "presubmitted", "partiallyfilled", "inactive", "cancelled"}:
                        break

            broker_errors = _collect_app_errors(app, error_cursor_before_submit, [int(ib_order_id), int(req_cursor)])
            fractional_issue = _first_fractional_issue(prepared, broker_errors)
            if fractional_issue is not None:
                fractional_reject_count += 1
                err_count += 1
                issue_entry = _issue_entry(prepared, fractional_issue, paths.orders_csv)
                _upsert_broker_log_entry(broker_log, issue_entry)
                fill_entries.append(issue_entry)
                manual_review_entries.append(_manual_review_entry(prepared, fractional_issue, paths.orders_csv))
                print(
                    f"[BROKER][{paths.name}][FRACTIONAL_REJECT] symbol={prepared.symbol} broker_symbol={prepared.broker_symbol} side={prepared.order_side} qty={prepared.qty:.8f} code={fractional_issue.broker_error.error_code if fractional_issue.broker_error else 0} msg={fractional_issue.broker_error.error_string if fractional_issue.broker_error else fractional_issue.message}"
                )
                continue

            latest = app.orders_by_ib_id.get(int(ib_order_id), ib_entry)
            latest.setdefault("submitted_at_utc", _utc_now_iso())
            generic_issue = _first_generic_issue(broker_errors)
            status_lower = str(latest.get("status", "")).strip().lower()
            if generic_issue is not None and status_lower not in {"filled", "submitted", "presubmitted", "partiallyfilled"}:
                err_count += 1
                issue_entry = _issue_entry(prepared, generic_issue, paths.orders_csv)
                _upsert_broker_log_entry(broker_log, issue_entry)
                fill_entries.append(issue_entry)
                print(
                    f"[BROKER][{paths.name}][API_ERROR] symbol={prepared.symbol} broker_symbol={prepared.broker_symbol} side={prepared.order_side} qty={prepared.qty:.8f} code={generic_issue.broker_error.error_code if generic_issue.broker_error else 0} msg={generic_issue.broker_error.error_string if generic_issue.broker_error else generic_issue.message}"
                )
                continue

            entry = _entry_from_ib(prepared, latest, paths.orders_csv)
            _upsert_broker_log_entry(broker_log, entry)
            fill_entries.append(entry)
            sent_count += 1
            print(
                f"[BROKER][{paths.name}][SEND] symbol={prepared.symbol} broker_symbol={prepared.broker_symbol} side={prepared.order_side} qty={prepared.qty:.8f} status={entry['status']} ib_order_id={entry['broker_order_id']}"
            )
        except Exception as exc:
            err_count += 1
            error_entry = {
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
                "filled_avg_price": 0.0,
                "order_notional": float(prepared.order_notional),
                "fill_notional": 0.0,
                "status": f"error:{type(exc).__name__}",
                "submitted_at": _utc_now_iso(),
                "filled_at": "",
                "mode": "ibkr_gateway",
                "request": {"account": IB_ACCOUNT_CODE, "host": IB_HOST, "port": IB_PORT, "client_id": IB_CLIENT_ID},
                "response": {"error": str(exc)},
            }
            _upsert_broker_log_entry(broker_log, error_entry)
            fill_entries.append(error_entry)
            print(f"[BROKER][{paths.name}][ERROR] symbol={prepared.symbol} side={prepared.order_side} qty={prepared.qty:.8f} error={exc}")

    _append_or_replace_fills(paths.fills_csv, fill_entries)
    _append_manual_review_csv(manual_review_csv, manual_review_entries)
    _save_broker_log(paths.broker_log_json, broker_log)

    print(
        f"[BROKER][{paths.name}][SUMMARY] sent={sent_count} duplicate_skipped={dup_count} errors={err_count} fractional_rejected={fractional_reject_count} fills_csv={paths.fills_csv} broker_log_json={paths.broker_log_json} manual_review_csv={manual_review_csv}"
    )
    return req_cursor


def main() -> int:
    _enable_line_buffering()
    symbol_map = _load_symbol_map()
    print(f"[CFG] execution_root={EXECUTION_ROOT}")
    print(f"[CFG] broker_name={BROKER_NAME} broker_platform={BROKER_PLATFORM}")
    print(f"[CFG] configs={CONFIG_NAMES}")
    print(f"[CFG] ib_host={IB_HOST} ib_port={IB_PORT} ib_client_id={IB_CLIENT_ID}")
    print(f"[CFG] ib_account_code={IB_ACCOUNT_CODE}")
    print(f"[CFG] ib_order_type={IB_ORDER_TYPE} ib_time_in_force={IB_TIME_IN_FORCE} ib_outside_rth={int(IB_OUTSIDE_RTH)}")
    print(f"[CFG] ib_exchange={IB_EXCHANGE} ib_primary_exchange={IB_PRIMARY_EXCHANGE} ib_currency={IB_CURRENCY} ib_security_type={IB_SECURITY_TYPE}")
    print(f"[CFG] ib_allow_fractional={int(IB_ALLOW_FRACTIONAL)}")
    print(f"[CFG] ib_fractional_reject_codes={IB_FRACTIONAL_REJECT_CODES} ib_fractional_failure_mode={IB_FRACTIONAL_FAILURE_MODE}")
    print(
        f"[CFG] ib_refresh_positions_on_connect={int(IB_REFRESH_POSITIONS_ON_CONNECT)} ib_require_positions_on_connect={int(IB_REQUIRE_POSITIONS_ON_CONNECT)} ib_positions_timeout_sec={IB_POSITIONS_TIMEOUT_SEC}"
    )
    if IB_ALLOW_FRACTIONAL and IB_EXCHANGE != "SMART":
        print(f"[CFG][WARN] IB_ALLOW_FRACTIONAL=1 but IB_EXCHANGE={IB_EXCHANGE}; IBKR fractional routing often requires SMART routing.")
    print(f"[CFG] symbol_map_entries={len(symbol_map)}")

    app = _bootstrap_connection()
    req_cursor = 100000
    try:
        for config_name in CONFIG_NAMES:
            paths = _cfg_paths(config_name)
            req_cursor = _run_one_config(app, paths, symbol_map, req_cursor)
    finally:
        _teardown_connection(app)

    print("[FINAL] IBKR broker adapter complete")
    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 1
    _safe_exit(rc)
