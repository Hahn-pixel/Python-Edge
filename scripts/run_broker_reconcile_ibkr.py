from __future__ import annotations

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
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    from ibapi.client import EClient
    from ibapi.common import OrderId
    from ibapi.contract import Contract
    from ibapi.order import Order
    from ibapi.wrapper import EWrapper
except Exception as import_exc:
    raise RuntimeError(
        "Failed to import ibapi. Install IB API first, for example: pip install ibapi"
    ) from import_exc


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
for p in [ROOT, SRC_DIR]:
    ps = str(p)
    if ps not in sys.path:
        sys.path.insert(0, ps)


# Double-click runnable.
# Never auto-close.


def _env_flag(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip().lower() not in {"0", "false", "no", "off", ""}


def _enable_line_buffering() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    try:
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass


def _should_pause() -> bool:
    if _env_flag("PYTHON_EDGE_NO_PAUSE", "0"):
        return False
    return True


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
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _num_scalar(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        if isinstance(value, str) and not value.strip():
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _norm_symbol(value: Any) -> str:
    return str(value or "").strip().upper()


def _safe_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("w", encoding=encoding, newline="") as fh:
        fh.write(text)
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except Exception:
            pass
    os.replace(tmp, path)


RECONCILE_CONFIG_NAMES = [
    x.strip()
    for x in str(os.getenv("RECONCILE_CONFIG_NAMES", os.getenv("CONFIG_NAMES", "optimal|aggressive"))).split("|")
    if x.strip()
]
EXECUTION_ROOT = Path(os.getenv("EXECUTION_ROOT", "artifacts/execution_loop"))
IB_HOST = str(os.getenv("IB_HOST", "127.0.0.1")).strip()
IB_PORT = int(os.getenv("IB_PORT", "4002"))
IB_CLIENT_ID = int(os.getenv("RECONCILE_IB_CLIENT_ID", os.getenv("IB_CLIENT_ID", "42")))
IB_TIMEOUT_SEC = float(os.getenv("IB_TIMEOUT_SEC", "20.0"))
IB_POSITIONS_TIMEOUT_SEC = float(os.getenv("IB_POSITIONS_TIMEOUT_SEC", str(max(IB_TIMEOUT_SEC, 45.0))))
IB_POSITIONS_RETRIES = int(os.getenv("IB_POSITIONS_RETRIES", "2"))
IB_POSITIONS_SETTLE_SEC = float(os.getenv("IB_POSITIONS_SETTLE_SEC", "2.0"))
BROKER_ACCOUNT_ID = str(os.getenv("BROKER_ACCOUNT_ID", "")).strip()
IB_ACCOUNT_CODE = str(os.getenv("IB_ACCOUNT_CODE", BROKER_ACCOUNT_ID)).strip()
TOPK_PRINT = int(os.getenv("TOPK_PRINT", "50"))
REQ_OPEN_ORDERS_MODE = str(os.getenv("RECONCILE_OPEN_ORDERS_MODE", "open")).strip().lower()


@dataclass(frozen=True)
class ConfigPaths:
    name: str
    execution_dir: Path
    state_json: Path
    orders_csv: Path
    report_json: Path
    diff_csv: Path
    broker_positions_csv: Path
    broker_open_orders_csv: Path


@dataclass(frozen=True)
class LocalExpectedSource:
    source: str
    rows: pd.DataFrame


class IBKRReconApp(EWrapper, EClient):
    def __init__(self) -> None:
        EClient.__init__(self, self)
        self._network_thread: Optional[threading.Thread] = None
        self._next_valid_id_queue: queue.Queue[int] = queue.Queue()
        self._managed_accounts_queue: queue.Queue[str] = queue.Queue()
        self._errors: List[Dict[str, Any]] = []
        self.done_open_orders = False
        self.done_positions = False
        self.positions_request_started_at: Optional[float] = None
        self.positions_last_callback_at: Optional[float] = None
        self.open_orders_last_callback_at: Optional[float] = None
        self.orders_by_ib_id: Dict[int, Dict[str, Any]] = {}
        self.position_rows: List[Dict[str, Any]] = []

    def start_network_loop(self) -> None:
        if self._network_thread is not None and self._network_thread.is_alive():
            return
        self._network_thread = threading.Thread(target=self.run, name="IBKRReconNetwork", daemon=True)
        self._network_thread.start()

    def error(self, reqId: int, errorCode: int, errorString: str, advancedOrderRejectJson: str = "") -> None:
        payload = {
            "req_id": int(reqId or 0),
            "error_code": int(errorCode or 0),
            "error_string": str(errorString or ""),
            "advanced": str(advancedOrderRejectJson or ""),
            "ts": _utc_now_iso(),
        }
        self._errors.append(payload)
        print(f"[IB][ERROR] reqId={payload['req_id']} code={payload['error_code']} msg={payload['error_string']}")

    def connectAck(self) -> None:
        print("[IB] connectAck received")

    def nextValidId(self, orderId: int) -> None:
        try:
            self._next_valid_id_queue.put_nowait(int(orderId))
        except Exception:
            pass
        print(f"[IB] nextValidId={int(orderId)}")

    def managedAccounts(self, accountsList: str) -> None:
        try:
            self._managed_accounts_queue.put_nowait(str(accountsList or ""))
        except Exception:
            pass
        print(f"[IB] managedAccounts={accountsList}")

    def openOrder(self, orderId: OrderId, contract: Contract, order: Order, orderState) -> None:
        self.open_orders_last_callback_at = time.time()
        entry = self.orders_by_ib_id.setdefault(
            int(orderId),
            {
                "ib_order_id": int(orderId),
                "account": str(getattr(order, "account", "") or ""),
                "symbol": str(getattr(contract, "symbol", "") or ""),
                "broker_symbol": str(getattr(contract, "localSymbol", "") or getattr(contract, "symbol", "") or ""),
                "sec_type": str(getattr(contract, "secType", "") or ""),
                "currency": str(getattr(contract, "currency", "") or ""),
                "exchange": str(getattr(contract, "exchange", "") or ""),
                "primary_exchange": str(getattr(contract, "primaryExchange", "") or ""),
                "status": str(getattr(orderState, "status", "") or ""),
                "action": str(getattr(order, "action", "") or ""),
                "total_qty": _num_scalar(getattr(order, "totalQuantity", 0.0)),
                "filled_qty": 0.0,
                "remaining_qty": _num_scalar(getattr(order, "totalQuantity", 0.0)),
                "lmt_price": _num_scalar(getattr(order, "lmtPrice", 0.0)),
                "aux_price": _num_scalar(getattr(order, "auxPrice", 0.0)),
                "order_type": str(getattr(order, "orderType", "") or ""),
                "tif": str(getattr(order, "tif", "") or ""),
                "outside_rth": int(bool(getattr(order, "outsideRth", False))),
                "perm_id": int(getattr(order, "permId", 0) or 0),
                "client_tag": str(getattr(order, "orderRef", "") or ""),
                "submitted_at_utc": _utc_now_iso(),
                "avg_fill_price": 0.0,
                "last_fill_price": 0.0,
            },
        )
        entry["account"] = str(getattr(order, "account", "") or entry.get("account", ""))
        entry["symbol"] = str(getattr(contract, "symbol", "") or entry.get("symbol", ""))
        entry["broker_symbol"] = str(getattr(contract, "localSymbol", "") or getattr(contract, "symbol", "") or entry.get("broker_symbol", ""))
        entry["sec_type"] = str(getattr(contract, "secType", "") or entry.get("sec_type", ""))
        entry["currency"] = str(getattr(contract, "currency", "") or entry.get("currency", ""))
        entry["exchange"] = str(getattr(contract, "exchange", "") or entry.get("exchange", ""))
        entry["primary_exchange"] = str(getattr(contract, "primaryExchange", "") or entry.get("primary_exchange", ""))
        entry["status"] = str(getattr(orderState, "status", "") or entry.get("status", ""))
        entry["action"] = str(getattr(order, "action", "") or entry.get("action", ""))
        entry["total_qty"] = _num_scalar(getattr(order, "totalQuantity", entry.get("total_qty", 0.0)))
        entry["lmt_price"] = _num_scalar(getattr(order, "lmtPrice", entry.get("lmt_price", 0.0)))
        entry["aux_price"] = _num_scalar(getattr(order, "auxPrice", entry.get("aux_price", 0.0)))
        entry["order_type"] = str(getattr(order, "orderType", "") or entry.get("order_type", ""))
        entry["tif"] = str(getattr(order, "tif", "") or entry.get("tif", ""))
        entry["outside_rth"] = int(bool(getattr(order, "outsideRth", entry.get("outside_rth", False))))
        entry["perm_id"] = int(getattr(order, "permId", entry.get("perm_id", 0)) or 0)
        entry["client_tag"] = str(getattr(order, "orderRef", "") or entry.get("client_tag", ""))

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
                "account": "",
                "symbol": "",
                "broker_symbol": "",
                "sec_type": "",
                "currency": "",
                "exchange": "",
                "primary_exchange": "",
                "status": str(status or ""),
                "action": "",
                "total_qty": float(filled or 0.0) + float(remaining or 0.0),
                "filled_qty": float(filled or 0.0),
                "remaining_qty": float(remaining or 0.0),
                "lmt_price": 0.0,
                "aux_price": 0.0,
                "order_type": "",
                "tif": "",
                "outside_rth": 0,
                "perm_id": int(permId or 0),
                "client_tag": "",
                "submitted_at_utc": _utc_now_iso(),
                "avg_fill_price": float(avgFillPrice or 0.0),
                "last_fill_price": float(lastFillPrice or 0.0),
            },
        )
        entry["status"] = str(status or entry.get("status", ""))
        entry["filled_qty"] = float(filled or 0.0)
        entry["remaining_qty"] = float(remaining or 0.0)
        entry["avg_fill_price"] = float(avgFillPrice or 0.0)
        entry["last_fill_price"] = float(lastFillPrice or 0.0)
        entry["perm_id"] = int(permId or entry.get("perm_id", 0) or 0)

    def position(self, account: str, contract: Contract, position: float, avgCost: float) -> None:
        self.positions_last_callback_at = time.time()
        self.position_rows.append(
            {
                "account": str(account or ""),
                "symbol": str(getattr(contract, "symbol", "") or ""),
                "broker_symbol": str(getattr(contract, "localSymbol", "") or getattr(contract, "symbol", "") or ""),
                "sec_type": str(getattr(contract, "secType", "") or ""),
                "currency": str(getattr(contract, "currency", "") or ""),
                "exchange": str(getattr(contract, "exchange", "") or ""),
                "primary_exchange": str(getattr(contract, "primaryExchange", "") or ""),
                "position": _num_scalar(position),
                "avg_cost": _num_scalar(avgCost),
            }
        )

    def positionEnd(self) -> None:
        self.done_positions = True
        self.positions_last_callback_at = time.time()
        print("[IB] positionEnd")

    def wait_for_next_valid_id(self, timeout_sec: float) -> int:
        return int(self._next_valid_id_queue.get(timeout=timeout_sec))

    def wait_for_managed_accounts(self, timeout_sec: float) -> str:
        return str(self._managed_accounts_queue.get(timeout=timeout_sec))

    def wait_until_open_orders_end(self, timeout_sec: float) -> None:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            if self.done_open_orders:
                return
            time.sleep(0.1)
        raise TimeoutError("Timed out waiting for openOrderEnd")

    def wait_until_positions_end(self, timeout_sec: float, settle_sec: float) -> None:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            if self.done_positions:
                return
            if self.positions_request_started_at is not None and self.positions_last_callback_at is not None:
                if self.positions_last_callback_at >= self.positions_request_started_at:
                    idle_sec = time.time() - self.positions_last_callback_at
                    if idle_sec >= settle_sec:
                        print(
                            "[IB][WARN] positionEnd not received; using settled position snapshot "
                            f"rows={len(self.position_rows)} idle_sec={idle_sec:.2f}"
                        )
                        return
            time.sleep(0.1)
        raise TimeoutError("Timed out waiting for positionEnd")



def _config_paths(name: str) -> ConfigPaths:
    execution_dir = EXECUTION_ROOT / name
    return ConfigPaths(
        name=name,
        execution_dir=execution_dir,
        state_json=execution_dir / "portfolio_state.json",
        orders_csv=execution_dir / "orders.csv",
        report_json=execution_dir / "reconcile_report.json",
        diff_csv=execution_dir / "reconcile_diff.csv",
        broker_positions_csv=execution_dir / "broker_positions.csv",
        broker_open_orders_csv=execution_dir / "broker_open_orders.csv",
    )



def _bootstrap_connection() -> IBKRReconApp:
    app = IBKRReconApp()
    print(f"[IB] connecting host={IB_HOST} port={IB_PORT} client_id={IB_CLIENT_ID}")
    app.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID)
    app.start_network_loop()
    _ = app.wait_for_next_valid_id(timeout_sec=IB_TIMEOUT_SEC)
    managed_accounts = app.wait_for_managed_accounts(timeout_sec=IB_TIMEOUT_SEC)
    if IB_ACCOUNT_CODE:
        account_list = [x.strip() for x in managed_accounts.split(",") if x.strip()]
        if IB_ACCOUNT_CODE not in account_list:
            raise RuntimeError(
                f"IB_ACCOUNT_CODE={IB_ACCOUNT_CODE!r} not present in managed accounts: {managed_accounts!r}"
            )
    return app



def _refresh_open_orders(app: IBKRReconApp) -> pd.DataFrame:
    app.done_open_orders = False
    app.open_orders_last_callback_at = None
    if REQ_OPEN_ORDERS_MODE == "all":
        app.reqAllOpenOrders()
    else:
        app.reqOpenOrders()
    app.wait_until_open_orders_end(timeout_sec=IB_TIMEOUT_SEC)
    rows = list(app.orders_by_ib_id.values())
    df = pd.DataFrame(rows)
    if not len(df):
        return pd.DataFrame(
            columns=[
                "account", "symbol", "broker_symbol", "sec_type", "currency", "exchange",
                "primary_exchange", "status", "action", "total_qty", "filled_qty", "remaining_qty",
                "lmt_price", "aux_price", "order_type", "tif", "outside_rth", "perm_id",
                "client_tag", "ib_order_id", "submitted_at_utc", "avg_fill_price", "last_fill_price",
            ]
        )
    if IB_ACCOUNT_CODE and "account" in df.columns:
        df = df[df["account"].astype(str).str.strip() == IB_ACCOUNT_CODE].copy()
    df["symbol"] = df.get("symbol", pd.Series(dtype="object")).astype(str).str.upper()
    df["broker_symbol"] = df.get("broker_symbol", pd.Series(dtype="object")).astype(str)
    df = df.sort_values(["symbol", "ib_order_id"], ascending=[True, True]).reset_index(drop=True)
    return df



def _refresh_positions_once(app: IBKRReconApp) -> pd.DataFrame:
    app.done_positions = False
    app.position_rows = []
    app.positions_request_started_at = time.time()
    app.positions_last_callback_at = None
    app.reqPositions()
    app.wait_until_positions_end(timeout_sec=IB_POSITIONS_TIMEOUT_SEC, settle_sec=IB_POSITIONS_SETTLE_SEC)
    try:
        app.cancelPositions()
    except Exception:
        pass

    df = pd.DataFrame(app.position_rows)
    if not len(df):
        return pd.DataFrame(
            columns=[
                "account", "symbol", "broker_symbol", "sec_type", "currency", "exchange",
                "primary_exchange", "position", "avg_cost",
            ]
        )
    if IB_ACCOUNT_CODE and "account" in df.columns:
        df = df[df["account"].astype(str).str.strip() == IB_ACCOUNT_CODE].copy()
    df["symbol"] = df.get("symbol", pd.Series(dtype="object")).astype(str).str.upper()
    df["broker_symbol"] = df.get("broker_symbol", pd.Series(dtype="object")).astype(str)
    df["position"] = pd.to_numeric(df.get("position", 0.0), errors="coerce").fillna(0.0)
    df["avg_cost"] = pd.to_numeric(df.get("avg_cost", 0.0), errors="coerce").fillna(0.0)
    df = (
        df.groupby(["account", "symbol", "broker_symbol", "sec_type", "currency", "exchange", "primary_exchange"], dropna=False, as_index=False)
        .agg(position=("position", "sum"), avg_cost=("avg_cost", "last"))
        .sort_values(["symbol", "broker_symbol"], ascending=[True, True])
        .reset_index(drop=True)
    )
    return df



def _refresh_positions(app: IBKRReconApp) -> pd.DataFrame:
    last_exc: Optional[Exception] = None
    for attempt in range(1, IB_POSITIONS_RETRIES + 1):
        try:
            print(
                f"[IB][POSITIONS] attempt={attempt}/{IB_POSITIONS_RETRIES} "
                f"timeout_sec={IB_POSITIONS_TIMEOUT_SEC} settle_sec={IB_POSITIONS_SETTLE_SEC}"
            )
            df = _refresh_positions_once(app)
            print(f"[IB][POSITIONS] rows={len(df)}")
            return df
        except TimeoutError as exc:
            last_exc = exc
            try:
                app.cancelPositions()
            except Exception:
                pass
            received_rows = len(app.position_rows)
            print(
                f"[IB][WARN] reqPositions timeout attempt={attempt}/{IB_POSITIONS_RETRIES} "
                f"received_rows={received_rows}"
            )
            if received_rows > 0:
                print("[IB][WARN] Using partial position snapshot because at least one position callback was received.")
                df = pd.DataFrame(app.position_rows)
                if IB_ACCOUNT_CODE and "account" in df.columns:
                    df = df[df["account"].astype(str).str.strip() == IB_ACCOUNT_CODE].copy()
                if len(df):
                    df["symbol"] = df.get("symbol", pd.Series(dtype="object")).astype(str).str.upper()
                    df["broker_symbol"] = df.get("broker_symbol", pd.Series(dtype="object")).astype(str)
                    df["position"] = pd.to_numeric(df.get("position", 0.0), errors="coerce").fillna(0.0)
                    df["avg_cost"] = pd.to_numeric(df.get("avg_cost", 0.0), errors="coerce").fillna(0.0)
                    df = (
                        df.groupby(["account", "symbol", "broker_symbol", "sec_type", "currency", "exchange", "primary_exchange"], dropna=False, as_index=False)
                        .agg(position=("position", "sum"), avg_cost=("avg_cost", "last"))
                        .sort_values(["symbol", "broker_symbol"], ascending=[True, True])
                        .reset_index(drop=True)
                    )
                else:
                    df = pd.DataFrame(
                        columns=[
                            "account", "symbol", "broker_symbol", "sec_type", "currency", "exchange",
                            "primary_exchange", "position", "avg_cost",
                        ]
                    )
                return df
            if attempt < IB_POSITIONS_RETRIES:
                time.sleep(1.5)
                continue
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Unexpected positions refresh failure")



def _load_portfolio_state(state_json: Path) -> LocalExpectedSource:
    with state_json.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid portfolio state json at {state_json}")
    positions = payload.get("positions", {})
    if positions is None:
        positions = {}
    if not isinstance(positions, dict):
        raise RuntimeError(f"State positions must be a dict at {state_json}")
    rows: List[Dict[str, Any]] = []
    for symbol, pos in positions.items():
        symbol_norm = _norm_symbol(symbol)
        if not symbol_norm:
            continue
        pos_dict = pos if isinstance(pos, dict) else {}
        expected_shares = _num_scalar(pos_dict.get("shares", 0.0))
        rows.append(
            {
                "symbol": symbol_norm,
                "expected_shares": expected_shares,
                "source_priority": 1,
                "source": "portfolio_state.json",
                "target_shares": expected_shares,
                "current_shares": _num_scalar(pos_dict.get("shares", expected_shares)),
                "delta_shares": 0.0,
                "last_price": _num_scalar(pos_dict.get("last_price", 0.0)),
                "market_value": _num_scalar(pos_dict.get("market_value", 0.0)),
            }
        )
    df = pd.DataFrame(rows)
    if not len(df):
        df = pd.DataFrame(columns=[
            "symbol", "expected_shares", "source_priority", "source", "target_shares",
            "current_shares", "delta_shares", "last_price", "market_value",
        ])
    else:
        df = df.sort_values(["symbol"], ascending=[True]).reset_index(drop=True)
    return LocalExpectedSource(source="portfolio_state.json", rows=df)



def _compute_expected_from_orders(orders_csv: Path) -> LocalExpectedSource:
    if not orders_csv.exists():
        raise RuntimeError(f"Missing orders.csv fallback at {orders_csv}")
    df = pd.read_csv(orders_csv)
    if not len(df):
        out = pd.DataFrame(columns=[
            "symbol", "expected_shares", "source_priority", "source", "target_shares",
            "current_shares", "delta_shares", "price", "date", "order_side", "skip_reason",
        ])
        return LocalExpectedSource(source="orders.csv", rows=out)

    df = df.copy()
    if "symbol" not in df.columns:
        raise RuntimeError(f"orders.csv missing required column 'symbol': {orders_csv}")

    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df = df[df["symbol"] != ""].copy()

    for col in ["target_shares", "current_shares", "delta_shares", "price"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "__row_order__" not in df.columns:
        df["__row_order__"] = range(len(df))

    if "date" in df.columns:
        date_rank = pd.to_datetime(df["date"], errors="coerce")
        df["__date_rank__"] = date_rank
    else:
        df["__date_rank__"] = pd.NaT

    df = df.sort_values(["symbol", "__date_rank__", "__row_order__"], ascending=[True, True, True]).reset_index(drop=True)

    rows: List[Dict[str, Any]] = []
    for symbol, g in df.groupby("symbol", sort=True):
        row = g.iloc[-1]
        target_shares_present = "target_shares" in g.columns and pd.notna(row.get("target_shares"))
        current_shares_present = "current_shares" in g.columns and pd.notna(row.get("current_shares"))
        delta_shares_present = "delta_shares" in g.columns and pd.notna(row.get("delta_shares"))

        target_shares = _num_scalar(row.get("target_shares", 0.0)) if target_shares_present else 0.0
        current_shares = _num_scalar(row.get("current_shares", 0.0)) if current_shares_present else 0.0
        delta_shares = _num_scalar(row.get("delta_shares", 0.0)) if delta_shares_present else 0.0

        if target_shares_present:
            expected_shares = target_shares
            source_formula = "target_shares"
        elif current_shares_present and delta_shares_present:
            expected_shares = current_shares + delta_shares
            source_formula = "current_shares + delta_shares"
        elif delta_shares_present:
            expected_shares = delta_shares
            source_formula = "delta_shares"
        else:
            expected_shares = 0.0
            source_formula = "missing_shares_columns"

        rows.append(
            {
                "symbol": symbol,
                "expected_shares": float(expected_shares),
                "source_priority": 2,
                "source": "orders.csv",
                "source_formula": source_formula,
                "target_shares": target_shares if target_shares_present else None,
                "current_shares": current_shares if current_shares_present else None,
                "delta_shares": delta_shares if delta_shares_present else None,
                "price": _num_scalar(row.get("price", 0.0)),
                "date": str(row.get("date", "") or ""),
                "order_side": str(row.get("order_side", "") or ""),
                "skip_reason": str(row.get("skip_reason", "") or ""),
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(["symbol"], ascending=[True]).reset_index(drop=True)
    return LocalExpectedSource(source="orders.csv", rows=out)



def _load_local_expected(paths: ConfigPaths) -> LocalExpectedSource:
    if paths.state_json.exists():
        return _load_portfolio_state(paths.state_json)
    return _compute_expected_from_orders(paths.orders_csv)



def _build_reconcile_diff(local_source: LocalExpectedSource, broker_positions: pd.DataFrame, broker_open_orders: pd.DataFrame) -> pd.DataFrame:
    local_df = local_source.rows.copy()
    if not len(local_df):
        local_df = pd.DataFrame(columns=["symbol", "expected_shares", "source", "source_priority"])
    if "symbol" not in local_df.columns:
        local_df["symbol"] = pd.Series(dtype="object")
    local_df["symbol"] = local_df["symbol"].astype(str).str.upper()
    local_df["expected_shares"] = pd.to_numeric(local_df.get("expected_shares", 0.0), errors="coerce").fillna(0.0)

    broker_positions_agg = broker_positions.copy()
    if not len(broker_positions_agg):
        broker_positions_agg = pd.DataFrame(columns=["symbol", "broker_position"])
    else:
        broker_positions_agg = (
            broker_positions_agg.groupby("symbol", as_index=False)
            .agg(
                broker_position=("position", "sum"),
                broker_avg_cost=("avg_cost", "last"),
                broker_symbol=("broker_symbol", "last"),
                broker_currency=("currency", "last"),
                broker_exchange=("exchange", "last"),
                broker_primary_exchange=("primary_exchange", "last"),
                broker_sec_type=("sec_type", "last"),
                broker_account=("account", "last"),
            )
            .sort_values(["symbol"], ascending=[True])
            .reset_index(drop=True)
        )

    open_orders_agg = broker_open_orders.copy()
    if not len(open_orders_agg):
        open_orders_agg = pd.DataFrame(columns=["symbol", "broker_open_order_count", "broker_open_order_net_qty"])
    else:
        if "remaining_qty" not in open_orders_agg.columns:
            open_orders_agg["remaining_qty"] = 0.0
        open_orders_agg["remaining_qty"] = pd.to_numeric(open_orders_agg["remaining_qty"], errors="coerce").fillna(0.0)
        signed_remaining = open_orders_agg["remaining_qty"].where(
            open_orders_agg.get("action", pd.Series(dtype="object")).astype(str).str.upper() == "BUY",
            -open_orders_agg["remaining_qty"],
        )
        open_orders_agg = open_orders_agg.assign(__signed_remaining__=signed_remaining)
        open_orders_agg = (
            open_orders_agg.groupby("symbol", as_index=False)
            .agg(
                broker_open_order_count=("ib_order_id", "count"),
                broker_open_order_net_qty=("__signed_remaining__", "sum"),
                broker_open_order_remaining_qty=("remaining_qty", "sum"),
                broker_open_order_actions=("action", lambda s: "|".join(sorted({str(x) for x in s if str(x)}))),
                broker_open_order_statuses=("status", lambda s: "|".join(sorted({str(x) for x in s if str(x)}))),
            )
            .sort_values(["symbol"], ascending=[True])
            .reset_index(drop=True)
        )

    merged = pd.merge(local_df, broker_positions_agg, on="symbol", how="outer")
    merged = pd.merge(merged, open_orders_agg, on="symbol", how="outer")

        if "expected_shares" not in merged.columns:
        merged["expected_shares"] = 0.0
    if "broker_position" not in merged.columns:
        merged["broker_position"] = 0.0
    if "broker_open_order_count" not in merged.columns:
        merged["broker_open_order_count"] = 0
    if "broker_open_order_net_qty" not in merged.columns:
        merged["broker_open_order_net_qty"] = 0.0
    if "broker_open_order_remaining_qty" not in merged.columns:
        merged["broker_open_order_remaining_qty"] = 0.0
    if "source" not in merged.columns:
        merged["source"] = local_source.source

    merged["expected_shares"] = pd.to_numeric(merged["expected_shares"], errors="coerce").fillna(0.0)
    merged["broker_position"] = pd.to_numeric(merged["broker_position"], errors="coerce").fillna(0.0)
    merged["broker_open_order_count"] = pd.to_numeric(merged["broker_open_order_count"], errors="coerce").fillna(0).astype(int)
    merged["broker_open_order_net_qty"] = pd.to_numeric(merged["broker_open_order_net_qty"], errors="coerce").fillna(0.0)
    merged["broker_open_order_remaining_qty"] = pd.to_numeric(merged["broker_open_order_remaining_qty"], errors="coerce").fillna(0.0)

    merged["source"] = merged["source"].fillna(local_source.source).astype(str)
    merged["drift_shares"] = merged["broker_position"] - merged["expected_shares"]
    merged["abs_drift_shares"] = merged["drift_shares"].abs()
    merged["positions_match"] = merged["abs_drift_shares"] <= 1e-9
    merged["has_open_orders"] = merged["broker_open_order_count"] > 0
    merged["expected_nonzero"] = merged["expected_shares"].abs() > 1e-9
    merged["broker_nonzero"] = merged["broker_position"].abs() > 1e-9

    status: List[str] = []
    for _, row in merged.iterrows():
        expected_nonzero = bool(row["expected_nonzero"])
        broker_nonzero = bool(row["broker_nonzero"])
        positions_match = bool(row["positions_match"])
        has_open_orders = bool(row["has_open_orders"])
        if positions_match and not has_open_orders:
            status.append("match")
        elif positions_match and has_open_orders:
            status.append("match_with_open_orders")
        elif expected_nonzero and not broker_nonzero:
            status.append("missing_at_broker")
        elif broker_nonzero and not expected_nonzero:
            status.append("unexpected_at_broker")
        else:
            status.append("drift")
    merged["reconcile_status"] = status

    preferred_cols = [
        "symbol",
        "source",
        "expected_shares",
        "broker_position",
        "drift_shares",
        "abs_drift_shares",
        "reconcile_status",
        "positions_match",
        "has_open_orders",
        "broker_open_order_count",
        "broker_open_order_net_qty",
        "broker_open_order_remaining_qty",
        "target_shares",
        "current_shares",
        "delta_shares",
        "source_formula",
        "date",
        "order_side",
        "skip_reason",
        "broker_avg_cost",
        "broker_symbol",
        "broker_currency",
        "broker_exchange",
        "broker_primary_exchange",
        "broker_sec_type",
        "broker_account",
        "broker_open_order_actions",
        "broker_open_order_statuses",
        "price",
        "last_price",
        "market_value",
    ]
    final_cols = [c for c in preferred_cols if c in merged.columns] + [c for c in merged.columns if c not in preferred_cols]
    merged = merged[final_cols].sort_values(["abs_drift_shares", "symbol"], ascending=[False, True]).reset_index(drop=True)
    return merged



def _write_df_csv_safe(df: pd.DataFrame, path: Path) -> None:
    csv_text = df.to_csv(index=False)
    _safe_write_text(path, csv_text, encoding="utf-8")



def _write_json_safe(payload: Dict[str, Any], path: Path) -> None:
    _safe_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")



def _build_report(paths: ConfigPaths, local_source: LocalExpectedSource, broker_positions: pd.DataFrame, broker_open_orders: pd.DataFrame, diff_df: pd.DataFrame, app: IBKRReconApp) -> Dict[str, Any]:
    exact_match_count = int((diff_df["reconcile_status"] == "match").sum()) if len(diff_df) else 0
    match_with_open_orders_count = int((diff_df["reconcile_status"] == "match_with_open_orders").sum()) if len(diff_df) else 0
    drift_count = int((diff_df["reconcile_status"] == "drift").sum()) if len(diff_df) else 0
    missing_at_broker_count = int((diff_df["reconcile_status"] == "missing_at_broker").sum()) if len(diff_df) else 0
    unexpected_at_broker_count = int((diff_df["reconcile_status"] == "unexpected_at_broker").sum()) if len(diff_df) else 0
    report = {
        "generated_at_utc": _utc_now_iso(),
        "config": paths.name,
        "execution_dir": str(paths.execution_dir),
        "broker": {
            "host": IB_HOST,
            "port": IB_PORT,
            "client_id": IB_CLIENT_ID,
            "account_code": IB_ACCOUNT_CODE,
            "open_orders_mode": REQ_OPEN_ORDERS_MODE,
            "positions_timeout_sec": IB_POSITIONS_TIMEOUT_SEC,
            "positions_retries": IB_POSITIONS_RETRIES,
            "positions_settle_sec": IB_POSITIONS_SETTLE_SEC,
        },
        "local_source": {
            "source": local_source.source,
            "state_json_exists": paths.state_json.exists(),
            "orders_csv_exists": paths.orders_csv.exists(),
            "local_rows": int(len(local_source.rows)),
        },
        "summary": {
            "symbols_total": int(len(diff_df)),
            "local_expected_symbols": int(len(local_source.rows)),
            "broker_position_symbols": int(len(broker_positions)),
            "broker_open_orders": int(len(broker_open_orders)),
            "exact_match_count": exact_match_count,
            "match_with_open_orders_count": match_with_open_orders_count,
            "drift_count": drift_count,
            "missing_at_broker_count": missing_at_broker_count,
            "unexpected_at_broker_count": unexpected_at_broker_count,
            "max_abs_drift_shares": float(diff_df["abs_drift_shares"].max()) if len(diff_df) else 0.0,
            "sum_abs_drift_shares": float(diff_df["abs_drift_shares"].sum()) if len(diff_df) else 0.0,
        },
        "top_drifts": diff_df.head(min(TOPK_PRINT, len(diff_df))).to_dict(orient="records") if len(diff_df) else [],
        "artifacts": {
            "reconcile_report_json": str(paths.report_json),
            "reconcile_diff_csv": str(paths.diff_csv),
            "broker_positions_csv": str(paths.broker_positions_csv),
            "broker_open_orders_csv": str(paths.broker_open_orders_csv),
        },
        "ib_errors": app._errors,
    }
    return report



def _run_one_config(app: IBKRReconApp, paths: ConfigPaths) -> None:
    print(f"[STEP][{paths.name}] reconcile")
    paths.execution_dir.mkdir(parents=True, exist_ok=True)

    local_source = _load_local_expected(paths)
    print(
        f"[LOCAL][{paths.name}] source={local_source.source} rows={len(local_source.rows)} "
        f"state_json_exists={int(paths.state_json.exists())} orders_csv_exists={int(paths.orders_csv.exists())}"
    )

    broker_positions = _refresh_positions(app)
    broker_open_orders = _refresh_open_orders(app)

    print(
        f"[BROKER][{paths.name}] positions={len(broker_positions)} open_orders={len(broker_open_orders)} "
        f"account={IB_ACCOUNT_CODE or '<all>'}"
    )

    diff_df = _build_reconcile_diff(local_source, broker_positions, broker_open_orders)
    report = _build_report(paths, local_source, broker_positions, broker_open_orders, diff_df, app)

    _write_df_csv_safe(broker_positions, paths.broker_positions_csv)
    _write_df_csv_safe(broker_open_orders, paths.broker_open_orders_csv)
    _write_df_csv_safe(diff_df, paths.diff_csv)
    _write_json_safe(report, paths.report_json)

    exact_match_count = int((diff_df["reconcile_status"] == "match").sum()) if len(diff_df) else 0
    match_with_open_orders_count = int((diff_df["reconcile_status"] == "match_with_open_orders").sum()) if len(diff_df) else 0
    drift_count = int((diff_df["reconcile_status"] == "drift").sum()) if len(diff_df) else 0
    missing_at_broker_count = int((diff_df["reconcile_status"] == "missing_at_broker").sum()) if len(diff_df) else 0
    unexpected_at_broker_count = int((diff_df["reconcile_status"] == "unexpected_at_broker").sum()) if len(diff_df) else 0
    max_abs_drift = float(diff_df["abs_drift_shares"].max()) if len(diff_df) else 0.0

    print(
        f"[RECON][{paths.name}][SUMMARY] symbols_total={len(diff_df)} exact_match={exact_match_count} "
        f"match_with_open_orders={match_with_open_orders_count} drift={drift_count} "
        f"missing_at_broker={missing_at_broker_count} unexpected_at_broker={unexpected_at_broker_count} "
        f"max_abs_drift_shares={max_abs_drift:.8f}"
    )

    if len(diff_df):
        print(f"[RECON][{paths.name}][TOP]")
        print(diff_df.head(min(TOPK_PRINT, len(diff_df))).to_string(index=False))

    print(f"[ARTIFACT] {paths.report_json}")
    print(f"[ARTIFACT] {paths.diff_csv}")
    print(f"[ARTIFACT] {paths.broker_positions_csv}")
    print(f"[ARTIFACT] {paths.broker_open_orders_csv}")



def main() -> int:
    _enable_line_buffering()
    print(f"[CFG] execution_root={EXECUTION_ROOT}")
    print(f"[CFG] reconcile_config_names={RECONCILE_CONFIG_NAMES}")
    print(f"[CFG] ib_host={IB_HOST} ib_port={IB_PORT} ib_client_id={IB_CLIENT_ID} ib_timeout_sec={IB_TIMEOUT_SEC}")
    print(f"[CFG] ib_positions_timeout_sec={IB_POSITIONS_TIMEOUT_SEC} ib_positions_retries={IB_POSITIONS_RETRIES} ib_positions_settle_sec={IB_POSITIONS_SETTLE_SEC}")
    print(f"[CFG] ib_account_code={IB_ACCOUNT_CODE}")
    print(f"[CFG] reconcile_open_orders_mode={REQ_OPEN_ORDERS_MODE}")

    if not RECONCILE_CONFIG_NAMES:
        raise RuntimeError("No reconcile configs provided. Set RECONCILE_CONFIG_NAMES or CONFIG_NAMES.")

    app: Optional[IBKRReconApp] = None
    try:
        app = _bootstrap_connection()
        for name in RECONCILE_CONFIG_NAMES:
            paths = _config_paths(name)
            _run_one_config(app, paths)
    finally:
        if app is not None:
            try:
                if app.isConnected():
                    app.disconnect()
            except Exception:
                pass

    print("[FINAL] IBKR reconcile complete")
    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 1
    _safe_exit(rc)
