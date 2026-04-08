from __future__ import annotations

import json
import math
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

PAUSE_ON_EXIT = str(os.getenv("PAUSE_ON_EXIT", "auto")).strip().lower()
EXECUTION_ROOT = Path(os.getenv("EXECUTION_ROOT", "artifacts/execution_loop"))
CONFIG_NAMES = [x.strip() for x in str(os.getenv("CONFIG_NAMES", "optimal|aggressive")).split("|") if x.strip()]
IB_HOST = str(os.getenv("IB_HOST", "127.0.0.1")).strip()
IB_PORT = int(os.getenv("IB_PORT", "4002"))
IB_CLIENT_ID = int(os.getenv("IB_CLIENT_ID", "61"))
IB_ACCOUNT_CODE = str(os.getenv("IB_ACCOUNT_CODE", "")).strip()
IB_TIMEOUT_SEC = float(os.getenv("IB_TIMEOUT_SEC", "20.0"))
IB_OPEN_ORDERS_TIMEOUT_SEC = float(os.getenv("IB_OPEN_ORDERS_TIMEOUT_SEC", str(IB_TIMEOUT_SEC)))
IB_POSITIONS_TIMEOUT_SEC = float(os.getenv("IB_POSITIONS_TIMEOUT_SEC", str(IB_TIMEOUT_SEC)))
DRIFT_TOLERANCE_SHARES = float(os.getenv("DRIFT_TOLERANCE_SHARES", "0.000001"))
TOPK_PRINT = int(os.getenv("TOPK_PRINT", "50"))
PREFER_STATE_OVER_ORDERS = str(os.getenv("PREFER_STATE_OVER_ORDERS", "1")).strip().lower() not in {"0", "false", "no", "off"}
REQUIRE_BROKER_REFRESH = str(os.getenv("REQUIRE_BROKER_REFRESH", "1")).strip().lower() not in {"0", "false", "no", "off"}
ALLOW_EXISTING_BROKER_POSITIONS_CSV = str(os.getenv("ALLOW_EXISTING_BROKER_POSITIONS_CSV", "1")).strip().lower() not in {"0", "false", "no", "off"}
CLEANUP_PREVIEW_MODE = str(os.getenv("CLEANUP_PREVIEW_MODE", "1")).strip().lower() not in {"0", "false", "no", "off"}
CLEANUP_INCLUDE_UNEXPECTED = str(os.getenv("CLEANUP_INCLUDE_UNEXPECTED", "1")).strip().lower() not in {"0", "false", "no", "off"}
CLEANUP_INCLUDE_DRIFT = str(os.getenv("CLEANUP_INCLUDE_DRIFT", "1")).strip().lower() not in {"0", "false", "no", "off"}
CLEANUP_ONLY_SYMBOLS = {x.strip().upper() for x in str(os.getenv("CLEANUP_ONLY_SYMBOLS", "")).split("|") if x.strip()}
CLEANUP_EXCLUDE_SYMBOLS = {x.strip().upper() for x in str(os.getenv("CLEANUP_EXCLUDE_SYMBOLS", "")).split("|") if x.strip()}
CLEANUP_MIN_ABS_SHARES = float(os.getenv("CLEANUP_MIN_ABS_SHARES", "1.0"))
BROKER_PRICE_COLS = [
    "marketPrice",
    "market_price",
    "markPrice",
    "mark_price",
    "lastPrice",
    "last_price",
    "price",
    "avgCost",
    "avg_cost",
    "averageCost",
    "average_cost",
    "costBasisPrice",
    "cost_basis_price",
]


@dataclass(frozen=True)
class ConfigPaths:
    name: str
    execution_dir: Path
    state_json: Path
    orders_csv: Path
    broker_positions_csv: Path
    broker_open_orders_csv: Path
    broker_pending_csv: Path
    reconcile_csv: Path
    reconcile_summary_json: Path
    cleanup_preview_csv: Path


class IBReconApp(EWrapper, EClient):
    def __init__(self) -> None:
        EClient.__init__(self, self)
        self._thread: Optional[threading.Thread] = None
        self._next_valid_id_q: "queue.Queue[int]" = queue.Queue()
        self._managed_accounts_q: "queue.Queue[str]" = queue.Queue()
        self._open_orders_end = threading.Event()
        self._positions_end = threading.Event()
        self._errors: List[Dict[str, Any]] = []
        self.open_orders: List[Dict[str, Any]] = []
        self.positions_rows: List[Dict[str, Any]] = []

    def start_network_loop(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self.run, name="ibapi-network", daemon=True)
        self._thread.start()

    def error(self, reqId: int, errorCode: int, errorString: str, advancedOrderRejectJson: str = "") -> None:
        row = {
            "ts_utc": utc_now_iso(),
            "reqId": int(reqId or 0),
            "errorCode": int(errorCode or 0),
            "errorString": str(errorString or ""),
            "advancedOrderRejectJson": str(advancedOrderRejectJson or ""),
        }
        self._errors.append(row)
        print(f"[IB][ERR] reqId={row['reqId']} code={row['errorCode']} msg={row['errorString']}")

    def nextValidId(self, orderId: OrderId) -> None:
        self._next_valid_id_q.put(int(orderId))

    def managedAccounts(self, accountsList: str) -> None:
        self._managed_accounts_q.put(str(accountsList or ""))

    def openOrder(self, orderId: OrderId, contract: Contract, order: Order, orderState) -> None:
        self.open_orders.append(
            {
                "ib_order_id": int(orderId),
                "symbol": str(getattr(contract, "symbol", "") or ""),
                "localSymbol": str(getattr(contract, "localSymbol", "") or ""),
                "exchange": str(getattr(contract, "exchange", "") or ""),
                "primaryExchange": str(getattr(contract, "primaryExchange", "") or ""),
                "currency": str(getattr(contract, "currency", "") or ""),
                "secType": str(getattr(contract, "secType", "") or ""),
                "action": str(getattr(order, "action", "") or ""),
                "orderType": str(getattr(order, "orderType", "") or ""),
                "totalQuantity": float(to_float(getattr(order, "totalQuantity", 0.0))),
                "lmtPrice": float(to_float(getattr(order, "lmtPrice", 0.0))),
                "auxPrice": float(to_float(getattr(order, "auxPrice", 0.0))),
                "tif": str(getattr(order, "tif", "") or ""),
                "outsideRth": int(bool(getattr(order, "outsideRth", False))),
                "account": str(getattr(order, "account", "") or ""),
                "orderRef": str(getattr(order, "orderRef", "") or ""),
                "status": str(getattr(orderState, "status", "") or ""),
            }
        )

    def openOrderEnd(self) -> None:
        self._open_orders_end.set()
        print("[IB] openOrderEnd")

    def position(self, account: str, contract: Contract, position: float, avgCost: float) -> None:
        self.positions_rows.append(
            {
                "account": str(account or ""),
                "symbol": str(getattr(contract, "symbol", "") or ""),
                "localSymbol": str(getattr(contract, "localSymbol", "") or ""),
                "exchange": str(getattr(contract, "exchange", "") or ""),
                "primaryExchange": str(getattr(contract, "primaryExchange", "") or ""),
                "currency": str(getattr(contract, "currency", "") or ""),
                "secType": str(getattr(contract, "secType", "") or ""),
                "position": float(to_float(position)),
                "avgCost": float(to_float(avgCost)),
            }
        )

    def positionEnd(self) -> None:
        self._positions_end.set()
        print("[IB] positionEnd")

    def wait_for_next_valid_id(self, timeout_sec: float) -> int:
        return int(self._next_valid_id_q.get(timeout=timeout_sec))

    def wait_for_managed_accounts(self, timeout_sec: float) -> str:
        return str(self._managed_accounts_q.get(timeout=timeout_sec))

    def wait_until_open_orders_end(self, timeout_sec: float) -> None:
        if not self._open_orders_end.wait(timeout_sec):
            raise TimeoutError(f"Timed out waiting for openOrderEnd after {timeout_sec:.1f}s")

    def wait_until_positions_end(self, timeout_sec: float) -> None:
        if not self._positions_end.wait(timeout_sec):
            raise TimeoutError(f"Timed out waiting for positionEnd after {timeout_sec:.1f}s")


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


def to_float(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def normalize_symbol(symbol: Any) -> str:
    return str(symbol or "").strip().upper()


def normalize_side(side: str) -> str:
    out = str(side or "").strip().upper()
    if out not in {"BUY", "SELL", "HOLD"}:
        return "HOLD"
    return out


def must_exist(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def cfg_paths(config_name: str) -> ConfigPaths:
    base = EXECUTION_ROOT / config_name
    return ConfigPaths(
        name=config_name,
        execution_dir=base,
        state_json=base / "portfolio_state.json",
        orders_csv=base / "orders.csv",
        broker_positions_csv=base / "broker_positions.csv",
        broker_open_orders_csv=base / "broker_open_orders.csv",
        broker_pending_csv=base / "broker_pending.csv",
        reconcile_csv=base / "broker_reconcile.csv",
        reconcile_summary_json=base / "broker_reconcile_summary.json",
        cleanup_preview_csv=base / "broker_cleanup_preview.csv",
    )


def load_state_positions(state_json: Path) -> pd.DataFrame:
    must_exist(state_json, "portfolio_state.json")
    payload = json.loads(state_json.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid state JSON: {state_json}")
    positions = payload.get("positions", {})
    if not isinstance(positions, dict):
        raise RuntimeError(f"Invalid positions object in state JSON: {state_json}")
    rows: List[Dict[str, Any]] = []
    for symbol, pos in positions.items():
        if not isinstance(pos, dict):
            continue
        rows.append(
            {
                "symbol": normalize_symbol(symbol),
                "expected_shares_state": float(to_float(pos.get("shares", 0.0))),
                "state_last_price": float(to_float(pos.get("last_price", 0.0))),
                "state_market_value": float(to_float(pos.get("market_value", 0.0))),
                "state_price_source": str(pos.get("price_source", "") or ""),
                "state_is_priced": int(bool(pos.get("is_priced", False))),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["symbol", "expected_shares_state", "state_last_price", "state_market_value", "state_price_source", "state_is_priced"])
    return df.sort_values(["symbol"]).reset_index(drop=True)


def load_orders_targets(orders_csv: Path) -> pd.DataFrame:
    must_exist(orders_csv, "orders.csv")
    df = pd.read_csv(orders_csv)
    if df.empty:
        return pd.DataFrame(columns=["symbol", "expected_shares_orders", "price_hint", "price_source_orders"])
    if "symbol" not in df.columns:
        raise RuntimeError(f"orders.csv missing symbol column: {orders_csv}")
    if "target_shares" not in df.columns:
        return pd.DataFrame(columns=["symbol", "expected_shares_orders", "price_hint", "price_source_orders"])
    out = df.copy()
    out["symbol"] = out["symbol"].astype(str).map(normalize_symbol)
    out["target_shares"] = pd.to_numeric(out["target_shares"], errors="coerce").fillna(0.0)
    price_col = "price" if "price" in out.columns else None
    source_col = "price_source" if "price_source" in out.columns else None
    agg = out.groupby("symbol", as_index=False).agg(expected_shares_orders=("target_shares", "sum"))
    if price_col is not None:
        first_price = out.groupby("symbol", as_index=False).agg(price_hint=(price_col, "first"))
        agg = agg.merge(first_price, on="symbol", how="left")
    else:
        agg["price_hint"] = 0.0
    if source_col is not None:
        first_source = out.groupby("symbol", as_index=False).agg(price_source_orders=(source_col, "first"))
        agg = agg.merge(first_source, on="symbol", how="left")
    else:
        agg["price_source_orders"] = ""
    return agg.sort_values(["symbol"]).reset_index(drop=True)


def save_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def refresh_open_orders(app: IBReconApp) -> pd.DataFrame:
    app.open_orders = []
    app._open_orders_end.clear()
    app.reqOpenOrders()
    app.wait_until_open_orders_end(timeout_sec=IB_OPEN_ORDERS_TIMEOUT_SEC)
    df = pd.DataFrame(app.open_orders)
    if df.empty:
        return pd.DataFrame(columns=["ib_order_id", "symbol", "localSymbol", "exchange", "primaryExchange", "currency", "secType", "action", "orderType", "totalQuantity", "lmtPrice", "auxPrice", "tif", "outsideRth", "account", "orderRef", "status"])
    df["symbol"] = df["symbol"].astype(str).map(normalize_symbol)
    return df.sort_values(["symbol", "ib_order_id"]).reset_index(drop=True)


def refresh_positions(app: IBReconApp) -> pd.DataFrame:
    app.positions_rows = []
    app._positions_end.clear()
    app.reqPositions()
    app.wait_until_positions_end(timeout_sec=IB_POSITIONS_TIMEOUT_SEC)
    try:
        app.cancelPositions()
    except Exception:
        pass
    df = pd.DataFrame(app.positions_rows)
    if df.empty:
        return pd.DataFrame(columns=["account", "symbol", "localSymbol", "exchange", "primaryExchange", "currency", "secType", "position", "avgCost"])
    df["symbol"] = df["symbol"].astype(str).map(normalize_symbol)
    df["position"] = pd.to_numeric(df["position"], errors="coerce").fillna(0.0)
    df["avgCost"] = pd.to_numeric(df["avgCost"], errors="coerce")
    df = df.groupby("symbol", as_index=False).agg(
        account=("account", "first"),
        localSymbol=("localSymbol", "first"),
        exchange=("exchange", "first"),
        primaryExchange=("primaryExchange", "first"),
        currency=("currency", "first"),
        secType=("secType", "first"),
        position=("position", "sum"),
        avgCost=("avgCost", "first"),
    )
    return df.sort_values(["symbol"]).reset_index(drop=True)


def load_or_refresh_broker_positions(paths: ConfigPaths, app: Optional[IBReconApp]) -> pd.DataFrame:
    if app is not None:
        df = refresh_positions(app)
        save_df(df, paths.broker_positions_csv)
        return df
    if ALLOW_EXISTING_BROKER_POSITIONS_CSV and paths.broker_positions_csv.exists():
        df = pd.read_csv(paths.broker_positions_csv)
        if "symbol" in df.columns:
            df["symbol"] = df["symbol"].astype(str).map(normalize_symbol)
        return df
    raise RuntimeError(f"Broker positions unavailable for config={paths.name}")


def load_or_refresh_open_orders(paths: ConfigPaths, app: Optional[IBReconApp]) -> pd.DataFrame:
    if app is not None:
        df = refresh_open_orders(app)
        save_df(df, paths.broker_open_orders_csv)
        return df
    if paths.broker_open_orders_csv.exists():
        df = pd.read_csv(paths.broker_open_orders_csv)
        if "symbol" in df.columns:
            df["symbol"] = df["symbol"].astype(str).map(normalize_symbol)
        return df
    return pd.DataFrame(columns=["symbol", "status", "action", "totalQuantity", "lmtPrice", "orderRef"])


def build_pending_from_open_orders(open_orders_df: pd.DataFrame) -> pd.DataFrame:
    if open_orders_df.empty:
        return pd.DataFrame(columns=["symbol", "pending_side", "pending_qty", "pending_status", "pending_order_refs"])
    work = open_orders_df.copy()
    work["symbol"] = work["symbol"].astype(str).map(normalize_symbol)
    work["status_norm"] = work["status"].astype(str).str.strip().str.lower()
    work = work.loc[work["status_norm"].isin({"presubmitted", "submitted", "pendingsubmit", "pendingcancel", "api_pending"})].copy()
    if work.empty:
        return pd.DataFrame(columns=["symbol", "pending_side", "pending_qty", "pending_status", "pending_order_refs"])
    work["action"] = work["action"].astype(str).map(normalize_side)
    work["signed_qty"] = work.apply(lambda r: abs(to_float(r.get("totalQuantity", 0.0))) * (1.0 if r["action"] == "BUY" else -1.0 if r["action"] == "SELL" else 0.0), axis=1)
    grouped = work.groupby("symbol", as_index=False).agg(
        pending_signed_qty=("signed_qty", "sum"),
        pending_status=("status", lambda s: "|".join(sorted({str(x) for x in s if str(x)}))),
        pending_order_refs=("orderRef", lambda s: "|".join(sorted({str(x) for x in s if str(x)}))),
    )
    grouped["pending_side"] = grouped["pending_signed_qty"].map(lambda x: "BUY" if x > 0 else "SELL" if x < 0 else "HOLD")
    grouped["pending_qty"] = grouped["pending_signed_qty"].abs()
    return grouped[["symbol", "pending_side", "pending_qty", "pending_status", "pending_order_refs"]].sort_values(["symbol"]).reset_index(drop=True)


def merge_expected_and_broker(state_df: pd.DataFrame, orders_df: pd.DataFrame, broker_df: pd.DataFrame, pending_df: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(state_df, orders_df, on="symbol", how="outer")
    merged = pd.merge(merged, broker_df.rename(columns={"position": "broker_position", "avgCost": "broker_avg_cost"}), on="symbol", how="outer")
    merged = pd.merge(merged, pending_df, on="symbol", how="left")
    if "expected_shares_state" not in merged.columns:
        merged["expected_shares_state"] = 0.0
    if "expected_shares_orders" not in merged.columns:
        merged["expected_shares_orders"] = 0.0
    if "broker_position" not in merged.columns:
        merged["broker_position"] = 0.0
    merged["expected_shares_state"] = pd.to_numeric(merged["expected_shares_state"], errors="coerce").fillna(0.0)
    merged["expected_shares_orders"] = pd.to_numeric(merged["expected_shares_orders"], errors="coerce").fillna(0.0)
    merged["broker_position"] = pd.to_numeric(merged["broker_position"], errors="coerce").fillna(0.0)
    merged["pending_qty"] = pd.to_numeric(merged.get("pending_qty", 0.0), errors="coerce").fillna(0.0)
    merged["expected_shares"] = merged["expected_shares_state"] if PREFER_STATE_OVER_ORDERS else merged["expected_shares_orders"]
    missing_state_mask = merged["expected_shares_state"].abs() <= DRIFT_TOLERANCE_SHARES
    if PREFER_STATE_OVER_ORDERS:
        merged.loc[missing_state_mask, "expected_shares"] = merged.loc[missing_state_mask, "expected_shares_orders"]
    merged["drift_shares"] = merged["broker_position"] - merged["expected_shares"]
    merged["abs_drift_shares"] = merged["drift_shares"].abs()
    merged["broker_has_position"] = merged["broker_position"].abs() > DRIFT_TOLERANCE_SHARES
    merged["expected_has_position"] = merged["expected_shares"].abs() > DRIFT_TOLERANCE_SHARES
    merged["pending_signed_qty"] = merged.apply(lambda r: float(r["pending_qty"]) * (1.0 if str(r.get("pending_side", "")) == "BUY" else -1.0 if str(r.get("pending_side", "")) == "SELL" else 0.0), axis=1)
    merged["drift_after_pending"] = merged["broker_position"] + merged["pending_signed_qty"] - merged["expected_shares"]
    merged["abs_drift_after_pending"] = merged["drift_after_pending"].abs()
    merged["drift"] = merged["abs_drift_shares"] > DRIFT_TOLERANCE_SHARES
    merged["drift_after_pending_flag"] = merged["abs_drift_after_pending"] > DRIFT_TOLERANCE_SHARES
    merged["missing_at_broker"] = merged["expected_has_position"] & ~merged["broker_has_position"]
    merged["unexpected_at_broker"] = ~merged["expected_has_position"] & merged["broker_has_position"]
    merged["pending_covers_drift"] = merged["drift"] & (~merged["drift_after_pending_flag"])
    merged["issue_kind"] = merged.apply(classify_issue_kind, axis=1)
    merged["cleanup_side"] = merged["broker_position"].map(lambda x: "SELL" if x > DRIFT_TOLERANCE_SHARES else "BUY" if x < -DRIFT_TOLERANCE_SHARES else "HOLD")
    merged["cleanup_qty"] = merged["broker_position"].abs()
    return merged.sort_values(["abs_drift_shares", "symbol"], ascending=[False, True]).reset_index(drop=True)


def classify_issue_kind(row: pd.Series) -> str:
    if bool(row.get("pending_covers_drift", False)):
        return "pending_covers_drift"
    if bool(row.get("missing_at_broker", False)):
        return "missing_at_broker"
    if bool(row.get("unexpected_at_broker", False)):
        return "unexpected_at_broker"
    if bool(row.get("drift", False)):
        return "drift"
    return "ok"


def build_cleanup_preview(merged: pd.DataFrame) -> pd.DataFrame:
    if merged.empty:
        return pd.DataFrame(columns=["symbol", "cleanup_side", "cleanup_qty", "cleanup_reason", "broker_avg_cost", "broker_position", "expected_shares", "pending_side", "pending_qty"])
    work = merged.copy()
    mask = pd.Series(False, index=work.index)
    if CLEANUP_INCLUDE_UNEXPECTED:
        mask = mask | work["unexpected_at_broker"].fillna(False)
    if CLEANUP_INCLUDE_DRIFT:
        mask = mask | (work["drift"].fillna(False) & ~work["pending_covers_drift"].fillna(False))
    work = work.loc[mask].copy()
    work = work.loc[work["cleanup_qty"].abs() >= float(CLEANUP_MIN_ABS_SHARES)].copy()
    if CLEANUP_ONLY_SYMBOLS:
        work = work.loc[work["symbol"].astype(str).isin(CLEANUP_ONLY_SYMBOLS)].copy()
    if CLEANUP_EXCLUDE_SYMBOLS:
        work = work.loc[~work["symbol"].astype(str).isin(CLEANUP_EXCLUDE_SYMBOLS)].copy()
    if work.empty:
        return pd.DataFrame(columns=["symbol", "cleanup_side", "cleanup_qty", "cleanup_reason", "broker_avg_cost", "broker_position", "expected_shares", "pending_side", "pending_qty"])
    work["cleanup_reason"] = work["issue_kind"].astype(str)
    return work[[
        "symbol",
        "cleanup_side",
        "cleanup_qty",
        "cleanup_reason",
        "broker_avg_cost",
        "broker_position",
        "expected_shares",
        "pending_side",
        "pending_qty",
        "pending_status",
        "pending_order_refs",
    ]].sort_values(["cleanup_qty", "symbol"], ascending=[False, True]).reset_index(drop=True)


def summary_from_merged(config_name: str, merged: pd.DataFrame, cleanup_df: pd.DataFrame) -> Dict[str, Any]:
    drift_df = merged.loc[merged["drift"].fillna(False)].copy()
    pending_covers = merged.loc[merged["pending_covers_drift"].fillna(False)].copy()
    payload = {
        "config": config_name,
        "generated_at_utc": utc_now_iso(),
        "symbols_total": int(len(merged)),
        "drift": int(drift_df.shape[0]),
        "missing_at_broker": int(merged["missing_at_broker"].fillna(False).sum()),
        "unexpected_at_broker": int(merged["unexpected_at_broker"].fillna(False).sum()),
        "pending_covers_drift": int(pending_covers.shape[0]),
        "max_abs_drift_shares": float(merged["abs_drift_shares"].max()) if len(merged) else 0.0,
        "cleanup_preview_rows": int(len(cleanup_df)),
        "cleanup_preview_enabled": int(CLEANUP_PREVIEW_MODE),
    }
    return payload


def print_top(tag: str, df: pd.DataFrame, cols: List[str]) -> None:
    if df.empty:
        print(f"{tag} none")
        return
    available = [c for c in cols if c in df.columns]
    print(tag)
    print(df[available].head(min(TOPK_PRINT, len(df))).to_string(index=False))


def connect_app() -> IBReconApp:
    app = IBReconApp()
    print(f"[IB] connecting host={IB_HOST} port={IB_PORT} client_id={IB_CLIENT_ID}")
    app.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID)
    app.start_network_loop()
    next_id = app.wait_for_next_valid_id(timeout_sec=IB_TIMEOUT_SEC)
    managed = app.wait_for_managed_accounts(timeout_sec=IB_TIMEOUT_SEC)
    print(f"[IB] connected next_valid_id={next_id} managed_accounts={managed}")
    if IB_ACCOUNT_CODE:
        accounts = [x.strip() for x in managed.split(",") if x.strip()]
        if IB_ACCOUNT_CODE not in accounts:
            raise RuntimeError(f"IB_ACCOUNT_CODE={IB_ACCOUNT_CODE!r} not present in managed accounts: {managed!r}")
    return app


def disconnect_app(app: Optional[IBReconApp]) -> None:
    if app is None:
        return
    try:
        if app.isConnected():
            app.disconnect()
    except Exception:
        pass
    time.sleep(0.25)


def run_one_config(paths: ConfigPaths, app: Optional[IBReconApp]) -> None:
    print(f"[RECON][{paths.name}] execution_dir={paths.execution_dir}")
    paths.execution_dir.mkdir(parents=True, exist_ok=True)

    state_df = load_state_positions(paths.state_json)
    orders_df = load_orders_targets(paths.orders_csv)
    broker_df = load_or_refresh_broker_positions(paths, app)
    open_orders_df = load_or_refresh_open_orders(paths, app)
    pending_df = build_pending_from_open_orders(open_orders_df)

    save_df(open_orders_df, paths.broker_open_orders_csv)
    save_df(pending_df, paths.broker_pending_csv)

    merged = merge_expected_and_broker(state_df, orders_df, broker_df, pending_df)
    cleanup_df = build_cleanup_preview(merged) if CLEANUP_PREVIEW_MODE else pd.DataFrame()
    summary = summary_from_merged(paths.name, merged, cleanup_df)

    save_df(merged, paths.reconcile_csv)
    save_df(cleanup_df, paths.cleanup_preview_csv)
    paths.reconcile_summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"[RECON][{paths.name}][SUMMARY] symbols_total={summary['symbols_total']} drift={summary['drift']} "
        f"missing_at_broker={summary['missing_at_broker']} unexpected_at_broker={summary['unexpected_at_broker']} "
        f"pending_covers_drift={summary['pending_covers_drift']} max_abs_drift_shares={summary['max_abs_drift_shares']:.8f} "
        f"cleanup_preview_rows={summary['cleanup_preview_rows']}"
    )

    print_top(
        f"[RECON][{paths.name}][TOP]",
        merged.loc[merged["drift"].fillna(False)].copy(),
        [
            "symbol",
            "expected_shares",
            "broker_position",
            "drift_shares",
            "pending_side",
            "pending_qty",
            "drift_after_pending",
            "issue_kind",
            "broker_avg_cost",
            "state_last_price",
            "price_hint",
        ],
    )
    print_top(
        f"[RECON][{paths.name}][CLEANUP_PREVIEW]",
        cleanup_df,
        [
            "symbol",
            "cleanup_side",
            "cleanup_qty",
            "cleanup_reason",
            "broker_avg_cost",
            "broker_position",
            "expected_shares",
            "pending_side",
            "pending_qty",
        ],
    )
    print(f"[ARTIFACT] {paths.broker_positions_csv}")
    print(f"[ARTIFACT] {paths.broker_open_orders_csv}")
    print(f"[ARTIFACT] {paths.broker_pending_csv}")
    print(f"[ARTIFACT] {paths.reconcile_csv}")
    print(f"[ARTIFACT] {paths.cleanup_preview_csv}")
    print(f"[ARTIFACT] {paths.reconcile_summary_json}")


def main() -> int:
    enable_line_buffering()
    print(f"[CFG] execution_root={EXECUTION_ROOT}")
    print(f"[CFG] configs={CONFIG_NAMES}")
    print(f"[CFG] ib_host={IB_HOST} ib_port={IB_PORT} ib_client_id={IB_CLIENT_ID} ib_account_code={IB_ACCOUNT_CODE}")
    print(f"[CFG] require_broker_refresh={int(REQUIRE_BROKER_REFRESH)} allow_existing_broker_positions_csv={int(ALLOW_EXISTING_BROKER_POSITIONS_CSV)}")
    print(f"[CFG] drift_tolerance_shares={DRIFT_TOLERANCE_SHARES:.8f} prefer_state_over_orders={int(PREFER_STATE_OVER_ORDERS)}")
    print(f"[CFG] cleanup_preview_mode={int(CLEANUP_PREVIEW_MODE)} cleanup_include_unexpected={int(CLEANUP_INCLUDE_UNEXPECTED)} cleanup_include_drift={int(CLEANUP_INCLUDE_DRIFT)} cleanup_min_abs_shares={CLEANUP_MIN_ABS_SHARES:.8f}")
    if CLEANUP_ONLY_SYMBOLS:
        print(f"[CFG] cleanup_only_symbols={sorted(CLEANUP_ONLY_SYMBOLS)}")
    if CLEANUP_EXCLUDE_SYMBOLS:
        print(f"[CFG] cleanup_exclude_symbols={sorted(CLEANUP_EXCLUDE_SYMBOLS)}")

    app: Optional[IBReconApp] = None
    try:
        if REQUIRE_BROKER_REFRESH:
            app = connect_app()
        for config_name in CONFIG_NAMES:
            run_one_config(cfg_paths(config_name), app)
    finally:
        disconnect_app(app)

    print("[FINAL] IBKR broker reconcile complete")
    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 1
    safe_exit(rc)
