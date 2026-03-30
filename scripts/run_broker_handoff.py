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
from ibapi.client import EClient
from ibapi.contract import Contract
from ibapi.ticktype import TickTypeEnum
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
IB_HOST = str(os.getenv("IB_HOST", "127.0.0.1")).strip()
IB_PORT = int(os.getenv("IB_PORT", "4002"))
IB_CLIENT_ID = int(os.getenv("IB_CLIENT_ID", "51"))
IB_TIMEOUT_SEC = float(os.getenv("IB_TIMEOUT_SEC", "20.0"))
IB_MKT_TIMEOUT_SEC = float(os.getenv("IB_MKT_TIMEOUT_SEC", "8.0"))
IB_EXCHANGE = str(os.getenv("IB_EXCHANGE", "SMART")).strip().upper() or "SMART"
IB_PRIMARY_EXCHANGE = str(os.getenv("IB_PRIMARY_EXCHANGE", "")).strip().upper()
IB_CURRENCY = str(os.getenv("IB_CURRENCY", "USD")).strip().upper() or "USD"
IB_SECURITY_TYPE = str(os.getenv("IB_SECURITY_TYPE", "STK")).strip().upper() or "STK"
PRICE_MAX_AGE_SEC = float(os.getenv("PRICE_MAX_AGE_SEC", "15.0"))
MAX_PRICE_DEVIATION_PCT = float(os.getenv("MAX_PRICE_DEVIATION_PCT", "8.0"))
TOPK_PRINT = int(os.getenv("TOPK_PRINT", "50"))
ALLOW_LAST_FALLBACK = str(os.getenv("ALLOW_LAST_FALLBACK", "1")).strip().lower() not in {"0", "false", "no", "off"}

REQUIRED_ORDER_COLUMNS = ["symbol", "order_side", "delta_shares"]


@dataclass(frozen=True)
class ConfigPaths:
    name: str
    execution_dir: Path
    orders_csv: Path
    handoff_summary_json: Path


@dataclass
class QuoteState:
    req_id: int
    symbol: str
    bid: float = 0.0
    ask: float = 0.0
    last: float = 0.0
    close: float = 0.0
    bid_size: float = 0.0
    ask_size: float = 0.0
    last_size: float = 0.0
    done: bool = False
    ts_utc: str = ""


class QuoteApp(EWrapper, EClient):
    def __init__(self) -> None:
        EClient.__init__(self, self)
        self._next_valid_id_queue: queue.Queue[int] = queue.Queue()
        self._managed_accounts_queue: queue.Queue[str] = queue.Queue()
        self._network_thread: Optional[threading.Thread] = None
        self.quotes: Dict[int, QuoteState] = {}
        self.errors: List[dict] = []

    def error(self, reqId: int, errorCode: int, errorString: str, advancedOrderRejectJson: str = "") -> None:
        payload = {
            "ts_utc": utc_now_iso(),
            "reqId": int(reqId),
            "errorCode": int(errorCode),
            "errorString": str(errorString),
            "advancedOrderRejectJson": str(advancedOrderRejectJson or ""),
        }
        self.errors.append(payload)
        print(f"[IB][ERROR] reqId={reqId} code={errorCode} msg={errorString}")

    def nextValidId(self, orderId: int) -> None:
        self._next_valid_id_queue.put_nowait(int(orderId))
        print(f"[IB] nextValidId={orderId}")

    def managedAccounts(self, accountsList: str) -> None:
        self._managed_accounts_queue.put_nowait(str(accountsList))
        print(f"[IB] managedAccounts={accountsList}")

    def tickPrice(self, reqId: int, tickType: int, price: float, attrib) -> None:
        q = self.quotes.get(int(reqId))
        if q is None:
            return
        q.ts_utc = utc_now_iso()
        if tickType == TickTypeEnum.BID:
            q.bid = to_float(price)
        elif tickType == TickTypeEnum.ASK:
            q.ask = to_float(price)
        elif tickType == TickTypeEnum.LAST:
            q.last = to_float(price)
        elif tickType == TickTypeEnum.CLOSE:
            q.close = to_float(price)

    def tickSize(self, reqId: int, tickType: int, size: float) -> None:
        q = self.quotes.get(int(reqId))
        if q is None:
            return
        if tickType == TickTypeEnum.BID_SIZE:
            q.bid_size = to_float(size)
        elif tickType == TickTypeEnum.ASK_SIZE:
            q.ask_size = to_float(size)
        elif tickType == TickTypeEnum.LAST_SIZE:
            q.last_size = to_float(size)

    def tickSnapshotEnd(self, reqId: int) -> None:
        q = self.quotes.get(int(reqId))
        if q is not None:
            q.done = True
            q.ts_utc = utc_now_iso()

    def start_network_loop(self) -> None:
        if self._network_thread is not None and self._network_thread.is_alive():
            return
        self._network_thread = threading.Thread(target=self.run, name="ibapi-network", daemon=True)
        self._network_thread.start()

    def wait_for_next_valid_id(self, timeout_sec: float) -> int:
        return int(self._next_valid_id_queue.get(timeout=timeout_sec))

    def wait_for_managed_accounts(self, timeout_sec: float) -> str:
        return str(self._managed_accounts_queue.get(timeout=timeout_sec))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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


def cfg_paths(name: str) -> ConfigPaths:
    execution_dir = EXECUTION_ROOT / name
    return ConfigPaths(
        name=name,
        execution_dir=execution_dir,
        orders_csv=execution_dir / "orders.csv",
        handoff_summary_json=execution_dir / "broker_handoff_summary.json",
    )


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


def build_contract(symbol: str) -> Contract:
    contract = Contract()
    contract.symbol = normalize_symbol(symbol)
    contract.secType = IB_SECURITY_TYPE
    contract.exchange = IB_EXCHANGE
    contract.currency = IB_CURRENCY
    if IB_PRIMARY_EXCHANGE:
        contract.primaryExchange = IB_PRIMARY_EXCHANGE
    return contract


def load_orders_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"orders.csv not found: {path}")
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


def connect_quote_app() -> QuoteApp:
    app = QuoteApp()
    print(f"[IB] connecting host={IB_HOST} port={IB_PORT} client_id={IB_CLIENT_ID}")
    app.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID)
    app.start_network_loop()
    next_id = app.wait_for_next_valid_id(timeout_sec=IB_TIMEOUT_SEC)
    managed = app.wait_for_managed_accounts(timeout_sec=IB_TIMEOUT_SEC)
    print(f"[IB] connected next_valid_id={next_id} managed_accounts={managed}")
    return app


def disconnect_quote_app(app: QuoteApp) -> None:
    try:
        if app.isConnected():
            app.disconnect()
    except Exception:
        pass
    time.sleep(0.25)


def request_quotes(app: QuoteApp, symbols: List[str]) -> Dict[str, QuoteState]:
    states: Dict[str, QuoteState] = {}
    for idx, symbol in enumerate(symbols, start=1):
        req_id = 900000 + idx
        state = QuoteState(req_id=req_id, symbol=symbol)
        app.quotes[req_id] = state
        states[symbol] = state
        app.reqMktData(req_id, build_contract(symbol), "", True, False, [])
    deadline = time.time() + IB_MKT_TIMEOUT_SEC
    while time.time() < deadline:
        if all(q.done for q in states.values()):
            break
        time.sleep(0.1)
    for q in states.values():
        try:
            app.cancelMktData(q.req_id)
        except Exception:
            pass
    return states


def pick_execution_price(side: str, quote: QuoteState) -> Tuple[float, str]:
    side_up = normalize_order_side(side)
    bid = to_float(quote.bid)
    ask = to_float(quote.ask)
    last = to_float(quote.last)
    close = to_float(quote.close)
    mid = (bid + ask) / 2.0 if bid > 0.0 and ask > 0.0 else 0.0

    if side_up == "BUY":
        if ask > 0.0:
            return float(ask), "ask"
        if mid > 0.0:
            return float(mid), "mid"
        if ALLOW_LAST_FALLBACK and last > 0.0:
            return float(last), "last"
        if close > 0.0:
            return float(close), "close"
    elif side_up == "SELL":
        if bid > 0.0:
            return float(bid), "bid"
        if mid > 0.0:
            return float(mid), "mid"
        if ALLOW_LAST_FALLBACK and last > 0.0:
            return float(last), "last"
        if close > 0.0:
            return float(close), "close"
    raise RuntimeError(f"No usable live price for symbol={quote.symbol} side={side_up}")


def enrich_orders_with_live_prices(df: pd.DataFrame, quotes: Dict[str, QuoteState]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if df.empty:
        return df.copy(), {"rows": 0, "updated": 0, "price_sanity_fail": 0}

    out = df.copy()
    if "price" not in out.columns:
        out["price"] = 0.0
    out["model_price_reference"] = out["price"].astype(float)
    out["price_hint_source"] = ""
    out["quote_ts_utc"] = ""
    out["bid_price"] = 0.0
    out["ask_price"] = 0.0
    out["mid_price"] = 0.0
    out["last_price"] = 0.0
    out["close_price"] = 0.0
    out["spread_abs"] = 0.0
    out["spread_bps"] = 0.0
    out["price_sanity_ok"] = 0
    out["price_deviation_pct"] = 0.0

    updated = 0
    sanity_fail = 0
    for idx in out.index:
        symbol = normalize_symbol(str(out.at[idx, "symbol"]))
        side = normalize_order_side(str(out.at[idx, "order_side"]))
        q = quotes.get(symbol)
        if q is None:
            raise RuntimeError(f"Missing live quote for symbol={symbol}")
        bid = to_float(q.bid)
        ask = to_float(q.ask)
        last = to_float(q.last)
        close = to_float(q.close)
        mid = (bid + ask) / 2.0 if bid > 0.0 and ask > 0.0 else 0.0
        live_price, src = pick_execution_price(side, q)
        model_price = to_float(out.at[idx, "model_price_reference"])
        deviation_pct = 0.0
        sanity_ok = 1
        if model_price > 0.0:
            deviation_pct = abs(live_price - model_price) / model_price * 100.0
            if deviation_pct > MAX_PRICE_DEVIATION_PCT:
                sanity_ok = 0
                sanity_fail += 1
                print(
                    f"[PRICE_SANITY_WARN] symbol={symbol} side={side} model_price={model_price:.4f} live_price={live_price:.4f} deviation_pct={deviation_pct:.2f} src={src}"
                )
        out.at[idx, "price"] = float(live_price)
        out.at[idx, "price_hint_source"] = src
        out.at[idx, "quote_ts_utc"] = q.ts_utc or utc_now_iso()
        out.at[idx, "bid_price"] = float(bid)
        out.at[idx, "ask_price"] = float(ask)
        out.at[idx, "mid_price"] = float(mid)
        out.at[idx, "last_price"] = float(last)
        out.at[idx, "close_price"] = float(close)
        out.at[idx, "spread_abs"] = float(max(0.0, ask - bid)) if bid > 0.0 and ask > 0.0 else 0.0
        out.at[idx, "spread_bps"] = float(((ask - bid) / mid) * 10000.0) if bid > 0.0 and ask > 0.0 and mid > 0.0 else 0.0
        out.at[idx, "price_sanity_ok"] = int(sanity_ok)
        out.at[idx, "price_deviation_pct"] = float(deviation_pct)
        if "order_notional" in out.columns:
            out.at[idx, "order_notional"] = abs(to_float(out.at[idx, "delta_shares"]) * live_price)
        updated += 1

    summary = {
        "rows": int(len(out)),
        "updated": int(updated),
        "price_sanity_fail": int(sanity_fail),
    }
    return out, summary


def run_one_config(app: QuoteApp, paths: ConfigPaths) -> None:
    print(f"[HANDOFF][{paths.name}] orders_csv={paths.orders_csv}")
    paths.execution_dir.mkdir(parents=True, exist_ok=True)
    df = load_orders_df(paths.orders_csv)
    live_df = select_live_orders(df)
    symbols = sorted({normalize_symbol(str(x)) for x in live_df.get("symbol", []) if str(x).strip()})
    print(f"[HANDOFF][{paths.name}] total_rows={len(df)} live_rows={len(live_df)} symbols={len(symbols)}")
    if not symbols:
        summary = {"config": paths.name, "rows": int(len(df)), "live_rows": int(len(live_df)), "symbols": 0, "updated": 0, "price_sanity_fail": 0}
        paths.handoff_summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[HANDOFF][{paths.name}][SUMMARY] updated=0 price_sanity_fail=0 summary_json={paths.handoff_summary_json}")
        return

    quotes = request_quotes(app, symbols)
    updated_df, enrich_summary = enrich_orders_with_live_prices(df, quotes)
    updated_df.to_csv(paths.orders_csv, index=False)

    summary = {
        "config": paths.name,
        "rows": int(len(df)),
        "live_rows": int(len(live_df)),
        "symbols": int(len(symbols)),
        **enrich_summary,
    }
    paths.handoff_summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    preview_cols = [c for c in [
        "symbol", "order_side", "delta_shares", "model_price_reference", "price", "price_hint_source",
        "bid_price", "ask_price", "mid_price", "last_price", "spread_bps", "price_deviation_pct"
    ] if c in updated_df.columns]
    if preview_cols:
        print(f"[HANDOFF][{paths.name}][PREVIEW]")
        print(updated_df[preview_cols].head(min(TOPK_PRINT, len(updated_df))).to_string(index=False))

    print(
        f"[HANDOFF][{paths.name}][SUMMARY] updated={summary['updated']} price_sanity_fail={summary['price_sanity_fail']} orders_csv={paths.orders_csv} summary_json={paths.handoff_summary_json}"
    )


def main() -> int:
    enable_line_buffering()
    print(f"[CFG] execution_root={EXECUTION_ROOT}")
    print(f"[CFG] configs={CONFIG_NAMES}")
    print(f"[CFG] ib_host={IB_HOST} ib_port={IB_PORT} ib_client_id={IB_CLIENT_ID}")
    print(f"[CFG] exchange={IB_EXCHANGE} primary_exchange={IB_PRIMARY_EXCHANGE} currency={IB_CURRENCY} sec_type={IB_SECURITY_TYPE}")
    print(f"[CFG] ib_mkt_timeout_sec={IB_MKT_TIMEOUT_SEC} max_price_deviation_pct={MAX_PRICE_DEVIATION_PCT} allow_last_fallback={int(ALLOW_LAST_FALLBACK)}")
    app = connect_quote_app()
    try:
        for config_name in CONFIG_NAMES:
            run_one_config(app, cfg_paths(config_name))
    finally:
        disconnect_quote_app(app)
    print("[FINAL] broker handoff complete")
    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 1
    safe_exit(rc)
