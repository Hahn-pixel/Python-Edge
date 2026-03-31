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
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
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
IB_MARKET_DATA_TYPE = int(os.getenv("IB_MARKET_DATA_TYPE", "3"))
PRICE_MAX_AGE_SEC = float(os.getenv("PRICE_MAX_AGE_SEC", "15.0"))
MAX_PRICE_DEVIATION_PCT = float(os.getenv("MAX_PRICE_DEVIATION_PCT", "8.0"))
TOPK_PRINT = int(os.getenv("TOPK_PRINT", "50"))
ALLOW_LAST_FALLBACK = str(os.getenv("ALLOW_LAST_FALLBACK", "1")).strip().lower() not in {"0", "false", "no", "off"}

MASSIVE_API_KEY = str(os.getenv("MASSIVE_API_KEY", "")).strip()
MASSIVE_BASE_URL = str(os.getenv("MASSIVE_BASE_URL", "https://api.massive.com")).strip() or "https://api.massive.com"
MASSIVE_TIMEOUT_SEC = float(os.getenv("MASSIVE_TIMEOUT_SEC", "20.0"))
ENABLE_MASSIVE_FALLBACK = str(os.getenv("ENABLE_MASSIVE_FALLBACK", "1")).strip().lower() not in {"0", "false", "no", "off"}

REQUIRED_ORDER_COLUMNS = ["symbol", "order_side", "delta_shares"]

TICK_TYPE_BY_NAME: Dict[str, int] = {
    "BID": int(TickTypeEnum.BID),
    "ASK": int(TickTypeEnum.ASK),
    "LAST": int(TickTypeEnum.LAST),
    "CLOSE": int(TickTypeEnum.CLOSE),
    "BID_SIZE": int(TickTypeEnum.BID_SIZE),
    "ASK_SIZE": int(TickTypeEnum.ASK_SIZE),
    "LAST_SIZE": int(TickTypeEnum.LAST_SIZE),
    "DELAYED_BID": int(TickTypeEnum.DELAYED_BID),
    "DELAYED_ASK": int(TickTypeEnum.DELAYED_ASK),
    "DELAYED_LAST": int(TickTypeEnum.DELAYED_LAST),
    "DELAYED_CLOSE": int(TickTypeEnum.DELAYED_CLOSE),
    "DELAYED_BID_SIZE": int(TickTypeEnum.DELAYED_BID_SIZE),
    "DELAYED_ASK_SIZE": int(TickTypeEnum.DELAYED_ASK_SIZE),
    "DELAYED_LAST_SIZE": int(TickTypeEnum.DELAYED_LAST_SIZE),
}


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
    provider: str = "ib"
    market_data_type: int = 0
    timeframe: str = ""
    bid: float = 0.0
    ask: float = 0.0
    last: float = 0.0
    close: float = 0.0
    bid_size: float = 0.0
    ask_size: float = 0.0
    last_size: float = 0.0
    done: bool = False
    ts_utc: str = ""
    raw_error_codes: str = ""

    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0 if self.bid > 0.0 and self.ask > 0.0 else 0.0

    def has_any_price(self) -> bool:
        return any(x > 0.0 for x in [self.bid, self.ask, self.last, self.close, self.mid()])


@dataclass(frozen=True)
class MassiveClient:
    api_key: str
    base_url: str
    timeout_sec: float

    def __post_init__(self) -> None:
        if not self.api_key:
            raise RuntimeError("MASSIVE_API_KEY is missing from env")
        object.__setattr__(self, "_session", requests.Session())

    def _request_json(self, url: str, params: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        final_params = dict(params or {})
        final_params.setdefault("apiKey", self.api_key)
        response = self._session.get(url, params=final_params, timeout=self.timeout_sec)
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict) and str(payload.get("status", "")).upper() in {"ERROR", "NOT_AUTHORIZED"}:
            raise RuntimeError(f"Massive API error for {url}: {payload}")
        if not isinstance(payload, dict):
            raise RuntimeError(f"Unexpected Massive response type for {url}: {type(payload).__name__}")
        return payload

    def get_stock_snapshot(self, symbol: str) -> QuoteState:
        url = f"{self.base_url.rstrip('/')}/v2/snapshot/locale/us/markets/stocks/tickers/{normalize_symbol(symbol)}"
        payload = self._request_json(url)
        ticker = payload.get("ticker")
        if not isinstance(ticker, dict):
            raise RuntimeError(f"Massive snapshot missing ticker payload for symbol={symbol}")
        last_quote = ticker.get("lastQuote")
        last_trade = ticker.get("lastTrade")
        day = ticker.get("day")
        prev_day = ticker.get("prevDay")
        updated_raw = ticker.get("updated")

        bid = 0.0
        ask = 0.0
        bid_size = 0.0
        ask_size = 0.0
        if isinstance(last_quote, dict):
            bid = to_float(last_quote.get("p"))
            ask = to_float(last_quote.get("P"))
            bid_size = to_float(last_quote.get("s"))
            ask_size = to_float(last_quote.get("S"))

        last = 0.0
        last_size = 0.0
        if isinstance(last_trade, dict):
            last = to_float(last_trade.get("p"))
            last_size = to_float(last_trade.get("s"))

        close = 0.0
        if isinstance(day, dict):
            close = to_float(day.get("c"))
        if close <= 0.0 and isinstance(prev_day, dict):
            close = to_float(prev_day.get("c"))
        if close <= 0.0 and isinstance(ticker.get("min"), dict):
            close = to_float(ticker["min"].get("c"))

        state = QuoteState(
            req_id=0,
            symbol=normalize_symbol(symbol),
            provider="massive",
            market_data_type=0,
            timeframe="LIVE",
            bid=bid,
            ask=ask,
            last=last,
            close=close,
            bid_size=bid_size,
            ask_size=ask_size,
            last_size=last_size,
            done=True,
            ts_utc=ns_or_ms_to_utc_iso(updated_raw),
            raw_error_codes="",
        )
        if not state.ts_utc:
            state.ts_utc = utc_now_iso()
        return state


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
        req_id_int = int(reqId)
        q = self.quotes.get(req_id_int)
        if q is not None:
            codes = [x for x in str(q.raw_error_codes).split("|") if x]
            codes.append(str(int(errorCode)))
            q.raw_error_codes = "|".join(codes)
        print(f"[IB][ERROR] reqId={reqId} code={errorCode} msg={errorString}")

    def nextValidId(self, orderId: int) -> None:
        self._next_valid_id_queue.put_nowait(int(orderId))
        print(f"[IB] nextValidId={orderId}")

    def managedAccounts(self, accountsList: str) -> None:
        self._managed_accounts_queue.put_nowait(str(accountsList))
        print(f"[IB] managedAccounts={accountsList}")

    def marketDataType(self, reqId: int, marketDataType: int) -> None:
        q = self.quotes.get(int(reqId))
        if q is None:
            return
        q.market_data_type = int(marketDataType)
        q.ts_utc = utc_now_iso()

    def tickPrice(self, reqId: int, tickType: int, price: float, attrib) -> None:
        q = self.quotes.get(int(reqId))
        if q is None:
            return
        q.ts_utc = utc_now_iso()
        tick_type = int(tickType)

        if tick_type in {TICK_TYPE_BY_NAME["BID"], TICK_TYPE_BY_NAME["DELAYED_BID"]}:
            q.bid = to_float(price)
        elif tick_type in {TICK_TYPE_BY_NAME["ASK"], TICK_TYPE_BY_NAME["DELAYED_ASK"]}:
            q.ask = to_float(price)
        elif tick_type in {TICK_TYPE_BY_NAME["LAST"], TICK_TYPE_BY_NAME["DELAYED_LAST"]}:
            q.last = to_float(price)
        elif tick_type in {TICK_TYPE_BY_NAME["CLOSE"], TICK_TYPE_BY_NAME["DELAYED_CLOSE"]}:
            q.close = to_float(price)

    def tickSize(self, reqId: int, tickType: int, size: float) -> None:
        q = self.quotes.get(int(reqId))
        if q is None:
            return
        tick_type = int(tickType)
        if tick_type in {TICK_TYPE_BY_NAME["BID_SIZE"], TICK_TYPE_BY_NAME["DELAYED_BID_SIZE"]}:
            q.bid_size = to_float(size)
        elif tick_type in {TICK_TYPE_BY_NAME["ASK_SIZE"], TICK_TYPE_BY_NAME["DELAYED_ASK_SIZE"]}:
            q.ask_size = to_float(size)
        elif tick_type in {TICK_TYPE_BY_NAME["LAST_SIZE"], TICK_TYPE_BY_NAME["DELAYED_LAST_SIZE"]}:
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


def ns_or_ms_to_utc_iso(value: Any) -> str:
    try:
        if value is None:
            return ""
        iv = int(value)
        if iv <= 0:
            return ""
        if iv >= 10**18:
            ts = iv / 1_000_000_000.0
        elif iv >= 10**15:
            ts = iv / 1_000_000.0
        elif iv >= 10**12:
            ts = iv / 1_000.0
        else:
            ts = float(iv)
        return datetime.fromtimestamp(ts, tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    except Exception:
        return ""


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


def request_ib_quotes(app: QuoteApp, symbols: List[str]) -> Dict[str, QuoteState]:
    states: Dict[str, QuoteState] = {}
    app.reqMarketDataType(int(IB_MARKET_DATA_TYPE))
    time.sleep(0.25)

    for idx, symbol in enumerate(symbols, start=1):
        req_id = 900000 + idx
        state = QuoteState(req_id=req_id, symbol=symbol, provider="ib")
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

    for q in states.values():
        q.done = True
        if not q.ts_utc:
            q.ts_utc = utc_now_iso()
        if q.market_data_type == 3:
            q.timeframe = "DELAYED"
        elif q.market_data_type == 4:
            q.timeframe = "DELAYED_FROZEN"
        elif q.market_data_type == 1:
            q.timeframe = "LIVE"
        elif q.market_data_type == 2:
            q.timeframe = "FROZEN"
        elif not q.timeframe:
            q.timeframe = f"TYPE_{q.market_data_type}" if q.market_data_type else ""

    return states


def pick_execution_price(side: str, quote: QuoteState) -> Tuple[float, str]:
    side_up = normalize_order_side(side)
    bid = to_float(quote.bid)
    ask = to_float(quote.ask)
    last = to_float(quote.last)
    close = to_float(quote.close)
    mid = quote.mid()

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
    raise RuntimeError(f"No usable execution price for symbol={quote.symbol} side={side_up} provider={quote.provider}")


def needs_massive_fallback(side: str, quote: QuoteState) -> bool:
    try:
        pick_execution_price(side, quote)
        return False
    except Exception:
        return True


def build_massive_client() -> Optional[MassiveClient]:
    if not ENABLE_MASSIVE_FALLBACK:
        return None
    if not MASSIVE_API_KEY:
        return None
    return MassiveClient(
        api_key=MASSIVE_API_KEY,
        base_url=MASSIVE_BASE_URL,
        timeout_sec=MASSIVE_TIMEOUT_SEC,
    )


def request_massive_quotes(symbols: List[str]) -> Dict[str, QuoteState]:
    client = build_massive_client()
    if client is None:
        raise RuntimeError("Massive fallback requested but MASSIVE_API_KEY is missing or fallback disabled")
    states: Dict[str, QuoteState] = {}
    for idx, symbol in enumerate(symbols, start=1):
        state = client.get_stock_snapshot(symbol)
        states[symbol] = state
        print(
            f"[MASSIVE][SNAPSHOT] {idx}/{len(symbols)} symbol={symbol} "
            f"bid={state.bid:.4f} ask={state.ask:.4f} last={state.last:.4f} close={state.close:.4f} timeframe={state.timeframe}"
        )
    return states


def overlay_massive_fallback_quotes(df: pd.DataFrame, ib_quotes: Dict[str, QuoteState]) -> Tuple[Dict[str, QuoteState], Dict[str, int]]:
    final_quotes: Dict[str, QuoteState] = dict(ib_quotes)
    fallback_symbols: List[str] = []

    for idx in df.index:
        symbol = normalize_symbol(str(df.at[idx, "symbol"]))
        side = normalize_order_side(str(df.at[idx, "order_side"]))
        q = final_quotes.get(symbol)
        if q is None or needs_massive_fallback(side, q):
            fallback_symbols.append(symbol)

    fallback_symbols = sorted(set(fallback_symbols))
    used_massive = 0
    if fallback_symbols:
        print(f"[HANDOFF][FALLBACK] symbols_needing_massive={len(fallback_symbols)}")
        massive_quotes = request_massive_quotes(fallback_symbols)
        for symbol in fallback_symbols:
            if symbol in massive_quotes:
                final_quotes[symbol] = massive_quotes[symbol]
                used_massive += 1

    counts = {
        "ib_total": int(len(ib_quotes)),
        "massive_used": int(used_massive),
        "ib_only": int(max(0, len(final_quotes) - used_massive)),
        "fallback_needed": int(len(fallback_symbols)),
    }
    return final_quotes, counts


def enrich_orders_with_prices(df: pd.DataFrame, quotes: Dict[str, QuoteState]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if df.empty:
        return df.copy(), {"rows": 0, "updated": 0, "price_sanity_fail": 0, "massive_rows": 0, "ib_rows": 0}

    out = df.copy()
    if "price" not in out.columns:
        out["price"] = 0.0
    out["model_price_reference"] = out["price"].astype(float)
    out["quote_provider"] = ""
    out["quote_market_data_type"] = 0
    out["quote_timeframe"] = ""
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
    out["quote_raw_error_codes"] = ""

    updated = 0
    sanity_fail = 0
    massive_rows = 0
    ib_rows = 0

    for idx in out.index:
        symbol = normalize_symbol(str(out.at[idx, "symbol"]))
        side = normalize_order_side(str(out.at[idx, "order_side"]))
        if side == "HOLD" or abs(to_float(out.at[idx, "delta_shares"])) <= 1e-12:
            continue

        q = quotes.get(symbol)
        if q is None:
            raise RuntimeError(f"Missing quote for symbol={symbol}")

        bid = to_float(q.bid)
        ask = to_float(q.ask)
        last = to_float(q.last)
        close = to_float(q.close)
        mid = q.mid()
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
                    f"[PRICE_SANITY_WARN] symbol={symbol} side={side} model_price={model_price:.4f} "
                    f"live_price={live_price:.4f} deviation_pct={deviation_pct:.2f} src={q.provider}:{src}"
                )

        out.at[idx, "price"] = float(live_price)
        out.at[idx, "quote_provider"] = q.provider
        out.at[idx, "quote_market_data_type"] = int(q.market_data_type)
        out.at[idx, "quote_timeframe"] = q.timeframe
        out.at[idx, "price_hint_source"] = f"{q.provider}_{src}"
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
        out.at[idx, "quote_raw_error_codes"] = str(q.raw_error_codes or "")
        if "order_notional" in out.columns:
            out.at[idx, "order_notional"] = abs(to_float(out.at[idx, "delta_shares"]) * live_price)

        if q.provider == "massive":
            massive_rows += 1
        else:
            ib_rows += 1
        updated += 1

    summary = {
        "rows": int(len(out)),
        "updated": int(updated),
        "price_sanity_fail": int(sanity_fail),
        "massive_rows": int(massive_rows),
        "ib_rows": int(ib_rows),
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
        summary = {
            "config": paths.name,
            "rows": int(len(df)),
            "live_rows": int(len(live_df)),
            "symbols": 0,
            "updated": 0,
            "price_sanity_fail": 0,
            "massive_rows": 0,
            "ib_rows": 0,
        }
        paths.handoff_summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[HANDOFF][{paths.name}][SUMMARY] updated=0 price_sanity_fail=0 summary_json={paths.handoff_summary_json}")
        return

    ib_quotes = request_ib_quotes(app, symbols)
    final_quotes, quote_counts = overlay_massive_fallback_quotes(live_df, ib_quotes)
    updated_df, enrich_summary = enrich_orders_with_prices(df, final_quotes)
    updated_df.to_csv(paths.orders_csv, index=False)

    summary = {
        "config": paths.name,
        "rows": int(len(df)),
        "live_rows": int(len(live_df)),
        "symbols": int(len(symbols)),
        **quote_counts,
        **enrich_summary,
    }
    paths.handoff_summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    preview_cols = [
        c
        for c in [
            "symbol",
            "order_side",
            "delta_shares",
            "model_price_reference",
            "price",
            "quote_provider",
            "quote_timeframe",
            "price_hint_source",
            "bid_price",
            "ask_price",
            "mid_price",
            "last_price",
            "close_price",
            "spread_bps",
            "price_deviation_pct",
        ]
        if c in updated_df.columns
    ]
    if preview_cols:
        print(f"[HANDOFF][{paths.name}][PREVIEW]")
        print(updated_df[preview_cols].head(min(TOPK_PRINT, len(updated_df))).to_string(index=False))

    print(
        f"[HANDOFF][{paths.name}][SUMMARY] updated={summary['updated']} "
        f"ib_rows={summary['ib_rows']} massive_rows={summary['massive_rows']} "
        f"price_sanity_fail={summary['price_sanity_fail']} "
        f"orders_csv={paths.orders_csv} summary_json={paths.handoff_summary_json}"
    )


def main() -> int:
    enable_line_buffering()
    print(f"[CFG] execution_root={EXECUTION_ROOT}")
    print(f"[CFG] configs={CONFIG_NAMES}")
    print(f"[CFG] ib_host={IB_HOST} ib_port={IB_PORT} ib_client_id={IB_CLIENT_ID}")
    print(f"[CFG] exchange={IB_EXCHANGE} primary_exchange={IB_PRIMARY_EXCHANGE} currency={IB_CURRENCY} sec_type={IB_SECURITY_TYPE}")
    print(
        f"[CFG] ib_market_data_type={IB_MARKET_DATA_TYPE} ib_mkt_timeout_sec={IB_MKT_TIMEOUT_SEC} "
        f"max_price_deviation_pct={MAX_PRICE_DEVIATION_PCT} allow_last_fallback={int(ALLOW_LAST_FALLBACK)}"
    )
    print(
        f"[CFG] enable_massive_fallback={int(ENABLE_MASSIVE_FALLBACK)} "
        f"massive_base_url={MASSIVE_BASE_URL} massive_timeout_sec={MASSIVE_TIMEOUT_SEC}"
    )
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
