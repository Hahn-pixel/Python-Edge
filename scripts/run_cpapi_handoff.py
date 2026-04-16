from __future__ import annotations

import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

ROOT    = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
for _p in [ROOT, SRC_DIR]:
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

from python_edge.broker.cpapi_client import CpapiClient

# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------

def _env_str(k: str, d: str)   -> str:   return str(os.getenv(k, d)).strip()
def _env_float(k: str, d: float) -> float:
    try: return float(os.getenv(k, str(d)))
    except Exception: return d
def _env_bool(k: str, d: bool) -> bool:
    return str(os.getenv(k, "1" if d else "0")).strip().lower() not in {"0","false","no","off"}

PAUSE_ON_EXIT          = _env_str("PAUSE_ON_EXIT", "auto")
EXECUTION_ROOT         = Path(_env_str("EXECUTION_ROOT", "artifacts/execution_loop"))
CONFIG_NAMES           = [x for x in _env_str("CONFIG_NAMES", "optimal|aggressive").split("|") if x]
CPAPI_BASE_URL         = _env_str("CPAPI_BASE_URL",    "https://localhost:5000")
CPAPI_VERIFY_SSL       = _env_bool("CPAPI_VERIFY_SSL",  False)
CPAPI_TIMEOUT_SEC      = _env_float("CPAPI_TIMEOUT_SEC", 10.0)
CPAPI_SNAPSHOT_FIELDS  = _env_str("CPAPI_SNAPSHOT_FIELDS", "31,84,86,88")  # bid,ask,last,close
CPAPI_SNAPSHOT_WAIT_SEC= _env_float("CPAPI_SNAPSHOT_WAIT_SEC", 2.0)   # wait for streaming to populate
CPAPI_SNAPSHOT_RETRIES = int(os.getenv("CPAPI_SNAPSHOT_RETRIES", "3"))
CPAPI_INTER_REQ_SEC    = _env_float("CPAPI_INTER_REQ_SEC", 0.1)

MAX_PRICE_DEVIATION_PCT        = _env_float("MAX_PRICE_DEVIATION_PCT", 8.0)
ALLOW_LAST_FALLBACK            = _env_bool("ALLOW_LAST_FALLBACK", True)
FORCE_MASSIVE_WHEN_NO_BBO      = _env_bool("FORCE_MASSIVE_WHEN_IB_NO_BBO", True)
FORCE_MASSIVE_WHEN_CLOSE_ONLY  = _env_bool("FORCE_MASSIVE_WHEN_IB_CLOSE_ONLY", True)
ENABLE_MASSIVE_FALLBACK        = _env_bool("ENABLE_MASSIVE_FALLBACK", True)
MASSIVE_API_KEY                = _env_str("MASSIVE_API_KEY", "")
MASSIVE_BASE_URL               = _env_str("MASSIVE_BASE_URL", "https://api.massive.com")
MASSIVE_TIMEOUT_SEC            = _env_float("MASSIVE_TIMEOUT_SEC", 20.0)
TOPK_PRINT                     = int(os.getenv("TOPK_PRINT", "50"))

# CPAPI tick field IDs → semantic names
# https://www.interactivebrokers.com/campus/ibkr-api-page/cpapi-ref/#market-data
_FIELD_BID   = "84"
_FIELD_ASK   = "86"
_FIELD_LAST  = "31"
_FIELD_CLOSE = "7295"  # prior close; fallback fields: 7284, 7762
_FIELD_CLOSE_ALT = ["7762", "7284", "70"]

REQUIRED_ORDER_COLUMNS = ["symbol","order_side","delta_shares"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00","Z")

def _to_float(v: Any) -> float:
    try: return 0.0 if v is None else float(v)
    except Exception: return 0.0

def _norm(s: Any) -> str: return str(s or "").strip().upper()

def _norm_side(s: Any) -> str:
    v = _norm(s)
    return v if v in {"BUY","SELL","HOLD"} else "HOLD"

def _should_pause() -> bool:
    if PAUSE_ON_EXIT in {"0","false","no","off"}: return False
    if PAUSE_ON_EXIT in {"1","true","yes","on"}:  return True
    si,so = getattr(sys,"stdin",None), getattr(sys,"stdout",None)
    return bool(si and so and hasattr(si,"isatty") and si.isatty() and so.isatty())

def _safe_exit(code: int) -> None:
    if _should_pause():
        try:
            print(f"\n[EXIT] code={code}")
            input("Press Enter to exit...")
        except Exception: pass
    raise SystemExit(code)

def _enable_lb() -> None:
    for nm in ["stdout","stderr"]:
        s = getattr(sys,nm,None)
        if s and hasattr(s,"reconfigure"):
            try: s.reconfigure(line_buffering=True)
            except Exception: pass

# ---------------------------------------------------------------------------
# Quote container
# ---------------------------------------------------------------------------

class QuoteState:
    __slots__ = ("symbol","bid","ask","last","close","provider","ts_utc","conid")
    def __init__(self, symbol: str):
        self.symbol   = symbol
        self.bid      = 0.0
        self.ask      = 0.0
        self.last     = 0.0
        self.close    = 0.0
        self.provider = "cpapi"
        self.ts_utc   = ""
        self.conid    = ""

    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0 if self.bid > 0.0 and self.ask > 0.0 else 0.0

    def has_bbo(self) -> bool:
        return self.bid > 0.0 and self.ask > 0.0

    def has_any(self) -> bool:
        return any(x > 0.0 for x in [self.bid, self.ask, self.last, self.close, self.mid()])

# ---------------------------------------------------------------------------
# conid cache helper (re-uses cpapi_conid_resolver cache file)
# ---------------------------------------------------------------------------

def _load_conid_cache(orders_csv: Path) -> Dict[str, str]:
    cache_path = orders_csv.parent / "conid_cache.json"
    if not cache_path.exists():
        return {}
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        return {str(k): str(v) for k, v in data.items() if k and v}
    except Exception:
        return {}

# ---------------------------------------------------------------------------
# CPAPI market data snapshot
# ---------------------------------------------------------------------------

def _snapshot_batch(
    client: CpapiClient,
    conids: List[str],
    fields: str,
) -> Dict[str, Dict[str, Any]]:
    """
    GET /iserver/marketdata/snapshot?conids=...&fields=...
    Returns {conid: {field_id: value, ...}}
    """
    conids_str = ",".join(conids)
    path = f"/v1/api/iserver/marketdata/snapshot?conids={conids_str}&fields={fields}"
    try:
        raw = client._get(path)
    except Exception as exc:
        print(f"[HANDOFF][SNAPSHOT][FAIL] conids={conids_str}: {exc}")
        return {}
    rows = raw if isinstance(raw, list) else []
    result: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        conid = str(row.get("conid","") or "")
        if conid:
            result[conid] = dict(row)
    return result


def _request_cpapi_quotes(
    client: CpapiClient,
    symbol_to_conid: Dict[str, str],
) -> Dict[str, QuoteState]:
    """
    Fetch market data snapshots for all symbols via CPAPI.
    Retries up to CPAPI_SNAPSHOT_RETRIES times with a short wait
    between attempts (streaming data may not be immediately populated).
    """
    quotes: Dict[str, QuoteState] = {sym: QuoteState(sym) for sym in symbol_to_conid}
    conid_to_sym = {v: k for k, v in symbol_to_conid.items()}
    conids = list(symbol_to_conid.values())
    fields = CPAPI_SNAPSHOT_FIELDS

    for attempt in range(1, CPAPI_SNAPSHOT_RETRIES + 1):
        if attempt > 1:
            time.sleep(CPAPI_SNAPSHOT_WAIT_SEC)
        snapshot = _snapshot_batch(client, conids, fields)
        for conid, row in snapshot.items():
            sym = conid_to_sym.get(conid)
            if sym is None:
                continue
            q = quotes[sym]
            q.conid   = conid
            q.ts_utc  = _utc_now_iso()
            q.bid     = max(q.bid,   _to_float(row.get(_FIELD_BID,  0.0)))
            q.ask     = max(q.ask,   _to_float(row.get(_FIELD_ASK,  0.0)))
            q.last    = max(q.last,  _to_float(row.get(_FIELD_LAST, 0.0)))
            # close: try primary field then alternates
            if q.close <= 0.0:
                q.close = _to_float(row.get(_FIELD_CLOSE, 0.0))
            for alt in _FIELD_CLOSE_ALT:
                if q.close > 0.0:
                    break
                q.close = _to_float(row.get(alt, 0.0))

        # if all symbols have at least a price → stop early
        if all(quotes[s].has_any() for s in symbol_to_conid):
            break
        missing = [s for s in symbol_to_conid if not quotes[s].has_any()]
        print(f"[HANDOFF][SNAPSHOT] attempt={attempt}/{CPAPI_SNAPSHOT_RETRIES} "
              f"missing_prices={len(missing)}: {missing[:10]}")

    for q in quotes.values():
        if not q.ts_utc:
            q.ts_utc = _utc_now_iso()

    return quotes

# ---------------------------------------------------------------------------
# Massive fallback (identical logic to run_broker_handoff.py)
# ---------------------------------------------------------------------------

def _ns_to_utc(value: Any) -> str:
    try:
        iv = int(value)
        if iv <= 0: return ""
        ts = iv/1e9 if iv>=10**18 else iv/1e6 if iv>=10**15 else iv/1e3 if iv>=10**12 else float(iv)
        return datetime.fromtimestamp(ts, tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00","Z")
    except Exception:
        return ""


def _massive_snapshot(symbol: str) -> QuoteState:
    url = f"{MASSIVE_BASE_URL.rstrip('/')}/v2/snapshot/locale/us/markets/stocks/tickers/{_norm(symbol)}"
    resp = requests.get(url, params={"apiKey": MASSIVE_API_KEY}, timeout=MASSIVE_TIMEOUT_SEC)
    resp.raise_for_status()
    payload = resp.json()
    ticker = payload.get("ticker", {})
    lq  = ticker.get("lastQuote",  {}) or {}
    lt  = ticker.get("lastTrade",  {}) or {}
    day = ticker.get("day",        {}) or {}
    prv = ticker.get("prevDay",    {}) or {}

    q         = QuoteState(symbol)
    q.bid     = _to_float(lq.get("p"))
    q.ask     = _to_float(lq.get("P"))
    q.last    = _to_float(lt.get("p"))
    q.close   = _to_float(day.get("c")) or _to_float(prv.get("c"))
    q.ts_utc  = _ns_to_utc(ticker.get("updated")) or _utc_now_iso()
    q.provider = "massive"
    return q


def _needs_massive(side: str, q: QuoteState) -> Tuple[bool, str]:
    if not q.has_any():
        return True, "no_price"
    if FORCE_MASSIVE_WHEN_NO_BBO and not q.has_bbo():
        return True, "no_bbo"
    if FORCE_MASSIVE_WHEN_CLOSE_ONLY:
        bid, ask, last = q.bid, q.ask, q.last
        if bid <= 0.0 and ask <= 0.0 and last <= 0.0 and q.close > 0.0:
            return True, "close_only"
    return False, ""


def _pick_price(side: str, q: QuoteState) -> Tuple[float, str]:
    s = _norm_side(side)
    if s == "BUY":
        if q.ask  > 0.0: return q.ask,  "ask"
        if q.mid() > 0.0: return q.mid(),"mid"
        if ALLOW_LAST_FALLBACK and q.last  > 0.0: return q.last,  "last"
        if q.close > 0.0: return q.close,"close"
    else:
        if q.bid  > 0.0: return q.bid,  "bid"
        if q.mid() > 0.0: return q.mid(),"mid"
        if ALLOW_LAST_FALLBACK and q.last  > 0.0: return q.last,  "last"
        if q.close > 0.0: return q.close,"close"
    raise RuntimeError(f"No usable price for symbol={q.symbol} side={s} provider={q.provider}")

# ---------------------------------------------------------------------------
# orders.csv loading / enrichment
# ---------------------------------------------------------------------------

def _load_orders(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"orders.csv not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=REQUIRED_ORDER_COLUMNS)
    missing = [c for c in REQUIRED_ORDER_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(f"orders.csv missing columns {missing}: {path}")
    return df.copy()


def _enrich(
    df: pd.DataFrame,
    quotes: Dict[str, QuoteState],
    fallback_reasons: Dict[str, str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    out = df.copy()
    # Ensure all columns exist
    for col in ["price","model_price_reference","quote_provider","quote_market_data_type",
                "quote_timeframe","price_hint_source","quote_ts","quote_ts_utc",
                "bid","ask","mid","last","bid_price","ask_price","mid_price",
                "last_price","close_price","spread_abs","spread_bps","price_sanity_ok",
                "price_deviation_vs_model","price_deviation_pct","quote_raw_error_codes","fallback_reason"]:
        if col not in out.columns:
            out[col] = 0.0 if col not in {"quote_provider","quote_timeframe",
                                           "price_hint_source","quote_ts","quote_ts_utc",
                                           "quote_raw_error_codes","fallback_reason"} else ""

    out["model_price_reference"] = out["price"].astype(float)
    updated = sanity_fail = massive_rows = cpapi_rows = 0

    for idx in out.index:
        symbol = _norm(str(out.at[idx,"symbol"]))
        side   = _norm_side(str(out.at[idx,"order_side"]))
        if side == "HOLD" or abs(_to_float(out.at[idx,"delta_shares"])) <= 1e-12:
            continue
        q = quotes.get(symbol)
        if q is None:
            raise RuntimeError(f"Missing quote for symbol={symbol}")

        live_price, src = _pick_price(side, q)
        model_price     = _to_float(out.at[idx,"model_price_reference"])
        dev_pct = 0.0
        sanity  = 1
        if model_price > 0.0:
            dev_pct = abs(live_price - model_price) / model_price * 100.0
            if dev_pct > MAX_PRICE_DEVIATION_PCT:
                sanity = 0
                sanity_fail += 1
                print(f"[PRICE_SANITY_WARN] symbol={symbol} side={side} "
                      f"model={model_price:.4f} live={live_price:.4f} dev={dev_pct:.2f} src={q.provider}:{src}")

        mid = q.mid()
        out.at[idx,"price"]                    = float(live_price)
        out.at[idx,"quote_provider"]           = q.provider
        out.at[idx,"quote_market_data_type"]   = 0
        out.at[idx,"quote_timeframe"]          = "LIVE" if q.provider == "cpapi" else "MASSIVE"
        out.at[idx,"price_hint_source"]        = f"{q.provider}_{src}"
        out.at[idx,"quote_ts"]                 = q.ts_utc or _utc_now_iso()
        out.at[idx,"quote_ts_utc"]             = q.ts_utc or _utc_now_iso()
        out.at[idx,"bid"]                      = float(q.bid)
        out.at[idx,"ask"]                      = float(q.ask)
        out.at[idx,"mid"]                      = float(mid)
        out.at[idx,"last"]                     = float(q.last)
        out.at[idx,"bid_price"]                = float(q.bid)
        out.at[idx,"ask_price"]                = float(q.ask)
        out.at[idx,"mid_price"]                = float(mid)
        out.at[idx,"last_price"]               = float(q.last)
        out.at[idx,"close_price"]              = float(q.close)
        out.at[idx,"spread_abs"]               = float(max(0.0, q.ask - q.bid)) if q.has_bbo() else 0.0
        out.at[idx,"spread_bps"]               = float(((q.ask-q.bid)/mid)*10000.0) if q.has_bbo() and mid>0.0 else 0.0
        out.at[idx,"price_sanity_ok"]          = int(sanity)
        out.at[idx,"price_deviation_vs_model"] = float(dev_pct)
        out.at[idx,"price_deviation_pct"]      = float(dev_pct)
        out.at[idx,"quote_raw_error_codes"]    = ""
        out.at[idx,"fallback_reason"]          = str(fallback_reasons.get(symbol,""))
        if "order_notional" in out.columns:
            out.at[idx,"order_notional"] = abs(_to_float(out.at[idx,"delta_shares"]) * live_price)

        if q.provider == "massive": massive_rows += 1
        else:                       cpapi_rows   += 1
        updated += 1

    return out, {"rows": len(out), "updated": updated, "price_sanity_fail": sanity_fail,
                 "massive_rows": massive_rows, "cpapi_rows": cpapi_rows}

# ---------------------------------------------------------------------------
# Per-config runner
# ---------------------------------------------------------------------------

def _run_one(config_name: str, client: CpapiClient) -> None:
    base        = EXECUTION_ROOT / config_name
    orders_csv  = base / "orders.csv"
    summary_json= base / "broker_handoff_summary.json"
    base.mkdir(parents=True, exist_ok=True)

    df      = _load_orders(orders_csv)
    live_df = df.loc[(df["order_side"].str.upper() != "HOLD") &
                     (df["delta_shares"].abs() > 1e-12)].copy()
    symbols = sorted({_norm(str(x)) for x in live_df.get("symbol",[]) if str(x).strip()})

    print(f"[HANDOFF][{config_name}] total_rows={len(df)} live_rows={len(live_df)} symbols={len(symbols)}")

    if not symbols:
        summary_json.write_text(json.dumps({
            "config": config_name, "rows": len(df), "live_rows": len(live_df),
            "symbols": 0, "updated": 0, "price_sanity_fail": 0,
            "massive_rows": 0, "cpapi_rows": 0,
        }, indent=2), encoding="utf-8")
        return

    # Build symbol → conid mapping from cache
    conid_cache = _load_conid_cache(orders_csv)
    symbol_to_conid: Dict[str, str] = {}
    no_conid: List[str] = []
    for sym in symbols:
        cid = conid_cache.get(sym, "")
        if cid:
            symbol_to_conid[sym] = cid
        else:
            no_conid.append(sym)
    if no_conid:
        print(f"[HANDOFF][{config_name}][WARN] no conid for {len(no_conid)} symbols — will use massive: {no_conid}")

    # CPAPI quotes for symbols with conid
    quotes: Dict[str, QuoteState] = {}
    if symbol_to_conid:
        quotes = _request_cpapi_quotes(client, symbol_to_conid)

    # Determine which symbols need massive fallback
    fallback_syms: List[str] = list(no_conid)
    fallback_reasons: Dict[str, str] = {s: "no_conid" for s in no_conid}

    for sym in symbol_to_conid:
        side = _norm_side(str(live_df.loc[live_df["symbol"]==sym, "order_side"].iloc[0]
                              if sym in live_df["symbol"].values else "BUY"))
        q = quotes.get(sym, QuoteState(sym))
        need, reason = _needs_massive(side, q)
        if need:
            fallback_syms.append(sym)
            fallback_reasons[sym] = reason

    fallback_syms = sorted(set(fallback_syms))
    massive_used  = 0
    if fallback_syms and ENABLE_MASSIVE_FALLBACK:
        if not MASSIVE_API_KEY:
            print(f"[HANDOFF][WARN] massive fallback needed for {fallback_syms} but MASSIVE_API_KEY missing")
        else:
            print(f"[HANDOFF][FALLBACK] symbols={len(fallback_syms)}: {fallback_syms}")
            for sym in fallback_syms:
                try:
                    q = _massive_snapshot(sym)
                    quotes[sym] = q
                    massive_used += 1
                    print(f"[MASSIVE] {sym} bid={q.bid:.4f} ask={q.ask:.4f} last={q.last:.4f} close={q.close:.4f}")
                except Exception as exc:
                    print(f"[MASSIVE][FAIL] {sym}: {exc}")
                    quotes[sym] = QuoteState(sym)
                time.sleep(CPAPI_INTER_REQ_SEC)
    elif fallback_syms:
        for sym in fallback_syms:
            if sym not in quotes:
                quotes[sym] = QuoteState(sym)

    # Fill missing symbols with empty quotes
    for sym in symbols:
        if sym not in quotes:
            quotes[sym] = QuoteState(sym)

    updated_df, enrich_sum = _enrich(df, quotes, fallback_reasons)
    updated_df.to_csv(orders_csv, index=False)

    summary = {"config": config_name, "rows": len(df), "live_rows": len(live_df),
               "symbols": len(symbols), "cpapi_symbols": len(symbol_to_conid),
               "massive_used": massive_used, "fallback_needed": len(fallback_syms),
               **enrich_sum}
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    preview_cols = [c for c in ["symbol","order_side","delta_shares","model_price_reference",
                                "price","quote_provider","quote_timeframe","price_hint_source",
                                "quote_ts","fallback_reason","bid","ask","mid","last",
                                "close_price","spread_bps","price_deviation_vs_model"]
                    if c in updated_df.columns]
    print(f"[HANDOFF][{config_name}][PREVIEW]")
    print(updated_df[preview_cols].head(min(TOPK_PRINT, len(updated_df))).to_string(index=False))
    print(f"[HANDOFF][{config_name}][SUMMARY] updated={enrich_sum['updated']} "
          f"cpapi_rows={enrich_sum['cpapi_rows']} massive_rows={enrich_sum['massive_rows']} "
          f"price_sanity_fail={enrich_sum['price_sanity_fail']}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    _enable_lb()
    print(f"[CFG] execution_root={EXECUTION_ROOT} configs={CONFIG_NAMES}")
    print(f"[CFG] cpapi_url={CPAPI_BASE_URL} snapshot_fields={CPAPI_SNAPSHOT_FIELDS} "
          f"retries={CPAPI_SNAPSHOT_RETRIES} wait={CPAPI_SNAPSHOT_WAIT_SEC}s")
    print(f"[CFG] enable_massive_fallback={int(ENABLE_MASSIVE_FALLBACK)} "
          f"force_massive_no_bbo={int(FORCE_MASSIVE_WHEN_NO_BBO)}")

    client = CpapiClient(CPAPI_BASE_URL, CPAPI_TIMEOUT_SEC, CPAPI_VERIFY_SSL)
    client.assert_authenticated()
    client.start_tickle_loop()
    try:
        for cfg in CONFIG_NAMES:
            _run_one(cfg, client)
    finally:
        client.stop_tickle_loop()

    print("[FINAL] CPAPI handoff complete")
    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 1
    _safe_exit(rc)
