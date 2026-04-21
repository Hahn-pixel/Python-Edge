from __future__ import annotations

import hashlib
import json
import os
import statistics
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

ROOT    = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
for _p in [ROOT, SRC_DIR]:
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

from python_edge.broker.cpapi_client    import CpapiClient
from python_edge.broker.cpapi_models    import ExecutionIntent, ExecState, OrderSide
from python_edge.broker.cpapi_execution import run_batch
from python_edge.broker.cpapi_conid_resolver import (
    _CACHE_FILENAME,
    load_conid_cache,
    resolve_for_orders_csv,
)
from python_edge.broker.cpapi_storage import (
    append_or_replace_fills,
    build_broker_log_entry,
    duplicate_fill_entry,
    existing_duplicate_status,
    load_broker_log,
    save_broker_log,
    upsert_broker_log_entry,
)

# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------

def _env_str(k: str, d: str) -> str:   return str(os.getenv(k, d)).strip()
def _env_float(k: str, d: float) -> float:
    try: return float(os.getenv(k, str(d)))
    except Exception: return d
def _env_bool(k: str, d: bool) -> bool:
    return str(os.getenv(k, "1" if d else "0")).strip().lower() not in {"0","false","no","off"}

PAUSE_ON_EXIT      = _env_str("PAUSE_ON_EXIT", "auto")
EXECUTION_ROOT     = Path(_env_str("EXECUTION_ROOT", "artifacts/execution_loop"))
CONFIG_NAMES       = [x for x in _env_str("CONFIG_NAMES", "optimal|aggressive").split("|") if x]
BROKER_NAME        = _env_str("BROKER_NAME",        "MEXEM")
BROKER_PLATFORM    = _env_str("BROKER_PLATFORM",    "IBKR_CPAPI")
BROKER_ACCOUNT_ID  = _env_str("BROKER_ACCOUNT_ID",  "")
CPAPI_BASE_URL     = _env_str("CPAPI_BASE_URL",     "https://localhost:5000")
CPAPI_VERIFY_SSL   = _env_bool("CPAPI_VERIFY_SSL",   False)
CPAPI_TIMEOUT_SEC  = _env_float("CPAPI_TIMEOUT_SEC",  10.0)
WHOLE_TIMEOUT_SEC  = _env_float("CPAPI_WHOLE_TIMEOUT_SEC",  60.0)
FRAC_TIMEOUT_SEC   = _env_float("CPAPI_FRAC_TIMEOUT_SEC",   20.0)
TIF                = _env_str("CPAPI_TIF",           "DAY")

# Frac slippage — базове значення; адаптивне розраховується з spread_bps
FRAC_SLIPPAGE_BPS   = _env_float("CPAPI_FRAC_SLIPPAGE_BPS",    5.0)
FRAC_SLIPPAGE_MIN   = _env_float("CPAPI_FRAC_SLIPPAGE_MIN",    5.0)
FRAC_SLIPPAGE_MAX   = _env_float("CPAPI_FRAC_SLIPPAGE_MAX",   30.0)
FRAC_SLIPPAGE_ADDON = _env_float("CPAPI_FRAC_SLIPPAGE_ADDON",  2.0)

# Whole leg slippage від mid (LMT замість MIDPRICE)
WHOLE_SLIPPAGE_BPS  = _env_float("CPAPI_WHOLE_SLIPPAGE_BPS",  20.0)

# Parent guard ±N% від mid
PARENT_GUARD_PCT    = _env_float("CPAPI_PARENT_GUARD_PCT",     3.0)

RESET_BROKER_LOG     = _env_bool("RESET_BROKER_LOG",     False)
CPAPI_RESOLVE_CONIDS = _env_bool("CPAPI_RESOLVE_CONIDS",  True)

REQUIRED_ORDER_COLUMNS = ["symbol", "order_side", "delta_shares"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00","Z")

def _should_pause() -> bool:
    if PAUSE_ON_EXIT in {"0","false","no","off"}: return False
    if PAUSE_ON_EXIT in {"1","true","yes","on"}:  return True
    si, so = getattr(sys,"stdin",None), getattr(sys,"stdout",None)
    return bool(si and so and hasattr(si,"isatty") and si.isatty() and so.isatty())

def _safe_exit(code: int) -> None:
    if _should_pause():
        try:
            print(f"\n[EXIT] code={code}")
            input("Press Enter to exit...")
        except Exception: pass
    raise SystemExit(code)

def _must_exist(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")

def _to_float(v: Any) -> float:
    try: return 0.0 if v is None else float(v)
    except Exception: return 0.0

def _normalize_symbol(s: str) -> str: return str(s or "").strip().upper()

def _normalize_side(s: str) -> str:
    v = str(s or "").strip().upper()
    if v not in {"BUY","SELL","HOLD"}:
        raise RuntimeError(f"Unsupported order_side: {s!r}")
    return v

def _enable_line_buffering() -> None:
    for name in ["stdout","stderr"]:
        stream = getattr(sys, name, None)
        if stream and hasattr(stream, "reconfigure"):
            try: stream.reconfigure(line_buffering=True)
            except Exception: pass

# ---------------------------------------------------------------------------
# Adaptive frac slippage
# ---------------------------------------------------------------------------

def _adaptive_slippage_bps(spread_bps: float) -> float:
    """
    half_spread + addon, обмежений [MIN, MAX].
    spread=0 → fallback до FRAC_SLIPPAGE_BPS.
    Приклад: spread=22.7 → 22.7/2+2=13.35bps
    """
    if spread_bps <= 0.0:
        return float(FRAC_SLIPPAGE_BPS)
    return float(max(FRAC_SLIPPAGE_MIN, min(FRAC_SLIPPAGE_MAX, spread_bps / 2.0 + FRAC_SLIPPAGE_ADDON)))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _config_paths(cfg: str) -> Dict[str, Path]:
    d = EXECUTION_ROOT / cfg
    return {
        "execution_dir": d,
        "orders_csv":    d / "orders.csv",
        "fills_csv":     d / "fills.csv",
        "broker_log":    d / "broker_log.json",
    }

# ---------------------------------------------------------------------------
# orders.csv
# ---------------------------------------------------------------------------

def _load_orders_csv(path: Path) -> pd.DataFrame:
    _must_exist(path, "orders.csv")
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_ORDER_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(f"orders.csv missing columns {missing}: {path}")
    df["symbol"]       = df["symbol"].astype(str).map(_normalize_symbol)
    df["order_side"]   = df["order_side"].astype(str).map(_normalize_side)
    df["delta_shares"] = pd.to_numeric(df["delta_shares"], errors="coerce").fillna(0.0)
    return df.copy()

def _select_live_orders(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df.copy()
    mask = (df["order_side"].str.upper() != "HOLD") & (df["delta_shares"].abs() > 1e-12)
    return df.loc[mask].reset_index(drop=True)

# ---------------------------------------------------------------------------
# conid
# ---------------------------------------------------------------------------

def _resolve_conid(row: Dict[str, Any], symbol: str, cache: Dict[str, str]) -> str:
    for col in ("conid", "broker_conid"):
        val = str(row.get(col, "") or "").strip()
        if val and val not in {"nan","None","0"}:
            return val
    cached = cache.get(symbol, "").strip()
    if cached:
        return cached
    raise RuntimeError(
        f"conid not found for symbol={symbol}. "
        f"Run with CPAPI_RESOLVE_CONIDS=1 or add 'conid' column to orders.csv."
    )

# ---------------------------------------------------------------------------
# Build ExecutionIntent
# ---------------------------------------------------------------------------

def _build_intent(
    config_name: str,
    row: Dict[str, Any],
    ikey: str,
    conid_cache: Dict[str, str],
) -> tuple[ExecutionIntent, float, float]:
    """
    Повертає (intent, reference_price, frac_slippage_bps).

    reference_price = mid (BBO midpoint) > price (ask/bid або close fallback)
    frac_slippage   = адаптивний від spread_bps
    parent_cap/floor = ±PARENT_GUARD_PCT% від reference
    """
    symbol   = _normalize_symbol(str(row.get("symbol", "")))
    side_str = _normalize_side(str(row.get("order_side", "")))
    qty      = abs(_to_float(row.get("delta_shares", 0.0)))
    conid    = _resolve_conid(row, symbol, conid_cache)

    # Reference price: mid > price
    mid_val   = _to_float(row.get("mid", 0.0)) or _to_float(row.get("mid_price", 0.0))
    price_col = _to_float(row.get("price", 0.0))

    if mid_val > 0.0:
        reference_price = mid_val
        ref_source = "mid"
    elif price_col > 0.0:
        reference_price = price_col
        ref_source = "price_col"
    else:
        reference_price = 0.0
        ref_source = "none"

    print(
        f"[INTENT] {symbol} side={side_str} qty={qty:.0f} "
        f"ref={reference_price:.4f}({ref_source}) "
        f"mid={mid_val:.4f} price_col={price_col:.4f}"
    )

    # Adaptive frac slippage
    spread_bps = _to_float(row.get("spread_bps", 0.0))
    frac_slippage = _adaptive_slippage_bps(spread_bps)
    print(f"[INTENT] {symbol} spread_bps={spread_bps:.2f} frac_slippage={frac_slippage:.2f}bps")

    # Parent guard ±PARENT_GUARD_PCT% від reference
    guard = float(PARENT_GUARD_PCT) / 100.0
    parent_cap:   Optional[float] = None
    parent_floor: Optional[float] = None
    if reference_price > 0.0:
        if side_str == "BUY":
            parent_cap   = reference_price * (1.0 + guard)
        else:
            parent_floor = reference_price * (1.0 - guard)
    else:
        # fallback до старої логіки
        p = price_col
        if side_str == "BUY":
            parent_cap   = _to_float(row.get("parent_cap",   0.0)) or (p * 1.20 if p > 0 else None)
        else:
            parent_floor = _to_float(row.get("parent_floor", 0.0)) or (p * 0.80 if p > 0 else None)

    intent = ExecutionIntent(
        symbol       = symbol,
        conid        = conid,
        side         = OrderSide(side_str),
        target_qty   = qty,
        parent_cap   = parent_cap,
        parent_floor = parent_floor,
        client_tag   = f"pe-{ikey[:24]}",
        account_id   = BROKER_ACCOUNT_ID,
    )
    return intent, reference_price, frac_slippage


def _make_ikey(config_name: str, row: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps({
        "config":         config_name,
        "date":           str(row.get("date","")).strip(),
        "symbol":         _normalize_symbol(str(row.get("symbol",""))),
        "order_side":     _normalize_side(str(row.get("order_side","HOLD"))),
        "delta_shares":   round(_to_float(row.get("delta_shares",0.0)), 8),
        "order_notional": round(_to_float(row.get("order_notional",0.0)), 8),
        "target_weight":  round(_to_float(row.get("target_weight",0.0)), 8),
    }, sort_keys=True, separators=(",",":"), ensure_ascii=False).encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------------
# Per-config execution
# ---------------------------------------------------------------------------

def _run_config(client: CpapiClient, config_name: str) -> None:
    paths = _config_paths(config_name)
    _must_exist(paths["orders_csv"], f"orders.csv [{config_name}]")

    broker_log = load_broker_log(
        path=paths["broker_log"], config_name=config_name,
        broker_name=BROKER_NAME, broker_platform=BROKER_PLATFORM,
        broker_account_id=BROKER_ACCOUNT_ID, utc_now_iso=_utc_now_iso,
        reset=RESET_BROKER_LOG,
    )

    df      = _load_orders_csv(paths["orders_csv"])
    live_df = _select_live_orders(df)
    print(f"[CFG] config={config_name} orders_total={len(df)} live_orders={len(live_df)}")

    # conid cache
    cache_path = paths["execution_dir"] / _CACHE_FILENAME
    if CPAPI_RESOLVE_CONIDS:
        conid_cache, unresolved = resolve_for_orders_csv(client, paths["orders_csv"])
        if unresolved:
            print(f"[CONID][WARN] unresolved: {unresolved}")
    else:
        conid_cache = load_conid_cache(cache_path)
        print(f"[CONID_CACHE] loaded {len(conid_cache)} entries (CPAPI_RESOLVE_CONIDS=0)")

    intents:           List[ExecutionIntent] = []
    ikeys:             List[str]             = []
    row_metas:         List[Dict[str,Any]]   = []
    reference_prices:  Dict[str,float]       = {}
    slippage_per_sym:  Dict[str,float]       = {}
    duplicate_entries: List[dict]            = []
    debug_dup = debug_conid_missing = 0

    for _, row in live_df.iterrows():
        row_dict = {str(k): row[k] for k in live_df.columns}
        symbol   = _normalize_symbol(str(row_dict.get("symbol","")))
        side_str = _normalize_side(str(row_dict.get("order_side","")))
        qty      = abs(_to_float(row_dict.get("delta_shares",0.0)))
        if qty <= 0.0:
            print(f"[SKIP] {symbol} delta_shares=0")
            continue

        ikey = _make_ikey(config_name, row_dict)
        dup  = existing_duplicate_status(broker_log, ikey)
        if dup is not None:
            debug_dup += 1
            duplicate_entries.append(duplicate_fill_entry(
                idempotency_key=ikey, client_tag=f"pe-{ikey[:24]}",
                symbol=symbol, side=side_str, qty=qty,
                price_hint=_to_float(row_dict.get("price",0.0)),
                order_notional=_to_float(row_dict.get("order_notional",0.0)),
                order_date=str(row_dict.get("date","")),
                config_name=config_name,
                source_order_path=str(paths["orders_csv"]),
                duplicate_status=dup, broker_log=broker_log,
            ))
            print(f"[DUP] {symbol} {side_str} qty={qty:.0f} status={dup} — skipped")
            continue

        try:
            intent, ref_price, frac_slippage = _build_intent(config_name, row_dict, ikey, conid_cache)
        except RuntimeError as exc:
            debug_conid_missing += 1
            print(f"[SKIP][CONID_MISSING] {symbol}: {exc}")
            continue

        if ref_price <= 0.0:
            print(f"[SKIP] {symbol} ref_price=0 — no mid or price available")
            continue

        intents.append(intent)
        ikeys.append(ikey)
        row_metas.append({
            "price_hint":        _to_float(row_dict.get("price",0.0)),
            "order_notional":    _to_float(row_dict.get("order_notional",0.0)),
            "order_date":        str(row_dict.get("date","")),
            "source_order_path": str(paths["orders_csv"]),
        })
        reference_prices[symbol] = ref_price
        slippage_per_sym[symbol] = frac_slippage

    if duplicate_entries:
        append_or_replace_fills(paths["fills_csv"], duplicate_entries)

    print(
        f"[CFG] config={config_name} to_execute={len(intents)} "
        f"dup_skipped={debug_dup} conid_missing={debug_conid_missing}"
    )

    if not intents:
        print(f"[CFG] config={config_name} no intents to execute")
        return

    # Median frac slippage для батчу
    all_slippages = list(slippage_per_sym.values())
    batch_frac_slippage = statistics.median(all_slippages) if all_slippages else FRAC_SLIPPAGE_BPS
    print(
        f"[CFG] batch_frac_slippage={batch_frac_slippage:.2f}bps "
        f"(median, min={min(all_slippages):.2f} max={max(all_slippages):.2f})"
    )

    engine_kwargs = {
        "whole_timeout_sec":  WHOLE_TIMEOUT_SEC,
        "frac_timeout_sec":   FRAC_TIMEOUT_SEC,
        "tif":                TIF,
        "frac_slippage_bps":  batch_frac_slippage,
        "whole_slippage_bps": WHOLE_SLIPPAGE_BPS,
    }
    results = run_batch(client, intents, reference_prices, engine_kwargs)

    sent = errors = 0
    fills_to_write: List[dict] = []

    for intent, result, ikey, meta in zip(intents, results, ikeys, row_metas):
        try:
            log_entry = build_broker_log_entry(
                config_name=config_name, order_date=meta["order_date"],
                order_notional=meta["order_notional"], price_hint=meta["price_hint"],
                source_order_path=meta["source_order_path"],
                idempotency_key=ikey, intent=intent, result=result, utc_now_iso=_utc_now_iso,
            )
            upsert_broker_log_entry(broker_log, log_entry, _utc_now_iso)
            save_broker_log(paths["broker_log"], broker_log, _utc_now_iso)
            if result.total_filled > 0.0:
                fills_to_write.append(log_entry)
            print(
                f"[SEND] {intent.symbol} {intent.side.value} qty={intent.target_qty:.0f} "
                f"state={result.final_state.value} filled={result.total_filled:.4f} avg_px={result.avg_price:.4f}"
            )
            if result.final_state is ExecState.DONE: sent += 1
            else: errors += 1
        except Exception as exc:
            errors += 1
            traceback.print_exc()
            print(f"[ERR] {intent.symbol} persist failed: {exc}")

    if fills_to_write:
        append_or_replace_fills(paths["fills_csv"], fills_to_write)

    print(
        f"[SUMMARY] config={config_name} sent={sent} errors={errors} "
        f"dup_skipped={debug_dup} conid_missing={debug_conid_missing}"
    )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    _enable_line_buffering()
    print(
        f"[ENV] broker={BROKER_NAME} account={BROKER_ACCOUNT_ID} "
        f"whole_timeout={WHOLE_TIMEOUT_SEC}s whole_slippage={WHOLE_SLIPPAGE_BPS}bps "
        f"frac_slippage_min={FRAC_SLIPPAGE_MIN}bps frac_slippage_max={FRAC_SLIPPAGE_MAX}bps "
        f"parent_guard={PARENT_GUARD_PCT}%"
    )
    if not BROKER_ACCOUNT_ID:
        print("[FATAL] BROKER_ACCOUNT_ID required")
        return 1

    client = CpapiClient(CPAPI_BASE_URL, CPAPI_TIMEOUT_SEC, CPAPI_VERIFY_SSL)
    try:
        client.assert_authenticated()
    except Exception as exc:
        print(f"[FATAL] auth failed: {exc}")
        return 1

    client.start_tickle_loop()
    try:
        for cfg in CONFIG_NAMES:
            try:
                _run_config(client, cfg)
            except Exception as exc:
                traceback.print_exc()
                print(f"[ERR] config={cfg} failed: {exc}")
        dbg = client.debug_summary()
        print(
            f"[CPAPI_DEBUG] tickle_ok={dbg['tickle_ok']} tickle_fail={dbg['tickle_fail']} "
            f"order_submit={dbg['order_submit']} poll_calls={dbg['poll_calls']}"
        )
        print("[FINAL] CPAPI execution complete")
        return 0
    finally:
        client.stop_tickle_loop()


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 1
    _safe_exit(int(rc))
