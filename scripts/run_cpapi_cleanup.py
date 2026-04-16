from __future__ import annotations

import json
import os
import shutil
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
from python_edge.broker.cpapi_storage   import (
    append_or_replace_fills,
    build_broker_log_entry,
    duplicate_fill_entry,
    existing_duplicate_status,
    load_broker_log,
    save_broker_log,
    upsert_broker_log_entry,
)
from python_edge.broker.cpapi_conid_resolver import (
    _CACHE_FILENAME,
    load_conid_cache,
    update_conid_cache,
)

# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------

def _env_str(k: str, d: str) -> str: return str(os.getenv(k, d)).strip()
def _env_float(k: str, d: float) -> float:
    try: return float(os.getenv(k, str(d)))
    except Exception: return d
def _env_bool(k: str, d: bool) -> bool:
    return str(os.getenv(k, "1" if d else "0")).strip().lower() not in {"0","false","no","off"}
def _env_set(k: str) -> set:
    return {x.strip().upper() for x in str(os.getenv(k,"")).split("|") if x.strip()}

PAUSE_ON_EXIT         = _env_str("PAUSE_ON_EXIT", "auto")
EXECUTION_ROOT        = Path(_env_str("EXECUTION_ROOT", "artifacts/execution_loop"))
CONFIG_NAMES          = [x for x in _env_str("CONFIG_NAMES", "optimal|aggressive").split("|") if x]
BROKER_NAME           = _env_str("BROKER_NAME",       "MEXEM")
BROKER_PLATFORM       = _env_str("BROKER_PLATFORM",   "IBKR_CPAPI")
BROKER_ACCOUNT_ID     = _env_str("BROKER_ACCOUNT_ID", "")
CPAPI_BASE_URL        = _env_str("CPAPI_BASE_URL",     "https://localhost:5000")
CPAPI_VERIFY_SSL      = _env_bool("CPAPI_VERIFY_SSL",   False)
CPAPI_TIMEOUT_SEC     = _env_float("CPAPI_TIMEOUT_SEC",  10.0)
WHOLE_TIMEOUT_SEC     = _env_float("CPAPI_WHOLE_TIMEOUT_SEC", 30.0)
FRAC_TIMEOUT_SEC      = _env_float("CPAPI_FRAC_TIMEOUT_SEC",  20.0)
TIF                   = _env_str("CPAPI_TIF", "DAY")
FRAC_SLIPPAGE_BPS     = _env_float("CPAPI_FRAC_SLIPPAGE_BPS", 5.0)

CLEANUP_SEND_PREVIEW_ONLY               = _env_bool("CLEANUP_SEND_PREVIEW_ONLY", True)
CLEANUP_SEND_ONLY_SYMBOLS               = _env_set("CLEANUP_SEND_ONLY_SYMBOLS")
CLEANUP_SEND_EXCLUDE_SYMBOLS            = _env_set("CLEANUP_SEND_EXCLUDE_SYMBOLS")
CLEANUP_SEND_MAX_ROWS                   = int(os.getenv("CLEANUP_SEND_MAX_ROWS", "1000000"))
CLEANUP_SEND_REQUIRE_EMPTY_PENDING_REFS = _env_bool("CLEANUP_SEND_REQUIRE_EMPTY_PENDING_REFS", False)
CLEANUP_SEND_REQUIRE_SIDE               = _env_str("CLEANUP_SEND_REQUIRE_SIDE", "BUY|SELL")
CLEANUP_SEND_REASON_TAG                 = _env_str("CLEANUP_SEND_REASON_TAG", "broker_cleanup_send")
RESET_BROKER_LOG                        = _env_bool("RESET_BROKER_LOG", False)
TOPK_PRINT                              = int(os.getenv("TOPK_PRINT", "50"))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00","Z")

def _to_float(v: Any) -> float:
    try: return 0.0 if v is None else float(v)
    except Exception: return 0.0

def _norm(s: Any) -> str: return str(s or "").strip().upper()

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
# Load + filter cleanup orders
# ---------------------------------------------------------------------------

def _load_cleanup_orders(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"cleanup orders not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["symbol","order_side","delta_shares",
                                     "cleanup_reason","pending_order_refs"])
    df["symbol"]       = df["symbol"].astype(str).map(_norm)
    df["order_side"]   = df["order_side"].astype(str).str.strip().str.upper()
    df["delta_shares"] = pd.to_numeric(df["delta_shares"], errors="coerce").fillna(0.0)
    for col in ("pending_order_refs","cleanup_reason"):
        if col not in df.columns: df[col] = ""
        df[col] = df[col].astype(str).fillna("")
    return df.sort_values(["delta_shares","symbol"], ascending=[False,True]).reset_index(drop=True)


def _passes_filters(row: pd.Series) -> tuple[bool, str]:
    symbol = _norm(row.get("symbol",""))
    side   = str(row.get("order_side","") or "").strip().upper()
    qty    = abs(_to_float(row.get("delta_shares", 0.0)))
    refs   = str(row.get("pending_order_refs","") or "").strip()

    if not symbol:
        return False, "empty_symbol"
    if CLEANUP_SEND_ONLY_SYMBOLS and symbol not in CLEANUP_SEND_ONLY_SYMBOLS:
        return False, "not_in_only_symbols"
    if CLEANUP_SEND_EXCLUDE_SYMBOLS and symbol in CLEANUP_SEND_EXCLUDE_SYMBOLS:
        return False, "excluded_symbol"
    if qty <= 0.0:
        return False, "non_positive_qty"
    allowed = {x.strip().upper() for x in CLEANUP_SEND_REQUIRE_SIDE.split("|") if x.strip()}
    if side not in allowed:
        return False, "side_not_allowed"
    if CLEANUP_SEND_REQUIRE_EMPTY_PENDING_REFS and refs:
        return False, "pending_refs_present"
    return True, "emit"


def _build_send_plan(
    paths: Dict[str, Path],
    cleanup_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    rows: List[Dict[str, Any]] = []
    counters: Dict[str, int] = {
        "cleanup_rows": len(cleanup_df),
        "emitted_rows": 0,
        "blocked_empty_symbol": 0,
        "blocked_not_in_only_symbols": 0,
        "blocked_excluded_symbol": 0,
        "blocked_non_positive_qty": 0,
        "blocked_side_not_allowed": 0,
        "blocked_pending_refs_present": 0,
        "blocked_max_rows": 0,
    }
    emitted = 0
    for _, row in cleanup_df.iterrows():
        ok, reason = _passes_filters(row)
        if not ok:
            key = f"blocked_{reason}"
            counters[key] = counters.get(key, 0) + 1
            continue
        if emitted >= CLEANUP_SEND_MAX_ROWS:
            counters["blocked_max_rows"] += 1
            continue
        emitted += 1
        counters["emitted_rows"] = emitted
        sym   = _norm(row["symbol"])
        side  = str(row["order_side"] or "").strip().upper()
        qty   = abs(_to_float(row["delta_shares"]))
        avg_c = _to_float(row.get("broker_avg_cost", 0.0))
        rows.append({
            "date":             "",
            "symbol":           sym,
            "order_side":       side,
            "delta_shares":     qty,
            "price":            avg_c if avg_c > 0.0 else None,
            "price_source":     "cleanup_send_plan",
            "price_hint_source":"cleanup_broker_avg_cost" if avg_c > 0.0 else "cleanup_no_price",
            "quote_ts":         _utc_now_iso(),
            "quote_provider":   "cleanup_preview",
            "quote_timeframe":  "N/A",
            "model_price_reference": avg_c if avg_c > 0.0 else 0.0,
            "bid": 0.0, "ask": 0.0, "mid": 0.0, "last": 0.0, "close_price": 0.0,
            "spread_bps": 0.0, "price_deviation_vs_model": 0.0,
            "is_priced":        1 if avg_c > 0.0 else 0,
            "target_weight":    0.0, "target_notional":   0.0,
            "target_shares_raw":0.0,
            "current_shares":   _to_float(row.get("broker_position", 0.0)),
            "target_shares":    _to_float(row.get("expected_shares", 0.0)),
            "raw_delta_shares": qty if side == "BUY" else -qty,
            "order_notional":   0.0, "order_notional_abs": 0.0,
            "estimated_commission": 0.0, "estimated_slippage": 0.0, "estimated_total_cost": 0.0,
            "skip_reason":      "",
            "cleanup_reason":   str(row.get("cleanup_reason","") or ""),
            "cleanup_tag":      CLEANUP_SEND_REASON_TAG,
            "pending_order_refs": str(row.get("pending_order_refs","") or ""),
            "fallback_reason":  str(row.get("cleanup_reason","") or ""),
        })

    plan_df = pd.DataFrame(rows)
    if not plan_df.empty:
        plan_df = plan_df.sort_values(["delta_shares","symbol"],ascending=[False,True]).reset_index(drop=True)
    summary = {
        "config":           paths["execution_dir"].name,
        "generated_at":     _utc_now_iso(),
        "preview_only":     int(CLEANUP_SEND_PREVIEW_ONLY),
        "counters":         counters,
        "filters": {
            "only_symbols":           sorted(CLEANUP_SEND_ONLY_SYMBOLS),
            "exclude_symbols":        sorted(CLEANUP_SEND_EXCLUDE_SYMBOLS),
            "max_rows":               CLEANUP_SEND_MAX_ROWS,
            "require_empty_pending":  int(CLEANUP_SEND_REQUIRE_EMPTY_PENDING_REFS),
            "require_side":           CLEANUP_SEND_REQUIRE_SIDE,
        },
    }
    return plan_df, summary

# ---------------------------------------------------------------------------
# Execute plan via CPAPI
# ---------------------------------------------------------------------------

def _execute_plan(
    client: CpapiClient,
    config_name: str,
    plan_df: pd.DataFrame,
    paths: Dict[str, Path],
) -> None:
    """
    Reuse the same execution engine as run_cpapi_execution.py.
    Uses broker_avg_cost as reference_price (cleanup orders have no live quote).

    Auto-resolves missing conids via CPAPI before building intents.
    This makes cleanup self-sufficient regardless of what prior pipeline
    steps put in conid_cache.json.
    """
    cache_path = paths["execution_dir"] / _CACHE_FILENAME

    # ── Auto-resolve any conids missing from cache ─────────────────────
    # Done here (not in a prior pipeline step) so cleanup is always
    # self-sufficient even if step 5b was skipped or wrote to a wrong path.
    all_symbols = sorted(
        plan_df["symbol"].dropna().astype(str).str.strip().str.upper().unique().tolist()
    )
    if all_symbols:
        conid_cache, unresolved_syms = update_conid_cache(
            cache_path, client, all_symbols, force_refresh=False
        )
        if unresolved_syms:
            print(
                f"[CLEANUP][{config_name}][CONID_WARN] "
                f"{len(unresolved_syms)} symbols still unresolved after auto-resolve: "
                f"{unresolved_syms} — these will be skipped"
            )
    else:
        conid_cache = load_conid_cache(cache_path)

    broker_log = load_broker_log(
        path              = paths["broker_log"],
        config_name       = config_name,
        broker_name       = BROKER_NAME,
        broker_platform   = BROKER_PLATFORM,
        broker_account_id = BROKER_ACCOUNT_ID,
        utc_now_iso       = _utc_now_iso,
        reset             = RESET_BROKER_LOG,
    )

    import hashlib
    from datetime import date as _date
    intents: List[ExecutionIntent] = []
    ikeys:   List[str]             = []
    metas:   List[Dict[str,Any]]   = []
    ref_prices: Dict[str, float]   = {}
    dup_entries: List[dict]        = []
    debug_no_conid = 0
    debug_no_price = 0
    debug_dup      = 0

    # run_date makes cOID unique per calendar day — prevents Gateway from
    # rejecting re-submitted cleanup orders with "Local order ID already registered"
    run_date = _date.today().isoformat()   # e.g. "2026-04-16"

    for _, row in plan_df.iterrows():
        sym   = _norm(str(row.get("symbol","")))
        side  = str(row.get("order_side","") or "").strip().upper()
        qty   = abs(_to_float(row.get("delta_shares", 0.0)))
        price = _to_float(row.get("price", 0.0)) or _to_float(row.get("model_price_reference", 0.0))

        if qty <= 0.0:
            continue

        ikey = hashlib.sha256(
            json.dumps({"config": config_name, "symbol": sym, "side": side,
                        "delta_shares": round(qty,8), "cleanup_tag": CLEANUP_SEND_REASON_TAG,
                        "run_date": run_date},
                       sort_keys=True, separators=(",",":")).encode()
        ).hexdigest()

        dup = existing_duplicate_status(broker_log, ikey)
        if dup is not None:
            debug_dup += 1
            dup_entries.append(duplicate_fill_entry(
                idempotency_key=ikey, client_tag=f"cl-{ikey[:24]}",
                symbol=sym, side=side, qty=qty, price_hint=price,
                order_notional=qty*price, order_date="", config_name=config_name,
                source_order_path=str(paths["cleanup_orders"]),
                duplicate_status=dup, broker_log=broker_log,
            ))
            continue

        conid = conid_cache.get(sym, "")
        if not conid:
            debug_no_conid += 1
            print(f"[CLEANUP][SKIP][NO_CONID] symbol={sym}")
            continue

        if price <= 0.0:
            debug_no_price += 1
            print(f"[CLEANUP][SKIP][NO_PRICE] symbol={sym}")
            continue

        # For cleanup: use avg_cost as both cap and floor (50% deviation — legacy positions)
        parent_cap   = price * 1.50 if side == "BUY"  else None
        parent_floor = price * 0.50 if side == "SELL" else None

        intents.append(ExecutionIntent(
            symbol=sym, conid=conid, side=OrderSide(side),
            target_qty=qty, parent_cap=parent_cap, parent_floor=parent_floor,
            client_tag=f"cl-{ikey[:24]}", account_id=BROKER_ACCOUNT_ID,
        ))
        ikeys.append(ikey)
        metas.append({"price_hint": price, "order_notional": qty*price,
                      "order_date": "", "source_order_path": str(paths["cleanup_orders"])})
        ref_prices[sym] = price

    if dup_entries:
        append_or_replace_fills(paths["fills"], dup_entries)

    print(
        f"[CLEANUP][{config_name}] to_execute={len(intents)} "
        f"dup={debug_dup} no_conid={debug_no_conid} no_price={debug_no_price}"
    )

    if not intents:
        return

    engine_kwargs = {
        "whole_timeout_sec": WHOLE_TIMEOUT_SEC,
        "frac_timeout_sec":  FRAC_TIMEOUT_SEC,
        "tif":               TIF,
        "frac_slippage_bps": FRAC_SLIPPAGE_BPS,
    }
    results = run_batch(client, intents, ref_prices, engine_kwargs)

    sent = errors = 0
    fills_to_write: List[dict] = []
    for intent, result, ikey, meta in zip(intents, results, ikeys, metas):
        try:
            log_entry = build_broker_log_entry(
                config_name=config_name, order_date=meta["order_date"],
                order_notional=meta["order_notional"], price_hint=meta["price_hint"],
                source_order_path=meta["source_order_path"], idempotency_key=ikey,
                intent=intent, result=result, utc_now_iso=_utc_now_iso,
            )
            upsert_broker_log_entry(broker_log, log_entry, _utc_now_iso)
            save_broker_log(paths["broker_log"], broker_log, _utc_now_iso)
            if result.total_filled > 0.0:
                fills_to_write.append(log_entry)
            if result.final_state is ExecState.DONE:
                sent  += 1
            else:
                errors += 1
            print(
                f"[CLEANUP][SEND] symbol={intent.symbol} side={intent.side.value} "
                f"qty={intent.target_qty:.4f} state={result.final_state.value} "
                f"filled={result.total_filled:.4f}"
            )
        except Exception as exc:
            errors += 1
            traceback.print_exc()
            print(f"[CLEANUP][ERR] symbol={intent.symbol}: {exc}")

    if fills_to_write:
        append_or_replace_fills(paths["fills"], fills_to_write)

    print(f"[CLEANUP][{config_name}][SEND_SUMMARY] sent={sent} errors={errors}")

# ---------------------------------------------------------------------------
# Per-config runner
# ---------------------------------------------------------------------------

def _run_one(config_name: str, client: Optional[CpapiClient]) -> None:
    base = EXECUTION_ROOT / config_name
    base.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, Path] = {
        "execution_dir":   base,
        "cleanup_orders":  base / "broker_cleanup_orders.csv",
        "orders":          base / "orders.csv",
        "orders_backup":   base / "orders_pre_cleanup_backup.csv",
        "send_plan":       base / "broker_cleanup_send_plan.csv",
        "send_summary":    base / "broker_cleanup_send_summary.json",
        "broker_log":      base / "broker_log.json",
        "fills":           base / "fills.csv",
    }

    cleanup_df       = _load_cleanup_orders(paths["cleanup_orders"])
    plan_df, summary = _build_send_plan(paths, cleanup_df)

    # Save plan
    plan_df.to_csv(paths["send_plan"], index=False)
    paths["send_summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    counters = summary["counters"]
    print(
        f"[CLEANUP][{config_name}][PLAN] "
        f"cleanup_rows={counters['cleanup_rows']} emitted_rows={counters['emitted_rows']} "
        f"preview_only={summary['preview_only']} "
        f"blocked_pending_refs={counters.get('blocked_pending_refs_present',0)}"
    )
    if not plan_df.empty:
        print(plan_df.head(min(TOPK_PRINT, len(plan_df))).to_string(index=False))
    print(f"[ARTIFACT] {paths['send_plan']}")
    print(f"[ARTIFACT] {paths['send_summary']}")

    if CLEANUP_SEND_PREVIEW_ONLY:
        print(f"[CLEANUP][{config_name}] preview-only — adapter not called")
        return
    if plan_df.empty:
        print(f"[CLEANUP][{config_name}] no rows after filters — adapter not called")
        return
    if client is None:
        raise RuntimeError("CpapiClient is required for non-preview cleanup execution")

    # Swap orders.csv → cleanup plan (mirror of ibkr cleanup behaviour)
    if paths["orders"].exists():
        shutil.copy2(paths["orders"], paths["orders_backup"])
    plan_df.to_csv(paths["orders"], index=False)
    try:
        _execute_plan(client, config_name, plan_df, paths)
    finally:
        if paths["orders_backup"].exists():
            shutil.copy2(paths["orders_backup"], paths["orders"])
            print(f"[CLEANUP][{config_name}] original orders.csv restored")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    _enable_lb()
    print(f"[CFG] execution_root={EXECUTION_ROOT} configs={CONFIG_NAMES}")
    print(f"[CFG] cpapi_url={CPAPI_BASE_URL} account={BROKER_ACCOUNT_ID}")
    print(
        f"[CFG] cleanup_send_preview_only={int(CLEANUP_SEND_PREVIEW_ONLY)} "
        f"require_empty_pending={int(CLEANUP_SEND_REQUIRE_EMPTY_PENDING_REFS)}"
    )

    if not BROKER_ACCOUNT_ID and not CLEANUP_SEND_PREVIEW_ONLY:
        print("[FATAL] BROKER_ACCOUNT_ID is required for non-preview cleanup")
        return 1

    client: Optional[CpapiClient] = None
    if not CLEANUP_SEND_PREVIEW_ONLY:
        client = CpapiClient(CPAPI_BASE_URL, CPAPI_TIMEOUT_SEC, CPAPI_VERIFY_SSL)
        client.assert_authenticated()
        client.start_tickle_loop()

    try:
        for cfg in CONFIG_NAMES:
            _run_one(cfg, client)
    finally:
        if client is not None:
            client.stop_tickle_loop()

    print("[FINAL] CPAPI cleanup send complete")
    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 1
    _safe_exit(rc)
