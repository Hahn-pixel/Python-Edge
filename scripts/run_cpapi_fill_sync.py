"""
run_cpapi_fill_sync.py — синхронізація fills після закриття

Читає broker_log_YYYY-MM-DD.json (або broker_log.json як fallback),
звіряє з /iserver/account/orders і /iserver/account/trades,
оновлює broker_log і fills.csv.

Запускати о ~16:05 ET (21:05 Київ) того ж дня що і pipeline.
Подвійний клік — вікно не закривається до натискання Enter.
"""

from __future__ import annotations

import json
import os
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

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from python_edge.broker.cpapi_client import CpapiClient

# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------

def _env_str(k: str, d: str) -> str:   return str(os.getenv(k, d)).strip()
def _env_float(k: str, d: float) -> float:
    try: return float(os.getenv(k, str(d)))
    except Exception: return d
def _env_bool(k: str, d: bool) -> bool:
    return str(os.getenv(k, "1" if d else "0")).strip().lower() not in {"0","false","no","off"}

PAUSE_ON_EXIT     = _env_str("PAUSE_ON_EXIT", "auto")
EXECUTION_ROOT    = Path(_env_str("EXECUTION_ROOT", "artifacts/execution_loop"))
CONFIG_NAMES      = [x for x in _env_str("CONFIG_NAMES", "optimal|aggressive").split("|") if x]
CPAPI_BASE_URL    = _env_str("CPAPI_BASE_URL", "https://localhost:5000")
CPAPI_VERIFY_SSL  = _env_bool("CPAPI_VERIFY_SSL", False)
CPAPI_TIMEOUT_SEC = _env_float("CPAPI_TIMEOUT_SEC", 10.0)
BROKER_ACCOUNT_ID = _env_str("BROKER_ACCOUNT_ID", "")
# Дата для вибору broker_log_YYYY-MM-DD.json. Default = сьогодні UTC.
# Передавати явно якщо sync запускається після опівночі UTC.
SYNC_DATE         = _env_str("SYNC_DATE", "")

_FILLED_STATUSES    = frozenset({"Filled","FILLED","filled"})
_CANCELLED_STATUSES = frozenset({"Cancelled","CANCELLED","cancelled"})

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00","Z")

def _should_pause() -> bool:
    if PAUSE_ON_EXIT in {"0","false","no","off"}: return False
    if PAUSE_ON_EXIT in {"1","true","yes","on"}:  return True
    si, so = getattr(sys,"stdin",None), getattr(sys,"stdout",None)
    return bool(si and so and hasattr(si,"isatty") and si.isatty() and so.isatty())

def _to_float(v: Any) -> float:
    try: return 0.0 if v is None else float(v)
    except Exception: return 0.0


def _find_log_path(cfg_dir: Path, date: str) -> Optional[Path]:
    """
    Пріоритет:
      1. broker_log_YYYY-MM-DD.json (архів поточного дня)
      2. broker_log.json (поточний, якщо датований ще не створений)
    """
    dated = cfg_dir / f"broker_log_{date}.json"
    if dated.exists():
        return dated
    fallback = cfg_dir / "broker_log.json"
    if fallback.exists():
        return fallback
    return None

# ---------------------------------------------------------------------------
# Broker data
# ---------------------------------------------------------------------------

def _build_broker_map(client: CpapiClient) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    try:
        raw = client._get("/v1/api/iserver/account/orders")
        lst = raw.get("orders", raw) if isinstance(raw, dict) else raw
        for o in (lst if isinstance(lst, list) else []):
            if isinstance(o, dict):
                oid = str(o.get("orderId","") or o.get("order_id","") or "")
                if oid: result[oid] = o
    except Exception as exc:
        print(f"[FILL_SYNC][WARN] fetch orders failed: {exc}")
    return result


def _build_trades_map(client: CpapiClient) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    try:
        raw = client._get("/v1/api/iserver/account/trades")
        lst = raw if isinstance(raw, list) else raw.get("trades", [])
        for t in (lst if isinstance(lst, list) else []):
            if isinstance(t, dict):
                oid = str(t.get("orderId","") or t.get("order_id","") or "")
                if oid: result[oid] = t
    except Exception as exc:
        print(f"[FILL_SYNC][WARN] fetch trades failed: {exc}")
    return result

# ---------------------------------------------------------------------------
# Per-config sync
# ---------------------------------------------------------------------------

def _sync_config(
    config_name: str,
    date: str,
    broker_orders: Dict[str, Dict[str, Any]],
    broker_trades: Dict[str, Dict[str, Any]],
) -> Dict[str, int]:
    cfg_dir    = EXECUTION_ROOT / config_name
    fills_path = cfg_dir / "fills.csv"

    counters = {
        "working_found": 0, "sync_filled": 0, "sync_partial": 0,
        "sync_still_working": 0, "sync_not_found": 0, "sync_cancelled": 0,
    }

    log_path = _find_log_path(cfg_dir, date)
    if log_path is None:
        print(f"[FILL_SYNC][{config_name}] no broker_log found for date={date} — skip")
        return counters

    print(f"[FILL_SYNC][{config_name}] reading {log_path.name}")
    log = json.loads(log_path.read_text(encoding="utf-8"))
    orders_dict = log.get("orders", {})
    if not isinstance(orders_dict, dict):
        print(f"[FILL_SYNC][{config_name}] orders not a dict — skip")
        return counters

    new_fills: List[dict] = []
    modified = False

    for ikey, entry in orders_dict.items():
        outcome = str(entry.get("outcome","") or entry.get("status",""))
        if outcome not in {"working","submitted"}:
            continue

        broker_order_id = str(
            entry.get("broker_order_id","")
            or entry.get("response",{}).get("whole_order_id","")
            or ""
        )
        if not broker_order_id:
            continue

        counters["working_found"] += 1
        symbol = str(entry.get("symbol",""))
        side   = str(entry.get("side",""))

        broker_row = broker_orders.get(broker_order_id) or broker_trades.get(broker_order_id)
        if broker_row is None:
            counters["sync_not_found"] += 1
            print(f"[FILL_SYNC][{config_name}] NOT_FOUND {symbol} order_id={broker_order_id}")
            continue

        status      = str(broker_row.get("status","") or broker_row.get("order_ccp_status",""))
        filled_qty  = _to_float(broker_row.get("filledQuantity",0.0) or broker_row.get("filled",0.0))
        remaining   = _to_float(broker_row.get("remainingQuantity",0.0))
        avg_price   = _to_float(broker_row.get("avgPrice",0.0) or broker_row.get("price",0.0))
        fill_notional = round(filled_qty * avg_price, 4)

        is_filled    = status in _FILLED_STATUSES or (filled_qty > 0.0 and remaining <= 1e-9)
        is_partial   = filled_qty > 0.0 and remaining > 1e-9
        is_cancelled = status in _CANCELLED_STATUSES

        if is_filled:
            counters["sync_filled"] += 1
            entry.update({
                "filled_qty": filled_qty, "filled_avg_price": avg_price,
                "fill_notional": fill_notional, "remaining_qty": 0.0,
                "filled_at": _utc_now_iso(), "updated_at_utc": _utc_now_iso(),
                "status": "filled", "outcome": "filled",
            })
            resp = entry.setdefault("response", {})
            resp.update({
                "avg_fill_price": avg_price, "whole_filled_qty": filled_qty,
                "whole_avg_price": avg_price, "outcome": "filled", "status": "filled",
            })
            modified = True
            new_fills.append({
                "config": config_name, "date": str(entry.get("date","")),
                "symbol": symbol, "side": side, "qty": filled_qty,
                "avg_price": avg_price, "fill_notional": fill_notional,
                "broker_order_id": broker_order_id,
                "client_order_id": str(entry.get("client_order_id","")),
                "filled_at": entry["filled_at"], "source": "fill_sync",
            })
            print(
                f"[FILL_SYNC][{config_name}] FILLED {symbol} "
                f"side={side} qty={filled_qty} avg_px={avg_price:.4f} "
                f"notional={fill_notional:.2f} order_id={broker_order_id}"
            )

        elif is_partial:
            counters["sync_partial"] += 1
            entry.update({
                "filled_qty": filled_qty, "filled_avg_price": avg_price,
                "fill_notional": fill_notional, "remaining_qty": remaining,
                "updated_at_utc": _utc_now_iso(), "status": "partial", "outcome": "partial",
            })
            modified = True
            print(
                f"[FILL_SYNC][{config_name}] PARTIAL {symbol} "
                f"filled={filled_qty} remaining={remaining} avg_px={avg_price:.4f}"
            )

        elif is_cancelled:
            counters["sync_cancelled"] += 1
            entry.update({"status": "cancelled", "outcome": "cancelled", "updated_at_utc": _utc_now_iso()})
            modified = True
            print(f"[FILL_SYNC][{config_name}] CANCELLED {symbol} order_id={broker_order_id}")

        else:
            counters["sync_still_working"] += 1
            print(
                f"[FILL_SYNC][{config_name}] STILL_WORKING {symbol} "
                f"status={status} filled={filled_qty} remaining={remaining}"
            )

    if modified:
        log["orders"] = orders_dict
        log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
        # Синхронізуємо і broker_log.json якщо читали датований файл
        current_log = cfg_dir / "broker_log.json"
        if log_path != current_log:
            import shutil
            shutil.copy2(str(log_path), str(current_log))
            print(f"[FILL_SYNC][{config_name}] broker_log.json synced from {log_path.name}")
        else:
            print(f"[FILL_SYNC][{config_name}] broker_log.json updated")

    if new_fills:
        fills_df = pd.DataFrame(new_fills)
        if fills_path.exists():
            existing = pd.read_csv(fills_path)
            existing = existing.loc[
                ~existing["broker_order_id"].astype(str).isin(
                    set(fills_df["broker_order_id"].astype(str))
                )
            ]
            out = pd.concat([existing, fills_df], ignore_index=True)
        else:
            out = fills_df
        out.to_csv(fills_path, index=False)
        print(f"[FILL_SYNC][{config_name}] fills.csv updated rows={len(new_fills)}")

    return counters

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    date = SYNC_DATE or _utc_today()

    print("=" * 60)
    print("  Python-Edge — CPAPI Fill Sync")
    print("=" * 60)
    print(f"  ROOT:    {ROOT}")
    print(f"  ACCOUNT: {BROKER_ACCOUNT_ID}")
    print(f"  CONFIGS: {'|'.join(CONFIG_NAMES)}")
    print(f"  DATE:    {date}")
    print()

    if not BROKER_ACCOUNT_ID:
        print("[FATAL] BROKER_ACCOUNT_ID required")
        return 1

    client = CpapiClient(CPAPI_BASE_URL, CPAPI_TIMEOUT_SEC, CPAPI_VERIFY_SSL)
    try:
        client.assert_authenticated()
    except Exception as exc:
        print(f"[FATAL] auth failed: {exc}")
        return 1

    print("[FETCH] fetching broker orders and trades...")
    broker_orders = _build_broker_map(client)
    broker_trades = _build_trades_map(client)
    print(f"[FETCH] orders={len(broker_orders)} trades={len(broker_trades)}")

    total_filled = 0
    for cfg in CONFIG_NAMES:
        print(f"\n--- {cfg} ---")
        counters = _sync_config(cfg, date, broker_orders, broker_trades)
        total_filled += counters["sync_filled"]
        print(
            f"[FILL_SYNC][{cfg}][SUMMARY] "
            f"working_found={counters['working_found']} "
            f"sync_filled={counters['sync_filled']} "
            f"sync_partial={counters['sync_partial']} "
            f"sync_still_working={counters['sync_still_working']} "
            f"sync_not_found={counters['sync_not_found']} "
            f"sync_cancelled={counters['sync_cancelled']}"
        )

    print(f"\n[FINAL] fill sync complete — total_filled={total_filled}")
    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        print("\n[CRASHED]")
    finally:
        print()
        input("Press Enter to exit...")
    sys.exit(rc)
