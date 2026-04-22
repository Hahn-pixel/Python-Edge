"""
launch_full_cycle_cpapi.py — повний цикл виконання (CPAPI)

Замінює run_full_cycle_cpapi.ps1. Запускає всі кроки pipeline
напряму через subprocess без PowerShell.

PIPELINE:
  0)  Очищення артефактів execution_loop
  1)  run_cpapi_reconcile.py   — positions -> broker_positions.csv
  1b) умовне видалення portfolio_state.json (тільки якщо немає позицій)
  2a) run_execution_loop.py    — optimal
  2b) run_execution_loop.py    — aggressive
  2c) run_exit_scanner.py      — price-based exits
  2d) freeze gate diagnostics
  3)  run_cpapi_handoff.py     — live BBO
  4)  run_cpapi_execution.py   — відправка ордерів
  5)  run_cpapi_reconcile.py   — post-execution reconcile
  5a) sync cleanup preview -> orders
  5b) resolve conids for cleanup
  6)  run_cpapi_cleanup.py     — cleanup send

Передумова: Client Portal Gateway запущений на https://localhost:5000
            і авторизований через браузер.

Подвійний клік — вікно не закривається до натискання Enter.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from datetime import date
from pathlib import Path

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ── Налаштування ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent

IB_ACCOUNT_CODE  = "DUP561175"
MASSIVE_API_KEY  = os.getenv("MASSIVE_API_KEY", "")
MASSIVE_BASE_URL = "https://api.massive.com"
CPAPI_BASE_URL   = "https://localhost:5000"

# Exit policy (порожнє = використовувати дефолти з exit_policy.py)
EXIT_OPTIMAL_STOP_LOSS_PCT     = ""   # default: 0.08
EXIT_OPTIMAL_TAKE_PROFIT_PCT   = ""   # default: 0.25
EXIT_OPTIMAL_TRAIL_K           = ""   # default: 1.5
EXIT_OPTIMAL_TRAIL_MIN         = ""   # default: 0.03
EXIT_OPTIMAL_TRAIL_MAX         = ""   # default: 0.20
EXIT_OPTIMAL_MAX_HOLD_DAYS     = ""   # default: 30

EXIT_AGGRESSIVE_STOP_LOSS_PCT     = ""   # default: 0.10
EXIT_AGGRESSIVE_TAKE_PROFIT_PCT   = ""   # default: 0.30
EXIT_AGGRESSIVE_TRAIL_K           = ""   # default: 1.5
EXIT_AGGRESSIVE_TRAIL_MIN         = ""   # default: 0.03
EXIT_AGGRESSIVE_TRAIL_MAX         = ""   # default: 0.20
EXIT_AGGRESSIVE_MAX_HOLD_DAYS     = ""   # default: 20

# ──────────────────────────────────────────────────────────────

LOCK_FILE = ROOT / "artifacts" / "pipeline.lock"

SCRIPT_RECONCILE    = ROOT / "scripts" / "run_cpapi_reconcile.py"
SCRIPT_EXEC         = ROOT / "scripts" / "run_execution_loop.py"
SCRIPT_EXIT_SCANNER = ROOT / "scripts" / "run_exit_scanner.py"
SCRIPT_HANDOFF      = ROOT / "scripts" / "run_cpapi_handoff.py"
SCRIPT_EXEC_CPAPI   = ROOT / "scripts" / "run_cpapi_execution.py"
SCRIPT_CLEANUP      = ROOT / "scripts" / "run_cpapi_cleanup.py"


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def _check_gateway() -> bool:
    try:
        import requests
        resp = requests.get(
            f"{CPAPI_BASE_URL}/v1/api/iserver/auth/status",
            verify=False, timeout=5,
        )
        data = resp.json()
        authenticated = bool(data.get("authenticated", False))
        print(f"[GATEWAY] authenticated={authenticated} connected={data.get('connected')}")
        return authenticated
    except Exception as exc:
        print(f"[GATEWAY] unreachable: {exc}")
        return False


def _fetch_nav() -> str:
    try:
        import requests
        s = requests.Session()
        s.verify = False
        resp = s.get(f"{CPAPI_BASE_URL}/v1/api/iserver/accounts", timeout=10)
        accounts = resp.json().get("accounts", [])
        if not accounts:
            return ""
        acct = accounts[0]
        resp2 = s.get(f"{CPAPI_BASE_URL}/v1/api/portfolio/{acct}/summary", timeout=10)
        nl = resp2.json().get("netliquidation", {})
        nav = str(nl.get("amount", "")) if isinstance(nl, dict) else str(nl)
        if nav:
            print(f"[NAV] {nav} (from CPAPI)")
        return nav
    except Exception as exc:
        print(f"[NAV] fetch failed: {exc}")
        return ""


def _run(label: str, script: Path, env: dict) -> None:
    """Запустити скрипт. При ненульовому exitcode — raise."""
    if not script.exists():
        raise FileNotFoundError(f"Script not found: {script}")
    print(f"\n{'='*60}")
    print(f"[STEP] {label}")
    print(f"{'='*60}")
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(ROOT),
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"[{label}] FAILED exit={result.returncode}")
    print(f"[{label}] OK")


def _run_inline(label: str, code: str, env: dict) -> None:
    """Запустити Python-код inline через -c."""
    print(f"\n[STEP] {label}")
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(ROOT),
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"[{label}] FAILED exit={result.returncode}")
    print(f"[{label}] OK")


def _clean_artifacts(env: dict) -> None:
    print("\n=== STEP 0: CLEAN ARTIFACTS ===")
    to_delete = [
        "orders.csv", "target_book.csv", "broker_log.json", "fills.csv",
        "execution_log.csv", "positions_mark_to_market.csv",
        "broker_handoff_summary.json", "broker_positions.csv",
        "broker_open_orders.csv", "broker_pending.csv",
        "broker_reconcile.csv", "broker_reconcile_summary.json",
        "broker_cleanup_preview.csv", "broker_cleanup_orders.csv",
        "broker_cleanup_emit_summary.json", "broker_cleanup_send_plan.csv",
        "broker_cleanup_send_summary.json", "orders_exec_backup.csv",
        "orders_pre_cleanup_backup.csv", "state_alignment_once.json",
    ]
    for cfg in ["optimal", "aggressive"]:
        cfg_dir = ROOT / "artifacts" / "execution_loop" / cfg
        if not cfg_dir.exists():
            continue
        for f in to_delete:
            p = cfg_dir / f
            if p.exists():
                p.unlink()
                print(f"  [{cfg}] deleted {f}")
        print(f"  [{cfg}] clean OK (portfolio_state.json preserved)")


def _create_orders_stubs() -> None:
    for cfg in ["optimal", "aggressive"]:
        cfg_dir = ROOT / "artifacts" / "execution_loop" / cfg
        cfg_dir.mkdir(parents=True, exist_ok=True)
        orders = cfg_dir / "orders.csv"
        if not orders.exists():
            orders.write_text(
                "symbol,order_side,delta_shares,target_shares,price,price_source\n",
                encoding="utf-8",
            )
            print(f"  [{cfg}] created empty orders.csv stub")


def _broker_positions_has_data(cfg: str) -> bool:
    """
    Повертає True якщо broker_positions.csv для config містить >=1 рядок з позицією.
    Використовується для вирішення: видаляти state чи ні.
    """
    p = ROOT / "artifacts" / "execution_loop" / cfg / "broker_positions.csv"
    if not p.exists():
        return False
    try:
        import csv
        with p.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Будь-який рядок з непустим symbol — це реальна позиція
                sym = (row.get("symbol") or "").strip()
                if sym:
                    return True
    except Exception:
        pass
    return False


def _conditional_remove_state_for_alignment() -> None:
    """
    STEP 1b — умовне видалення portfolio_state.json.

    Логіка:
    - Якщо broker_positions.csv порожній (paper account без позицій):
        → видаляємо state → STEP 2 зробить bootstrap alignment,
          але потім ЗМОЖЕ генерувати ордери на вхід у нові позиції
          (бо broker_positions порожній = стан 0, ціль > 0 → order_count > 0).
    - Якщо broker_positions.csv має реальні позиції:
        → НЕ видаляємо state → STEP 2 порівняє state з broker_positions
          і згенерує тільки drift-ордери.

    Старий підхід (завжди видаляти) призводив до bootstrap_only=1 і нуль ордерів
    коли broker_positions порожній, бо після alignment state = 0 позицій,
    і система вважала що вже "на цілі".
    """
    print("[1b] conditional state removal for alignment")
    for cfg in ["optimal", "aggressive"]:
        has_positions = _broker_positions_has_data(cfg)
        cfg_dir = ROOT / "artifacts" / "execution_loop" / cfg
        if has_positions:
            # Є реальні позиції → зберігаємо state, видаляємо тільки маркер alignment
            marker = cfg_dir / "state_alignment_once.json"
            if marker.exists():
                marker.unlink()
                print(f"  [{cfg}] broker has positions → keeping state, removed alignment marker")
            else:
                print(f"  [{cfg}] broker has positions → keeping state (no marker)")
        else:
            # Порожній broker → видаляємо state щоб alignment пройшов коректно,
            # але STEP 2 все одно згенерує ордери бо поточний стан = 0, ціль > 0
            for fname in ["portfolio_state.json", "state_alignment_once.json"]:
                p = cfg_dir / fname
                if p.exists():
                    p.unlink()
                    print(f"  [{cfg}] broker empty → removed {fname}")
                else:
                    print(f"  [{cfg}] broker empty → {fname} already absent")


def main() -> int:
    today_str = date.today().isoformat()

    print("=" * 60)
    print("  Python-Edge — Full Cycle CPAPI (Execution)")
    print("=" * 60)
    print(f"  ROOT:    {ROOT}")
    print(f"  ACCOUNT: {IB_ACCOUNT_CODE}")
    print(f"  CPAPI:   {CPAPI_BASE_URL}")
    print(f"  DATE:    {today_str}")
    print()

    # Pre-flight checks
    print("[CHECK] Gateway status...")
    if not _check_gateway():
        print("[ERROR] Gateway not authenticated")
        print("        Відкрийте браузер: https://localhost:5000")
        print("        та авторизуйтесь через IBKR")
        return 1

    print("[NAV] Fetching current NAV...")
    nav = _fetch_nav()
    if not nav:
        print("[WARN] Could not fetch NAV — pipeline will use freeze-based sizing")
        nav = "0"

    if not MASSIVE_API_KEY:
        print("[WARN] MASSIVE_API_KEY not set — massive fallback disabled")

    # Preflight scripts check
    print("\n[PREFLIGHT] Checking scripts...")
    missing = [s for s in [
        SCRIPT_RECONCILE, SCRIPT_EXEC, SCRIPT_EXIT_SCANNER,
        SCRIPT_HANDOFF, SCRIPT_EXEC_CPAPI, SCRIPT_CLEANUP,
    ] if not s.exists()]
    if missing:
        for s in missing:
            print(f"  MISSING: {s}")
        return 1

    # Базовий env — спільний для всіх кроків
    base_env = os.environ.copy()
    base_env.update({
        "EXECUTION_ROOT":    "artifacts/execution_loop",
        "BROKER_ACCOUNT_ID": IB_ACCOUNT_CODE,
        "CPAPI_BASE_URL":    CPAPI_BASE_URL,
        "CPAPI_VERIFY_SSL":  "0",
        "CPAPI_TIMEOUT_SEC": "10.0",
        "MASSIVE_API_KEY":   MASSIVE_API_KEY,
        "MASSIVE_BASE_URL":  MASSIVE_BASE_URL,
        "PAUSE_ON_EXIT":     "0",
    })

    # Exit policy env
    for k, v in {
        "EXIT_OPTIMAL_STOP_LOSS_PCT":        EXIT_OPTIMAL_STOP_LOSS_PCT,
        "EXIT_OPTIMAL_TAKE_PROFIT_PCT":      EXIT_OPTIMAL_TAKE_PROFIT_PCT,
        "EXIT_OPTIMAL_TRAIL_K":              EXIT_OPTIMAL_TRAIL_K,
        "EXIT_OPTIMAL_TRAIL_MIN":            EXIT_OPTIMAL_TRAIL_MIN,
        "EXIT_OPTIMAL_TRAIL_MAX":            EXIT_OPTIMAL_TRAIL_MAX,
        "EXIT_OPTIMAL_MAX_HOLD_DAYS":        EXIT_OPTIMAL_MAX_HOLD_DAYS,
        "EXIT_AGGRESSIVE_STOP_LOSS_PCT":     EXIT_AGGRESSIVE_STOP_LOSS_PCT,
        "EXIT_AGGRESSIVE_TAKE_PROFIT_PCT":   EXIT_AGGRESSIVE_TAKE_PROFIT_PCT,
        "EXIT_AGGRESSIVE_TRAIL_K":           EXIT_AGGRESSIVE_TRAIL_K,
        "EXIT_AGGRESSIVE_TRAIL_MIN":         EXIT_AGGRESSIVE_TRAIL_MIN,
        "EXIT_AGGRESSIVE_TRAIL_MAX":         EXIT_AGGRESSIVE_TRAIL_MAX,
        "EXIT_AGGRESSIVE_MAX_HOLD_DAYS":     EXIT_AGGRESSIVE_MAX_HOLD_DAYS,
    }.items():
        if v:
            base_env[k] = v

    # LOCK
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOCK_FILE.write_text(f"pid={os.getpid()}", encoding="utf-8")
    print(f"[LOCK] created: {LOCK_FILE}")

    try:
        # ── STEP 0 ────────────────────────────────────────────
        _clean_artifacts(base_env)

        # ── STEP 1: Reconcile pre-alignment ───────────────────
        print("\n=== STEP 1: CPAPI RECONCILE (pre-alignment) ===")
        _create_orders_stubs()

        env1 = {**base_env, **{
            "CONFIG_NAMES":                        "optimal|aggressive",
            "DRIFT_TOLERANCE_SHARES":              "0.000001",
            "PREFER_STATE_OVER_ORDERS":            "1",
            "REQUIRE_BROKER_REFRESH":              "1",
            "ALLOW_EXISTING_BROKER_POSITIONS_CSV": "1",
            "CLEANUP_PREVIEW_MODE":                "1",
            "CLEANUP_INCLUDE_UNEXPECTED":          "1",
            "CLEANUP_INCLUDE_DRIFT":               "1",
            "CLEANUP_ONLY_SYMBOLS":                "",
            "CLEANUP_EXCLUDE_SYMBOLS":             "",
            "CLEANUP_MIN_ABS_SHARES":              "1.0",
            "TOPK_PRINT":                          "50",
        }}
        _run("CPAPI_RECONCILE_PRE", SCRIPT_RECONCILE, env1)

        # ── STEP 1b: умовне видалення state ───────────────────
        _conditional_remove_state_for_alignment()

        # ── STEP 2a: Execution loop optimal ───────────────────
        print("\n=== STEP 2: EXECUTION LOOP ===")
        env2 = {**base_env, **{
            "LIVE_FEATURE_SNAPSHOT_FILE":               "artifacts/live_alpha/live_feature_snapshot.parquet",
            "FREEZE_ROOT":                              "artifacts/freeze_runner",
            "ACCOUNT_NAV":                              nav,
            "ALLOW_FRACTIONAL_SHARES":                  "0",
            "FRACTIONAL_MODE":                          "integer",
            "MIN_ORDER_NOTIONAL":                       "25.0",
            "DEFAULT_PRICE_FALLBACK":                   "100.0",
            "RESET_STATE":                              "0",
            "SKIP_EMPTY_FREEZE_CONFIGS":                "1",
            "REQUIRE_FREEZE_DATE_MATCH_LIVE_PRICES":    "1",
            "MAX_SINGLE_NAME_WEIGHT":                   "0.05",
            "MAX_SINGLE_NAME_NOTIONAL":                 "10000.0",
            "MIN_PRICE_TO_TRADE":                       "1.0",
            "MAX_PRICE_TO_TRADE":                       "1000000.0",
            "COMMISSION_BPS":                           "0.50",
            "COMMISSION_MIN_PER_ORDER":                 "0.35",
            "SLIPPAGE_BPS":                             "1.50",
            "ALIGN_STATE_FROM_BROKER_ONCE":             "1",
            "ALIGN_STATE_REQUIRE_BROKER_POSITIONS":     "0",
            "ALIGN_STATE_CASH_MODE":                    "preserve_nav",
            "SKIP_LEGACY_SYMBOLS_NOT_IN_TARGET":        "1",
            "USE_STATE_PRICE_FALLBACK_FOR_LEGACY":      "1",
            "USE_STATE_PRICE_FALLBACK_FOR_MTM":         "1",
            "USE_BROKER_POSITIONS_PRICE_FALLBACK_FOR_LEGACY": "1",
            "USE_BROKER_POSITIONS_PRICE_FALLBACK_FOR_MTM":    "1",
            "SANITIZE_STATE_PRICES":                    "1",
            "PERSIST_DEFAULT_FALLBACK_TO_STATE":        "0",
            "TOPK_PRINT":                               "50",
        }}

        env2_optimal    = {**env2, "CONFIG_NAMES": "optimal"}
        env2_aggressive = {**env2, "CONFIG_NAMES": "aggressive"}

        print("[2a] optimal")
        _run("EXEC_LOOP_OPTIMAL", SCRIPT_EXEC, env2_optimal)

        print("[2b] aggressive")
        _run("EXEC_LOOP_AGGRESSIVE", SCRIPT_EXEC, env2_aggressive)

        # ── STEP 2c: Exit scanner ─────────────────────────────
        print("\n=== STEP 2c: EXIT SCANNER ===")
        env2c = {**base_env, **{
            "CONFIG_NAMES":                   "optimal|aggressive",
            "LIVE_FEATURE_SNAPSHOT_FILE":     "artifacts/live_alpha/live_feature_snapshot.parquet",
            "EXIT_SCANNER_DRY_RUN":           "0",
            "EXIT_SCANNER_USE_STATE_PRICE":   "0",
        }}
        _run("EXIT_SCANNER", SCRIPT_EXIT_SCANNER, env2c)

        # ── STEP 2d: Freeze gate diagnostics ──────────────────
        print("\n=== STEP 2d: FREEZE GATE DIAGNOSTICS ===")
        gate_ok_all = True
        for cfg in ["optimal", "aggressive"]:
            freeze_summary = (
                ROOT / "artifacts" / "freeze_runner" / cfg
                / "freeze_current_summary.json"
            )
            if freeze_summary.exists():
                try:
                    data = json.loads(freeze_summary.read_text(encoding="utf-8"))
                    gate    = data.get("live_gate_status", "UNKNOWN")
                    reason  = data.get("live_gate_reason", "n/a")
                    active  = data.get("live_active_names", "?")
                    raw_n   = data.get("live_active_names_raw", active)
                    sharpe  = data.get("live_sharpe", "?")
                    cumret  = data.get("live_cumret", "?")
                    snap_dt = data.get("live_snapshot_current_date", "?")
                    print(
                        f"  [{cfg}] gate={gate} reason={reason} "
                        f"live_active={active}(raw={raw_n}) "
                        f"sharpe={sharpe} cumret={cumret} date={snap_dt}"
                    )
                    if gate != "PASSED":
                        gate_ok_all = False
                except Exception as exc:
                    print(f"  [{cfg}] WARN freeze summary parse error: {exc}")
                    gate_ok_all = False
            else:
                print(f"  [{cfg}] WARN freeze_current_summary.json not found")
                gate_ok_all = False

        if gate_ok_all:
            print("  [ALL] live gate passed")
        else:
            print("  [WARN] one or more configs did not pass live gate — orders may be empty")

        # ── STEP 3: Handoff ───────────────────────────────────
        print("\n=== STEP 3: CPAPI HANDOFF (live BBO) ===")
        env3 = {**base_env, **{
            "CONFIG_NAMES":                       "optimal|aggressive",
            "CPAPI_SNAPSHOT_FIELDS":              "31,84,86,88",
            "CPAPI_SNAPSHOT_WAIT_SEC":            "2.0",
            "CPAPI_SNAPSHOT_RETRIES":             "3",
            "CPAPI_INTER_REQ_SEC":                "0.1",
            "MAX_PRICE_DEVIATION_PCT":            "8.0",
            "ALLOW_LAST_FALLBACK":                "1",
            "FORCE_MASSIVE_WHEN_IB_NO_BBO":       "1",
            "FORCE_MASSIVE_WHEN_IB_CLOSE_ONLY":   "1",
            "ENABLE_MASSIVE_FALLBACK":            "1",
            "MASSIVE_TIMEOUT_SEC":                "20.0",
            "TOPK_PRINT":                         "50",
        }}
        _run("CPAPI_HANDOFF", SCRIPT_HANDOFF, env3)

        # ── STEP 4: Execution ─────────────────────────────────
        print("\n=== STEP 4: CPAPI EXECUTION ===")
        env4 = {**base_env, **{
            "CONFIG_NAMES":            "optimal|aggressive",
            "BROKER_NAME":             "MEXEM",
            "BROKER_PLATFORM":         "IBKR_CPAPI",
            "CPAPI_WHOLE_TIMEOUT_SEC": "60.0",
            "CPAPI_FRAC_TIMEOUT_SEC":  "20.0",
            "CPAPI_TIF":               "DAY",
            "CPAPI_FRAC_SLIPPAGE_BPS": "5.0",
            "CPAPI_RESOLVE_CONIDS":    "1",
            "RESET_BROKER_LOG":        "1",
        }}
        _run("CPAPI_EXEC_ORDERS", SCRIPT_EXEC_CPAPI, env4)

        # ── STEP 4b: Cancel working orders (TD-2) ─────────────
        print("\n=== STEP 4b: CANCEL WORKING ORDERS (TD-2) ===")
        cancel_code = """
import os, sys, time
from pathlib import Path
ROOT = Path(".").resolve()
SRC_DIR = ROOT / "src"
for p in [str(ROOT), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)
from python_edge.broker.cpapi_client import CpapiClient
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
base_url   = os.getenv("CPAPI_BASE_URL", "https://localhost:5000")
verify_ssl = os.getenv("CPAPI_VERIFY_SSL", "0").lower() not in {"1","true","yes","on"}
timeout    = float(os.getenv("CPAPI_TIMEOUT_SEC", "10.0"))
account    = os.getenv("BROKER_ACCOUNT_ID", "")
client = CpapiClient(base_url, timeout, verify_ssl)
try:
    resp = client._get(f"/iserver/account/orders?accountId={account}")
    orders = resp if isinstance(resp, list) else (resp.get("orders") or [])
    pe_orders = [o for o in orders if str(o.get("orderId","")).startswith("pe-")]
    print(f"[CANCEL_WORKING] found {len(pe_orders)} pe-* orders")
    cancelled = 0
    for o in pe_orders:
        oid = o.get("orderId")
        status = o.get("status","")
        if status in ("Submitted","PreSubmitted","Working","Filled_partially"):
            try:
                client._delete(f"/iserver/account/{account}/order/{oid}")
                print(f"[CANCEL_WORKING] cancelled {oid} (status={status})")
                cancelled += 1
            except Exception as ex:
                print(f"[CANCEL_WORKING] cancel {oid} error: {ex}")
    print(f"[CANCEL_WORKING] cancelled={cancelled}")
except Exception as exc:
    print(f"[CANCEL_WORKING] fetch failed: {exc}")
"""
        cancel_env = {**base_env, "CONFIG_NAMES": "optimal|aggressive"}
        try:
            _run_inline("CANCEL_WORKING", cancel_code, cancel_env)
        except RuntimeError as exc:
            print(f"[4b] non-fatal: {exc}")
        print("[4b] OK")

        # ── STEP 5: Reconcile post ────────────────────────────
        print("\n=== STEP 5: CPAPI RECONCILE (post-execution) ===")
        _run("CPAPI_RECONCILE_POST", SCRIPT_RECONCILE, env1)

        # ── STEP 5a: Sync cleanup preview ─────────────────────
        print("\n=== STEP 5a: SYNC CLEANUP PREVIEW ===")
        sync_code = """
import pandas as pd
from pathlib import Path
for cfg in ["optimal", "aggressive"]:
    preview = Path(f"artifacts/execution_loop/{cfg}/broker_cleanup_preview.csv")
    orders  = Path(f"artifacts/execution_loop/{cfg}/broker_cleanup_orders.csv")
    if not preview.exists():
        print(f"[SYNC][{cfg}] preview not found -- skip")
        continue
    df = pd.read_csv(preview)
    if "cleanup_side" in df.columns:
        df = df.rename(columns={"cleanup_side": "order_side", "cleanup_qty": "delta_shares"})
    df.to_csv(orders, index=False)
    print(f"[SYNC][{cfg}] rows={len(df)} -> broker_cleanup_orders.csv")
"""
        _run_inline("CLEANUP_SYNC", sync_code, base_env)

        # ── STEP 5b: Resolve conids for cleanup ───────────────
        print("\n=== STEP 5b: RESOLVE CONIDS FOR CLEANUP ===")
        resolve_code = """
import json, sys, os
from pathlib import Path
ROOT = Path(".").resolve()
SRC_DIR = ROOT / "src"
for p in [str(ROOT), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)
from python_edge.broker.cpapi_client import CpapiClient
from python_edge.broker.cpapi_conid_resolver import update_conid_cache
import pandas as pd
base_url   = os.getenv("CPAPI_BASE_URL", "https://localhost:5000")
verify_ssl = os.getenv("CPAPI_VERIFY_SSL", "0").lower() not in {"1","true","yes","on"}
timeout    = float(os.getenv("CPAPI_TIMEOUT_SEC", "30.0"))
exec_root  = Path(os.getenv("EXECUTION_ROOT", "artifacts/execution_loop"))
configs    = [x for x in os.getenv("CONFIG_NAMES","optimal|aggressive").split("|") if x]
client = CpapiClient(base_url, timeout, verify_ssl)
for cfg in configs:
    cleanup_csv = exec_root / cfg / "broker_cleanup_orders.csv"
    cache_path  = exec_root / cfg / "conid_cache.json"
    if not cleanup_csv.exists():
        print(f"[CONID_CLEANUP][{cfg}] no broker_cleanup_orders.csv -- skip")
        continue
    df = pd.read_csv(cleanup_csv)
    if "symbol" not in df.columns or df.empty:
        print(f"[CONID_CLEANUP][{cfg}] empty -- skip")
        continue
    symbols = sorted(df["symbol"].dropna().astype(str).str.strip().str.upper().unique().tolist())
    print(f"[CONID_CLEANUP][{cfg}] resolving {len(symbols)} symbols: {symbols}")
    cache, unresolved = update_conid_cache(cache_path, client, symbols, force_refresh=False)
    if unresolved:
        print(f"[CONID_CLEANUP][{cfg}] UNRESOLVED: {unresolved}", file=sys.stderr)
    print(f"[CONID_CLEANUP][{cfg}] cache_size={len(cache)} unresolved={len(unresolved)}")
"""
        resolve_env = {**base_env, "CONFIG_NAMES": "optimal|aggressive"}
        _run_inline("CONID_CLEANUP_RESOLVE", resolve_code, resolve_env)

        # ── STEP 6: Cleanup send ──────────────────────────────
        print("\n=== STEP 6: CPAPI CLEANUP SEND ===")
        env6 = {**base_env, **{
            "CONFIG_NAMES":                            "optimal|aggressive",
            "BROKER_NAME":                             "MEXEM",
            "BROKER_PLATFORM":                         "IBKR_CPAPI",
            "CPAPI_WHOLE_TIMEOUT_SEC":                 "90.0",
            "CPAPI_FRAC_TIMEOUT_SEC":                  "20.0",
            "CPAPI_TIF":                               "DAY",
            "CPAPI_FRAC_SLIPPAGE_BPS":                 "5.0",
            "CLEANUP_SEND_PREVIEW_ONLY":               "0",
            "CLEANUP_SEND_ONLY_SYMBOLS":               "",
            "CLEANUP_SEND_EXCLUDE_SYMBOLS":            "",
            "CLEANUP_SEND_MAX_ROWS":                   "1000000",
            "CLEANUP_SEND_REQUIRE_EMPTY_PENDING_REFS": "0",
            "CLEANUP_SEND_REQUIRE_SIDE":               "BUY|SELL",
            "CLEANUP_SEND_REASON_TAG":                 "broker_cleanup_send",
            "RESET_BROKER_LOG":                        "0",
            "TOPK_PRINT":                              "50",
        }}
        _run("CPAPI_CLEANUP_SEND", SCRIPT_CLEANUP, env6)

    finally:
        if LOCK_FILE.exists():
            LOCK_FILE.unlink()
            print(f"\n[LOCK] released")

    # ── Summary ───────────────────────────────────────────────
    print("\n=== PIPELINE COMPLETE ===")
    dated_log = f"broker_log_{today_str}.json"
    for cfg in ["optimal", "aggressive"]:
        cfg_dir = ROOT / "artifacts" / "execution_loop" / cfg
        for fname in [
            "orders.csv", "fills.csv",
            "broker_log.json", dated_log,
            "portfolio_state.json", "broker_reconcile_summary.json",
        ]:
            p = cfg_dir / fname
            status = "OK " if p.exists() else "-- "
            print(f"  {status} {cfg}/{fname}")

    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        print("\n[CRASHED]")
        if LOCK_FILE.exists():
            try:
                LOCK_FILE.unlink()
                print("[LOCK] released on crash")
            except Exception:
                pass
    finally:
        print()
        input("Press Enter to exit...")
    sys.exit(rc)
