"""
launch_full_cycle_cpapi.py — повний цикл виконання (CPAPI)

Замінює run_full_cycle_cpapi.ps1. Запускає всі кроки pipeline
напряму через subprocess без PowerShell.

PIPELINE:
  0)  Очищення артефактів execution_loop
  1)  run_cpapi_reconcile.py   — positions -> broker_positions.csv
  2a) run_execution_loop.py    — optimal
  2b) run_execution_loop.py    — aggressive
  2c) run_exit_scanner.py      — price-based exits
  2d) [ДІАГНОСТИКА] freeze gate summary — live_gate_reason per config
  3)  run_cpapi_handoff.py     — live BBO
  4)  run_cpapi_execution.py   — відправка ордерів
  4b) [TD-2] cancel working orders — скасування незакритих ордерів поточного циклу
  5)  run_cpapi_reconcile.py   — post-execution reconcile
  5a) sync cleanup preview -> orders
  5b) resolve conids for cleanup
  6)  run_cpapi_cleanup.py     — cleanup send (TD-3: timeout=90s для великих qty)

Передумова: Client Portal Gateway запущений на https://localhost:5000
            і авторизований через браузер.

Подвійний клік — вікно не закривається до натискання Enter.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
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


def _remove_state_for_alignment() -> None:
    print("[1b] removing portfolio_state.json for fresh alignment")
    for cfg in ["optimal", "aggressive"]:
        cfg_dir = ROOT / "artifacts" / "execution_loop" / cfg
        for fname in ["portfolio_state.json", "state_alignment_once.json"]:
            p = cfg_dir / fname
            if p.exists():
                p.unlink()


# ──────────────────────────────────────────────────────────────
# TD-5: Діагностика freeze gate для кожного конфігу
# ──────────────────────────────────────────────────────────────
def _print_freeze_gate_diagnostics() -> None:
    """
    Читає freeze_current_summary.json для optimal і aggressive
    і виводить live_gate_passed / live_gate_reason / live_active_names.
    Якщо aggressive заблокований — явно попереджає.
    """
    print("\n=== STEP 2d: FREEZE GATE DIAGNOSTICS ===")
    freeze_root = ROOT / "artifacts" / "freeze_runner"
    configs_warned: list[str] = []
    for cfg_name in ["optimal", "aggressive"]:
        summary_path = freeze_root / cfg_name / "freeze_current_summary.json"
        if not summary_path.exists():
            print(f"  [{cfg_name}] freeze_current_summary.json not found — skipping diagnostics")
            continue
        try:
            with summary_path.open("r", encoding="utf-8") as fh:
                s = json.load(fh)
        except Exception as exc:
            print(f"  [{cfg_name}] failed to read freeze summary: {exc}")
            continue

        live_gate_passed   = int(s.get("live_gate_passed", -1))
        live_gate_reason   = str(s.get("live_gate_reason", "unknown"))
        live_active_names  = int(s.get("live_active_names", 0))
        live_active_raw    = int(s.get("live_active_names_raw", 0))
        replay_sharpe      = s.get("replay_sharpe_last_fold", float("nan"))
        replay_cumret      = s.get("replay_cumret_last_fold", float("nan"))
        live_current_date  = str(s.get("live_current_date", ""))
        use_regime         = int(s.get("use_regime_overlay", 0))
        use_budgets        = int(s.get("use_dynamic_side_budgets", 0))

        gate_str = "PASSED" if live_gate_passed == 1 else ("BLOCKED" if live_gate_passed == 0 else "UNKNOWN")
        print(
            f"  [{cfg_name}] gate={gate_str} reason={live_gate_reason} "
            f"live_active_names={live_active_names} live_active_names_raw={live_active_raw} "
            f"replay_sharpe={replay_sharpe:.4f} replay_cumret={replay_cumret:.4f} "
            f"live_current_date={live_current_date} "
            f"use_regime_overlay={use_regime} use_dynamic_side_budgets={use_budgets}"
        )

        if live_gate_passed == 0:
            configs_warned.append(cfg_name)
            print(
                f"  [{cfg_name}] *** LIVE GATE BLOCKED *** — orders.csv will be empty for this config.\n"
                f"             To diagnose: check freeze_basket_search.csv and freeze_daily_replay_last_fold.csv.\n"
                f"             Possible fixes:\n"
                f"               • Знизити FREEZE_MIN_REPLAY_SHARPE (зараз в launch_daily_update.py не встановлено → default={0.00})\n"
                f"               • Знизити FREEZE_MIN_LIVE_ACTIVE_NAMES (default={1})\n"
                f"               • Перевірити basket_search: можливо regime_overlay/side_budgets обнуляють всі ваги"
            )

    if not configs_warned:
        print("  [ALL] live gate passed for all configs")
    else:
        print(f"  [WARN] {len(configs_warned)} config(s) blocked by live gate: {configs_warned}")


# ──────────────────────────────────────────────────────────────
# TD-2: Cancel working orders після execution
# ──────────────────────────────────────────────────────────────
def _cancel_working_orders(base_env: dict) -> None:
    """
    STEP 4b: після виконання ордерів — знаходить всі working/submitted
    ордери з prefix 'pe-' (поточний цикл) через /iserver/account/orders,
    відправляє cancel на кожен, чекає 5с, перевіряє статус.
    Логує debug counters: found / cancelled / still_working.
    Не кидає виняток — тільки друкує попередження.
    """
    print("\n=== STEP 4b: CANCEL WORKING ORDERS (TD-2) ===")
    cancel_code = r"""
import os, sys, time, json
from pathlib import Path
ROOT = Path(".").resolve()
SRC_DIR = ROOT / "src"
for p in [str(ROOT), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from python_edge.broker.cpapi_client import CpapiClient

base_url   = os.getenv("CPAPI_BASE_URL", "https://localhost:5000")
account_id = os.getenv("BROKER_ACCOUNT_ID", "")
verify_ssl = os.getenv("CPAPI_VERIFY_SSL", "0").lower() not in {"1","true","yes","on"}
timeout    = float(os.getenv("CPAPI_TIMEOUT_SEC", "30.0"))

client = CpapiClient(base_url, timeout, verify_ssl)

# Отримати всі ордери брокера
try:
    orders_raw = client._get("/v1/api/iserver/account/orders")
    if isinstance(orders_raw, dict):
        orders_list = orders_raw.get("orders", [])
    elif isinstance(orders_raw, list):
        orders_list = orders_raw
    else:
        orders_list = []
except Exception as exc:
    print(f"[CANCEL_WORKING] failed to fetch orders: {exc}")
    sys.exit(0)

WORKING_STATUSES = {"Submitted", "PreSubmitted", "PendingSubmit",
                    "SUBMITTED", "PRESUBMITTED", "PENDINGSUBMIT"}

# Фільтруємо тільки ордери поточного циклу (cOID починається з 'pe-')
working = [
    o for o in orders_list
    if str(o.get("status", "")).strip() in WORKING_STATUSES
    and str(o.get("orderRef", "") or o.get("cOID", "") or "").startswith("pe-")
]

debug_found         = len(working)
debug_cancel_sent   = 0
debug_cancel_ok     = 0
debug_still_working = 0

print(f"[CANCEL_WORKING] found={debug_found} working orders with prefix 'pe-'")

for order in working:
    order_id = str(order.get("orderId", "") or order.get("order_id", "")).strip()
    symbol   = str(order.get("ticker", "") or order.get("symbol", "")).strip()
    side     = str(order.get("side", "")).strip()
    qty      = order.get("remainingQuantity", order.get("remaining", "?"))
    if not order_id:
        print(f"[CANCEL_WORKING] skipping order without orderId: {order}")
        continue
    try:
        result = client._delete(f"/v1/api/iserver/account/{account_id}/order/{order_id}")
        debug_cancel_sent += 1
        print(f"[CANCEL_WORKING] cancel sent order_id={order_id} symbol={symbol} side={side} remaining={qty} result={result}")
        debug_cancel_ok += 1
    except Exception as exc:
        print(f"[CANCEL_WORKING] cancel failed order_id={order_id} symbol={symbol}: {exc}")

if debug_cancel_sent > 0:
    print(f"[CANCEL_WORKING] waiting 5s for cancel propagation...")
    time.sleep(5)

    # Перевірка: чи залишились ще working
    try:
        orders_raw2 = client._get("/v1/api/iserver/account/orders")
        if isinstance(orders_raw2, dict):
            orders_list2 = orders_raw2.get("orders", [])
        elif isinstance(orders_raw2, list):
            orders_list2 = orders_raw2
        else:
            orders_list2 = []
        still = [
            o for o in orders_list2
            if str(o.get("status", "")).strip() in WORKING_STATUSES
            and str(o.get("orderRef", "") or o.get("cOID", "") or "").startswith("pe-")
        ]
        debug_still_working = len(still)
        if still:
            for o in still:
                print(f"[CANCEL_WORKING][WARN] still working: order_id={o.get('orderId')} symbol={o.get('ticker')} status={o.get('status')} remaining={o.get('remainingQuantity')}")
    except Exception as exc:
        print(f"[CANCEL_WORKING] post-cancel status check failed: {exc}")

print(
    f"[CANCEL_WORKING][SUMMARY] "
    f"found={debug_found} "
    f"cancel_sent={debug_cancel_sent} "
    f"cancel_ok={debug_cancel_ok} "
    f"still_working={debug_still_working}"
)
if debug_still_working > 0:
    print(f"[CANCEL_WORKING][WARN] {debug_still_working} order(s) still working after cancel — reconcile may show partial fills")

sys.exit(0)
"""
    # Виконуємо як inline — не кидаємо помилку якщо cancel не вдався
    print("[4b] cancelling working orders from current cycle...")
    result = subprocess.run(
        [sys.executable, "-c", cancel_code],
        cwd=str(ROOT),
        env=base_env,
    )
    if result.returncode != 0:
        print(f"[4b][WARN] cancel working orders script exited {result.returncode} — continuing pipeline")
    else:
        print("[4b] cancel working orders OK")


def main() -> int:
    print("=" * 60)
    print("  Python-Edge — Full Cycle CPAPI (Execution)")
    print("=" * 60)
    print(f"  ROOT:    {ROOT}")
    print(f"  ACCOUNT: {IB_ACCOUNT_CODE}")
    print(f"  CPAPI:   {CPAPI_BASE_URL}")
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
        "EXECUTION_ROOT":   "artifacts/execution_loop",
        "BROKER_ACCOUNT_ID": IB_ACCOUNT_CODE,
        "CPAPI_BASE_URL":   CPAPI_BASE_URL,
        "CPAPI_VERIFY_SSL": "0",
        "CPAPI_TIMEOUT_SEC": "10.0",
        "MASSIVE_API_KEY":  MASSIVE_API_KEY,
        "MASSIVE_BASE_URL": MASSIVE_BASE_URL,
        "PAUSE_ON_EXIT":    "0",
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
        _remove_state_for_alignment()

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
            "ALIGN_STATE_REQUIRE_BROKER_POSITIONS":     "1",
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

        env2_optimal     = {**env2, "CONFIG_NAMES": "optimal"}
        env2_aggressive  = {**env2, "CONFIG_NAMES": "aggressive"}

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

        # ── STEP 2d: Freeze gate diagnostics (TD-5) ───────────
        _print_freeze_gate_diagnostics()

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
            "CPAPI_WHOLE_TIMEOUT_SEC": "30.0",
            "CPAPI_FRAC_TIMEOUT_SEC":  "20.0",
            "CPAPI_TIF":               "DAY",
            "CPAPI_FRAC_SLIPPAGE_BPS": "5.0",
            "CPAPI_RESOLVE_CONIDS":    "0",
            "RESET_BROKER_LOG":        "1",
        }}
        _run("CPAPI_EXEC_ORDERS", SCRIPT_EXEC_CPAPI, env4)

        # ── STEP 4b: Cancel working orders (TD-2) ─────────────
        _cancel_working_orders(base_env)

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
        # TD-3: CPAPI_WHOLE_TIMEOUT_SEC=90.0 для cleanup (vs 30.0 для execution)
        # Cleanup ордери — це legacy/великі qty (APLS 1269 тощо), потребують більше часу
        print("\n=== STEP 6: CPAPI CLEANUP SEND ===")
        env6 = {**base_env, **{
            "CONFIG_NAMES":                            "optimal|aggressive",
            "BROKER_NAME":                             "MEXEM",
            "BROKER_PLATFORM":                         "IBKR_CPAPI",
            "CPAPI_WHOLE_TIMEOUT_SEC":                 "90.0",   # TD-3: 30→90 для великих cleanup ордерів
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
        # Завжди знімаємо lock
        if LOCK_FILE.exists():
            LOCK_FILE.unlink()
            print(f"\n[LOCK] released")

    # ── Summary ───────────────────────────────────────────────
    print("\n=== PIPELINE COMPLETE ===")
    for cfg in ["optimal", "aggressive"]:
        cfg_dir = ROOT / "artifacts" / "execution_loop" / cfg
        for fname in ["orders.csv", "fills.csv", "broker_log.json",
                      "portfolio_state.json", "broker_reconcile_summary.json"]:
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
        # Знімаємо lock при краші
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
