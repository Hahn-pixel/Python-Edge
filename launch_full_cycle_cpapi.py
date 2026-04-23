"""
launch_full_cycle_cpapi.py — повний цикл виконання (CPAPI)

Замінює run_full_cycle_cpapi.ps1. Запускає всі кроки pipeline
напряму через subprocess без PowerShell.

КОНФІГУРАЦІЇ:
  ACTIVE_CONFIGS = ["aggressive"]        ← тільки paper account (DUP561175)
  ACTIVE_CONFIGS = ["optimal"]           ← тільки real account (після відкриття)
  ACTIVE_CONFIGS = ["optimal", "aggressive"] ← обидва (різні акаунти)

PIPELINE:
  0)  Очищення артефактів execution_loop
  1)  run_cpapi_reconcile.py   — positions -> broker_positions.csv
  1b) умовне видалення portfolio_state.json (тільки якщо немає позицій)
  2a) run_execution_loop.py    — по кожному активному конфігу
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

# Які конфіги торгують зараз.
# Варіант B (paper only): тільки aggressive на DUP561175.
# Коли з'явиться real account — додати "optimal" і прописати його номер нижче.
ACTIVE_CONFIGS: list[str] = ["aggressive"]

# Акаунт per-config. Якщо конфіг відсутній — використовується ACCOUNT_DEFAULT.
ACCOUNT_DEFAULT  = "DUP561175"
ACCOUNT_BY_CONFIG: dict[str, str] = {
    "aggressive": "DUP561175",
    # "optimal": "U1234567",  # ← розкоментувати після відкриття real account
}

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
    for cfg in ACTIVE_CONFIGS:
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
    for cfg in ACTIVE_CONFIGS:
        cfg_dir = ROOT / "artifacts" / "execution_loop" / cfg
        cfg_dir.mkdir(parents=True, exist_ok=True)
        orders = cfg_dir / "orders.csv"
        if not orders.exists():
            orders.write_text(
                "symbol,order_side,delta_shares,target_shares,price,price_source\n",
                encoding="utf-8",
            )
            print(f"  [{cfg}] created empty orders.csv stub")


def _state_has_positions(cfg: str) -> bool:
    """
    Повертає True якщо portfolio_state.json для config містить >=1 позицію
    з ненульовою кількістю акцій.

    Це єдиний надійний критерій для вирішення: видаляти state чи ні.
    broker_positions.csv не підходить — він порожній і на paper account
    без fillів, і на першому запуску, тобто не дозволяє розрізнити ці випадки.
    """
    p = ROOT / "artifacts" / "execution_loop" / cfg / "portfolio_state.json"
    if not p.exists():
        return False
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        positions = data.get("positions", {})
        if not isinstance(positions, dict):
            return False
        for pos in positions.values():
            try:
                shares = float(pos.get("shares", 0.0))
            except Exception:
                shares = 0.0
            if abs(shares) > 1e-9:
                return True
    except Exception:
        pass
    return False


def _conditional_remove_state_for_alignment() -> None:
    """
    STEP 1b — умовне видалення portfolio_state.json.

    Логіка (перевіряємо STATE, не broker_positions):
    - Якщо state має позиції (попередній цикл вже будував портфель):
        → НЕ видаляємо state → STEP 2 порівняє state з цільовими вагами
          і згенерує тільки delta-ордери (вхід/вихід/drift).
        → Видаляємо тільки маркер state_alignment_once.json щоб alignment
          міг пройти заново (broker може мати нові позиції після fills).
    - Якщо state порожній (справжній перший запуск або RESET_STATE):
        → Видаляємо state і маркер → STEP 2 зробить fresh alignment
          з broker_positions (порожній на paper → стан 0),
          після чого відразу згенерує ордери на вхід (ціль > 0).

    Стара логіка (перевіряти broker_positions) була хибною: broker_positions
    порожній і на першому запуску, і після кожного щоденного запуску до fills,
    тому завжди видаляла state → bootstrap_only=1 → нуль ордерів.
    """
    print("[1b] conditional state removal for alignment")
    for cfg in ACTIVE_CONFIGS:
        has_positions = _state_has_positions(cfg)
        cfg_dir = ROOT / "artifacts" / "execution_loop" / cfg
        if has_positions:
            # State має позиції → зберігаємо, видаляємо тільки маркер alignment
            marker = cfg_dir / "state_alignment_once.json"
            if marker.exists():
                marker.unlink()
                print(f"  [{cfg}] state has positions → keeping state, removed alignment marker")
            else:
                print(f"  [{cfg}] state has positions → keeping state (marker already absent)")
        else:
            # State порожній → справжній перший запуск → видаляємо все
            for fname in ["portfolio_state.json", "state_alignment_once.json"]:
                p = cfg_dir / fname
                if p.exists():
                    p.unlink()
                    print(f"  [{cfg}] state empty → removed {fname}")
                else:
                    print(f"  [{cfg}] state empty → {fname} already absent")


def main() -> int:
    today_str = date.today().isoformat()

    print("=" * 60)
    print("  Python-Edge — Full Cycle CPAPI (Execution)")
    print("=" * 60)
    print(f"  ROOT:    {ROOT}")
    print(f"  CONFIGS: {ACTIVE_CONFIGS}")
    for cfg in ACTIVE_CONFIGS:
        acct = ACCOUNT_BY_CONFIG.get(cfg, ACCOUNT_DEFAULT)
        print(f"  ACCOUNT[{cfg}]: {acct}")
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

    # Базовий env — спільний для всіх кроків (без BROKER_ACCOUNT_ID — він per-config)
    base_env = os.environ.copy()
    base_env.update({
        "EXECUTION_ROOT":    "artifacts/execution_loop",
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

        configs_pipe = "|".join(ACTIVE_CONFIGS)
        # Для reconcile використовуємо акаунт першого конфігу (або default).
        # Reconcile читає позиції з брокера — якщо конфіги на одному акаунті,
        # достатньо одного виклику; якщо на різних — потрібні окремі виклики.
        # Зараз ACTIVE_CONFIGS=[aggressive] → один акаунт → один виклик.
        reconcile_account = ACCOUNT_BY_CONFIG.get(ACTIVE_CONFIGS[0], ACCOUNT_DEFAULT)
        env1 = {**base_env, **{
            "CONFIG_NAMES":                        configs_pipe,
            "BROKER_ACCOUNT_ID":                   reconcile_account,
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

        # ── STEP 2: Execution loop ────────────────────────────
        print("\n=== STEP 2: EXECUTION LOOP ===")
        env2_base = {**base_env, **{
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

        for idx, cfg in enumerate(ACTIVE_CONFIGS):
            label = f"[{idx + 1}/{len(ACTIVE_CONFIGS)}]"
            print(f"{label} {cfg}")
            env2_cfg = {
                **env2_base,
                "CONFIG_NAMES":    cfg,
                "BROKER_ACCOUNT_ID": ACCOUNT_BY_CONFIG.get(cfg, ACCOUNT_DEFAULT),
            }
            _run(f"EXEC_LOOP_{cfg.upper()}", SCRIPT_EXEC, env2_cfg)

        # ── STEP 2c: Exit scanner ─────────────────────────────
        print("\n=== STEP 2c: EXIT SCANNER ===")
        env2c = {**base_env, **{
            "CONFIG_NAMES":                   configs_pipe,
            "BROKER_ACCOUNT_ID":              reconcile_account,
            "LIVE_FEATURE_SNAPSHOT_FILE":     "artifacts/live_alpha/live_feature_snapshot.parquet",
            "EXIT_SCANNER_DRY_RUN":           "0",
            "EXIT_SCANNER_USE_STATE_PRICE":   "0",
        }}
        _run("EXIT_SCANNER", SCRIPT_EXIT_SCANNER, env2c)

        # ── STEP 2d: Freeze gate diagnostics ──────────────────
        print("\n=== STEP 2d: FREEZE GATE DIAGNOSTICS ===")
        gate_ok_all = True
        for cfg in ACTIVE_CONFIGS:
            freeze_summary = (
                ROOT / "artifacts" / "freeze_runner" / cfg
                / "freeze_current_summary.json"
            )
            if freeze_summary.exists():
                try:
                    data = json.loads(freeze_summary.read_text(encoding="utf-8"))
                    # live_gate_status — рядок ("PASSED"/"FAILED"), або відсутній.
                    # Fallback: live_gate_passed (int 0/1) який завжди є у freeze runner.
                    gate_status = data.get("live_gate_status", "")
                    if not gate_status:
                        gate_passed_int = int(data.get("live_gate_passed", 0))
                        gate_status = "PASSED" if gate_passed_int else "FAILED"
                    gate    = gate_status
                    reason  = data.get("live_gate_reason", "n/a")
                    active  = data.get("live_active_names", "?")
                    raw_n   = data.get("live_active_names_raw", active)
                    sharpe  = data.get("live_sharpe",
                              data.get("replay_sharpe_last_fold", "?"))
                    cumret  = data.get("live_cumret",
                              data.get("replay_cumret_last_fold", "?"))
                    snap_dt = data.get("live_snapshot_current_date",
                              data.get("live_current_date", "?"))
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
            "CONFIG_NAMES":                       configs_pipe,
            "BROKER_ACCOUNT_ID":                  reconcile_account,
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
            "CONFIG_NAMES":            configs_pipe,
            "BROKER_ACCOUNT_ID":       reconcile_account,
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
        cancel_env = {**base_env, "CONFIG_NAMES": configs_pipe, "BROKER_ACCOUNT_ID": reconcile_account}
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
        active_configs_repr = repr(ACTIVE_CONFIGS)
        sync_code = f"""
import pandas as pd
from pathlib import Path
for cfg in {active_configs_repr}:
    preview = Path(f"artifacts/execution_loop/{{cfg}}/broker_cleanup_preview.csv")
    orders  = Path(f"artifacts/execution_loop/{{cfg}}/broker_cleanup_orders.csv")
    if not preview.exists():
        print(f"[SYNC][{{cfg}}] preview not found -- skip")
        continue
    df = pd.read_csv(preview)
    if "cleanup_side" in df.columns:
        df = df.rename(columns={{"cleanup_side": "order_side", "cleanup_qty": "delta_shares"}})
    df.to_csv(orders, index=False)
    print(f"[SYNC][{{cfg}}] rows={{len(df)}} -> broker_cleanup_orders.csv")
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
configs    = [x for x in os.getenv("CONFIG_NAMES","aggressive").split("|") if x]
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
        resolve_env = {**base_env, "CONFIG_NAMES": configs_pipe, "BROKER_ACCOUNT_ID": reconcile_account}
        _run_inline("CONID_CLEANUP_RESOLVE", resolve_code, resolve_env)

        # ── STEP 6: Cleanup send ──────────────────────────────
        print("\n=== STEP 6: CPAPI CLEANUP SEND ===")
        env6 = {**base_env, **{
            "CONFIG_NAMES":                            configs_pipe,
            "BROKER_ACCOUNT_ID":                       reconcile_account,
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
    for cfg in ACTIVE_CONFIGS:
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
