"""
launch_full_cycle_cpapi.py — повний цикл виконання (CPAPI)

Замінює run_full_cycle_cpapi.ps1. Запускає всі кроки pipeline
напряму через subprocess без PowerShell.

PIPELINE:
  0)  Очищення артефактів execution_loop
      (orders.csv НЕ видаляється — він вже згенерований daily update)
  1)  run_cpapi_reconcile.py   — positions -> broker_positions.csv
  2)  run_execution_loop.py    — тільки якщо orders.csv порожній або відсутній
  2c) run_exit_scanner.py      — price-based exits
  2d) freeze gate diagnostics
  3)  run_cpapi_handoff.py     — live BBO
  4)  run_cpapi_execution.py   — відправка ордерів
  4b) cancel working orders (TD-2)
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
from pathlib import Path

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ── Налаштування ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent

# Конфіги для виконання (можна змінити на "optimal|aggressive")
CONFIGS = ["aggressive"]

IB_ACCOUNT_MAP = {
    "optimal":    "DUP561175",
    "aggressive": "DUP561175",
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


def _clean_artifacts(configs: list[str]) -> None:
    """
    STEP 0: очищення broker-артефактів.

    orders.csv НЕ видаляється — він вже згенерований launch_daily_update.py.
    Якщо orders.csv відсутній або порожній, STEP 2 перегенерує його.
    """
    print("\n=== STEP 0: CLEAN ARTIFACTS ===")
    # orders.csv навмисно відсутній у цьому списку
    to_delete = [
        "target_book.csv",
        "execution_log.csv", "positions_mark_to_market.csv",
        "broker_handoff_summary.json", "broker_positions.csv",
        "broker_open_orders.csv", "broker_pending.csv",
        "broker_reconcile.csv", "broker_reconcile_summary.json",
        "broker_cleanup_preview.csv", "broker_cleanup_orders.csv",
        "broker_cleanup_emit_summary.json", "broker_cleanup_send_plan.csv",
        "broker_cleanup_send_summary.json", "orders_exec_backup.csv",
        "orders_pre_cleanup_backup.csv", "state_alignment_once.json",
    ]
    for cfg in configs:
        cfg_dir = ROOT / "artifacts" / "execution_loop" / cfg
        if not cfg_dir.exists():
            continue
        for f in to_delete:
            p = cfg_dir / f
            if p.exists():
                p.unlink()
                print(f"  [{cfg}] deleted {f}")
        # Перевіряємо orders.csv
        orders_path = cfg_dir / "orders.csv"
        if orders_path.exists():
            # Рахуємо живі рядки (не враховуючи header)
            try:
                lines = [l for l in orders_path.read_text(encoding="utf-8").splitlines() if l.strip()]
                live_rows = max(0, len(lines) - 1)
            except Exception:
                live_rows = 0
            print(f"  [{cfg}] orders.csv preserved ({live_rows} order rows)")
        else:
            print(f"  [{cfg}] orders.csv absent — will be generated in STEP 2")
        print(f"  [{cfg}] clean OK (portfolio_state.json + orders.csv preserved)")


def _create_orders_stubs(configs: list[str]) -> None:
    for cfg in configs:
        cfg_dir = ROOT / "artifacts" / "execution_loop" / cfg
        cfg_dir.mkdir(parents=True, exist_ok=True)
        orders = cfg_dir / "orders.csv"
        if not orders.exists():
            orders.write_text(
                "symbol,order_side,delta_shares,target_shares,price,price_source\n",
                encoding="utf-8",
            )
            print(f"  [{cfg}] created empty orders.csv stub")


def _orders_have_live_rows(cfg: str) -> bool:
    """True якщо orders.csv існує і містить хоча б один не-header рядок."""
    orders_path = ROOT / "artifacts" / "execution_loop" / cfg / "orders.csv"
    if not orders_path.exists():
        return False
    try:
        lines = [l for l in orders_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        return len(lines) > 1
    except Exception:
        return False


def _conditional_state_removal(configs: list[str]) -> None:
    """
    Після STEP 1 (reconcile): видаляємо portfolio_state.json тільки якщо
    він порожній (нема позицій). Якщо є позиції — зберігаємо.
    """
    print("[1b] conditional state removal for alignment")
    for cfg in configs:
        cfg_dir = ROOT / "artifacts" / "execution_loop" / cfg
        state_path = cfg_dir / "portfolio_state.json"
        marker_path = cfg_dir / "state_alignment_once.json"
        if not state_path.exists():
            print(f"  [{cfg}] no state — skip")
            continue
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            positions = state.get("positions", {})
            has_positions = bool(positions)
        except Exception:
            has_positions = False
        if has_positions:
            print(f"  [{cfg}] state has positions → keeping state (marker already absent)")
        else:
            print(f"  [{cfg}] state empty → removing for fresh alignment")
            state_path.unlink(missing_ok=True)
        marker_path.unlink(missing_ok=True)


def main() -> int:
    configs_str = "|".join(CONFIGS)

    print("=" * 60)
    print("  Python-Edge — Full Cycle CPAPI (Execution)")
    print("=" * 60)
    print(f"  ROOT:    {ROOT}")
    print(f"  CONFIGS: {CONFIGS}")
    for cfg in CONFIGS:
        print(f"  ACCOUNT[{cfg}]: {IB_ACCOUNT_MAP.get(cfg, 'UNKNOWN')}")
    print(f"  CPAPI:   {CPAPI_BASE_URL}")
    import datetime
    print(f"  DATE:    {datetime.date.today()}")
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
        "BROKER_ACCOUNT_ID": IB_ACCOUNT_MAP.get(CONFIGS[0], IB_ACCOUNT_MAP.get("aggressive", "")),
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
        # orders.csv НЕ видаляється — він вже готовий від daily update
        _clean_artifacts(CONFIGS)

        # ── STEP 1: Reconcile pre-alignment ───────────────────
        print("\n=== STEP 1: CPAPI RECONCILE (pre-alignment) ===")
        _create_orders_stubs(CONFIGS)

        env1 = {**base_env, **{
            "CONFIG_NAMES":                        configs_str,
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
        _conditional_state_removal(CONFIGS)

        # ── STEP 2: Execution loop — тільки якщо orders порожній ──
        print("\n=== STEP 2: EXECUTION LOOP ===")

        # Env для exec loop:
        # ALIGN_STATE_FROM_BROKER_ONCE=0 — не обнуляти стан з порожнього broker_positions
        # ALIGN_STATE_REQUIRE_BROKER_POSITIONS=0 — не вимагати позицій від брокера
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
            # Ключова зміна: не робити alignment з broker_positions
            "ALIGN_STATE_FROM_BROKER_ONCE":             "0",
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

        # Перевіряємо кожен конфіг окремо
        for i, cfg in enumerate(CONFIGS, 1):
            label = f"EXEC_LOOP_{cfg.upper()}"
            if _orders_have_live_rows(cfg):
                print(f"[{i}/{len(CONFIGS)}] {cfg} — orders.csv вже містить ордери → пропускаємо STEP 2")
                print(f"[{label}] SKIPPED (orders already present)")
            else:
                print(f"[{i}/{len(CONFIGS)}] {cfg} — orders.csv порожній → генеруємо")
                env2_cfg = {**env2, "CONFIG_NAMES": cfg}
                _run(label, SCRIPT_EXEC, env2_cfg)

        # ── STEP 2c: Exit scanner ─────────────────────────────
        print("\n=== STEP 2c: EXIT SCANNER ===")
        env2c = {**base_env, **{
            "CONFIG_NAMES":                   configs_str,
            "LIVE_FEATURE_SNAPSHOT_FILE":     "artifacts/live_alpha/live_feature_snapshot.parquet",
            "EXIT_SCANNER_DRY_RUN":           "0",
            # Використовуємо state price (Gateway SSL може бути недоступний)
            "EXIT_SCANNER_USE_STATE_PRICE":   "1",
        }}
        _run("EXIT_SCANNER", SCRIPT_EXIT_SCANNER, env2c)

        # ── STEP 2d: Freeze gate diagnostics ─────────────────
        print("\n=== STEP 2d: FREEZE GATE DIAGNOSTICS ===")
        gate_code = """
import json, os
from pathlib import Path
freeze_root = Path(os.getenv("FREEZE_ROOT", "artifacts/freeze_runner"))
configs = [x for x in os.getenv("CONFIG_NAMES", "optimal|aggressive").split("|") if x]
all_passed = True
for cfg in configs:
    p = freeze_root / cfg / "freeze_current_summary.json"
    if not p.exists():
        print(f"  [{cfg}] freeze summary not found — skip")
        continue
    s = json.loads(p.read_text(encoding="utf-8"))
    passed = s.get("live_gate_passed", 0)
    reason = s.get("live_gate_reason", "unknown")
    active = s.get("live_active_names", 0)
    active_raw = s.get("live_active_names_raw", active)
    sharpe = s.get("replay_sharpe_last_fold", 0.0)
    cumret = s.get("replay_cumret_last_fold", 0.0)
    date   = s.get("live_current_date", "?")
    gate_str = "PASSED" if passed else "BLOCKED"
    print(f"  [{cfg}] gate={gate_str} reason={reason} live_active={active}(raw={active_raw}) sharpe={sharpe} cumret={cumret} date={date}")
    if not passed:
        all_passed = False
if all_passed:
    print("  [ALL] live gate passed")
else:
    print("  [WARN] some configs blocked — check freeze summaries")
"""
        gate_env = {**base_env,
                    "CONFIG_NAMES": configs_str,
                    "FREEZE_ROOT":  "artifacts/freeze_runner"}
        _run_inline("FREEZE_GATE_DIAG", gate_code, gate_env)

        # ── STEP 3: Handoff ───────────────────────────────────
        print("\n=== STEP 3: CPAPI HANDOFF (live BBO) ===")
        env3 = {**base_env, **{
            "CONFIG_NAMES":                       configs_str,
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
            "CONFIG_NAMES":            configs_str,
            "BROKER_NAME":             "MEXEM",
            "BROKER_PLATFORM":         "IBKR_CPAPI",
            "CPAPI_WHOLE_TIMEOUT_SEC": "60.0",
            "CPAPI_FRAC_TIMEOUT_SEC":  "20.0",
            "CPAPI_TIF":               "DAY",
            "CPAPI_FRAC_SLIPPAGE_BPS": "5.0",
            "CPAPI_RESOLVE_CONIDS":    "0",
            "RESET_BROKER_LOG":        "1",
        }}
        _run("CPAPI_EXEC_ORDERS", SCRIPT_EXEC_CPAPI, env4)

        # ── STEP 4b: Cancel working orders (TD-2) ────────────
        print("\n=== STEP 4b: CANCEL WORKING ORDERS (TD-2) ===")
        cancel_code = """
import os, requests, time
urllib3_imported = False
try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    urllib3_imported = True
except ImportError:
    pass
base_url = os.getenv("CPAPI_BASE_URL", "https://localhost:5000")
account  = os.getenv("BROKER_ACCOUNT_ID", "")
s = requests.Session()
s.verify = False
cancelled = 0
errors = 0
try:
    resp = s.get(f"{base_url}/v1/api/iserver/account/orders?accountId={account}",
                 timeout=(15, 30))
    orders = resp.json()
    if isinstance(orders, dict):
        orders = orders.get("orders", [])
    working = [o for o in (orders or [])
               if isinstance(o, dict)
               and str(o.get("status","")).upper() in {"SUBMITTED","PRESUBMITTED","WORKING"}
               and str(o.get("orderId","")).startswith("pe-")]
    print(f"[CANCEL_WORKING] working pe-* orders found: {len(working)}")
    for o in working:
        oid = o.get("orderId") or o.get("order_id")
        try:
            r = s.delete(f"{base_url}/v1/api/iserver/account/{account}/order/{oid}",
                         timeout=(15, 30))
            print(f"[CANCEL_WORKING] cancel {oid} status={r.status_code}")
            cancelled += 1
            time.sleep(0.3)
        except Exception as e:
            print(f"[CANCEL_WORKING] cancel {oid} error: {e}")
            errors += 1
except Exception as e:
    print(f"[CANCEL_WORKING] fetch failed: {e}")
    errors += 1
print(f"[CANCEL_WORKING] done cancelled={cancelled} errors={errors}")
"""
        cancel_env = {**base_env}
        _run_inline("CANCEL_WORKING", cancel_code, cancel_env)
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
        resolve_env = {**base_env, "CONFIG_NAMES": configs_str}
        _run_inline("CONID_CLEANUP_RESOLVE", resolve_code, resolve_env)

        # ── STEP 6: Cleanup send ──────────────────────────────
        print("\n=== STEP 6: CPAPI CLEANUP SEND ===")
        env6 = {**base_env, **{
            "CONFIG_NAMES":                            configs_str,
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
        # Завжди знімаємо lock
        if LOCK_FILE.exists():
            LOCK_FILE.unlink()
            print(f"\n[LOCK] released")

    # ── Summary ───────────────────────────────────────────────
    print("\n=== PIPELINE COMPLETE ===")
    for cfg in CONFIGS:
        cfg_dir = ROOT / "artifacts" / "execution_loop" / cfg
        broker_log_date = f"broker_log_{__import__('datetime').date.today()}.json"
        for fname in ["orders.csv", "fills.csv", "broker_log.json",
                      broker_log_date, "portfolio_state.json",
                      "broker_reconcile_summary.json"]:
            p = cfg_dir / fname
            status = "OK " if p.exists() else "-- "
            print(f"  {status} {cfg}/{fname}")
    print()
    print()

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
