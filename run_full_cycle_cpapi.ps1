Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ============================================================
# Python-Edge — повний цикл (повністю CPAPI, без ibapi)
#
# PIPELINE:
#   0)  Очищення артефактів execution_loop
#   1)  run_cpapi_reconcile.py   — positions → broker_positions.csv
#   2)  run_execution_loop.py    — alignment + order generation
#   2b) run_exit_scanner.py      — price-based exits: stop/tp/trailing/max_hold
#   3)  run_cpapi_handoff.py     — live BBO через CPAPI snapshot
#   4)  run_cpapi_execution.py   — відправка ордерів через CPAPI
#   5)  run_cpapi_reconcile.py   — post-execution reconcile
#   5а) Синхронізація preview → orders
#   6)  run_cpapi_cleanup.py     — cleanup send через CPAPI
#
# Передумова: Client Portal Gateway запущений на https://localhost:5000
#             і авторизований через браузер
# ============================================================

$ROOT   = "C:\Users\Dmytro Govor\Documents\Python-Edge"
$PYTHON = "python"

$SCRIPT_EXEC         = "$ROOT\scripts\run_execution_loop.py"
$SCRIPT_RECONCILE    = "$ROOT\scripts\run_cpapi_reconcile.py"
$SCRIPT_HANDOFF      = "$ROOT\scripts\run_cpapi_handoff.py"
$SCRIPT_EXEC_CPAPI   = "$ROOT\scripts\run_cpapi_execution.py"
$SCRIPT_CLEANUP      = "$ROOT\scripts\run_cpapi_cleanup.py"
$SCRIPT_EXIT_SCANNER = "$ROOT\scripts\run_exit_scanner.py"

# ── Налаштування ──────────────────────────────────────────────
$IB_ACCOUNT_CODE  = ""             # ваш акаунт IBKR (обов'язково)
$NAV_OPTIMAL      = "55204.032916"
$NAV_AGGRESSIVE   = "60986.205462"
$MASSIVE_API_KEY  = $env:MASSIVE_API_KEY
$MASSIVE_BASE_URL = "https://api.massive.com"
$CPAPI_BASE_URL   = "https://localhost:5000"

# ── Exit policy параметри (per-config) ───────────────────────
# Залиште порожнім щоб використовувати дефолти з exit_policy.py
$EXIT_OPTIMAL_STOP_LOSS_PCT     = ""   # default: 0.08
$EXIT_OPTIMAL_TAKE_PROFIT_PCT   = ""   # default: 0.25
$EXIT_OPTIMAL_TRAILING_STOP_PCT = ""   # default: 0.12
$EXIT_OPTIMAL_MAX_HOLD_DAYS     = ""   # default: 30

$EXIT_AGGRESSIVE_STOP_LOSS_PCT     = ""   # default: 0.10
$EXIT_AGGRESSIVE_TAKE_PROFIT_PCT   = ""   # default: 0.30
$EXIT_AGGRESSIVE_TRAILING_STOP_PCT = ""   # default: 0.15
$EXIT_AGGRESSIVE_MAX_HOLD_DAYS     = ""   # default: 20

# ── Допоміжна функція ─────────────────────────────────────────
function Assert-Exit {
    param([string]$Label)
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[$Label] FAILED exit=$LASTEXITCODE" -ForegroundColor Red
        throw "Pipeline aborted: $Label"
    }
    Write-Host "[$Label] OK" -ForegroundColor Green
}

Set-Location $ROOT

# ============================================================
# КРОК 0 — Очищення артефактів execution_loop
# ============================================================
Write-Host ""
Write-Host "=== STEP 0: CLEAN ARTIFACTS ===" -ForegroundColor Cyan

foreach ($cfg in @("optimal", "aggressive")) {
    $dir = "$ROOT\artifacts\execution_loop\$cfg"
    if (-not (Test-Path $dir)) { continue }

    $toDelete = @(
        "orders.csv", "target_book.csv", "broker_log.json", "fills.csv",
        "execution_log.csv", "positions_mark_to_market.csv",
        "broker_handoff_summary.json", "broker_positions.csv",
        "broker_open_orders.csv", "broker_pending.csv",
        "broker_reconcile.csv", "broker_reconcile_summary.json",
        "broker_cleanup_preview.csv", "broker_cleanup_orders.csv",
        "broker_cleanup_emit_summary.json", "broker_cleanup_send_plan.csv",
        "broker_cleanup_send_summary.json", "orders_exec_backup.csv",
        "orders_pre_cleanup_backup.csv", "state_alignment_once.json"
    )
    foreach ($f in $toDelete) {
        $path = "$dir\$f"
        if (Test-Path $path) {
            Remove-Item $path -Force
            Write-Host "  [$cfg] deleted $f" -ForegroundColor DarkGray
        }
    }
    Write-Host "  [$cfg] clean OK (portfolio_state.json preserved)" -ForegroundColor Green
}

# ============================================================
# PRE-FLIGHT CHECK
# ============================================================
Write-Host ""
Write-Host "=== PRE-FLIGHT CHECK ===" -ForegroundColor Cyan
@(
    $SCRIPT_EXEC, $SCRIPT_RECONCILE, $SCRIPT_HANDOFF, $SCRIPT_EXEC_CPAPI, $SCRIPT_CLEANUP,
    $SCRIPT_EXIT_SCANNER,
    "$ROOT\artifacts\live_alpha\live_feature_snapshot.parquet",
    "$ROOT\artifacts\freeze_runner\optimal\freeze_current_book.csv",
    "$ROOT\artifacts\freeze_runner\optimal\freeze_current_summary.json",
    "$ROOT\artifacts\freeze_runner\aggressive\freeze_current_book.csv",
    "$ROOT\artifacts\freeze_runner\aggressive\freeze_current_summary.json",
    "$ROOT\artifacts\execution_loop\optimal\portfolio_state.json",
    "$ROOT\artifacts\execution_loop\aggressive\portfolio_state.json"
) | ForEach-Object {
    if (-not (Test-Path $_)) {
        Write-Host "  MISSING: $_" -ForegroundColor Yellow
    } else {
        Write-Host "  OK: $_" -ForegroundColor Green
    }
}

# ============================================================
# КРОК 1 — CPAPI reconcile (pre-alignment)
# ============================================================
Write-Host ""
Write-Host "=== STEP 1: CPAPI RECONCILE (pre-alignment) ===" -ForegroundColor Cyan

foreach ($cfg in @("optimal", "aggressive")) {
    $ordersPath = "$ROOT\artifacts\execution_loop\$cfg\orders.csv"
    $dirPath    = "$ROOT\artifacts\execution_loop\$cfg"
    if (-not (Test-Path $dirPath)) { New-Item -ItemType Directory -Path $dirPath -Force | Out-Null }
    if (-not (Test-Path $ordersPath)) {
        "symbol,order_side,delta_shares,target_shares,price,price_source" | Out-File $ordersPath -Encoding utf8
        Write-Host "  [$cfg] created empty orders.csv stub" -ForegroundColor DarkGray
    }
}

$env:EXECUTION_ROOT                      = "artifacts/execution_loop"
$env:CONFIG_NAMES                        = "optimal|aggressive"
$env:BROKER_ACCOUNT_ID                   = $IB_ACCOUNT_CODE
$env:CPAPI_BASE_URL                      = $CPAPI_BASE_URL
$env:CPAPI_VERIFY_SSL                    = "0"
$env:CPAPI_TIMEOUT_SEC                   = "10.0"
$env:DRIFT_TOLERANCE_SHARES              = "0.000001"
$env:PREFER_STATE_OVER_ORDERS            = "1"
$env:REQUIRE_BROKER_REFRESH              = "1"
$env:ALLOW_EXISTING_BROKER_POSITIONS_CSV = "1"
$env:CLEANUP_PREVIEW_MODE                = "1"
$env:CLEANUP_INCLUDE_UNEXPECTED          = "1"
$env:CLEANUP_INCLUDE_DRIFT               = "1"
$env:CLEANUP_ONLY_SYMBOLS                = ""
$env:CLEANUP_EXCLUDE_SYMBOLS             = ""
$env:CLEANUP_MIN_ABS_SHARES              = "1.0"
$env:TOPK_PRINT                          = "50"
$env:PAUSE_ON_EXIT                       = "0"

& $PYTHON $SCRIPT_RECONCILE
Assert-Exit "CPAPI_RECONCILE_PRE"

Write-Host "[1b] removing portfolio_state.json for fresh alignment" -ForegroundColor DarkGray
foreach ($cfg in @("optimal", "aggressive")) {
    $statePath = "$ROOT\artifacts\execution_loop\$cfg\portfolio_state.json"
    $alignPath = "$ROOT\artifacts\execution_loop\$cfg\state_alignment_once.json"
    if (Test-Path $statePath) { Remove-Item $statePath -Force }
    if (Test-Path $alignPath) { Remove-Item $alignPath -Force }
}

# ============================================================
# КРОК 2 — Execution loop (alignment + order generation)
# ============================================================
Write-Host ""
Write-Host "=== STEP 2: EXECUTION LOOP ===" -ForegroundColor Cyan

Write-Host "[2a] optimal" -ForegroundColor White
$env:LIVE_FEATURE_SNAPSHOT_FILE               = "artifacts/live_alpha/live_feature_snapshot.parquet"
$env:FREEZE_ROOT                              = "artifacts/freeze_runner"
$env:EXECUTION_ROOT                           = "artifacts/execution_loop"
$env:ACCOUNT_NAV                              = $NAV_OPTIMAL
$env:CONFIG_NAMES                             = "optimal"
$env:ALLOW_FRACTIONAL_SHARES                  = "0"
$env:FRACTIONAL_MODE                          = "integer"
$env:MIN_ORDER_NOTIONAL                       = "25.0"
$env:DEFAULT_PRICE_FALLBACK                   = "100.0"
$env:RESET_STATE                              = "0"
$env:SKIP_EMPTY_FREEZE_CONFIGS                = "1"
$env:REQUIRE_FREEZE_DATE_MATCH_LIVE_PRICES    = "1"
$env:MAX_SINGLE_NAME_WEIGHT                   = "0.05"
$env:MAX_SINGLE_NAME_NOTIONAL                 = "10000.0"
$env:MIN_PRICE_TO_TRADE                       = "1.0"
$env:MAX_PRICE_TO_TRADE                       = "1000000.0"
$env:COMMISSION_BPS                           = "0.50"
$env:COMMISSION_MIN_PER_ORDER                 = "0.35"
$env:SLIPPAGE_BPS                             = "1.50"
$env:ALIGN_STATE_FROM_BROKER_ONCE             = "1"
$env:ALIGN_STATE_REQUIRE_BROKER_POSITIONS     = "1"
$env:ALIGN_STATE_CASH_MODE                    = "preserve_nav"
$env:SKIP_LEGACY_SYMBOLS_NOT_IN_TARGET        = "1"
$env:USE_STATE_PRICE_FALLBACK_FOR_LEGACY      = "1"
$env:USE_STATE_PRICE_FALLBACK_FOR_MTM         = "1"
$env:USE_BROKER_POSITIONS_PRICE_FALLBACK_FOR_LEGACY = "1"
$env:USE_BROKER_POSITIONS_PRICE_FALLBACK_FOR_MTM    = "1"
$env:SANITIZE_STATE_PRICES                    = "1"
$env:PERSIST_DEFAULT_FALLBACK_TO_STATE        = "0"
$env:PAUSE_ON_EXIT                            = "0"
$env:TOPK_PRINT                               = "50"

& $PYTHON $SCRIPT_EXEC
Assert-Exit "EXEC_LOOP_OPTIMAL"

Write-Host "[2b-exec] aggressive" -ForegroundColor White
$env:ACCOUNT_NAV  = $NAV_AGGRESSIVE
$env:CONFIG_NAMES = "aggressive"
& $PYTHON $SCRIPT_EXEC
Assert-Exit "EXEC_LOOP_AGGRESSIVE"

# ============================================================
# КРОК 2b — Exit Scanner (price-based exits)
# Виконується після order generation, до handoff/execution.
# Додає SELL/BUY ордери закриття в orders.csv для позицій що
# спрацювали на stop_loss / take_profit / trailing_stop / max_hold.
# ============================================================
Write-Host ""
Write-Host "=== STEP 2b: EXIT SCANNER ===" -ForegroundColor Cyan

$env:EXECUTION_ROOT                  = "artifacts/execution_loop"
$env:CONFIG_NAMES                    = "optimal|aggressive"
$env:CPAPI_BASE_URL                  = $CPAPI_BASE_URL
$env:CPAPI_VERIFY_SSL                = "0"
$env:CPAPI_TIMEOUT_SEC               = "10.0"
$env:EXIT_SCANNER_DRY_RUN            = "0"
$env:EXIT_SCANNER_USE_STATE_PRICE    = "0"   # 1 = не ходити в CPAPI
$env:PAUSE_ON_EXIT                   = "0"

# Exit policy: optimal
if ($EXIT_OPTIMAL_STOP_LOSS_PCT     -ne "") { $env:EXIT_OPTIMAL_STOP_LOSS_PCT     = $EXIT_OPTIMAL_STOP_LOSS_PCT }
if ($EXIT_OPTIMAL_TAKE_PROFIT_PCT   -ne "") { $env:EXIT_OPTIMAL_TAKE_PROFIT_PCT   = $EXIT_OPTIMAL_TAKE_PROFIT_PCT }
if ($EXIT_OPTIMAL_TRAILING_STOP_PCT -ne "") { $env:EXIT_OPTIMAL_TRAILING_STOP_PCT = $EXIT_OPTIMAL_TRAILING_STOP_PCT }
if ($EXIT_OPTIMAL_MAX_HOLD_DAYS     -ne "") { $env:EXIT_OPTIMAL_MAX_HOLD_DAYS     = $EXIT_OPTIMAL_MAX_HOLD_DAYS }

# Exit policy: aggressive
if ($EXIT_AGGRESSIVE_STOP_LOSS_PCT     -ne "") { $env:EXIT_AGGRESSIVE_STOP_LOSS_PCT     = $EXIT_AGGRESSIVE_STOP_LOSS_PCT }
if ($EXIT_AGGRESSIVE_TAKE_PROFIT_PCT   -ne "") { $env:EXIT_AGGRESSIVE_TAKE_PROFIT_PCT   = $EXIT_AGGRESSIVE_TAKE_PROFIT_PCT }
if ($EXIT_AGGRESSIVE_TRAILING_STOP_PCT -ne "") { $env:EXIT_AGGRESSIVE_TRAILING_STOP_PCT = $EXIT_AGGRESSIVE_TRAILING_STOP_PCT }
if ($EXIT_AGGRESSIVE_MAX_HOLD_DAYS     -ne "") { $env:EXIT_AGGRESSIVE_MAX_HOLD_DAYS     = $EXIT_AGGRESSIVE_MAX_HOLD_DAYS }

& $PYTHON $SCRIPT_EXIT_SCANNER
Assert-Exit "EXIT_SCANNER"

# ============================================================
# КРОК 3 — CPAPI handoff (live BBO)
# Також будує/оновлює conid_cache.json для кроку 4
# ============================================================
Write-Host ""
Write-Host "=== STEP 3: CPAPI HANDOFF (live BBO) ===" -ForegroundColor Cyan

$env:EXECUTION_ROOT               = "artifacts/execution_loop"
$env:CONFIG_NAMES                 = "optimal|aggressive"
$env:BROKER_ACCOUNT_ID            = $IB_ACCOUNT_CODE
$env:CPAPI_BASE_URL               = $CPAPI_BASE_URL
$env:CPAPI_VERIFY_SSL             = "0"
$env:CPAPI_TIMEOUT_SEC            = "10.0"
$env:CPAPI_SNAPSHOT_FIELDS        = "31,84,86,88"
$env:CPAPI_SNAPSHOT_WAIT_SEC      = "2.0"
$env:CPAPI_SNAPSHOT_RETRIES       = "3"
$env:CPAPI_INTER_REQ_SEC          = "0.1"
$env:MAX_PRICE_DEVIATION_PCT      = "8.0"
$env:ALLOW_LAST_FALLBACK          = "1"
$env:FORCE_MASSIVE_WHEN_IB_NO_BBO      = "1"
$env:FORCE_MASSIVE_WHEN_IB_CLOSE_ONLY  = "1"
$env:ENABLE_MASSIVE_FALLBACK      = "1"
$env:MASSIVE_API_KEY              = $MASSIVE_API_KEY
$env:MASSIVE_BASE_URL             = $MASSIVE_BASE_URL
$env:MASSIVE_TIMEOUT_SEC          = "20.0"
$env:TOPK_PRINT                   = "50"
$env:PAUSE_ON_EXIT                = "0"

& $PYTHON $SCRIPT_HANDOFF
Assert-Exit "CPAPI_HANDOFF"

# ============================================================
# КРОК 4 — CPAPI execution
# CPAPI_RESOLVE_CONIDS=0: conid_cache вже є після кроку 3
# ============================================================
Write-Host ""
Write-Host "=== STEP 4: CPAPI EXECUTION ===" -ForegroundColor Cyan

$env:EXECUTION_ROOT          = "artifacts/execution_loop"
$env:CONFIG_NAMES            = "optimal|aggressive"
$env:BROKER_NAME             = "MEXEM"
$env:BROKER_PLATFORM         = "IBKR_CPAPI"
$env:BROKER_ACCOUNT_ID       = $IB_ACCOUNT_CODE
$env:CPAPI_BASE_URL          = $CPAPI_BASE_URL
$env:CPAPI_VERIFY_SSL        = "0"
$env:CPAPI_TIMEOUT_SEC       = "10.0"
$env:CPAPI_WHOLE_TIMEOUT_SEC = "30.0"
$env:CPAPI_FRAC_TIMEOUT_SEC  = "20.0"
$env:CPAPI_TIF               = "DAY"
$env:CPAPI_FRAC_SLIPPAGE_BPS = "5.0"
$env:CPAPI_RESOLVE_CONIDS    = "0"
$env:RESET_BROKER_LOG        = "1"
$env:PAUSE_ON_EXIT           = "0"

& $PYTHON $SCRIPT_EXEC_CPAPI
Assert-Exit "CPAPI_EXEC_ORDERS"

# ============================================================
# КРОК 5 — CPAPI reconcile (post-execution)
# ============================================================
Write-Host ""
Write-Host "=== STEP 5: CPAPI RECONCILE (post-execution) ===" -ForegroundColor Cyan

$env:EXECUTION_ROOT                      = "artifacts/execution_loop"
$env:CONFIG_NAMES                        = "optimal|aggressive"
$env:BROKER_ACCOUNT_ID                   = $IB_ACCOUNT_CODE
$env:CPAPI_BASE_URL                      = $CPAPI_BASE_URL
$env:CPAPI_VERIFY_SSL                    = "0"
$env:CPAPI_TIMEOUT_SEC                   = "10.0"
$env:DRIFT_TOLERANCE_SHARES              = "0.000001"
$env:PREFER_STATE_OVER_ORDERS            = "1"
$env:REQUIRE_BROKER_REFRESH              = "1"
$env:ALLOW_EXISTING_BROKER_POSITIONS_CSV = "1"
$env:CLEANUP_PREVIEW_MODE                = "1"
$env:CLEANUP_INCLUDE_UNEXPECTED          = "1"
$env:CLEANUP_INCLUDE_DRIFT               = "1"
$env:CLEANUP_ONLY_SYMBOLS                = ""
$env:CLEANUP_EXCLUDE_SYMBOLS             = ""
$env:CLEANUP_MIN_ABS_SHARES              = "1.0"
$env:TOPK_PRINT                          = "50"
$env:PAUSE_ON_EXIT                       = "0"

& $PYTHON $SCRIPT_RECONCILE
Assert-Exit "CPAPI_RECONCILE_POST"

# ============================================================
# КРОК 5а — Синхронізація preview → orders
# ============================================================
Write-Host ""
Write-Host "=== STEP 5a: SYNC CLEANUP PREVIEW ===" -ForegroundColor Cyan

$syncScript = @'
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
'@
& $PYTHON -c $syncScript
Assert-Exit "CLEANUP_SYNC"

# ============================================================
# КРОК 5b — Резолвінг conid для cleanup символів
# ============================================================
Write-Host ""
Write-Host "=== STEP 5b: RESOLVE CONIDS FOR CLEANUP ===" -ForegroundColor Cyan

$resolveScript = @'
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] if "__file__" in dir() else Path(".")
SRC_DIR = ROOT / "src"
for p in [ROOT, SRC_DIR]:
    ps = str(p)
    if ps not in sys.path:
        sys.path.insert(0, ps)

import os
from python_edge.broker.cpapi_client import CpapiClient
from python_edge.broker.cpapi_conid_resolver import update_conid_cache

base_url   = os.getenv("CPAPI_BASE_URL",    "https://localhost:5000")
verify_ssl = os.getenv("CPAPI_VERIFY_SSL",  "0").lower() not in {"1","true","yes","on"}
timeout    = float(os.getenv("CPAPI_TIMEOUT_SEC", "30.0"))
exec_root  = Path(os.getenv("EXECUTION_ROOT", "artifacts/execution_loop"))
configs    = [x for x in os.getenv("CONFIG_NAMES","optimal|aggressive").split("|") if x]

client = CpapiClient(base_url, timeout, verify_ssl)

for cfg in configs:
    cleanup_csv = exec_root / cfg / "broker_cleanup_orders.csv"
    cache_path  = exec_root / cfg / "conid_cache.json"
    if not cleanup_csv.exists():
        print(f"[CONID_CLEANUP][{cfg}] no broker_cleanup_orders.csv — skip")
        continue
    import pandas as pd
    df = pd.read_csv(cleanup_csv)
    if "symbol" not in df.columns or df.empty:
        print(f"[CONID_CLEANUP][{cfg}] empty — skip")
        continue
    symbols = sorted(df["symbol"].dropna().astype(str).str.strip().str.upper().unique().tolist())
    print(f"[CONID_CLEANUP][{cfg}] resolving {len(symbols)} symbols: {symbols}")
    cache, unresolved = update_conid_cache(cache_path, client, symbols, force_refresh=False)
    if unresolved:
        print(f"[CONID_CLEANUP][{cfg}] UNRESOLVED: {unresolved}", file=sys.stderr)
    print(f"[CONID_CLEANUP][{cfg}] cache_size={len(cache)} unresolved={len(unresolved)}")
'@
& $PYTHON -c $resolveScript
Assert-Exit "CONID_CLEANUP_RESOLVE"

# ============================================================
# КРОК 6 — CPAPI cleanup send
# ============================================================
Write-Host ""
Write-Host "=== STEP 6: CPAPI CLEANUP SEND ===" -ForegroundColor Cyan

$env:EXECUTION_ROOT                          = "artifacts/execution_loop"
$env:CONFIG_NAMES                            = "optimal|aggressive"
$env:BROKER_NAME                             = "MEXEM"
$env:BROKER_PLATFORM                         = "IBKR_CPAPI"
$env:BROKER_ACCOUNT_ID                       = $IB_ACCOUNT_CODE
$env:CPAPI_BASE_URL                          = $CPAPI_BASE_URL
$env:CPAPI_VERIFY_SSL                        = "0"
$env:CPAPI_TIMEOUT_SEC                       = "10.0"
$env:CPAPI_WHOLE_TIMEOUT_SEC                 = "30.0"
$env:CPAPI_FRAC_TIMEOUT_SEC                  = "20.0"
$env:CPAPI_TIF                               = "DAY"
$env:CPAPI_FRAC_SLIPPAGE_BPS                 = "5.0"
$env:CLEANUP_SEND_PREVIEW_ONLY               = "0"
$env:CLEANUP_SEND_ONLY_SYMBOLS               = ""
$env:CLEANUP_SEND_EXCLUDE_SYMBOLS            = ""
$env:CLEANUP_SEND_MAX_ROWS                   = "1000000"
$env:CLEANUP_SEND_REQUIRE_EMPTY_PENDING_REFS = "0"
$env:CLEANUP_SEND_REQUIRE_SIDE               = "BUY|SELL"
$env:CLEANUP_SEND_REASON_TAG                 = "broker_cleanup_send"
$env:RESET_BROKER_LOG                        = "0"
$env:TOPK_PRINT                              = "50"
$env:PAUSE_ON_EXIT                           = "0"

& $PYTHON $SCRIPT_CLEANUP
Assert-Exit "CPAPI_CLEANUP_SEND"

# ============================================================
# ПІДСУМОК
# ============================================================
Write-Host ""
Write-Host "=== PIPELINE COMPLETE ===" -ForegroundColor Green
Write-Host "Artifacts:" -ForegroundColor Cyan
@(
    "artifacts\execution_loop\optimal\orders.csv",
    "artifacts\execution_loop\optimal\portfolio_state.json",
    "artifacts\execution_loop\optimal\broker_positions.csv",
    "artifacts\execution_loop\optimal\broker_reconcile.csv",
    "artifacts\execution_loop\optimal\broker_cleanup_send_plan.csv",
    "artifacts\execution_loop\optimal\broker_log.json",
    "artifacts\execution_loop\optimal\fills.csv",
    "artifacts\execution_loop\aggressive\orders.csv",
    "artifacts\execution_loop\aggressive\portfolio_state.json",
    "artifacts\execution_loop\aggressive\broker_positions.csv",
    "artifacts\execution_loop\aggressive\broker_reconcile.csv",
    "artifacts\execution_loop\aggressive\broker_cleanup_send_plan.csv",
    "artifacts\execution_loop\aggressive\broker_log.json",
    "artifacts\execution_loop\aggressive\fills.csv"
) | ForEach-Object {
    $full = "$ROOT\$_"
    if (Test-Path $full) {
        Write-Host "  OK  $_" -ForegroundColor Green
    } else {
        Write-Host "  --  $_" -ForegroundColor DarkGray
    }
}

Write-Host ""
Write-Host "Press Enter to exit..."
$null = Read-Host
