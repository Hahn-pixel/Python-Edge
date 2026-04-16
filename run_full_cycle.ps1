Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ============================================================
# Python-Edge — повний цикл одним запуском
#
# PIPELINE:
#   0)  Очищення артефактів execution_loop
#   0а) Патч clip_limit_from_202 для SELL
#   1)  run_broker_reconcile_ibkr.py — live positions → broker_positions.csv
#   2)  run_execution_loop.py     — alignment з broker + optimal + aggressive
#   3)  run_broker_handoff.py     — live BBO з IB (client_id=51)
#   4)  run_broker_adapter_ibkr.py — відправка execution orders
#   5)  run_broker_reconcile_ibkr.py — post-execution reconcile → cleanup_preview
#   5а) Синхронізація preview → orders
#   6)  run_broker_cleanup_send_ibkr.py — cleanup send
# ============================================================

$ROOT   = "C:\Users\Dmytro Govor\Documents\Python-Edge"
$PYTHON = "python"

$SCRIPT_EXEC      = "$ROOT\scripts\run_execution_loop.py"
$SCRIPT_HANDOFF   = "$ROOT\scripts\run_broker_handoff.py"
$SCRIPT_BROKER    = "$ROOT\scripts\run_broker_adapter_ibkr.py"
$SCRIPT_RECONCILE = "$ROOT\scripts\run_broker_reconcile_ibkr.py"
$SCRIPT_CLEANUP   = "$ROOT\scripts\run_broker_cleanup_send_ibkr.py"

$IB_ACCOUNT_CODE  = ""
$NAV_OPTIMAL      = "55204.032916"
$NAV_AGGRESSIVE   = "60986.205462"
$MASSIVE_API_KEY  = $env:MASSIVE_API_KEY
$MASSIVE_BASE_URL = "https://api.massive.com"

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
# Зберігаємо portfolio_state.json (поточний стан позицій)
# Видаляємо всі накопичені broker_log, fills, orders, cleanup файли
# ============================================================
Write-Host ""
Write-Host "=== STEP 0: CLEAN ARTIFACTS ===" -ForegroundColor Cyan

foreach ($cfg in @("optimal", "aggressive")) {
    $dir = "$ROOT\artifacts\execution_loop\$cfg"
    if (-not (Test-Path $dir)) { continue }

    # файли що видаляємо (всі артефакти крім portfolio_state.json)
    # portfolio_state.json видаляємо після першого reconcile (кроку 1)
    $toDelete = @(
        "orders.csv",
        "target_book.csv",
        "broker_log.json",
        "fills.csv",
        "execution_log.csv",
        "positions_mark_to_market.csv",
        "broker_handoff_summary.json",
        "broker_positions.csv",
        "broker_open_orders.csv",
        "broker_pending.csv",
        "broker_reconcile.csv",
        "broker_reconcile_summary.json",
        "broker_cleanup_preview.csv",
        "broker_cleanup_orders.csv",
        "broker_cleanup_emit_summary.json",
        "broker_cleanup_send_plan.csv",
        "broker_cleanup_send_summary.json",
        "orders_exec_backup.csv",
        "orders_pre_cleanup_backup.csv",
        "state_alignment_once.json"
    )
    foreach ($f in $toDelete) {
        $path = "$dir\$f"
        if (Test-Path $path) {
            Remove-Item $path -Force
            Write-Host "  [$cfg] deleted $f" -ForegroundColor DarkGray
        }
    }
    Write-Host "  [$cfg] clean OK (portfolio_state.json preserved for reconcile)" -ForegroundColor Green
}

# ============================================================
# КРОК 0а — Патч clip_limit_from_202 вже застосований локально
# boundary + tick * clip_ticks для SELL — OK
# ============================================================
Write-Host "[0a] clip_limit_from_202 patch — already applied, skip" -ForegroundColor DarkGray

# ============================================================
# PRE-FLIGHT CHECK
# ============================================================
Write-Host ""
Write-Host "=== PRE-FLIGHT CHECK ===" -ForegroundColor Cyan
@(
    $SCRIPT_EXEC, $SCRIPT_HANDOFF, $SCRIPT_BROKER, $SCRIPT_RECONCILE, $SCRIPT_CLEANUP,
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
# КРОК 1 — Broker reconcile (client_id=61)
# Запускаємо ПЕРШИМ щоб отримати broker_positions.csv
# до execution loop який використає його для alignment
# ============================================================
Write-Host ""
Write-Host "=== STEP 1: BROKER RECONCILE (pre-alignment) ===" -ForegroundColor Cyan

# reconcile вимагає orders.csv — створюємо порожній stub якщо відсутній
foreach ($cfg in @("optimal", "aggressive")) {
    $ordersPath = "$ROOT\artifacts\execution_loop\$cfg\orders.csv"
    $dirPath    = "$ROOT\artifacts\execution_loop\$cfg"
    if (-not (Test-Path $dirPath)) { New-Item -ItemType Directory -Path $dirPath -Force | Out-Null }
    if (-not (Test-Path $ordersPath)) {
        "symbol,order_side,delta_shares,target_shares,price,price_source" | Out-File $ordersPath -Encoding utf8
        Write-Host "  [$cfg] created empty orders.csv stub" -ForegroundColor DarkGray
    }
}

$env:EXECUTION_ROOT                        = "artifacts/execution_loop"
$env:CONFIG_NAMES                          = "optimal|aggressive"
$env:IB_HOST                               = "127.0.0.1"
$env:IB_PORT                               = "4002"
$env:IB_CLIENT_ID                          = "61"
$env:IB_ACCOUNT_CODE                       = $IB_ACCOUNT_CODE
$env:IB_TIMEOUT_SEC                        = "20.0"
$env:IB_OPEN_ORDERS_TIMEOUT_SEC            = "20.0"
$env:IB_POSITIONS_TIMEOUT_SEC              = "20.0"
$env:DRIFT_TOLERANCE_SHARES                = "0.000001"
$env:PREFER_STATE_OVER_ORDERS              = "1"
$env:REQUIRE_BROKER_REFRESH                = "1"
$env:ALLOW_EXISTING_BROKER_POSITIONS_CSV   = "1"
$env:CLEANUP_PREVIEW_MODE                  = "1"
$env:CLEANUP_INCLUDE_UNEXPECTED            = "1"
$env:CLEANUP_INCLUDE_DRIFT                 = "1"
$env:CLEANUP_ONLY_SYMBOLS                  = ""
$env:CLEANUP_EXCLUDE_SYMBOLS               = ""
$env:CLEANUP_MIN_ABS_SHARES                = "1.0"
$env:TOPK_PRINT                            = "50"
$env:PAUSE_ON_EXIT                         = "0"

& $PYTHON $SCRIPT_RECONCILE
Assert-Exit "BROKER_RECONCILE_PRE"

# Тепер portfolio_state.json можна видалити — broker_positions.csv вже є
# execution loop з ALIGN_STATE_FROM_BROKER_ONCE=1 побудує новий state
Write-Host "[1b] removing portfolio_state.json for fresh alignment" -ForegroundColor DarkGray
foreach ($cfg in @("optimal", "aggressive")) {
    $statePath = "$ROOT\artifacts\execution_loop\$cfg\portfolio_state.json"
    $alignPath = "$ROOT\artifacts\execution_loop\$cfg\state_alignment_once.json"
    if (Test-Path $statePath) { Remove-Item $statePath -Force }
    if (Test-Path $alignPath) { Remove-Item $alignPath -Force }
}

# ============================================================
# КРОК 2 — Execution loop з broker alignment
# ALIGN_STATE_FROM_BROKER_ONCE=1: читає broker_positions.csv
# і вирівнює portfolio_state.json з реальними позиціями
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

Write-Host "[2b] aggressive" -ForegroundColor White
$env:ACCOUNT_NAV  = $NAV_AGGRESSIVE
$env:CONFIG_NAMES = "aggressive"
& $PYTHON $SCRIPT_EXEC
Assert-Exit "EXEC_LOOP_AGGRESSIVE"

# ============================================================
# КРОК 2 — Broker handoff (client_id=51)
# ============================================================
Write-Host ""
Write-Host "=== STEP 3: BROKER HANDOFF ===" -ForegroundColor Cyan

$env:EXECUTION_ROOT           = "artifacts/execution_loop"
$env:CONFIG_NAMES             = "optimal|aggressive"
$env:IB_HOST                  = "127.0.0.1"
$env:IB_PORT                  = "4002"
$env:IB_CLIENT_ID             = "51"
$env:IB_TIMEOUT_SEC           = "20.0"
$env:IB_MKT_TIMEOUT_SEC       = "8.0"
$env:IB_EXCHANGE              = "SMART"
$env:IB_PRIMARY_EXCHANGE      = ""
$env:IB_CURRENCY              = "USD"
$env:IB_SECURITY_TYPE         = "STK"
$env:IB_MARKET_DATA_TYPE      = "3"
$env:MAX_PRICE_DEVIATION_PCT  = "8.0"
$env:ALLOW_LAST_FALLBACK      = "1"
$env:FORCE_MASSIVE_WHEN_IB_CLOSE_ONLY = "1"
$env:FORCE_MASSIVE_WHEN_IB_NO_BBO     = "1"
$env:ENABLE_MASSIVE_FALLBACK  = "1"
$env:MASSIVE_API_KEY          = $MASSIVE_API_KEY
$env:MASSIVE_BASE_URL         = $MASSIVE_BASE_URL
$env:MASSIVE_TIMEOUT_SEC      = "20.0"
$env:TOPK_PRINT               = "50"
$env:PAUSE_ON_EXIT            = "0"

& $PYTHON $SCRIPT_HANDOFF
Assert-Exit "BROKER_HANDOFF"

# ============================================================
# КРОК 3 — Broker adapter: execution orders (client_id=41)
# ============================================================
Write-Host ""
Write-Host "=== STEP 4: BROKER ADAPTER — EXECUTION ORDERS ===" -ForegroundColor Cyan

$env:EXECUTION_ROOT                           = "artifacts/execution_loop"
$env:CONFIG_NAMES                             = "optimal|aggressive"
$env:BROKER_NAME                              = "MEXEM"
$env:BROKER_PLATFORM                          = "IBKR"
$env:BROKER_ACCOUNT_ID                        = $IB_ACCOUNT_CODE
$env:IB_HOST                                  = "127.0.0.1"
$env:IB_PORT                                  = "4002"
$env:IB_CLIENT_ID                             = "41"
$env:IB_TIMEOUT_SEC                           = "20.0"
$env:IB_ACCOUNT_CODE                          = $IB_ACCOUNT_CODE
$env:IB_TIME_IN_FORCE                         = "DAY"
$env:IB_OUTSIDE_RTH                           = "1"
$env:IB_ALLOW_FRACTIONAL                      = "0"
$env:IB_EXCHANGE                              = "SMART"
$env:IB_PRIMARY_EXCHANGE                      = ""
$env:IB_CURRENCY                              = "USD"
$env:IB_SECURITY_TYPE                         = "STK"
$env:IB_OPEN_ORDERS_TIMEOUT_SEC               = "20.0"
$env:IB_POSITIONS_TIMEOUT_SEC                 = "20.0"
$env:IB_REFRESH_POSITIONS_ON_CONNECT          = "0"
$env:IB_REQUIRE_POSITIONS_ON_CONNECT          = "0"
$env:RESET_BROKER_LOG                         = "1"
$env:BROKER_SYMBOL_MAP_FILE                   = ""
$env:BROKER_SYMBOL_MAP_JSON                   = ""
$env:IB_FRACTIONAL_REJECT_CODES               = "10243|10247|10248|10249|10250"
$env:IB_LMT_PRICE_MIN_ABS                     = "0.01"
$env:IB_POLL_VERBOSE                          = "changes"
$env:IB_POLL_PRINT_EVERY                      = "5"
$env:IB_CONTRACT_DETAILS_RETRIES              = "3"
$env:IB_CONTRACT_DETAILS_RETRY_SLEEP_SEC      = "1.0"
$env:IB_ALLOW_SUBMIT_WITHOUT_CONTRACT_DETAILS  = "1"
$env:IB_ALLOW_PRIMARY_EXCHANGE_FALLBACK        = "1"
$env:IB_RETRY_202_ENABLED                     = "1"
$env:IB_RETRY_202_CLIP_TICKS                  = "2"
$env:IB_REPRICE_ENABLED                       = "1"
$env:IB_REPRICE_WAIT_SEC                      = "8.0"
$env:IB_REPRICE_STEPS_BPS                     = "15|10|5|0"
$env:IB_REPRICE_FINAL_MODE                    = "marketable_lmt"
$env:IB_REPRICE_FINAL_MARKETABLE_BPS          = "12.0"
$env:IB_REPRICE_MAX_DEVIATION_PCT             = "1.25"
$env:IB_REPRICE_CANCEL_POLL_ATTEMPTS          = "6"
$env:IB_REPRICE_CANCEL_POLL_SLEEP_SEC         = "1.0"
$env:REQUIRE_PRICE_HINT_SOURCE                = "1"
$env:REQUIRE_QUOTE_TS                         = "1"
$env:ENFORCE_OPEN_ORDER_DUP_GUARD             = "1"
$env:ENFORCE_OPEN_ORDER_DUP_GUARD_STRICT      = "0"
$env:ENFORCE_OPEN_ORDER_DUP_GUARD_QTY_TOL    = "0.000001"
$env:PAUSE_ON_EXIT                            = "0"

& $PYTHON $SCRIPT_BROKER
Assert-Exit "BROKER_EXEC_ORDERS"

# ============================================================
# КРОК 4 — Broker reconcile (client_id=61)
# ============================================================
Write-Host ""
Write-Host "=== STEP 5: BROKER RECONCILE ===" -ForegroundColor Cyan

$env:EXECUTION_ROOT                        = "artifacts/execution_loop"
$env:CONFIG_NAMES                          = "optimal|aggressive"
$env:IB_HOST                               = "127.0.0.1"
$env:IB_PORT                               = "4002"
$env:IB_CLIENT_ID                          = "61"
$env:IB_ACCOUNT_CODE                       = $IB_ACCOUNT_CODE
$env:IB_TIMEOUT_SEC                        = "20.0"
$env:IB_OPEN_ORDERS_TIMEOUT_SEC            = "20.0"
$env:IB_POSITIONS_TIMEOUT_SEC              = "20.0"
$env:DRIFT_TOLERANCE_SHARES                = "0.000001"
$env:PREFER_STATE_OVER_ORDERS              = "1"
$env:REQUIRE_BROKER_REFRESH                = "1"
$env:ALLOW_EXISTING_BROKER_POSITIONS_CSV   = "1"
$env:CLEANUP_PREVIEW_MODE                  = "1"
$env:CLEANUP_INCLUDE_UNEXPECTED            = "1"
$env:CLEANUP_INCLUDE_DRIFT                 = "1"
$env:CLEANUP_ONLY_SYMBOLS                  = ""
$env:CLEANUP_EXCLUDE_SYMBOLS               = ""
$env:CLEANUP_MIN_ABS_SHARES                = "1.0"
$env:TOPK_PRINT                            = "50"
$env:PAUSE_ON_EXIT                         = "0"

& $PYTHON $SCRIPT_RECONCILE
Assert-Exit "BROKER_RECONCILE"

# ============================================================
# КРОК 5а — Синхронізація preview → orders (rename колонок)
# reconcile: cleanup_side/cleanup_qty → order_side/delta_shares
# ============================================================
$syncScript = @'
import pandas as pd
from pathlib import Path

for cfg in ["optimal", "aggressive"]:
    preview = Path(f"artifacts/execution_loop/{cfg}/broker_cleanup_preview.csv")
    orders  = Path(f"artifacts/execution_loop/{cfg}/broker_cleanup_orders.csv")
    if not preview.exists():
        print(f"[SYNC][{cfg}] preview not found — skip")
        continue
    df = pd.read_csv(preview)
    if "cleanup_side" in df.columns:
        df = df.rename(columns={"cleanup_side": "order_side", "cleanup_qty": "delta_shares"})
    df.to_csv(orders, index=False)
    print(f"[SYNC][{cfg}] rows={len(df)} → broker_cleanup_orders.csv")
'@
& $PYTHON -c $syncScript
Assert-Exit "CLEANUP_SYNC"

# ============================================================
# КРОК 5 — Cleanup send (client_id=41 через subprocess)
# IB_OUTSIDE_RTH=0: cleanup тільки в RTH
# IB_REPRICE_MAX_DEVIATION_PCT=50: avg_cost може сильно відрізнятись
# CLEANUP_SEND_REQUIRE_EMPTY_PENDING_REFS=0: reconcile вже перевірив
# ============================================================
Write-Host ""
Write-Host "=== STEP 6: CLEANUP SEND ===" -ForegroundColor Cyan

$env:EXECUTION_ROOT                           = "artifacts/execution_loop"
$env:CONFIG_NAMES                             = "optimal|aggressive"
$env:CLEANUP_SEND_PREVIEW_ONLY                = "0"
$env:CLEANUP_SEND_ONLY_SYMBOLS                = ""
$env:CLEANUP_SEND_EXCLUDE_SYMBOLS             = ""
$env:CLEANUP_SEND_MAX_ROWS                    = "1000000"
$env:CLEANUP_SEND_REQUIRE_EMPTY_PENDING_REFS  = "0"
$env:CLEANUP_SEND_REQUIRE_SIDE                = "BUY|SELL"
$env:CLEANUP_SEND_REASON_TAG                  = "broker_cleanup_send"
$env:TOPK_PRINT                               = "50"
$env:BROKER_NAME                              = "MEXEM"
$env:BROKER_PLATFORM                          = "IBKR"
$env:BROKER_ACCOUNT_ID                        = $IB_ACCOUNT_CODE
$env:IB_HOST                                  = "127.0.0.1"
$env:IB_PORT                                  = "4002"
$env:IB_CLIENT_ID                             = "41"
$env:IB_TIMEOUT_SEC                           = "20.0"
$env:IB_ACCOUNT_CODE                          = $IB_ACCOUNT_CODE
$env:IB_TIME_IN_FORCE                         = "DAY"
$env:IB_OUTSIDE_RTH                           = "0"
$env:IB_ALLOW_FRACTIONAL                      = "0"
$env:IB_EXCHANGE                              = "SMART"
$env:IB_PRIMARY_EXCHANGE                      = ""
$env:IB_CURRENCY                              = "USD"
$env:IB_SECURITY_TYPE                         = "STK"
$env:IB_OPEN_ORDERS_TIMEOUT_SEC               = "20.0"
$env:IB_POSITIONS_TIMEOUT_SEC                 = "20.0"
$env:IB_REFRESH_POSITIONS_ON_CONNECT          = "0"
$env:IB_REQUIRE_POSITIONS_ON_CONNECT          = "0"
$env:RESET_BROKER_LOG                         = "0"
$env:BROKER_SYMBOL_MAP_FILE                   = ""
$env:BROKER_SYMBOL_MAP_JSON                   = ""
$env:IB_FRACTIONAL_REJECT_CODES               = "10243|10247|10248|10249|10250"
$env:IB_LMT_PRICE_MIN_ABS                     = "0.01"
$env:IB_POLL_VERBOSE                          = "changes"
$env:IB_POLL_PRINT_EVERY                      = "5"
$env:IB_CONTRACT_DETAILS_RETRIES              = "3"
$env:IB_CONTRACT_DETAILS_RETRY_SLEEP_SEC      = "1.0"
$env:IB_ALLOW_SUBMIT_WITHOUT_CONTRACT_DETAILS  = "1"
$env:IB_ALLOW_PRIMARY_EXCHANGE_FALLBACK        = "1"
$env:IB_RETRY_202_ENABLED                     = "1"
$env:IB_RETRY_202_CLIP_TICKS                  = "2"
$env:IB_REPRICE_ENABLED                       = "1"
$env:IB_REPRICE_WAIT_SEC                      = "8.0"
$env:IB_REPRICE_STEPS_BPS                     = "15|10|5|0"
$env:IB_REPRICE_FINAL_MODE                    = "marketable_lmt"
$env:IB_REPRICE_FINAL_MARKETABLE_BPS          = "12.0"
$env:IB_REPRICE_MAX_DEVIATION_PCT             = "50.0"
$env:IB_REPRICE_CANCEL_POLL_ATTEMPTS          = "6"
$env:IB_REPRICE_CANCEL_POLL_SLEEP_SEC         = "1.0"
$env:REQUIRE_PRICE_HINT_SOURCE                = "1"
$env:REQUIRE_QUOTE_TS                         = "1"
$env:ENFORCE_OPEN_ORDER_DUP_GUARD             = "1"
$env:ENFORCE_OPEN_ORDER_DUP_GUARD_STRICT      = "0"
$env:ENFORCE_OPEN_ORDER_DUP_GUARD_QTY_TOL    = "0.000001"
$env:PAUSE_ON_EXIT                            = "0"

& $PYTHON $SCRIPT_CLEANUP
Assert-Exit "CLEANUP_SEND"

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
