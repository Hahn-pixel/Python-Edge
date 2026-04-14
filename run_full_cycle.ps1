Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ============================================================
# Python-Edge — повний цикл одним запуском
#
# PIPELINE (строгий порядок):
#   1) run_execution_loop.py     — optimal + aggressive
#   2) run_broker_handoff.py     — live BBO з IB (client_id=51)
#   3) run_broker_adapter_ibkr.py — відправка execution orders
#   4) run_broker_reconcile_ibkr.py — live positions з IB (client_id=61)
#                                     → broker_cleanup_preview.csv
#   5) run_broker_cleanup_send_ibkr.py — cleanup send з broker_avg_cost
# ============================================================

$ROOT   = "C:\Users\Dmytro Govor\Documents\Python-Edge"
$PYTHON = "python"

$SCRIPT_EXEC      = "$ROOT\scripts\run_execution_loop.py"
$SCRIPT_HANDOFF   = "$ROOT\scripts\run_broker_handoff.py"
$SCRIPT_BROKER    = "$ROOT\scripts\run_broker_adapter_ibkr.py"
$SCRIPT_RECONCILE = "$ROOT\scripts\run_broker_reconcile_ibkr.py"
$SCRIPT_CLEANUP   = "$ROOT\scripts\run_broker_cleanup_send_ibkr.py"

# ── акаунт IB ────────────────────────────────────────────────
$IB_ACCOUNT_CODE = ""   # напр. "U1234567"

# ── NAV по конфігах ──────────────────────────────────────────
$NAV_OPTIMAL    = "55204.032916"
$NAV_AGGRESSIVE = "60986.205462"

# ── Massive API ───────────────────────────────────────────────
$MASSIVE_API_KEY  = $env:MASSIVE_API_KEY   # береться з env якщо є
$MASSIVE_BASE_URL = "https://api.massive.com"

# ============================================================
# Хелпер: перевірка exit code
# ============================================================
function Assert-Exit {
    param([string]$Label)
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "[$Label] FAILED with exit code $LASTEXITCODE" -ForegroundColor Red
        throw "Pipeline aborted at step: $Label"
    }
    Write-Host "[$Label] OK" -ForegroundColor Green
}

# ============================================================
# PRE-FLIGHT CHECK
# ============================================================
Write-Host "=== PRE-FLIGHT CHECK ===" -ForegroundColor Cyan
@(
    $SCRIPT_EXEC,
    $SCRIPT_HANDOFF,
    $SCRIPT_BROKER,
    $SCRIPT_RECONCILE,
    $SCRIPT_CLEANUP,
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

Set-Location $ROOT

# ============================================================
# КРОК 1 — Execution loop
# ============================================================
Write-Host ""
Write-Host "=== STEP 1: EXECUTION LOOP ===" -ForegroundColor Cyan

# --- optimal ---
Write-Host "[1a] optimal" -ForegroundColor White
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
$env:ALIGN_STATE_FROM_BROKER_ONCE             = "0"
$env:ALIGN_STATE_REQUIRE_BROKER_POSITIONS     = "0"
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

# --- aggressive ---
Write-Host "[1b] aggressive" -ForegroundColor White
$env:ACCOUNT_NAV  = $NAV_AGGRESSIVE
$env:CONFIG_NAMES = "aggressive"

& $PYTHON $SCRIPT_EXEC
Assert-Exit "EXEC_LOOP_AGGRESSIVE"

# ============================================================
# КРОК 2 — Broker handoff (client_id=51)
# Запитує live BBO з IB, збагачує orders.csv:
# price, bid/ask/mid, price_hint_source, quote_ts
# Fallback на massive якщо IB повертає тільки close або без BBO
# ============================================================
Write-Host ""
Write-Host "=== STEP 2: BROKER HANDOFF ===" -ForegroundColor Cyan

$env:EXECUTION_ROOT                       = "artifacts/execution_loop"
$env:CONFIG_NAMES                         = "optimal|aggressive"
$env:IB_HOST                              = "127.0.0.1"
$env:IB_PORT                              = "4002"
$env:IB_CLIENT_ID                         = "51"
$env:IB_TIMEOUT_SEC                       = "20.0"
$env:IB_MKT_TIMEOUT_SEC                   = "8.0"
$env:IB_EXCHANGE                          = "SMART"
$env:IB_PRIMARY_EXCHANGE                  = ""
$env:IB_CURRENCY                          = "USD"
$env:IB_SECURITY_TYPE                     = "STK"
$env:IB_MARKET_DATA_TYPE                  = "3"
$env:MAX_PRICE_DEVIATION_PCT              = "8.0"
$env:ALLOW_LAST_FALLBACK                  = "1"
$env:FORCE_MASSIVE_WHEN_IB_CLOSE_ONLY     = "1"
$env:FORCE_MASSIVE_WHEN_IB_NO_BBO         = "1"
$env:ENABLE_MASSIVE_FALLBACK              = "1"
$env:MASSIVE_API_KEY                      = $MASSIVE_API_KEY
$env:MASSIVE_BASE_URL                     = $MASSIVE_BASE_URL
$env:MASSIVE_TIMEOUT_SEC                  = "20.0"
$env:TOPK_PRINT                           = "50"
$env:PAUSE_ON_EXIT                        = "0"

& $PYTHON $SCRIPT_HANDOFF
Assert-Exit "BROKER_HANDOFF"

# ============================================================
# КРОК 3 — Broker adapter: execution orders (client_id=41)
# orders.csv збагачений handoff — price_hint_source і quote_ts є
# ============================================================
Write-Host ""
Write-Host "=== STEP 3: BROKER ADAPTER — EXECUTION ORDERS ===" -ForegroundColor Cyan

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
# handoff гарантує price_hint_source і quote_ts
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
# Підключається до IB, отримує live positions + open orders
# Порівнює з portfolio_state.json → broker_cleanup_preview.csv
# ============================================================
Write-Host ""
Write-Host "=== STEP 4: BROKER RECONCILE ===" -ForegroundColor Cyan

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
# КРОК 5 — Broker cleanup send (client_id=41 через subprocess)
# run_broker_cleanup_send_ibkr.py:
#   - читає broker_cleanup_preview.csv (згенерований reconcile)
#   - будує send_plan з broker_avg_cost як price hint
#   - підміняє orders.csv ← send_plan
#   - викликає broker_adapter_ibkr через subprocess
#   - відновлює orders.csv у finally
# IB_OUTSIDE_RTH=0: виконується тільки в RTH (тихі 202 поза RTH)
# IB_REPRICE_MAX_DEVIATION_PCT=50: avg_cost може сильно відрізнятись від ринку
# ============================================================
Write-Host ""
Write-Host "=== STEP 5: CLEANUP SEND ===" -ForegroundColor Cyan

$env:EXECUTION_ROOT                           = "artifacts/execution_loop"
$env:CONFIG_NAMES                             = "optimal|aggressive"
$env:CLEANUP_SEND_PREVIEW_ONLY                = "0"
$env:CLEANUP_SEND_ONLY_SYMBOLS                = ""
$env:CLEANUP_SEND_EXCLUDE_SYMBOLS             = ""
$env:CLEANUP_SEND_MAX_ROWS                    = "1000000"
$env:CLEANUP_SEND_REQUIRE_EMPTY_PENDING_REFS  = "1"
$env:CLEANUP_SEND_REQUIRE_SIDE                = "BUY|SELL"
$env:CLEANUP_SEND_REASON_TAG                  = "broker_cleanup_send"
$env:TOPK_PRINT                               = "50"
# broker_adapter env — передається через os.environ.copy() у subprocess
$env:BROKER_NAME                              = "MEXEM"
$env:BROKER_PLATFORM                          = "IBKR"
$env:BROKER_ACCOUNT_ID                        = $IB_ACCOUNT_CODE
$env:IB_HOST                                  = "127.0.0.1"
$env:IB_PORT                                  = "4002"
$env:IB_CLIENT_ID                             = "41"
$env:IB_TIMEOUT_SEC                           = "20.0"
$env:IB_ACCOUNT_CODE                          = $IB_ACCOUNT_CODE
$env:IB_TIME_IN_FORCE                         = "DAY"
# cleanup виконується тільки в RTH — поза RTH тихі 202 без msg для багатьох паперів
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
# cleanup закриває старі позиції — avg_cost може сильно відрізнятись від ринку
# 50% cap дозволяє 202 override дійти до broker-guided ціни навіть при великому gap
$env:IB_REPRICE_MAX_DEVIATION_PCT             = "50.0"
$env:IB_REPRICE_CANCEL_POLL_ATTEMPTS          = "6"
$env:IB_REPRICE_CANCEL_POLL_SLEEP_SEC         = "1.0"
# cleanup_send сам ставить price_hint_source і quote_ts через broker_avg_cost
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
    "artifacts\execution_loop\optimal\broker_handoff_summary.json",
    "artifacts\execution_loop\optimal\broker_positions.csv",
    "artifacts\execution_loop\optimal\broker_reconcile.csv",
    "artifacts\execution_loop\optimal\broker_cleanup_preview.csv",
    "artifacts\execution_loop\optimal\broker_cleanup_send_plan.csv",
    "artifacts\execution_loop\optimal\broker_cleanup_send_summary.json",
    "artifacts\execution_loop\optimal\broker_log.json",
    "artifacts\execution_loop\optimal\fills.csv",
    "artifacts\execution_loop\aggressive\orders.csv",
    "artifacts\execution_loop\aggressive\portfolio_state.json",
    "artifacts\execution_loop\aggressive\broker_handoff_summary.json",
    "artifacts\execution_loop\aggressive\broker_positions.csv",
    "artifacts\execution_loop\aggressive\broker_reconcile.csv",
    "artifacts\execution_loop\aggressive\broker_cleanup_preview.csv",
    "artifacts\execution_loop\aggressive\broker_cleanup_send_plan.csv",
    "artifacts\execution_loop\aggressive\broker_cleanup_send_summary.json",
    "artifacts\execution_loop\aggressive\broker_log.json",
    "artifacts\execution_loop\aggressive\fills.csv"
) | ForEach-Object {
    $full = "$ROOT\$_"
    if (Test-Path $full) {
        Write-Host "  OK  $_" -ForegroundColor Green
    } else {
        Write-Host "  --  $_ (not yet created)" -ForegroundColor DarkGray
    }
}

Write-Host ""
Write-Host "Press Enter to exit..."
$null = Read-Host
