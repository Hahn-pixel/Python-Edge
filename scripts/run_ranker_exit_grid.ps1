Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$srcPath = (Resolve-Path ".\src").Path
if ([string]::IsNullOrWhiteSpace($env:PYTHONPATH)) {
    $env:PYTHONPATH = $srcPath
} else {
    $env:PYTHONPATH = "$srcPath;$env:PYTHONPATH"
}

$env:PAUSE_ON_EXIT = "0"

# Fixed baseline
$env:WF_TRAIN_DAYS = "252"
$env:WF_TEST_DAYS = "63"
$env:WF_STEP_DAYS = "63"
$env:WF_PURGE_DAYS = "5"
$env:WF_EMBARGO_DAYS = "5"
$env:TOP_PCT = "0.10"
$env:ENTER_PCT = "0.10"
$env:EXIT_PCT = "0.20"
$env:FINAL_WEIGHT_CAP = "0.08"
$env:MAX_ADV_PARTICIPATION = "0.05"
$env:PORTFOLIO_NOTIONAL = "1.0"
$env:APPLY_DYNAMIC_BUDGETS = "1"
$env:BUDGET_INPUT_LAG_DAYS = "1"
$env:LOW_PRICE_MIN = "5.0"
$env:LOW_DV_MIN = "1000000"
$env:FEE_BPS = "1.0"
$env:BASE_SLIPPAGE_BPS = "2.0"
$env:BORROW_BPS_DAILY = "1.0"
$env:SPREAD_BPS = "3.0"
$env:IMPACT_BPS = "8.0"
$env:LOW_PRICE_PENALTY_BPS = "4.0"
$env:HTB_BORROW_BPS_DAILY = "8.0"

# Fixed baseline candidate #1
$env:MAX_DAILY_TURNOVER = "0.80"
$env:CAPITAL_POLICY = "scale_up_to_target"

# Exit grid only
$peakDrops = @("0.05", "0.08", "0.12", "0.20", "0.30")

$runStamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outDir = Join-Path (Resolve-Path ".").Path ("data\wf_exit_grid_" + $runStamp)
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

Write-Host "[GRID] output_dir=$outDir"
Write-Host "[GRID] peakDrops=$($peakDrops -join ', ')"

function Get-LastMatchLine {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Pattern
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        return ""
    }

    $matches = @(
        Select-String -Path $Path -Pattern $Pattern -Encoding UTF8
    )
    if ($matches.Length -eq 0) {
        return ""
    }
    return [string]$matches[-1].Line
}

function Extract-Metric {
    param(
        [Parameter(Mandatory = $true)][string]$Line,
        [Parameter(Mandatory = $true)][string]$Key
    )

    if ([string]::IsNullOrWhiteSpace($Line)) {
        return $null
    }

    $pattern = [regex]::Escape($Key) + '=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'
    $m = [regex]::Match($Line, $pattern)
    if (-not $m.Success) {
        return $null
    }
    return [double]$m.Groups[1].Value
}

$summaryRows = @()
$totalRuns = $peakDrops.Length
$runIdx = 0

foreach ($peakDrop in $peakDrops) {
    $runIdx += 1
    $env:PEAK_TRAIL_DROP_LONG = $peakDrop
    $env:PEAK_TRAIL_DROP_SHORT = $peakDrop

    $tag = "peak_${peakDrop}__turnover_0.80__capital_scale_up_to_target"
    $safeTag = $tag.Replace('.', '_')
    $logFile = Join-Path $outDir ("run__" + $safeTag + ".log")

    Write-Host ""
    Write-Host "=============================================================="
    Write-Host "[RUN $runIdx/$totalRuns] $tag"
    Write-Host "[LOG] $logFile"
    Write-Host "=============================================================="

    if (Test-Path -LiteralPath $logFile) {
        Remove-Item -LiteralPath $logFile -Force
    }

    $startTs = Get-Date
    python .\scripts\run_ranker_wf.py *>> $logFile
    $exitCode = $LASTEXITCODE
    $endTs = Get-Date
    $elapsedSec = [int](New-TimeSpan -Start $startTs -End $endTs).TotalSeconds

    Write-Host "[DONE] tag=$tag exit_code=$exitCode elapsed=${elapsedSec}s"

    $overallMain = Get-LastMatchLine -Path $logFile -Pattern '^\[WF\]\[full_regime_stack_neutralized_sized_barbell_peaktrail_priority_exec\]\[OVERALL\]'
    $overallDiag = Get-LastMatchLine -Path $logFile -Pattern '^\[WF\]\[diag_barbell_no_exits\]\[OVERALL\]'

    $summaryRows += [pscustomobject]@{
        peak_drop       = [double]$peakDrop
        exit_code       = $exitCode
        elapsed_sec     = $elapsedSec
        main_cum_ret    = Extract-Metric -Line $overallMain -Key "cum_ret"
        main_avg_turnover = Extract-Metric -Line $overallMain -Key "avg_turnover"
        main_cap_hit_rate = Extract-Metric -Line $overallMain -Key "cap_hit_rate"
        main_avg_hold_days = Extract-Metric -Line $overallMain -Key "avg_hold_days"
        main_exit_rate    = Extract-Metric -Line $overallMain -Key "exit_rate"
        diag_cum_ret      = Extract-Metric -Line $overallDiag -Key "cum_ret"
        delta_cum_ret     = $null
        overall_line      = $overallMain
        diag_line         = $overallDiag
        log_file          = $logFile
    }
}

foreach ($row in $summaryRows) {
    if ($null -ne $row.main_cum_ret -and $null -ne $row.diag_cum_ret) {
        $row.delta_cum_ret = [double]$row.main_cum_ret - [double]$row.diag_cum_ret
    }
}

$summaryCsv = Join-Path $outDir "wf_exit_grid_summary.csv"
$summaryRows | Sort-Object peak_drop | Export-Csv -Path $summaryCsv -NoTypeInformation -Encoding UTF8

Write-Host ""
Write-Host "==================== EXIT GRID SUMMARY ===================="
$summaryRows | Sort-Object peak_drop | Format-Table -AutoSize
Write-Host ""
Write-Host "[SAVED] $summaryCsv"
Read-Host "Press Enter to exit"