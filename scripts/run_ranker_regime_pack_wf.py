from __future__ import annotations

import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
for _p in [ROOT, SRC_DIR]:
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

import numpy as np
import pandas as pd

from python_edge.execution.cost_model import attach_execution_costs
from python_edge.model.cross_sectional_signal import build_cross_sectional_signal
from python_edge.model.neutralize import add_beta_proxy, neutralize_score_cross_section
from python_edge.model.risk_model import build_risk_model
from python_edge.portfolio.budget_allocation import attach_dynamic_side_budgets, apply_side_budgets
try:
    from python_edge.portfolio.exit_rules import apply_adaptive_exit_rules
except Exception:
    def apply_adaptive_exit_rules(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        out = df.copy()
        out["dbg_exit_rules_fallback"] = 1
        return out
from python_edge.portfolio.holding_inertia import apply_holding_inertia
from python_edge.portfolio.position_limits import apply_position_filters, cap_final_weight, normalize_gross_exposure, renormalize_after_caps
from python_edge.portfolio.signal_sizing import apply_signal_strength_sizing
from python_edge.portfolio.turnover_control import cap_daily_turnover
from python_edge.wf.evaluate_ranker import evaluate_long_short, print_summary, summarize_daily_returns
from python_edge.wf.splits import build_walkforward_splits, print_split_summary

EPS = 1e-12
FEATURE_FILE = Path("data") / "features" / "feature_matrix_v1.parquet"
DIAG_DIR = Path("data") / "diagnostics"
DIAG_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = str(os.getenv("TARGET_COL", "target_fwd_ret_1d")).strip()
TRAIN_DAYS = int(os.getenv("WF_TRAIN_DAYS", "252"))
TEST_DAYS = int(os.getenv("WF_TEST_DAYS", "63"))
STEP_DAYS = int(os.getenv("WF_STEP_DAYS", "63"))
PURGE_DAYS = int(os.getenv("WF_PURGE_DAYS", "5"))
EMBARGO_DAYS = int(os.getenv("WF_EMBARGO_DAYS", "5"))
ENTER_PCT = float(os.getenv("ENTER_PCT", "0.10"))
EXIT_PCT = float(os.getenv("EXIT_PCT", "0.20"))
MIN_TEST_ROWS = int(os.getenv("MIN_TEST_ROWS", "100"))
MAX_DAILY_TURNOVER = float(os.getenv("MAX_DAILY_TURNOVER", "0.80"))
FINAL_WEIGHT_CAP = float(os.getenv("FINAL_WEIGHT_CAP", "0.08"))
CAPITAL_POLICY = str(os.getenv("CAPITAL_POLICY", "scale_up_to_target")).strip().lower()
MAX_ADV_PARTICIPATION = float(os.getenv("MAX_ADV_PARTICIPATION", "0.05"))
PORTFOLIO_NOTIONAL = float(os.getenv("PORTFOLIO_NOTIONAL", "1.0"))
APPLY_DYNAMIC_BUDGETS = str(os.getenv("APPLY_DYNAMIC_BUDGETS", "1")).strip() == "1"
BUDGET_INPUT_LAG_DAYS = int(os.getenv("BUDGET_INPUT_LAG_DAYS", "1"))
LOW_PRICE_MIN = float(os.getenv("LOW_PRICE_MIN", "5.0"))
LOW_DV_MIN = float(os.getenv("LOW_DV_MIN", "1000000"))
FEE_BPS = float(os.getenv("FEE_BPS", "1.0"))
BASE_SLIPPAGE_BPS = float(os.getenv("BASE_SLIPPAGE_BPS", "2.0"))
BORROW_BPS_DAILY = float(os.getenv("BORROW_BPS_DAILY", "1.0"))
SPREAD_BPS = float(os.getenv("SPREAD_BPS", "3.0"))
IMPACT_BPS = float(os.getenv("IMPACT_BPS", "8.0"))
LOW_PRICE_PENALTY_BPS = float(os.getenv("LOW_PRICE_PENALTY_BPS", "4.0"))
HTB_BORROW_BPS_DAILY = float(os.getenv("HTB_BORROW_BPS_DAILY", "8.0"))
PEAK_TRAIL_DROP_LONG = float(os.getenv("PEAK_TRAIL_DROP_LONG", "0.10"))
PEAK_TRAIL_DROP_SHORT = float(os.getenv("PEAK_TRAIL_DROP_SHORT", "0.10"))
PEAK_TRAIL_MIN_AGE_LONG = int(os.getenv("PEAK_TRAIL_MIN_AGE_LONG", "2"))
PEAK_TRAIL_MIN_AGE_SHORT = int(os.getenv("PEAK_TRAIL_MIN_AGE_SHORT", "1"))
CS_WINSOR_LOWER_Q = float(os.getenv("CS_WINSOR_LOWER_Q", "0.02"))
CS_WINSOR_UPPER_Q = float(os.getenv("CS_WINSOR_UPPER_Q", "0.98"))
CS_CONF_FLOOR = float(os.getenv("CS_CONF_FLOOR", "0.35"))
CS_CONF_CAP = float(os.getenv("CS_CONF_CAP", "1.25"))
CS_FINAL_SCORE_CAP = float(os.getenv("CS_FINAL_SCORE_CAP", "6.0"))
RISK_BETA_PENALTY = float(os.getenv("RISK_BETA_PENALTY", "0.35"))
RISK_LIQ_PENALTY = float(os.getenv("RISK_LIQ_PENALTY", "0.35"))
RISK_VOL_PENALTY = float(os.getenv("RISK_VOL_PENALTY", "0.45"))
RISK_MARKET_REGIME_PENALTY = float(os.getenv("RISK_MARKET_REGIME_PENALTY", "0.15"))
RISK_FLOOR = float(os.getenv("RISK_FLOOR", "0.35"))
RISK_CAP = float(os.getenv("RISK_CAP", "3.50"))
REGIME_HIGH_Q = float(os.getenv("REGIME_HIGH_Q", "0.70"))
REGIME_LOW_Q = float(os.getenv("REGIME_LOW_Q", "0.30"))
ALPHA_W_REV_HI_RVOL = float(os.getenv("ALPHA_W_REV_HI_RVOL", "1.00"))
ALPHA_W_GAP_Z_RVOL = float(os.getenv("ALPHA_W_GAP_Z_RVOL", "1.00"))
ALPHA_W_PRESSURE_HI_LIQ = float(os.getenv("ALPHA_W_PRESSURE_HI_LIQ", "0.60"))
SIZING_PRESET = str(os.getenv("SIZING_PRESET", "production_residual")).strip()
PAUSE_ON_EXIT_ENV = str(os.getenv("PAUSE_ON_EXIT", "auto")).strip().lower()


@dataclass(frozen=True)
class FoldArtifacts:
    fold_id: int
    daily: pd.DataFrame
    portfolio: pd.DataFrame


# ------------------------------------------------------------
# RUNTIME
# ------------------------------------------------------------

def _enable_line_buffering() -> None:
    for stream_name in ["stdout", "stderr"]:
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(line_buffering=True)
            except Exception:
                pass


def _should_pause_on_exit() -> bool:
    if PAUSE_ON_EXIT_ENV in {"0", "false", "no", "off"}:
        return False
    if PAUSE_ON_EXIT_ENV in {"1", "true", "yes", "on"}:
        return True
    stdin_obj = getattr(sys, "stdin", None)
    stdout_obj = getattr(sys, "stdout", None)
    stdin_is_tty = bool(stdin_obj is not None and hasattr(stdin_obj, "isatty") and stdin_obj.isatty())
    stdout_is_tty = bool(stdout_obj is not None and hasattr(stdout_obj, "isatty") and stdout_obj.isatty())
    return stdin_is_tty and stdout_is_tty


def _press_enter_exit(code: int) -> None:
    if _should_pause_on_exit():
        try:
            print(f"\n[EXIT] code={code}")
            input("Press Enter to exit...")
        except Exception:
            pass
    raise SystemExit(code)


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def _safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("float64")


def _series_or_zeros(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=df.index, dtype="float64")


def _cs_rank_pct(df: pd.DataFrame, col: str) -> pd.Series:
    return _safe_numeric(df[col]).groupby(df["date"], sort=False).rank(method="average", pct=True)


def _cs_zscore(df: pd.DataFrame, col: str) -> pd.Series:
    x = _safe_numeric(df[col])
    grp = x.groupby(df["date"], sort=False)
    mean = grp.transform("mean")
    std = grp.transform("std").replace(0.0, np.nan)
    out = (x - mean) / std
    return out.replace([np.inf, -np.inf], np.nan)


def _slice_by_date(df: pd.DataFrame, start_date: object, end_date: object) -> pd.DataFrame:
    return df.loc[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()


def _debug_counter_summary(df: pd.DataFrame) -> Dict[str, float]:
    return {
        "rows": float(len(df)),
        "dates": float(df["date"].nunique()),
        "symbols": float(df["symbol"].nunique()),
        "alpha_rev_hi_rvol_non_na": float(df["alpha_rev_hi_rvol"].notna().sum()) if "alpha_rev_hi_rvol" in df.columns else 0.0,
        "alpha_gap_z_rvol_non_na": float(df["alpha_gap_z_rvol"].notna().sum()) if "alpha_gap_z_rvol" in df.columns else 0.0,
        "alpha_pressure_hi_liq_non_na": float(df["alpha_pressure_hi_liq"].notna().sum()) if "alpha_pressure_hi_liq" in df.columns else 0.0,
        "dbg_mask_rvol_hi_sum": float(_series_or_zeros(df, "dbg_mask_rvol_hi").sum()),
        "dbg_mask_liq_hi_sum": float(_series_or_zeros(df, "dbg_mask_liq_hi").sum()),
        "dbg_exit_rules_fallback_sum": float(_series_or_zeros(df, "dbg_exit_rules_fallback").sum()),
        "beta_proxy_non_na": float(df["beta_proxy_60d"].notna().sum()) if "beta_proxy_60d" in df.columns else 0.0,
        "score_model_non_na": float(df["score_model"].notna().sum()) if "score_model" in df.columns else 0.0,
        "intraday_rs_non_na": float(df["intraday_rs"].notna().sum()) if "intraday_rs" in df.columns else 0.0,
        "intraday_pressure_non_na": float(df["intraday_pressure"].notna().sum()) if "intraday_pressure" in df.columns else 0.0,
    }


# ------------------------------------------------------------
# FEATURE PREP
# ------------------------------------------------------------

def _ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "symbol" not in out.columns:
        if "ticker" in out.columns:
            out = out.rename(columns={"ticker": "symbol"})
        elif "sym" in out.columns:
            out = out.rename(columns={"sym": "symbol"})
    if "date" not in out.columns:
        raise RuntimeError("feature matrix missing date")
    if "symbol" not in out.columns:
        raise RuntimeError("feature matrix missing symbol")
    if TARGET_COL not in out.columns:
        raise RuntimeError(f"feature matrix missing target column: {TARGET_COL}")
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in out.columns:
            raise RuntimeError(f"feature matrix missing required raw column: {col}")
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out = out.sort_values(["date", "symbol"]).reset_index(drop=True)
    return out


def _derive_regime_inputs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    prev_close = _safe_numeric(out.groupby("symbol", sort=False)["close"].shift(1))

    if "ret_1d_simple" not in out.columns:
        out["ret_1d_simple"] = _safe_numeric(out.groupby("symbol", sort=False)["close"].pct_change())
    if "gap_ret" not in out.columns:
        out["gap_ret"] = (_safe_numeric(out["open"]) / (prev_close + EPS)) - 1.0
    if "oc_body_pct" not in out.columns:
        out["oc_body_pct"] = (_safe_numeric(out["close"]) - _safe_numeric(out["open"])) / (_safe_numeric(out["open"]) + EPS)
    if "dollar_vol" not in out.columns:
        out["dollar_vol"] = _safe_numeric(out["close"]) * _safe_numeric(out["volume"])
    if "liq" not in out.columns:
        out["liq"] = np.log1p(_safe_numeric(out["dollar_vol"]).clip(lower=0.0))
    if "volume_shock" not in out.columns:
        vol_mean_20 = out.groupby("symbol", sort=False)["volume"].rolling(20, min_periods=10).mean().reset_index(level=0, drop=True)
        out["volume_shock"] = _safe_numeric(out["volume"]) / (_safe_numeric(vol_mean_20) + EPS)
    if "rel_volume_20" not in out.columns:
        out["rel_volume_20"] = _safe_numeric(out["volume_shock"])
    if "market_ret_mean" not in out.columns:
        out["market_ret_mean"] = out.groupby("date", sort=False)["ret_1d_simple"].transform(lambda s: pd.to_numeric(s, errors="coerce").mean())
    if "market_breadth" not in out.columns:
        out["market_breadth"] = out.groupby("date", sort=False)["ret_1d_simple"].transform(lambda s: pd.to_numeric(s, errors="coerce").gt(0).mean())
    if "meta_dollar_volume" not in out.columns:
        out["meta_dollar_volume"] = _safe_numeric(out["dollar_vol"])
    if "meta_price" not in out.columns:
        out["meta_price"] = _safe_numeric(out["close"])
    if "liq_rank" not in out.columns:
        out["liq_rank"] = _cs_rank_pct(out, "liq")

    out["rev1_base"] = -_cs_zscore(out, "ret_1d_simple")
    out["gap_base"] = _cs_zscore(out, "gap_ret")
    out["pressure_base"] = _cs_zscore(out, "oc_body_pct")

    if "intraday_rs" not in out.columns:
        out["intraday_rs"] = out["rev1_base"]
    if "intraday_pressure" not in out.columns:
        out["intraday_pressure"] = out["pressure_base"]
    out["rvol_z"] = _cs_zscore(out, "rel_volume_20")
    out["rvol_rank_pct"] = _cs_rank_pct(out, "rel_volume_20")
    out["liq_rank_pct"] = _cs_rank_pct(out, "liq")
    out["dbg_mask_rvol_hi"] = (out["rvol_rank_pct"] >= REGIME_HIGH_Q).astype(int)
    out["dbg_mask_liq_hi"] = (out["liq_rank_pct"] >= REGIME_HIGH_Q).astype(int)

    out["alpha_rev_hi_rvol"] = out["rev1_base"] * out["dbg_mask_rvol_hi"]
    out["alpha_gap_z_rvol"] = out["gap_base"] * out["rvol_z"]
    out["alpha_pressure_hi_liq"] = out["pressure_base"] * out["dbg_mask_liq_hi"]

    out["score_model"] = (
        ALPHA_W_REV_HI_RVOL * out["alpha_rev_hi_rvol"].fillna(0.0)
        + ALPHA_W_GAP_Z_RVOL * out["alpha_gap_z_rvol"].fillna(0.0)
        + ALPHA_W_PRESSURE_HI_LIQ * out["alpha_pressure_hi_liq"].fillna(0.0)
    )
    out["score_model_abs_rank_pct"] = out.groupby("date", sort=False)["score_model"].transform(lambda s: pd.to_numeric(s, errors="coerce").abs().rank(method="average", pct=True))
    out["fresh_dislocation_flag"] = ((out["dbg_mask_rvol_hi"] == 1) | (out["dbg_mask_liq_hi"] == 1)).astype(int)
    if "dbg_exit_rules_fallback" not in out.columns:
        out["dbg_exit_rules_fallback"] = 0
    return out


# ------------------------------------------------------------
# SIGNAL / PORTFOLIO LAYERS
# ------------------------------------------------------------

def _build_signal_layer(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = neutralize_score_cross_section(
        out,
        score_col="score_model",
        exposure_cols=["liq_rank", "beta_proxy_60d"],
        out_col="score_raw",
    )
    out = build_cross_sectional_signal(
        out,
        score_col="score_raw",
        date_col="date",
        winsor_lower_q=CS_WINSOR_LOWER_Q,
        winsor_upper_q=CS_WINSOR_UPPER_Q,
        conf_floor=CS_CONF_FLOOR,
        conf_cap=CS_CONF_CAP,
        final_score_cap=CS_FINAL_SCORE_CAP,
    )
    out = build_risk_model(
        out,
        score_col="score_final",
        date_col="date",
        symbol_col="symbol",
        beta_penalty=RISK_BETA_PENALTY,
        liq_penalty=RISK_LIQ_PENALTY,
        vol_penalty=RISK_VOL_PENALTY,
        market_regime_penalty=RISK_MARKET_REGIME_PENALTY,
        risk_floor=RISK_FLOOR,
        risk_cap=RISK_CAP,
    )
    out["score"] = pd.to_numeric(out["score_risk_adj"], errors="coerce")
    out["score_abs_rank_pct"] = out.groupby("date", sort=False)["score"].transform(lambda s: pd.to_numeric(s, errors="coerce").abs().rank(method="average", pct=True))
    return out


def _build_portfolio(test_df: pd.DataFrame) -> pd.DataFrame:
    port = apply_holding_inertia(test_df.copy(), enter_pct=ENTER_PCT, exit_pct=EXIT_PCT)
    port = apply_adaptive_exit_rules(
        port,
        side_col="side",
        score_col="score",
        peak_trail_drop_long=PEAK_TRAIL_DROP_LONG,
        peak_trail_drop_short=PEAK_TRAIL_DROP_SHORT,
        peak_trail_min_age_long=PEAK_TRAIL_MIN_AGE_LONG,
        peak_trail_min_age_short=PEAK_TRAIL_MIN_AGE_SHORT,
    )
    if "dbg_exit_rules_fallback" not in port.columns:
        port["dbg_exit_rules_fallback"] = 0
    return port


def _apply_execution_layer(port_df: pd.DataFrame) -> pd.DataFrame:
    out = port_df.copy()
    side_col = "side"
    out = apply_position_filters(out, side_col=side_col, min_price=LOW_PRICE_MIN, min_dollar_volume=LOW_DV_MIN)
    out = apply_signal_strength_sizing(out, side_col=side_col, score_col="score", out_col="side_sized", preset_name=SIZING_PRESET)
    side_col = "side_sized"
    out = normalize_gross_exposure(out, side_col=side_col, gross_target=1.0, out_col="weight")
    if APPLY_DYNAMIC_BUDGETS:
        out = attach_dynamic_side_budgets(out, input_lag_days=BUDGET_INPUT_LAG_DAYS)
        out = apply_side_budgets(out, weight_col="weight")
    else:
        out["long_budget"] = 0.50
        out["short_budget"] = 0.50
        out["budget_signal_lag_days"] = 0
    out = cap_final_weight(
        out,
        weight_col="weight",
        cap_abs_weight=FINAL_WEIGHT_CAP,
        max_adv_participation=MAX_ADV_PARTICIPATION,
        portfolio_notional=PORTFOLIO_NOTIONAL,
    )
    out = renormalize_after_caps(out, weight_col="weight", gross_target=1.0, capital_policy=CAPITAL_POLICY)
    out = cap_daily_turnover(out, weight_col="weight", max_daily_turnover=MAX_DAILY_TURNOVER)
    out = attach_execution_costs(
        out,
        weight_col="weight",
        fee_bps=FEE_BPS,
        slippage_bps=BASE_SLIPPAGE_BPS,
        borrow_bps_daily=BORROW_BPS_DAILY,
        spread_bps=SPREAD_BPS,
        impact_bps=IMPACT_BPS,
        low_price_penalty_bps=LOW_PRICE_PENALTY_BPS,
        htb_borrow_bps_daily=HTB_BORROW_BPS_DAILY,
        max_participation=MAX_ADV_PARTICIPATION,
        portfolio_notional=PORTFOLIO_NOTIONAL,
    )
    if "dbg_exit_rules_fallback" not in out.columns:
        out["dbg_exit_rules_fallback"] = 0
    return out


# ------------------------------------------------------------
# WF RUN
# ------------------------------------------------------------

def _run_walkforward(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[FoldArtifacts]]:
    splits = build_walkforward_splits(
        df["date"],
        train_days=TRAIN_DAYS,
        test_days=TEST_DAYS,
        step_days=STEP_DAYS,
        purge_days=PURGE_DAYS,
        embargo_days=EMBARGO_DAYS,
    )
    print_split_summary(splits)

    fold_artifacts: List[FoldArtifacts] = []
    all_test_daily: List[pd.DataFrame] = []

    for sp in splits:
        print(f"[WF][regime_pack][FOLD {sp.fold_id}] start")
        train_df = _slice_by_date(df, sp.train_start, sp.train_end)
        test_df = _slice_by_date(df, sp.test_start, sp.test_end)
        print(f"[WF][regime_pack][FOLD {sp.fold_id}] train_rows={len(train_df)} test_rows={len(test_df)}")
        if len(test_df) < MIN_TEST_ROWS:
            print(f"[WF][regime_pack][FOLD {sp.fold_id}][SKIP] test_rows<{MIN_TEST_ROWS}")
            continue

        scored_test = _build_signal_layer(test_df)
        port_test = _build_portfolio(scored_test)
        port_test = _apply_execution_layer(port_test)
        daily_test = evaluate_long_short(port_test, target_col=TARGET_COL)
        daily_test["fold_id"] = sp.fold_id
        daily_test["model"] = "regime_pack_fixed"

        port_test.to_parquet(DIAG_DIR / f"portfolio__regime_pack_fixed__fold{sp.fold_id}.parquet", index=False)
        fold_summary = summarize_daily_returns(daily_test)
        print_summary(f"[WF][regime_pack][FOLD {sp.fold_id}][SUMMARY]", fold_summary)

        fold_artifacts.append(FoldArtifacts(fold_id=sp.fold_id, daily=daily_test, portfolio=port_test))
        all_test_daily.append(daily_test)

    if not all_test_daily:
        raise RuntimeError("[WF][regime_pack] no valid folds survived")

    overall = pd.concat(all_test_daily, axis=0, ignore_index=True)
    overall = overall.sort_values(["date", "fold_id"]).reset_index(drop=True)
    overall_summary = summarize_daily_returns(overall)
    print_summary("[WF][regime_pack][OVERALL]", overall_summary)
    print("[WF][regime_pack][TAIL]")
    print(overall.tail(10).to_string(index=False))
    return overall, fold_artifacts


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main() -> int:
    _enable_line_buffering()
    print(f"[CFG] feature_file={FEATURE_FILE}")
    print(f"[CFG] target_col={TARGET_COL}")
    print(f"[CFG] train_days={TRAIN_DAYS} test_days={TEST_DAYS} step_days={STEP_DAYS} purge_days={PURGE_DAYS} embargo_days={EMBARGO_DAYS}")
    print(f"[CFG] sizing_preset={SIZING_PRESET} final_weight_cap={FINAL_WEIGHT_CAP} max_daily_turnover={MAX_DAILY_TURNOVER}")
    print(f"[CFG] regime_high_q={REGIME_HIGH_Q} regime_low_q={REGIME_LOW_Q}")
    print(f"[CFG] alpha_weights=rev_hi_rvol:{ALPHA_W_REV_HI_RVOL} gap_z_rvol:{ALPHA_W_GAP_Z_RVOL} pressure_hi_liq:{ALPHA_W_PRESSURE_HI_LIQ}")

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(f"Feature file not found: {FEATURE_FILE}")

    df = pd.read_parquet(FEATURE_FILE)
    if df.empty:
        raise RuntimeError("Loaded feature matrix is empty")

    df = _ensure_required_columns(df)
    df = _derive_regime_inputs(df)
    df = add_beta_proxy(df, lookback=60)

    needed = [
        "date", "symbol", TARGET_COL,
        "market_ret_mean", "market_breadth", "meta_dollar_volume", "meta_price", "liq_rank", "beta_proxy_60d",
        "intraday_rs", "intraday_pressure",
        "alpha_rev_hi_rvol", "alpha_gap_z_rvol", "alpha_pressure_hi_liq", "score_model",
        "fresh_dislocation_flag", "score_model_abs_rank_pct", "dbg_exit_rules_fallback",
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise RuntimeError(f"main: missing columns after derivation: {missing}")

    dbg = _debug_counter_summary(df)
    print("[DEBUG] " + " ".join(f"{k}={v}" for k, v in dbg.items()))

    overall, folds = _run_walkforward(df)
    overall.to_parquet(DIAG_DIR / "wf__regime_pack_fixed__overall.parquet", index=False)
    print(f"[ARTIFACT] {DIAG_DIR / 'wf__regime_pack_fixed__overall.parquet'}")
    print(f"[FINAL] folds={len(folds)} rows={len(overall)}")
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 1
    _press_enter_exit(rc)
