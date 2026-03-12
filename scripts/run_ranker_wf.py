from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

import pandas as pd

from python_edge.execution.cost_model import attach_execution_costs
from python_edge.model.conditional_factors import add_conditional_factors
from python_edge.model.cross_sectional_signal import build_cross_sectional_signal
from python_edge.model.cs_normalize import cs_zscore
from python_edge.model.neutralize import add_beta_proxy, neutralize_score_cross_section
from python_edge.model.ranker_linear import apply_linear_score, fit_corr_weights
from python_edge.model.risk_model import build_risk_model
from python_edge.portfolio.budget_allocation import attach_dynamic_side_budgets, apply_side_budgets
from python_edge.portfolio.construct import build_long_short_portfolio
from python_edge.portfolio.exit_rules import apply_adaptive_exit_rules
from python_edge.portfolio.holding_inertia import apply_holding_inertia
from python_edge.portfolio.position_limits import apply_position_filters, cap_final_weight, normalize_gross_exposure, renormalize_after_caps
from python_edge.portfolio.regime_allocation import build_regime_aware_long_short_portfolio
from python_edge.portfolio.signal_sizing import apply_signal_strength_sizing
from python_edge.portfolio.turnover_control import cap_daily_turnover
from python_edge.wf.evaluate_ranker import evaluate_long_short, print_summary, summarize_daily_returns
from python_edge.wf.splits import build_walkforward_splits, print_split_summary


FEATURE_FILE = Path("data") / "features" / "feature_matrix_v1.parquet"
DIAG_DIR = Path("data") / "diagnostics"
DIAG_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "target_fwd_ret_1d"
TRAIN_DAYS = int(os.getenv("WF_TRAIN_DAYS", "252"))
TEST_DAYS = int(os.getenv("WF_TEST_DAYS", "63"))
STEP_DAYS = int(os.getenv("WF_STEP_DAYS", "63"))
PURGE_DAYS = int(os.getenv("WF_PURGE_DAYS", "5"))
EMBARGO_DAYS = int(os.getenv("WF_EMBARGO_DAYS", "5"))
TOP_PCT = float(os.getenv("TOP_PCT", "0.10"))
ENTER_PCT = float(os.getenv("ENTER_PCT", "0.10"))
EXIT_PCT = float(os.getenv("EXIT_PCT", "0.20"))
MIN_TRAIN_ROWS = int(os.getenv("MIN_TRAIN_ROWS", "500"))
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
PAUSE_ON_EXIT_ENV = str(os.getenv("PAUSE_ON_EXIT", "auto")).strip().lower()

INTRADAY_CORE_FEATURES = [
    "intraday_rs",
    "volume_shock",
    "intraday_pressure",
    "intraday_rs_x_volume_shock",
    "intraday_pressure_x_volume_shock",
    "liq_rank_x_intraday_rs",
]
REGIME_BREADTH_FEATURES = [
    "cond_breadth_trend_intraday_rs",
    "cond_breadth_trend_mom_compression",
    "cond_breadth_range_str",
    "cond_breadth_weak_overnight",
]
RECOVERED_DAILY_FEATURES = [
    "cond_momentum_liq_trend",
    "cond_str_weak_breadth",
    "cond_overnight_trend_follow",
    "cond_vol_compression_liq_breakout",
]
RISK_FILTER_FEATURES = [
    "cond_ivol_lowliq_penalty",
]
FULL_FEATURE_STACK = INTRADAY_CORE_FEATURES + REGIME_BREADTH_FEATURES + RECOVERED_DAILY_FEATURES + RISK_FILTER_FEATURES

BASELINE_MODEL = "full_regime_stack_neutralized_sized_barbell_peaktrail_priority_exec"

ABLATIONS: dict[str, dict[str, object]] = {
    "full_regime_stack_neutralized_sized_barbell_peaktrail_priority_exec": {
        "features": FULL_FEATURE_STACK,
        "portfolio_mode": "regime_inertia",
        "neutralize": True,
        "sizing": True,
        "sizing_preset": "barbell",
        "adaptive_exits": True,
    },
    "diag_barbell_no_exits": {
        "features": FULL_FEATURE_STACK,
        "portfolio_mode": "regime_inertia",
        "neutralize": True,
        "sizing": True,
        "sizing_preset": "barbell",
        "adaptive_exits": False,
    },
}



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



def _prepare_base_frame(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    out = df.copy()
    if "symbol" not in out.columns:
        if "ticker" in out.columns:
            out = out.rename(columns={"ticker": "symbol"})
        elif "sym" in out.columns:
            out = out.rename(columns={"sym": "symbol"})
    if "date" not in out.columns:
        raise RuntimeError("_prepare_base_frame: missing date")
    if "symbol" not in out.columns:
        raise RuntimeError("_prepare_base_frame: missing symbol before transforms")

    out = add_conditional_factors(out)
    out = add_beta_proxy(out, lookback=60)
    out = cs_zscore(out, features)

    zcols = [f"z_{f}" for f in features]
    needed = [
        "date",
        "symbol",
        TARGET_COL,
        "market_breadth",
        "meta_dollar_volume",
        "meta_price",
        "liq_rank",
        "beta_proxy_60d",
        "intraday_rs",
        "volume_shock",
        "intraday_pressure",
    ] + features + zcols
    missing = [c for c in needed if c not in out.columns]
    if missing:
        raise RuntimeError(f"_prepare_base_frame: missing columns: {missing}")

    out = out.sort_values(["date", "symbol"]).reset_index(drop=True)
    out["date"] = pd.to_datetime(out["date"])
    return out



def _slice_by_date(df: pd.DataFrame, start_date: object, end_date: object) -> pd.DataFrame:
    return df.loc[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()



def _weights_to_frame(model_name: str, fold_id: int, weights: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "model": model_name,
            "fold_id": fold_id,
            "feature": feature_name,
            "weight": float(weight_value),
            "abs_weight": abs(float(weight_value)),
            "sign": 0 if float(weight_value) == 0.0 else (1 if float(weight_value) > 0.0 else -1),
        }
        for feature_name, weight_value in weights.items()
    ])



def _print_weight_stability(model_name: str, weights_df: pd.DataFrame) -> None:
    if weights_df.empty:
        print(f"[WF][{model_name}][WEIGHTS][WARN] empty weights_df")
        return
    grouped = weights_df.groupby("feature", as_index=False).agg(
        folds=("fold_id", "nunique"),
        mean_weight=("weight", "mean"),
        std_weight=("weight", "std"),
        mean_abs_weight=("abs_weight", "mean"),
        pos_folds=("sign", lambda s: int((s > 0).sum())),
        neg_folds=("sign", lambda s: int((s < 0).sum())),
        zero_folds=("sign", lambda s: int((s == 0).sum())),
    )
    grouped = grouped.sort_values(["mean_abs_weight", "feature"], ascending=[False, True]).reset_index(drop=True)
    print(f"[WF][{model_name}][WEIGHTS][STABILITY]")
    print(grouped.to_string(index=False))



def _build_signal_layer(df: pd.DataFrame, score_col: str) -> tuple[pd.DataFrame, str]:
    out = df.copy()
    out["score_raw"] = pd.to_numeric(out[score_col], errors="coerce")
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
    return out, "score_final"



def _build_portfolio(test_df: pd.DataFrame, portfolio_mode: str, score_col: str, adaptive_exits: bool) -> pd.DataFrame:
    temp = test_df.copy()
    if score_col != "score":
        temp["score"] = pd.to_numeric(temp[score_col], errors="coerce")

    if portfolio_mode == "plain":
        port = build_long_short_portfolio(temp, top_pct=TOP_PCT)
    elif portfolio_mode == "regime":
        port = build_regime_aware_long_short_portfolio(temp)
    elif portfolio_mode == "plain_inertia":
        port = apply_holding_inertia(temp, enter_pct=ENTER_PCT, exit_pct=EXIT_PCT)
    elif portfolio_mode == "regime_inertia":
        reg = build_regime_aware_long_short_portfolio(temp)
        reg = reg.drop(columns=[c for c in ["rank", "side"] if c in reg.columns])
        port = apply_holding_inertia(reg, enter_pct=ENTER_PCT, exit_pct=EXIT_PCT)
    else:
        raise RuntimeError(f"Unknown portfolio_mode={portfolio_mode!r}")

    if adaptive_exits:
        port = apply_adaptive_exit_rules(
            port,
            side_col="side",
            score_col="score",
            peak_trail_drop_long=PEAK_TRAIL_DROP_LONG,
            peak_trail_drop_short=PEAK_TRAIL_DROP_SHORT,
            peak_trail_min_age_long=PEAK_TRAIL_MIN_AGE_LONG,
            peak_trail_min_age_short=PEAK_TRAIL_MIN_AGE_SHORT,
        )
    else:
        port["exit_any"] = 0
        port["exit_time_stop_long"] = 0
        port["exit_time_stop_short"] = 0
        port["exit_rank_decay_long"] = 0
        port["exit_rank_decay_short"] = 0
        port["exit_score_peak_long"] = 0
        port["exit_score_peak_short"] = 0
        port["exit_signal_flip"] = 0
        port["exit_signal_flat"] = 0

    return port



def _apply_execution_layer(port_df: pd.DataFrame, use_signal_sizing: bool, sizing_preset: str | None) -> pd.DataFrame:
    out = port_df.copy()
    side_col = "side"

    out = apply_position_filters(out, side_col=side_col, min_price=LOW_PRICE_MIN, min_dollar_volume=LOW_DV_MIN)

    if use_signal_sizing:
        preset_name = str(sizing_preset or "baseline")
        out = apply_signal_strength_sizing(out, side_col=side_col, score_col="score", out_col="side_sized", preset_name=preset_name)
        side_col = "side_sized"
    else:
        out["sizing_preset"] = "none"
        out["conviction_mult"] = 1.0
        out["conviction_bucket"] = "flat"

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
    return out



def _run_one_model(
    model_name: str,
    raw_df: pd.DataFrame,
    features: list[str],
    portfolio_mode: str,
    neutralize: bool,
    sizing: bool,
    sizing_preset: str | None,
    adaptive_exits: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(
        f"[WF][{model_name}] portfolio_mode={portfolio_mode} neutralize={neutralize} sizing={sizing} "
        f"sizing_preset={sizing_preset} adaptive_exits={adaptive_exits} max_daily_turnover={MAX_DAILY_TURNOVER} "
        f"final_weight_cap={FINAL_WEIGHT_CAP} purge_days={PURGE_DAYS} embargo_days={EMBARGO_DAYS} capital_policy={CAPITAL_POLICY}"
    )
    df = _prepare_base_frame(raw_df, features)
    splits = build_walkforward_splits(
        df["date"],
        train_days=TRAIN_DAYS,
        test_days=TEST_DAYS,
        step_days=STEP_DAYS,
        purge_days=PURGE_DAYS,
        embargo_days=EMBARGO_DAYS,
    )
    print_split_summary(splits)

    all_test_daily: list[pd.DataFrame] = []
    all_weight_frames: list[pd.DataFrame] = []

    for sp in splits:
        print(f"[WF][{model_name}][FOLD {sp.fold_id}] start")
        train_df = _slice_by_date(df, sp.train_start, sp.train_end)
        test_df = _slice_by_date(df, sp.test_start, sp.test_end)
        print(f"[WF][{model_name}][FOLD {sp.fold_id}] train_rows={len(train_df)} test_rows={len(test_df)}")

        if len(train_df) < MIN_TRAIN_ROWS:
            print(f"[WF][{model_name}][FOLD {sp.fold_id}][SKIP] train_rows<{MIN_TRAIN_ROWS}")
            continue
        if len(test_df) < MIN_TEST_ROWS:
            print(f"[WF][{model_name}][FOLD {sp.fold_id}][SKIP] test_rows<{MIN_TEST_ROWS}")
            continue

        zcols = [f"z_{f}" for f in features]
        fit = fit_corr_weights(train_df=train_df, zcols=zcols, target_col=TARGET_COL)
        all_weight_frames.append(_weights_to_frame(model_name, sp.fold_id, fit.weights))

        scored_test = apply_linear_score(test_df, fit=fit, out_col="score_model")
        score_col = "score_model"
        if neutralize:
            scored_test = neutralize_score_cross_section(
                scored_test,
                score_col="score_model",
                exposure_cols=["liq_rank", "beta_proxy_60d"],
                out_col="score_raw",
            )
            score_col = "score_raw"
        else:
            scored_test["score_raw"] = pd.to_numeric(scored_test[score_col], errors="coerce")
            score_col = "score_raw"

        scored_test, score_col = _build_signal_layer(scored_test, score_col=score_col)
        scored_test = build_risk_model(
            scored_test,
            score_col=score_col,
            date_col="date",
            symbol_col="symbol",
            beta_penalty=RISK_BETA_PENALTY,
            liq_penalty=RISK_LIQ_PENALTY,
            vol_penalty=RISK_VOL_PENALTY,
            market_regime_penalty=RISK_MARKET_REGIME_PENALTY,
            risk_floor=RISK_FLOOR,
            risk_cap=RISK_CAP,
        )
        scored_test["score"] = pd.to_numeric(scored_test["score_risk_adj"], errors="coerce")

        port_test = _build_portfolio(scored_test, portfolio_mode=portfolio_mode, score_col="score", adaptive_exits=adaptive_exits)
        port_test = _apply_execution_layer(port_test, use_signal_sizing=sizing, sizing_preset=sizing_preset)
        daily_test = evaluate_long_short(port_test, target_col=TARGET_COL)
        daily_test["fold_id"] = sp.fold_id
        daily_test["model"] = model_name

        safe_model_name = model_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
        port_test.to_parquet(DIAG_DIR / f"portfolio__{safe_model_name}__fold{sp.fold_id}.parquet", index=False)

        fold_summary = summarize_daily_returns(daily_test)
        print_summary(f"[WF][{model_name}][FOLD {sp.fold_id}][SUMMARY]", fold_summary)
        all_test_daily.append(daily_test)

    if not all_test_daily:
        raise RuntimeError(f"[WF][{model_name}] no valid folds survived")

    overall = pd.concat(all_test_daily, axis=0, ignore_index=True)
    overall = overall.sort_values(["date", "fold_id"]).reset_index(drop=True)
    overall_summary = summarize_daily_returns(overall)
    print_summary(f"[WF][{model_name}][OVERALL]", overall_summary)

    weights_df = pd.concat(all_weight_frames, axis=0, ignore_index=True) if all_weight_frames else pd.DataFrame()
    _print_weight_stability(model_name, weights_df)

    print(f"[WF][{model_name}][TAIL]")
    print(overall.tail(10).to_string(index=False))
    return overall, weights_df



def main() -> int:
    _enable_line_buffering()
    print(f"[CFG] feature_file={FEATURE_FILE}")
    print(f"[CFG] target_col={TARGET_COL}")
    print(
        f"[CFG] train_days={TRAIN_DAYS} test_days={TEST_DAYS} step_days={STEP_DAYS} purge_days={PURGE_DAYS} embargo_days={EMBARGO_DAYS}"
    )
    print(f"[CFG] top_pct={TOP_PCT} enter_pct={ENTER_PCT} exit_pct={EXIT_PCT}")
    print(
        f"[CFG] max_daily_turnover={MAX_DAILY_TURNOVER} final_weight_cap={FINAL_WEIGHT_CAP} capital_policy={CAPITAL_POLICY} "
        f"max_adv_participation={MAX_ADV_PARTICIPATION} portfolio_notional={PORTFOLIO_NOTIONAL}"
    )
    print(
        f"[CFG] dynamic_budgets={int(APPLY_DYNAMIC_BUDGETS)} budget_input_lag_days={BUDGET_INPUT_LAG_DAYS} "
        f"fee_bps={FEE_BPS} spread_bps={SPREAD_BPS} impact_bps={IMPACT_BPS} borrow_bps_daily={BORROW_BPS_DAILY}"
    )
    print(
        f"[CFG] peak_trail_drop_long={PEAK_TRAIL_DROP_LONG} peak_trail_drop_short={PEAK_TRAIL_DROP_SHORT} "
        f"peak_trail_min_age_long={PEAK_TRAIL_MIN_AGE_LONG} peak_trail_min_age_short={PEAK_TRAIL_MIN_AGE_SHORT}"
    )
    print(
        f"[CFG] cs_winsor_lower_q={CS_WINSOR_LOWER_Q} cs_winsor_upper_q={CS_WINSOR_UPPER_Q} "
        f"cs_conf_floor={CS_CONF_FLOOR} cs_conf_cap={CS_CONF_CAP} cs_final_score_cap={CS_FINAL_SCORE_CAP}"
    )
    print(
        f"[CFG] risk_beta_penalty={RISK_BETA_PENALTY} risk_liq_penalty={RISK_LIQ_PENALTY} "
        f"risk_vol_penalty={RISK_VOL_PENALTY} risk_market_regime_penalty={RISK_MARKET_REGIME_PENALTY} "
        f"risk_floor={RISK_FLOOR} risk_cap={RISK_CAP}"
    )
    print(f"[CFG] pause_on_exit={PAUSE_ON_EXIT_ENV}")

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(f"Feature file not found: {FEATURE_FILE}")

    raw_df = pd.read_parquet(FEATURE_FILE)
    if raw_df.empty:
        raise RuntimeError("Loaded feature matrix is empty")

    model_summaries: list[dict[str, float | str | bool]] = []
    for model_name, cfg in ABLATIONS.items():
        overall, _weights_df = _run_one_model(
            model_name=model_name,
            raw_df=raw_df,
            features=list(cfg["features"]),
            portfolio_mode=str(cfg["portfolio_mode"]),
            neutralize=bool(cfg["neutralize"]),
            sizing=bool(cfg["sizing"]),
            sizing_preset=None if cfg["sizing_preset"] is None else str(cfg["sizing_preset"]),
            adaptive_exits=bool(cfg["adaptive_exits"]),
        )
        summary = summarize_daily_returns(overall)
        model_summaries.append(
            {
                "model": model_name,
                "portfolio_mode": str(cfg["portfolio_mode"]),
                "neutralize": bool(cfg["neutralize"]),
                "sizing": bool(cfg["sizing"]),
                "sizing_preset": "none" if cfg["sizing_preset"] is None else str(cfg["sizing_preset"]),
                "adaptive_exits": bool(cfg["adaptive_exits"]),
                "days": int(summary["days"]),
                "avg_daily_ret": float(summary["avg_daily_ret"]),
                "std_daily_ret": float(summary["std_daily_ret"]),
                "win_rate_days": float(summary["win_rate_days"]),
                "cum_ret": float(summary["cum_ret"]),
                "avg_raw_turnover": float(summary.get("avg_raw_turnover", 0.0)),
                "avg_turnover": float(summary.get("avg_turnover", 0.0)),
                "cap_hit_rate": float(summary.get("cap_hit_rate", 0.0)),
                "avg_gross_ret": float(summary.get("avg_gross_ret", 0.0)),
                "avg_trading_costs": float(summary.get("avg_trading_costs", 0.0)),
                "avg_borrow_costs": float(summary.get("avg_borrow_costs", 0.0)),
                "avg_costs": float(summary.get("avg_costs", 0.0)),
                "avg_long_gross_ret": float(summary.get("avg_long_gross_ret", 0.0)),
                "avg_short_gross_ret": float(summary.get("avg_short_gross_ret", 0.0)),
                "avg_long_costs": float(summary.get("avg_long_costs", 0.0)),
                "avg_short_costs": float(summary.get("avg_short_costs", 0.0)),
                "avg_gross_long_exposure": float(summary.get("avg_gross_long_exposure", 0.0)),
                "avg_gross_short_exposure": float(summary.get("avg_gross_short_exposure", 0.0)),
                "avg_cash_weight": float(summary.get("avg_cash_weight", 0.0)),
                "avg_deployed_gross": float(summary.get("avg_deployed_gross", 0.0)),
                "avg_execution_participation": float(summary.get("avg_execution_participation", 0.0)),
                "participation_limit_hit_rate": float(summary.get("participation_limit_hit_rate", 0.0)),
                "avg_hold_days": float(summary.get("avg_hold_days", 0.0)),
                "exit_rate": float(summary.get("exit_rate", 0.0)),
                "avg_score_conf": float(summary.get("avg_score_conf", 0.0)),
                "avg_cs_dispersion": float(summary.get("avg_cs_dispersion", 0.0)),
                "avg_cs_top_bottom_spread": float(summary.get("avg_cs_top_bottom_spread", 0.0)),
                "avg_cs_signal_breadth": float(summary.get("avg_cs_signal_breadth", 0.0)),
                "avg_cs_nonzero_frac": float(summary.get("avg_cs_nonzero_frac", 0.0)),
                "avg_cs_signal_count": float(summary.get("avg_cs_signal_count", 0.0)),
                "avg_cs_signal_quality_flag": float(summary.get("avg_cs_signal_quality_flag", 0.0)),
                "avg_score_abs_rank_pct": float(summary.get("avg_score_abs_rank_pct", 0.0)),
                "avg_risk_unit": float(summary.get("avg_risk_unit", 0.0)),
                "avg_score_risk_adj": float(summary.get("avg_score_risk_adj", 0.0)),
                "avg_alpha_to_risk": float(summary.get("avg_alpha_to_risk", 0.0)),
                "avg_risk_penalty_rate": float(summary.get("avg_risk_penalty_rate", 0.0)),
                "avg_risk_beta_rank": float(summary.get("avg_risk_beta_rank", 0.0)),
                "avg_risk_vol_rank": float(summary.get("avg_risk_vol_rank", 0.0)),
                "avg_risk_liq_penalty": float(summary.get("avg_risk_liq_penalty", 0.0)),
                "avg_risk_market_penalty": float(summary.get("avg_risk_market_penalty", 0.0)),
                "avg_risk_quality_flag": float(summary.get("avg_risk_quality_flag", 0.0)),
            }
        )

    summary_df = pd.DataFrame(model_summaries).sort_values(["cum_ret", "avg_daily_ret"], ascending=[False, False]).reset_index(drop=True)
    print("[WF][ABLATION][SUMMARY]")
    print(summary_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except Exception:
        print("[ERROR] Unhandled exception:")
        print()
        traceback.print_exc()
        rc = 1
    finally:
        if _should_pause_on_exit():
            try:
                input("Press Enter to exit...")
            except EOFError:
                pass
    raise SystemExit(rc)