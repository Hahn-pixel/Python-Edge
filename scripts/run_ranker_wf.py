from __future__ import annotations

import os
import traceback
from pathlib import Path

import pandas as pd

from python_edge.execution.cost_model import attach_execution_costs
from python_edge.model.conditional_factors import add_conditional_factors
from python_edge.model.cs_normalize import cs_zscore
from python_edge.model.neutralize import add_beta_proxy, neutralize_score_cross_section
from python_edge.model.ranker_linear import apply_linear_score, fit_corr_weights
from python_edge.portfolio.construct import build_long_short_portfolio
from python_edge.portfolio.holding_inertia import apply_holding_inertia
from python_edge.portfolio.position_limits import apply_position_filters, cap_single_name_weight, normalize_gross_exposure
from python_edge.portfolio.regime_allocation import build_regime_aware_long_short_portfolio
from python_edge.portfolio.signal_sizing import apply_signal_strength_sizing
from python_edge.portfolio.turnover_control import cap_daily_turnover
from python_edge.wf.evaluate_ranker import evaluate_long_short, print_summary, summarize_daily_returns
from python_edge.wf.splits import build_walkforward_splits, print_split_summary


FEATURE_FILE = Path("data") / "features" / "feature_matrix_v1.parquet"
TARGET_COL = "target_fwd_ret_1d"
TRAIN_DAYS = 252
TEST_DAYS = 63
STEP_DAYS = 63
TOP_PCT = 0.10
ENTER_PCT = 0.10
EXIT_PCT = 0.20
MIN_TRAIN_ROWS = 500
MIN_TEST_ROWS = 100
MAX_DAILY_TURNOVER = float(os.getenv("MAX_DAILY_TURNOVER", "0.60"))

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

ABLATIONS: dict[str, dict[str, object]] = {
    "intraday_core_only": {
        "features": INTRADAY_CORE_FEATURES,
        "portfolio_mode": "plain_inertia",
        "neutralize": False,
        "sizing": False,
        "sizing_preset": None,
    },
    "intraday_plus_breadth_regime": {
        "features": INTRADAY_CORE_FEATURES + REGIME_BREADTH_FEATURES,
        "portfolio_mode": "regime_inertia",
        "neutralize": False,
        "sizing": False,
        "sizing_preset": None,
    },
    "full_regime_stack": {
        "features": FULL_FEATURE_STACK,
        "portfolio_mode": "regime_inertia",
        "neutralize": False,
        "sizing": False,
        "sizing_preset": None,
    },
    "full_regime_stack_neutralized": {
        "features": FULL_FEATURE_STACK,
        "portfolio_mode": "regime_inertia",
        "neutralize": True,
        "sizing": False,
        "sizing_preset": None,
    },
    "full_regime_stack_neutralized_sized_baseline": {
        "features": FULL_FEATURE_STACK,
        "portfolio_mode": "regime_inertia",
        "neutralize": True,
        "sizing": True,
        "sizing_preset": "baseline",
    },
    "full_regime_stack_neutralized_sized_aggressive": {
        "features": FULL_FEATURE_STACK,
        "portfolio_mode": "regime_inertia",
        "neutralize": True,
        "sizing": True,
        "sizing_preset": "aggressive",
    },
    "full_regime_stack_neutralized_sized_conservative": {
        "features": FULL_FEATURE_STACK,
        "portfolio_mode": "regime_inertia",
        "neutralize": True,
        "sizing": True,
        "sizing_preset": "conservative",
    },
    "full_regime_stack_neutralized_sized_barbell": {
        "features": FULL_FEATURE_STACK,
        "portfolio_mode": "regime_inertia",
        "neutralize": True,
        "sizing": True,
        "sizing_preset": "barbell",
    },
}



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
    ] + features + zcols
    missing = [c for c in needed if c not in out.columns]
    if missing:
        raise RuntimeError(f"_prepare_base_frame: missing columns: {missing}")
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



def _build_portfolio(test_df: pd.DataFrame, portfolio_mode: str, score_col: str) -> pd.DataFrame:
    temp = test_df.copy()
    if score_col != "score":
        temp["score"] = pd.to_numeric(temp[score_col], errors="coerce")
    if portfolio_mode == "plain":
        return build_long_short_portfolio(temp, top_pct=TOP_PCT)
    if portfolio_mode == "regime":
        return build_regime_aware_long_short_portfolio(temp)
    if portfolio_mode == "plain_inertia":
        return apply_holding_inertia(temp, enter_pct=ENTER_PCT, exit_pct=EXIT_PCT)
    if portfolio_mode == "regime_inertia":
        reg = build_regime_aware_long_short_portfolio(temp)
        reg = reg.drop(columns=[c for c in ["rank", "side"] if c in reg.columns])
        return apply_holding_inertia(reg, enter_pct=ENTER_PCT, exit_pct=EXIT_PCT)
    raise RuntimeError(f"Unknown portfolio_mode={portfolio_mode!r}")



def _apply_execution_layer(
    port_df: pd.DataFrame,
    use_signal_sizing: bool,
    sizing_preset: str | None,
) -> pd.DataFrame:
    out = port_df.copy()
    side_col = "side"

    out = apply_position_filters(out, side_col=side_col, min_price=5.0, min_dollar_volume=1_000_000.0)

    if use_signal_sizing:
        preset_name = str(sizing_preset or "baseline")
        out = apply_signal_strength_sizing(
            out,
            side_col=side_col,
            score_col="score",
            out_col="side_sized",
            preset_name=preset_name,
        )
        side_col = "side_sized"
    else:
        out["sizing_preset"] = "none"

    out = cap_single_name_weight(out, side_col=side_col, cap_abs_weight=0.08)
    out = normalize_gross_exposure(out, side_col=side_col, gross_target=1.0)
    out = cap_daily_turnover(out, weight_col="weight", max_daily_turnover=MAX_DAILY_TURNOVER)
    out = attach_execution_costs(out, weight_col="weight", fee_bps=1.0, slippage_bps=2.0, borrow_bps_daily=1.0)
    return out



def _run_one_model(
    model_name: str,
    raw_df: pd.DataFrame,
    features: list[str],
    portfolio_mode: str,
    neutralize: bool,
    sizing: bool,
    sizing_preset: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(
        f"[WF][{model_name}] portfolio_mode={portfolio_mode} "
        f"neutralize={neutralize} sizing={sizing} sizing_preset={sizing_preset} "
        f"max_daily_turnover={MAX_DAILY_TURNOVER} features={features}"
    )
    df = _prepare_base_frame(raw_df, features)
    splits = build_walkforward_splits(df["date"], train_days=TRAIN_DAYS, test_days=TEST_DAYS, step_days=STEP_DAYS)
    print_split_summary(splits)

    all_test_daily: list[pd.DataFrame] = []
    all_weight_frames: list[pd.DataFrame] = []

    for sp in splits:
        print(f"[WF][{model_name}][FOLD {sp.fold_id}] start")
        train_df = _slice_by_date(df, sp.train_start, sp.train_end)
        test_df = _slice_by_date(df, sp.test_start, sp.test_end)
        print(f"[WF][{model_name}][FOLD {sp.fold_id}] train_rows={len(train_df)} test_rows={len(test_df)}")
        if len(train_df) < MIN_TRAIN_ROWS:
            raise RuntimeError(f"[WF][{model_name}][FOLD {sp.fold_id}] train_rows too small: {len(train_df)}")
        if len(test_df) < MIN_TEST_ROWS:
            raise RuntimeError(f"[WF][{model_name}][FOLD {sp.fold_id}] test_rows too small: {len(test_df)}")

        zcols = [f"z_{f}" for f in features]
        fit = fit_corr_weights(train_df=train_df, zcols=zcols, target_col=TARGET_COL)
        all_weight_frames.append(_weights_to_frame(model_name, sp.fold_id, fit.weights))

        scored_test = apply_linear_score(test_df, fit=fit, out_col="score")
        score_col = "score"
        if neutralize:
            scored_test = neutralize_score_cross_section(
                scored_test,
                score_col="score",
                exposure_cols=["liq_rank", "beta_proxy_60d"],
                out_col="score_neutral",
            )
            score_col = "score_neutral"

        port_test = _build_portfolio(scored_test, portfolio_mode=portfolio_mode, score_col=score_col)
        if score_col != "score":
            port_test["score"] = pd.to_numeric(port_test[score_col], errors="coerce")

        port_test = _apply_execution_layer(
            port_test,
            use_signal_sizing=sizing,
            sizing_preset=sizing_preset,
        )

        daily_test = evaluate_long_short(port_test, target_col=TARGET_COL)
        daily_test["fold_id"] = sp.fold_id
        daily_test["model"] = model_name

        fold_summary = summarize_daily_returns(daily_test)
        print_summary(f"[WF][{model_name}][FOLD {sp.fold_id}][SUMMARY]", fold_summary)
        all_test_daily.append(daily_test)

    if not all_test_daily:
        raise RuntimeError(f"[WF][{model_name}] no test daily results were produced")

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
    print(f"[CFG] feature_file={FEATURE_FILE}")
    print(f"[CFG] target_col={TARGET_COL}")
    print(f"[CFG] train_days={TRAIN_DAYS} test_days={TEST_DAYS} step_days={STEP_DAYS} top_pct={TOP_PCT}")
    print(f"[CFG] enter_pct={ENTER_PCT} exit_pct={EXIT_PCT}")
    print(f"[CFG] max_daily_turnover={MAX_DAILY_TURNOVER}")

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(f"Feature file not found: {FEATURE_FILE}")

    print("[LOAD] feature matrix")
    raw_df = pd.read_parquet(FEATURE_FILE)
    if raw_df.empty:
        raise RuntimeError("Loaded feature matrix is empty")

    model_summaries: list[dict[str, float | str | bool]] = []
    for model_name, cfg in ABLATIONS.items():
        features = list(cfg["features"])
        portfolio_mode = str(cfg["portfolio_mode"])
        neutralize = bool(cfg["neutralize"])
        sizing = bool(cfg["sizing"])
        sizing_preset = cfg.get("sizing_preset")
        overall, _weights_df = _run_one_model(
            model_name,
            raw_df,
            features,
            portfolio_mode,
            neutralize,
            sizing,
            None if sizing_preset is None else str(sizing_preset),
        )
        summary = summarize_daily_returns(overall)
        model_summaries.append(
            {
                "model": model_name,
                "portfolio_mode": portfolio_mode,
                "neutralize": neutralize,
                "sizing": sizing,
                "sizing_preset": "none" if sizing_preset is None else str(sizing_preset),
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
        try:
            input("Press Enter to exit...")
        except EOFError:
            pass
    raise SystemExit(rc)