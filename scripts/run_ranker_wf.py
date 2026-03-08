from __future__ import annotations

import traceback
from pathlib import Path

import pandas as pd

from python_edge.model.conditional_factors import CONDITIONAL_FEATURE_COLS, add_conditional_factors
from python_edge.model.cs_normalize import cs_zscore
from python_edge.model.ranker_linear import apply_linear_score, fit_corr_weights, print_fit_summary
from python_edge.portfolio.construct import build_long_short_portfolio
from python_edge.wf.evaluate_ranker import evaluate_long_short, print_summary, summarize_daily_returns
from python_edge.wf.splits import build_walkforward_splits, print_split_summary


FEATURE_FILE = Path("data") / "features" / "feature_matrix_v1.parquet"
BASE_FEATURES = [
    "momentum_20d",
    "str_3d",
    "overnight_drift_20d",
    "volume_shock",
    "ivol_20d",
    "vol_compression",
    "intraday_rs",
    "intraday_pressure",
    "liq_rank",
    "market_breadth",
    "mom_x_volume_shock",
    "intraday_rs_x_volume_shock",
    "mom_x_vol_compression",
    "mom_x_market_breadth",
    "intraday_pressure_x_volume_shock",
    "liq_rank_x_intraday_rs",
]
FEATURES = BASE_FEATURES + CONDITIONAL_FEATURE_COLS
TARGET_COL = "target_fwd_ret_1d"
TRAIN_DAYS = 252
TEST_DAYS = 63
STEP_DAYS = 63
TOP_PCT = 0.10
MIN_TRAIN_ROWS = 500
MIN_TEST_ROWS = 100



def _prepare_base_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" not in out.columns:
        raise RuntimeError("_prepare_base_frame: missing date")

    out = add_conditional_factors(out)
    out = cs_zscore(out, FEATURES)

    zcols = [f"z_{f}" for f in FEATURES]
    needed = ["date", "symbol", TARGET_COL] + FEATURES + zcols
    missing = [c for c in needed if c not in out.columns]
    if missing:
        raise RuntimeError(f"_prepare_base_frame: missing columns: {missing}")
    return out



def _slice_by_date(df: pd.DataFrame, start_date: object, end_date: object) -> pd.DataFrame:
    return df.loc[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()



def _weights_to_frame(fold_id: int, weights: dict[str, float]) -> pd.DataFrame:
    rows = []
    for feature_name, weight_value in weights.items():
        rows.append(
            {
                "fold_id": fold_id,
                "feature": feature_name,
                "weight": float(weight_value),
                "abs_weight": abs(float(weight_value)),
                "sign": 0 if float(weight_value) == 0.0 else (1 if float(weight_value) > 0.0 else -1),
            }
        )
    return pd.DataFrame(rows)



def _print_weight_stability(weights_df: pd.DataFrame) -> None:
    if weights_df.empty:
        print("[WF][WEIGHTS][WARN] empty weights_df")
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

    print("[WF][WEIGHTS][STABILITY]")
    print(grouped.to_string(index=False))



def main() -> int:
    print(f"[CFG] feature_file={FEATURE_FILE}")
    print(f"[CFG] target_col={TARGET_COL}")
    print(f"[CFG] train_days={TRAIN_DAYS} test_days={TEST_DAYS} step_days={STEP_DAYS} top_pct={TOP_PCT}")

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(f"Feature file not found: {FEATURE_FILE}")

    print("[LOAD] feature matrix")
    df = pd.read_parquet(FEATURE_FILE)
    if df.empty:
        raise RuntimeError("Loaded feature matrix is empty")

    df = _prepare_base_frame(df)
    splits = build_walkforward_splits(df["date"], train_days=TRAIN_DAYS, test_days=TEST_DAYS, step_days=STEP_DAYS)
    print_split_summary(splits)

    all_test_daily: list[pd.DataFrame] = []
    all_weight_frames: list[pd.DataFrame] = []

    for sp in splits:
        print(f"[WF][FOLD {sp.fold_id}] start")
        train_df = _slice_by_date(df, sp.train_start, sp.train_end)
        test_df = _slice_by_date(df, sp.test_start, sp.test_end)

        print(f"[WF][FOLD {sp.fold_id}] train_rows={len(train_df)} test_rows={len(test_df)}")
        if len(train_df) < MIN_TRAIN_ROWS:
            raise RuntimeError(f"[WF][FOLD {sp.fold_id}] train_rows too small: {len(train_df)}")
        if len(test_df) < MIN_TEST_ROWS:
            raise RuntimeError(f"[WF][FOLD {sp.fold_id}] test_rows too small: {len(test_df)}")

        zcols = [f"z_{f}" for f in FEATURES]
        fit = fit_corr_weights(train_df=train_df, zcols=zcols, target_col=TARGET_COL)
        print_fit_summary(fit)
        all_weight_frames.append(_weights_to_frame(sp.fold_id, fit.weights))

        scored_test = apply_linear_score(test_df, fit=fit, out_col="score")
        port_test = build_long_short_portfolio(scored_test, top_pct=TOP_PCT)
        daily_test = evaluate_long_short(port_test, target_col=TARGET_COL)
        daily_test["fold_id"] = sp.fold_id

        fold_summary = summarize_daily_returns(daily_test)
        print_summary(f"[WF][FOLD {sp.fold_id}][SUMMARY]", fold_summary)
        all_test_daily.append(daily_test)

    if not all_test_daily:
        raise RuntimeError("No test daily results were produced")

    overall = pd.concat(all_test_daily, axis=0, ignore_index=True)
    overall = overall.sort_values(["date", "fold_id"]).reset_index(drop=True)
    overall_summary = summarize_daily_returns(overall)
    print_summary("[WF][OVERALL]", overall_summary)

    weights_df = pd.concat(all_weight_frames, axis=0, ignore_index=True) if all_weight_frames else pd.DataFrame()
    _print_weight_stability(weights_df)

    print("[WF][TAIL]")
    print(overall.tail(10).to_string(index=False))
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