from __future__ import annotations

import traceback
from pathlib import Path

import pandas as pd

from python_edge.model.cs_normalize import cs_zscore
from python_edge.model.ranker_linear import apply_linear_score, fit_corr_weights, print_fit_summary
from python_edge.portfolio.construct import build_long_short_portfolio
from python_edge.wf.evaluate_ranker import evaluate_long_short, print_summary, summarize_daily_returns
from python_edge.wf.splits import build_walkforward_splits, print_split_summary


FEATURE_FILE = Path("data") / "features" / "feature_matrix_v1.parquet"
FEATURES = [
    "momentum_20d",
    "str_3d",
    "overnight_drift_20d",
    "volume_shock",
    "ivol_20d",
]
TARGET_COL = "target_fwd_ret_1d"
TRAIN_DAYS = 252
TEST_DAYS = 63
STEP_DAYS = 63
TOP_PCT = 0.10



def _prepare_base_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" not in out.columns:
        raise RuntimeError("_prepare_base_frame: missing date")
    out = cs_zscore(out, FEATURES)
    zcols = [f"z_{f}" for f in FEATURES]
    needed = ["date", "symbol", TARGET_COL] + FEATURES + zcols
    missing = [c for c in needed if c not in out.columns]
    if missing:
        raise RuntimeError(f"_prepare_base_frame: missing columns: {missing}")
    return out



def _slice_by_date(df: pd.DataFrame, start_date: object, end_date: object) -> pd.DataFrame:
    return df.loc[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()



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

    for sp in splits:
        print(f"[WF][FOLD {sp.fold_id}] start")
        train_df = _slice_by_date(df, sp.train_start, sp.train_end)
        test_df = _slice_by_date(df, sp.test_start, sp.test_end)

        print(f"[WF][FOLD {sp.fold_id}] train_rows={len(train_df)} test_rows={len(test_df)}")
        if train_df.empty:
            raise RuntimeError(f"[WF][FOLD {sp.fold_id}] empty train_df")
        if test_df.empty:
            raise RuntimeError(f"[WF][FOLD {sp.fold_id}] empty test_df")

        zcols = [f"z_{f}" for f in FEATURES]
        fit = fit_corr_weights(train_df=train_df, zcols=zcols, target_col=TARGET_COL)
        print_fit_summary(fit)

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