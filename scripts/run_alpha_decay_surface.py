from __future__ import annotations

import json
import traceback
from pathlib import Path

import pandas as pd

from python_edge.analysis.alpha_decay import AlphaDecayConfig, build_alpha_decay_surface, print_alpha_decay_summary, summarize_decay_turning_points
from python_edge.model.conditional_factors import add_conditional_factors
from python_edge.model.cs_normalize import cs_zscore
from python_edge.model.neutralize import add_beta_proxy, neutralize_score_cross_section
from python_edge.model.ranker_linear import apply_linear_score, fit_corr_weights
from python_edge.wf.splits import build_walkforward_splits, print_split_summary


FEATURE_FILE = Path("data") / "features" / "feature_matrix_v1.parquet"
OUT_DIR = Path("data") / "analysis"
OUT_SURFACE_CSV = OUT_DIR / "alpha_decay_surface.csv"
OUT_TURNING_JSON = OUT_DIR / "alpha_decay_turning_points.json"
TARGET_COL = "target_fwd_ret_1d"
TRAIN_DAYS = 252
TEST_DAYS = 63
STEP_DAYS = 63

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
        raise RuntimeError("_prepare_base_frame: missing symbol")
    if "close" not in out.columns:
        raise RuntimeError("_prepare_base_frame: missing close")

    out = add_conditional_factors(out)
    out = add_beta_proxy(out, lookback=60)
    out = cs_zscore(out, FULL_FEATURE_STACK)
    return out



def _slice_by_date(df: pd.DataFrame, start_date: object, end_date: object) -> pd.DataFrame:
    return df.loc[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()



def _build_oos_scored_frame(df: pd.DataFrame) -> pd.DataFrame:
    splits = build_walkforward_splits(df["date"], train_days=TRAIN_DAYS, test_days=TEST_DAYS, step_days=STEP_DAYS)
    print_split_summary(splits)

    scored_parts: list[pd.DataFrame] = []

    for sp in splits:
        print(f"[ALPHA_DECAY][FOLD {sp.fold_id}] start")
        train_df = _slice_by_date(df, sp.train_start, sp.train_end)
        test_df = _slice_by_date(df, sp.test_start, sp.test_end)
        print(f"[ALPHA_DECAY][FOLD {sp.fold_id}] train_rows={len(train_df)} test_rows={len(test_df)}")

        zcols = [f"z_{f}" for f in FULL_FEATURE_STACK]
        fit = fit_corr_weights(train_df=train_df, zcols=zcols, target_col=TARGET_COL)
        scored_test = apply_linear_score(test_df, fit=fit, out_col="score")
        scored_test = neutralize_score_cross_section(
            scored_test,
            score_col="score",
            exposure_cols=["liq_rank", "beta_proxy_60d"],
            out_col="score_neutral",
        )
        scored_test["fold_id"] = sp.fold_id
        scored_parts.append(scored_test)

    if not scored_parts:
        raise RuntimeError("_build_oos_scored_frame: no scored OOS parts produced")

    out = pd.concat(scored_parts, axis=0, ignore_index=True)
    out = out.sort_values(["date", "symbol"]).reset_index(drop=True)
    return out



def main() -> int:
    print(f"[CFG] feature_file={FEATURE_FILE}")
    print(f"[CFG] out_surface_csv={OUT_SURFACE_CSV}")
    print(f"[CFG] out_turning_json={OUT_TURNING_JSON}")

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(f"Feature file not found: {FEATURE_FILE}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[LOAD] feature matrix")
    raw_df = pd.read_parquet(FEATURE_FILE)
    if raw_df.empty:
        raise RuntimeError("Loaded feature matrix is empty")

    prepared = _prepare_base_frame(raw_df, FULL_FEATURE_STACK)
    oos_scored = _build_oos_scored_frame(prepared)

    cfg = AlphaDecayConfig(
        score_col="score_neutral",
        symbol_col="symbol",
        date_col="date",
        close_col="close",
        horizons=(1, 2, 3, 5, 10, 15, 20),
        quantiles=(0.02, 0.05, 0.10, 0.20),
    )

    surface = build_alpha_decay_surface(oos_scored, cfg)
    turning = summarize_decay_turning_points(surface)
    print_alpha_decay_summary(surface)

    surface.to_csv(OUT_SURFACE_CSV, index=False, encoding="utf-8-sig")
    OUT_TURNING_JSON.write_text(turning.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")

    print(f"[SAVE] surface_csv={OUT_SURFACE_CSV}")
    print(f"[SAVE] turning_json={OUT_TURNING_JSON}")
    print("[DONE] alpha decay surface completed")
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