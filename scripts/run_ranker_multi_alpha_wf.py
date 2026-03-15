from __future__ import annotations

import json
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ALPHA_LIB_FILE = Path(os.getenv("ALPHA_LIB_FILE", r"data\alpha_library\alpha_library_v1.parquet"))
OUT_DIR = Path(os.getenv("MULTI_ALPHA_WF_OUT_DIR", r"artifacts\multi_alpha_wf"))
PAUSE_ON_EXIT_ENV = str(os.getenv("PAUSE_ON_EXIT", "auto")).strip().lower()

TARGET_COL = str(os.getenv("TARGET_COL", "target_fwd_ret_1d")).strip()
TRAIN_DAYS = int(os.getenv("WF_TRAIN_DAYS", "252"))
TEST_DAYS = int(os.getenv("WF_TEST_DAYS", "63"))
STEP_DAYS = int(os.getenv("WF_STEP_DAYS", "63"))
PURGE_DAYS = int(os.getenv("WF_PURGE_DAYS", "5"))
EMBARGO_DAYS = int(os.getenv("WF_EMBARGO_DAYS", "5"))
MIN_TRAIN_ROWS = int(os.getenv("MIN_TRAIN_ROWS", "3000"))
MIN_TEST_ROWS = int(os.getenv("MIN_TEST_ROWS", "300"))
MIN_ALPHA_DAYS = int(os.getenv("MIN_ALPHA_DAYS", "40"))
MIN_ALPHA_ABS_IC = float(os.getenv("MIN_ALPHA_ABS_IC", "0.002"))
MAX_ALPHAS = int(os.getenv("MAX_ALPHAS", "24"))
RIDGE_L2 = float(os.getenv("RIDGE_L2", "8.0"))
ENTER_PCT = float(os.getenv("ENTER_PCT", "0.10"))
EXIT_PCT = float(os.getenv("EXIT_PCT", "0.20"))
GROSS_TARGET = float(os.getenv("GROSS_TARGET", "1.0"))
WEIGHT_CAP = float(os.getenv("WEIGHT_CAP", "0.08"))
MAX_DAILY_TURNOVER = float(os.getenv("MAX_DAILY_TURNOVER", "0.80"))
COST_BPS = float(os.getenv("COST_BPS", "8.0"))
BORROW_BPS_DAILY = float(os.getenv("BORROW_BPS_DAILY", "1.0"))
SIDE = str(os.getenv("SIDE", "long_short")).strip().lower()
TOPK_DEBUG = int(os.getenv("TOPK_DEBUG", "10"))
EPS = 1e-12


@dataclass(frozen=True)
class WFSplit:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


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


def _alpha_cols(df: pd.DataFrame) -> List[str]:
    return sorted([c for c in df.columns if c.startswith("alpha_")])


def _robust_zscore_series(s: pd.Series) -> pd.Series:
    x = _safe_numeric(s)
    valid = x.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=x.index, dtype="float64")
    med = float(valid.median())
    mad = float((valid - med).abs().median())
    if mad > EPS:
        out = (x - med) / (1.4826 * mad)
    else:
        mean = float(valid.mean())
        std = float(valid.std(ddof=0))
        out = (x - mean) / (std + EPS)
    return out.replace([np.inf, -np.inf], np.nan)


def _cs_zscore_df(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col] = out.groupby("date", sort=False)[col].transform(_robust_zscore_series)
    return out


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    wsum = float(np.sum(weights))
    if wsum <= EPS:
        return float(np.mean(values))
    return float(np.sum(values * weights) / wsum)


def _rank_abs_pct_by_date(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby("date", sort=False)[col].transform(lambda s: _safe_numeric(s).abs().rank(method="average", pct=True))


# ------------------------------------------------------------
# LOAD / SPLITS
# ------------------------------------------------------------

def load_alpha_library() -> pd.DataFrame:
    if not ALPHA_LIB_FILE.exists():
        raise FileNotFoundError(f"Alpha library not found: {ALPHA_LIB_FILE}")
    df = pd.read_parquet(ALPHA_LIB_FILE)
    if df.empty:
        raise RuntimeError("Alpha library is empty")
    required = ["date", "symbol", TARGET_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Alpha library missing required columns: {missing}")
    alpha_cols = _alpha_cols(df)
    if not alpha_cols:
        raise RuntimeError("No alpha_ columns found in alpha library")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)
    return df


def build_walkforward_splits(dates: Sequence[pd.Timestamp]) -> List[WFSplit]:
    uniq = pd.Index(sorted(pd.to_datetime(pd.Series(dates)).dt.normalize().unique()))
    if len(uniq) < (TRAIN_DAYS + TEST_DAYS + PURGE_DAYS + EMBARGO_DAYS + 5):
        raise RuntimeError("Not enough dates for requested walk-forward configuration")
    splits: List[WFSplit] = []
    fold_id = 1
    train_end_idx = TRAIN_DAYS - 1
    while True:
        test_start_idx = train_end_idx + 1 + PURGE_DAYS + EMBARGO_DAYS
        test_end_idx = test_start_idx + TEST_DAYS - 1
        if test_end_idx >= len(uniq):
            break
        train_start_idx = train_end_idx - TRAIN_DAYS + 1
        splits.append(
            WFSplit(
                fold_id=fold_id,
                train_start=uniq[train_start_idx],
                train_end=uniq[train_end_idx],
                test_start=uniq[test_start_idx],
                test_end=uniq[test_end_idx],
            )
        )
        fold_id += 1
        train_end_idx += STEP_DAYS
    if not splits:
        raise RuntimeError("No walk-forward splits generated")
    return splits


# ------------------------------------------------------------
# ALPHA SELECTION / WEIGHTS
# ------------------------------------------------------------

def _daily_ic(train_df: pd.DataFrame, factor_col: str, target_col: str, min_cs: int = 20) -> Tuple[float, int]:
    vals: List[float] = []
    for _, g in train_df.groupby("date", sort=False):
        x = g[[factor_col, target_col]].dropna()
        if len(x) < min_cs:
            continue
        if x[factor_col].nunique(dropna=True) <= 1:
            continue
        if x[target_col].nunique(dropna=True) <= 1:
            continue
        ic = x[factor_col].corr(x[target_col], method="spearman")
        if pd.notna(ic):
            vals.append(float(ic))
    if not vals:
        return float("nan"), 0
    return float(np.nanmean(vals)), int(len(vals))


def select_alphas(train_df: pd.DataFrame, alpha_cols: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for col in alpha_cols:
        ic, n_days = _daily_ic(train_df, col, TARGET_COL)
        rows.append({
            "alpha": col,
            "ic": ic,
            "abs_ic": abs(ic) if pd.notna(ic) else np.nan,
            "n_days": n_days,
        })
    sel = pd.DataFrame(rows)
    if sel.empty:
        raise RuntimeError("select_alphas: no alpha diagnostics produced")
    sel = sel.loc[(sel["n_days"] >= MIN_ALPHA_DAYS) & (sel["abs_ic"] >= MIN_ALPHA_ABS_IC)].copy()
    if sel.empty:
        raise RuntimeError("select_alphas: no alpha passed min alpha thresholds")
    sel = sel.sort_values(["abs_ic", "n_days", "alpha"], ascending=[False, False, True]).reset_index(drop=True)

    picked: List[str] = []
    if len(sel) > 0:
        corr_source = train_df[[c for c in sel["alpha"].tolist() if c in train_df.columns]].copy()
        corr = corr_source.corr(method="spearman", min_periods=200)
        for alpha in sel["alpha"].tolist():
            if len(picked) >= MAX_ALPHAS:
                break
            ok = True
            for prev in picked:
                c = corr.loc[alpha, prev] if alpha in corr.index and prev in corr.columns else np.nan
                if pd.notna(c) and abs(float(c)) >= 0.92:
                    ok = False
                    break
            if ok:
                picked.append(alpha)
    sel["selected"] = sel["alpha"].isin(picked).astype(int)
    sel = sel.sort_values(["selected", "abs_ic", "n_days", "alpha"], ascending=[False, False, False, True]).reset_index(drop=True)
    return sel


def fit_ridge_weights(train_df: pd.DataFrame, selected_alphas: Sequence[str]) -> pd.DataFrame:
    if not selected_alphas:
        raise RuntimeError("fit_ridge_weights: selected_alphas empty")

    xdf = train_df[list(selected_alphas)].copy()
    xdf = xdf.apply(_safe_numeric)
    y = _safe_numeric(train_df[TARGET_COL]).to_numpy(dtype="float64")

    # Multi-alpha libraries are sparse by construction: many alphas are active only
    # in specific regimes. Requiring a complete-case intersection across every
    # selected alpha collapses the sample to zero rows. Treat missing alpha values
    # as inactive signal (0.0) after cross-sectional normalization.
    valid_target = np.isfinite(y)
    active_counts = np.isfinite(xdf.to_numpy(dtype="float64")).sum(axis=1)
    min_active = max(1, min(3, len(selected_alphas)))
    usable_mask = valid_target & (active_counts >= min_active)

    x = xdf.loc[usable_mask, list(selected_alphas)].fillna(0.0).to_numpy(dtype="float64")
    y = y[usable_mask]
    if len(y) < MIN_TRAIN_ROWS:
        raise RuntimeError(
            f"fit_ridge_weights: too few usable train rows ({len(y)}) < {MIN_TRAIN_ROWS}; "
            f"selected_alphas={len(selected_alphas)} min_active={min_active}"
        )

    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    x_std = np.where(x_std <= EPS, 1.0, x_std)
    xz = (x - x_mean) / x_std
    y_mean = float(np.mean(y))
    y_std = float(np.std(y))
    yz = (y - y_mean) / (y_std + EPS)

    xtx = xz.T @ xz
    reg = RIDGE_L2 * np.eye(xtx.shape[0], dtype="float64")
    xty = xz.T @ yz
    w_std = np.linalg.solve(xtx + reg, xty)
    w_raw = w_std / x_std

    out = pd.DataFrame({
        "alpha": list(selected_alphas),
        "coef_std": w_std,
        "coef_raw": w_raw,
        "x_mean": x_mean,
        "x_std": x_std,
    })
    out["abs_coef_std"] = out["coef_std"].abs()
    scale = out["abs_coef_std"].sum()
    out["blend_weight"] = out["coef_std"] / (scale + EPS)
    out = out.sort_values(["abs_coef_std", "alpha"], ascending=[False, True]).reset_index(drop=True)
    return out


# ------------------------------------------------------------
# SCORING / PORTFOLIO
# ------------------------------------------------------------

def apply_weights(test_df: pd.DataFrame, weights_df: pd.DataFrame) -> pd.DataFrame:
    out = test_df.copy()
    selected = weights_df["alpha"].tolist()
    for alpha in selected:
        out[alpha] = _safe_numeric(out[alpha])
    score = np.zeros(len(out), dtype="float64")
    for _, row in weights_df.iterrows():
        score += row["blend_weight"] * out[row["alpha"]].fillna(0.0).to_numpy(dtype="float64")
    out["score_model"] = score
    out["score_model_z"] = out.groupby("date", sort=False)["score_model"].transform(_robust_zscore_series)
    out["score_abs_rank_pct"] = _rank_abs_pct_by_date(out, "score_model_z")
    return out


def build_portfolio(scored_df: pd.DataFrame) -> pd.DataFrame:
    out = scored_df.copy()
    out["signal_side"] = 0
    if SIDE == "long_only":
        out.loc[out["score_model_z"] > 0, "signal_side"] = 1
    elif SIDE == "short_only":
        out.loc[out["score_model_z"] < 0, "signal_side"] = -1
    else:
        out.loc[out["score_model_z"] > 0, "signal_side"] = 1
        out.loc[out["score_model_z"] < 0, "signal_side"] = -1

    if SIDE != "short_only":
        out["long_rank_pct"] = out.groupby("date", sort=False)["score_model_z"].transform(lambda s: _safe_numeric(s).rank(method="average", pct=True))
    else:
        out["long_rank_pct"] = np.nan
    if SIDE != "long_only":
        out["short_rank_pct"] = out.groupby("date", sort=False)["score_model_z"].transform(lambda s: _safe_numeric(-s).rank(method="average", pct=True))
    else:
        out["short_rank_pct"] = np.nan

    out["side"] = 0
    if SIDE != "short_only":
        out.loc[(out["signal_side"] > 0) & (out["long_rank_pct"] >= 1.0 - ENTER_PCT), "side"] = 1
    if SIDE != "long_only":
        out.loc[(out["signal_side"] < 0) & (out["short_rank_pct"] >= 1.0 - ENTER_PCT), "side"] = -1

    out["raw_strength"] = out["score_model_z"].abs().fillna(0.0)
    out.loc[out["side"] == 0, "raw_strength"] = 0.0

    pieces: List[pd.DataFrame] = []
    for _, g in out.groupby("date", sort=False):
        gg = g.copy()
        pos_long = gg.loc[gg["side"] > 0, "raw_strength"]
        pos_short = gg.loc[gg["side"] < 0, "raw_strength"]
        long_sum = float(pos_long.sum())
        short_sum = float(pos_short.sum())
        gg["weight"] = 0.0
        if SIDE != "short_only" and long_sum > EPS:
            gg.loc[gg["side"] > 0, "weight"] = 0.5 * gg.loc[gg["side"] > 0, "raw_strength"] / long_sum
        if SIDE != "long_only" and short_sum > EPS:
            gg.loc[gg["side"] < 0, "weight"] = -0.5 * gg.loc[gg["side"] < 0, "raw_strength"] / short_sum
        if SIDE == "long_only" and long_sum > EPS:
            gg.loc[gg["side"] > 0, "weight"] = GROSS_TARGET * gg.loc[gg["side"] > 0, "raw_strength"] / long_sum
        if SIDE == "short_only" and short_sum > EPS:
            gg.loc[gg["side"] < 0, "weight"] = -GROSS_TARGET * gg.loc[gg["side"] < 0, "raw_strength"] / short_sum
        gg["weight"] = gg["weight"].clip(lower=-WEIGHT_CAP, upper=WEIGHT_CAP)
        gross = float(gg["weight"].abs().sum())
        if gross > EPS:
            gg["weight"] = gg["weight"] * (GROSS_TARGET / gross)
        pieces.append(gg)
    out = pd.concat(pieces, ignore_index=True)
    out = out.sort_values(["symbol", "date"]).reset_index(drop=True)
    out["prev_weight"] = out.groupby("symbol", sort=False)["weight"].shift(1).fillna(0.0)
    out["turnover_name"] = (out["weight"] - out["prev_weight"]).abs()

    daily_turn = out.groupby("date", sort=False)["turnover_name"].transform("sum")
    out["daily_turnover_pre_cap"] = daily_turn
    scale = np.minimum(1.0, MAX_DAILY_TURNOVER / (daily_turn + EPS))
    out["turnover_scale"] = scale
    out["weight"] = out["prev_weight"] + (out["weight"] - out["prev_weight"]) * out["turnover_scale"]
    out["turnover"] = (out["weight"] - out["prev_weight"]).abs()
    return out


def evaluate_portfolio(port_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    df = port_df.copy()
    df["gross_pnl"] = _safe_numeric(df["weight"]) * _safe_numeric(df[TARGET_COL])
    df["cost"] = _safe_numeric(df["turnover"]) * (COST_BPS / 10000.0)
    df.loc[df["weight"] < 0, "cost"] += _safe_numeric(df.loc[df["weight"] < 0, "weight"]).abs() * (BORROW_BPS_DAILY / 10000.0)
    daily = df.groupby("date", sort=False).agg(
        gross_ret=("gross_pnl", "sum"),
        cost_ret=("cost", "sum"),
        turnover=("turnover", "sum"),
        names=("symbol", "count"),
        gross=("weight", lambda s: float(np.sum(np.abs(pd.to_numeric(s, errors="coerce"))))),
        longs=("weight", lambda s: int((pd.to_numeric(s, errors="coerce") > 0).sum())),
        shorts=("weight", lambda s: int((pd.to_numeric(s, errors="coerce") < 0).sum())),
    ).reset_index()
    daily["net_ret"] = daily["gross_ret"] - daily["cost_ret"]
    daily["equity"] = (1.0 + daily["net_ret"].fillna(0.0)).cumprod()
    daily["cum_ret"] = daily["equity"] - 1.0

    mean = float(daily["net_ret"].mean()) if len(daily) else float("nan")
    std = float(daily["net_ret"].std(ddof=0)) if len(daily) else float("nan")
    sharpe = (mean / (std + EPS)) * np.sqrt(252.0) if len(daily) else float("nan")
    hit = float((daily["net_ret"] > 0).mean()) if len(daily) else float("nan")
    maxdd = float((daily["equity"] / daily["equity"].cummax() - 1.0).min()) if len(daily) else float("nan")
    summary = {
        "days": float(len(daily)),
        "mean_daily": mean,
        "std_daily": std,
        "sharpe": sharpe,
        "hit_rate": hit,
        "cum_ret": float(daily["cum_ret"].iloc[-1]) if len(daily) else float("nan"),
        "max_drawdown": maxdd,
        "avg_turnover": float(daily["turnover"].mean()) if len(daily) else float("nan"),
        "avg_gross": float(daily["gross"].mean()) if len(daily) else float("nan"),
    }
    return daily, summary


# ------------------------------------------------------------
# MAIN WF
# ------------------------------------------------------------

def run_fold(df: pd.DataFrame, split: WFSplit, alpha_cols: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    train_df = df.loc[(df["date"] >= split.train_start) & (df["date"] <= split.train_end)].copy()
    test_df = df.loc[(df["date"] >= split.test_start) & (df["date"] <= split.test_end)].copy()
    if len(train_df) < MIN_TRAIN_ROWS:
        raise RuntimeError(f"fold {split.fold_id}: train rows too small: {len(train_df)}")
    if len(test_df) < MIN_TEST_ROWS:
        raise RuntimeError(f"fold {split.fold_id}: test rows too small: {len(test_df)}")

    train_df = _cs_zscore_df(train_df, alpha_cols)
    test_df = _cs_zscore_df(test_df, alpha_cols)

    selected_df = select_alphas(train_df, alpha_cols)
    selected = selected_df.loc[selected_df["selected"] == 1, "alpha"].tolist()
    if not selected:
        raise RuntimeError(f"fold {split.fold_id}: selected alpha list is empty")
    weights_df = fit_ridge_weights(train_df, selected)
    scored = apply_weights(test_df, weights_df)
    port = build_portfolio(scored)
    daily, summary = evaluate_portfolio(port)
    daily["fold_id"] = split.fold_id
    daily["train_start"] = split.train_start
    daily["train_end"] = split.train_end
    daily["test_start"] = split.test_start
    daily["test_end"] = split.test_end
    summary["selected_alpha_count"] = float(len(selected))
    summary["selected_alpha_list"] = ",".join(selected)
    summary["usable_train_rows"] = float(int(np.isfinite(_safe_numeric(train_df[TARGET_COL]).to_numpy(dtype="float64")).sum()))
    summary["selected_alpha_nonzero_mean"] = float(
        train_df[selected].fillna(0.0).abs().sum(axis=1).gt(0.0).mean()
    )
    summary["fold_id"] = float(split.fold_id)
    return selected_df, weights_df, daily, summary


def main() -> int:
    _enable_line_buffering()
    print(f"[CFG] alpha_lib_file={ALPHA_LIB_FILE}")
    print(f"[CFG] out_dir={OUT_DIR}")
    print(f"[CFG] target_col={TARGET_COL}")
    print(f"[CFG] train_days={TRAIN_DAYS} test_days={TEST_DAYS} step_days={STEP_DAYS} purge_days={PURGE_DAYS} embargo_days={EMBARGO_DAYS}")
    print(f"[CFG] ridge_l2={RIDGE_L2} max_alphas={MAX_ALPHAS} min_alpha_days={MIN_ALPHA_DAYS} min_alpha_abs_ic={MIN_ALPHA_ABS_IC}")
    print(f"[CFG] enter_pct={ENTER_PCT} weight_cap={WEIGHT_CAP} max_daily_turnover={MAX_DAILY_TURNOVER} cost_bps={COST_BPS}")

    df = load_alpha_library()
    alpha_cols = _alpha_cols(df)
    print(f"[DATA] rows={len(df)} dates={df['date'].nunique()} symbols={df['symbol'].nunique()} alpha_cols={len(alpha_cols)}")

    splits = build_walkforward_splits(df["date"])
    print(f"[WF] folds={len(splits)}")
    for sp in splits:
        print(f"[WF][FOLD {sp.fold_id}] train={sp.train_start.date()}..{sp.train_end.date()} test={sp.test_start.date()}..{sp.test_end.date()}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries: List[Dict[str, float]] = []
    all_daily: List[pd.DataFrame] = []
    top_weights_preview: List[Dict[str, object]] = []

    for sp in splits:
        print(f"[WF][FOLD {sp.fold_id}] start")
        selected_df, weights_df, daily, summary = run_fold(df, sp, alpha_cols)
        selected_path = OUT_DIR / f"wf_selected_alphas__fold{sp.fold_id}.csv"
        weights_path = OUT_DIR / f"wf_alpha_weights__fold{sp.fold_id}.csv"
        daily_path = OUT_DIR / f"wf_daily__fold{sp.fold_id}.csv"
        selected_df.to_csv(selected_path, index=False)
        weights_df.to_csv(weights_path, index=False)
        daily.to_csv(daily_path, index=False)
        print(f"[WF][FOLD {sp.fold_id}][SUMMARY] sharpe={summary['sharpe']:.4f} mean_daily={summary['mean_daily']:.6f} cum_ret={summary['cum_ret']:.4f} maxdd={summary['max_drawdown']:.4f} selected={int(summary['selected_alpha_count'])}")
        print(f"[WF][FOLD {sp.fold_id}][TOP_WEIGHTS]")
        print(weights_df.head(TOPK_DEBUG).to_string(index=False))
        for _, row in weights_df.head(TOPK_DEBUG).iterrows():
            top_weights_preview.append({
                "fold_id": sp.fold_id,
                "alpha": row["alpha"],
                "coef_std": float(row["coef_std"]),
                "blend_weight": float(row["blend_weight"]),
            })
        summaries.append(summary)
        all_daily.append(daily)

    overall_daily = pd.concat(all_daily, ignore_index=True).sort_values(["date", "fold_id"]).reset_index(drop=True)
    overall_path = OUT_DIR / "wf_multi_alpha_overall.csv"
    overall_daily.to_csv(overall_path, index=False)
    overall_summary = {
        "days": float(len(overall_daily)),
        "mean_daily": float(overall_daily["net_ret"].mean()),
        "std_daily": float(overall_daily["net_ret"].std(ddof=0)),
        "sharpe": float((overall_daily["net_ret"].mean() / (overall_daily["net_ret"].std(ddof=0) + EPS)) * np.sqrt(252.0)),
        "hit_rate": float((overall_daily["net_ret"] > 0).mean()),
        "cum_ret": float(((1.0 + overall_daily["net_ret"].fillna(0.0)).cumprod().iloc[-1]) - 1.0),
        "max_drawdown": float((((1.0 + overall_daily["net_ret"].fillna(0.0)).cumprod()) / ((1.0 + overall_daily["net_ret"].fillna(0.0)).cumprod()).cummax() - 1.0).min()),
        "avg_turnover": float(overall_daily["turnover"].mean()),
        "folds": int(len(summaries)),
    }
    summary_df = pd.DataFrame(summaries)
    summary_path = OUT_DIR / "wf_fold_summaries.csv"
    summary_df.to_csv(summary_path, index=False)

    meta = {
        "alpha_lib_file": str(ALPHA_LIB_FILE),
        "target_col": TARGET_COL,
        "train_days": TRAIN_DAYS,
        "test_days": TEST_DAYS,
        "step_days": STEP_DAYS,
        "purge_days": PURGE_DAYS,
        "embargo_days": EMBARGO_DAYS,
        "min_train_rows": MIN_TRAIN_ROWS,
        "min_test_rows": MIN_TEST_ROWS,
        "min_alpha_days": MIN_ALPHA_DAYS,
        "min_alpha_abs_ic": MIN_ALPHA_ABS_IC,
        "max_alphas": MAX_ALPHAS,
        "ridge_l2": RIDGE_L2,
        "enter_pct": ENTER_PCT,
        "gross_target": GROSS_TARGET,
        "weight_cap": WEIGHT_CAP,
        "max_daily_turnover": MAX_DAILY_TURNOVER,
        "cost_bps": COST_BPS,
        "borrow_bps_daily": BORROW_BPS_DAILY,
        "side": SIDE,
        "fold_summaries": summaries,
        "overall_summary": overall_summary,
        "top_weights_preview": top_weights_preview,
        "alpha_count": len(alpha_cols),
    }
    meta_path = OUT_DIR / "wf_multi_alpha_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OVERALL] sharpe={overall_summary['sharpe']:.4f} mean_daily={overall_summary['mean_daily']:.6f} cum_ret={overall_summary['cum_ret']:.4f} maxdd={overall_summary['max_drawdown']:.4f} avg_turnover={overall_summary['avg_turnover']:.4f}")
    print(f"[ARTIFACT] {overall_path}")
    print(f"[ARTIFACT] {summary_path}")
    print(f"[ARTIFACT] {meta_path}")
    print("[FINAL] multi-alpha walk-forward complete")
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 1
    _press_enter_exit(rc)
