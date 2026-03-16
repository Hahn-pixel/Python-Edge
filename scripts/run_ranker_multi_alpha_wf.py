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
SRC_DIR = ROOT / "src"
for p in [ROOT, SRC_DIR]:
    ps = str(p)
    if ps not in sys.path:
        sys.path.insert(0, ps)

try:
    from python_edge.portfolio.holding_inertia import apply_holding_inertia
    HAS_HOLDING_INERTIA = True
except Exception:
    HAS_HOLDING_INERTIA = False

    def apply_holding_inertia(df: pd.DataFrame, enter_pct: float, exit_pct: float) -> pd.DataFrame:
        out = df.copy()
        out["side"] = 0
        rank_pct = out.groupby("date", sort=False)["score"].transform(lambda s: pd.to_numeric(s, errors="coerce").rank(method="average", pct=True))
        out.loc[rank_pct >= 1.0 - enter_pct, "side"] = 1
        out.loc[rank_pct <= enter_pct, "side"] = -1
        out["dbg_holding_inertia_fallback"] = 1
        return out

try:
    from python_edge.portfolio.turnover_control import cap_daily_turnover
    HAS_TURNOVER_CONTROL = True
except Exception:
    HAS_TURNOVER_CONTROL = False

    def cap_daily_turnover(df: pd.DataFrame, weight_col: str = "weight", max_daily_turnover: float = 0.20) -> pd.DataFrame:
        out = df.copy()
        out = out.sort_values(["symbol", "date"]).reset_index(drop=True)
        out["prev_weight"] = out.groupby("symbol", sort=False)[weight_col].shift(1).fillna(0.0)
        out["turnover"] = (pd.to_numeric(out[weight_col], errors="coerce") - pd.to_numeric(out["prev_weight"], errors="coerce")).abs()
        day_turn = out.groupby("date", sort=False)["turnover"].transform("sum")
        scale = np.where(day_turn > max_daily_turnover, max_daily_turnover / (day_turn + 1e-12), 1.0)
        out[weight_col] = pd.to_numeric(out[weight_col], errors="coerce") * scale
        out["turnover"] = pd.to_numeric(out["turnover"], errors="coerce") * scale
        out["dbg_turnover_control_fallback"] = 1
        return out

EPS = 1e-12

ALPHA_LIB_FILE = Path(os.getenv("ALPHA_LIB_FILE", "data/alpha_library_v2/alpha_library_v2.parquet"))
ALPHA_SHORTLIST_CSV = Path(os.getenv("ALPHA_SHORTLIST_CSV", "data/alpha_library_v2/diagnostics/alpha_candidate_shortlist.csv"))
ALPHA_SHORTLIST_REQUIRED = str(os.getenv("ALPHA_SHORTLIST_REQUIRED", "1")).strip().lower() not in {"0", "false", "no", "off"}
OUT_DIR = Path(os.getenv("MULTI_ALPHA_WF_OUT_DIR", "artifacts/multi_alpha_wf"))
TARGET_COL = str(os.getenv("TARGET_COL", "target_fwd_ret_1d")).strip()
PAUSE_ON_EXIT = str(os.getenv("PAUSE_ON_EXIT", "auto")).strip().lower()

TRAIN_DAYS = int(os.getenv("WF_TRAIN_DAYS", "252"))
TEST_DAYS = int(os.getenv("WF_TEST_DAYS", "63"))
STEP_DAYS = int(os.getenv("WF_STEP_DAYS", "63"))
PURGE_DAYS = int(os.getenv("WF_PURGE_DAYS", "5"))
EMBARGO_DAYS = int(os.getenv("WF_EMBARGO_DAYS", "5"))
MIN_TRAIN_ROWS = int(os.getenv("MIN_TRAIN_ROWS", "3000"))
MIN_TEST_ROWS = int(os.getenv("MIN_TEST_ROWS", "300"))
MIN_ALPHA_DAYS = int(os.getenv("MIN_ALPHA_DAYS", "60"))
MIN_ALPHA_ABS_IC = float(os.getenv("MIN_ALPHA_ABS_IC", "0.0055"))
MIN_BLOCK_IC_ABS = float(os.getenv("MIN_BLOCK_IC_ABS", "0.0020"))
MIN_STABILITY_BLOCKS = int(os.getenv("MIN_STABILITY_BLOCKS", "3"))
MIN_SIGN_CONSISTENCY = float(os.getenv("MIN_SIGN_CONSISTENCY", "0.75"))
RIDGE_L2 = float(os.getenv("RIDGE_L2", "25.0"))
MAX_ALPHAS = int(os.getenv("MAX_ALPHAS", "6"))
MIN_SELECTED_ALPHAS = int(os.getenv("MIN_SELECTED_ALPHAS", "3"))
BASE_ENTER_PCT = float(os.getenv("ENTER_PCT", "0.06"))
BASE_EXIT_PCT = float(os.getenv("EXIT_PCT", "0.14"))
BASE_WEIGHT_CAP = float(os.getenv("WEIGHT_CAP", "0.05"))
BASE_GROSS_TARGET = float(os.getenv("GROSS_TARGET", "0.85"))
BASE_MAX_DAILY_TURNOVER = float(os.getenv("MAX_DAILY_TURNOVER", "0.20"))
COST_BPS = float(os.getenv("COST_BPS", "8.0"))
TOPK_DEBUG = int(os.getenv("TOPK_DEBUG", "10"))
CORR_PRUNE = float(os.getenv("CORR_PRUNE", "0.80"))
MIN_FINAL_SELECT_SCORE = float(os.getenv("MIN_FINAL_SELECT_SCORE", "0.160"))

CONSENSUS_MODE = str(os.getenv("CONSENSUS_MODE", "recent_weighted")).strip().lower()
CONSENSUS_REQUIRED_LAST_FOLD = str(os.getenv("CONSENSUS_REQUIRED_LAST_FOLD", "1")).strip().lower() not in {"0", "false", "no", "off"}
CONSENSUS_MIN_FOLDS = int(os.getenv("CONSENSUS_MIN_FOLDS", "2"))
CONSENSUS_MAX_SIGN_FLIP = float(os.getenv("CONSENSUS_MAX_SIGN_FLIP", "0.25"))
CONSENSUS_MIN_MEAN_ABS_IC = float(os.getenv("CONSENSUS_MIN_MEAN_ABS_IC", "0.0050"))
CONSENSUS_HISTORY_BLEND = float(os.getenv("CONSENSUS_HISTORY_BLEND", "0.55"))
CONSENSUS_FAMILY_CAP = int(os.getenv("CONSENSUS_FAMILY_CAP", "1"))
CURRENT_FAMILY_CAP = int(os.getenv("CURRENT_FAMILY_CAP", "2"))


@dataclass(frozen=True)
class WFSplit:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass(frozen=True)
class ShellVariant:
    name: str
    enter_pct: float
    exit_pct: float
    weight_cap: float
    gross_target: float
    max_daily_turnover: float


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


def _should_pause() -> bool:
    if PAUSE_ON_EXIT in {"0", "false", "no", "off"}:
        return False
    if PAUSE_ON_EXIT in {"1", "true", "yes", "on"}:
        return True
    stdin_obj = getattr(sys, "stdin", None)
    stdout_obj = getattr(sys, "stdout", None)
    stdin_ok = bool(stdin_obj is not None and hasattr(stdin_obj, "isatty") and stdin_obj.isatty())
    stdout_ok = bool(stdout_obj is not None and hasattr(stdout_obj, "isatty") and stdout_obj.isatty())
    return stdin_ok and stdout_ok


def _safe_exit(code: int) -> None:
    if _should_pause():
        try:
            print("")
            print(f"[EXIT] code={code}")
            input("Press Enter to exit...")
        except Exception:
            pass
    raise SystemExit(code)


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


def _must_exist(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def _block_dates(date_series: Sequence[pd.Timestamp], n_blocks: int = 4) -> List[pd.Index]:
    uniq = pd.Index(sorted(pd.to_datetime(pd.Series(date_series)).dt.normalize().unique()))
    if len(uniq) == 0:
        return []
    out: List[pd.Index] = []
    for idx in np.array_split(np.arange(len(uniq)), n_blocks):
        if len(idx) == 0:
            continue
        out.append(pd.Index(uniq[idx]))
    return out


def _daily_ic(frame: pd.DataFrame, factor_col: str, target_col: str, min_cs: int = 20) -> Tuple[float, int, float]:
    values: List[float] = []
    pos_days = 0
    used_days = 0
    for _, g in frame.groupby("date", sort=False):
        x = g[[factor_col, target_col]].dropna()
        if len(x) < min_cs:
            continue
        if x[factor_col].nunique(dropna=True) <= 1:
            continue
        if x[target_col].nunique(dropna=True) <= 1:
            continue
        ic = x[factor_col].corr(x[target_col], method="spearman")
        if pd.notna(ic):
            values.append(float(ic))
            used_days += 1
            if float(ic) > 0:
                pos_days += 1
    if not values:
        return float("nan"), 0, float("nan")
    return float(np.nanmean(values)), int(len(values)), float(pos_days / max(1, used_days))


def _alpha_stats(frame: pd.DataFrame, alpha_cols: Sequence[str], fold_id: int, family_map: Dict[str, str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    blocks = _block_dates(frame["date"], 4)
    for alpha in alpha_cols:
        ic, n_days, pos_rate = _daily_ic(frame, alpha, TARGET_COL)
        block_ics: List[float] = []
        for block_dates in blocks:
            block_df = frame.loc[frame["date"].isin(block_dates)].copy()
            bic, block_days, _ = _daily_ic(block_df, alpha, TARGET_COL)
            if block_days > 0 and pd.notna(bic):
                block_ics.append(float(bic))
        block_hit_count = int(sum(1 for x in block_ics if abs(x) >= MIN_BLOCK_IC_ABS))
        signs = [np.sign(x) for x in block_ics if abs(x) >= MIN_BLOCK_IC_ABS]
        pos_sign = int(sum(1 for s in signs if s > 0))
        neg_sign = int(sum(1 for s in signs if s < 0))
        sign_consistency = float(max(pos_sign, neg_sign) / max(1, len(signs))) if signs else 0.0
        row = {
            "fold_id": int(fold_id),
            "alpha": alpha,
            "family": family_map.get(alpha, "unknown"),
            "ic": ic,
            "abs_ic": abs(ic) if pd.notna(ic) else np.nan,
            "n_days": int(n_days),
            "pos_rate": pos_rate,
            "block_hit_count": float(block_hit_count),
            "sign_consistency": float(sign_consistency),
            "block_ic_mean": float(np.mean(block_ics)) if block_ics else float("nan"),
        }
        row["current_score"] = 0.65 * (row["abs_ic"] if pd.notna(row["abs_ic"]) else 0.0) + 0.20 * (row["block_ic_mean"] if pd.notna(row["block_ic_mean"]) else 0.0) + 0.15 * row["sign_consistency"]
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("alpha stats empty")
    return out


def _load_shortlist_required() -> Tuple[List[str], Dict[str, object], pd.DataFrame, Dict[str, str]]:
    counters: Dict[str, object] = {
        "shortlist_required": int(ALPHA_SHORTLIST_REQUIRED),
        "shortlist_path": str(ALPHA_SHORTLIST_CSV),
        "shortlist_csv_exists": int(ALPHA_SHORTLIST_CSV.exists()),
        "shortlist_rows_raw": 0,
        "shortlist_rows_unique": 0,
        "shortlist_duplicates_removed": 0,
        "shortlist_alpha_missing_name": 0,
    }
    if not ALPHA_SHORTLIST_CSV.exists():
        if ALPHA_SHORTLIST_REQUIRED:
            raise FileNotFoundError(f"ALPHA_SHORTLIST_REQUIRED=1 but shortlist csv not found: {ALPHA_SHORTLIST_CSV}")
        return [], counters, pd.DataFrame(columns=["alpha"]), {}
    shortlist_df = pd.read_csv(ALPHA_SHORTLIST_CSV)
    if "alpha" not in shortlist_df.columns:
        raise RuntimeError(f"Shortlist csv missing alpha column: {ALPHA_SHORTLIST_CSV}")
    counters["shortlist_rows_raw"] = int(len(shortlist_df))
    shortlist_df["alpha"] = shortlist_df["alpha"].astype(str).str.strip()
    bad_mask = shortlist_df["alpha"].eq("") | shortlist_df["alpha"].eq("nan")
    counters["shortlist_alpha_missing_name"] = int(bad_mask.sum())
    shortlist_df = shortlist_df.loc[~bad_mask].copy()
    shortlist_df = shortlist_df.drop_duplicates(subset=["alpha"]).reset_index(drop=True)
    counters["shortlist_rows_unique"] = int(len(shortlist_df))
    counters["shortlist_duplicates_removed"] = int(int(counters["shortlist_rows_raw"]) - int(counters["shortlist_alpha_missing_name"]) - int(counters["shortlist_rows_unique"]))
    family_map: Dict[str, str] = {}
    if "family" in shortlist_df.columns:
        family_map = dict(zip(shortlist_df["alpha"].astype(str), shortlist_df["family"].astype(str)))
    return shortlist_df["alpha"].tolist(), counters, shortlist_df, family_map


def load_alpha_library() -> Tuple[pd.DataFrame, List[str], Dict[str, object], pd.DataFrame, Dict[str, str]]:
    _must_exist(ALPHA_LIB_FILE, "Alpha library")
    df = pd.read_parquet(ALPHA_LIB_FILE)
    if df.empty:
        raise RuntimeError("Alpha library is empty")
    required = ["date", "symbol", TARGET_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Alpha library missing required columns: {missing}")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)
    alpha_before = _alpha_cols(df)
    if not alpha_before:
        raise RuntimeError("No alpha columns found")
    shortlist_cols, shortlist_counters, shortlist_df, family_map = _load_shortlist_required()
    shortlist_set = set(shortlist_cols)
    found = [c for c in alpha_before if c in shortlist_set]
    missing_in_parquet = [c for c in shortlist_cols if c not in set(alpha_before)]
    shortlist_counters["alpha_cols_before_shortlist"] = int(len(alpha_before))
    shortlist_counters["shortlist_found_in_parquet"] = int(len(found))
    shortlist_counters["shortlist_missing_in_parquet"] = int(len(missing_in_parquet))
    shortlist_counters["shortlist_missing_in_parquet_list"] = missing_in_parquet
    if ALPHA_SHORTLIST_REQUIRED and not found:
        raise RuntimeError("Shortlist required but no shortlist alphas found in parquet")
    if ALPHA_SHORTLIST_REQUIRED and missing_in_parquet:
        raise RuntimeError(f"Shortlist required and some shortlist alphas are missing in parquet: {missing_in_parquet[:10]}")
    alpha_cols = found if shortlist_cols else alpha_before
    if not alpha_cols:
        raise RuntimeError("No alpha columns remain after shortlist filtering")
    keep_cols = [c for c in df.columns if not c.startswith("alpha_")] + alpha_cols
    df = df[keep_cols].copy()
    shortlist_counters["alpha_cols_after_shortlist"] = int(len(alpha_cols))
    shortlist_counters["shortlist_filter_applied"] = int(bool(shortlist_cols))
    return df, alpha_cols, shortlist_counters, shortlist_df, family_map


def build_walkforward_splits(dates: Sequence[pd.Timestamp]) -> List[WFSplit]:
    uniq = pd.Index(sorted(pd.to_datetime(pd.Series(dates)).dt.normalize().unique()))
    if len(uniq) < (TRAIN_DAYS + TEST_DAYS + PURGE_DAYS + EMBARGO_DAYS + 5):
        raise RuntimeError("Not enough dates for walkforward configuration")
    splits: List[WFSplit] = []
    fold_id = 1
    train_end_idx = TRAIN_DAYS - 1
    while True:
        test_start_idx = train_end_idx + 1 + PURGE_DAYS + EMBARGO_DAYS
        test_end_idx = test_start_idx + TEST_DAYS - 1
        if test_end_idx >= len(uniq):
            break
        train_start_idx = train_end_idx - TRAIN_DAYS + 1
        splits.append(WFSplit(fold_id=fold_id, train_start=uniq[train_start_idx], train_end=uniq[train_end_idx], test_start=uniq[test_start_idx], test_end=uniq[test_end_idx]))
        fold_id += 1
        train_end_idx += STEP_DAYS
    if not splits:
        raise RuntimeError("No walkforward splits generated")
    return splits


def build_consensus_from_history(history_stats: pd.DataFrame) -> pd.DataFrame:
    if history_stats.empty:
        return pd.DataFrame(columns=["alpha", "family", "history_folds", "mean_abs_ic", "last_ic", "sign_flip_rate", "history_score", "consensus_admitted"])
    work = history_stats.sort_values(["alpha", "fold_id"]).reset_index(drop=True)
    rows: List[Dict[str, object]] = []
    for alpha, g in work.groupby("alpha", sort=False):
        fam = str(g["family"].iloc[0]) if "family" in g.columns else "unknown"
        ics = [float(x) for x in g["ic"].tolist() if pd.notna(x)]
        signs = [int(np.sign(x)) for x in ics if abs(x) >= MIN_BLOCK_IC_ABS]
        sign_flips = 0
        transitions = 0
        prev_sign = None
        for s in signs:
            if prev_sign is None:
                prev_sign = s
                continue
            transitions += 1
            if s != prev_sign:
                sign_flips += 1
            prev_sign = s
        flip_rate = float(sign_flips / max(1, transitions)) if transitions > 0 else 0.0
        mean_abs_ic = float(np.mean(np.abs(ics))) if ics else float("nan")
        last_ic = float(g["ic"].iloc[-1]) if len(g) else float("nan")
        mean_current = float(g["current_score"].mean()) if "current_score" in g.columns and len(g) else float("nan")
        folds = int(g["fold_id"].nunique())
        admitted = int((folds >= CONSENSUS_MIN_FOLDS) and pd.notna(mean_abs_ic) and (mean_abs_ic >= CONSENSUS_MIN_MEAN_ABS_IC) and (flip_rate <= CONSENSUS_MAX_SIGN_FLIP))
        history_score = 0.35 * (mean_abs_ic if pd.notna(mean_abs_ic) else 0.0) + (0.20 * abs(last_ic) if pd.notna(last_ic) else 0.0) + 0.25 * (1.0 - flip_rate) + 0.20 * (mean_current if pd.notna(mean_current) else 0.0)
        rows.append({"alpha": alpha, "family": fam, "history_folds": folds, "mean_abs_ic": mean_abs_ic, "last_ic": last_ic, "sign_flip_rate": flip_rate, "history_score": float(history_score), "consensus_admitted": admitted})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["consensus_admitted", "history_score", "alpha"], ascending=[False, False, True]).reset_index(drop=True)
    if CONSENSUS_FAMILY_CAP > 0 and len(out):
        out["family_rank"] = out.groupby("family", sort=False).cumcount() + 1
        out["consensus_admitted"] = np.where((out["consensus_admitted"] == 1) & (out["family_rank"] <= CONSENSUS_FAMILY_CAP), 1, 0)
    return out


def select_current_fold_alphas(current_stats: pd.DataFrame, history_consensus: pd.DataFrame, is_last_fold: bool) -> Tuple[pd.DataFrame, Dict[str, object]]:
    counters: Dict[str, object] = {
        "input_current": int(len(current_stats)),
        "after_current_thresholds": 0,
        "history_rows": int(len(history_consensus)),
        "dropped_by_fold_consensus": 0,
        "dropped_by_sign_flip": 0,
        "dropped_by_family_dominance": 0,
        "dropped_by_weight_instability": 0,
        "dropped_by_low_final_score": 0,
        "corr_pruned": 0,
        "selected": 0,
        "history_gate_active": 0,
    }
    work = current_stats.copy()
    work = work.loc[(work["n_days"] >= MIN_ALPHA_DAYS) & (work["abs_ic"] >= MIN_ALPHA_ABS_IC)].copy()
    work = work.loc[(work["block_hit_count"] >= MIN_STABILITY_BLOCKS) & (work["sign_consistency"] >= MIN_SIGN_CONSISTENCY)].copy()
    counters["after_current_thresholds"] = int(len(work))
    if work.empty:
        raise RuntimeError("No alpha passed current fold thresholds")
    if CONSENSUS_MODE != "off" and len(history_consensus):
        counters["history_gate_active"] = 1
        hist = history_consensus[["alpha", "family", "history_folds", "mean_abs_ic", "sign_flip_rate", "history_score", "consensus_admitted"]].copy()
        work = work.merge(hist, on=["alpha", "family"], how="left")
        work["history_folds"] = pd.to_numeric(work["history_folds"], errors="coerce").fillna(0.0)
        work["mean_abs_ic"] = pd.to_numeric(work["mean_abs_ic"], errors="coerce")
        work["sign_flip_rate"] = pd.to_numeric(work["sign_flip_rate"], errors="coerce")
        work["history_score"] = pd.to_numeric(work["history_score"], errors="coerce").fillna(0.0)
        work["consensus_admitted"] = pd.to_numeric(work["consensus_admitted"], errors="coerce").fillna(0.0)
        sign_flip_drop_mask = work["sign_flip_rate"].fillna(0.0) > CONSENSUS_MAX_SIGN_FLIP
        counters["dropped_by_sign_flip"] = int(sign_flip_drop_mask.sum())
        work = work.loc[~sign_flip_drop_mask].copy()
        if is_last_fold and CONSENSUS_REQUIRED_LAST_FOLD:
            consensus_drop_mask = work["consensus_admitted"].fillna(0.0) <= 0.0
            counters["dropped_by_fold_consensus"] = int(consensus_drop_mask.sum())
            work = work.loc[~consensus_drop_mask].copy()
        else:
            work["consensus_admitted"] = work["consensus_admitted"].fillna(0.0)
    else:
        work["history_score"] = 0.0
        work["consensus_admitted"] = 0.0
    if work.empty:
        raise RuntimeError("No alpha remained after fold consensus gate")
    history_blend = CONSENSUS_HISTORY_BLEND if (CONSENSUS_MODE != "off" and counters["history_gate_active"] == 1) else 0.0
    work["final_select_score"] = (1.0 - history_blend) * pd.to_numeric(work["current_score"], errors="coerce").fillna(0.0) + history_blend * pd.to_numeric(work["history_score"], errors="coerce").fillna(0.0)
    work["final_select_score"] += 0.02 * pd.to_numeric(work["consensus_admitted"], errors="coerce").fillna(0.0)
    low_score_mask = pd.to_numeric(work["final_select_score"], errors="coerce").fillna(0.0) < MIN_FINAL_SELECT_SCORE
    counters["dropped_by_low_final_score"] = int(low_score_mask.sum())
    work = work.loc[~low_score_mask].copy()
    if work.empty:
        raise RuntimeError("No alpha remained after final score floor")
    work = work.sort_values(["final_select_score", "abs_ic", "alpha"], ascending=[False, False, True]).reset_index(drop=True)
    if CURRENT_FAMILY_CAP > 0:
        work["family_rank"] = work.groupby("family", sort=False).cumcount() + 1
        fam_drop_mask = work["family_rank"] > CURRENT_FAMILY_CAP
        counters["dropped_by_family_dominance"] = int(fam_drop_mask.sum())
        work = work.loc[~fam_drop_mask].copy()
    if work.empty:
        raise RuntimeError("No alpha remained after family cap")
    return work.reset_index(drop=True), counters


def corr_prune_selected(train_df: pd.DataFrame, selected: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    if selected.empty:
        return selected.copy(), 0
    cols = [c for c in selected["alpha"].tolist() if c in train_df.columns]
    if len(cols) <= 1:
        return selected.copy(), 0
    corr = train_df[cols].fillna(0.0).corr(method="spearman", min_periods=200)
    keep: List[str] = []
    removed = 0
    for alpha in cols:
        ok = True
        for prev in keep:
            val = corr.loc[alpha, prev]
            if pd.notna(val) and abs(float(val)) >= CORR_PRUNE:
                ok = False
                removed += 1
                break
        if ok:
            keep.append(alpha)
        if len(keep) >= MAX_ALPHAS:
            break
    out = selected[selected["alpha"].isin(set(keep))].copy().reset_index(drop=True)
    return out, int(removed)


def fit_ridge_weights(train_df: pd.DataFrame, selected_alphas: Sequence[str]) -> pd.DataFrame:
    if not selected_alphas:
        raise RuntimeError("Selected alpha list is empty")
    x = train_df[list(selected_alphas)].copy().apply(_safe_numeric).fillna(0.0).to_numpy(dtype="float64")
    y = _safe_numeric(train_df[TARGET_COL]).fillna(0.0).to_numpy(dtype="float64")
    valid_mask = np.isfinite(y)
    x = x[valid_mask]
    y = y[valid_mask]
    if len(y) < MIN_TRAIN_ROWS:
        raise RuntimeError(f"Too few usable train rows for ridge: {len(y)}")
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    x_std = np.where(x_std <= EPS, 1.0, x_std)
    xz = (x - x_mean) / x_std
    yz = (y - y.mean()) / (y.std() + EPS)
    xtx = xz.T @ xz
    xty = xz.T @ yz
    coef_std = np.linalg.solve(xtx + RIDGE_L2 * np.eye(xtx.shape[0], dtype="float64"), xty)
    out = pd.DataFrame({"alpha": list(selected_alphas), "coef_std": coef_std, "x_mean": x_mean, "x_std": x_std})
    out["abs_coef_std"] = out["coef_std"].abs()
    total_abs = float(out["abs_coef_std"].sum())
    out["blend_weight"] = out["coef_std"] / (total_abs + EPS)
    out = out.sort_values(["abs_coef_std", "alpha"], ascending=[False, True]).reset_index(drop=True)
    return out


def apply_weights(test_df: pd.DataFrame, weights_df: pd.DataFrame) -> pd.DataFrame:
    out = test_df.copy()
    score = np.zeros(len(out), dtype="float64")
    for _, row in weights_df.iterrows():
        alpha = str(row["alpha"])
        score += float(row["blend_weight"]) * _safe_numeric(out[alpha]).fillna(0.0).to_numpy(dtype="float64")
    out["score_model"] = score
    out["score_model_z"] = out.groupby("date", sort=False)["score_model"].transform(_robust_zscore_series)
    return out


def build_portfolio(scored_df: pd.DataFrame, shell: ShellVariant) -> pd.DataFrame:
    base = scored_df.copy()
    base["score"] = _safe_numeric(base["score_model_z"])
    out = apply_holding_inertia(base, enter_pct=shell.enter_pct, exit_pct=shell.exit_pct)
    if "side" not in out.columns:
        raise RuntimeError("Holding inertia did not return side column")
    out["raw_strength"] = _safe_numeric(out["score_model_z"]).abs().fillna(0.0)
    out.loc[out["side"] == 0, "raw_strength"] = 0.0
    pieces: List[pd.DataFrame] = []
    for _, g in out.groupby("date", sort=False):
        gg = g.copy()
        gg["weight"] = 0.0
        long_strength = float(gg.loc[gg["side"] > 0, "raw_strength"].sum())
        short_strength = float(gg.loc[gg["side"] < 0, "raw_strength"].sum())
        if long_strength > EPS:
            gg.loc[gg["side"] > 0, "weight"] = 0.5 * gg.loc[gg["side"] > 0, "raw_strength"] / long_strength
        if short_strength > EPS:
            gg.loc[gg["side"] < 0, "weight"] = -0.5 * gg.loc[gg["side"] < 0, "raw_strength"] / short_strength
        gg["weight"] = gg["weight"].clip(lower=-shell.weight_cap, upper=shell.weight_cap)
        gross = float(gg["weight"].abs().sum())
        if gross > EPS:
            gg["weight"] = gg["weight"] * (shell.gross_target / gross)
        pieces.append(gg)
    out = pd.concat(pieces, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    out = cap_daily_turnover(out, weight_col="weight", max_daily_turnover=shell.max_daily_turnover)
    if "turnover" not in out.columns:
        out = out.sort_values(["symbol", "date"]).reset_index(drop=True)
        out["prev_weight"] = out.groupby("symbol", sort=False)["weight"].shift(1).fillna(0.0)
        out["turnover"] = (pd.to_numeric(out["weight"], errors="coerce") - pd.to_numeric(out["prev_weight"], errors="coerce")).abs()
    out["shell_name"] = shell.name
    out["shell_enter_pct"] = float(shell.enter_pct)
    out["shell_exit_pct"] = float(shell.exit_pct)
    out["shell_weight_cap"] = float(shell.weight_cap)
    out["shell_gross_target"] = float(shell.gross_target)
    out["shell_max_daily_turnover"] = float(shell.max_daily_turnover)
    return out


def evaluate_portfolio(port_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    daily = port_df.groupby("date", sort=False).agg(
        turnover=("turnover", "sum"),
        gross=("weight", lambda s: float(np.sum(np.abs(pd.to_numeric(s, errors="coerce"))))),
        shell_name=("shell_name", "first"),
        shell_enter_pct=("shell_enter_pct", "first"),
        shell_exit_pct=("shell_exit_pct", "first"),
        shell_weight_cap=("shell_weight_cap", "first"),
        shell_gross_target=("shell_gross_target", "first"),
        shell_max_daily_turnover=("shell_max_daily_turnover", "first"),
    ).reset_index()
    pnl = port_df.copy()
    pnl["gross_pnl"] = _safe_numeric(pnl["weight"]) * _safe_numeric(pnl[TARGET_COL])
    gross_by_day = pnl.groupby("date", sort=False)["gross_pnl"].sum().rename("gross_ret").reset_index()
    daily = daily.merge(gross_by_day, on="date", how="left")
    daily["cost_ret"] = pd.to_numeric(daily["turnover"], errors="coerce") * (COST_BPS / 10000.0)
    daily["net_ret"] = pd.to_numeric(daily["gross_ret"], errors="coerce") - pd.to_numeric(daily["cost_ret"], errors="coerce")
    daily["equity"] = (1.0 + daily["net_ret"].fillna(0.0)).cumprod()
    daily["cum_ret"] = daily["equity"] - 1.0
    mean = float(daily["net_ret"].mean()) if len(daily) else float("nan")
    std = float(daily["net_ret"].std(ddof=0)) if len(daily) else float("nan")
    sharpe = float((mean / (std + EPS)) * np.sqrt(252.0)) if len(daily) else float("nan")
    summary = {
        "days": float(len(daily)),
        "mean_daily": mean,
        "std_daily": std,
        "sharpe": sharpe,
        "cum_ret": float(daily["cum_ret"].iloc[-1]) if len(daily) else float("nan"),
        "max_drawdown": float((daily["equity"] / daily["equity"].cummax() - 1.0).min()) if len(daily) else float("nan"),
        "avg_turnover": float(daily["turnover"].mean()) if len(daily) else float("nan"),
        "avg_gross": float(daily["gross"].mean()) if len(daily) else float("nan"),
    }
    return daily, summary


def shell_variants() -> List[ShellVariant]:
    return [
        ShellVariant(
            name="base_hardened",
            enter_pct=BASE_ENTER_PCT,
            exit_pct=BASE_EXIT_PCT,
            weight_cap=BASE_WEIGHT_CAP,
            gross_target=BASE_GROSS_TARGET,
            max_daily_turnover=BASE_MAX_DAILY_TURNOVER,
        ),
        ShellVariant(
            name="looser_turnover",
            enter_pct=BASE_ENTER_PCT,
            exit_pct=BASE_EXIT_PCT,
            weight_cap=BASE_WEIGHT_CAP,
            gross_target=BASE_GROSS_TARGET,
            max_daily_turnover=max(BASE_MAX_DAILY_TURNOVER, 0.35),
        ),
        ShellVariant(
            name="looser_inertia",
            enter_pct=max(BASE_ENTER_PCT, 0.10),
            exit_pct=max(BASE_EXIT_PCT, 0.22),
            weight_cap=BASE_WEIGHT_CAP,
            gross_target=BASE_GROSS_TARGET,
            max_daily_turnover=BASE_MAX_DAILY_TURNOVER,
        ),
    ]


def run_fold(
    df: pd.DataFrame,
    split: WFSplit,
    alpha_cols: Sequence[str],
    family_map: Dict[str, str],
    history_stats: pd.DataFrame,
    is_last_fold: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object], Dict[str, float], Dict[str, pd.DataFrame], Dict[str, Dict[str, float]]]:
    train_df = df.loc[(df["date"] >= split.train_start) & (df["date"] <= split.train_end)].copy()
    test_df = df.loc[(df["date"] >= split.test_start) & (df["date"] <= split.test_end)].copy()
    if len(train_df) < MIN_TRAIN_ROWS:
        raise RuntimeError(f"Fold {split.fold_id}: train rows too small: {len(train_df)}")
    if len(test_df) < MIN_TEST_ROWS:
        raise RuntimeError(f"Fold {split.fold_id}: test rows too small: {len(test_df)}")
    train_df = _cs_zscore_df(train_df, alpha_cols)
    test_df = _cs_zscore_df(test_df, alpha_cols)
    current_stats = _alpha_stats(train_df, alpha_cols, split.fold_id, family_map)
    history_consensus = build_consensus_from_history(history_stats)
    candidate_df, counters = select_current_fold_alphas(current_stats, history_consensus, is_last_fold=is_last_fold)
    candidate_df, corr_removed = corr_prune_selected(train_df, candidate_df)
    counters["corr_pruned"] = int(corr_removed)
    selected = candidate_df["alpha"].head(MAX_ALPHAS).tolist()
    counters["selected"] = int(len(selected))
    if len(selected) < MIN_SELECTED_ALPHAS:
        raise RuntimeError(f"Fold {split.fold_id}: selected alpha list too small after hardening: {len(selected)} < {MIN_SELECTED_ALPHAS}")
    weights_df = fit_ridge_weights(train_df, selected)
    scored = apply_weights(test_df, weights_df)

    shell_daily_map: Dict[str, pd.DataFrame] = {}
    shell_summary_map: Dict[str, Dict[str, float]] = {}
    for shell in shell_variants():
        port = build_portfolio(scored, shell)
        daily, summary = evaluate_portfolio(port)
        daily["fold_id"] = split.fold_id
        daily["train_start"] = split.train_start
        daily["train_end"] = split.train_end
        daily["test_start"] = split.test_start
        daily["test_end"] = split.test_end
        summary = dict(summary)
        summary["selected_alpha_count"] = float(len(selected))
        summary["selected_alpha_list"] = ",".join(selected)
        summary["fold_alpha_input_count"] = float(len(alpha_cols))
        summary["fold_id"] = float(split.fold_id)
        summary["shell_name"] = shell.name
        summary["shell_enter_pct"] = float(shell.enter_pct)
        summary["shell_exit_pct"] = float(shell.exit_pct)
        summary["shell_weight_cap"] = float(shell.weight_cap)
        summary["shell_gross_target"] = float(shell.gross_target)
        summary["shell_max_daily_turnover"] = float(shell.max_daily_turnover)
        shell_daily_map[shell.name] = daily
        shell_summary_map[shell.name] = summary
    return current_stats, history_consensus, candidate_df, weights_df, counters, shell_summary_map["base_hardened"], shell_daily_map, shell_summary_map


def main() -> int:
    _enable_line_buffering()
    print(f"[CFG] alpha_lib_file={ALPHA_LIB_FILE}")
    print(f"[CFG] alpha_shortlist_csv={ALPHA_SHORTLIST_CSV}")
    print(f"[CFG] alpha_shortlist_required={int(ALPHA_SHORTLIST_REQUIRED)}")
    print(f"[CFG] consensus_mode={CONSENSUS_MODE}")
    print(f"[CFG] consensus_required_last_fold={int(CONSENSUS_REQUIRED_LAST_FOLD)}")
    print(f"[CFG] consensus_min_folds={CONSENSUS_MIN_FOLDS} consensus_max_sign_flip={CONSENSUS_MAX_SIGN_FLIP} consensus_min_mean_abs_ic={CONSENSUS_MIN_MEAN_ABS_IC}")
    print(f"[CFG] consensus_history_blend={CONSENSUS_HISTORY_BLEND} consensus_family_cap={CONSENSUS_FAMILY_CAP} current_family_cap={CURRENT_FAMILY_CAP}")
    print(f"[CFG] hardening max_alphas={MAX_ALPHAS} min_selected_alphas={MIN_SELECTED_ALPHAS} min_final_select_score={MIN_FINAL_SELECT_SCORE} corr_prune={CORR_PRUNE}")
    print(f"[CFG] base_shell gross_target={BASE_GROSS_TARGET} weight_cap={BASE_WEIGHT_CAP} max_daily_turnover={BASE_MAX_DAILY_TURNOVER} enter_pct={BASE_ENTER_PCT} exit_pct={BASE_EXIT_PCT}")

    df, alpha_cols, shortlist_counters, shortlist_df, family_map = load_alpha_library()
    print(f"[DATA] rows={len(df)} dates={df['date'].nunique()} symbols={df['symbol'].nunique()} alpha_cols={len(alpha_cols)}")
    print("[ALPHA_SHORTLIST]")
    print(json.dumps(shortlist_counters, ensure_ascii=False, indent=2))
    if len(shortlist_df):
        print("[ALPHA_SHORTLIST][HEAD]")
        preview_cols = [c for c in ["shortlist_rank", "alpha", "family", "wave", "transform", "interaction", "regime", "selector_score"] if c in shortlist_df.columns]
        if not preview_cols:
            preview_cols = ["alpha"]
        print(shortlist_df[preview_cols].head(20).to_string(index=False))

    splits = build_walkforward_splits(df["date"])
    print(f"[WF] folds={len(splits)}")
    for sp in splits:
        print(f"[WF][FOLD {sp.fold_id}] train={sp.train_start.date()}..{sp.train_end.date()} test={sp.test_start.date()}..{sp.test_end.date()}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_history_stats: List[pd.DataFrame] = []
    fold_summaries: List[Dict[str, float]] = []
    fold_debug_rows: List[Dict[str, object]] = []
    shell_summary_rows: List[Dict[str, float]] = []
    shell_daily_map_all: Dict[str, List[pd.DataFrame]] = {shell.name: [] for shell in shell_variants()}

    shortlist_debug_path = OUT_DIR / "wf_alpha_shortlist_debug.json"
    shortlist_debug_path.write_text(json.dumps(shortlist_counters, ensure_ascii=False, indent=2), encoding="utf-8")

    for idx, split in enumerate(splits, start=1):
        is_last_fold = idx == len(splits)
        history_df = pd.concat(all_history_stats, ignore_index=True) if all_history_stats else pd.DataFrame()
        print(f"[WF][FOLD {split.fold_id}] start history_rows={len(history_df)} is_last_fold={int(is_last_fold)}")
        current_stats, history_consensus, candidate_df, weights_df, counters, base_summary, fold_shell_daily, fold_shell_summary = run_fold(df, split, alpha_cols, family_map, history_df, is_last_fold=is_last_fold)
        all_history_stats.append(current_stats)
        fold_summaries.append(base_summary)
        fold_debug_rows.append({"fold_id": split.fold_id, **counters})

        current_stats.to_csv(OUT_DIR / f"wf_alpha_stats__fold{split.fold_id}.csv", index=False)
        history_consensus.to_csv(OUT_DIR / f"wf_consensus__fold{split.fold_id}.csv", index=False)
        candidate_df.to_csv(OUT_DIR / f"wf_selected_alphas__fold{split.fold_id}.csv", index=False)
        weights_df.to_csv(OUT_DIR / f"wf_alpha_weights__fold{split.fold_id}.csv", index=False)

        for shell_name, daily in fold_shell_daily.items():
            shell_daily_map_all[shell_name].append(daily)
            daily.to_csv(OUT_DIR / f"wf_daily__fold{split.fold_id}__{shell_name}.csv", index=False)
            shell_summary_rows.append(fold_shell_summary[shell_name])
            print(f"[WF][FOLD {split.fold_id}][{shell_name}] sharpe={fold_shell_summary[shell_name]['sharpe']:.4f} mean_daily={fold_shell_summary[shell_name]['mean_daily']:.6f} cum_ret={fold_shell_summary[shell_name]['cum_ret']:.4f} maxdd={fold_shell_summary[shell_name]['max_drawdown']:.4f} avg_turn={fold_shell_summary[shell_name]['avg_turnover']:.4f}")

        print(f"[WF][FOLD {split.fold_id}][CONSENSUS] selected={counters['selected']} dropped_by_fold_consensus={counters['dropped_by_fold_consensus']} dropped_by_sign_flip={counters['dropped_by_sign_flip']} dropped_by_family_dominance={counters['dropped_by_family_dominance']} dropped_by_low_final_score={counters['dropped_by_low_final_score']} corr_pruned={counters['corr_pruned']}")
        print(f"[WF][FOLD {split.fold_id}][TOP_WEIGHTS]")
        print(weights_df.head(TOPK_DEBUG).to_string(index=False))

    pd.DataFrame(fold_summaries).to_csv(OUT_DIR / "wf_fold_summaries.csv", index=False)
    pd.DataFrame(fold_debug_rows).to_csv(OUT_DIR / "wf_fold_consensus_debug.csv", index=False)
    shell_summary_df = pd.DataFrame(shell_summary_rows)
    shell_summary_df.to_csv(OUT_DIR / "wf_shell_fold_summaries.csv", index=False)

    shell_overall_rows: List[Dict[str, float]] = []
    for shell in shell_variants():
        shell_daily = pd.concat(shell_daily_map_all[shell.name], ignore_index=True).sort_values(["date", "fold_id"]).reset_index(drop=True)
        shell_daily.to_csv(OUT_DIR / f"wf_multi_alpha_overall__{shell.name}.csv", index=False)
        equity = (1.0 + shell_daily["net_ret"].fillna(0.0)).cumprod()
        row = {
            "shell_name": shell.name,
            "days": float(len(shell_daily)),
            "mean_daily": float(shell_daily["net_ret"].mean()),
            "std_daily": float(shell_daily["net_ret"].std(ddof=0)),
            "sharpe": float((shell_daily["net_ret"].mean() / (shell_daily["net_ret"].std(ddof=0) + EPS)) * np.sqrt(252.0)),
            "cum_ret": float(equity.iloc[-1] - 1.0) if len(equity) else float("nan"),
            "max_drawdown": float((equity / equity.cummax() - 1.0).min()) if len(equity) else float("nan"),
            "avg_turnover": float(shell_daily["turnover"].mean()) if len(shell_daily) else float("nan"),
            "folds": int(len(splits)),
            "enter_pct": float(shell.enter_pct),
            "exit_pct": float(shell.exit_pct),
            "weight_cap": float(shell.weight_cap),
            "gross_target": float(shell.gross_target),
            "max_daily_turnover": float(shell.max_daily_turnover),
        }
        shell_overall_rows.append(row)

    shell_overall_df = pd.DataFrame(shell_overall_rows).sort_values(["sharpe", "cum_ret", "shell_name"], ascending=[False, False, True]).reset_index(drop=True)
    shell_overall_df.to_csv(OUT_DIR / "wf_shell_comparison.csv", index=False)
    best_shell = shell_overall_df.iloc[0].to_dict() if len(shell_overall_df) else {}

    meta = {
        "alpha_lib_file": str(ALPHA_LIB_FILE),
        "alpha_shortlist_csv": str(ALPHA_SHORTLIST_CSV),
        "alpha_shortlist_required": int(ALPHA_SHORTLIST_REQUIRED),
        "consensus_mode": CONSENSUS_MODE,
        "consensus_required_last_fold": int(CONSENSUS_REQUIRED_LAST_FOLD),
        "consensus_min_folds": CONSENSUS_MIN_FOLDS,
        "consensus_max_sign_flip": CONSENSUS_MAX_SIGN_FLIP,
        "consensus_min_mean_abs_ic": CONSENSUS_MIN_MEAN_ABS_IC,
        "consensus_history_blend": CONSENSUS_HISTORY_BLEND,
        "consensus_family_cap": CONSENSUS_FAMILY_CAP,
        "current_family_cap": CURRENT_FAMILY_CAP,
        "max_alphas": MAX_ALPHAS,
        "min_selected_alphas": MIN_SELECTED_ALPHAS,
        "min_final_select_score": MIN_FINAL_SELECT_SCORE,
        "shell_variants": [shell.__dict__ for shell in shell_variants()],
        "best_shell": best_shell,
        "alpha_shortlist_debug": shortlist_counters,
    }
    meta_path = OUT_DIR / "wf_multi_alpha_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if len(shell_overall_df):
        print("[SHELL_COMPARISON]")
        print(shell_overall_df.to_string(index=False))
        print(f"[BEST_SHELL] name={best_shell.get('shell_name', '')} sharpe={best_shell.get('sharpe', float('nan')):.4f} cum_ret={best_shell.get('cum_ret', float('nan')):.4f} avg_turnover={best_shell.get('avg_turnover', float('nan')):.4f}")
    print(f"[ARTIFACT] {OUT_DIR / 'wf_fold_summaries.csv'}")
    print(f"[ARTIFACT] {OUT_DIR / 'wf_fold_consensus_debug.csv'}")
    print(f"[ARTIFACT] {OUT_DIR / 'wf_shell_fold_summaries.csv'}")
    print(f"[ARTIFACT] {OUT_DIR / 'wf_shell_comparison.csv'}")
    print(f"[ARTIFACT] {meta_path}")
    print(f"[ARTIFACT] {shortlist_debug_path}")
    print("[FINAL] multi-alpha walkforward shell comparison complete")
    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 1
    _safe_exit(rc)