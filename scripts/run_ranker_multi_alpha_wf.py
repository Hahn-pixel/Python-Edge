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
for _p in [ROOT, SRC_DIR]:
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

from python_edge.portfolio.holding_inertia import apply_holding_inertia
from python_edge.portfolio.turnover_control import cap_daily_turnover

try:
    from python_edge.portfolio.position_limits import apply_position_filters, cap_final_weight, renormalize_after_caps, normalize_gross_exposure
    _HAS_POSITION_LIMITS = True
except Exception:
    _HAS_POSITION_LIMITS = False

    def apply_position_filters(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        out = df.copy()
        out["dbg_position_limits_fallback"] = 1
        return out

    def cap_final_weight(df: pd.DataFrame, weight_col: str = "weight", **kwargs) -> pd.DataFrame:
        out = df.copy()
        out["dbg_position_limits_fallback"] = 1
        return out

    def renormalize_after_caps(df: pd.DataFrame, weight_col: str = "weight", gross_target: float = 1.0, **kwargs) -> pd.DataFrame:
        out = df.copy()
        gross = out.groupby("date", sort=False)[weight_col].transform(lambda s: float(np.sum(np.abs(pd.to_numeric(s, errors="coerce")))))
        scale = np.where(gross > EPS, gross_target / gross, 1.0)
        out[weight_col] = pd.to_numeric(out[weight_col], errors="coerce") * scale
        out["dbg_position_limits_fallback"] = 1
        return out

    def normalize_gross_exposure(df: pd.DataFrame, side_col: str = "side", gross_target: float = 1.0, out_col: str = "weight", **kwargs) -> pd.DataFrame:
        out = df.copy()
        out[out_col] = pd.to_numeric(out[side_col], errors="coerce").fillna(0.0)
        gross = out.groupby("date", sort=False)[out_col].transform(lambda s: float(np.sum(np.abs(pd.to_numeric(s, errors="coerce")))))
        scale = np.where(gross > EPS, gross_target / gross, 0.0)
        out[out_col] = pd.to_numeric(out[out_col], errors="coerce") * scale
        out["dbg_position_limits_fallback"] = 1
        return out

try:
    from python_edge.execution.cost_model import attach_execution_costs
    _HAS_COST_MODEL = True
except Exception:
    _HAS_COST_MODEL = False

    def attach_execution_costs(df: pd.DataFrame, weight_col: str = "weight", **kwargs) -> pd.DataFrame:
        out = df.copy()
        if "turnover" not in out.columns:
            out = out.sort_values(["symbol", "date"]).reset_index(drop=True)
            out["prev_weight"] = out.groupby("symbol", sort=False)[weight_col].shift(1).fillna(0.0)
            out["turnover"] = (pd.to_numeric(out[weight_col], errors="coerce") - pd.to_numeric(out["prev_weight"], errors="coerce")).abs()
        cost_bps = float(kwargs.get("fee_bps", 0.0)) + float(kwargs.get("slippage_bps", 0.0)) + float(kwargs.get("spread_bps", 0.0)) + float(kwargs.get("impact_bps", 0.0))
        out["exec_cost_ret"] = pd.to_numeric(out["turnover"], errors="coerce") * (cost_bps / 10000.0)
        out["dbg_cost_model_fallback"] = 1
        return out


ALPHA_LIB_FILE = Path(os.getenv("ALPHA_LIB_FILE", r"data\alpha_library_v2\alpha_library_v2.parquet"))
ALPHA_SHORTLIST_CSV = Path(os.getenv("ALPHA_SHORTLIST_CSV", r"data\alpha_library_v2\diagnostics\alpha_candidate_shortlist.csv"))
ALPHA_SHORTLIST_REQUIRED = str(os.getenv("ALPHA_SHORTLIST_REQUIRED", "1")).strip().lower() not in {"0", "false", "no", "off"}
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
MIN_ALPHA_DAYS = int(os.getenv("MIN_ALPHA_DAYS", "50"))
MIN_ALPHA_ABS_IC = float(os.getenv("MIN_ALPHA_ABS_IC", "0.004"))
MIN_STABILITY_BLOCKS = int(os.getenv("MIN_STABILITY_BLOCKS", "3"))
MIN_SIGN_CONSISTENCY = float(os.getenv("MIN_SIGN_CONSISTENCY", "0.67"))
MIN_BLOCK_IC_ABS = float(os.getenv("MIN_BLOCK_IC_ABS", "0.0015"))
MAX_ALPHAS = int(os.getenv("MAX_ALPHAS", "12"))
RIDGE_L2 = float(os.getenv("RIDGE_L2", "20.0"))
ENTER_PCT = float(os.getenv("ENTER_PCT", "0.08"))
EXIT_PCT = float(os.getenv("EXIT_PCT", "0.16"))
GROSS_TARGET = float(os.getenv("GROSS_TARGET", "1.0"))
WEIGHT_CAP = float(os.getenv("WEIGHT_CAP", "0.06"))
MAX_DAILY_TURNOVER = float(os.getenv("MAX_DAILY_TURNOVER", "0.35"))
COST_BPS = float(os.getenv("COST_BPS", "8.0"))
BORROW_BPS_DAILY = float(os.getenv("BORROW_BPS_DAILY", "1.0"))
LOW_PRICE_MIN = float(os.getenv("LOW_PRICE_MIN", "5.0"))
LOW_DV_MIN = float(os.getenv("LOW_DV_MIN", "1000000"))
MAX_ADV_PARTICIPATION = float(os.getenv("MAX_ADV_PARTICIPATION", "0.05"))
PORTFOLIO_NOTIONAL = float(os.getenv("PORTFOLIO_NOTIONAL", "1.0"))
FEE_BPS = float(os.getenv("FEE_BPS", "1.0"))
SLIPPAGE_BPS = float(os.getenv("SLIPPAGE_BPS", "2.0"))
SPREAD_BPS = float(os.getenv("SPREAD_BPS", "3.0"))
IMPACT_BPS = float(os.getenv("IMPACT_BPS", "8.0"))
LOW_PRICE_PENALTY_BPS = float(os.getenv("LOW_PRICE_PENALTY_BPS", "4.0"))
HTB_BORROW_BPS_DAILY = float(os.getenv("HTB_BORROW_BPS_DAILY", "8.0"))
SIDE = str(os.getenv("SIDE", "long_short")).strip().lower()
TOPK_DEBUG = int(os.getenv("TOPK_DEBUG", "10"))
CORR_PRUNE = float(os.getenv("CORR_PRUNE", "0.85"))
BUDGET_MODE = str(os.getenv("BUDGET_MODE", "dynamic")).strip().lower()
REGIME_MODE = str(os.getenv("REGIME_MODE", "dynamic")).strip().lower()
BREADTH_STRONG = float(os.getenv("BREADTH_STRONG", "0.58"))
BREADTH_WEAK = float(os.getenv("BREADTH_WEAK", "0.42"))
PRESSURE_STRONG = float(os.getenv("PRESSURE_STRONG", "0.15"))
PRESSURE_WEAK = float(os.getenv("PRESSURE_WEAK", "-0.15"))
VOLSHOCK_STRONG = float(os.getenv("VOLSHOCK_STRONG", "0.25"))
TOP_PCT_STRONG = float(os.getenv("TOP_PCT_STRONG", "0.06"))
TOP_PCT_NEUTRAL = float(os.getenv("TOP_PCT_NEUTRAL", "0.08"))
TOP_PCT_WEAK = float(os.getenv("TOP_PCT_WEAK", "0.10"))
LONG_MULT_STRONG = float(os.getenv("LONG_MULT_STRONG", "1.20"))
LONG_MULT_WEAK = float(os.getenv("LONG_MULT_WEAK", "0.85"))
SHORT_MULT_STRONG = float(os.getenv("SHORT_MULT_STRONG", "1.20"))
SHORT_MULT_WEAK = float(os.getenv("SHORT_MULT_WEAK", "0.85"))
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


def _rank_abs_pct_by_date(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby("date", sort=False)[col].transform(lambda s: _safe_numeric(s).abs().rank(method="average", pct=True))


def _must_exist(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _load_shortlist_required() -> Tuple[List[str], Dict[str, object], pd.DataFrame]:
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
        return [], counters, pd.DataFrame(columns=["alpha"])

    shortlist_df = pd.read_csv(ALPHA_SHORTLIST_CSV)
    if "alpha" not in shortlist_df.columns:
        raise RuntimeError(f"Shortlist csv missing 'alpha' column: {ALPHA_SHORTLIST_CSV}")

    counters["shortlist_rows_raw"] = int(len(shortlist_df))
    shortlist_df["alpha"] = shortlist_df["alpha"].astype(str).str.strip()
    missing_name_mask = shortlist_df["alpha"].eq("") | shortlist_df["alpha"].eq("nan")
    counters["shortlist_alpha_missing_name"] = int(missing_name_mask.sum())
    shortlist_df = shortlist_df.loc[~missing_name_mask].copy()
    shortlist_df = shortlist_df.drop_duplicates(subset=["alpha"]).reset_index(drop=True)
    counters["shortlist_rows_unique"] = int(len(shortlist_df))
    counters["shortlist_duplicates_removed"] = int(int(counters["shortlist_rows_raw"]) - int(counters["shortlist_alpha_missing_name"]) - int(counters["shortlist_rows_unique"]))

    if ALPHA_SHORTLIST_REQUIRED and shortlist_df.empty:
        raise RuntimeError(f"ALPHA_SHORTLIST_REQUIRED=1 but shortlist is empty after cleanup: {ALPHA_SHORTLIST_CSV}")
    return shortlist_df["alpha"].tolist(), counters, shortlist_df


# ------------------------------------------------------------
# LOAD / SPLITS
# ------------------------------------------------------------

def load_alpha_library() -> Tuple[pd.DataFrame, List[str], Dict[str, object], pd.DataFrame]:
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

    alpha_cols_before = _alpha_cols(df)
    if not alpha_cols_before:
        raise RuntimeError("No alpha_ columns found in alpha library")

    shortlist_cols, shortlist_counters, shortlist_df = _load_shortlist_required()
    shortlist_set = set(shortlist_cols)
    found = [c for c in alpha_cols_before if c in shortlist_set]
    missing_in_parquet = [c for c in shortlist_cols if c not in set(alpha_cols_before)]

    shortlist_counters["alpha_cols_before_shortlist"] = int(len(alpha_cols_before))
    shortlist_counters["shortlist_found_in_parquet"] = int(len(found))
    shortlist_counters["shortlist_missing_in_parquet"] = int(len(missing_in_parquet))
    shortlist_counters["shortlist_missing_in_parquet_list"] = missing_in_parquet

    if ALPHA_SHORTLIST_REQUIRED and not found:
        raise RuntimeError(
            f"ALPHA_SHORTLIST_REQUIRED=1 but no shortlist alphas were found in parquet; "
            f"shortlist_path={ALPHA_SHORTLIST_CSV} alpha_lib_file={ALPHA_LIB_FILE}"
        )

    if ALPHA_SHORTLIST_REQUIRED and missing_in_parquet:
        raise RuntimeError(
            f"ALPHA_SHORTLIST_REQUIRED=1 and shortlist has missing alpha columns in parquet: "
            f"missing_count={len(missing_in_parquet)} first_missing={missing_in_parquet[:10]}"
        )

    if shortlist_cols:
        alpha_cols = found
    else:
        alpha_cols = alpha_cols_before

    if not alpha_cols:
        raise RuntimeError("No alpha columns remain after shortlist filtering")

    keep_cols = [c for c in df.columns if not c.startswith("alpha_")] + alpha_cols
    df = df[keep_cols].copy()
    shortlist_counters["alpha_cols_after_shortlist"] = int(len(alpha_cols))
    shortlist_counters["shortlist_filter_applied"] = int(bool(shortlist_cols))
    return df, alpha_cols, shortlist_counters, shortlist_df


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

def _daily_ic(train_df: pd.DataFrame, factor_col: str, target_col: str, min_cs: int = 20) -> Tuple[float, int, float]:
    vals: List[float] = []
    pos_days = 0
    used_days = 0
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
            used_days += 1
            if float(ic) > 0:
                pos_days += 1
    if not vals:
        return float("nan"), 0, float("nan")
    return float(np.nanmean(vals)), int(len(vals)), float(pos_days / max(1, used_days))


def _block_dates(date_index: pd.Index, n_blocks: int = 4) -> List[pd.Index]:
    uniq = pd.Index(sorted(pd.to_datetime(pd.Series(date_index)).dt.normalize().unique()))
    if len(uniq) == 0:
        return []
    parts = np.array_split(np.arange(len(uniq)), n_blocks)
    out: List[pd.Index] = []
    for p in parts:
        if len(p) == 0:
            continue
        out.append(pd.Index(uniq[p]))
    return out


def _alpha_stability_stats(train_df: pd.DataFrame, factor_col: str, target_col: str) -> Dict[str, float]:
    blocks = _block_dates(train_df["date"], 4)
    block_ics: List[float] = []
    for block_dates in blocks:
        block_df = train_df.loc[train_df["date"].isin(block_dates)].copy()
        ic, n_days, _ = _daily_ic(block_df, factor_col, target_col)
        if n_days > 0 and pd.notna(ic):
            block_ics.append(float(ic))
    if not block_ics:
        return {
            "block_count": 0.0,
            "block_hit_count": 0.0,
            "sign_consistency": float("nan"),
            "block_ic_mean": float("nan"),
            "block_ic_median": float("nan"),
            "block_ic_worst": float("nan"),
        }
    signs = [np.sign(x) for x in block_ics if abs(x) >= MIN_BLOCK_IC_ABS]
    pos = sum(1 for s in signs if s > 0)
    neg = sum(1 for s in signs if s < 0)
    sign_consistency = max(pos, neg) / max(1, len(signs)) if signs else 0.0
    return {
        "block_count": float(len(block_ics)),
        "block_hit_count": float(sum(1 for x in block_ics if abs(x) >= MIN_BLOCK_IC_ABS)),
        "sign_consistency": float(sign_consistency),
        "block_ic_mean": float(np.mean(block_ics)),
        "block_ic_median": float(np.median(block_ics)),
        "block_ic_worst": float(min(block_ics, key=lambda x: abs(x))),
    }


def select_alphas(train_df: pd.DataFrame, alpha_cols: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for col in alpha_cols:
        ic, n_days, pos_rate = _daily_ic(train_df, col, TARGET_COL)
        stab = _alpha_stability_stats(train_df, col, TARGET_COL)
        row = {
            "alpha": col,
            "ic": ic,
            "abs_ic": abs(ic) if pd.notna(ic) else np.nan,
            "n_days": n_days,
            "pos_rate": pos_rate,
            **stab,
        }
        row["stability_score"] = (
            (0.55 * (row["abs_ic"] if pd.notna(row["abs_ic"]) else 0.0))
            + (0.20 * (row["block_ic_mean"] if pd.notna(row["block_ic_mean"]) else 0.0))
            + (0.15 * (row["sign_consistency"] if pd.notna(row["sign_consistency"]) else 0.0))
            + (0.10 * (row["block_hit_count"] / 4.0 if pd.notna(row["block_hit_count"]) else 0.0))
        )
        rows.append(row)
    sel = pd.DataFrame(rows)
    if sel.empty:
        raise RuntimeError("select_alphas: no alpha diagnostics produced")
    sel = sel.loc[(sel["n_days"] >= MIN_ALPHA_DAYS) & (sel["abs_ic"] >= MIN_ALPHA_ABS_IC)].copy()
    if len(sel):
        sel = sel.loc[(sel["pos_rate"] <= 0.45) | (sel["pos_rate"] >= 0.55)].copy()
        sel = sel.loc[(sel["block_hit_count"] >= MIN_STABILITY_BLOCKS) & (sel["sign_consistency"] >= MIN_SIGN_CONSISTENCY)].copy()
    if sel.empty:
        raise RuntimeError("select_alphas: no alpha passed min alpha thresholds")
    sel = sel.sort_values(["stability_score", "abs_ic", "n_days", "alpha"], ascending=[False, False, False, True]).reset_index(drop=True)

    picked: List[str] = []
    corr_source = train_df[[c for c in sel["alpha"].tolist() if c in train_df.columns]].copy().fillna(0.0)
    corr = corr_source.corr(method="spearman", min_periods=200)
    for alpha in sel["alpha"].tolist():
        if len(picked) >= MAX_ALPHAS:
            break
        ok = True
        for prev in picked:
            c = corr.loc[alpha, prev] if alpha in corr.index and prev in corr.columns else np.nan
            if pd.notna(c) and abs(float(c)) >= CORR_PRUNE:
                ok = False
                break
        if ok:
            picked.append(alpha)
    sel["selected"] = sel["alpha"].isin(picked).astype(int)
    sel = sel.sort_values(["selected", "stability_score", "abs_ic", "n_days", "alpha"], ascending=[False, False, False, False, True]).reset_index(drop=True)
    return sel


def fit_ridge_weights(train_df: pd.DataFrame, selected_alphas: Sequence[str]) -> pd.DataFrame:
    if not selected_alphas:
        raise RuntimeError("fit_ridge_weights: selected_alphas empty")

    xdf = train_df[list(selected_alphas)].copy().apply(_safe_numeric)
    y = _safe_numeric(train_df[TARGET_COL]).to_numpy(dtype="float64")

    valid_target = np.isfinite(y)
    active_counts = np.isfinite(xdf.to_numpy(dtype="float64")).sum(axis=1)

    candidate_min_active = [max(1, min(3, len(selected_alphas))), 2, 1]
    chosen_min_active = None
    usable_mask = None
    usable_rows = 0
    for min_active in candidate_min_active:
        mask = valid_target & (active_counts >= min_active)
        rows = int(mask.sum())
        if rows >= MIN_TRAIN_ROWS:
            chosen_min_active = min_active
            usable_mask = mask
            usable_rows = rows
            break
    if usable_mask is None:
        mask = valid_target
        rows = int(mask.sum())
        if rows <= 0:
            raise RuntimeError(f"fit_ridge_weights: zero usable train rows; selected_alphas={len(selected_alphas)}")
        chosen_min_active = 0
        usable_mask = mask
        usable_rows = rows

    x = xdf.loc[usable_mask, list(selected_alphas)].fillna(0.0).to_numpy(dtype="float64")
    y = y[usable_mask]
    if len(y) < MIN_TRAIN_ROWS:
        raise RuntimeError(
            f"fit_ridge_weights: too few usable train rows ({len(y)}) < {MIN_TRAIN_ROWS}; "
            f"selected_alphas={len(selected_alphas)} chosen_min_active={chosen_min_active} usable_rows={usable_rows}"
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
    out["chosen_min_active"] = float(chosen_min_active)
    out["usable_rows"] = float(usable_rows)
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


def attach_market_state(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "meta_price" not in out.columns and "close" in out.columns:
        out["meta_price"] = _safe_numeric(out["close"])
    if "meta_dollar_volume" not in out.columns:
        if "dollar_vol" in out.columns:
            out["meta_dollar_volume"] = _safe_numeric(out["dollar_vol"])
        elif all(c in out.columns for c in ["close", "volume"]):
            out["meta_dollar_volume"] = _safe_numeric(out["close"]) * _safe_numeric(out["volume"])
    if "market_breadth" not in out.columns:
        out["market_breadth"] = out.groupby("date", sort=False)[TARGET_COL].transform(lambda s: _safe_numeric(s).gt(0).mean())
    if "intraday_rs" in out.columns:
        out["market_intraday_rs"] = out.groupby("date", sort=False)["intraday_rs"].transform("mean")
    else:
        out["market_intraday_rs"] = 0.0
    if "intraday_pressure" in out.columns:
        out["market_intraday_pressure"] = out.groupby("date", sort=False)["intraday_pressure"].transform("mean")
    else:
        out["market_intraday_pressure"] = 0.0
    if "rel_volume_20" in out.columns:
        out["market_volume_shock"] = out.groupby("date", sort=False)["rel_volume_20"].transform("mean")
    else:
        out["market_volume_shock"] = 0.0
    return out


def attach_dynamic_allocations(df: pd.DataFrame) -> pd.DataFrame:
    out = attach_market_state(df)
    out["top_pct"] = TOP_PCT_NEUTRAL
    out["long_budget"] = 0.50
    out["short_budget"] = 0.50
    out["long_mult"] = 1.0
    out["short_mult"] = 1.0
    out["alloc_regime"] = "neutral"

    strong_long = ((_safe_numeric(out["market_breadth"]) >= BREADTH_STRONG) & (_safe_numeric(out["market_intraday_pressure"]) >= PRESSURE_STRONG))
    strong_short = ((_safe_numeric(out["market_breadth"]) <= BREADTH_WEAK) & (_safe_numeric(out["market_intraday_pressure"]) <= PRESSURE_WEAK))
    panic = _safe_numeric(out["market_volume_shock"]) >= VOLSHOCK_STRONG

    if REGIME_MODE == "dynamic":
        out.loc[panic, "top_pct"] = TOP_PCT_STRONG
        out.loc[~panic, "top_pct"] = TOP_PCT_NEUTRAL
        out.loc[strong_long | strong_short, "top_pct"] = TOP_PCT_WEAK

    if BUDGET_MODE == "dynamic":
        out.loc[strong_long, ["long_budget", "short_budget", "long_mult", "short_mult", "alloc_regime"]] = [0.60, 0.40, LONG_MULT_STRONG, SHORT_MULT_WEAK, "strong_long"]
        out.loc[strong_short, ["long_budget", "short_budget", "long_mult", "short_mult", "alloc_regime"]] = [0.40, 0.60, LONG_MULT_WEAK, SHORT_MULT_STRONG, "strong_short"]
        out.loc[panic & ~strong_long & ~strong_short, ["long_budget", "short_budget", "alloc_regime"]] = [0.50, 0.50, "panic"]

    return out


def build_portfolio(scored_df: pd.DataFrame) -> pd.DataFrame:
    out = attach_dynamic_allocations(scored_df.copy())
    out["score"] = _safe_numeric(out["score_model_z"])
    out = apply_holding_inertia(out, enter_pct=ENTER_PCT, exit_pct=EXIT_PCT)
    if "side" not in out.columns:
        raise RuntimeError("apply_holding_inertia did not return side column")

    out["raw_strength"] = out["score"].abs().fillna(0.0)
    out.loc[out["side"] == 0, "raw_strength"] = 0.0
    out.loc[out["side"] > 0, "raw_strength"] *= _safe_numeric(out["long_mult"])
    out.loc[out["side"] < 0, "raw_strength"] *= _safe_numeric(out["short_mult"])

    pieces: List[pd.DataFrame] = []
    for _, g in out.groupby("date", sort=False):
        gg = g.copy()
        pos_long = gg.loc[gg["side"] > 0, "raw_strength"]
        pos_short = gg.loc[gg["side"] < 0, "raw_strength"]
        long_sum = float(pos_long.sum())
        short_sum = float(pos_short.sum())
        gg["weight"] = 0.0
        long_budget = float(_safe_numeric(gg["long_budget"]).iloc[0]) if len(gg) else 0.5
        short_budget = float(_safe_numeric(gg["short_budget"]).iloc[0]) if len(gg) else 0.5
        if SIDE != "short_only" and long_sum > EPS:
            gg.loc[gg["side"] > 0, "weight"] = long_budget * gg.loc[gg["side"] > 0, "raw_strength"] / long_sum
        if SIDE != "long_only" and short_sum > EPS:
            gg.loc[gg["side"] < 0, "weight"] = -short_budget * gg.loc[gg["side"] < 0, "raw_strength"] / short_sum
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
    out = out.sort_values(["date", "symbol"]).reset_index(drop=True)

    out["dbg_position_limits_fallback"] = 0
    out = apply_position_filters(out, side_col="side", min_price=LOW_PRICE_MIN, min_dollar_volume=LOW_DV_MIN)
    out = cap_final_weight(
        out,
        weight_col="weight",
        cap_abs_weight=WEIGHT_CAP,
        max_adv_participation=MAX_ADV_PARTICIPATION,
        portfolio_notional=PORTFOLIO_NOTIONAL,
    )
    out = renormalize_after_caps(out, weight_col="weight", gross_target=GROSS_TARGET)

    out["trade_strength"] = out["raw_strength"].fillna(0.0)
    out["mandatory_exit"] = 0
    out["risk_trim"] = 0
    out["keep_add"] = ((out["side"] != 0) & (out["weight"].abs() > 0)).astype(int)
    out["new_entry"] = ((out["side"] != 0) & (out["weight"].abs() > 0)).astype(int)
    out = cap_daily_turnover(out, weight_col="weight", max_daily_turnover=MAX_DAILY_TURNOVER)
    if "turnover" not in out.columns:
        out = out.sort_values(["symbol", "date"]).reset_index(drop=True)
        out["prev_weight"] = out.groupby("symbol", sort=False)["weight"].shift(1).fillna(0.0)
        out["turnover"] = (out["weight"] - out["prev_weight"]).abs()

    out["dbg_cost_model_fallback"] = 0
    out = attach_execution_costs(
        out,
        weight_col="weight",
        fee_bps=FEE_BPS,
        slippage_bps=SLIPPAGE_BPS,
        borrow_bps_daily=BORROW_BPS_DAILY,
        spread_bps=SPREAD_BPS,
        impact_bps=IMPACT_BPS,
        low_price_penalty_bps=LOW_PRICE_PENALTY_BPS,
        htb_borrow_bps_daily=HTB_BORROW_BPS_DAILY,
        max_participation=MAX_ADV_PARTICIPATION,
        portfolio_notional=PORTFOLIO_NOTIONAL,
    )
    return out


def evaluate_portfolio(port_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    df = port_df.copy()
    df["gross_pnl"] = _safe_numeric(df["weight"]) * _safe_numeric(df[TARGET_COL])
    if "exec_cost_ret" in df.columns:
        df["cost"] = _safe_numeric(df["exec_cost_ret"]).fillna(0.0)
    else:
        df["cost"] = _safe_numeric(df["turnover"]) * (COST_BPS / 10000.0)
        df.loc[df["weight"] < 0, "cost"] += _safe_numeric(df.loc[df["weight"] < 0, "weight"]).abs() * (BORROW_BPS_DAILY / 10000.0)
    daily = df.groupby("date", sort=False).agg(
        gross_ret=("gross_pnl", "sum"),
        cost_ret=("cost", "sum"),
        turnover=("turnover", "sum"),
        gross=("weight", lambda s: float(np.sum(np.abs(pd.to_numeric(s, errors="coerce"))))),
        longs=("weight", lambda s: int((pd.to_numeric(s, errors="coerce") > 0).sum())),
        shorts=("weight", lambda s: int((pd.to_numeric(s, errors="coerce") < 0).sum())),
        top_pct=("top_pct", "first"),
        long_budget=("long_budget", "first"),
        short_budget=("short_budget", "first"),
        regime=("alloc_regime", "first"),
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
        "avg_top_pct": float(daily["top_pct"].mean()) if len(daily) else float("nan"),
        "position_limits_fallback_days": float(daily["date"].map(df.groupby("date", sort=False)["dbg_position_limits_fallback"].max()).fillna(0.0).sum()) if len(daily) else float("nan"),
        "cost_model_fallback_days": float(daily["date"].map(df.groupby("date", sort=False)["dbg_cost_model_fallback"].max()).fillna(0.0).sum()) if len(daily) else float("nan"),
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
    summary["selected_alpha_nonzero_mean"] = float(train_df[selected].fillna(0.0).abs().sum(axis=1).gt(0.0).mean())
    summary["ridge_chosen_min_active"] = float(weights_df["chosen_min_active"].iloc[0]) if "chosen_min_active" in weights_df.columns and len(weights_df) else float("nan")
    summary["ridge_usable_rows"] = float(weights_df["usable_rows"].iloc[0]) if "usable_rows" in weights_df.columns and len(weights_df) else float("nan")
    summary["fold_alpha_input_count"] = float(len(alpha_cols))
    return selected_df, weights_df, daily, summary


def main() -> int:
    _enable_line_buffering()
    print(f"[CFG] alpha_lib_file={ALPHA_LIB_FILE}")
    print(f"[CFG] alpha_shortlist_csv={ALPHA_SHORTLIST_CSV}")
    print(f"[CFG] alpha_shortlist_required={int(ALPHA_SHORTLIST_REQUIRED)}")
    print(f"[CFG] out_dir={OUT_DIR}")
    print(f"[CFG] target_col={TARGET_COL}")
    print(f"[CFG] train_days={TRAIN_DAYS} test_days={TEST_DAYS} step_days={STEP_DAYS} purge_days={PURGE_DAYS} embargo_days={EMBARGO_DAYS}")
    print(f"[CFG] ridge_l2={RIDGE_L2} max_alphas={MAX_ALPHAS} min_alpha_days={MIN_ALPHA_DAYS} min_alpha_abs_ic={MIN_ALPHA_ABS_IC} corr_prune={CORR_PRUNE}")
    print(f"[CFG] stability min_blocks={MIN_STABILITY_BLOCKS} min_sign_consistency={MIN_SIGN_CONSISTENCY} min_block_ic_abs={MIN_BLOCK_IC_ABS}")
    print(f"[CFG] enter_pct={ENTER_PCT} exit_pct={EXIT_PCT} weight_cap={WEIGHT_CAP} max_daily_turnover={MAX_DAILY_TURNOVER}")
    print(f"[CFG] position_limits_module={_HAS_POSITION_LIMITS} cost_model_module={_HAS_COST_MODEL}")
    print(f"[CFG] low_price_min={LOW_PRICE_MIN} low_dv_min={LOW_DV_MIN} max_adv_participation={MAX_ADV_PARTICIPATION}")
    print(f"[CFG] top_pct strong/neutral/weak={TOP_PCT_STRONG}/{TOP_PCT_NEUTRAL}/{TOP_PCT_WEAK} budgets={BUDGET_MODE} regime={REGIME_MODE}")

    df, alpha_cols, shortlist_counters, shortlist_df = load_alpha_library()
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
    summaries: List[Dict[str, float]] = []
    all_daily: List[pd.DataFrame] = []
    top_weights_preview: List[Dict[str, object]] = []

    shortlist_meta_path = OUT_DIR / "wf_alpha_shortlist_debug.json"
    shortlist_meta_path.write_text(json.dumps(shortlist_counters, ensure_ascii=False, indent=2), encoding="utf-8")

    for sp in splits:
        print(f"[WF][FOLD {sp.fold_id}] start")
        selected_df, weights_df, daily, summary = run_fold(df, sp, alpha_cols)
        selected_path = OUT_DIR / f"wf_selected_alphas__fold{sp.fold_id}.csv"
        weights_path = OUT_DIR / f"wf_alpha_weights__fold{sp.fold_id}.csv"
        daily_path = OUT_DIR / f"wf_daily__fold{sp.fold_id}.csv"
        selected_df.to_csv(selected_path, index=False)
        weights_df.to_csv(weights_path, index=False)
        daily.to_csv(daily_path, index=False)
        print(
            f"[WF][FOLD {sp.fold_id}][SUMMARY] sharpe={summary['sharpe']:.4f} mean_daily={summary['mean_daily']:.6f} "
            f"cum_ret={summary['cum_ret']:.4f} maxdd={summary['max_drawdown']:.4f} selected={int(summary['selected_alpha_count'])} "
            f"avg_turn={summary['avg_turnover']:.4f} alpha_input={int(summary['fold_alpha_input_count'])}"
        )
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
    equity = (1.0 + overall_daily["net_ret"].fillna(0.0)).cumprod()
    overall_summary = {
        "days": float(len(overall_daily)),
        "mean_daily": float(overall_daily["net_ret"].mean()),
        "std_daily": float(overall_daily["net_ret"].std(ddof=0)),
        "sharpe": float((overall_daily["net_ret"].mean() / (overall_daily["net_ret"].std(ddof=0) + EPS)) * np.sqrt(252.0)),
        "hit_rate": float((overall_daily["net_ret"] > 0).mean()),
        "cum_ret": float(equity.iloc[-1] - 1.0),
        "max_drawdown": float((equity / equity.cummax() - 1.0).min()),
        "avg_turnover": float(overall_daily["turnover"].mean()),
        "folds": int(len(summaries)),
    }
    summary_df = pd.DataFrame(summaries)
    summary_path = OUT_DIR / "wf_fold_summaries.csv"
    summary_df.to_csv(summary_path, index=False)

    meta = {
        "alpha_lib_file": str(ALPHA_LIB_FILE),
        "alpha_shortlist_csv": str(ALPHA_SHORTLIST_CSV),
        "alpha_shortlist_required": int(ALPHA_SHORTLIST_REQUIRED),
        "alpha_shortlist_debug": shortlist_counters,
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
        "corr_prune": CORR_PRUNE,
        "enter_pct": ENTER_PCT,
        "exit_pct": EXIT_PCT,
        "gross_target": GROSS_TARGET,
        "weight_cap": WEIGHT_CAP,
        "max_daily_turnover": MAX_DAILY_TURNOVER,
        "cost_bps": COST_BPS,
        "borrow_bps_daily": BORROW_BPS_DAILY,
        "side": SIDE,
        "budget_mode": BUDGET_MODE,
        "regime_mode": REGIME_MODE,
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
    print(f"[ARTIFACT] {shortlist_meta_path}")
    print("[FINAL] multi-alpha walk-forward complete")
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 1
    _press_enter_exit(rc)