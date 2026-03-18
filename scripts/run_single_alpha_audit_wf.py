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
ALPHA_SHORTLIST_CSV = Path(os.getenv("ALPHA_SHORTLIST_CSV", "data/alpha_library_v2/diagnostics/alpha_candidate_shortlist__clean.csv"))
ALPHA_SHORTLIST_REQUIRED = str(os.getenv("ALPHA_SHORTLIST_REQUIRED", "1")).strip().lower() not in {"0", "false", "no", "off"}
TARGET_COL = str(os.getenv("TARGET_COL", "target_fwd_ret_1d")).strip()
OUT_DIR = Path(os.getenv("SINGLE_ALPHA_AUDIT_OUT_DIR", "artifacts/single_alpha_audit_wf"))
PAUSE_ON_EXIT = str(os.getenv("PAUSE_ON_EXIT", "auto")).strip().lower()

TRAIN_DAYS = int(os.getenv("WF_TRAIN_DAYS", "252"))
TEST_DAYS = int(os.getenv("WF_TEST_DAYS", "63"))
STEP_DAYS = int(os.getenv("WF_STEP_DAYS", "63"))
PURGE_DAYS = int(os.getenv("WF_PURGE_DAYS", "5"))
EMBARGO_DAYS = int(os.getenv("WF_EMBARGO_DAYS", "5"))
MIN_TRAIN_ROWS = int(os.getenv("MIN_TRAIN_ROWS", "3000"))
MIN_TEST_ROWS = int(os.getenv("MIN_TEST_ROWS", "300"))
MIN_DAILY_IC_CS = int(os.getenv("MIN_DAILY_IC_CS", "20"))

ENTER_PCT = float(os.getenv("ENTER_PCT", "0.10"))
EXIT_PCT = float(os.getenv("EXIT_PCT", "0.22"))
WEIGHT_CAP = float(os.getenv("WEIGHT_CAP", "0.05"))
GROSS_TARGET = float(os.getenv("GROSS_TARGET", "0.85"))
MAX_DAILY_TURNOVER = float(os.getenv("MAX_DAILY_TURNOVER", "0.20"))
COST_BPS = float(os.getenv("COST_BPS", "8.0"))

ALPHA_LIMIT = int(os.getenv("ALPHA_LIMIT", "0"))
ALPHA_NAME_FILTER = str(os.getenv("ALPHA_NAME_FILTER", "")).strip()
TOPK_PRINT = int(os.getenv("TOPK_PRINT", "30"))

ALIVE_MIN_LAST_FOLD_SHARPE = float(os.getenv("ALIVE_MIN_LAST_FOLD_SHARPE", "0.0"))
ALIVE_MIN_LAST2_MEAN_SHARPE = float(os.getenv("ALIVE_MIN_LAST2_MEAN_SHARPE", "0.0"))
ALIVE_MIN_MEAN_IC = float(os.getenv("ALIVE_MIN_MEAN_IC", "0.0"))
ALIVE_MIN_SIGN_STABILITY = float(os.getenv("ALIVE_MIN_SIGN_STABILITY", "0.50"))
ALIVE_MIN_POSITIVE_SHARPE_FRACTION = float(os.getenv("ALIVE_MIN_POSITIVE_SHARPE_FRACTION", "0.50"))


@dataclass(frozen=True)
class WFSplit:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass(frozen=True)
class AuditConfig:
    enter_pct: float
    exit_pct: float
    weight_cap: float
    gross_target: float
    max_daily_turnover: float
    cost_bps: float


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


def _must_exist(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


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


def _daily_ic_series(frame: pd.DataFrame, factor_col: str, target_col: str, min_cs: int) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for dt, g in frame.groupby("date", sort=False):
        x = g[[factor_col, target_col]].dropna()
        if len(x) < min_cs:
            continue
        if x[factor_col].nunique(dropna=True) <= 1:
            continue
        if x[target_col].nunique(dropna=True) <= 1:
            continue
        ic = x[factor_col].corr(x[target_col], method="spearman")
        if pd.notna(ic):
            rows.append({"date": pd.Timestamp(dt).normalize(), "daily_ic": float(ic), "cross_section_n": int(len(x))})
    return pd.DataFrame(rows)


def _family_from_shortlist(shortlist_df: pd.DataFrame) -> Dict[str, str]:
    if shortlist_df.empty or "family" not in shortlist_df.columns:
        return {}
    work = shortlist_df[["alpha", "family"]].copy()
    work["alpha"] = work["alpha"].astype(str)
    work["family"] = work["family"].astype(str)
    return dict(zip(work["alpha"], work["family"]))


def _load_shortlist_required(alpha_cols_available: Sequence[str]) -> Tuple[List[str], Dict[str, object], pd.DataFrame, Dict[str, str]]:
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
    found = [c for c in shortlist_df["alpha"].tolist() if c in set(alpha_cols_available)]
    missing_in_parquet = [c for c in shortlist_df["alpha"].tolist() if c not in set(alpha_cols_available)]
    counters["alpha_cols_before_shortlist"] = int(len(alpha_cols_available))
    counters["shortlist_found_in_parquet"] = int(len(found))
    counters["shortlist_missing_in_parquet"] = int(len(missing_in_parquet))
    counters["shortlist_missing_in_parquet_list"] = missing_in_parquet
    family_map = _family_from_shortlist(shortlist_df)
    return found, counters, shortlist_df, family_map


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
    shortlist_cols, shortlist_counters, shortlist_df, family_map = _load_shortlist_required(alpha_before)
    if ALPHA_SHORTLIST_REQUIRED and not shortlist_cols:
        raise RuntimeError("Shortlist required but no shortlist alphas found in parquet")
    alpha_cols = shortlist_cols if shortlist_cols else alpha_before
    if ALPHA_NAME_FILTER:
        alpha_cols = [c for c in alpha_cols if ALPHA_NAME_FILTER.lower() in c.lower()]
    if ALPHA_LIMIT > 0:
        alpha_cols = alpha_cols[:ALPHA_LIMIT]
    if not alpha_cols:
        raise RuntimeError("No alpha columns remain after shortlist/name filtering")
    keep_cols = [c for c in df.columns if not c.startswith("alpha_")] + alpha_cols
    df = df[keep_cols].copy()
    shortlist_counters["alpha_cols_after_filters"] = int(len(alpha_cols))
    shortlist_counters["alpha_name_filter"] = ALPHA_NAME_FILTER
    shortlist_counters["alpha_limit"] = ALPHA_LIMIT
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


def _split_frame(df: pd.DataFrame, split: WFSplit) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df.loc[(df["date"] >= split.train_start) & (df["date"] <= split.train_end)].copy()
    test_df = df.loc[(df["date"] >= split.test_start) & (df["date"] <= split.test_end)].copy()
    return train_df, test_df


def _orient_sign(train_df: pd.DataFrame, alpha: str) -> Tuple[float, float, int, float]:
    train_ic_df = _daily_ic_series(train_df, alpha, TARGET_COL, min_cs=MIN_DAILY_IC_CS)
    if train_ic_df.empty:
        return 1.0, float("nan"), 0, float("nan")
    mean_ic = float(train_ic_df["daily_ic"].mean())
    sign = 1.0 if mean_ic >= 0.0 else -1.0
    pos_rate = float((train_ic_df["daily_ic"] > 0.0).mean())
    return sign, mean_ic, int(len(train_ic_df)), pos_rate


def _prepare_single_alpha_scores(test_df: pd.DataFrame, alpha: str, sign: float) -> pd.DataFrame:
    out = test_df[["date", "symbol", TARGET_COL, alpha]].copy()
    out[alpha] = _safe_numeric(out[alpha])
    out["raw_alpha"] = out[alpha]
    out["score"] = out.groupby("date", sort=False)[alpha].transform(_robust_zscore_series) * float(sign)
    out = out.drop(columns=[alpha])
    return out


def _build_portfolio(scored_df: pd.DataFrame, cfg: AuditConfig) -> pd.DataFrame:
    out = scored_df.copy()
    out = apply_holding_inertia(out, enter_pct=cfg.enter_pct, exit_pct=cfg.exit_pct)
    if "side" not in out.columns:
        raise RuntimeError("apply_holding_inertia did not return side column")
    abs_side_sum = out.groupby("date", sort=False)["side"].transform(lambda s: pd.to_numeric(s, errors="coerce").abs().sum())
    out["raw_weight"] = np.where(abs_side_sum > 0, pd.to_numeric(out["side"], errors="coerce") / abs_side_sum, 0.0)
    out["weight"] = pd.to_numeric(out["raw_weight"], errors="coerce").clip(-cfg.weight_cap, cfg.weight_cap)
    gross = out.groupby("date", sort=False)["weight"].transform(lambda s: pd.to_numeric(s, errors="coerce").abs().sum())
    out["weight"] = np.where(gross > 0, pd.to_numeric(out["weight"], errors="coerce") * (cfg.gross_target / gross), 0.0)
    out = cap_daily_turnover(out, weight_col="weight", max_daily_turnover=cfg.max_daily_turnover)
    out = out.sort_values(["symbol", "date"]).reset_index(drop=True)
    out["prev_weight"] = out.groupby("symbol", sort=False)["weight"].shift(1).fillna(0.0)
    out["turnover"] = (pd.to_numeric(out["weight"], errors="coerce") - pd.to_numeric(out["prev_weight"], errors="coerce")).abs()
    out["gross_ret"] = pd.to_numeric(out["weight"], errors="coerce") * pd.to_numeric(out[TARGET_COL], errors="coerce")
    out["cost_ret"] = pd.to_numeric(out["turnover"], errors="coerce") * (cfg.cost_bps / 10000.0)
    out["net_ret"] = pd.to_numeric(out["gross_ret"], errors="coerce") - pd.to_numeric(out["cost_ret"], errors="coerce")
    return out


def _evaluate_daily_portfolio(port_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    daily = port_df.groupby("date", sort=False, as_index=False).agg(
        gross_ret=("gross_ret", "sum"),
        cost_ret=("cost_ret", "sum"),
        net_ret=("net_ret", "sum"),
        turnover=("turnover", "sum"),
        gross_exposure=("weight", lambda s: float(pd.to_numeric(s, errors="coerce").abs().sum())),
        names_active=("side", lambda s: int((pd.to_numeric(s, errors="coerce").abs() > 0).sum())),
    )
    daily = daily.sort_values("date").reset_index(drop=True)
    equity = (1.0 + daily["net_ret"].fillna(0.0)).cumprod()
    daily["equity"] = equity
    daily["rolling_peak"] = equity.cummax()
    daily["drawdown"] = np.where(daily["rolling_peak"] > 0, equity / daily["rolling_peak"] - 1.0, 0.0)
    mean_daily = float(daily["net_ret"].mean()) if len(daily) else float("nan")
    std_daily = float(daily["net_ret"].std(ddof=0)) if len(daily) else float("nan")
    sharpe = float((mean_daily / (std_daily + EPS)) * np.sqrt(252.0)) if len(daily) else float("nan")
    summary = {
        "days": float(len(daily)),
        "mean_daily": mean_daily,
        "std_daily": std_daily,
        "sharpe": sharpe,
        "cum_ret": float(equity.iloc[-1] - 1.0) if len(equity) else float("nan"),
        "max_drawdown": float(daily["drawdown"].min()) if len(daily) else float("nan"),
        "avg_turnover": float(daily["turnover"].mean()) if len(daily) else float("nan"),
        "avg_names_active": float(daily["names_active"].mean()) if len(daily) else float("nan"),
        "avg_gross_exposure": float(daily["gross_exposure"].mean()) if len(daily) else float("nan"),
    }
    return daily, summary


def _safe_mean(values: Sequence[float]) -> float:
    vals = [float(x) for x in values if pd.notna(x)]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def _safe_fraction_positive(values: Sequence[float]) -> float:
    vals = [float(x) for x in values if pd.notna(x)]
    if not vals:
        return float("nan")
    return float(sum(1 for x in vals if x > 0.0) / len(vals))


def _last_n_mean(values: Sequence[float], n: int) -> float:
    vals = [float(x) for x in values if pd.notna(x)]
    if not vals:
        return float("nan")
    return float(np.mean(vals[-n:]))


def _alive_flag(last_fold_sharpe: float, last2_fold_mean_sharpe: float, mean_ic: float, sign_stability: float, positive_sharpe_fraction: float) -> Tuple[int, str]:
    reasons: List[str] = []
    alive = True
    if pd.isna(last_fold_sharpe) or float(last_fold_sharpe) < ALIVE_MIN_LAST_FOLD_SHARPE:
        alive = False
        reasons.append(f"last_fold_sharpe<{ALIVE_MIN_LAST_FOLD_SHARPE}")
    if pd.isna(last2_fold_mean_sharpe) or float(last2_fold_mean_sharpe) < ALIVE_MIN_LAST2_MEAN_SHARPE:
        alive = False
        reasons.append(f"last2_fold_mean_sharpe<{ALIVE_MIN_LAST2_MEAN_SHARPE}")
    if pd.isna(mean_ic) or float(mean_ic) < ALIVE_MIN_MEAN_IC:
        alive = False
        reasons.append(f"mean_ic<{ALIVE_MIN_MEAN_IC}")
    if pd.isna(sign_stability) or float(sign_stability) < ALIVE_MIN_SIGN_STABILITY:
        alive = False
        reasons.append(f"sign_stability<{ALIVE_MIN_SIGN_STABILITY}")
    if pd.isna(positive_sharpe_fraction) or float(positive_sharpe_fraction) < ALIVE_MIN_POSITIVE_SHARPE_FRACTION:
        alive = False
        reasons.append(f"positive_sharpe_fraction<{ALIVE_MIN_POSITIVE_SHARPE_FRACTION}")
    if alive:
        return 1, "ok"
    return 0, ";".join(reasons)


def _run_single_alpha_fold(df: pd.DataFrame, split: WFSplit, alpha: str, cfg: AuditConfig) -> Tuple[Dict[str, object], pd.DataFrame, pd.DataFrame]:
    train_df, test_df = _split_frame(df, split)
    if len(train_df) < MIN_TRAIN_ROWS:
        raise RuntimeError(f"Fold {split.fold_id} alpha {alpha}: too few train rows: {len(train_df)}")
    if len(test_df) < MIN_TEST_ROWS:
        raise RuntimeError(f"Fold {split.fold_id} alpha {alpha}: too few test rows: {len(test_df)}")
    sign, train_mean_ic, train_ic_days, train_ic_pos_rate = _orient_sign(train_df, alpha)
    scored_df = _prepare_single_alpha_scores(test_df, alpha, sign=sign)
    port_df = _build_portfolio(scored_df, cfg)
    daily_df, port_summary = _evaluate_daily_portfolio(port_df)
    ic_df = _daily_ic_series(scored_df, "score", TARGET_COL, min_cs=MIN_DAILY_IC_CS)
    if not ic_df.empty:
        ic_df["alpha"] = alpha
        ic_df["fold_id"] = int(split.fold_id)
        ic_df["train_sign"] = float(sign)
        ic_df["train_mean_ic"] = float(train_mean_ic) if pd.notna(train_mean_ic) else float("nan")
        ic_df["train_ic_days"] = int(train_ic_days)
    fold_row: Dict[str, object] = {
        "alpha": alpha,
        "fold_id": int(split.fold_id),
        "train_start": split.train_start,
        "train_end": split.train_end,
        "test_start": split.test_start,
        "test_end": split.test_end,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_mean_ic_raw": float(train_mean_ic) if pd.notna(train_mean_ic) else float("nan"),
        "train_ic_days": int(train_ic_days),
        "train_ic_positive_rate": float(train_ic_pos_rate) if pd.notna(train_ic_pos_rate) else float("nan"),
        "train_sign": float(sign),
        "oos_mean_ic": float(ic_df["daily_ic"].mean()) if len(ic_df) else float("nan"),
        "oos_mean_abs_ic": float(ic_df["daily_ic"].abs().mean()) if len(ic_df) else float("nan"),
        "oos_ic_days": int(len(ic_df)),
        "oos_sign_stability_day": float((ic_df["daily_ic"] > 0.0).mean()) if len(ic_df) else float("nan"),
        **port_summary,
    }
    daily_df = daily_df.copy()
    daily_df["alpha"] = alpha
    daily_df["fold_id"] = int(split.fold_id)
    daily_df["train_sign"] = float(sign)
    return fold_row, daily_df, ic_df


def _summarize_alpha(alpha: str, family: str, fold_df: pd.DataFrame, daily_ic_df: pd.DataFrame) -> Dict[str, object]:
    fold_df = fold_df.sort_values("fold_id").reset_index(drop=True)
    sharpes = [float(x) for x in fold_df["sharpe"].tolist() if pd.notna(x)]
    mean_ics = [float(x) for x in fold_df["oos_mean_ic"].tolist() if pd.notna(x)]
    last_fold_sharpe = float(fold_df["sharpe"].iloc[-1]) if len(fold_df) else float("nan")
    last2_fold_mean_sharpe = _last_n_mean(fold_df["sharpe"].tolist(), 2)
    mean_ic = float(daily_ic_df["daily_ic"].mean()) if len(daily_ic_df) else _safe_mean(mean_ics)
    mean_abs_ic = float(daily_ic_df["daily_ic"].abs().mean()) if len(daily_ic_df) else float("nan")
    sign_stability = float((daily_ic_df["daily_ic"] > 0.0).mean()) if len(daily_ic_df) else float("nan")
    fold_positive_ic_fraction = _safe_fraction_positive(fold_df["oos_mean_ic"].tolist())
    positive_sharpe_fraction = _safe_fraction_positive(fold_df["sharpe"].tolist())
    alive, alive_reason = _alive_flag(
        last_fold_sharpe=last_fold_sharpe,
        last2_fold_mean_sharpe=last2_fold_mean_sharpe,
        mean_ic=mean_ic,
        sign_stability=sign_stability,
        positive_sharpe_fraction=positive_sharpe_fraction,
    )
    return {
        "alpha": alpha,
        "family": family,
        "folds": int(len(fold_df)),
        "oos_sharpe_mean": _safe_mean(sharpes),
        "oos_sharpe_median": float(np.median(sharpes)) if sharpes else float("nan"),
        "oos_sharpe_std": float(np.std(sharpes, ddof=0)) if sharpes else float("nan"),
        "oos_sharpe_min": float(np.min(sharpes)) if sharpes else float("nan"),
        "oos_sharpe_max": float(np.max(sharpes)) if sharpes else float("nan"),
        "last_fold_sharpe": last_fold_sharpe,
        "last_2_fold_mean_sharpe": last2_fold_mean_sharpe,
        "daily_ic_mean": mean_ic,
        "daily_ic_mean_abs": mean_abs_ic,
        "fold_ic_mean": _safe_mean(mean_ics),
        "sign_stability": sign_stability,
        "fold_positive_ic_fraction": fold_positive_ic_fraction,
        "positive_sharpe_fraction": positive_sharpe_fraction,
        "max_drawdown": float(fold_df["max_drawdown"].min()) if len(fold_df) else float("nan"),
        "turnover_mean": float(fold_df["avg_turnover"].mean()) if len(fold_df) else float("nan"),
        "turnover_last_fold": float(fold_df["avg_turnover"].iloc[-1]) if len(fold_df) else float("nan"),
        "cum_ret_total": float(np.prod(1.0 + pd.to_numeric(fold_df["cum_ret"], errors="coerce").fillna(0.0)) - 1.0) if len(fold_df) else float("nan"),
        "standalone_alive": int(alive),
        "alive_reason": alive_reason,
    }


def main() -> int:
    _enable_line_buffering()
    cfg = AuditConfig(
        enter_pct=ENTER_PCT,
        exit_pct=EXIT_PCT,
        weight_cap=WEIGHT_CAP,
        gross_target=GROSS_TARGET,
        max_daily_turnover=MAX_DAILY_TURNOVER,
        cost_bps=COST_BPS,
    )

    print(f"[CFG] alpha_lib_file={ALPHA_LIB_FILE}")
    print(f"[CFG] alpha_shortlist_csv={ALPHA_SHORTLIST_CSV}")
    print(f"[CFG] alpha_shortlist_required={int(ALPHA_SHORTLIST_REQUIRED)}")
    print(f"[CFG] target_col={TARGET_COL}")
    print(f"[CFG] wf train_days={TRAIN_DAYS} test_days={TEST_DAYS} step_days={STEP_DAYS} purge_days={PURGE_DAYS} embargo_days={EMBARGO_DAYS}")
    print(f"[CFG] shell gross_target={GROSS_TARGET} weight_cap={WEIGHT_CAP} max_daily_turnover={MAX_DAILY_TURNOVER} enter_pct={ENTER_PCT} exit_pct={EXIT_PCT} cost_bps={COST_BPS}")
    print(f"[CFG] alpha_limit={ALPHA_LIMIT} alpha_name_filter={ALPHA_NAME_FILTER!r}")
    print(f"[CFG] alive min_last_fold_sharpe={ALIVE_MIN_LAST_FOLD_SHARPE} min_last2_mean_sharpe={ALIVE_MIN_LAST2_MEAN_SHARPE} min_mean_ic={ALIVE_MIN_MEAN_IC} min_sign_stability={ALIVE_MIN_SIGN_STABILITY} min_positive_sharpe_fraction={ALIVE_MIN_POSITIVE_SHARPE_FRACTION}")

    df, alpha_cols, shortlist_counters, shortlist_df, family_map = load_alpha_library()
    print(f"[DATA] rows={len(df)} dates={df['date'].nunique()} symbols={df['symbol'].nunique()} alpha_cols={len(alpha_cols)}")
    print("[ALPHA_SHORTLIST]")
    print(json.dumps(shortlist_counters, ensure_ascii=False, indent=2))
    if len(shortlist_df):
        preview_cols = [c for c in ["shortlist_rank", "alpha", "family", "wave", "transform", "interaction", "regime", "selector_score"] if c in shortlist_df.columns]
        if not preview_cols:
            preview_cols = ["alpha"]
        print("[ALPHA_SHORTLIST][HEAD]")
        print(shortlist_df[preview_cols].head(20).to_string(index=False))

    splits = build_walkforward_splits(df["date"])
    print(f"[WF] folds={len(splits)}")
    for sp in splits:
        print(f"[WF][FOLD {sp.fold_id}] train={sp.train_start.date()}..{sp.train_end.date()} test={sp.test_start.date()}..{sp.test_end.date()}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "per_fold_daily").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "per_fold_ic").mkdir(parents=True, exist_ok=True)

    shortlist_debug_path = OUT_DIR / "audit_alpha_shortlist_debug.json"
    shortlist_debug_path.write_text(json.dumps(shortlist_counters, ensure_ascii=False, indent=2), encoding="utf-8")

    all_fold_rows: List[Dict[str, object]] = []
    all_daily_ic_rows: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []

    for idx, alpha in enumerate(alpha_cols, start=1):
        family = family_map.get(alpha, "unknown")
        print(f"[AUDIT] {idx}/{len(alpha_cols)} alpha={alpha} family={family}")
        alpha_fold_rows: List[Dict[str, object]] = []
        alpha_daily_frames: List[pd.DataFrame] = []
        alpha_ic_frames: List[pd.DataFrame] = []
        try:
            for split in splits:
                fold_row, daily_df, ic_df = _run_single_alpha_fold(df, split, alpha, cfg)
                fold_row["family"] = family
                alpha_fold_rows.append(fold_row)
                alpha_daily_frames.append(daily_df)
                alpha_ic_frames.append(ic_df)
                print(
                    f"[AUDIT][{alpha}][FOLD {split.fold_id}] "
                    f"sharpe={fold_row['sharpe']:.4f} "
                    f"mean_ic={fold_row['oos_mean_ic']:.5f} "
                    f"maxdd={fold_row['max_drawdown']:.4f} "
                    f"turnover={fold_row['avg_turnover']:.4f} "
                    f"train_sign={int(fold_row['train_sign'])}"
                )
            alpha_fold_df = pd.DataFrame(alpha_fold_rows).sort_values("fold_id").reset_index(drop=True)
            alpha_ic_df = pd.concat(alpha_ic_frames, ignore_index=True) if alpha_ic_frames else pd.DataFrame(columns=["date", "daily_ic"])
            alpha_daily_df = pd.concat(alpha_daily_frames, ignore_index=True) if alpha_daily_frames else pd.DataFrame(columns=["date", "net_ret"])
            alpha_fold_df.to_csv(OUT_DIR / "per_fold_daily" / f"{alpha}__fold_summary.csv", index=False)
            alpha_daily_df.to_csv(OUT_DIR / "per_fold_daily" / f"{alpha}__daily.csv", index=False)
            alpha_ic_df.to_csv(OUT_DIR / "per_fold_ic" / f"{alpha}__daily_ic.csv", index=False)
            all_fold_rows.extend(alpha_fold_rows)
            if len(alpha_ic_df):
                all_daily_ic_rows.append(alpha_ic_df)
            summary_row = _summarize_alpha(alpha, family, alpha_fold_df, alpha_ic_df)
            summary_rows.append(summary_row)
        except Exception as exc:
            err = {"alpha": alpha, "family": family, "error": f"{type(exc).__name__}: {exc}"}
            failures.append(err)
            print(f"[AUDIT][FAIL] alpha={alpha} error={err['error']}")

    fold_df_all = pd.DataFrame(all_fold_rows)
    summary_df = pd.DataFrame(summary_rows)
    if len(summary_df):
        summary_df = summary_df.sort_values(
            ["standalone_alive", "last_fold_sharpe", "last_2_fold_mean_sharpe", "daily_ic_mean", "sign_stability", "alpha"],
            ascending=[False, False, False, False, False, True],
        ).reset_index(drop=True)
    daily_ic_all_df = pd.concat(all_daily_ic_rows, ignore_index=True) if all_daily_ic_rows else pd.DataFrame(columns=["alpha", "fold_id", "date", "daily_ic"])
    failures_df = pd.DataFrame(failures)

    fold_df_all.to_csv(OUT_DIR / "single_alpha_audit__fold_metrics.csv", index=False)
    summary_df.to_csv(OUT_DIR / "single_alpha_audit__summary.csv", index=False)
    daily_ic_all_df.to_csv(OUT_DIR / "single_alpha_audit__daily_ic.csv", index=False)
    failures_df.to_csv(OUT_DIR / "single_alpha_audit__failures.csv", index=False)

    if len(summary_df):
        alive_df = summary_df.loc[summary_df["standalone_alive"] == 1].copy()
        dead_df = summary_df.loc[summary_df["standalone_alive"] != 1].copy()
        alive_df.to_csv(OUT_DIR / "single_alpha_audit__alive.csv", index=False)
        dead_df.to_csv(OUT_DIR / "single_alpha_audit__dead.csv", index=False)
        print("[SUMMARY][TOP]")
        print(summary_df.head(TOPK_PRINT).to_string(index=False))
        print(f"[SUMMARY] total={len(summary_df)} alive={len(alive_df)} dead={len(dead_df)} failures={len(failures_df)}")
    else:
        print(f"[SUMMARY] no completed alpha audits; failures={len(failures_df)}")

    meta = {
        "alpha_lib_file": str(ALPHA_LIB_FILE),
        "alpha_shortlist_csv": str(ALPHA_SHORTLIST_CSV),
        "alpha_shortlist_required": int(ALPHA_SHORTLIST_REQUIRED),
        "target_col": TARGET_COL,
        "wf": {
            "train_days": TRAIN_DAYS,
            "test_days": TEST_DAYS,
            "step_days": STEP_DAYS,
            "purge_days": PURGE_DAYS,
            "embargo_days": EMBARGO_DAYS,
            "folds": len(splits),
        },
        "shell": cfg.__dict__,
        "alpha_limit": ALPHA_LIMIT,
        "alpha_name_filter": ALPHA_NAME_FILTER,
        "alive_thresholds": {
            "min_last_fold_sharpe": ALIVE_MIN_LAST_FOLD_SHARPE,
            "min_last2_mean_sharpe": ALIVE_MIN_LAST2_MEAN_SHARPE,
            "min_mean_ic": ALIVE_MIN_MEAN_IC,
            "min_sign_stability": ALIVE_MIN_SIGN_STABILITY,
            "min_positive_sharpe_fraction": ALIVE_MIN_POSITIVE_SHARPE_FRACTION,
        },
        "holding_inertia_imported": int(HAS_HOLDING_INERTIA),
        "turnover_control_imported": int(HAS_TURNOVER_CONTROL),
        "shortlist_debug": shortlist_counters,
        "summary_rows": int(len(summary_df)),
        "failure_rows": int(len(failures_df)),
    }
    meta_path = OUT_DIR / "single_alpha_audit__meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ARTIFACT] {OUT_DIR / 'single_alpha_audit__summary.csv'}")
    print(f"[ARTIFACT] {OUT_DIR / 'single_alpha_audit__fold_metrics.csv'}")
    print(f"[ARTIFACT] {OUT_DIR / 'single_alpha_audit__daily_ic.csv'}")
    print(f"[ARTIFACT] {OUT_DIR / 'single_alpha_audit__failures.csv'}")
    print(f"[ARTIFACT] {meta_path}")
    print(f"[ARTIFACT] {shortlist_debug_path}")
    print("[FINAL] single-alpha walk-forward audit complete")
    return 0 if len(summary_df) else 1


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 1
    _safe_exit(rc)
