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

    def apply_holding_inertia(df: pd.DataFrame, enter_pct: float = 0.10, exit_pct: float = 0.22) -> pd.DataFrame:
        out = df.copy().sort_values(["date", "symbol"]).reset_index(drop=True)
        out["rank_pct"] = out.groupby("date", sort=False)["score"].rank(method="average", pct=True)
        out["side"] = 0.0
        prev_side_by_symbol: Dict[str, float] = {}
        for dt in sorted(out["date"].dropna().unique()):
            idx = out.index[out["date"] == dt]
            day = out.loc[idx, ["symbol", "rank_pct"]].copy()
            next_side = pd.Series(0.0, index=day.index, dtype="float64")
            for row_idx, row in day.iterrows():
                sym = str(row["symbol"])
                rp = float(row["rank_pct"])
                prev_side = float(prev_side_by_symbol.get(sym, 0.0))
                if rp >= (1.0 - enter_pct):
                    curr_side = 1.0
                elif rp <= enter_pct:
                    curr_side = -1.0
                elif prev_side > 0.0 and rp >= (1.0 - exit_pct):
                    curr_side = 1.0
                elif prev_side < 0.0 and rp <= exit_pct:
                    curr_side = -1.0
                else:
                    curr_side = 0.0
                next_side.loc[row_idx] = curr_side
                prev_side_by_symbol[sym] = curr_side
            out.loc[idx, "side"] = next_side.values
        return out

try:
    from python_edge.portfolio.turnover_control import cap_daily_turnover
    HAS_TURNOVER_CONTROL = True
except Exception:
    HAS_TURNOVER_CONTROL = False

    def cap_daily_turnover(df: pd.DataFrame, weight_col: str = "weight", max_daily_turnover: float = 0.20) -> pd.DataFrame:
        out = df.copy().sort_values(["symbol", "date"]).reset_index(drop=True)
        out["prev_weight"] = out.groupby("symbol", sort=False)[weight_col].shift(1).fillna(0.0)
        out["trade_delta"] = pd.to_numeric(out[weight_col], errors="coerce") - pd.to_numeric(out["prev_weight"], errors="coerce")
        out["trade_abs_after"] = out["trade_delta"].abs()
        day_turn = out.groupby("date", sort=False)["trade_abs_after"].transform("sum")
        scale = np.where(day_turn > max_daily_turnover, max_daily_turnover / (day_turn + 1e-12), 1.0)
        out[weight_col] = pd.to_numeric(out[weight_col], errors="coerce") * scale
        out["trade_delta"] = pd.to_numeric(out["trade_delta"], errors="coerce") * scale
        out["trade_abs_after"] = pd.to_numeric(out["trade_abs_after"], errors="coerce") * scale
        out["cap_hit"] = (day_turn > max_daily_turnover).astype(int)
        return out

EPS = 1e-12

INTERACTION_SUMMARY_FILE = Path(os.getenv("INTERACTION_SUMMARY_FILE", "artifacts/residual_layer_fs2_interactions/residual_component_summary.csv"))
FEATURE_V2_FILE = Path(os.getenv("FEATURE_V2_FILE", "data/features/feature_matrix_v2.parquet"))
ALPHA_LIB_FILE = Path(os.getenv("ALPHA_LIB_FILE", "data/alpha_library_fs2_base/alpha_library_fs2_base.parquet"))
TARGET_COL = str(os.getenv("TARGET_COL", "target_fwd_ret_1d")).strip()
OUT_DIR = Path(os.getenv("PORTFOLIO_OPT_OUT_DIR", "artifacts/portfolio_opt_wf_robustness"))
PAUSE_ON_EXIT = str(os.getenv("PAUSE_ON_EXIT", "auto")).strip().lower()

DEFAULT_TRAIN_DAYS = int(os.getenv("WF_TRAIN_DAYS", "252"))
DEFAULT_TEST_DAYS = int(os.getenv("WF_TEST_DAYS", "63"))
DEFAULT_STEP_DAYS = int(os.getenv("WF_STEP_DAYS", "63"))
WF_PURGE_DAYS = int(os.getenv("WF_PURGE_DAYS", "5"))
WF_EMBARGO_DAYS = int(os.getenv("WF_EMBARGO_DAYS", "5"))
MIN_TRAIN_ROWS = int(os.getenv("MIN_TRAIN_ROWS", "3000"))
MIN_TEST_ROWS = int(os.getenv("MIN_TEST_ROWS", "300"))
MIN_DAILY_IC_CS = int(os.getenv("MIN_DAILY_IC_CS", "20"))

ENTER_PCT = float(os.getenv("ENTER_PCT", "0.10"))
EXIT_PCT = float(os.getenv("EXIT_PCT", "0.22"))
WEIGHT_CAP = float(os.getenv("WEIGHT_CAP", "0.05"))
GROSS_TARGET = float(os.getenv("GROSS_TARGET", "0.85"))
BASE_COST_BPS = float(os.getenv("COST_BPS", "8.0"))
BASE_MAX_DAILY_TURNOVER = float(os.getenv("MAX_DAILY_TURNOVER", "0.20"))
SIGN_LOCK_IC_ABS = float(os.getenv("SIGN_LOCK_IC_ABS", "0.0100"))
SIGN_LOCK_POS_RATE = float(os.getenv("SIGN_LOCK_POS_RATE", "0.55"))
TAIL_TOP_PCT = float(os.getenv("TAIL_TOP_PCT", "0.10"))
TOPK_PRINT = int(os.getenv("TOPK_PRINT", "40"))

COMPONENT_COUNT_LIMIT = int(os.getenv("COMPONENT_COUNT_LIMIT", "2"))
ENABLE_EQUAL_WEIGHT = str(os.getenv("ENABLE_EQUAL_WEIGHT", "1")).strip().lower() not in {"0", "false", "no", "off"}
ENABLE_IC_WEIGHT = str(os.getenv("ENABLE_IC_WEIGHT", "1")).strip().lower() not in {"0", "false", "no", "off"}
ENABLE_SHARPE_WEIGHT = str(os.getenv("ENABLE_SHARPE_WEIGHT", "1")).strip().lower() not in {"0", "false", "no", "off"}
ENABLE_MR_LEG = str(os.getenv("ENABLE_MR_LEG", "1")).strip().lower() not in {"0", "false", "no", "off"}
MR_WEIGHT = float(os.getenv("MR_WEIGHT", "0.50"))
MR_GROSS_MULT = float(os.getenv("MR_GROSS_MULT", "0.35"))
MR_ALPHA_OVERRIDE = str(os.getenv("MR_ALPHA_OVERRIDE", "alpha_fs2_intraday_strength_rvol_interaction__lag1")).strip()
MR_ONLY_TREND_LO = str(os.getenv("MR_ONLY_TREND_LO", "1")).strip().lower() not in {"0", "false", "no", "off"}
AUTO_BUILD_FS2_IF_MISSING = str(os.getenv("AUTO_BUILD_FS2_IF_MISSING", "1")).strip().lower() not in {"0", "false", "no", "off"}
FS2_ENABLE_RAW = str(os.getenv("FS2_ENABLE_RAW", "1")).strip().lower() not in {"0", "false", "no", "off"}
FS2_ENABLE_SIGNED_LOG = str(os.getenv("FS2_ENABLE_SIGNED_LOG", "1")).strip().lower() not in {"0", "false", "no", "off"}
FS2_ENABLE_EMA3 = str(os.getenv("FS2_ENABLE_EMA3", "1")).strip().lower() not in {"0", "false", "no", "off"}
FS2_ENABLE_LAG1 = str(os.getenv("FS2_ENABLE_LAG1", "1")).strip().lower() not in {"0", "false", "no", "off"}
FS2_ENABLE_TANH_Z = str(os.getenv("FS2_ENABLE_TANH_Z", "1")).strip().lower() not in {"0", "false", "no", "off"}

ROBUST_COST_BPS_GRID = str(os.getenv("ROBUST_COST_BPS_GRID", "5|8|12")).strip()
ROBUST_TURNOVER_CAP_GRID = str(os.getenv("ROBUST_TURNOVER_CAP_GRID", "0.15|0.20|0.25")).strip()
ROBUST_WF_GRID = str(os.getenv("ROBUST_WF_GRID", "200:40:40|252:63:63|300:80:80")).strip()


@dataclass(frozen=True)
class WFSplit:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass(frozen=True)
class PortfolioSpec:
    name: str
    weighting: str
    component_count: int
    mr_leg: int


@dataclass(frozen=True)
class RobustConfig:
    name: str
    train_days: int
    test_days: int
    step_days: int
    cost_bps: float
    turnover_cap: float


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


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("float64")


def _safe_nanmean(values: Sequence[float] | np.ndarray | pd.Series) -> float:
    arr = np.asarray(values, dtype="float64")
    if arr.size == 0:
        return float("nan")
    mask = ~np.isnan(arr)
    if not mask.any():
        return float("nan")
    return float(arr[mask].mean())


def _robust_zscore_series(s: pd.Series) -> pd.Series:
    x = _num(s)
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


def _cs_zscore_by_date(df: pd.DataFrame, values: pd.Series) -> pd.Series:
    tmp = pd.DataFrame({"date": df["date"].values, "v": _num(values).values}, index=df.index)
    return tmp.groupby("date", sort=False)["v"].transform(_robust_zscore_series)


def _safe_mean(values: Sequence[float]) -> float:
    vals = [float(x) for x in values if pd.notna(x)]
    return float(np.mean(vals)) if vals else float("nan")


def _safe_last(values: Sequence[float]) -> float:
    vals = [float(x) for x in values if pd.notna(x)]
    return float(vals[-1]) if vals else float("nan")


def _last_n_mean(values: Sequence[float], n: int) -> float:
    vals = [float(x) for x in values if pd.notna(x)]
    return float(np.mean(vals[-n:])) if vals else float("nan")


def _positive_fraction(values: Sequence[float]) -> float:
    vals = [float(x) for x in values if pd.notna(x)]
    return float(sum(1 for x in vals if x > 0.0) / len(vals)) if vals else float("nan")


def _parse_float_grid(text: str, default: Sequence[float]) -> List[float]:
    parts = [x.strip() for x in text.split("|") if x.strip()]
    vals: List[float] = []
    for p in parts:
        try:
            vals.append(float(p))
        except Exception:
            pass
    return vals if vals else list(default)


def _parse_wf_grid(text: str) -> List[Tuple[int, int, int]]:
    out: List[Tuple[int, int, int]] = []
    for part in [x.strip() for x in text.split("|") if x.strip()]:
        pieces = [y.strip() for y in part.split(":") if y.strip()]
        if len(pieces) != 3:
            continue
        try:
            train_days = int(pieces[0])
            test_days = int(pieces[1])
            step_days = int(pieces[2])
            out.append((train_days, test_days, step_days))
        except Exception:
            pass
    if not out:
        out.append((DEFAULT_TRAIN_DAYS, DEFAULT_TEST_DAYS, DEFAULT_STEP_DAYS))
    return out


def _daily_ic_series(frame: pd.DataFrame, factor_col: str, target_col: str, min_cs: int) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for dt, g in frame.groupby("date", sort=False):
        x = g[[factor_col, target_col]].dropna()
        if len(x) < min_cs:
            continue
        if x[factor_col].nunique(dropna=True) <= 1 or x[target_col].nunique(dropna=True) <= 1:
            continue
        ic = x[factor_col].corr(x[target_col], method="spearman")
        if pd.notna(ic):
            rows.append({"date": pd.Timestamp(dt).normalize(), "daily_ic": float(ic)})
    return pd.DataFrame(rows)


def _tail_spread_series(frame: pd.DataFrame, factor_col: str, target_col: str, top_pct: float) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for dt, g in frame.groupby("date", sort=False):
        x = g[[factor_col, target_col]].dropna().copy()
        if len(x) < max(MIN_DAILY_IC_CS, 10):
            continue
        x["rank_pct"] = x[factor_col].rank(method="average", pct=True)
        top = x.loc[x["rank_pct"] >= (1.0 - top_pct), target_col]
        bot = x.loc[x["rank_pct"] <= top_pct, target_col]
        if len(top) == 0 or len(bot) == 0:
            continue
        rows.append({"date": pd.Timestamp(dt).normalize(), "tail_spread": float(top.mean() - bot.mean())})
    return pd.DataFrame(rows)


def _build_walkforward_splits(dates: Sequence[pd.Timestamp], train_days: int, test_days: int, step_days: int) -> List[WFSplit]:
    uniq = pd.Index(sorted(pd.to_datetime(pd.Series(dates)).dt.normalize().unique()))
    if len(uniq) < (train_days + test_days + WF_PURGE_DAYS + WF_EMBARGO_DAYS + 5):
        raise RuntimeError("Not enough dates for walkforward configuration")
    splits: List[WFSplit] = []
    fold_id = 1
    train_end_idx = train_days - 1
    while True:
        test_start_idx = train_end_idx + 1 + WF_PURGE_DAYS + WF_EMBARGO_DAYS
        test_end_idx = test_start_idx + test_days - 1
        if test_end_idx >= len(uniq):
            break
        train_start_idx = train_end_idx - train_days + 1
        splits.append(WFSplit(fold_id=fold_id, train_start=uniq[train_start_idx], train_end=uniq[train_end_idx], test_start=uniq[test_start_idx], test_end=uniq[test_end_idx]))
        fold_id += 1
        train_end_idx += step_days
    if not splits:
        raise RuntimeError("No walkforward splits generated")
    return splits


def _sign_decision(train_ic_df: pd.DataFrame) -> Dict[str, object]:
    if train_ic_df.empty:
        return {
            "train_mean_ic": float("nan"),
            "train_abs_mean_ic": float("nan"),
            "train_pos_rate": float("nan"),
            "train_sign_naive": 1.0,
            "train_sign_locked": 1.0,
            "sign_lock_triggered": 1,
            "sign_lock_reason": "no_train_ic",
        }
    mean_ic = float(train_ic_df["daily_ic"].mean())
    abs_mean_ic = float(train_ic_df["daily_ic"].abs().mean())
    pos_rate = float((train_ic_df["daily_ic"] > 0.0).mean())
    naive_sign = 1.0 if mean_ic >= 0.0 else -1.0
    lock = False
    reason = "ok"
    locked_sign = naive_sign
    if abs(mean_ic) < SIGN_LOCK_IC_ABS:
        lock = True
        reason = "weak_mean_ic"
    elif max(pos_rate, 1.0 - pos_rate) < SIGN_LOCK_POS_RATE:
        lock = True
        reason = "weak_pos_rate"
    if lock:
        if pos_rate > 0.50:
            locked_sign = 1.0
        elif pos_rate < 0.50:
            locked_sign = -1.0
        else:
            locked_sign = 1.0
    return {
        "train_mean_ic": mean_ic,
        "train_abs_mean_ic": abs_mean_ic,
        "train_pos_rate": pos_rate,
        "train_sign_naive": naive_sign,
        "train_sign_locked": locked_sign,
        "sign_lock_triggered": int(lock),
        "sign_lock_reason": reason,
    }


def _load_component_universe() -> pd.DataFrame:
    _must_exist(INTERACTION_SUMMARY_FILE, "Interaction summary")
    df = pd.read_csv(INTERACTION_SUMMARY_FILE)
    if df.empty:
        raise RuntimeError("Interaction summary is empty")
    required = ["candidate", "alpha", "regime", "kind", "oos_sharpe_mean", "last_2_fold_mean_sharpe", "test_mean_ic"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Interaction summary missing columns: {missing}")
    df = df.sort_values(["last_2_fold_mean_sharpe", "oos_sharpe_mean", "test_mean_ic", "candidate"], ascending=[False, False, False, True]).reset_index(drop=True)
    if COMPONENT_COUNT_LIMIT > 0:
        df = df.head(COMPONENT_COUNT_LIMIT).copy()
    if df.empty:
        raise RuntimeError("No components selected for portfolio optimization")
    return df.reset_index(drop=True)


def _pick_regime_source(df: pd.DataFrame, candidates: Sequence[str]) -> Tuple[pd.Series, str]:
    for c in candidates:
        if c in df.columns:
            return _num(df[c]), c
    return pd.Series(np.nan, index=df.index, dtype="float64"), "none"


def _build_explicit_regimes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    _, trend_name = _pick_regime_source(out, ["fs2_intraday_pressure_rs_mix", "z_fs2_intraday_pressure_rs_mix", "fs2_intraday_rs_proxy", "z_fs2_intraday_rs_proxy"])
    day_level = out.groupby("date", sort=False).agg(
        trend_proxy=(trend_name if trend_name != "none" else TARGET_COL, lambda s: float(_safe_nanmean(_num(s)))),
    ).reset_index()
    trend_valid = _num(day_level["trend_proxy"]).fillna(0.0)
    valid = trend_valid.dropna()
    if valid.empty:
        day_level["regime_trend_hi"] = 0.0
        day_level["regime_trend_lo"] = 0.0
    else:
        q_hi = float(valid.quantile(0.67))
        q_lo = float(valid.quantile(0.33))
        day_level["regime_trend_hi"] = np.where(trend_valid >= q_hi, 1.0, 0.0)
        day_level["regime_trend_lo"] = np.where(trend_valid <= q_lo, 1.0, 0.0)
    return out.merge(day_level, on="date", how="left")


def _signed_log(s: pd.Series) -> pd.Series:
    x = _num(s)
    return np.sign(x) * np.log1p(np.abs(x))


def _ema3_by_symbol(frame: pd.DataFrame, col: str) -> pd.Series:
    return frame.groupby("symbol", sort=False)[col].transform(lambda s: _num(s).ewm(span=3, adjust=False, min_periods=1).mean())


def _lag1_by_symbol(frame: pd.DataFrame, col: str) -> pd.Series:
    return frame.groupby("symbol", sort=False)[col].shift(1)


def _auto_build_requested_alpha_columns(df: pd.DataFrame, alpha_names: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    needed = [a for a in alpha_names if a not in out.columns]
    if not needed:
        return out
    built: Dict[str, pd.Series] = {}
    for alpha in needed:
        if not str(alpha).startswith("alpha_") or "__" not in str(alpha):
            continue
        body = str(alpha)[len("alpha_"):]
        family, transform = body.rsplit("__", 1)
        base_col = family if family in out.columns else f"z_{family}" if f"z_{family}" in out.columns else None
        if base_col is None:
            continue
        base = _num(out[base_col])
        if transform == "raw":
            built[alpha] = base
        elif transform == "signed_log":
            built[alpha] = _signed_log(base)
        elif transform == "ema3":
            built[alpha] = _ema3_by_symbol(out.assign(_x=base), "_x")
        elif transform == "lag1":
            built[alpha] = _lag1_by_symbol(out.assign(_x=base), "_x")
        elif transform == "tanh_z":
            built[alpha] = np.tanh(_cs_zscore_by_date(out, base))
    if built:
        out = pd.concat([out, pd.DataFrame(built, index=out.index)], axis=1).copy()
    still_missing = [a for a in alpha_names if a not in out.columns]
    if still_missing:
        raise RuntimeError(f"Missing alpha columns in portfolio opt after auto-build: {still_missing[:10]}")
    return out


def _load_feature_panel(components_df: pd.DataFrame) -> pd.DataFrame:
    _must_exist(FEATURE_V2_FILE, "Feature v2 file")
    df = pd.read_parquet(FEATURE_V2_FILE)
    if df.empty:
        raise RuntimeError("Feature v2 file is empty")
    required = ["date", "symbol", TARGET_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Feature v2 file missing columns: {missing}")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)

    alpha_names = sorted(set(components_df["alpha"].astype(str).tolist())) if "alpha" in components_df.columns else []
    if MR_ALPHA_OVERRIDE:
        alpha_names = sorted(set(alpha_names + [MR_ALPHA_OVERRIDE]))
    if alpha_names:
        if ALPHA_LIB_FILE.exists():
            alpha_df = pd.read_parquet(ALPHA_LIB_FILE)
            if not alpha_df.empty:
                alpha_df["date"] = pd.to_datetime(alpha_df["date"]).dt.normalize()
                keep_alpha_cols = [c for c in alpha_names if c in alpha_df.columns]
                if keep_alpha_cols:
                    df = df.merge(alpha_df[["date", "symbol"] + keep_alpha_cols].copy(), on=["date", "symbol"], how="left")
        if any(a not in df.columns for a in alpha_names):
            if not AUTO_BUILD_FS2_IF_MISSING:
                missing_alpha = [a for a in alpha_names if a not in df.columns]
                raise RuntimeError(f"Missing alpha columns in portfolio opt and AUTO_BUILD_FS2_IF_MISSING=0: {missing_alpha[:10]}")
            df = _auto_build_requested_alpha_columns(df, alpha_names)

    df = _build_explicit_regimes(df)
    return df


def _build_candidate_signal(df: pd.DataFrame, alpha_col: str, regime_col: str, kind: str) -> pd.Series:
    alpha_z = _cs_zscore_by_date(df, _num(df[alpha_col]))
    if kind == "base" or regime_col == "none":
        return alpha_z
    regime = _num(df[regime_col]).fillna(0.0)
    if kind == "gate":
        return alpha_z * regime
    if kind == "sign":
        return alpha_z * (2.0 * regime - 1.0)
    raise RuntimeError(f"Unsupported candidate kind: {kind}")


def _split_frame(df: pd.DataFrame, split: WFSplit) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df.loc[(df["date"] >= split.train_start) & (df["date"] <= split.train_end)].copy()
    test_df = df.loc[(df["date"] >= split.test_start) & (df["date"] <= split.test_end)].copy()
    return train_df, test_df


def _portfolio_weights(spec: PortfolioSpec, comp_train_info: pd.DataFrame) -> Dict[str, float]:
    comp_info = comp_train_info.head(spec.component_count).copy()
    if spec.weighting == "equal":
        vals = {row["candidate"]: 1.0 for _, row in comp_info.iterrows()}
    elif spec.weighting == "ic":
        vals = {row["candidate"]: max(float(pd.to_numeric(row.get("train_abs_mean_ic", np.nan), errors="coerce") or 0.0), EPS) for _, row in comp_info.iterrows()}
    elif spec.weighting == "sharpe":
        vals = {row["candidate"]: max(float(pd.to_numeric(row.get("proxy_train_sharpe", np.nan), errors="coerce") or 0.0), EPS) for _, row in comp_info.iterrows()}
    else:
        raise RuntimeError(f"Unsupported weighting: {spec.weighting}")
    total = float(sum(vals.values()))
    return {k: float(v / total) for k, v in vals.items()} if total > 0.0 else {k: 1.0 / len(vals) for k in vals}


def _mr_alpha_source(components_df: pd.DataFrame) -> str:
    if MR_ALPHA_OVERRIDE:
        return MR_ALPHA_OVERRIDE
    for alpha in components_df["alpha"].astype(str).tolist():
        if "intraday_strength_rvol_interaction" in alpha:
            return alpha
    return str(components_df["alpha"].iloc[0])


def _build_mr_signal(df: pd.DataFrame, alpha_col: str) -> pd.Series:
    return -1.0 * _cs_zscore_by_date(df, _num(df[alpha_col]))


def _build_portfolio_scores(test_df: pd.DataFrame, signal_cols: Dict[str, pd.Series], weights: Dict[str, float], mr_signal: pd.Series | None) -> pd.DataFrame:
    out = test_df[["date", "symbol", TARGET_COL, "regime_trend_hi", "regime_trend_lo"]].copy()
    score = pd.Series(0.0, index=test_df.index, dtype="float64")
    for cand, w in weights.items():
        score = score + (_num(signal_cols[cand]) * float(w))
    out["trend_score_raw"] = _num(score) * _num(test_df["regime_trend_hi"]).fillna(0.0)
    out["mr_score_raw"] = 0.0
    if mr_signal is not None:
        if MR_ONLY_TREND_LO:
            out["mr_score_raw"] = _num(mr_signal) * _num(test_df["regime_trend_lo"]).fillna(0.0) * float(MR_WEIGHT)
        else:
            out["mr_score_raw"] = _num(mr_signal) * float(MR_WEIGHT)
    out["score_raw"] = _num(out["trend_score_raw"]) + _num(out["mr_score_raw"])
    out["score"] = _cs_zscore_by_date(out, _num(out["score_raw"]))
    return out


def _build_portfolio(scored_df: pd.DataFrame, mr_leg: int, turnover_cap: float, cost_bps: float) -> pd.DataFrame:
    base = scored_df.copy()
    base["score"] = _num(base["score"]).fillna(0.0)
    out = apply_holding_inertia(base, enter_pct=ENTER_PCT, exit_pct=EXIT_PCT)
    out["raw_strength"] = _num(out["score"]).abs().fillna(0.0)
    out.loc[_num(out["side"]) == 0.0, "raw_strength"] = 0.0
    pieces: List[pd.DataFrame] = []
    for _, g in out.groupby("date", sort=False):
        gg = g.copy()
        gg["weight"] = 0.0
        long_mask = _num(gg["side"]) > 0.0
        short_mask = _num(gg["side"]) < 0.0
        long_strength = float(gg.loc[long_mask, "raw_strength"].sum())
        short_strength = float(gg.loc[short_mask, "raw_strength"].sum())
        if long_strength > EPS:
            gg.loc[long_mask, "weight"] = 0.5 * gg.loc[long_mask, "raw_strength"] / long_strength
        if short_strength > EPS:
            gg.loc[short_mask, "weight"] = -0.5 * gg.loc[short_mask, "raw_strength"] / short_strength
        gg["weight"] = _num(gg["weight"]).clip(lower=-WEIGHT_CAP, upper=WEIGHT_CAP)
        trend_hi = float(_num(gg.get("regime_trend_hi", pd.Series(0.0, index=gg.index))).iloc[0])
        trend_lo = float(_num(gg.get("regime_trend_lo", pd.Series(0.0, index=gg.index))).iloc[0])
        if trend_hi >= 0.5:
            gross_mult = 1.0
            gross_reason = "trend_hi"
        elif trend_lo >= 0.5 and mr_leg == 1:
            gross_mult = float(MR_GROSS_MULT)
            gross_reason = "trend_lo_mr"
        else:
            gross_mult = 0.50
            gross_reason = "neutral"
        dynamic_gross_target = float(GROSS_TARGET) * gross_mult
        gg["dynamic_gross_target"] = float(dynamic_gross_target)
        gg["dynamic_gross_reason"] = gross_reason
        gross = float(gg["weight"].abs().sum())
        if gross > EPS:
            gg["weight"] = gg["weight"] * (float(dynamic_gross_target) / gross)
        pieces.append(gg)
    out = pd.concat(pieces, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    out = cap_daily_turnover(out, weight_col="weight", max_daily_turnover=turnover_cap)
    if "trade_abs_after" in out.columns:
        out["turnover"] = _num(out["trade_abs_after"])
    else:
        out["prev_weight"] = out.groupby("symbol", sort=False)["weight"].shift(1).fillna(0.0)
        out["turnover"] = (_num(out["weight"]) - _num(out["prev_weight"])).abs()
    out["gross_ret"] = _num(out["weight"]) * _num(out[TARGET_COL])
    out["cost_ret"] = _num(out["turnover"]) * (cost_bps / 10000.0)
    out["net_ret"] = _num(out["gross_ret"]) - _num(out["cost_ret"])
    return out


def _evaluate_portfolio(port_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    daily = port_df.groupby("date", sort=False, as_index=False).agg(
        gross_ret=("gross_ret", "sum"),
        cost_ret=("cost_ret", "sum"),
        net_ret=("net_ret", "sum"),
        turnover=("turnover", "sum"),
        gross_exposure=("weight", lambda s: float(_num(s).abs().sum())),
        names_active=("side", lambda s: int((_num(s).abs() > 0).sum())),
        dynamic_gross_target=("dynamic_gross_target", "first"),
    )
    daily = daily.sort_values("date").reset_index(drop=True)
    daily["equity"] = (1.0 + daily["net_ret"].fillna(0.0)).cumprod()
    daily["rolling_peak"] = daily["equity"].cummax()
    daily["drawdown"] = np.where(daily["rolling_peak"] > 0.0, daily["equity"] / daily["rolling_peak"] - 1.0, 0.0)
    mean_daily = _safe_nanmean(daily["net_ret"].to_numpy(dtype="float64")) if len(daily) else float("nan")
    std_daily = float(pd.to_numeric(daily["net_ret"], errors="coerce").std(ddof=0)) if len(daily) else float("nan")
    sharpe = float((mean_daily / (std_daily + EPS)) * np.sqrt(252.0)) if len(daily) else float("nan")
    return daily, {
        "days": float(len(daily)),
        "mean_daily": mean_daily,
        "std_daily": std_daily,
        "sharpe": sharpe,
        "cum_ret": float(daily["equity"].iloc[-1] - 1.0) if len(daily) else float("nan"),
        "max_drawdown": float(daily["drawdown"].min()) if len(daily) else float("nan"),
        "avg_turnover": _safe_nanmean(daily["turnover"].to_numpy(dtype="float64")) if len(daily) else float("nan"),
        "avg_names_active": _safe_nanmean(daily["names_active"].to_numpy(dtype="float64")) if len(daily) else float("nan"),
        "avg_gross_exposure": _safe_nanmean(daily["gross_exposure"].to_numpy(dtype="float64")) if len(daily) else float("nan"),
        "avg_dynamic_gross_target": _safe_nanmean(daily["dynamic_gross_target"].to_numpy(dtype="float64")) if len(daily) else float("nan"),
    }


def _portfolio_specs(max_components: int) -> List[PortfolioSpec]:
    specs: List[PortfolioSpec] = []
    weightings: List[str] = []
    if ENABLE_EQUAL_WEIGHT:
        weightings.append("equal")
    if ENABLE_IC_WEIGHT:
        weightings.append("ic")
    if ENABLE_SHARPE_WEIGHT:
        weightings.append("sharpe")
    mr_options = [0, 1] if ENABLE_MR_LEG else [0]
    for n in range(1, max_components + 1):
        for weighting in weightings:
            for mr in mr_options:
                specs.append(PortfolioSpec(name=f"pf__n{n}__{weighting}__mr{mr}", weighting=weighting, component_count=n, mr_leg=mr))
    return specs


def _build_component_signals(train_df: pd.DataFrame, test_df: pd.DataFrame, components_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    info_rows: List[Dict[str, object]] = []
    train_signal_cols: Dict[str, pd.Series] = {}
    test_signal_cols: Dict[str, pd.Series] = {}
    for _, rec in components_df.iterrows():
        candidate = str(rec["candidate"])
        alpha = str(rec["alpha"])
        regime = str(rec["regime"])
        kind = str(rec["kind"])
        train_signal_raw = _build_candidate_signal(train_df, alpha, regime, kind)
        test_signal_raw = _build_candidate_signal(test_df, alpha, regime, kind)
        train_signal = _cs_zscore_by_date(train_df, train_signal_raw)
        test_signal = _cs_zscore_by_date(test_df, test_signal_raw)
        train_scored = pd.DataFrame({"date": train_df["date"].values, "symbol": train_df["symbol"].values, TARGET_COL: _num(train_df[TARGET_COL]).values, "score": _num(train_signal).values})
        train_ic_df = _daily_ic_series(train_scored, "score", TARGET_COL, MIN_DAILY_IC_CS)
        sign_info = _sign_decision(train_ic_df)
        if len(train_ic_df):
            proxy_train_mean = _safe_nanmean(train_ic_df["daily_ic"].to_numpy(dtype="float64"))
            proxy_train_std = float(pd.to_numeric(train_ic_df["daily_ic"], errors="coerce").std(ddof=0))
            proxy_train_sharpe = float((proxy_train_mean / (proxy_train_std + EPS)) * np.sqrt(252.0))
        else:
            proxy_train_sharpe = float("nan")
        info_rows.append({"candidate": candidate, "alpha": alpha, "regime": regime, "kind": kind, **sign_info, "proxy_train_sharpe": proxy_train_sharpe})
        train_signal_cols[candidate] = _num(train_signal) * float(sign_info["train_sign_locked"])
        test_signal_cols[candidate] = _num(test_signal) * float(sign_info["train_sign_locked"])
    mr_alpha = _mr_alpha_source(components_df)
    train_sig_df = pd.concat([train_df[["date", "symbol", TARGET_COL]].copy(), pd.DataFrame(train_signal_cols, index=train_df.index)], axis=1)
    test_sig_df = pd.concat([test_df[["date", "symbol", TARGET_COL, "regime_trend_hi", "regime_trend_lo"]].copy(), pd.DataFrame(test_signal_cols, index=test_df.index)], axis=1)
    return train_sig_df, test_sig_df, pd.DataFrame(info_rows), mr_alpha


def _run_fold(df: pd.DataFrame, components_df: pd.DataFrame, split: WFSplit, robust: RobustConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, test_df = _split_frame(df, split)
    if len(train_df) < MIN_TRAIN_ROWS:
        raise RuntimeError(f"Fold {split.fold_id}: too few train rows: {len(train_df)}")
    if len(test_df) < MIN_TEST_ROWS:
        raise RuntimeError(f"Fold {split.fold_id}: too few test rows: {len(test_df)}")

    train_sig_df, test_sig_df, comp_train_info, mr_alpha = _build_component_signals(train_df, test_df, components_df)
    if mr_alpha not in test_df.columns:
        raise RuntimeError(f"MR alpha source column not found: {mr_alpha}")
    mr_signal_test = _build_mr_signal(test_df, mr_alpha)
    specs = _portfolio_specs(max_components=min(len(components_df), COMPONENT_COUNT_LIMIT))

    result_rows: List[Dict[str, object]] = []
    daily_frames: List[pd.DataFrame] = []
    for spec in specs:
        chosen = comp_train_info.head(spec.component_count).copy()
        weights = _portfolio_weights(spec, chosen)
        mr_signal = mr_signal_test if spec.mr_leg == 1 else None
        scored = _build_portfolio_scores(test_df, {cand: test_sig_df[cand] for cand in weights.keys()}, weights, mr_signal)
        port_df = _build_portfolio(scored, spec.mr_leg, robust.turnover_cap, robust.cost_bps)
        daily_df, summary = _evaluate_portfolio(port_df)
        daily_df["fold_id"] = int(split.fold_id)
        daily_df["portfolio"] = spec.name
        daily_df["robust_name"] = robust.name
        daily_frames.append(daily_df)
        test_ic_df = _daily_ic_series(scored[["date", "symbol", TARGET_COL, "score"]].copy(), "score", TARGET_COL, MIN_DAILY_IC_CS)
        tail_df = _tail_spread_series(scored[["date", "symbol", TARGET_COL, "score"]].copy(), "score", TARGET_COL, TAIL_TOP_PCT)
        result_rows.append({
            "robust_name": robust.name,
            "fold_id": int(split.fold_id),
            "portfolio": spec.name,
            "weighting": spec.weighting,
            "component_count": int(spec.component_count),
            "mr_leg": int(spec.mr_leg),
            "mr_alpha_source": mr_alpha if spec.mr_leg == 1 else "none",
            "components": "|".join(chosen["candidate"].astype(str).tolist()),
            "weights": json.dumps(weights, ensure_ascii=False, sort_keys=True),
            "cost_bps": float(robust.cost_bps),
            "turnover_cap": float(robust.turnover_cap),
            "train_days": int(robust.train_days),
            "test_days": int(robust.test_days),
            "step_days": int(robust.step_days),
            **summary,
            "test_mean_ic": _safe_nanmean(test_ic_df["daily_ic"].to_numpy(dtype="float64")) if len(test_ic_df) else float("nan"),
            "test_sign_stability": float((test_ic_df["daily_ic"] > 0.0).mean()) if len(test_ic_df) else float("nan"),
            "tail_mean_spread": _safe_nanmean(tail_df["tail_spread"].to_numpy(dtype="float64")) if len(tail_df) else float("nan"),
            "tail_positive_rate": float((tail_df["tail_spread"] > 0.0).mean()) if len(tail_df) else float("nan"),
        })
    return pd.DataFrame(result_rows), pd.concat(daily_frames, ignore_index=True), comp_train_info.assign(robust_name=robust.name, fold_id=int(split.fold_id), mr_alpha_source=mr_alpha)


def _summarize_portfolios(portfolio_fold_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    if portfolio_fold_df.empty:
        return pd.DataFrame()
    grouped = portfolio_fold_df.sort_values(["robust_name", "portfolio", "fold_id"]).groupby(["robust_name", "portfolio"], sort=False)
    for (robust_name, portfolio), g in grouped:
        rows.append({
            "robust_name": robust_name,
            "portfolio": portfolio,
            "weighting": str(g["weighting"].iloc[0]),
            "component_count": int(pd.to_numeric(g["component_count"], errors="coerce").iloc[0]),
            "mr_leg": int(pd.to_numeric(g["mr_leg"], errors="coerce").iloc[0]),
            "mr_alpha_source": str(g["mr_alpha_source"].iloc[0]),
            "components": str(g["components"].iloc[0]),
            "cost_bps": float(pd.to_numeric(g["cost_bps"], errors="coerce").iloc[0]),
            "turnover_cap": float(pd.to_numeric(g["turnover_cap"], errors="coerce").iloc[0]),
            "train_days": int(pd.to_numeric(g["train_days"], errors="coerce").iloc[0]),
            "test_days": int(pd.to_numeric(g["test_days"], errors="coerce").iloc[0]),
            "step_days": int(pd.to_numeric(g["step_days"], errors="coerce").iloc[0]),
            "folds": int(len(g)),
            "oos_sharpe_mean": _safe_mean(g["sharpe"].tolist()),
            "last_fold_sharpe": _safe_last(g["sharpe"].tolist()),
            "last_2_fold_mean_sharpe": _last_n_mean(g["sharpe"].tolist(), 2),
            "test_mean_ic": _safe_mean(g["test_mean_ic"].tolist()),
            "sign_stability": _safe_mean(g["test_sign_stability"].tolist()),
            "tail_positive_rate": _safe_mean(g["tail_positive_rate"].tolist()),
            "max_drawdown": float(pd.to_numeric(g["max_drawdown"], errors="coerce").min()),
            "turnover_mean": _safe_mean(g["avg_turnover"].tolist()),
            "positive_sharpe_fraction": _positive_fraction(g["sharpe"].tolist()),
            "avg_dynamic_gross_target": _safe_mean(g["avg_dynamic_gross_target"].tolist()),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["last_2_fold_mean_sharpe", "oos_sharpe_mean", "robust_name", "portfolio"], ascending=[False, False, True, True]).reset_index(drop=True)
    return out


def _robustness_grid() -> List[RobustConfig]:
    cost_grid = _parse_float_grid(ROBUST_COST_BPS_GRID, [BASE_COST_BPS])
    turn_grid = _parse_float_grid(ROBUST_TURNOVER_CAP_GRID, [BASE_MAX_DAILY_TURNOVER])
    wf_grid = _parse_wf_grid(ROBUST_WF_GRID)
    rows: List[RobustConfig] = []
    for train_days, test_days, step_days in wf_grid:
        for cost_bps in cost_grid:
            for turnover_cap in turn_grid:
                name = f"wf{train_days}_{test_days}_{step_days}__cost{cost_bps:.1f}__turn{turnover_cap:.2f}"
                rows.append(RobustConfig(name=name, train_days=train_days, test_days=test_days, step_days=step_days, cost_bps=cost_bps, turnover_cap=turnover_cap))
    return rows


def main() -> int:
    _enable_line_buffering()
    print(f"[CFG] interaction_summary_file={INTERACTION_SUMMARY_FILE}")
    print(f"[CFG] feature_v2_file={FEATURE_V2_FILE}")
    print(f"[CFG] alpha_lib_file={ALPHA_LIB_FILE}")
    print(f"[CFG] out_dir={OUT_DIR}")
    print(f"[CFG] shell gross_target={GROSS_TARGET} weight_cap={WEIGHT_CAP} enter_pct={ENTER_PCT} exit_pct={EXIT_PCT}")
    print(f"[CFG] component_count_limit={COMPONENT_COUNT_LIMIT}")
    print(f"[CFG] weighting equal={int(ENABLE_EQUAL_WEIGHT)} ic={int(ENABLE_IC_WEIGHT)} sharpe={int(ENABLE_SHARPE_WEIGHT)}")
    print(f"[CFG] mr_leg={int(ENABLE_MR_LEG)} mr_weight={MR_WEIGHT} mr_gross_mult={MR_GROSS_MULT} mr_alpha_override={MR_ALPHA_OVERRIDE!r}")
    print(f"[CFG] robustness cost_grid={ROBUST_COST_BPS_GRID!r} turnover_grid={ROBUST_TURNOVER_CAP_GRID!r} wf_grid={ROBUST_WF_GRID!r}")

    components_df = _load_component_universe()
    print("[COMPONENTS][SELECTED]")
    print(components_df[["candidate", "alpha", "regime", "kind", "oos_sharpe_mean", "last_2_fold_mean_sharpe", "test_mean_ic"]].to_string(index=False))

    df = _load_feature_panel(components_df)
    print(f"[DATA] rows={len(df)} dates={df['date'].nunique()} symbols={df['symbol'].nunique()}")

    robust_grid = _robustness_grid()
    print(f"[ROBUSTNESS] configs={len(robust_grid)}")
    for cfg in robust_grid:
        print(f"[ROBUSTNESS][CFG] {cfg.name}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fold_rows_all: List[pd.DataFrame] = []
    daily_rows_all: List[pd.DataFrame] = []
    comp_info_all: List[pd.DataFrame] = []

    for robust in robust_grid:
        splits = _build_walkforward_splits(df["date"], robust.train_days, robust.test_days, robust.step_days)
        print(f"[WF][{robust.name}] folds={len(splits)}")
        for split in splits:
            fold_pf_df, fold_daily_df, fold_comp_info = _run_fold(df, components_df, split, robust)
            fold_rows_all.append(fold_pf_df)
            daily_rows_all.append(fold_daily_df)
            comp_info_all.append(fold_comp_info)
            fold_pf_df.to_csv(OUT_DIR / f"portfolio_fold_metrics__{robust.name}__fold{split.fold_id}.csv", index=False)
            fold_daily_df.to_csv(OUT_DIR / f"portfolio_daily__{robust.name}__fold{split.fold_id}.csv", index=False)
            fold_comp_info.to_csv(OUT_DIR / f"component_train_info__{robust.name}__fold{split.fold_id}.csv", index=False)
            for _, row in fold_pf_df.sort_values("portfolio").iterrows():
                print(f"[{robust.name}][FOLD {split.fold_id}][{row['portfolio']}] sharpe={row['sharpe']:.4f} mean_ic={row['test_mean_ic']:.5f} sign_stability={row['test_sign_stability']:.4f} maxdd={row['max_drawdown']:.4f} turnover={row['avg_turnover']:.4f}")

    portfolio_fold_df = pd.concat(fold_rows_all, ignore_index=True) if fold_rows_all else pd.DataFrame()
    daily_df = pd.concat(daily_rows_all, ignore_index=True) if daily_rows_all else pd.DataFrame()
    comp_info_df = pd.concat(comp_info_all, ignore_index=True) if comp_info_all else pd.DataFrame()
    portfolio_summary_df = _summarize_portfolios(portfolio_fold_df)

    portfolio_fold_df.to_csv(OUT_DIR / "portfolio_fold_metrics__all_folds.csv", index=False)
    daily_df.to_csv(OUT_DIR / "portfolio_daily__all_folds.csv", index=False)
    comp_info_df.to_csv(OUT_DIR / "component_train_info__all_folds.csv", index=False)
    portfolio_summary_df.to_csv(OUT_DIR / "portfolio_summary.csv", index=False)

    if len(portfolio_summary_df):
        print("[SUMMARY][PORTFOLIOS]")
        print(portfolio_summary_df.head(TOPK_PRINT).to_string(index=False))

    meta = {
        "interaction_summary_file": str(INTERACTION_SUMMARY_FILE),
        "feature_v2_file": str(FEATURE_V2_FILE),
        "alpha_lib_file": str(ALPHA_LIB_FILE),
        "target_col": TARGET_COL,
        "robustness": [x.__dict__ for x in robust_grid],
        "shell": {
            "enter_pct": ENTER_PCT,
            "exit_pct": EXIT_PCT,
            "weight_cap": WEIGHT_CAP,
            "gross_target": GROSS_TARGET,
            "base_cost_bps": BASE_COST_BPS,
            "base_turnover_cap": BASE_MAX_DAILY_TURNOVER,
        },
        "portfolio_grid": {
            "component_count_limit": COMPONENT_COUNT_LIMIT,
            "equal_weight": int(ENABLE_EQUAL_WEIGHT),
            "ic_weight": int(ENABLE_IC_WEIGHT),
            "sharpe_weight": int(ENABLE_SHARPE_WEIGHT),
            "mr_leg": int(ENABLE_MR_LEG),
            "mr_weight": MR_WEIGHT,
            "mr_gross_mult": MR_GROSS_MULT,
            "mr_alpha_override": MR_ALPHA_OVERRIDE,
        },
        "modules": {
            "holding_inertia": int(HAS_HOLDING_INERTIA),
            "turnover_control": int(HAS_TURNOVER_CONTROL),
        },
        "rows": {
            "portfolio_fold_rows": int(len(portfolio_fold_df)),
            "daily_rows": int(len(daily_df)),
            "component_train_info_rows": int(len(comp_info_df)),
            "portfolio_summary_rows": int(len(portfolio_summary_df)),
        },
    }
    meta_path = OUT_DIR / "portfolio_opt_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ARTIFACT] {OUT_DIR / 'portfolio_summary.csv'}")
    print(f"[ARTIFACT] {OUT_DIR / 'portfolio_fold_metrics__all_folds.csv'}")
    print(f"[ARTIFACT] {OUT_DIR / 'portfolio_daily__all_folds.csv'}")
    print(f"[ARTIFACT] {OUT_DIR / 'component_train_info__all_folds.csv'}")
    print(f"[ARTIFACT] {meta_path}")
    print("[FINAL] portfolio optimization robustness grid complete")
    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 1
    _safe_exit(rc)
