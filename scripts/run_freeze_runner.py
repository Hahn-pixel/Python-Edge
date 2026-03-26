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
except Exception:
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
except Exception:
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
UNIVERSE_SNAPSHOT_FILE = Path(os.getenv("UNIVERSE_SNAPSHOT_FILE", "artifacts/daily_cycle/universe/universe_snapshot.parquet"))
UNIVERSE_REQUIRE_SELECTED = str(os.getenv("UNIVERSE_REQUIRE_SELECTED", "1")).strip().lower() not in {"0", "false", "no", "off"}
ALPHA_LIB_FILE = Path(os.getenv("ALPHA_LIB_FILE", r"data/alpha_library_v2/alpha_library_v2.parquet"))
TARGET_COL = str(os.getenv("TARGET_COL", "target_fwd_ret_1d")).strip()
OUT_DIR = Path(os.getenv("FREEZE_OUT_DIR", "artifacts/freeze_runner"))
PAUSE_ON_EXIT = str(os.getenv("PAUSE_ON_EXIT", "auto")).strip().lower()

FREEZE_TRAIN_DAYS = int(os.getenv("FREEZE_TRAIN_DAYS", "200"))
FREEZE_TEST_DAYS = int(os.getenv("FREEZE_TEST_DAYS", "50"))
FREEZE_STEP_DAYS = int(os.getenv("FREEZE_STEP_DAYS", "50"))
FREEZE_COMPONENT_COUNT = int(os.getenv("FREEZE_COMPONENT_COUNT", "2"))
FREEZE_WEIGHTING_MODE = str(os.getenv("FREEZE_WEIGHTING_MODE", "ic")).strip().lower()
FREEZE_MR_ENABLED = str(os.getenv("FREEZE_MR_ENABLED", "1")).strip().lower() not in {"0", "false", "no", "off"}
FREEZE_MR_ALPHA = str(os.getenv("FREEZE_MR_ALPHA", "alpha_fs2_intraday_strength_rvol_interaction__lag1")).strip()
FREEZE_MR_ONLY_TREND_LO = str(os.getenv("FREEZE_MR_ONLY_TREND_LO", "1")).strip().lower() not in {"0", "false", "no", "off"}
FREEZE_MR_GROSS_MULT = float(os.getenv("FREEZE_MR_GROSS_MULT", "0.35"))
FALLBACK_ENABLE_RAW_FEATURE_COMPONENTS = str(os.getenv("FALLBACK_ENABLE_RAW_FEATURE_COMPONENTS", "1")).strip().lower() not in {"0", "false", "no", "off"}
FALLBACK_MAX_CANDIDATES = int(os.getenv("FALLBACK_MAX_CANDIDATES", "200"))

ENTER_PCT = float(os.getenv("ENTER_PCT", "0.10"))
EXIT_PCT = float(os.getenv("EXIT_PCT", "0.22"))
WEIGHT_CAP = float(os.getenv("WEIGHT_CAP", "0.05"))
GROSS_TARGET = float(os.getenv("GROSS_TARGET", "0.85"))
SIGN_LOCK_IC_ABS = float(os.getenv("SIGN_LOCK_IC_ABS", "0.0100"))
MIN_DAILY_IC_CS = int(os.getenv("MIN_DAILY_IC_CS", "20"))
MIN_TRAIN_ROWS = int(os.getenv("MIN_TRAIN_ROWS", "3000"))
WF_PURGE_DAYS = int(os.getenv("WF_PURGE_DAYS", "5"))
WF_EMBARGO_DAYS = int(os.getenv("WF_EMBARGO_DAYS", "5"))
TOPK_EXPORT = int(os.getenv("TOPK_EXPORT", "50"))
AUTO_BUILD_FS2_IF_MISSING = str(os.getenv("AUTO_BUILD_FS2_IF_MISSING", "1")).strip().lower() not in {"0", "false", "no", "off"}

NON_FEATURE_COLS = {
    "date", "symbol", TARGET_COL, "open", "high", "low", "close", "volume", "dollar_vol", "dollar_volume",
    "trend_bucket", "vol_bucket", "mr_bucket", "regime_trend_hi", "regime_trend_lo", "regime_vol_hi", "regime_vol_lo",
    "regime_mr_hi", "regime_mr_lo",
}


@dataclass(frozen=True)
class WFSplit:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass(frozen=True)
class FreezeConfig:
    name: str
    cost_bps: float
    turnover_cap: float
    mr_weight: float


FREEZE_CONFIGS: List[FreezeConfig] = [
    FreezeConfig(name="optimal", cost_bps=4.0, turnover_cap=0.10, mr_weight=0.50),
    FreezeConfig(name="aggressive", cost_bps=4.0, turnover_cap=0.20, mr_weight=0.50),
]


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
    return bool(stdin_obj and stdout_obj and hasattr(stdin_obj, "isatty") and hasattr(stdout_obj, "isatty") and stdin_obj.isatty() and stdout_obj.isatty())


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


def _build_walkforward_splits(dates: Sequence[pd.Timestamp], train_days: int, test_days: int, step_days: int) -> List[WFSplit]:
    uniq = pd.Index(sorted(pd.to_datetime(pd.Series(dates)).dt.normalize().unique()))
    if len(uniq) < (train_days + test_days + WF_PURGE_DAYS + WF_EMBARGO_DAYS + 5):
        raise RuntimeError("Not enough dates for walkforward configuration")
    splits: List[WFSplit] = []
    fold_id = 0
    start_idx = 0
    while True:
        train_start_idx = start_idx
        train_end_idx = train_start_idx + train_days - 1
        test_start_idx = train_end_idx + 1 + WF_PURGE_DAYS + WF_EMBARGO_DAYS
        test_end_idx = test_start_idx + test_days - 1
        if test_end_idx >= len(uniq):
            break
        fold_id += 1
        splits.append(
            WFSplit(
                fold_id=fold_id,
                train_start=pd.Timestamp(uniq[train_start_idx]).normalize(),
                train_end=pd.Timestamp(uniq[train_end_idx]).normalize(),
                test_start=pd.Timestamp(uniq[test_start_idx]).normalize(),
                test_end=pd.Timestamp(uniq[test_end_idx]).normalize(),
            )
        )
        start_idx += step_days
    if not splits:
        raise RuntimeError("No valid walkforward splits created")
    return splits


def _weight_from_train_info(train_info_df: pd.DataFrame) -> pd.Series:
    if train_info_df.empty:
        return pd.Series(dtype="float64")
    if FREEZE_WEIGHTING_MODE == "ic":
        w = _num(train_info_df["train_abs_mean_ic"]).clip(lower=0.0)
    elif FREEZE_WEIGHTING_MODE == "sharpe":
        w = _num(train_info_df["proxy_train_sharpe"]).abs().clip(lower=0.0)
    else:
        raise RuntimeError(f"Unsupported FREEZE_WEIGHTING_MODE: {FREEZE_WEIGHTING_MODE}")
    if float(w.sum()) <= EPS:
        w = pd.Series(np.ones(len(train_info_df), dtype="float64"), index=train_info_df.index)
    w = w / float(w.sum())
    return w


def _portfolio_weights(train_info_df: pd.DataFrame) -> Dict[str, float]:
    w = _weight_from_train_info(train_info_df)
    out: Dict[str, float] = {}
    for idx, row in train_info_df.reset_index(drop=True).iterrows():
        out[str(row["candidate"])] = float(w.iloc[idx])
    return out


def _build_explicit_regimes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "trend_bucket" in out.columns:
        tb = out["trend_bucket"].astype(str).str.lower()
        out["regime_trend_hi"] = (tb == "high").astype(float)
        out["regime_trend_lo"] = (tb == "low").astype(float)
    else:
        out["regime_trend_hi"] = 0.0
        out["regime_trend_lo"] = 0.0
    if "vol_bucket" in out.columns:
        vb = out["vol_bucket"].astype(str).str.lower()
        out["regime_vol_hi"] = (vb == "high").astype(float)
        out["regime_vol_lo"] = (vb == "low").astype(float)
    else:
        out["regime_vol_hi"] = 0.0
        out["regime_vol_lo"] = 0.0
    if "mr_bucket" in out.columns:
        mb = out["mr_bucket"].astype(str).str.lower()
        out["regime_mr_hi"] = (mb == "high").astype(float)
        out["regime_mr_lo"] = (mb == "low").astype(float)
    else:
        out["regime_mr_hi"] = 0.0
        out["regime_mr_lo"] = 0.0
    return out


def _ema3_by_symbol(frame: pd.DataFrame, col: str) -> pd.Series:
    return frame.groupby("symbol", sort=False)[col].transform(lambda s: _num(s).ewm(span=3, adjust=False, min_periods=1).mean())


def _lag1_by_symbol(frame: pd.DataFrame, col: str) -> pd.Series:
    return frame.groupby("symbol", sort=False)[col].shift(1)


def _auto_build_requested_alpha_columns(df: pd.DataFrame, alpha_names: Sequence[str]) -> tuple[pd.DataFrame, List[str], List[str]]:
    out = df.copy()
    built_any = True
    while built_any:
        built_any = False
        for alpha in alpha_names:
            if alpha in out.columns:
                continue
            if alpha.endswith("__ema3"):
                base_alpha = alpha[:-6]
                if base_alpha in out.columns:
                    base = _num(out[base_alpha])
                    out[alpha] = _ema3_by_symbol(out.assign(_x=base), "_x")
                    built_any = True
                    continue
            if alpha.endswith("__lag1"):
                base_alpha = alpha[:-6]
                if base_alpha in out.columns:
                    base = _num(out[base_alpha])
                    out[alpha] = _lag1_by_symbol(out.assign(_x=base), "_x")
                    built_any = True
                    continue
    available = sorted([a for a in alpha_names if a in out.columns])
    missing = sorted([a for a in alpha_names if a not in out.columns])
    return out, available, missing


def _load_interaction_summary() -> pd.DataFrame:
    _must_exist(INTERACTION_SUMMARY_FILE, "Interaction summary")
    df = pd.read_csv(INTERACTION_SUMMARY_FILE)
    if df.empty:
        raise RuntimeError("Interaction summary is empty")
    required = ["candidate", "alpha", "regime", "kind", "oos_sharpe_mean", "last_2_fold_mean_sharpe", "test_mean_ic"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Interaction summary missing required columns: {missing}")
    df = df.sort_values(["last_2_fold_mean_sharpe", "oos_sharpe_mean", "test_mean_ic", "candidate"], ascending=[False, False, False, True]).reset_index(drop=True)
    return df


def _load_selected_universe_symbols() -> tuple[list[str], dict]:
    _must_exist(UNIVERSE_SNAPSHOT_FILE, "Universe snapshot")
    snap = pd.read_parquet(UNIVERSE_SNAPSHOT_FILE)
    if snap.empty:
        raise RuntimeError("Universe snapshot is empty")
    required_cols = {"ticker", "selected"}
    missing = sorted(required_cols - set(snap.columns))
    if missing:
        raise RuntimeError(f"Universe snapshot missing required columns: {missing}")
    snap["ticker"] = snap["ticker"].astype(str).str.upper()
    snap["selected"] = snap["selected"].fillna(False).astype(bool)
    total_rows = int(len(snap))
    selected_df = snap.loc[snap["selected"]].copy()
    if UNIVERSE_REQUIRE_SELECTED and selected_df.empty:
        raise RuntimeError("Universe snapshot has zero selected rows while UNIVERSE_REQUIRE_SELECTED=1")
    selected_symbols = sorted(selected_df["ticker"].dropna().astype(str).str.upper().unique().tolist())
    summary = {
        "snapshot_path": str(UNIVERSE_SNAPSHOT_FILE),
        "snapshot_rows": total_rows,
        "selected_rows": int(len(selected_df)),
        "selected_symbols": int(len(selected_symbols)),
        "as_of_date": str(selected_df["trade_date"].iloc[0]) if (not selected_df.empty and "trade_date" in selected_df.columns) else None,
    }
    return selected_symbols, summary


def _discover_fallback_components(df: pd.DataFrame, full_components_df: pd.DataFrame) -> pd.DataFrame:
    if not FALLBACK_ENABLE_RAW_FEATURE_COMPONENTS:
        return pd.DataFrame(columns=full_components_df.columns)
    numeric_cols = []
    for col in df.columns:
        if col in NON_FEATURE_COLS:
            continue
        if str(col).startswith("signal__"):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(str(col))
    numeric_cols = sorted(numeric_cols)[:FALLBACK_MAX_CANDIDATES]
    if not numeric_cols:
        return pd.DataFrame(columns=full_components_df.columns)
    uniq_dates = pd.Index(sorted(pd.to_datetime(df["date"]).dt.normalize().unique()))
    last_idx = len(uniq_dates) - 1
    live_train_end_idx = last_idx - (WF_PURGE_DAYS + WF_EMBARGO_DAYS + 1)
    if live_train_end_idx < (FREEZE_TRAIN_DAYS - 1):
        raise RuntimeError("Not enough history for fallback component discovery")
    live_train_start_idx = live_train_end_idx - FREEZE_TRAIN_DAYS + 1
    live_train_start = pd.Timestamp(uniq_dates[live_train_start_idx]).normalize()
    live_train_end = pd.Timestamp(uniq_dates[live_train_end_idx]).normalize()
    train_df = df.loc[(df["date"] >= live_train_start) & (df["date"] <= live_train_end)].copy()
    rows: List[Dict[str, object]] = []
    for col in numeric_cols:
        scored = pd.DataFrame({
            "date": train_df["date"].values,
            "symbol": train_df["symbol"].values,
            TARGET_COL: _num(train_df[TARGET_COL]).values,
            "score": _cs_zscore_by_date(train_df, _num(train_df[col])).values,
        })
        ic_df = _daily_ic_series(scored, "score", TARGET_COL, MIN_DAILY_IC_CS)
        if ic_df.empty:
            continue
        mean_ic = float(ic_df["daily_ic"].mean())
        abs_mean_ic = float(ic_df["daily_ic"].abs().mean())
        std_ic = float(pd.to_numeric(ic_df["daily_ic"], errors="coerce").std(ddof=0))
        proxy_sharpe = float((mean_ic / (std_ic + EPS)) * np.sqrt(252.0))
        rows.append({
            "candidate": f"fallback::{col}",
            "alpha": col,
            "regime": "none",
            "kind": "base",
            "oos_sharpe_mean": proxy_sharpe,
            "last_2_fold_mean_sharpe": proxy_sharpe,
            "test_mean_ic": mean_ic,
            "source": "fallback_feature_matrix",
            "train_abs_mean_ic": abs_mean_ic,
        })
    if not rows:
        return pd.DataFrame(columns=full_components_df.columns)
    out = pd.DataFrame(rows)
    out = out.sort_values(["train_abs_mean_ic", "oos_sharpe_mean", "candidate"], ascending=[False, False, True]).reset_index(drop=True)
    return out.head(FREEZE_COMPONENT_COUNT).copy()


def _load_feature_panel(full_components_df: pd.DataFrame, requested_mr_alpha: str) -> tuple[pd.DataFrame, pd.DataFrame, str | None, dict]:
    _must_exist(FEATURE_V2_FILE, "Feature v2 file")
    selected_symbols, universe_summary = _load_selected_universe_symbols()
    df = pd.read_parquet(FEATURE_V2_FILE)
    if df.empty:
        raise RuntimeError("Feature v2 file is empty")
    if TARGET_COL not in df.columns:
        raise RuntimeError(f"Target column not found: {TARGET_COL}")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["symbol"] = df["symbol"].astype(str).str.upper()
    selected_set = {str(x).upper() for x in selected_symbols}
    if not selected_set:
        raise RuntimeError("Selected universe symbols are empty")
    before_rows = int(len(df))
    before_symbols = int(df["symbol"].nunique())
    df = df.loc[df["symbol"].isin(selected_set)].copy()
    after_rows = int(len(df))
    after_symbols = int(df["symbol"].nunique())
    print(f"[UNIVERSE] source={UNIVERSE_SNAPSHOT_FILE}")
    print(f"[UNIVERSE] snapshot_rows={universe_summary['snapshot_rows']} selected_rows={universe_summary['selected_rows']} selected_symbols_requested={len(selected_set)} feature_symbols_before={before_symbols} feature_symbols_after={after_symbols} rows_before={before_rows} rows_after={after_rows}")
    if df.empty:
        raise RuntimeError("Feature panel is empty after filtering by universe snapshot")
    missing_from_feature = sorted(selected_set - set(df["symbol"].astype(str).unique().tolist()))
    print(f"[UNIVERSE] missing_in_feature_panel={len(missing_from_feature)}")
    if missing_from_feature:
        print(f"[UNIVERSE][MISSING_SAMPLE] {missing_from_feature[:20]}")
    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)

    alpha_names = sorted(set(full_components_df["alpha"].astype(str).tolist() + ([requested_mr_alpha] if requested_mr_alpha else [])))
    if ALPHA_LIB_FILE.exists():
        alpha_df = pd.read_parquet(ALPHA_LIB_FILE)
        if not alpha_df.empty:
            alpha_df["date"] = pd.to_datetime(alpha_df["date"]).dt.normalize()
            alpha_df["symbol"] = alpha_df["symbol"].astype(str).str.upper()
            alpha_df = alpha_df.loc[alpha_df["symbol"].isin(selected_set)].copy()
            keep_alpha_cols = [c for c in alpha_names if c in alpha_df.columns]
            if keep_alpha_cols:
                df = df.merge(alpha_df[["date", "symbol"] + keep_alpha_cols].copy(), on=["date", "symbol"], how="left")

    available_before = sorted([a for a in alpha_names if a in df.columns])
    missing_before = sorted([a for a in alpha_names if a not in df.columns])
    if missing_before:
        print(f"[ALPHA] available_before_autobuild={len(available_before)} missing_before_autobuild={len(missing_before)}")
        print(f"[ALPHA][MISSING_BEFORE_AUTOBUILD] {missing_before[:20]}")

    if AUTO_BUILD_FS2_IF_MISSING:
        df, available_after, missing_after = _auto_build_requested_alpha_columns(df, alpha_names)
    else:
        available_after = sorted([a for a in alpha_names if a in df.columns])
        missing_after = sorted([a for a in alpha_names if a not in df.columns])

    print(f"[ALPHA] available_after_autobuild={len(available_after)} missing_after_autobuild={len(missing_after)}")
    if missing_after:
        print(f"[ALPHA][MISSING_AFTER_AUTOBUILD] {missing_after[:20]}")

    available_set = set(available_after)
    requested_df = full_components_df.head(FREEZE_COMPONENT_COUNT).copy().reset_index(drop=True)
    effective_components_df = full_components_df.loc[full_components_df["alpha"].astype(str).isin(available_set)].copy().head(FREEZE_COMPONENT_COUNT).reset_index(drop=True)
    if effective_components_df.empty:
        fallback_df = _discover_fallback_components(df, full_components_df)
        if fallback_df.empty:
            raise RuntimeError("No available freeze components found in interaction summary, and fallback raw-feature discovery found none")
        print(f"[FALLBACK] using raw feature components count={len(fallback_df)}")
        print(fallback_df[["candidate", "alpha", "regime", "kind", "last_2_fold_mean_sharpe", "test_mean_ic"]].to_string(index=False))
        effective_components_df = fallback_df.copy().reset_index(drop=True)
    else:
        dropped_requested = requested_df.loc[~requested_df["alpha"].astype(str).isin(available_set)].copy().reset_index(drop=True)
        if not dropped_requested.empty:
            print(f"[ALPHA][DROP_REQUESTED_COMPONENTS] dropped={len(dropped_requested)} because alpha column unavailable")
            print(dropped_requested[["candidate", "alpha", "regime", "kind"]].to_string(index=False))

    effective_mr_alpha: str | None = requested_mr_alpha if requested_mr_alpha in set(available_after) else None
    if requested_mr_alpha and effective_mr_alpha is None:
        print(f"[MR][DISABLED] requested alpha unavailable: {requested_mr_alpha}")

    summary = {
        "selected_symbols_requested": len(selected_set),
        "feature_symbols_after": after_symbols,
        "rows_after": after_rows,
        "missing_in_feature_panel": len(missing_from_feature),
        "alpha_available_after_autobuild": len(available_after),
        "alpha_missing_after_autobuild": len(missing_after),
        "requested_components": int(min(FREEZE_COMPONENT_COUNT, len(full_components_df))),
        "effective_components": int(len(effective_components_df)),
        "mr_alpha_requested": requested_mr_alpha,
        "mr_alpha_effective": effective_mr_alpha,
        "fallback_enabled": int(FALLBACK_ENABLE_RAW_FEATURE_COMPONENTS),
        "fallback_used": int(str(effective_components_df.iloc[0]["candidate"]).startswith("fallback::")) if len(effective_components_df) else 0,
    }
    return _build_explicit_regimes(df), effective_components_df.reset_index(drop=True), effective_mr_alpha, summary


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


def _sign_decision(train_ic_df: pd.DataFrame) -> Dict[str, float]:
    if train_ic_df.empty:
        return {"train_abs_mean_ic": float("nan"), "train_sign_locked": 1.0, "proxy_train_sharpe": float("nan")}
    mean_ic = float(train_ic_df["daily_ic"].mean())
    pos_rate = float((train_ic_df["daily_ic"] > 0.0).mean())
    sign_locked = 1.0 if mean_ic >= 0.0 else -1.0
    if abs(mean_ic) < SIGN_LOCK_IC_ABS and pos_rate != 0.5:
        sign_locked = 1.0 if pos_rate > 0.5 else -1.0
    std_ic = float(pd.to_numeric(train_ic_df["daily_ic"], errors="coerce").std(ddof=0))
    proxy_train_sharpe = float((mean_ic / (std_ic + EPS)) * np.sqrt(252.0))
    return {
        "train_abs_mean_ic": float(train_ic_df["daily_ic"].abs().mean()),
        "train_sign_locked": sign_locked,
        "proxy_train_sharpe": proxy_train_sharpe,
    }


def _build_mr_signal(df: pd.DataFrame, mr_alpha_col: str) -> pd.Series:
    if mr_alpha_col not in df.columns:
        raise RuntimeError(f"MR alpha column not found: {mr_alpha_col}")
    raw = -1.0 * _cs_zscore_by_date(df, _num(df[mr_alpha_col]))
    if FREEZE_MR_ONLY_TREND_LO:
        raw = raw * _num(df["regime_trend_lo"]).fillna(0.0)
    return raw


def _build_portfolio_scores(test_df: pd.DataFrame, component_signals: Dict[str, pd.Series], live_weights: Dict[str, float], cfg: FreezeConfig, mr_signal: pd.Series | None) -> pd.DataFrame:
    out = test_df[["date", "symbol", TARGET_COL, "regime_trend_hi", "regime_trend_lo"]].copy()
    score = pd.Series(0.0, index=out.index, dtype="float64")
    for candidate, weight in live_weights.items():
        sig = _num(component_signals[candidate]).fillna(0.0)
        out[f"signal__{candidate}"] = sig.values
        score = score + float(weight) * sig
    out["score_components"] = score.values
    if mr_signal is not None:
        mr = _num(mr_signal).fillna(0.0)
        out["signal__mr"] = mr.values
        score = (1.0 - cfg.mr_weight) * score + cfg.mr_weight * mr
    out["score"] = score.values
    return out


def _normalize_weights_with_caps(book: pd.DataFrame, gross_target: float, weight_cap: float) -> pd.DataFrame:
    out = book.copy()
    if out.empty:
        out["weight"] = pd.Series(dtype="float64")
        return out
    raw = _num(out["score"]).fillna(0.0)
    denom = float(raw.abs().sum())
    if denom <= EPS:
        out["weight"] = 0.0
        return out
    out["weight"] = gross_target * raw / denom
    out["weight"] = out["weight"].clip(lower=-weight_cap, upper=weight_cap)
    denom2 = float(out["weight"].abs().sum())
    if denom2 > EPS:
        out["weight"] = gross_target * _num(out["weight"]) / denom2
    out["weight"] = out["weight"].clip(lower=-weight_cap, upper=weight_cap)
    return out


def _build_portfolio(scored_df: pd.DataFrame, cfg: FreezeConfig) -> pd.DataFrame:
    pieces: List[pd.DataFrame] = []
    for dt, g in scored_df.groupby("date", sort=False):
        day = g[["date", "symbol", "score", TARGET_COL]].copy()
        inertia_input = day[["date", "symbol", "score"]].copy()
        inertia_out = apply_holding_inertia(inertia_input, enter_pct=ENTER_PCT, exit_pct=EXIT_PCT)
        day = day.merge(inertia_out[["date", "symbol", "rank_pct", "side"]], on=["date", "symbol"], how="left")
        day["active_score"] = _num(day["score"]).fillna(0.0) * _num(day["side"]).fillna(0.0)
        base_gross = float(GROSS_TARGET)
        reason = "base"
        trend_hi = float(scored_df.loc[scored_df["date"] == dt, "regime_trend_hi"].mean())
        trend_lo = float(scored_df.loc[scored_df["date"] == dt, "regime_trend_lo"].mean())
        if trend_lo > 0.5:
            dynamic_gross = base_gross * 0.75
            reason = "trend_lo"
        elif trend_hi > 0.5:
            dynamic_gross = base_gross * 1.00
            reason = "trend_hi"
        else:
            dynamic_gross = base_gross * 0.90
            reason = "mixed"
        book_input = day[["date", "symbol", "active_score", TARGET_COL, "rank_pct", "side"]].copy()
        book_input = book_input.rename(columns={"active_score": "score"})
        book = _normalize_weights_with_caps(book_input, gross_target=dynamic_gross, weight_cap=WEIGHT_CAP)
        book["dynamic_gross_target"] = dynamic_gross
        book["dynamic_gross_reason"] = reason
        pieces.append(book)
    out = pd.concat(pieces, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    out = cap_daily_turnover(out, weight_col="weight", max_daily_turnover=cfg.turnover_cap)
    if "trade_abs_after" not in out.columns:
        out["prev_weight"] = out.groupby("symbol", sort=False)["weight"].shift(1).fillna(0.0)
        out["trade_delta"] = _num(out["weight"]) - _num(out["prev_weight"])
        out["trade_abs_after"] = _num(out["trade_delta"]).abs()
        out["cap_hit"] = 0
    out["turnover"] = out.groupby("date", sort=False)["trade_abs_after"].transform("sum")
    out["pnl_gross"] = _num(out["weight"]) * _num(out[TARGET_COL])
    out["cost"] = (cfg.cost_bps / 10000.0) * _num(out["trade_abs_after"])
    out["pnl_net"] = _num(out["pnl_gross"]) - _num(out["cost"])
    return out


def _summarize_daily(port_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    daily = port_df.groupby("date", sort=False).agg(
        pnl_gross=("pnl_gross", "sum"),
        pnl_net=("pnl_net", "sum"),
        turnover=("turnover", "max"),
        gross_exposure=("weight", lambda s: float(_num(pd.Series(s)).abs().sum())),
        cap_hit=("cap_hit", "max"),
    ).reset_index()
    daily["equity"] = (1.0 + _num(daily["pnl_net"])).cumprod()
    running_max = _num(daily["equity"]).cummax()
    drawdown = (_num(daily["equity"]) / (running_max + EPS)) - 1.0
    mean_ret = _safe_nanmean(daily["pnl_net"]) if len(daily) else float("nan")
    std_ret = float(_num(daily["pnl_net"]).std(ddof=0)) if len(daily) else float("nan")
    sharpe = float((mean_ret / (std_ret + EPS)) * np.sqrt(252.0)) if len(daily) else float("nan")
    summary = {
        "days": int(len(daily)),
        "mean_daily_net": mean_ret,
        "std_daily_net": std_ret,
        "sharpe": sharpe,
        "cumret": float(_num(daily["equity"]).iloc[-1] - 1.0) if len(daily) else float("nan"),
        "max_drawdown": float(drawdown.min()) if len(drawdown) else float("nan"),
        "avg_turnover": float(_num(daily["turnover"]).mean()) if len(daily) else float("nan"),
        "avg_gross_exposure": float(_num(daily["gross_exposure"]).mean()) if len(daily) else float("nan"),
        "cap_hit_rate": float((_num(daily["cap_hit"]) > 0.0).mean()) if len(daily) else float("nan"),
    }
    return daily, summary


def _run_freeze_config(df: pd.DataFrame, components_df: pd.DataFrame, split: WFSplit, cfg: FreezeConfig, effective_mr_alpha: str | None) -> Dict[str, object]:
    train_df = df.loc[(df["date"] >= split.train_start) & (df["date"] <= split.train_end)].copy()
    test_df = df.loc[(df["date"] >= split.test_start) & (df["date"] <= split.test_end)].copy()
    if len(train_df) < MIN_TRAIN_ROWS:
        raise RuntimeError(f"Too few train rows for split {split.fold_id}: {len(train_df)}")
    if test_df.empty:
        raise RuntimeError(f"Empty test rows for split {split.fold_id}")

    info_rows: List[Dict[str, object]] = []
    test_signal_cols: Dict[str, pd.Series] = {}
    for _, rec in components_df.iterrows():
        candidate = str(rec["candidate"])
        alpha = str(rec["alpha"])
        regime = str(rec["regime"])
        kind = str(rec["kind"])
        train_signal = _cs_zscore_by_date(train_df, _build_candidate_signal(train_df, alpha, regime, kind))
        test_signal = _cs_zscore_by_date(test_df, _build_candidate_signal(test_df, alpha, regime, kind))
        train_scored = pd.DataFrame({"date": train_df["date"].values, "symbol": train_df["symbol"].values, TARGET_COL: _num(train_df[TARGET_COL]).values, "score": _num(train_signal).values})
        train_ic_df = _daily_ic_series(train_scored, "score", TARGET_COL, MIN_DAILY_IC_CS)
        sign_info = _sign_decision(train_ic_df)
        info_rows.append({"candidate": candidate, "alpha": alpha, "regime": regime, "kind": kind, **sign_info})
        test_signal_cols[candidate] = _num(test_signal) * float(sign_info["train_sign_locked"])

    train_info_df = pd.DataFrame(info_rows).head(FREEZE_COMPONENT_COUNT).copy()
    live_weights = _portfolio_weights(train_info_df)
    mr_signal = _build_mr_signal(test_df, effective_mr_alpha) if (FREEZE_MR_ENABLED and effective_mr_alpha) else None
    scored_df = _build_portfolio_scores(test_df, {cand: test_signal_cols[cand] for cand in live_weights.keys()}, live_weights, cfg, mr_signal)
    port_df = _build_portfolio(scored_df, cfg)
    daily_df, summary = _summarize_daily(port_df)
    current_date = pd.Timestamp(test_df["date"].max()).normalize()
    return {
        "component_info": train_info_df,
        "weights": live_weights,
        "scored_df": scored_df,
        "port_df": port_df,
        "daily_df": daily_df,
        "summary": summary,
        "current_date": current_date,
        "mr_enabled_effective": int(mr_signal is not None),
        "mr_alpha_effective": effective_mr_alpha,
    }


def main() -> int:
    _enable_line_buffering()
    print(f"[CFG] feature_v2_file={FEATURE_V2_FILE}")
    print(f"[CFG] universe_snapshot_file={UNIVERSE_SNAPSHOT_FILE}")
    print(f"[CFG] interaction_summary_file={INTERACTION_SUMMARY_FILE}")
    print(f"[CFG] alpha_lib_file={ALPHA_LIB_FILE}")
    print(f"[CFG] out_dir={OUT_DIR}")
    print(f"[CFG] frozen wf={FREEZE_TRAIN_DAYS}/{FREEZE_TEST_DAYS}/{FREEZE_STEP_DAYS} components={FREEZE_COMPONENT_COUNT} weighting={FREEZE_WEIGHTING_MODE}")
    print(f"[CFG] fallback_enable_raw_features={int(FALLBACK_ENABLE_RAW_FEATURE_COMPONENTS)} fallback_max_candidates={FALLBACK_MAX_CANDIDATES}")
    print(f"[CFG] configs={[cfg.name for cfg in FREEZE_CONFIGS]}")

    full_components_df = _load_interaction_summary()
    requested_components_df = full_components_df.head(FREEZE_COMPONENT_COUNT).copy().reset_index(drop=True)
    print("[COMPONENTS][REQUESTED_TOP]")
    print(requested_components_df[["candidate", "alpha", "regime", "kind", "oos_sharpe_mean", "last_2_fold_mean_sharpe", "test_mean_ic"]].to_string(index=False))

    df, components_df, effective_mr_alpha, load_summary = _load_feature_panel(full_components_df, FREEZE_MR_ALPHA)
    print(f"[COMPONENTS][EFFECTIVE] {len(components_df)}")
    print(components_df[["candidate", "alpha", "regime", "kind", "last_2_fold_mean_sharpe", "test_mean_ic"]].to_string(index=False))
    print(f"[MR] requested={FREEZE_MR_ALPHA} effective={effective_mr_alpha} enabled_requested={int(FREEZE_MR_ENABLED)} enabled_effective={int(FREEZE_MR_ENABLED and bool(effective_mr_alpha))}")

    last_date = pd.Timestamp(df["date"].max()).normalize()
    first_date = pd.Timestamp(df["date"].min()).normalize()
    print(f"[DATA] rows={len(df)} symbols={df['symbol'].nunique()} first_date={first_date.date()} last_date={last_date.date()}")

    splits = _build_walkforward_splits(df["date"], FREEZE_TRAIN_DAYS, FREEZE_TEST_DAYS, FREEZE_STEP_DAYS)
    current_split = splits[-1]
    print(f"[FREEZE] last_completed_fold fold_id={current_split.fold_id} train={current_split.train_start.date()}..{current_split.train_end.date()} test={current_split.test_start.date()}..{current_split.test_end.date()}")

    uniq_dates = pd.Index(sorted(pd.to_datetime(df["date"]).dt.normalize().unique()))
    last_idx = len(uniq_dates) - 1
    live_train_end_idx = last_idx - (WF_PURGE_DAYS + WF_EMBARGO_DAYS + 1)
    if live_train_end_idx < (FREEZE_TRAIN_DAYS - 1):
        raise RuntimeError("Not enough history for live-style freeze snapshot")
    live_train_start_idx = live_train_end_idx - FREEZE_TRAIN_DAYS + 1
    live_train_start = pd.Timestamp(uniq_dates[live_train_start_idx]).normalize()
    live_train_end = pd.Timestamp(uniq_dates[live_train_end_idx]).normalize()
    live_current_date = pd.Timestamp(uniq_dates[last_idx]).normalize()

    print(f"[FREEZE] live_snapshot train={live_train_start.date()}..{live_train_end.date()} current_date={live_current_date.date()}")

    train_df_live = df.loc[(df["date"] >= live_train_start) & (df["date"] <= live_train_end)].copy()
    current_df_live = df.loc[df["date"] == live_current_date].copy()
    if len(train_df_live) < MIN_TRAIN_ROWS:
        raise RuntimeError(f"Too few live-style train rows: {len(train_df_live)}")
    if current_df_live.empty:
        raise RuntimeError("No rows on live current date")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stale_days = int((pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() - last_date).days)
    summary_export: Dict[str, object] = {
        "last_data_date": str(last_date.date()),
        "days_stale_vs_now_utc": stale_days,
        "load_summary": load_summary,
        "last_completed_fold": {
            "fold_id": int(current_split.fold_id),
            "train_start": str(current_split.train_start.date()),
            "train_end": str(current_split.train_end.date()),
            "test_start": str(current_split.test_start.date()),
            "test_end": str(current_split.test_end.date()),
        },
        "live_snapshot": {
            "train_start": str(live_train_start.date()),
            "train_end": str(live_train_end.date()),
            "current_date": str(live_current_date.date()),
        },
        "frozen_base_config": {
            "train_days": FREEZE_TRAIN_DAYS,
            "test_days": FREEZE_TEST_DAYS,
            "step_days": FREEZE_STEP_DAYS,
            "component_count_requested": FREEZE_COMPONENT_COUNT,
            "component_count_effective": int(len(components_df)),
            "weighting_mode": FREEZE_WEIGHTING_MODE,
            "mr_enabled_requested": int(FREEZE_MR_ENABLED),
            "mr_alpha_requested": FREEZE_MR_ALPHA,
            "mr_alpha_effective": effective_mr_alpha,
            "mr_only_trend_lo": int(FREEZE_MR_ONLY_TREND_LO),
            "mr_gross_mult": FREEZE_MR_GROSS_MULT,
        },
        "configs": {},
    }

    for cfg in FREEZE_CONFIGS:
        replay_result = _run_freeze_config(df, components_df, current_split, cfg, effective_mr_alpha)

        info_rows: List[Dict[str, object]] = []
        current_signal_cols: Dict[str, pd.Series] = {}
        for _, rec in components_df.iterrows():
            candidate = str(rec["candidate"])
            alpha = str(rec["alpha"])
            regime = str(rec["regime"])
            kind = str(rec["kind"])
            train_signal = _cs_zscore_by_date(train_df_live, _build_candidate_signal(train_df_live, alpha, regime, kind))
            current_signal = _cs_zscore_by_date(current_df_live, _build_candidate_signal(current_df_live, alpha, regime, kind))
            train_scored = pd.DataFrame({"date": train_df_live["date"].values, "symbol": train_df_live["symbol"].values, TARGET_COL: _num(train_df_live[TARGET_COL]).values, "score": _num(train_signal).values})
            train_ic_df = _daily_ic_series(train_scored, "score", TARGET_COL, MIN_DAILY_IC_CS)
            sign_info = _sign_decision(train_ic_df)
            info_rows.append({"candidate": candidate, "alpha": alpha, "regime": regime, "kind": kind, **sign_info})
            current_signal_cols[candidate] = _num(current_signal) * float(sign_info["train_sign_locked"])

        live_comp_info = pd.DataFrame(info_rows).head(FREEZE_COMPONENT_COUNT).copy()
        live_weights = _portfolio_weights(live_comp_info)
        live_mr_signal = _build_mr_signal(current_df_live, effective_mr_alpha) if (FREEZE_MR_ENABLED and effective_mr_alpha) else None
        live_scored = _build_portfolio_scores(current_df_live, {cand: current_signal_cols[cand] for cand in live_weights.keys()}, live_weights, cfg, live_mr_signal)
        live_port = _build_portfolio(live_scored, cfg)
        live_book = live_port.loc[live_port["date"] == live_current_date].copy().sort_values(["weight", "symbol"], ascending=[False, True])
        live_book["abs_weight"] = _num(live_book["weight"]).abs()
        live_book = live_book.loc[live_book["abs_weight"] > 0.0].reset_index(drop=True)
        live_scores = live_scored.loc[live_scored["date"] == live_current_date].copy()
        live_scores["score_rank_pct"] = live_scores["score"].rank(method="average", pct=True)
        live_scores = live_scores.sort_values(["score", "symbol"], ascending=[False, True]).reset_index(drop=True)

        cfg_dir = OUT_DIR / cfg.name
        cfg_dir.mkdir(parents=True, exist_ok=True)
        replay_result["component_info"].to_csv(cfg_dir / "freeze_component_train_info.csv", index=False)
        replay_result["daily_df"].to_csv(cfg_dir / "freeze_daily_replay_last_fold.csv", index=False)
        replay_result["scored_df"].to_csv(cfg_dir / "freeze_scored_last_fold.csv", index=False)
        replay_result["port_df"].to_csv(cfg_dir / "freeze_portfolio_last_fold.csv", index=False)
        live_comp_info.to_csv(cfg_dir / "freeze_component_train_info_live_snapshot.csv", index=False)
        live_scored.to_csv(cfg_dir / "freeze_scored_live_snapshot.csv", index=False)
        live_port.to_csv(cfg_dir / "freeze_portfolio_live_snapshot.csv", index=False)
        live_scores.head(TOPK_EXPORT).to_csv(cfg_dir / "freeze_current_scores_top.csv", index=False)
        live_book.to_csv(cfg_dir / "freeze_current_book.csv", index=False)

        cfg_summary = {
            "cost_bps": cfg.cost_bps,
            "turnover_cap": cfg.turnover_cap,
            "mr_weight": cfg.mr_weight,
            "replay_current_date": str(replay_result["current_date"].date()),
            "live_current_date": str(live_current_date.date()),
            "weights": live_weights,
            "replay_evaluation": replay_result["summary"],
            "live_active_names": int(len(live_book)),
            "live_gross_exposure_current_day": float(_num(live_book["weight"]).abs().sum()) if len(live_book) else 0.0,
            "mr_enabled_effective": int(FREEZE_MR_ENABLED and bool(effective_mr_alpha)),
            "mr_alpha_effective": effective_mr_alpha,
        }
        (cfg_dir / "freeze_current_summary.json").write_text(json.dumps(cfg_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        summary_export["configs"][cfg.name] = cfg_summary

        print(f"[FREEZE][{cfg.name}] replay_current_date={replay_result['current_date'].date()} live_current_date={live_current_date.date()}")
        print(f"[FREEZE][{cfg.name}] live_active_names={len(live_book)} live_gross_exposure={cfg_summary['live_gross_exposure_current_day']:.4f}")
        print(f"[FREEZE][{cfg.name}] replay_sharpe_last_fold={replay_result['summary']['sharpe']:.4f} replay_maxdd_last_fold={replay_result['summary']['max_drawdown']:.4f} replay_avg_turnover_last_fold={replay_result['summary']['avg_turnover']:.4f}")
        if len(live_book):
            print(f"[FREEZE][{cfg.name}][LIVE_BOOK_TOP]")
            print(live_book[["symbol", "weight", "score", "side", "turnover", "dynamic_gross_target", "dynamic_gross_reason"]].head(min(TOPK_EXPORT, len(live_book))).to_string(index=False))
        else:
            print(f"[FREEZE][{cfg.name}][LIVE_BOOK_TOP] no active positions on live current date")

    (OUT_DIR / "freeze_all_configs_summary.json").write_text(json.dumps(summary_export, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ARTIFACT] {OUT_DIR / 'freeze_all_configs_summary.json'}")
    for cfg in FREEZE_CONFIGS:
        cfg_dir = OUT_DIR / cfg.name
        print(f"[ARTIFACT] {cfg_dir / 'freeze_current_summary.json'}")
        print(f"[ARTIFACT] {cfg_dir / 'freeze_component_train_info.csv'}")
        print(f"[ARTIFACT] {cfg_dir / 'freeze_daily_replay_last_fold.csv'}")
        print(f"[ARTIFACT] {cfg_dir / 'freeze_scored_last_fold.csv'}")
        print(f"[ARTIFACT] {cfg_dir / 'freeze_portfolio_last_fold.csv'}")
        print(f"[ARTIFACT] {cfg_dir / 'freeze_component_train_info_live_snapshot.csv'}")
        print(f"[ARTIFACT] {cfg_dir / 'freeze_scored_live_snapshot.csv'}")
        print(f"[ARTIFACT] {cfg_dir / 'freeze_portfolio_live_snapshot.csv'}")
        print(f"[ARTIFACT] {cfg_dir / 'freeze_current_scores_top.csv'}")
        print(f"[ARTIFACT] {cfg_dir / 'freeze_current_book.csv'}")
    print("[FINAL] freeze runner complete")
    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 1
    _safe_exit(rc)