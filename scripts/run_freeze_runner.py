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


def _pick_regime_source(df: pd.DataFrame, candidates: Sequence[str]) -> Tuple[pd.Series, str]:
    for c in candidates:
        if c in df.columns:
            return _num(df[c]), c
    return pd.Series(np.nan, index=df.index, dtype="float64"), "none"


def _build_explicit_regimes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    _, trend_name = _pick_regime_source(out, ["fs2_intraday_pressure_rs_mix", "z_fs2_intraday_pressure_rs_mix", "fs2_intraday_rs_proxy", "z_fs2_intraday_rs_proxy"])
    day_level = out.groupby("date", sort=False).agg(trend_proxy=(trend_name if trend_name != "none" else TARGET_COL, lambda s: float(_safe_nanmean(_num(s))))).reset_index()
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
        raise RuntimeError(f"Missing alpha columns after auto-build: {still_missing[:10]}")
    return out


def _load_component_universe() -> pd.DataFrame:
    _must_exist(INTERACTION_SUMMARY_FILE, "Interaction summary")
    df = pd.read_csv(INTERACTION_SUMMARY_FILE)
    if df.empty:
        raise RuntimeError("Interaction summary is empty")
    df = df.sort_values(["last_2_fold_mean_sharpe", "oos_sharpe_mean", "test_mean_ic", "candidate"], ascending=[False, False, False, True]).reset_index(drop=True)
    return df.head(FREEZE_COMPONENT_COUNT).copy()


def _load_feature_panel(components_df: pd.DataFrame) -> pd.DataFrame:
    _must_exist(FEATURE_V2_FILE, "Feature v2 file")
    df = pd.read_parquet(FEATURE_V2_FILE)
    if df.empty:
        raise RuntimeError("Feature v2 file is empty")
    if TARGET_COL not in df.columns:
        raise RuntimeError(f"Target column not found: {TARGET_COL}")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)
    alpha_names = sorted(set(components_df["alpha"].astype(str).tolist() + [FREEZE_MR_ALPHA]))
    if ALPHA_LIB_FILE.exists():
        alpha_df = pd.read_parquet(ALPHA_LIB_FILE)
        if not alpha_df.empty:
            alpha_df["date"] = pd.to_datetime(alpha_df["date"]).dt.normalize()
            keep_alpha_cols = [c for c in alpha_names if c in alpha_df.columns]
            if keep_alpha_cols:
                df = df.merge(alpha_df[["date", "symbol"] + keep_alpha_cols].copy(), on=["date", "symbol"], how="left")
    if any(a not in df.columns for a in alpha_names):
        if not AUTO_BUILD_FS2_IF_MISSING:
            missing = [a for a in alpha_names if a not in df.columns]
            raise RuntimeError(f"Missing alpha columns and auto-build disabled: {missing[:10]}")
        df = _auto_build_requested_alpha_columns(df, alpha_names)
    return _build_explicit_regimes(df)


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


def _portfolio_weights(comp_train_info: pd.DataFrame) -> Dict[str, float]:
    if FREEZE_WEIGHTING_MODE == "equal":
        vals = {row["candidate"]: 1.0 for _, row in comp_train_info.iterrows()}
    elif FREEZE_WEIGHTING_MODE == "ic":
        vals = {row["candidate"]: max(float(pd.to_numeric(row.get("train_abs_mean_ic", np.nan), errors="coerce") or 0.0), EPS) for _, row in comp_train_info.iterrows()}
    elif FREEZE_WEIGHTING_MODE == "sharpe":
        vals = {row["candidate"]: max(float(pd.to_numeric(row.get("proxy_train_sharpe", np.nan), errors="coerce") or 0.0), EPS) for _, row in comp_train_info.iterrows()}
    else:
        raise RuntimeError(f"Unsupported weighting mode: {FREEZE_WEIGHTING_MODE}")
    total = float(sum(vals.values()))
    return {k: float(v / total) for k, v in vals.items()} if total > 0.0 else {k: 1.0 / len(vals) for k in vals}


def _build_mr_signal(df: pd.DataFrame, alpha_col: str) -> pd.Series:
    return -1.0 * _cs_zscore_by_date(df, _num(df[alpha_col]))


def _build_portfolio_scores(test_df: pd.DataFrame, signal_cols: Dict[str, pd.Series], weights: Dict[str, float], cfg: FreezeConfig, mr_signal: pd.Series | None) -> pd.DataFrame:
    out = test_df[["date", "symbol", TARGET_COL, "regime_trend_hi", "regime_trend_lo"]].copy()
    trend_score = pd.Series(0.0, index=test_df.index, dtype="float64")
    for cand, w in weights.items():
        trend_score = trend_score + (_num(signal_cols[cand]) * float(w))
    out["trend_score_raw"] = _num(trend_score) * _num(test_df["regime_trend_hi"]).fillna(0.0)
    out["mr_score_raw"] = 0.0
    if mr_signal is not None:
        regime_lo = _num(test_df["regime_trend_lo"]).fillna(0.0) if FREEZE_MR_ONLY_TREND_LO else 1.0
        out["mr_score_raw"] = _num(mr_signal) * regime_lo * float(cfg.mr_weight)
    out["score_raw"] = _num(out["trend_score_raw"]) + _num(out["mr_score_raw"])
    out["score"] = _cs_zscore_by_date(out, _num(out["score_raw"]))
    return out


def _build_portfolio(scored_df: pd.DataFrame, cfg: FreezeConfig) -> pd.DataFrame:
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
        elif trend_lo >= 0.5 and FREEZE_MR_ENABLED:
            gross_mult = float(FREEZE_MR_GROSS_MULT)
            gross_reason = "trend_lo_mr"
        else:
            gross_mult = 0.50
            gross_reason = "neutral"
        dynamic_gross_target = float(GROSS_TARGET) * gross_mult
        gg["dynamic_gross_target"] = dynamic_gross_target
        gg["dynamic_gross_reason"] = gross_reason
        gross = float(gg["weight"].abs().sum())
        if gross > EPS:
            gg["weight"] = gg["weight"] * (dynamic_gross_target / gross)
        pieces.append(gg)
    out = pd.concat(pieces, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    out = cap_daily_turnover(out, weight_col="weight", max_daily_turnover=cfg.turnover_cap)
    if "trade_abs_after" in out.columns:
        out["turnover"] = _num(out["trade_abs_after"])
    else:
        out["prev_weight"] = out.groupby("symbol", sort=False)["weight"].shift(1).fillna(0.0)
        out["turnover"] = (_num(out["weight"]) - _num(out["prev_weight"])).abs()
    out["gross_ret"] = _num(out["weight"]) * _num(out[TARGET_COL])
    out["cost_ret"] = _num(out["turnover"]) * (cfg.cost_bps / 10000.0)
    out["net_ret"] = _num(out["gross_ret"]) - _num(out["cost_ret"])
    return out


def _evaluate_daily(port_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
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
        "days": int(len(daily)),
        "mean_daily": mean_daily,
        "std_daily": std_daily,
        "sharpe": sharpe,
        "cum_ret": float(daily["equity"].iloc[-1] - 1.0) if len(daily) else float("nan"),
        "max_drawdown": float(daily["drawdown"].min()) if len(daily) else float("nan"),
        "avg_turnover": _safe_nanmean(daily["turnover"].to_numpy(dtype="float64")) if len(daily) else float("nan"),
        "avg_gross_exposure": _safe_nanmean(daily["gross_exposure"].to_numpy(dtype="float64")) if len(daily) else float("nan"),
    }


def _run_freeze_config(df: pd.DataFrame, components_df: pd.DataFrame, current_split: WFSplit, cfg: FreezeConfig) -> Dict[str, object]:
    train_df = df.loc[(df["date"] >= current_split.train_start) & (df["date"] <= current_split.train_end)].copy()
    test_df = df.loc[(df["date"] >= current_split.test_start) & (df["date"] <= current_split.test_end)].copy()
    if len(train_df) < MIN_TRAIN_ROWS:
        raise RuntimeError(f"Too few train rows: {len(train_df)}")
    if test_df.empty:
        raise RuntimeError("Current test frame is empty")

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
        info_rows.append({
            "candidate": candidate,
            "alpha": alpha,
            "regime": regime,
            "kind": kind,
            **sign_info,
        })
        test_signal_cols[candidate] = _num(test_signal) * float(sign_info["train_sign_locked"])

    comp_info = pd.DataFrame(info_rows).head(FREEZE_COMPONENT_COUNT).copy()
    weights = _portfolio_weights(comp_info)
    mr_signal = _build_mr_signal(test_df, FREEZE_MR_ALPHA) if FREEZE_MR_ENABLED else None
    scored = _build_portfolio_scores(test_df, {cand: test_signal_cols[cand] for cand in weights.keys()}, weights, cfg, mr_signal)
    port_df = _build_portfolio(scored, cfg)
    daily_df, summary = _evaluate_daily(port_df)

    current_date = pd.Timestamp(test_df["date"].max()).normalize()
    current_book = port_df.loc[port_df["date"] == current_date].copy().sort_values(["weight", "symbol"], ascending=[False, True])
    current_book["abs_weight"] = _num(current_book["weight"]).abs()
    current_book = current_book.loc[current_book["abs_weight"] > 0.0].reset_index(drop=True)

    current_scores = scored.loc[scored["date"] == current_date].copy()
    current_scores["score_rank_pct"] = current_scores["score"].rank(method="average", pct=True)
    current_scores = current_scores.sort_values(["score", "symbol"], ascending=[False, True]).reset_index(drop=True)

    return {
        "cfg": cfg,
        "component_info": comp_info,
        "weights": weights,
        "summary": summary,
        "daily_df": daily_df,
        "scored_df": scored,
        "port_df": port_df,
        "current_book": current_book,
        "current_scores": current_scores,
        "current_date": current_date,
    }


def main() -> int:
    _enable_line_buffering()
    print(f"[CFG] feature_v2_file={FEATURE_V2_FILE}")
    print(f"[CFG] interaction_summary_file={INTERACTION_SUMMARY_FILE}")
    print(f"[CFG] alpha_lib_file={ALPHA_LIB_FILE}")
    print(f"[CFG] out_dir={OUT_DIR}")
    print(f"[CFG] frozen wf={FREEZE_TRAIN_DAYS}/{FREEZE_TEST_DAYS}/{FREEZE_STEP_DAYS} components={FREEZE_COMPONENT_COUNT} weighting={FREEZE_WEIGHTING_MODE}")
    print(f"[CFG] configs={[cfg.name for cfg in FREEZE_CONFIGS]}")

    components_df = _load_component_universe()
    print("[COMPONENTS][FROZEN]")
    print(components_df[["candidate", "alpha", "regime", "kind", "oos_sharpe_mean", "last_2_fold_mean_sharpe", "test_mean_ic"]].to_string(index=False))

    df = _load_feature_panel(components_df)
    last_date = pd.Timestamp(df["date"].max()).normalize()
    first_date = pd.Timestamp(df["date"].min()).normalize()
    print(f"[DATA] rows={len(df)} symbols={df['symbol'].nunique()} first_date={first_date.date()} last_date={last_date.date()}")

    splits = _build_walkforward_splits(df["date"], FREEZE_TRAIN_DAYS, FREEZE_TEST_DAYS, FREEZE_STEP_DAYS)
    current_split = splits[-1]
    print(f"[FREEZE] last_completed_fold fold_id={current_split.fold_id} train={current_split.train_start.date()}..{current_split.train_end.date()} test={current_split.test_start.date()}..{current_split.test_end.date()}")

    # Final patch: build live-style snapshot on the LAST AVAILABLE DATE, not on last completed fold end.
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
        "days_stale_vs_2026_03_21": stale_days,
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
            "component_count": FREEZE_COMPONENT_COUNT,
            "weighting_mode": FREEZE_WEIGHTING_MODE,
            "mr_enabled": int(FREEZE_MR_ENABLED),
            "mr_alpha": FREEZE_MR_ALPHA,
            "mr_only_trend_lo": int(FREEZE_MR_ONLY_TREND_LO),
            "mr_gross_mult": FREEZE_MR_GROSS_MULT,
        },
        "configs": {},
    }

    for cfg in FREEZE_CONFIGS:
        # Keep replay on last completed fold for diagnostics.
        replay_result = _run_freeze_config(df, components_df, current_split, cfg)

        # Build live-style current snapshot on last available date.
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
        live_mr_signal = _build_mr_signal(current_df_live, FREEZE_MR_ALPHA) if FREEZE_MR_ENABLED else None
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
