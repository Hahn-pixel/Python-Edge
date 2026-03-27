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

try:
    from python_edge.portfolio.budget_allocation import attach_dynamic_side_budgets, apply_side_budgets
except Exception:
    def attach_dynamic_side_budgets(df: pd.DataFrame, min_long_budget: float = 0.35, max_long_budget: float = 0.75, input_lag_days: int = 1) -> pd.DataFrame:
        out = df.copy()
        out["long_budget"] = 0.50
        out["short_budget"] = 0.50
        out["budget_signal_lag_days"] = int(input_lag_days)
        return out

    def apply_side_budgets(df: pd.DataFrame, weight_col: str = "weight") -> pd.DataFrame:
        return df.copy()

try:
    from python_edge.portfolio.regime_allocation import attach_regime_multipliers
except Exception:
    def attach_regime_multipliers(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["regime_long_mult"] = 1.0
        out["regime_short_mult"] = 1.0
        out["regime_top_pct"] = 0.10
        out["market_regime"] = "neutral"
        return out

EPS = 1e-12
PAUSE_ON_EXIT = str(os.getenv("PAUSE_ON_EXIT", "auto")).strip().lower()
LIVE_ALPHA_SNAPSHOT_FILE = Path(os.getenv("LIVE_ALPHA_SNAPSHOT_FILE", "artifacts/live_alpha/live_alpha_snapshot.parquet"))
TARGET_COL = str(os.getenv("TARGET_COL", "target_fwd_ret_1d")).strip()
OUT_DIR = Path(os.getenv("FREEZE_OUT_DIR", "artifacts/freeze_runner"))

FREEZE_TRAIN_DAYS = int(os.getenv("FREEZE_TRAIN_DAYS", "120"))
FREEZE_TEST_DAYS = int(os.getenv("FREEZE_TEST_DAYS", "20"))
FREEZE_STEP_DAYS = int(os.getenv("FREEZE_STEP_DAYS", "20"))
FREEZE_COMPONENT_COUNT = int(os.getenv("FREEZE_COMPONENT_COUNT", "4"))
FREEZE_WEIGHTING_MODE = str(os.getenv("FREEZE_WEIGHTING_MODE", "ic")).strip().lower()

ENTER_PCT = float(os.getenv("ENTER_PCT", "0.10"))
EXIT_PCT = float(os.getenv("EXIT_PCT", "0.22"))
WEIGHT_CAP = float(os.getenv("WEIGHT_CAP", "0.05"))
GROSS_TARGET = float(os.getenv("GROSS_TARGET", "0.85"))
MIN_DAILY_IC_CS = int(os.getenv("MIN_DAILY_IC_CS", "20"))
MIN_TRAIN_ROWS = int(os.getenv("MIN_TRAIN_ROWS", "3000"))
WF_PURGE_DAYS = int(os.getenv("WF_PURGE_DAYS", "5"))
WF_EMBARGO_DAYS = int(os.getenv("WF_EMBARGO_DAYS", "5"))
TOPK_EXPORT = int(os.getenv("TOPK_EXPORT", "50"))

FEATURE_BUDGET_COLS = ["market_breadth", "intraday_rs", "volume_shock", "intraday_pressure"]
FEATURE_REGIME_COLS = ["market_breadth"]


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
    enter_pct: float
    exit_pct: float
    gross_target: float
    weight_cap: float
    weighting_mode: str
    component_count: int
    use_regime_overlay: bool
    use_dynamic_side_budgets: bool
    budget_input_lag_days: int
    min_long_budget: float
    max_long_budget: float


FREEZE_CONFIGS: List[FreezeConfig] = [
    FreezeConfig(
        name="optimal",
        cost_bps=4.0,
        turnover_cap=0.10,
        enter_pct=ENTER_PCT,
        exit_pct=EXIT_PCT,
        gross_target=GROSS_TARGET,
        weight_cap=WEIGHT_CAP,
        weighting_mode=FREEZE_WEIGHTING_MODE,
        component_count=FREEZE_COMPONENT_COUNT,
        use_regime_overlay=False,
        use_dynamic_side_budgets=False,
        budget_input_lag_days=1,
        min_long_budget=0.35,
        max_long_budget=0.75,
    ),
    FreezeConfig(
        name="aggressive",
        cost_bps=4.0,
        turnover_cap=0.12,
        enter_pct=0.08,
        exit_pct=0.18,
        gross_target=1.00,
        weight_cap=0.05,
        weighting_mode="ic",
        component_count=max(3, FREEZE_COMPONENT_COUNT),
        use_regime_overlay=True,
        use_dynamic_side_budgets=True,
        budget_input_lag_days=1,
        min_long_budget=0.45,
        max_long_budget=0.90,
    ),
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


def _sign_decision(train_ic_df: pd.DataFrame) -> Dict[str, float]:
    if train_ic_df.empty:
        return {"train_abs_mean_ic": float("nan"), "train_sign_locked": 1.0, "proxy_train_sharpe": float("nan")}
    mean_ic = float(train_ic_df["daily_ic"].mean())
    sign_locked = 1.0 if mean_ic >= 0.0 else -1.0
    std_ic = float(pd.to_numeric(train_ic_df["daily_ic"], errors="coerce").std(ddof=0))
    proxy_train_sharpe = float((mean_ic / (std_ic + EPS)) * np.sqrt(252.0))
    return {
        "train_abs_mean_ic": float(train_ic_df["daily_ic"].abs().mean()),
        "train_sign_locked": sign_locked,
        "proxy_train_sharpe": proxy_train_sharpe,
    }


def _load_live_panel() -> tuple[pd.DataFrame, list[str]]:
    _must_exist(LIVE_ALPHA_SNAPSHOT_FILE, "Live alpha snapshot")
    df = pd.read_parquet(LIVE_ALPHA_SNAPSHOT_FILE)
    if df.empty:
        raise RuntimeError("Live alpha snapshot is empty")
    if TARGET_COL not in df.columns:
        raise RuntimeError(f"Target column not found: {TARGET_COL}")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["symbol"] = df["symbol"].astype(str).str.upper()
    alpha_cols = [str(c) for c in df.columns if str(c).startswith("alpha_")]
    if not alpha_cols:
        raise RuntimeError("No alpha_ columns found in live alpha snapshot")
    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)
    return df, alpha_cols


def _select_components(df: pd.DataFrame, alpha_cols: Sequence[str], train_start: pd.Timestamp, train_end: pd.Timestamp) -> pd.DataFrame:
    train_df = df.loc[(df["date"] >= train_start) & (df["date"] <= train_end)].copy()
    if len(train_df) < MIN_TRAIN_ROWS:
        raise RuntimeError(f"Too few train rows for alpha selection: {len(train_df)}")
    rows: List[Dict[str, object]] = []
    for alpha_col in alpha_cols:
        signal = _cs_zscore_by_date(train_df, _num(train_df[alpha_col]))
        scored = pd.DataFrame({
            "date": train_df["date"].values,
            "symbol": train_df["symbol"].values,
            TARGET_COL: _num(train_df[TARGET_COL]).values,
            "score": _num(signal).values,
        })
        ic_df = _daily_ic_series(scored, "score", TARGET_COL, MIN_DAILY_IC_CS)
        if ic_df.empty:
            continue
        sign_info = _sign_decision(ic_df)
        rows.append({
            "candidate": alpha_col,
            "alpha": alpha_col,
            "regime": "none",
            "kind": "base",
            **sign_info,
        })
    comp = pd.DataFrame(rows)
    if comp.empty:
        raise RuntimeError("No valid factory alpha components survived train-window IC screening")
    comp = comp.sort_values(["train_abs_mean_ic", "proxy_train_sharpe", "alpha"], ascending=[False, False, True]).reset_index(drop=True)
    return comp.copy()


def _weight_from_train_info(train_info_df: pd.DataFrame, weighting_mode: str) -> pd.Series:
    if weighting_mode == "sharpe":
        w = _num(train_info_df["proxy_train_sharpe"]).abs().clip(lower=0.0)
    else:
        w = _num(train_info_df["train_abs_mean_ic"]).clip(lower=0.0)
    if float(w.sum()) <= EPS:
        w = pd.Series(np.ones(len(train_info_df), dtype="float64"), index=train_info_df.index)
    return w / float(w.sum())


def _portfolio_weights(train_info_df: pd.DataFrame, weighting_mode: str) -> Dict[str, float]:
    w = _weight_from_train_info(train_info_df, weighting_mode=weighting_mode)
    out: Dict[str, float] = {}
    for idx, row in train_info_df.reset_index(drop=True).iterrows():
        out[str(row["candidate"])] = float(w.iloc[idx])
    return out


def _build_portfolio_scores(test_df: pd.DataFrame, component_info: pd.DataFrame, weighting_mode: str) -> pd.DataFrame:
    passthrough_cols = [c for c in FEATURE_BUDGET_COLS if c in test_df.columns]
    base_cols = ["date", "symbol", TARGET_COL] + passthrough_cols
    out = test_df[base_cols].copy()
    score = pd.Series(0.0, index=out.index, dtype="float64")
    weights = _portfolio_weights(component_info, weighting_mode=weighting_mode)
    for _, row in component_info.iterrows():
        alpha = str(row["alpha"])
        candidate = str(row["candidate"])
        sign_locked = float(row["train_sign_locked"])
        sig = _cs_zscore_by_date(test_df, _num(test_df[alpha])).fillna(0.0) * sign_locked
        out[f"signal__{candidate}"] = sig.values
        score = score + float(weights[candidate]) * sig
    out["score"] = score.values
    return out


def _normalize_weights_with_caps(book: pd.DataFrame, gross_target: float, weight_cap: float) -> pd.DataFrame:
    out = book.copy()
    raw = _num(out["score"]).fillna(0.0)
    denom = float(raw.abs().sum())
    if denom <= EPS:
        out["weight"] = 0.0
        return out
    out["weight"] = gross_target * raw / denom
    out["weight"] = out["weight"].clip(lower=-weight_cap, upper=weight_cap)
    denom2 = float(_num(out["weight"]).abs().sum())
    if denom2 > EPS:
        out["weight"] = gross_target * _num(out["weight"]) / denom2
    out["weight"] = out["weight"].clip(lower=-weight_cap, upper=weight_cap)
    return out


def _apply_regime_overlay(day: pd.DataFrame) -> pd.DataFrame:
    if not all(col in day.columns for col in FEATURE_REGIME_COLS):
        out = day.copy()
        out["market_regime"] = "na"
        out["regime_long_mult"] = 1.0
        out["regime_short_mult"] = 1.0
        out["regime_top_pct"] = 0.10
        out["score_after_regime"] = _num(out["active_score"]).fillna(0.0)
        return out
    out = attach_regime_multipliers(day.copy())
    active = _num(out["active_score"]).fillna(0.0)
    long_mult = _num(out.get("regime_long_mult", pd.Series(1.0, index=out.index))).fillna(1.0)
    short_mult = _num(out.get("regime_short_mult", pd.Series(1.0, index=out.index))).fillna(1.0)
    out["score_after_regime"] = np.where(active > 0.0, active * long_mult, np.where(active < 0.0, active * short_mult, active))
    return out


def _apply_side_budget_overlay(book: pd.DataFrame, cfg: FreezeConfig) -> Tuple[pd.DataFrame, int]:
    if not cfg.use_dynamic_side_budgets:
        out = book.copy()
        out["long_budget"] = 0.50
        out["short_budget"] = 0.50
        out["budget_signal_lag_days"] = cfg.budget_input_lag_days
        return out, 0
    if not all(col in book.columns for col in FEATURE_BUDGET_COLS):
        out = book.copy()
        out["long_budget"] = 0.50
        out["short_budget"] = 0.50
        out["budget_signal_lag_days"] = cfg.budget_input_lag_days
        return out, 0
    out = attach_dynamic_side_budgets(
        book.copy(),
        min_long_budget=cfg.min_long_budget,
        max_long_budget=cfg.max_long_budget,
        input_lag_days=cfg.budget_input_lag_days,
    )
    out = apply_side_budgets(out, weight_col="weight")
    return out, 1


def _build_portfolio(scored_df: pd.DataFrame, cfg: FreezeConfig) -> pd.DataFrame:
    pieces: List[pd.DataFrame] = []
    regime_days_used = 0
    budget_days_used = 0
    for dt, g in scored_df.groupby("date", sort=False):
        day = g.copy()
        day = day[[c for c in day.columns if c in (["date", "symbol", "score", TARGET_COL] + FEATURE_BUDGET_COLS)]].copy()
        inertia_input = day[["date", "symbol", "score"]].copy()
        inertia_out = apply_holding_inertia(inertia_input, enter_pct=cfg.enter_pct, exit_pct=cfg.exit_pct)
        day = day.merge(inertia_out[["date", "symbol", "rank_pct", "side"]], on=["date", "symbol"], how="left")
        day["active_score"] = _num(day["score"]).fillna(0.0) * _num(day["side"]).fillna(0.0)

        if cfg.use_regime_overlay:
            day = _apply_regime_overlay(day)
            regime_days_used += 1 if "market_regime" in day.columns and (day["market_regime"] != "na").any() else 0
            score_for_book = _num(day["score_after_regime"]).fillna(0.0)
        else:
            day["market_regime"] = "disabled"
            day["regime_long_mult"] = 1.0
            day["regime_short_mult"] = 1.0
            day["regime_top_pct"] = 0.10
            day["score_after_regime"] = _num(day["active_score"]).fillna(0.0)
            score_for_book = _num(day["active_score"]).fillna(0.0)

        book_input = day.copy()
        book_input["score"] = score_for_book
        book = _normalize_weights_with_caps(book_input, gross_target=cfg.gross_target, weight_cap=cfg.weight_cap)
        book, budget_used = _apply_side_budget_overlay(book, cfg)
        budget_days_used += int(budget_used)
        book["regime_overlay_used"] = int(cfg.use_regime_overlay)
        book["side_budget_overlay_used"] = int(budget_used)
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
    out["config_turnover_cap"] = cfg.turnover_cap
    out["config_gross_target"] = cfg.gross_target
    out["config_weight_cap"] = cfg.weight_cap
    out["config_enter_pct"] = cfg.enter_pct
    out["config_exit_pct"] = cfg.exit_pct
    out["config_weighting_mode"] = cfg.weighting_mode
    out["diag_regime_days_used"] = regime_days_used
    out["diag_budget_days_used"] = budget_days_used
    return out


def _summarize_daily(port_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    daily = port_df.groupby("date", sort=False).agg(
        pnl_gross=("pnl_gross", "sum"),
        pnl_net=("pnl_net", "sum"),
        turnover=("turnover", "max"),
        gross_exposure=("weight", lambda s: float(_num(pd.Series(s)).abs().sum())),
        cap_hit=("cap_hit", "max"),
        regime_overlay_used=("regime_overlay_used", "max"),
        side_budget_overlay_used=("side_budget_overlay_used", "max"),
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
        "regime_overlay_used_rate": float((_num(daily["regime_overlay_used"]) > 0.0).mean()) if len(daily) else float("nan"),
        "side_budget_overlay_used_rate": float((_num(daily["side_budget_overlay_used"]) > 0.0).mean()) if len(daily) else float("nan"),
    }
    return daily, summary


def _select_components_for_config(component_info_all: pd.DataFrame, cfg: FreezeConfig) -> pd.DataFrame:
    comp = component_info_all.copy()
    sort_cols = ["train_abs_mean_ic", "proxy_train_sharpe", "alpha"]
    ascending = [False, False, True] if cfg.weighting_mode != "sharpe" else [False, False, True]
    comp = comp.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    return comp.head(cfg.component_count).copy()


def main() -> int:
    _enable_line_buffering()
    print(f"[CFG] live_alpha_snapshot_file={LIVE_ALPHA_SNAPSHOT_FILE}")
    print(f"[CFG] out_dir={OUT_DIR}")
    print(f"[CFG] frozen wf={FREEZE_TRAIN_DAYS}/{FREEZE_TEST_DAYS}/{FREEZE_STEP_DAYS} components={FREEZE_COMPONENT_COUNT} weighting={FREEZE_WEIGHTING_MODE}")

    df, alpha_cols = _load_live_panel()
    print(f"[DATA] rows={len(df)} symbols={df['symbol'].nunique()} alpha_cols={len(alpha_cols)} first_date={df['date'].min().date()} last_date={df['date'].max().date()}")

    splits = _build_walkforward_splits(df["date"], FREEZE_TRAIN_DAYS, FREEZE_TEST_DAYS, FREEZE_STEP_DAYS)
    current_split = splits[-1]
    print(f"[FREEZE] last_completed_fold fold_id={current_split.fold_id} train={current_split.train_start.date()}..{current_split.train_end.date()} test={current_split.test_start.date()}..{current_split.test_end.date()}")

    component_info_all = _select_components(df, alpha_cols, current_split.train_start, current_split.train_end)
    print("[COMPONENTS][ALL_CANDIDATES_TOP]")
    print(component_info_all[["alpha", "train_abs_mean_ic", "proxy_train_sharpe", "train_sign_locked"]].head(max(FREEZE_COMPONENT_COUNT, 10)).to_string(index=False))

    replay_df = df.loc[(df["date"] >= current_split.test_start) & (df["date"] <= current_split.test_end)].copy()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_export: Dict[str, object] = {
        "last_data_date": str(pd.Timestamp(df["date"].max()).date()),
        "last_completed_fold": {
            "fold_id": int(current_split.fold_id),
            "train_start": str(current_split.train_start.date()),
            "train_end": str(current_split.train_end.date()),
            "test_start": str(current_split.test_start.date()),
            "test_end": str(current_split.test_end.date()),
        },
        "component_selection_all": component_info_all.to_dict(orient="records"),
        "configs": {},
    }

    live_current_date = pd.Timestamp(df["date"].max()).normalize()
    live_df = df.loc[df["date"] == live_current_date].copy()
    if live_df.empty:
        raise RuntimeError("No rows on live current date")

    for cfg in FREEZE_CONFIGS:
        component_info = _select_components_for_config(component_info_all, cfg)
        print(f"[COMPONENTS][{cfg.name}] weighting_mode={cfg.weighting_mode} component_count={cfg.component_count}")
        print(component_info[["alpha", "train_abs_mean_ic", "proxy_train_sharpe", "train_sign_locked"]].to_string(index=False))

        replay_scored = _build_portfolio_scores(replay_df, component_info, weighting_mode=cfg.weighting_mode)
        replay_port = _build_portfolio(replay_scored, cfg)
        replay_daily, replay_summary = _summarize_daily(replay_port)

        live_scored = _build_portfolio_scores(live_df, component_info, weighting_mode=cfg.weighting_mode)
        live_port = _build_portfolio(live_scored, cfg)
        live_book = live_port.loc[live_port["date"] == live_current_date].copy().sort_values(["weight", "symbol"], ascending=[False, True])
        live_book["abs_weight"] = _num(live_book["weight"]).abs()
        live_book = live_book.loc[live_book["abs_weight"] > 0.0].reset_index(drop=True)
        live_scores = live_scored.copy()
        live_scores["score_rank_pct"] = live_scores["score"].rank(method="average", pct=True)
        live_scores = live_scores.sort_values(["score", "symbol"], ascending=[False, True]).reset_index(drop=True)

        cfg_dir = OUT_DIR / cfg.name
        cfg_dir.mkdir(parents=True, exist_ok=True)
        component_info.to_csv(cfg_dir / "freeze_component_train_info.csv", index=False)
        replay_daily.to_csv(cfg_dir / "freeze_daily_replay_last_fold.csv", index=False)
        replay_scored.to_csv(cfg_dir / "freeze_scored_last_fold.csv", index=False)
        replay_port.to_csv(cfg_dir / "freeze_portfolio_last_fold.csv", index=False)
        component_info.to_csv(cfg_dir / "freeze_component_train_info_live_snapshot.csv", index=False)
        live_scored.to_csv(cfg_dir / "freeze_scored_live_snapshot.csv", index=False)
        live_port.to_csv(cfg_dir / "freeze_portfolio_live_snapshot.csv", index=False)
        live_scores.head(TOPK_EXPORT).to_csv(cfg_dir / "freeze_current_scores_top.csv", index=False)
        live_book.to_csv(cfg_dir / "freeze_current_book.csv", index=False)

        weight_series = _weight_from_train_info(component_info, weighting_mode=cfg.weighting_mode).reset_index(drop=True)
        cfg_summary = {
            "cost_bps": cfg.cost_bps,
            "turnover_cap": cfg.turnover_cap,
            "enter_pct": cfg.enter_pct,
            "exit_pct": cfg.exit_pct,
            "gross_target": cfg.gross_target,
            "weight_cap": cfg.weight_cap,
            "weighting_mode": cfg.weighting_mode,
            "component_count": cfg.component_count,
            "use_regime_overlay": int(cfg.use_regime_overlay),
            "use_dynamic_side_budgets": int(cfg.use_dynamic_side_budgets),
            "budget_input_lag_days": cfg.budget_input_lag_days,
            "min_long_budget": cfg.min_long_budget,
            "max_long_budget": cfg.max_long_budget,
            "replay_current_date": str(pd.Timestamp(replay_df["date"].max()).date()),
            "live_current_date": str(live_current_date.date()),
            "weights": {str(component_info.iloc[i]["alpha"]): float(weight_series.iloc[i]) for i in range(len(component_info))},
            "replay_evaluation": replay_summary,
            "live_active_names": int(len(live_book)),
            "live_gross_exposure_current_day": float(_num(live_book["weight"]).abs().sum()) if len(live_book) else 0.0,
            "mr_enabled_effective": 0,
            "mr_alpha_effective": None,
        }
        (cfg_dir / "freeze_current_summary.json").write_text(json.dumps(cfg_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        summary_export["configs"][cfg.name] = cfg_summary

        print(
            f"[FREEZE][{cfg.name}] live_active_names={len(live_book)} live_gross_exposure={cfg_summary['live_gross_exposure_current_day']:.4f} "
            f"replay_sharpe_last_fold={replay_summary['sharpe']:.4f} avg_turnover={replay_summary['avg_turnover']:.4f} avg_gross={replay_summary['avg_gross_exposure']:.4f}"
        )
        print(
            f"[FREEZE][{cfg.name}][DIAG] use_regime_overlay={int(cfg.use_regime_overlay)} use_dynamic_side_budgets={int(cfg.use_dynamic_side_budgets)} "
            f"enter_pct={cfg.enter_pct:.4f} exit_pct={cfg.exit_pct:.4f} gross_target={cfg.gross_target:.4f} weight_cap={cfg.weight_cap:.4f} turnover_cap={cfg.turnover_cap:.4f}"
        )
        if len(live_book):
            print(f"[FREEZE][{cfg.name}][LIVE_BOOK_TOP]")
            show_cols = [c for c in ["symbol", "weight", "score", "side", "market_regime", "long_budget", "short_budget", "turnover"] if c in live_book.columns]
            print(live_book[show_cols].head(min(TOPK_EXPORT, len(live_book))).to_string(index=False))
        else:
            print(f"[FREEZE][{cfg.name}][LIVE_BOOK_TOP] no active positions on live current date")

    (OUT_DIR / "freeze_all_configs_summary.json").write_text(json.dumps(summary_export, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ARTIFACT] {OUT_DIR / 'freeze_all_configs_summary.json'}")
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
