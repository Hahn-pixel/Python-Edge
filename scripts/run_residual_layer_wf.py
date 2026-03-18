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
        out = df.copy()
        out = out.sort_values(["date", "symbol"]).reset_index(drop=True)
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

try:
    from python_edge.portfolio.budget_allocation import attach_dynamic_side_budgets, apply_side_budgets
    HAS_BUDGET_ALLOCATION = True
except Exception:
    HAS_BUDGET_ALLOCATION = False

    def attach_dynamic_side_budgets(df: pd.DataFrame, min_long_budget: float = 0.35, max_long_budget: float = 0.75, input_lag_days: int = 1) -> pd.DataFrame:
        out = df.copy()
        out["long_budget"] = 0.50
        out["short_budget"] = 0.50
        out["budget_signal_lag_days"] = int(input_lag_days)
        out["dbg_budget_allocation_fallback"] = 1
        return out

    def apply_side_budgets(df: pd.DataFrame, weight_col: str = "weight") -> pd.DataFrame:
        return df.copy()

try:
    from python_edge.portfolio.regime_allocation import attach_regime_multipliers
    HAS_REGIME_ALLOCATION = True
except Exception:
    HAS_REGIME_ALLOCATION = False

    def attach_regime_multipliers(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["market_regime"] = "neutral"
        out["regime_long_mult"] = 1.0
        out["regime_short_mult"] = 1.0
        out["regime_top_pct"] = 0.10
        out["dbg_regime_allocation_fallback"] = 1
        return out

EPS = 1e-12

ALPHA_LIB_FILE = Path(os.getenv("ALPHA_LIB_FILE", "data/alpha_library_v2/alpha_library_v2.parquet"))
ALIVE_SUMMARY_CSV = Path(os.getenv("ALIVE_SUMMARY_CSV", "artifacts/single_alpha_audit_wf/single_alpha_audit__alive.csv"))
TARGET_COL = str(os.getenv("TARGET_COL", "target_fwd_ret_1d")).strip()
OUT_DIR = Path(os.getenv("RESIDUAL_WF_OUT_DIR", "artifacts/residual_layer_wf"))
PAUSE_ON_EXIT = str(os.getenv("PAUSE_ON_EXIT", "auto")).strip().lower()

TRAIN_DAYS = int(os.getenv("WF_TRAIN_DAYS", "252"))
TEST_DAYS = int(os.getenv("WF_TEST_DAYS", "63"))
STEP_DAYS = int(os.getenv("WF_STEP_DAYS", "63"))
PURGE_DAYS = int(os.getenv("WF_PURGE_DAYS", "5"))
EMBARGO_DAYS = int(os.getenv("WF_EMBARGO_DAYS", "5"))
MIN_TRAIN_ROWS = int(os.getenv("MIN_TRAIN_ROWS", "3000"))
MIN_TEST_ROWS = int(os.getenv("MIN_TEST_ROWS", "300"))
MIN_DAILY_IC_CS = int(os.getenv("MIN_DAILY_IC_CS", "20"))
MIN_RESIDUAL_TRAIN_ROWS = int(os.getenv("MIN_RESIDUAL_TRAIN_ROWS", "800"))

ENTER_PCT = float(os.getenv("ENTER_PCT", "0.10"))
EXIT_PCT = float(os.getenv("EXIT_PCT", "0.22"))
WEIGHT_CAP = float(os.getenv("WEIGHT_CAP", "0.05"))
GROSS_TARGET = float(os.getenv("GROSS_TARGET", "0.85"))
MAX_DAILY_TURNOVER = float(os.getenv("MAX_DAILY_TURNOVER", "0.20"))
COST_BPS = float(os.getenv("COST_BPS", "8.0"))

RESIDUAL_METHOD = str(os.getenv("RESIDUAL_METHOD", "ols")).strip().lower()
ALIVE_ALPHA_LIST = [x.strip() for x in str(os.getenv("ALIVE_ALPHA_LIST", "")).split(",") if x.strip()]
ALIVE_ALPHA_LIMIT = int(os.getenv("ALIVE_ALPHA_LIMIT", "3"))
CORE_ALPHA = str(os.getenv("CORE_ALPHA", "")).strip()
SIGN_LOCK_IC_ABS = float(os.getenv("SIGN_LOCK_IC_ABS", "0.0100"))
SIGN_LOCK_POS_RATE = float(os.getenv("SIGN_LOCK_POS_RATE", "0.55"))
TAIL_TOP_PCT = float(os.getenv("TAIL_TOP_PCT", "0.10"))
ENABLE_DYNAMIC_BUDGETS = str(os.getenv("ENABLE_DYNAMIC_BUDGETS", "0")).strip().lower() not in {"0", "false", "no", "off"}
ENABLE_REGIME_MULTIPLIERS = str(os.getenv("ENABLE_REGIME_MULTIPLIERS", "0")).strip().lower() not in {"0", "false", "no", "off"}
TOPK_PRINT = int(os.getenv("TOPK_PRINT", "20"))


@dataclass(frozen=True)
class WFSplit:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass(frozen=True)
class ShellConfig:
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


def _cs_zscore_df(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col] = out.groupby("date", sort=False)[col].transform(_robust_zscore_series)
    return out


def _fit_linear_beta(y: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    xtx = x.T @ x
    ridge = 1e-8 * np.eye(xtx.shape[0], dtype="float64")
    beta = np.linalg.solve(xtx + ridge, x.T @ y)
    yhat = x @ beta
    resid = y - yhat
    info = {
        "fit_r2": float(1.0 - (np.var(resid) / (np.var(y) + EPS))),
        "beta_l2": float(np.sqrt(np.sum(beta ** 2))),
    }
    return beta, info


def _residualize_against_prior(base_df: pd.DataFrame, alpha: str, prior_cols: Sequence[str]) -> Tuple[pd.Series, pd.Series, Dict[str, float]]:
    train_series = _safe_numeric(base_df[alpha]).copy()
    test_series = train_series.copy()
    if not prior_cols:
        zero_resid = train_series.copy()
        return zero_resid, zero_resid, {"fit_r2": 0.0, "beta_l2": 0.0, "prior_count": 0.0}
    x = base_df[list(prior_cols)].apply(_safe_numeric).copy()
    valid = pd.concat([train_series.rename("y"), x], axis=1).dropna()
    if len(valid) < MIN_RESIDUAL_TRAIN_ROWS:
        return train_series.copy(), test_series.copy(), {"fit_r2": 0.0, "beta_l2": 0.0, "prior_count": float(len(prior_cols)), "resid_fallback": 1.0}
    y_np = valid["y"].to_numpy(dtype="float64")
    x_np = valid[list(prior_cols)].to_numpy(dtype="float64")
    beta, info = _fit_linear_beta(y_np, x_np)
    x_all = x.fillna(0.0).to_numpy(dtype="float64")
    y_all = train_series.fillna(0.0).to_numpy(dtype="float64")
    resid_all = y_all - (x_all @ beta)
    resid_series = pd.Series(resid_all, index=base_df.index, dtype="float64")
    info["prior_count"] = float(len(prior_cols))
    return resid_series, resid_series.copy(), info


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
        spread = float(top.mean() - bot.mean())
        rows.append({"date": pd.Timestamp(dt).normalize(), "tail_spread": spread, "top_n": int(len(top)), "bot_n": int(len(bot))})
    return pd.DataFrame(rows)


def _build_day_regime_df(frame: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for dt, g in frame.groupby("date", sort=False):
        rec: Dict[str, object] = {"date": pd.Timestamp(dt).normalize()}
        if "market_breadth" in g.columns:
            rec["market_breadth"] = float(_safe_numeric(g["market_breadth"]).mean())
        else:
            rec["market_breadth"] = float("nan")
        if "volume_shock" in g.columns:
            rec["volume_shock"] = float(_safe_numeric(g["volume_shock"]).median())
        else:
            rec["volume_shock"] = float("nan")
        rec["target_abs_median"] = float(_safe_numeric(g[TARGET_COL]).abs().median())
        rows.append(rec)
    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    for col in ["market_breadth", "volume_shock", "target_abs_median"]:
        ser = _safe_numeric(out[col])
        q_hi = float(ser.quantile(0.67)) if ser.notna().any() else float("nan")
        q_lo = float(ser.quantile(0.33)) if ser.notna().any() else float("nan")
        out[f"{col}_hi"] = np.where(ser >= q_hi, 1, 0) if pd.notna(q_hi) else 0
        out[f"{col}_lo"] = np.where(ser <= q_lo, 1, 0) if pd.notna(q_lo) else 0
    return out


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


def _safe_mean(values: Sequence[float]) -> float:
    vals = [float(x) for x in values if pd.notna(x)]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def _safe_last(values: Sequence[float]) -> float:
    vals = [float(x) for x in values if pd.notna(x)]
    if not vals:
        return float("nan")
    return float(vals[-1])


def _last_n_mean(values: Sequence[float], n: int) -> float:
    vals = [float(x) for x in values if pd.notna(x)]
    if not vals:
        return float("nan")
    return float(np.mean(vals[-n:]))


def _positive_fraction(values: Sequence[float]) -> float:
    vals = [float(x) for x in values if pd.notna(x)]
    if not vals:
        return float("nan")
    return float(sum(1 for x in vals if x > 0.0) / len(vals))


def _compute_turnover_and_pnl(port_df: pd.DataFrame) -> pd.DataFrame:
    out = port_df.copy()
    out = out.sort_values(["symbol", "date"]).reset_index(drop=True)
    out["prev_weight"] = out.groupby("symbol", sort=False)["weight"].shift(1).fillna(0.0)
    out["turnover"] = (pd.to_numeric(out["weight"], errors="coerce") - pd.to_numeric(out["prev_weight"], errors="coerce")).abs()
    out["gross_ret"] = pd.to_numeric(out["weight"], errors="coerce") * pd.to_numeric(out[TARGET_COL], errors="coerce")
    out["cost_ret"] = pd.to_numeric(out["turnover"], errors="coerce") * (COST_BPS / 10000.0)
    out["net_ret"] = pd.to_numeric(out["gross_ret"], errors="coerce") - pd.to_numeric(out["cost_ret"], errors="coerce")
    return out


def _apply_optional_overlays(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if ENABLE_REGIME_MULTIPLIERS:
        out = attach_regime_multipliers(out)
    else:
        out["market_regime"] = "neutral"
        out["regime_long_mult"] = 1.0
        out["regime_short_mult"] = 1.0
        out["regime_top_pct"] = ENTER_PCT
    return out


def build_portfolio(scored_df: pd.DataFrame, shell: ShellConfig) -> pd.DataFrame:
    base = scored_df.copy()
    base["score"] = _safe_numeric(base["score"]).fillna(0.0)
    base = _apply_optional_overlays(base)
    out = apply_holding_inertia(base, enter_pct=shell.enter_pct, exit_pct=shell.exit_pct)
    if "side" not in out.columns:
        raise RuntimeError("Holding inertia did not return side column")
    out["raw_strength"] = _safe_numeric(out["score"]).abs().fillna(0.0)
    out.loc[pd.to_numeric(out["side"], errors="coerce") == 0.0, "raw_strength"] = 0.0
    pieces: List[pd.DataFrame] = []
    for _, g in out.groupby("date", sort=False):
        gg = g.copy()
        gg["weight"] = 0.0
        long_mask = pd.to_numeric(gg["side"], errors="coerce") > 0.0
        short_mask = pd.to_numeric(gg["side"], errors="coerce") < 0.0
        long_strength = float(gg.loc[long_mask, "raw_strength"].sum())
        short_strength = float(gg.loc[short_mask, "raw_strength"].sum())
        if long_strength > EPS:
            gg.loc[long_mask, "weight"] = 0.5 * gg.loc[long_mask, "raw_strength"] / long_strength
        if short_strength > EPS:
            gg.loc[short_mask, "weight"] = -0.5 * gg.loc[short_mask, "raw_strength"] / short_strength
        if ENABLE_REGIME_MULTIPLIERS:
            gg.loc[long_mask, "weight"] = pd.to_numeric(gg.loc[long_mask, "weight"], errors="coerce") * float(pd.to_numeric(gg["regime_long_mult"], errors="coerce").iloc[0])
            gg.loc[short_mask, "weight"] = pd.to_numeric(gg.loc[short_mask, "weight"], errors="coerce") * float(pd.to_numeric(gg["regime_short_mult"], errors="coerce").iloc[0])
        gg["weight"] = pd.to_numeric(gg["weight"], errors="coerce").clip(lower=-shell.weight_cap, upper=shell.weight_cap)
        gross = float(gg["weight"].abs().sum())
        if gross > EPS:
            gg["weight"] = gg["weight"] * (shell.gross_target / gross)
        pieces.append(gg)
    out = pd.concat(pieces, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    if ENABLE_DYNAMIC_BUDGETS:
        out = attach_dynamic_side_budgets(out)
        out = apply_side_budgets(out, weight_col="weight")
    out = cap_daily_turnover(out, weight_col="weight", max_daily_turnover=shell.max_daily_turnover)
    out = _compute_turnover_and_pnl(out)
    return out


def evaluate_portfolio(port_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    daily = port_df.groupby("date", sort=False, as_index=False).agg(
        gross_ret=("gross_ret", "sum"),
        cost_ret=("cost_ret", "sum"),
        net_ret=("net_ret", "sum"),
        turnover=("turnover", "sum"),
        gross_exposure=("weight", lambda s: float(pd.to_numeric(s, errors="coerce").abs().sum())),
        names_active=("side", lambda s: int((pd.to_numeric(s, errors="coerce").abs() > 0).sum())),
    )
    daily = daily.sort_values("date").reset_index(drop=True)
    daily["equity"] = (1.0 + daily["net_ret"].fillna(0.0)).cumprod()
    daily["rolling_peak"] = daily["equity"].cummax()
    daily["drawdown"] = np.where(daily["rolling_peak"] > 0.0, daily["equity"] / daily["rolling_peak"] - 1.0, 0.0)
    mean_daily = float(daily["net_ret"].mean()) if len(daily) else float("nan")
    std_daily = float(daily["net_ret"].std(ddof=0)) if len(daily) else float("nan")
    sharpe = float((mean_daily / (std_daily + EPS)) * np.sqrt(252.0)) if len(daily) else float("nan")
    summary = {
        "days": float(len(daily)),
        "mean_daily": mean_daily,
        "std_daily": std_daily,
        "sharpe": sharpe,
        "cum_ret": float(daily["equity"].iloc[-1] - 1.0) if len(daily) else float("nan"),
        "max_drawdown": float(daily["drawdown"].min()) if len(daily) else float("nan"),
        "avg_turnover": float(daily["turnover"].mean()) if len(daily) else float("nan"),
        "avg_names_active": float(daily["names_active"].mean()) if len(daily) else float("nan"),
        "avg_gross_exposure": float(daily["gross_exposure"].mean()) if len(daily) else float("nan"),
    }
    return daily, summary


def _choose_alive_alphas(df: pd.DataFrame) -> Tuple[List[str], pd.DataFrame]:
    alpha_available = set(_alpha_cols(df))
    if ALIVE_ALPHA_LIST:
        chosen = [a for a in ALIVE_ALPHA_LIST if a in alpha_available]
        if not chosen:
            raise RuntimeError("ALIVE_ALPHA_LIST provided but none are present in parquet")
        meta = pd.DataFrame({"alpha": chosen})
        meta["family"] = "unknown"
        meta["standalone_alive"] = 1
    else:
        _must_exist(ALIVE_SUMMARY_CSV, "Alive summary csv")
        meta = pd.read_csv(ALIVE_SUMMARY_CSV)
        if "alpha" not in meta.columns:
            raise RuntimeError(f"Alive summary missing alpha column: {ALIVE_SUMMARY_CSV}")
        if "standalone_alive" in meta.columns:
            meta = meta.loc[pd.to_numeric(meta["standalone_alive"], errors="coerce").fillna(0).astype(int) == 1].copy()
        if meta.empty:
            raise RuntimeError("Alive summary contains no alive alphas")
        sort_cols = [c for c in ["last_fold_sharpe", "last_2_fold_mean_sharpe", "daily_ic_mean", "alpha"] if c in meta.columns]
        asc = [False if c != "alpha" else True for c in sort_cols]
        meta = meta.sort_values(sort_cols, ascending=asc).reset_index(drop=True)
        meta = meta.loc[meta["alpha"].isin(alpha_available)].copy()
        if meta.empty:
            raise RuntimeError("Alive summary alphas are not present in parquet")
        if ALIVE_ALPHA_LIMIT > 0:
            meta = meta.head(ALIVE_ALPHA_LIMIT).copy()
        chosen = meta["alpha"].astype(str).tolist()
    if CORE_ALPHA:
        if CORE_ALPHA not in chosen:
            chosen = [CORE_ALPHA] + chosen
        else:
            chosen = [CORE_ALPHA] + [x for x in chosen if x != CORE_ALPHA]
    return chosen, meta.reset_index(drop=True)


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


def _component_order(alive_meta: pd.DataFrame, chosen: Sequence[str]) -> List[str]:
    if alive_meta.empty:
        return list(chosen)
    order = alive_meta[alive_meta["alpha"].isin(set(chosen))]["alpha"].astype(str).tolist()
    rest = [a for a in chosen if a not in set(order)]
    return order + rest


def _build_component_frames(train_df: pd.DataFrame, test_df: pd.DataFrame, ordered_alphas: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_work = train_df[["date", "symbol", TARGET_COL] + list(ordered_alphas)].copy()
    test_work = test_df[["date", "symbol", TARGET_COL] + list(ordered_alphas)].copy()
    coef_rows: List[Dict[str, object]] = []
    prior_components_train: List[str] = []
    prior_components_test: List[str] = []
    for idx, alpha in enumerate(ordered_alphas, start=1):
        comp_name = f"comp{idx}"
        if idx == 1:
            train_work[f"{comp_name}_raw"] = _safe_numeric(train_work[alpha])
            test_work[f"{comp_name}_raw"] = _safe_numeric(test_work[alpha])
            train_work[f"{comp_name}_resid_raw"] = _safe_numeric(train_work[alpha])
            test_work[f"{comp_name}_resid_raw"] = _safe_numeric(test_work[alpha])
            fit_info = {"fit_r2": 0.0, "beta_l2": 0.0, "prior_count": 0.0, "resid_fallback": 0.0}
        else:
            resid_train, _, fit_info = _residualize_against_prior(train_work[[alpha] + prior_components_train], alpha, prior_components_train)
            resid_test_base = test_work[[alpha] + prior_components_test].copy()
            if prior_components_test:
                x_train = train_work[[alpha] + prior_components_train].copy()
                valid = x_train.dropna()
                if len(valid) >= MIN_RESIDUAL_TRAIN_ROWS:
                    y_np = _safe_numeric(valid[alpha]).to_numpy(dtype="float64")
                    x_np = valid[list(prior_components_train)].apply(_safe_numeric).to_numpy(dtype="float64")
                    beta, _ = _fit_linear_beta(y_np, x_np)
                    x_test_np = resid_test_base[list(prior_components_test)].apply(_safe_numeric).fillna(0.0).to_numpy(dtype="float64")
                    y_test_np = _safe_numeric(resid_test_base[alpha]).fillna(0.0).to_numpy(dtype="float64")
                    resid_test = pd.Series(y_test_np - (x_test_np @ beta), index=resid_test_base.index, dtype="float64")
                else:
                    resid_test = _safe_numeric(test_work[alpha]).copy()
            else:
                resid_test = _safe_numeric(test_work[alpha]).copy()
            train_work[f"{comp_name}_raw"] = _safe_numeric(train_work[alpha])
            test_work[f"{comp_name}_raw"] = _safe_numeric(test_work[alpha])
            train_work[f"{comp_name}_resid_raw"] = resid_train
            test_work[f"{comp_name}_resid_raw"] = resid_test
        train_work[f"{comp_name}_resid"] = train_work.groupby("date", sort=False)[f"{comp_name}_resid_raw"].transform(_robust_zscore_series)
        test_work[f"{comp_name}_resid"] = test_work.groupby("date", sort=False)[f"{comp_name}_resid_raw"].transform(_robust_zscore_series)
        train_work[f"{comp_name}_signed"] = train_work[f"{comp_name}_resid"]
        test_work[f"{comp_name}_signed"] = test_work[f"{comp_name}_resid"]
        prior_components_train.append(f"{comp_name}_resid")
        prior_components_test.append(f"{comp_name}_resid")
        coef_rows.append({"component": comp_name, "alpha": alpha, **fit_info})
    coef_df = pd.DataFrame(coef_rows)
    return train_work, test_work, coef_df


def _sign_regime_diagnostics(test_component_df: pd.DataFrame, score_col: str, component: str, fold_id: int, layer_kind: str) -> Tuple[pd.DataFrame, Dict[str, object]]:
    ic_df = _daily_ic_series(test_component_df, score_col, TARGET_COL, min_cs=MIN_DAILY_IC_CS)
    sign_info = _sign_decision(ic_df)
    signed_df = test_component_df.copy()
    signed_df["score"] = _safe_numeric(signed_df[score_col]) * float(sign_info["train_sign_locked"])
    test_ic_df = _daily_ic_series(signed_df, "score", TARGET_COL, min_cs=MIN_DAILY_IC_CS)
    tail_df = _tail_spread_series(signed_df, "score", TARGET_COL, top_pct=TAIL_TOP_PCT)
    regime_df = _build_day_regime_df(test_component_df)
    diag_rows: List[Dict[str, object]] = []
    test_ic_merge = test_ic_df.merge(regime_df, on="date", how="left")
    for regime_col in ["market_breadth_hi", "market_breadth_lo", "volume_shock_hi", "volume_shock_lo", "target_abs_median_hi", "target_abs_median_lo"]:
        if regime_col not in test_ic_merge.columns:
            continue
        sub = test_ic_merge.loc[pd.to_numeric(test_ic_merge[regime_col], errors="coerce").fillna(0).astype(int) == 1].copy()
        diag_rows.append({
            "fold_id": int(fold_id),
            "component": component,
            "layer_kind": layer_kind,
            "regime": regime_col,
            "regime_days": int(len(sub)),
            "regime_mean_ic": float(sub["daily_ic"].mean()) if len(sub) else float("nan"),
            "regime_sign_stability": float((sub["daily_ic"] > 0.0).mean()) if len(sub) else float("nan"),
        })
    tail_merge = tail_df.merge(regime_df, on="date", how="left") if len(tail_df) else pd.DataFrame()
    tail_stability = float((tail_df["tail_spread"] > 0.0).mean()) if len(tail_df) else float("nan")
    head = {
        "fold_id": int(fold_id),
        "component": component,
        "layer_kind": layer_kind,
        **sign_info,
        "test_mean_ic": float(test_ic_df["daily_ic"].mean()) if len(test_ic_df) else float("nan"),
        "test_sign_stability": float((test_ic_df["daily_ic"] > 0.0).mean()) if len(test_ic_df) else float("nan"),
        "tail_mean_spread": float(tail_df["tail_spread"].mean()) if len(tail_df) else float("nan"),
        "tail_positive_rate": tail_stability,
    }
    return pd.DataFrame(diag_rows), head | {"signed_df": signed_df, "test_ic_df": test_ic_df, "tail_df": tail_df, "tail_merge_rows": int(len(tail_merge))}


def _corr_summary(test_df: pd.DataFrame, component_cols: Sequence[str], fold_id: int) -> pd.DataFrame:
    cols = [c for c in component_cols if c in test_df.columns]
    if not cols:
        return pd.DataFrame(columns=["fold_id", "left", "right", "spearman_corr"])
    corr = test_df[cols].apply(_safe_numeric).fillna(0.0).corr(method="spearman")
    rows: List[Dict[str, object]] = []
    for i, left in enumerate(cols):
        for right in cols[i + 1:]:
            rows.append({"fold_id": int(fold_id), "left": left, "right": right, "spearman_corr": float(corr.loc[left, right])})
    return pd.DataFrame(rows)


def run_fold(df: pd.DataFrame, split: WFSplit, ordered_alphas: Sequence[str], shell: ShellConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = df.loc[(df["date"] >= split.train_start) & (df["date"] <= split.train_end)].copy()
    test_df = df.loc[(df["date"] >= split.test_start) & (df["date"] <= split.test_end)].copy()
    if len(train_df) < MIN_TRAIN_ROWS:
        raise RuntimeError(f"Fold {split.fold_id}: too few train rows: {len(train_df)}")
    if len(test_df) < MIN_TEST_ROWS:
        raise RuntimeError(f"Fold {split.fold_id}: too few test rows: {len(test_df)}")
    train_df = _cs_zscore_df(train_df, ordered_alphas)
    test_df = _cs_zscore_df(test_df, ordered_alphas)
    train_work, test_work, coef_df = _build_component_frames(train_df, test_df, ordered_alphas)

    component_rows: List[Dict[str, object]] = []
    sign_rows: List[Dict[str, object]] = []
    daily_rows: List[pd.DataFrame] = []
    incremental_rows: List[Dict[str, object]] = []
    component_cols: List[str] = []

    for idx, alpha in enumerate(ordered_alphas, start=1):
        component = f"comp{idx}"
        component_cols.append(f"{component}_signed")
        diag_df, diag_head = _sign_regime_diagnostics(test_work[["date", "symbol", TARGET_COL, f"{component}_resid"]].rename(columns={f"{component}_resid": "feature"}), "feature", component, split.fold_id, "component")
        sign_rows.append({k: v for k, v in diag_head.items() if k != "signed_df" and k != "test_ic_df" and k != "tail_df"})
        if len(diag_df):
            sign_rows.extend(diag_df.to_dict("records"))
        signed_component_df = test_work[["date", "symbol", TARGET_COL]].copy()
        signed_component_df["score"] = _safe_numeric(test_work[f"{component}_resid"]) * float(diag_head["train_sign_locked"])
        port_df = build_portfolio(signed_component_df, shell)
        daily_df, summary = evaluate_portfolio(port_df)
        daily_df["fold_id"] = int(split.fold_id)
        daily_df["component"] = component
        daily_df["alpha"] = alpha
        daily_df["layer_kind"] = "component"
        daily_rows.append(daily_df)
        component_rows.append({
            "fold_id": int(split.fold_id),
            "component": component,
            "alpha": alpha,
            "layer_kind": "component",
            **summary,
            "train_sign_locked": float(diag_head["train_sign_locked"]),
            "sign_lock_triggered": int(diag_head["sign_lock_triggered"]),
            "sign_lock_reason": str(diag_head["sign_lock_reason"]),
            "test_mean_ic": float(diag_head["test_mean_ic"]),
            "test_sign_stability": float(diag_head["test_sign_stability"]),
            "tail_mean_spread": float(diag_head["tail_mean_spread"]),
            "tail_positive_rate": float(diag_head["tail_positive_rate"]),
        })

        cumulative_name = f"cum{idx}"
        cumulative_df = test_work[["date", "symbol", TARGET_COL]].copy()
        cumulative_df["score"] = 0.0
        for j in range(1, idx + 1):
            comp_j = f"comp{j}"
            sign_j = next(x for x in component_rows if x["fold_id"] == int(split.fold_id) and x["component"] == comp_j and x["layer_kind"] == "component")["train_sign_locked"]
            cumulative_df["score"] = pd.to_numeric(cumulative_df["score"], errors="coerce") + (_safe_numeric(test_work[f"{comp_j}_resid"]) * float(sign_j))
        cumulative_df["score"] = cumulative_df.groupby("date", sort=False)["score"].transform(_robust_zscore_series)
        cum_port_df = build_portfolio(cumulative_df, shell)
        cum_daily_df, cum_summary = evaluate_portfolio(cum_port_df)
        cum_daily_df["fold_id"] = int(split.fold_id)
        cum_daily_df["component"] = cumulative_name
        cum_daily_df["alpha"] = alpha
        cum_daily_df["layer_kind"] = "cumulative"
        daily_rows.append(cum_daily_df)
        prev_sharpe = component_rows[-2]["sharpe"] if len(component_rows) >= 2 else float("nan")
        incremental_rows.append({
            "fold_id": int(split.fold_id),
            "component": cumulative_name,
            "alpha_added": alpha,
            "component_count": int(idx),
            "cum_sharpe": float(cum_summary["sharpe"]),
            "cum_mean_daily": float(cum_summary["mean_daily"]),
            "cum_ret": float(cum_summary["cum_ret"]),
            "cum_max_drawdown": float(cum_summary["max_drawdown"]),
            "cum_turnover": float(cum_summary["avg_turnover"]),
            "delta_vs_prev_component_sharpe": float(cum_summary["sharpe"] - prev_sharpe) if pd.notna(prev_sharpe) else float("nan"),
        })

    corr_df = _corr_summary(test_work, [f"comp{i}_resid" for i in range(1, len(ordered_alphas) + 1)], split.fold_id)
    component_df = pd.DataFrame(component_rows)
    sign_df = pd.DataFrame(sign_rows)
    daily_all_df = pd.concat(daily_rows, ignore_index=True) if daily_rows else pd.DataFrame()
    incremental_df = pd.DataFrame(incremental_rows)
    coef_df["fold_id"] = int(split.fold_id)
    return component_df, sign_df, daily_all_df, incremental_df, coef_df, corr_df


def _summarize_by_component(component_fold_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    if component_fold_df.empty:
        return pd.DataFrame()
    work = component_fold_df.sort_values(["layer_kind", "component", "fold_id"]).reset_index(drop=True)
    for (layer_kind, component, alpha), g in work.groupby(["layer_kind", "component", "alpha"], sort=False):
        sharpes = [float(x) for x in g["sharpe"].tolist() if pd.notna(x)]
        rows.append({
            "layer_kind": layer_kind,
            "component": component,
            "alpha": alpha,
            "folds": int(len(g)),
            "oos_sharpe_mean": _safe_mean(sharpes),
            "last_fold_sharpe": _safe_last(g["sharpe"].tolist()),
            "last_2_fold_mean_sharpe": _last_n_mean(g["sharpe"].tolist(), 2),
            "test_mean_ic": _safe_mean(g["test_mean_ic"].tolist()),
            "sign_stability": _safe_mean(g["test_sign_stability"].tolist()),
            "tail_positive_rate": _safe_mean(g["tail_positive_rate"].tolist()),
            "max_drawdown": float(pd.to_numeric(g["max_drawdown"], errors="coerce").min()) if len(g) else float("nan"),
            "turnover_mean": _safe_mean(g["avg_turnover"].tolist()),
            "positive_sharpe_fraction": _positive_fraction(g["sharpe"].tolist()),
            "sign_lock_rate": _safe_mean(g["sign_lock_triggered"].tolist()),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["layer_kind", "last_2_fold_mean_sharpe", "oos_sharpe_mean", "component"], ascending=[True, False, False, True]).reset_index(drop=True)
    return out


def _summarize_incremental(incremental_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    if incremental_df.empty:
        return pd.DataFrame()
    for component, g in incremental_df.sort_values(["component", "fold_id"]).groupby("component", sort=False):
        rows.append({
            "component": component,
            "alpha_added": str(g["alpha_added"].iloc[0]),
            "folds": int(len(g)),
            "cum_sharpe_mean": _safe_mean(g["cum_sharpe"].tolist()),
            "last_fold_cum_sharpe": _safe_last(g["cum_sharpe"].tolist()),
            "last_2_fold_cum_sharpe": _last_n_mean(g["cum_sharpe"].tolist(), 2),
            "delta_vs_prev_component_sharpe_mean": _safe_mean(g["delta_vs_prev_component_sharpe"].tolist()),
            "cum_turnover_mean": _safe_mean(g["cum_turnover"].tolist()),
            "cum_max_drawdown": float(pd.to_numeric(g["cum_max_drawdown"], errors="coerce").min()) if len(g) else float("nan"),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["last_2_fold_cum_sharpe", "cum_sharpe_mean", "component"], ascending=[False, False, True]).reset_index(drop=True)
    return out


def main() -> int:
    _enable_line_buffering()
    shell = ShellConfig(enter_pct=ENTER_PCT, exit_pct=EXIT_PCT, weight_cap=WEIGHT_CAP, gross_target=GROSS_TARGET, max_daily_turnover=MAX_DAILY_TURNOVER)
    print(f"[CFG] alpha_lib_file={ALPHA_LIB_FILE}")
    print(f"[CFG] alive_summary_csv={ALIVE_SUMMARY_CSV}")
    print(f"[CFG] target_col={TARGET_COL}")
    print(f"[CFG] residual_method={RESIDUAL_METHOD} alive_alpha_limit={ALIVE_ALPHA_LIMIT} core_alpha={CORE_ALPHA!r}")
    print(f"[CFG] wf train_days={TRAIN_DAYS} test_days={TEST_DAYS} step_days={STEP_DAYS} purge_days={PURGE_DAYS} embargo_days={EMBARGO_DAYS}")
    print(f"[CFG] shell gross_target={GROSS_TARGET} weight_cap={WEIGHT_CAP} max_daily_turnover={MAX_DAILY_TURNOVER} enter_pct={ENTER_PCT} exit_pct={EXIT_PCT} cost_bps={COST_BPS}")
    print(f"[CFG] sign_lock_ic_abs={SIGN_LOCK_IC_ABS} sign_lock_pos_rate={SIGN_LOCK_POS_RATE} tail_top_pct={TAIL_TOP_PCT}")
    print(f"[CFG] enable_dynamic_budgets={int(ENABLE_DYNAMIC_BUDGETS)} has_budget_allocation={int(HAS_BUDGET_ALLOCATION)} enable_regime_multipliers={int(ENABLE_REGIME_MULTIPLIERS)} has_regime_allocation={int(HAS_REGIME_ALLOCATION)}")

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

    chosen_alphas, alive_meta = _choose_alive_alphas(df)
    ordered_alphas = _component_order(alive_meta, chosen_alphas)
    print(f"[ALIVE] selected_count={len(ordered_alphas)}")
    for idx, alpha in enumerate(ordered_alphas, start=1):
        print(f"[ALIVE][{idx}] {alpha}")

    splits = build_walkforward_splits(df["date"])
    print(f"[WF] folds={len(splits)}")
    for sp in splits:
        print(f"[WF][FOLD {sp.fold_id}] train={sp.train_start.date()}..{sp.train_end.date()} test={sp.test_start.date()}..{sp.test_end.date()}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_component_folds: List[pd.DataFrame] = []
    all_sign_rows: List[pd.DataFrame] = []
    all_daily_rows: List[pd.DataFrame] = []
    all_incremental_rows: List[pd.DataFrame] = []
    all_coef_rows: List[pd.DataFrame] = []
    all_corr_rows: List[pd.DataFrame] = []

    for split in splits:
        component_df, sign_df, daily_df, incremental_df, coef_df, corr_df = run_fold(df, split, ordered_alphas, shell)
        all_component_folds.append(component_df)
        all_sign_rows.append(sign_df)
        all_daily_rows.append(daily_df)
        all_incremental_rows.append(incremental_df)
        all_coef_rows.append(coef_df)
        all_corr_rows.append(corr_df)
        component_df.to_csv(OUT_DIR / f"residual_component_metrics__fold{split.fold_id}.csv", index=False)
        sign_df.to_csv(OUT_DIR / f"residual_sign_diagnostics__fold{split.fold_id}.csv", index=False)
        daily_df.to_csv(OUT_DIR / f"residual_daily__fold{split.fold_id}.csv", index=False)
        incremental_df.to_csv(OUT_DIR / f"residual_incremental__fold{split.fold_id}.csv", index=False)
        coef_df.to_csv(OUT_DIR / f"residual_coefficients__fold{split.fold_id}.csv", index=False)
        corr_df.to_csv(OUT_DIR / f"residual_corr__fold{split.fold_id}.csv", index=False)
        for _, row in component_df.sort_values(["layer_kind", "component"]).iterrows():
            print(f"[FOLD {split.fold_id}][{row['layer_kind']}][{row['component']}] alpha={row['alpha']} sharpe={row['sharpe']:.4f} mean_ic={row['test_mean_ic']:.5f} sign_stability={row['test_sign_stability']:.4f} maxdd={row['max_drawdown']:.4f} turnover={row['avg_turnover']:.4f}")
        for _, row in incremental_df.sort_values("component_count").iterrows():
            print(f"[FOLD {split.fold_id}][incremental][{row['component']}] add={row['alpha_added']} cum_sharpe={row['cum_sharpe']:.4f} delta_vs_prev_component_sharpe={row['delta_vs_prev_component_sharpe']:.4f}")

    component_fold_df = pd.concat(all_component_folds, ignore_index=True) if all_component_folds else pd.DataFrame()
    sign_df_all = pd.concat(all_sign_rows, ignore_index=True) if all_sign_rows else pd.DataFrame()
    daily_df_all = pd.concat(all_daily_rows, ignore_index=True) if all_daily_rows else pd.DataFrame()
    incremental_df_all = pd.concat(all_incremental_rows, ignore_index=True) if all_incremental_rows else pd.DataFrame()
    coef_df_all = pd.concat(all_coef_rows, ignore_index=True) if all_coef_rows else pd.DataFrame()
    corr_df_all = pd.concat(all_corr_rows, ignore_index=True) if all_corr_rows else pd.DataFrame()

    component_summary_df = _summarize_by_component(component_fold_df)
    incremental_summary_df = _summarize_incremental(incremental_df_all)

    component_fold_df.to_csv(OUT_DIR / "residual_component_metrics__all_folds.csv", index=False)
    sign_df_all.to_csv(OUT_DIR / "residual_sign_diagnostics__all_folds.csv", index=False)
    daily_df_all.to_csv(OUT_DIR / "residual_daily__all_folds.csv", index=False)
    incremental_df_all.to_csv(OUT_DIR / "residual_incremental__all_folds.csv", index=False)
    coef_df_all.to_csv(OUT_DIR / "residual_coefficients__all_folds.csv", index=False)
    corr_df_all.to_csv(OUT_DIR / "residual_corr__all_folds.csv", index=False)
    component_summary_df.to_csv(OUT_DIR / "residual_component_summary.csv", index=False)
    incremental_summary_df.to_csv(OUT_DIR / "residual_incremental_summary.csv", index=False)

    if len(component_summary_df):
        print("[SUMMARY][COMPONENTS]")
        print(component_summary_df.head(TOPK_PRINT).to_string(index=False))
    if len(incremental_summary_df):
        print("[SUMMARY][INCREMENTAL]")
        print(incremental_summary_df.head(TOPK_PRINT).to_string(index=False))

    meta = {
        "alpha_lib_file": str(ALPHA_LIB_FILE),
        "alive_summary_csv": str(ALIVE_SUMMARY_CSV),
        "target_col": TARGET_COL,
        "ordered_alphas": ordered_alphas,
        "residual_method": RESIDUAL_METHOD,
        "wf": {
            "train_days": TRAIN_DAYS,
            "test_days": TEST_DAYS,
            "step_days": STEP_DAYS,
            "purge_days": PURGE_DAYS,
            "embargo_days": EMBARGO_DAYS,
            "folds": len(splits),
        },
        "shell": {
            "enter_pct": ENTER_PCT,
            "exit_pct": EXIT_PCT,
            "weight_cap": WEIGHT_CAP,
            "gross_target": GROSS_TARGET,
            "max_daily_turnover": MAX_DAILY_TURNOVER,
            "cost_bps": COST_BPS,
        },
        "sign_lock": {
            "sign_lock_ic_abs": SIGN_LOCK_IC_ABS,
            "sign_lock_pos_rate": SIGN_LOCK_POS_RATE,
        },
        "modules": {
            "holding_inertia": int(HAS_HOLDING_INERTIA),
            "turnover_control": int(HAS_TURNOVER_CONTROL),
            "budget_allocation": int(HAS_BUDGET_ALLOCATION),
            "regime_allocation": int(HAS_REGIME_ALLOCATION),
        },
        "overlays": {
            "enable_dynamic_budgets": int(ENABLE_DYNAMIC_BUDGETS),
            "enable_regime_multipliers": int(ENABLE_REGIME_MULTIPLIERS),
        },
        "rows": {
            "component_fold_rows": int(len(component_fold_df)),
            "sign_rows": int(len(sign_df_all)),
            "daily_rows": int(len(daily_df_all)),
            "incremental_rows": int(len(incremental_df_all)),
            "coef_rows": int(len(coef_df_all)),
            "corr_rows": int(len(corr_df_all)),
        },
    }
    meta_path = OUT_DIR / "residual_layer_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ARTIFACT] {OUT_DIR / 'residual_component_summary.csv'}")
    print(f"[ARTIFACT] {OUT_DIR / 'residual_incremental_summary.csv'}")
    print(f"[ARTIFACT] {OUT_DIR / 'residual_sign_diagnostics__all_folds.csv'}")
    print(f"[ARTIFACT] {OUT_DIR / 'residual_corr__all_folds.csv'}")
    print(f"[ARTIFACT] {meta_path}")
    print("[FINAL] residual layer walk-forward complete")
    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 1
    _safe_exit(rc)
