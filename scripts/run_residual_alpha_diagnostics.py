# scripts/run_residual_alpha_diagnostics.py
# Massive factor sweep for residual stat-arb research
# Double-click runnable. Never auto-closes.
#
# Goal:
# - add a very broad feature universe in one pass
# - measure factor IC by horizon
# - export a ranked factor table for later interaction-model work

from __future__ import annotations

import json
import os
import random
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[1]
_ARTIFACT_DIR = _REPO_ROOT / "artifacts" / "residual_alpha_diagnostics"

EPS = 1e-12


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _press_enter_exit(code: int) -> None:
    try:
        print(f"\n[EXIT] code={code}")
        input("Press Enter to exit...")
    except Exception:
        pass
    raise SystemExit(code)


def _env_str(name: str, default: str = "") -> str:
    v = os.environ.get(name, default)
    return (v if v is not None else default).strip()


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or not str(v).strip():
        return default
    try:
        return int(float(str(v).strip()))
    except Exception:
        return default


def _safe(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("float64")


def _winsorize_series(x: pd.Series, lower_q: float = 0.02, upper_q: float = 0.98) -> pd.Series:
    s = _safe(x)
    valid = s.dropna()
    if valid.empty:
        return s
    lo = float(valid.quantile(lower_q))
    hi = float(valid.quantile(upper_q))
    return s.clip(lower=lo, upper=hi)


def _robust_zscore(x: pd.Series) -> pd.Series:
    s = _safe(x)
    valid = s.dropna()
    if valid.empty:
        return pd.Series(0.0, index=s.index, dtype="float64")
    med = float(valid.median())
    mad = float((valid - med).abs().median())
    if mad > EPS:
        return ((s - med) / (1.4826 * mad)).fillna(0.0)
    mean = float(valid.mean())
    std = float(valid.std())
    if std > EPS:
        return ((s - mean) / std).fillna(0.0)
    return pd.Series(0.0, index=s.index, dtype="float64")


def _rolling_autocorr_1(series: pd.Series, window: int) -> pd.Series:
    vals = _safe(series)
    out = pd.Series(np.nan, index=vals.index, dtype="float64")
    for i in range(window, len(vals) + 1):
        chunk = vals.iloc[i - window:i].dropna()
        if len(chunk) < max(5, window // 3):
            continue
        out.iloc[i - 1] = chunk.autocorr(lag=1)
    return out


def _group_rolling_apply(df: pd.DataFrame, col: str, func, new_col: str) -> pd.DataFrame:
    out = df.copy()
    out[new_col] = np.nan
    for _, idx in out.groupby("symbol", sort=False).groups.items():
        out.loc[idx, new_col] = func(out.loc[idx, col]).to_numpy()
    return out


@dataclass(frozen=True)
class Config:
    dataset_root: Path
    start: str
    end: str
    seed: int


def load_config() -> Config:
    return Config(
        dataset_root=Path(_env_str("DATASET_ROOT", r"D:\massive_dataset")),
        start=_env_str("START", "2023-01-01"),
        end=_env_str("END", "2026-02-28"),
        seed=_env_int("SEED", 7),
    )


# ------------------------------------------------------------
# DATA
# ------------------------------------------------------------

def _find_files(root: Path) -> List[Tuple[str, Path]]:
    out: List[Tuple[str, Path]] = []
    if not root.exists():
        raise RuntimeError(f"dataset_root not found: {root}")
    for sym_dir in sorted(root.iterdir()):
        if not sym_dir.is_dir():
            continue
        sym = sym_dir.name.upper()
        files = list(sym_dir.glob("aggs_1d_*.json"))
        if not files:
            continue
        best = max(files, key=lambda x: x.stat().st_size)
        out.append((sym, best))
    return out


def _load_file(sym: str, path: Path) -> pd.DataFrame:
    js = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    rows = js.get("results") or []
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "t" not in df.columns:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.date.astype(str)
    df["symbol"] = sym
    for c in ["o", "h", "l", "c", "v"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["date", "symbol", "o", "h", "l", "c", "v"]]


def load_dataset(cfg: Config) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []
    for sym, path in _find_files(cfg.dataset_root):
        d = _load_file(sym, path)
        if not d.empty:
            dfs.append(d)
    if not dfs:
        raise RuntimeError("No aggs_1d data loaded")
    df = pd.concat(dfs, ignore_index=True)
    df = df[(df["date"] >= cfg.start) & (df["date"] <= cfg.end)]
    return df.sort_values(["date", "symbol"]).reset_index(drop=True)


# ------------------------------------------------------------
# FACTORS
# ------------------------------------------------------------

def add_factors(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["o", "h", "l", "c", "v"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    prev_close = out.groupby("symbol")["c"].shift(1)
    prev_open = out.groupby("symbol")["o"].shift(1)
    prev_high = out.groupby("symbol")["h"].shift(1)
    prev_low = out.groupby("symbol")["l"].shift(1)

    out["ret1"] = out.groupby("symbol")["c"].pct_change(1, fill_method=None)
    out["ret2"] = out.groupby("symbol")["c"].pct_change(2, fill_method=None)
    out["ret3"] = out.groupby("symbol")["c"].pct_change(3, fill_method=None)
    out["ret5"] = out.groupby("symbol")["c"].pct_change(5, fill_method=None)
    out["ret10"] = out.groupby("symbol")["c"].pct_change(10, fill_method=None)
    out["ret20"] = out.groupby("symbol")["c"].pct_change(20, fill_method=None)

    out["open_close_ret"] = (out["c"] / out["o"]) - 1.0
    out["gap_ret"] = (out["o"] / prev_close) - 1.0
    out["high_close_ret"] = (out["c"] / prev_high) - 1.0
    out["close_low_ret"] = (out["c"] / prev_low) - 1.0

    out["range"] = (out["h"] - out["l"]) / (out["c"] + EPS)
    out["body"] = (out["c"] - out["o"]) / (out["o"] + EPS)
    out["upper_wick"] = (out["h"] - out[["o", "c"]].max(axis=1)) / (out["c"] + EPS)
    out["lower_wick"] = (out[["o", "c"]].min(axis=1) - out["l"]) / (out["c"] + EPS)
    out["wick_imbalance"] = out["upper_wick"] - out["lower_wick"]
    out["body_to_range"] = (out["body"] / (out["range"] + EPS)).replace([np.inf, -np.inf], np.nan)

    out["clv"] = ((out["c"] - out["l"]) - (out["h"] - out["c"])) / ((out["h"] - out["l"]) + EPS)
    out["pressure"] = (out["c"] - out["o"]) / ((out["h"] - out["l"]) + EPS)

    out["mom3"] = out["ret3"]
    out["mom5"] = out["ret5"]
    out["mom10"] = out["ret10"]
    out["mom20"] = out["ret20"]
    out["rev1"] = -out["ret1"]
    out["rev2"] = -out["ret2"]
    out["rev3"] = -out.groupby("symbol")["ret1"].rolling(3, min_periods=2).sum().reset_index(level=0, drop=True)
    out["rev5"] = -out.groupby("symbol")["ret1"].rolling(5, min_periods=3).sum().reset_index(level=0, drop=True)

    out["vol5"] = out.groupby("symbol")["ret1"].rolling(5, min_periods=3).std().reset_index(level=0, drop=True)
    out["vol10"] = out.groupby("symbol")["ret1"].rolling(10, min_periods=5).std().reset_index(level=0, drop=True)
    out["vol20"] = out.groupby("symbol")["ret1"].rolling(20, min_periods=10).std().reset_index(level=0, drop=True)
    out["vol40"] = out.groupby("symbol")["ret1"].rolling(40, min_periods=15).std().reset_index(level=0, drop=True)
    out["vol_spike_5_20"] = out["vol5"] / (out["vol20"] + EPS)
    out["vol_spike_10_40"] = out["vol10"] / (out["vol40"] + EPS)
    out["vol_comp_10_20"] = out["vol10"] / (out["vol20"] + EPS)
    out["vol_comp_5_40"] = out["vol5"] / (out["vol40"] + EPS)
    out["range_comp_5_20"] = (
        out.groupby("symbol")["range"].rolling(5, min_periods=3).mean().reset_index(level=0, drop=True)
        /
        (out.groupby("symbol")["range"].rolling(20, min_periods=10).mean().reset_index(level=0, drop=True) + EPS)
    )

    out["vol_mean_5"] = out.groupby("symbol")["v"].rolling(5, min_periods=3).mean().reset_index(level=0, drop=True)
    out["vol_mean_20"] = out.groupby("symbol")["v"].rolling(20, min_periods=10).mean().reset_index(level=0, drop=True)
    out["vol_mean_60"] = out.groupby("symbol")["v"].rolling(60, min_periods=20).mean().reset_index(level=0, drop=True)
    out["rel_volume_5"] = out["v"] / (out["vol_mean_5"] + EPS)
    out["rel_volume_20"] = out["v"] / (out["vol_mean_20"] + EPS)
    out["rel_volume_60"] = out["v"] / (out["vol_mean_60"] + EPS)
    out["vol_accel"] = out["rel_volume_5"] / (out.groupby("symbol")["rel_volume_5"].shift(1) + EPS)
    out["vol_persist_3"] = out.groupby("symbol")["rel_volume_20"].rolling(3, min_periods=2).mean().reset_index(level=0, drop=True)
    out["vol_persist_5"] = out.groupby("symbol")["rel_volume_20"].rolling(5, min_periods=3).mean().reset_index(level=0, drop=True)

    out["dollar_vol"] = out["v"] * out["c"]
    out["dollar_vol_mean_20"] = out.groupby("symbol")["dollar_vol"].rolling(20, min_periods=10).mean().reset_index(level=0, drop=True)
    out["liq"] = np.log(out["dollar_vol"] + 1.0)
    out["liq_shock"] = out["dollar_vol"] / (out["dollar_vol_mean_20"] + EPS)
    out["amihud_proxy"] = out["ret1"].abs() / (out["dollar_vol"] + EPS)
    out["amihud_shock"] = out["amihud_proxy"] / (out.groupby("symbol")["amihud_proxy"].rolling(20, min_periods=10).mean().reset_index(level=0, drop=True) + EPS)

    ema10 = out.groupby("symbol")["c"].transform(lambda s: s.ewm(span=10, adjust=False, min_periods=4).mean())
    ema20 = out.groupby("symbol")["c"].transform(lambda s: s.ewm(span=20, adjust=False, min_periods=7).mean())
    ema50 = out.groupby("symbol")["c"].transform(lambda s: s.ewm(span=50, adjust=False, min_periods=15).mean())
    out["dist_ema10"] = (out["c"] - ema10) / (ema10 + EPS)
    out["dist_ema20"] = (out["c"] - ema20) / (ema20 + EPS)
    out["dist_ema50"] = (out["c"] - ema50) / (ema50 + EPS)
    out["ema_stack_10_20"] = (ema10 - ema20) / (ema20 + EPS)
    out["ema_stack_20_50"] = (ema20 - ema50) / (ema50 + EPS)
    out["ema_slope_10"] = out.groupby("symbol")["dist_ema10"].diff(1)
    out["ema_slope_20"] = out.groupby("symbol")["dist_ema20"].diff(1)

    roll_max_20 = out.groupby("symbol")["c"].rolling(20, min_periods=10).max().reset_index(level=0, drop=True)
    roll_min_20 = out.groupby("symbol")["c"].rolling(20, min_periods=10).min().reset_index(level=0, drop=True)
    roll_max_60 = out.groupby("symbol")["c"].rolling(60, min_periods=20).max().reset_index(level=0, drop=True)
    roll_min_60 = out.groupby("symbol")["c"].rolling(60, min_periods=20).min().reset_index(level=0, drop=True)
    out["dist_high_20"] = (out["c"] / (roll_max_20 + EPS)) - 1.0
    out["dist_low_20"] = (out["c"] / (roll_min_20 + EPS)) - 1.0
    out["dist_high_60"] = (out["c"] / (roll_max_60 + EPS)) - 1.0
    out["dist_low_60"] = (out["c"] / (roll_min_60 + EPS)) - 1.0
    out["position_in_20d_range"] = (out["c"] - roll_min_20) / ((roll_max_20 - roll_min_20) + EPS)
    out["position_in_60d_range"] = (out["c"] - roll_min_60) / ((roll_max_60 - roll_min_60) + EPS)

    out["accel_3_1"] = out["mom3"] - out["ret1"]
    out["accel_5_3"] = out["mom5"] - out["mom3"]
    out["accel_10_5"] = out["mom10"] - out["mom5"]

    out["gap_vol_adj"] = out["gap_ret"] / (out["vol10"] + EPS)
    out["gap_range_adj"] = out["gap_ret"] / (out["range"] + EPS)
    out["gap_fill_pressure"] = out["gap_ret"] * (-out["pressure"])

    out["range_vol"] = out["range"] * out["rel_volume_20"]
    out["body_vol"] = out["body"] * out["rel_volume_20"]
    out["rev1_x_vol"] = out["rev1"] * out["rel_volume_20"]
    out["mom3_x_vol"] = out["mom3"] * out["rel_volume_20"]
    out["rev1_x_volcomp"] = out["rev1"] * out["vol_comp_10_20"]
    out["mom3_x_volcomp"] = out["mom3"] * out["vol_comp_10_20"]
    out["range_x_volcomp"] = out["range"] * out["vol_comp_10_20"]
    out["gap_x_relvol"] = out["gap_ret"] * out["rel_volume_20"]
    out["pressure_x_relvol"] = out["pressure"] * out["rel_volume_20"]

    out["ret_sign_imbalance_5"] = out.groupby("symbol")["ret1"].rolling(5, min_periods=3).apply(lambda x: float((x > 0).mean()) - float((x < 0).mean()), raw=False).reset_index(level=0, drop=True)
    out["ret_sign_imbalance_10"] = out.groupby("symbol")["ret1"].rolling(10, min_periods=5).apply(lambda x: float((x > 0).mean()) - float((x < 0).mean()), raw=False).reset_index(level=0, drop=True)
    out["ret_autocorr_10"] = np.nan
    out["ret_autocorr_20"] = np.nan
    for _, idx in out.groupby("symbol", sort=False).groups.items():
        s = out.loc[idx, "ret1"]
        out.loc[idx, "ret_autocorr_10"] = _rolling_autocorr_1(s, 10).to_numpy()
        out.loc[idx, "ret_autocorr_20"] = _rolling_autocorr_1(s, 20).to_numpy()

    out["mkt_ret1"] = out.groupby("date")["ret1"].transform("mean")
    out["mkt_vol20"] = out.groupby("date")["vol20"].transform("mean")
    out["mkt_range"] = out.groupby("date")["range"].transform("mean")
    out["breadth_up"] = out.groupby("date")["ret1"].transform(lambda s: float((s > 0).mean()))
    out["breadth_down"] = out.groupby("date")["ret1"].transform(lambda s: float((s < 0).mean()))
    out["breadth_thrust"] = out["breadth_up"] - out["breadth_down"]
    out["cs_ret_z"] = out.groupby("date")["ret1"].transform(lambda s: _robust_zscore(s))
    out["cs_gap_z"] = out.groupby("date")["gap_ret"].transform(lambda s: _robust_zscore(s))
    out["cs_range_z"] = out.groupby("date")["range"].transform(lambda s: _robust_zscore(s))
    out["cs_relvol_z"] = out.groupby("date")["rel_volume_20"].transform(lambda s: _robust_zscore(s))
    out["cs_liq_z"] = out.groupby("date")["liq"].transform(lambda s: _robust_zscore(s))
    out["cs_volshock_z"] = out.groupby("date")["vol_spike_5_20"].transform(lambda s: _robust_zscore(s))
    out["cs_mom5_rank"] = out.groupby("date")["mom5"].rank(method="average", pct=True)
    out["cs_rev1_rank"] = out.groupby("date")["rev1"].rank(method="average", pct=True)

    out["liq_imbalance"] = out["rel_volume_20"] / (out.groupby("date")["rel_volume_20"].transform("median") + EPS)
    out["turnover_imbalance"] = out["liq_shock"] / (out.groupby("date")["liq_shock"].transform("median") + EPS)
    out["amihud_imbalance"] = out["amihud_shock"] / (out.groupby("date")["amihud_shock"].transform("median") + EPS)

    out["overnight_pressure"] = out["gap_ret"] - out["body"]
    out["intraday_reversal_pressure"] = -out["body_to_range"] * out["range"]
    out["exhaustion_up"] = out["dist_high_20"] * out["rel_volume_20"]
    out["exhaustion_down"] = out["dist_low_20"] * out["rel_volume_20"]

    return out


# ------------------------------------------------------------
# FORWARD RETURNS
# ------------------------------------------------------------

def add_forward_returns(df: pd.DataFrame, horizons: Tuple[int, ...] = (1, 2, 3, 5)) -> pd.DataFrame:
    out = df.copy()
    entry = out.groupby("symbol")["o"].shift(-1)
    for h in horizons:
        exit_px = out.groupby("symbol")["c"].shift(-h)
        out[f"fwd_{h}d"] = (exit_px / entry) - 1.0
    return out


# ------------------------------------------------------------
# IC TEST
# ------------------------------------------------------------

def factor_ic(df: pd.DataFrame, factor: str, horizon: str) -> Tuple[float, int]:
    rows: List[float] = []
    for _, g in df.groupby("date", sort=False):
        x = g[[factor, horizon]].dropna().copy()
        if len(x) < 20:
            continue
        ic = x[factor].corr(x[horizon], method="spearman")
        rows.append(ic)
    if not rows:
        return float("nan"), 0
    return float(np.nanmean(rows)), int(len(rows))


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main() -> int:
    cfg = load_config()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    print("[LOAD] dataset")
    df = load_dataset(cfg)
    print("[ROWS]", len(df))

    print("[FACTORS] building huge factor universe")
    df = add_factors(df)
    df = add_forward_returns(df)

    factors = [
        "ret1", "ret2", "ret3", "ret5", "ret10", "ret20",
        "open_close_ret", "gap_ret", "high_close_ret", "close_low_ret",
        "range", "body", "upper_wick", "lower_wick", "wick_imbalance", "body_to_range",
        "clv", "pressure",
        "mom3", "mom5", "mom10", "mom20",
        "rev1", "rev2", "rev3", "rev5",
        "vol5", "vol10", "vol20", "vol40",
        "vol_spike_5_20", "vol_spike_10_40", "vol_comp_10_20", "vol_comp_5_40", "range_comp_5_20",
        "rel_volume_5", "rel_volume_20", "rel_volume_60", "vol_accel", "vol_persist_3", "vol_persist_5",
        "dollar_vol", "liq", "liq_shock", "amihud_proxy", "amihud_shock",
        "dist_ema10", "dist_ema20", "dist_ema50", "ema_stack_10_20", "ema_stack_20_50", "ema_slope_10", "ema_slope_20",
        "dist_high_20", "dist_low_20", "dist_high_60", "dist_low_60", "position_in_20d_range", "position_in_60d_range",
        "accel_3_1", "accel_5_3", "accel_10_5",
        "gap_vol_adj", "gap_range_adj", "gap_fill_pressure",
        "range_vol", "body_vol", "rev1_x_vol", "mom3_x_vol", "rev1_x_volcomp", "mom3_x_volcomp", "range_x_volcomp", "gap_x_relvol", "pressure_x_relvol",
        "ret_sign_imbalance_5", "ret_sign_imbalance_10", "ret_autocorr_10", "ret_autocorr_20",
        "breadth_up", "breadth_down", "breadth_thrust",
        "cs_ret_z", "cs_gap_z", "cs_range_z", "cs_relvol_z", "cs_liq_z", "cs_volshock_z", "cs_mom5_rank", "cs_rev1_rank",
        "liq_imbalance", "turnover_imbalance", "amihud_imbalance",
        "overnight_pressure", "intraday_reversal_pressure", "exhaustion_up", "exhaustion_down",
    ]

    horizons = ["fwd_1d", "fwd_2d", "fwd_3d", "fwd_5d"]
    rows: List[Dict[str, object]] = []

    for f in factors:
        for h in horizons:
            ic, n_days = factor_ic(df, f, h)
            rows.append({"factor": f, "horizon": h, "ic": ic, "n_days": n_days})

    res = pd.DataFrame(rows)
    res["abs_ic"] = res["ic"].abs()
    res = res.sort_values(["abs_ic", "ic"], ascending=[False, False]).reset_index(drop=True)

    print("\n[FACTOR IC TOP 150]")
    print(res.head(150).to_string(index=False))

    _ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    out = _ARTIFACT_DIR / "factor_ic_table.csv"
    res.to_csv(out, index=False)

    print("\n[ARTIFACT]", out)
    print(f"[SUMMARY] tested_factors={len(factors)} tested_rows={len(res)}")

    _log("DONE")
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 1
    _press_enter_exit(rc)
