# scripts/run_factor_interaction_scan.py
# Factor interaction scan for Python-Edge / massive dataset
# Double-click runnable. Never auto-closes.
#
# Purpose:
# - start from a curated core factor set
# - build pairwise interaction factors
# - measure IC by horizon
# - export ranked interaction table

from __future__ import annotations

import json
import os
import random
import traceback
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[1]
_ARTIFACT_DIR = _REPO_ROOT / "artifacts" / "factor_interaction_scan"

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


@dataclass(frozen=True)
class Config:
    dataset_root: Path
    start: str
    end: str
    seed: int
    top_n_base_factors: int
    horizons: Tuple[int, ...]
    max_pairs: int


def load_config() -> Config:
    return Config(
        dataset_root=Path(_env_str("DATASET_ROOT", r"D:\massive_dataset")),
        start=_env_str("START", "2023-01-01"),
        end=_env_str("END", "2026-02-28"),
        seed=_env_int("SEED", 7),
        top_n_base_factors=_env_int("TOP_N_BASE_FACTORS", 20),
        horizons=tuple(int(x) for x in _env_str("EDGE_HORIZONS", "1,2,3,5").split(",") if str(x).strip()),
        max_pairs=_env_int("MAX_INTERACTION_PAIRS", 200),
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
    prev_high = out.groupby("symbol")["h"].shift(1)
    prev_low = out.groupby("symbol")["l"].shift(1)

    out["ret1"] = out.groupby("symbol")["c"].pct_change(1, fill_method=None)
    out["ret2"] = out.groupby("symbol")["c"].pct_change(2, fill_method=None)
    out["ret3"] = out.groupby("symbol")["c"].pct_change(3, fill_method=None)
    out["ret5"] = out.groupby("symbol")["c"].pct_change(5, fill_method=None)
    out["ret10"] = out.groupby("symbol")["c"].pct_change(10, fill_method=None)

    out["gap_ret"] = (out["o"] / prev_close) - 1.0
    out["range"] = (out["h"] - out["l"]) / (out["c"] + EPS)
    out["body"] = (out["c"] - out["o"]) / (out["o"] + EPS)
    out["upper_wick"] = (out["h"] - out[["o", "c"]].max(axis=1)) / (out["c"] + EPS)
    out["lower_wick"] = (out[["o", "c"]].min(axis=1) - out["l"]) / (out["c"] + EPS)
    out["wick_imbalance"] = out["upper_wick"] - out["lower_wick"]
    out["body_to_range"] = (out["body"] / (out["range"] + EPS)).replace([np.inf, -np.inf], np.nan)
    out["pressure"] = (out["c"] - out["o"]) / ((out["h"] - out["l"]) + EPS)

    out["mom3"] = out["ret3"]
    out["mom5"] = out["ret5"]
    out["mom10"] = out["ret10"]
    out["rev1"] = -out["ret1"]
    out["rev3"] = -out.groupby("symbol")["ret1"].rolling(3, min_periods=2).sum().reset_index(level=0, drop=True)

    out["vol5"] = out.groupby("symbol")["ret1"].rolling(5, min_periods=3).std().reset_index(level=0, drop=True)
    out["vol10"] = out.groupby("symbol")["ret1"].rolling(10, min_periods=5).std().reset_index(level=0, drop=True)
    out["vol20"] = out.groupby("symbol")["ret1"].rolling(20, min_periods=10).std().reset_index(level=0, drop=True)
    out["vol40"] = out.groupby("symbol")["ret1"].rolling(40, min_periods=15).std().reset_index(level=0, drop=True)
    out["vol_spike_5_20"] = out["vol5"] / (out["vol20"] + EPS)
    out["vol_comp_10_20"] = out["vol10"] / (out["vol20"] + EPS)
    out["range_comp_5_20"] = (
        out.groupby("symbol")["range"].rolling(5, min_periods=3).mean().reset_index(level=0, drop=True)
        /
        (out.groupby("symbol")["range"].rolling(20, min_periods=10).mean().reset_index(level=0, drop=True) + EPS)
    )

    out["vol_mean_20"] = out.groupby("symbol")["v"].rolling(20, min_periods=10).mean().reset_index(level=0, drop=True)
    out["rel_volume_20"] = out["v"] / (out["vol_mean_20"] + EPS)
    out["dollar_vol"] = out["v"] * out["c"]
    out["liq"] = np.log(out["dollar_vol"] + 1.0)
    out["dollar_vol_mean_20"] = out.groupby("symbol")["dollar_vol"].rolling(20, min_periods=10).mean().reset_index(level=0, drop=True)
    out["liq_shock"] = out["dollar_vol"] / (out["dollar_vol_mean_20"] + EPS)
    out["amihud_proxy"] = out["ret1"].abs() / (out["dollar_vol"] + EPS)
    out["amihud_shock"] = out["amihud_proxy"] / (out.groupby("symbol")["amihud_proxy"].rolling(20, min_periods=10).mean().reset_index(level=0, drop=True) + EPS)

    ema20 = out.groupby("symbol")["c"].transform(lambda s: s.ewm(span=20, adjust=False, min_periods=7).mean())
    ema50 = out.groupby("symbol")["c"].transform(lambda s: s.ewm(span=50, adjust=False, min_periods=15).mean())
    out["dist_ema20"] = (out["c"] - ema20) / (ema20 + EPS)
    out["ema_stack_20_50"] = (ema20 - ema50) / (ema50 + EPS)

    roll_max_20 = out.groupby("symbol")["c"].rolling(20, min_periods=10).max().reset_index(level=0, drop=True)
    roll_min_20 = out.groupby("symbol")["c"].rolling(20, min_periods=10).min().reset_index(level=0, drop=True)
    roll_max_60 = out.groupby("symbol")["c"].rolling(60, min_periods=20).max().reset_index(level=0, drop=True)
    roll_min_60 = out.groupby("symbol")["c"].rolling(60, min_periods=20).min().reset_index(level=0, drop=True)
    out["dist_high_20"] = (out["c"] / (roll_max_20 + EPS)) - 1.0
    out["dist_low_20"] = (out["c"] / (roll_min_20 + EPS)) - 1.0
    out["dist_high_60"] = (out["c"] / (roll_max_60 + EPS)) - 1.0
    out["dist_low_60"] = (out["c"] / (roll_min_60 + EPS)) - 1.0
    out["position_in_60d_range"] = (out["c"] - roll_min_60) / ((roll_max_60 - roll_min_60) + EPS)

    out["accel_3_1"] = out["mom3"] - out["ret1"]
    out["gap_vol_adj"] = out["gap_ret"] / (out["vol10"] + EPS)
    out["range_vol"] = out["range"] * out["rel_volume_20"]
    out["rev1_x_vol"] = out["rev1"] * out["rel_volume_20"]
    out["mom3_x_vol"] = out["mom3"] * out["rel_volume_20"]
    out["rev1_x_volcomp"] = out["rev1"] * out["vol_comp_10_20"]
    out["mom3_x_volcomp"] = out["mom3"] * out["vol_comp_10_20"]
    out["pressure_x_relvol"] = out["pressure"] * out["rel_volume_20"]

    out["ret_autocorr_10"] = np.nan
    for _, idx in out.groupby("symbol", sort=False).groups.items():
        s = out.loc[idx, "ret1"]
        out.loc[idx, "ret_autocorr_10"] = _rolling_autocorr_1(s, 10).to_numpy()

    out["breadth_up"] = out.groupby("date")["ret1"].transform(lambda s: float((s > 0).mean()))
    out["breadth_down"] = out.groupby("date")["ret1"].transform(lambda s: float((s < 0).mean()))
    out["breadth_thrust"] = out["breadth_up"] - out["breadth_down"]
    out["cs_ret_z"] = out.groupby("date")["ret1"].transform(lambda s: _robust_zscore(s))
    out["cs_gap_z"] = out.groupby("date")["gap_ret"].transform(lambda s: _robust_zscore(s))
    out["cs_range_z"] = out.groupby("date")["range"].transform(lambda s: _robust_zscore(s))
    out["cs_relvol_z"] = out.groupby("date")["rel_volume_20"].transform(lambda s: _robust_zscore(s))
    out["cs_liq_z"] = out.groupby("date")["liq"].transform(lambda s: _robust_zscore(s))
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

def add_forward_returns(df: pd.DataFrame, horizons: Tuple[int, ...]) -> pd.DataFrame:
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
        if x[factor].nunique(dropna=True) <= 1:
            continue
        if x[horizon].nunique(dropna=True) <= 1:
            continue
        if float(x[factor].std(ddof=0)) <= 1e-12:
            continue
        if float(x[horizon].std(ddof=0)) <= 1e-12:
            continue
        ic = x[factor].corr(x[horizon], method="spearman")
        if pd.notna(ic):
            rows.append(float(ic))
    if not rows:
        return float("nan"), 0
    return float(np.nanmean(rows)), int(len(rows))


def select_core_factors(df: pd.DataFrame, candidate_factors: List[str], horizons: List[str], top_n: int) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for f in candidate_factors:
        for h in horizons:
            ic, n_days = factor_ic(df, f, h)
            rows.append({"factor": f, "horizon": h, "ic": ic, "n_days": n_days})
    base = pd.DataFrame(rows)
    base["abs_ic"] = base["ic"].abs()
    best = base.sort_values(["factor", "abs_ic"], ascending=[True, False]).groupby("factor", as_index=False).first()
    best = best.sort_values(["abs_ic", "ic"], ascending=[False, False]).reset_index(drop=True)
    return best.head(top_n)


def build_interaction_feature(df: pd.DataFrame, fa: str, fb: str) -> pd.Series:
    a = _safe(df[fa])
    b = _safe(df[fb])
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    for _, idx in df.groupby("date", sort=False).groups.items():
        az = _robust_zscore(_winsorize_series(a.loc[idx]))
        bz = _robust_zscore(_winsorize_series(b.loc[idx]))
        out.loc[idx] = (az * bz).to_numpy()
    return out


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

    print("[FACTORS] building base universe")
    df = add_factors(df)
    df = add_forward_returns(df, cfg.horizons)

    candidate_factors = [
        "ret1", "ret2", "ret3", "ret5", "ret10",
        "gap_ret", "range", "body", "wick_imbalance", "body_to_range", "pressure",
        "mom3", "mom5", "mom10", "rev1", "rev3",
        "vol5", "vol10", "vol20", "vol40", "vol_spike_5_20", "vol_comp_10_20", "range_comp_5_20",
        "rel_volume_20", "dollar_vol", "liq", "liq_shock", "amihud_proxy", "amihud_shock",
        "dist_ema20", "ema_stack_20_50", "dist_high_20", "dist_low_20", "dist_high_60", "dist_low_60", "position_in_60d_range",
        "accel_3_1", "gap_vol_adj", "range_vol", "rev1_x_vol", "mom3_x_vol", "rev1_x_volcomp", "mom3_x_volcomp", "pressure_x_relvol",
        "ret_autocorr_10", "breadth_thrust", "cs_ret_z", "cs_gap_z", "cs_range_z", "cs_relvol_z", "cs_liq_z",
        "liq_imbalance", "turnover_imbalance", "amihud_imbalance", "overnight_pressure", "intraday_reversal_pressure", "exhaustion_up", "exhaustion_down",
    ]
    horizons = [f"fwd_{h}d" for h in cfg.horizons]

    print("[SCAN] selecting core factors")
    core = select_core_factors(df, candidate_factors, horizons, cfg.top_n_base_factors)
    print("\n[CORE FACTORS]")
    print(core.to_string(index=False))

    core_list = core["factor"].tolist()
    pair_list = list(combinations(core_list, 2))
    if cfg.max_pairs > 0:
        pair_list = pair_list[:cfg.max_pairs]

    print(f"\n[INTERACTIONS] scanning_pairs={len(pair_list)}")
    interaction_rows: List[Dict[str, object]] = []

    for fa, fb in pair_list:
        col = f"ix__{fa}__x__{fb}"
        work = df[["date", "symbol", fa, fb] + horizons].copy()
        work[col] = build_interaction_feature(work, fa, fb)
        for h in horizons:
            ic, n_days = factor_ic(work, col, h)
            interaction_rows.append({
                "interaction": col,
                "factor_a": fa,
                "factor_b": fb,
                "horizon": h,
                "ic": ic,
                "n_days": n_days,
            })

    inter = pd.DataFrame(interaction_rows)
    inter["abs_ic"] = inter["ic"].abs()
    inter = inter.sort_values(["abs_ic", "ic"], ascending=[False, False]).reset_index(drop=True)

    print("\n[INTERACTION IC TOP 150]")
    print(inter.head(150).to_string(index=False))

    _ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    p_core = _ARTIFACT_DIR / "core_factor_table.csv"
    p_inter = _ARTIFACT_DIR / "interaction_ic_table.csv"
    core.to_csv(p_core, index=False)
    inter.to_csv(p_inter, index=False)

    print(f"\n[ARTIFACT] {p_core}")
    print(f"[ARTIFACT] {p_inter}")
    print(f"[SUMMARY] top_n_base_factors={len(core_list)} interaction_pairs={len(pair_list)} interaction_rows={len(inter)}")

    _log("DONE")
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 1
    _press_enter_exit(rc)
