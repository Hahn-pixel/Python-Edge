# scripts/run_residual_alpha_diagnostics.py
# Extended alpha diagnostics with MANY experimental factors
# Double-click runnable. Never auto-closes.

from __future__ import annotations

import json
import os
import random
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

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
    out = []
    for sym_dir in root.iterdir():
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

    df["date"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.date.astype(str)
    df["symbol"] = sym

    for c in ["o", "h", "l", "c", "v"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df[["date", "symbol", "o", "h", "l", "c", "v"]]


def load_dataset(cfg: Config) -> pd.DataFrame:

    dfs = []

    for sym, path in _find_files(cfg.dataset_root):

        d = _load_file(sym, path)

        if not d.empty:

            dfs.append(d)

    df = pd.concat(dfs, ignore_index=True)

    df = df[(df.date >= cfg.start) & (df.date <= cfg.end)]

    return df.sort_values(["date", "symbol"]).reset_index(drop=True)


# ------------------------------------------------------------
# FACTORS
# ------------------------------------------------------------

def add_factors(df: pd.DataFrame) -> pd.DataFrame:

    out = df.copy()

    out["ret1"] = out.groupby("symbol")["c"].pct_change()

    # momentum
    out["mom3"] = out.groupby("symbol")["c"].pct_change(3)
    out["mom5"] = out.groupby("symbol")["c"].pct_change(5)
    out["mom10"] = out.groupby("symbol")["c"].pct_change(10)

    # reversal
    out["rev1"] = -out["ret1"]
    out["rev3"] = -out.groupby("symbol")["ret1"].rolling(3).sum().reset_index(level=0, drop=True)

    # volatility
    out["vol10"] = out.groupby("symbol")["ret1"].rolling(10).std().reset_index(level=0, drop=True)
    out["vol20"] = out.groupby("symbol")["ret1"].rolling(20).std().reset_index(level=0, drop=True)

    # volatility compression
    out["vol_comp"] = out["vol10"] / (out["vol20"] + EPS)

    # range factors
    out["range"] = (out["h"] - out["l"]) / (out["c"] + EPS)
    out["range_norm"] = out["range"] / (out.groupby("symbol")["range"].rolling(20).mean().reset_index(level=0, drop=True) + EPS)

    # relative volume
    vol_mean = out.groupby("symbol")["v"].rolling(20).mean().reset_index(level=0, drop=True)

    out["rel_volume"] = out["v"] / (vol_mean + EPS)

    # turnover shock
    out["turnover"] = out["v"] * out["c"]

    turn_mean = out.groupby("symbol")["turnover"].rolling(20).mean().reset_index(level=0, drop=True)

    out["turnover_shock"] = out["turnover"] / (turn_mean + EPS)

    # liquidity proxy
    out["liq"] = np.log(out["turnover"] + 1)

    # price acceleration
    out["accel"] = out.groupby("symbol")["mom3"].diff()

    # gap
    prev_close = out.groupby("symbol")["c"].shift(1)

    out["gap"] = (out["o"] / prev_close) - 1

    # mean reversion pressure
    ema10 = out.groupby("symbol")["c"].transform(lambda x: x.ewm(span=10).mean())

    out["dist_ema"] = (out["c"] - ema10) / (ema10 + EPS)

    # abnormal range + volume
    out["range_vol"] = out["range"] * out["rel_volume"]

    return out


# ------------------------------------------------------------
# FORWARD RETURNS
# ------------------------------------------------------------

def add_forward_returns(df: pd.DataFrame, horizons=(1, 2, 3, 5)) -> pd.DataFrame:

    out = df.copy()

    entry = out.groupby("symbol")["o"].shift(-1)

    for h in horizons:

        exit_px = out.groupby("symbol")["c"].shift(-h)

        out[f"fwd_{h}d"] = (exit_px / entry) - 1

    return out


# ------------------------------------------------------------
# IC TEST
# ------------------------------------------------------------

def factor_ic(df: pd.DataFrame, factor: str, horizon: str) -> float:

    rows = []

    for _, g in df.groupby("date"):

        g = g[[factor, horizon]].dropna()

        if len(g) < 20:
            continue

        ic = g[factor].corr(g[horizon], method="spearman")

        rows.append(ic)

    if not rows:
        return float("nan")

    return float(np.nanmean(rows))


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

    print("[FACTORS]")

    df = add_factors(df)

    df = add_forward_returns(df)

    factors = [
        "mom3",
        "mom5",
        "mom10",
        "rev1",
        "rev3",
        "vol_comp",
        "range",
        "range_norm",
        "rel_volume",
        "turnover_shock",
        "liq",
        "accel",
        "gap",
        "dist_ema",
        "range_vol",
    ]

    horizons = ["fwd_1d", "fwd_2d", "fwd_3d", "fwd_5d"]

    rows = []

    for f in factors:

        for h in horizons:

            ic = factor_ic(df, f, h)

            rows.append({"factor": f, "horizon": h, "ic": ic})

    res = pd.DataFrame(rows)

    print("\n[FACTOR IC]")

    print(res.sort_values("ic", ascending=False).to_string(index=False))

    _ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    out = _ARTIFACT_DIR / "factor_ic_table.csv"

    res.to_csv(out, index=False)

    print("\n[ARTIFACT]", out)

    _log("DONE")

    return 0


if __name__ == "__main__":

    try:

        rc = main()

    except Exception:

        traceback.print_exc()

        rc = 1

    _press_enter_exit(rc)
