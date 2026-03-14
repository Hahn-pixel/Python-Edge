# scripts/build_feature_matrix.py
# Build a stable feature matrix parquet from massive daily aggregates.
# Double-click runnable. Never auto-closes.
#
# Outputs:
# - data/features/feature_matrix_v1.parquet
# - data/features/feature_matrix_v1.csv
# - data/features/feature_matrix_v1.meta.json
#
# Uses src/python_edge/features/build_features_daily.py as the canonical
# feature builder, then adds forward-return targets and explicit debug stats.

from __future__ import annotations

import json
import os
import sys
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.python_edge.features.build_features_daily import FeatureConfig, build_features_daily

EPS = 1e-12
DEFAULT_DATASET_ROOT = Path(r"D:\massive_dataset")
DEFAULT_OUT_DIR = ROOT / "data" / "features"
DEFAULT_OUT_PARQUET = DEFAULT_OUT_DIR / "feature_matrix_v1.parquet"
DEFAULT_OUT_CSV = DEFAULT_OUT_DIR / "feature_matrix_v1.csv"
DEFAULT_OUT_META = DEFAULT_OUT_DIR / "feature_matrix_v1.meta.json"


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
    mode = str(os.getenv("PAUSE_ON_EXIT", "auto")).strip().lower()
    if mode in {"0", "false", "no", "off"}:
        return False
    if mode in {"1", "true", "yes", "on"}:
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
# ENV
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

class Config:
    def __init__(self) -> None:
        self.dataset_root = Path(_env_str("DATASET_ROOT", str(DEFAULT_DATASET_ROOT)))
        self.start = _env_str("START", "2023-01-01")
        self.end = _env_str("END", "2026-02-28")
        self.out_dir = Path(_env_str("FEATURE_OUT_DIR", str(DEFAULT_OUT_DIR)))
        self.out_parquet = Path(_env_str("FEATURE_OUT_PARQUET", str(DEFAULT_OUT_PARQUET)))
        self.out_csv = Path(_env_str("FEATURE_OUT_CSV", str(DEFAULT_OUT_CSV)))
        self.out_meta = Path(_env_str("FEATURE_OUT_META", str(DEFAULT_OUT_META)))
        self.csv_max_rows = _env_int("FEATURE_CSV_MAX_ROWS", 250000)
        self.min_rows_per_symbol = _env_int("MIN_ROWS_PER_SYMBOL", 40)
        self.mom_lookbacks = tuple(int(x) for x in _env_str("MOM_LOOKBACKS", "1,3,5").split(",") if str(x).strip())
        self.rv_lookback = _env_int("RV_LOOKBACK", 10)
        self.atr_lookback = _env_int("ATR_LOOKBACK", 14)
        self.ema_fast = _env_int("EMA_FAST", 10)
        self.ema_slow = _env_int("EMA_SLOW", 30)
        self.target_horizons = tuple(int(x) for x in _env_str("TARGET_HORIZONS", "1,2,3,5").split(",") if str(x).strip())
        if not self.target_horizons:
            raise RuntimeError("TARGET_HORIZONS resolved to empty list")


# ------------------------------------------------------------
# LOAD RAW massive DATA
# ------------------------------------------------------------

def _discover_symbol_files(dataset_root: Path) -> List[Tuple[str, Path]]:
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset_root not found: {dataset_root}")
    found: List[Tuple[str, Path]] = []
    for sym_dir in sorted(dataset_root.iterdir()):
        if not sym_dir.is_dir():
            continue
        symbol = sym_dir.name.upper()
        files = list(sym_dir.glob("aggs_1d_*.json"))
        if not files:
            continue
        best = max(files, key=lambda p: p.stat().st_size)
        found.append((symbol, best))
    return found



def _load_one_symbol(symbol: str, path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    rows = payload.get("results") or []
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "t" not in df.columns:
        return pd.DataFrame()

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_localize(None).dt.normalize()
    out["symbol"] = symbol
    out["open"] = pd.to_numeric(df.get("o"), errors="coerce")
    out["high"] = pd.to_numeric(df.get("h"), errors="coerce")
    out["low"] = pd.to_numeric(df.get("l"), errors="coerce")
    out["close"] = pd.to_numeric(df.get("c"), errors="coerce")
    out["volume"] = pd.to_numeric(df.get("v"), errors="coerce")
    out["vwap"] = pd.to_numeric(df.get("vw"), errors="coerce") if "vw" in df.columns else np.nan
    out["n_trades"] = pd.to_numeric(df.get("n"), errors="coerce") if "n" in df.columns else np.nan
    return out



def load_massive_daily(cfg: Config) -> Tuple[pd.DataFrame, Dict[str, int]]:
    files = _discover_symbol_files(cfg.dataset_root)
    if not files:
        raise RuntimeError(f"No aggs_1d_*.json found under {cfg.dataset_root}")

    dfs: List[pd.DataFrame] = []
    dbg = {
        "symbols_seen": 0,
        "symbols_loaded": 0,
        "symbols_empty": 0,
        "symbols_too_short": 0,
    }

    for symbol, path in files:
        dbg["symbols_seen"] += 1
        df = _load_one_symbol(symbol, path)
        if df.empty:
            dbg["symbols_empty"] += 1
            continue
        df = df.sort_values("date").reset_index(drop=True)
        if len(df) < cfg.min_rows_per_symbol:
            dbg["symbols_too_short"] += 1
            continue
        dfs.append(df)
        dbg["symbols_loaded"] += 1

    if not dfs:
        raise RuntimeError("No usable symbol data loaded from massive dataset")

    out = pd.concat(dfs, ignore_index=True)
    out = out.sort_values(["symbol", "date"]).reset_index(drop=True)
    if cfg.start:
        out = out.loc[out["date"] >= pd.Timestamp(cfg.start)].copy()
    if cfg.end:
        out = out.loc[out["date"] <= pd.Timestamp(cfg.end)].copy()
    if out.empty:
        raise RuntimeError("Raw daily frame empty after START/END filter")

    return out, dbg


# ------------------------------------------------------------
# TARGETS / EXTRA FEATURES
# ------------------------------------------------------------

def add_targets(df: pd.DataFrame, horizons: Tuple[int, ...]) -> pd.DataFrame:
    out = df.copy()
    next_open = out.groupby("symbol", sort=False)["open"].shift(-1)
    for h in horizons:
        exit_close = out.groupby("symbol", sort=False)["close"].shift(-h)
        out[f"target_fwd_ret_{h}d"] = (exit_close / (next_open + EPS)) - 1.0
    return out



def add_extra_daily_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ret_1d_simple"] = out.groupby("symbol", sort=False)["close"].pct_change()
    out["gap_ret"] = (out["open"] / (out.groupby("symbol", sort=False)["close"].shift(1) + EPS)) - 1.0
    out["hl_range_pct"] = (out["high"] - out["low"]) / (out["close"] + EPS)
    out["oc_body_pct"] = (out["close"] - out["open"]) / (out["open"] + EPS)
    out["dollar_vol"] = out["close"] * out["volume"]
    out["liq"] = np.log1p(out["dollar_vol"].clip(lower=0.0))
    return out


# ------------------------------------------------------------
# DIAGNOSTICS
# ------------------------------------------------------------

def _nan_ratio(series: pd.Series) -> float:
    if len(series) == 0:
        return float("nan")
    return float(series.isna().mean())



def build_debug_summary(df: pd.DataFrame) -> Dict[str, object]:
    target_cols = [c for c in df.columns if c.startswith("target_fwd_ret_")]
    feature_cols = [
        c for c in df.columns
        if c not in {"date", "symbol"} and not c.startswith("target_fwd_ret_")
    ]
    nan_ratios = {
        c: round(_nan_ratio(df[c]), 6)
        for c in feature_cols[:200]
    }
    top_nan = sorted(nan_ratios.items(), key=lambda kv: (-(kv[1] if kv[1] == kv[1] else -1.0), kv[0]))[:30]
    per_symbol = df.groupby("symbol", sort=False).size()
    per_date = df.groupby("date", sort=False).size()
    return {
        "rows": int(len(df)),
        "symbols": int(df["symbol"].nunique()),
        "dates": int(df["date"].nunique()),
        "feature_cols": int(len(feature_cols)),
        "target_cols": int(len(target_cols)),
        "rows_per_symbol_min": int(per_symbol.min()) if not per_symbol.empty else 0,
        "rows_per_symbol_median": float(per_symbol.median()) if not per_symbol.empty else 0.0,
        "rows_per_symbol_max": int(per_symbol.max()) if not per_symbol.empty else 0,
        "rows_per_date_min": int(per_date.min()) if not per_date.empty else 0,
        "rows_per_date_median": float(per_date.median()) if not per_date.empty else 0.0,
        "rows_per_date_max": int(per_date.max()) if not per_date.empty else 0,
        "top_nan_ratio_features": top_nan,
    }


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main() -> int:
    _enable_line_buffering()
    cfg = Config()

    print(f"[CFG] dataset_root={cfg.dataset_root}")
    print(f"[CFG] start={cfg.start} end={cfg.end}")
    print(f"[CFG] out_parquet={cfg.out_parquet}")
    print(f"[CFG] target_horizons={cfg.target_horizons}")
    print(
        "[CFG] feature_builder="
        f"mom={cfg.mom_lookbacks} rv={cfg.rv_lookback} atr={cfg.atr_lookback} "
        f"ema_fast={cfg.ema_fast} ema_slow={cfg.ema_slow}"
    )

    raw_df, load_dbg = load_massive_daily(cfg)
    print(
        "[LOAD] "
        f"rows={len(raw_df)} symbols={raw_df['symbol'].nunique()} dates={raw_df['date'].nunique()} "
        f"loaded_symbols={load_dbg['symbols_loaded']}"
    )

    feat_cfg = FeatureConfig(
        mom_lookbacks=list(cfg.mom_lookbacks),
        rv_lookback=cfg.rv_lookback,
        atr_lookback=cfg.atr_lookback,
        ema_fast=cfg.ema_fast,
        ema_slow=cfg.ema_slow,
    )

    feat_input = raw_df[["date", "symbol", "close"]].copy()
    feat_df = build_features_daily(feat_input, feat_cfg)
    print(f"[FEATURES] rows={len(feat_df)} cols={len(feat_df.columns)}")

    merged = raw_df.merge(feat_df, on=["date", "symbol", "close"], how="left", suffixes=("", ""))
    merged = add_extra_daily_fields(merged)
    merged = add_targets(merged, cfg.target_horizons)
    merged = merged.sort_values(["date", "symbol"]).reset_index(drop=True)

    debug_summary = build_debug_summary(merged)
    print(
        "[DEBUG] "
        f"rows={debug_summary['rows']} symbols={debug_summary['symbols']} dates={debug_summary['dates']} "
        f"feature_cols={debug_summary['feature_cols']} target_cols={debug_summary['target_cols']}"
    )

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(cfg.out_parquet, index=False)

    csv_rows = min(len(merged), cfg.csv_max_rows)
    merged.head(csv_rows).to_csv(cfg.out_csv, index=False)

    meta = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_root": str(cfg.dataset_root),
        "start": cfg.start,
        "end": cfg.end,
        "out_parquet": str(cfg.out_parquet),
        "out_csv": str(cfg.out_csv),
        "out_meta": str(cfg.out_meta),
        "csv_max_rows": cfg.csv_max_rows,
        "min_rows_per_symbol": cfg.min_rows_per_symbol,
        "target_horizons": list(cfg.target_horizons),
        "feature_config": asdict(feat_cfg),
        "load_debug": load_dbg,
        "debug_summary": debug_summary,
        "columns": list(merged.columns),
    }
    cfg.out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ARTIFACT] {cfg.out_parquet}")
    print(f"[ARTIFACT] {cfg.out_csv}")
    print(f"[ARTIFACT] {cfg.out_meta}")
    print("[FINAL] feature_matrix build complete")
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 1
    _press_enter_exit(rc)
