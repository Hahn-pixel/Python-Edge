# scripts/run_regime_interaction_scan.py
# Regime-conditioned interaction scan on top of feature_matrix_v1.parquet.
# Double-click runnable. Never auto-closes.
#
# This is the parquet-native follow-up step after build_feature_matrix.py.
# It does NOT rebuild the factor universe from raw massive JSON each run.
#
# Outputs:
# - artifacts/regime_interaction_scan/regime_variant_ic_table.csv
# - artifacts/regime_interaction_scan/regime_summary_table.csv
# - artifacts/regime_interaction_scan/regime_retained_candidates.csv
# - artifacts/regime_interaction_scan/regime_scan_meta.json

from __future__ import annotations

import json
import os
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

EPS = 1e-12
ROOT = Path(__file__).resolve().parents[1]
FEATURE_FILE_DEFAULT = ROOT / "data" / "features" / "feature_matrix_v1.parquet"
OUT_DIR_DEFAULT = ROOT / "artifacts" / "regime_interaction_scan"

DEFAULT_SCAN_SPECS: Tuple[Tuple[str, str], ...] = (
    ("ret_1d_simple", "liq"),
    ("ret_1d_simple", "dollar_vol"),
    ("gap_ret", "dollar_vol"),
    ("gap_ret", "liq"),
    ("oc_body_pct", "liq"),
    ("oc_body_pct", "hl_range_pct"),
)

FACTOR_ALIASES: Dict[str, Tuple[str, ...]] = {
    "rev1": ("intraday_rs", "ret_1d_simple", "ret1"),
    "mom3": ("mom_3d", "mom3", "ret_3d"),
    "gap_ret": ("gap_ret",),
    "pressure": ("intraday_pressure", "oc_body_pct", "pressure"),
    "wick_imbalance": ("wick_imbalance", "oc_body_pct"),
    "dist_high_20": ("dist_high_20",),
    "dist_low_20": ("dist_low_20",),
    "rel_volume_20": ("volume_shock", "rel_volume_20", "dollar_vol"),
    "vol_spike_5_20": ("vol_spike_5_20", "rv_10d", "atr_14d"),
    "liq": ("liq", "dollar_vol"),
    "cs_liq_z": ("z_liq", "liq", "dollar_vol"),
    "range_comp_5_20": ("compression_10d", "hl_range_pct", "atr_14d"),
}


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
# ENV / CONFIG
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



def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None or not str(v).strip():
        return default
    try:
        return float(str(v).strip())
    except Exception:
        return default



def _parse_scan_specs(text: str) -> Tuple[Tuple[str, str], ...]:
    raw = text.strip()
    if not raw:
        return DEFAULT_SCAN_SPECS
    specs: List[Tuple[str, str]] = []
    for token in raw.split(";"):
        piece = token.strip()
        if not piece:
            continue
        if ":" not in piece:
            raise RuntimeError(
                "Invalid REGIME_SCAN_SPECS token. Use 'factor:regime;factor:regime'. "
                f"Bad token: {piece}"
            )
        left, right = piece.split(":", 1)
        factor = left.strip()
        regime = right.strip()
        if not factor or not regime:
            raise RuntimeError(f"Invalid REGIME_SCAN_SPECS token: {piece}")
        specs.append((factor, regime))
    if not specs:
        raise RuntimeError("REGIME_SCAN_SPECS resolved to empty list")
    return tuple(specs)


@dataclass(frozen=True)
class Config:
    feature_file: Path
    out_dir: Path
    start: str
    end: str
    horizons: Tuple[int, ...]
    high_q: float
    low_q: float
    min_cross_section: int
    min_ic_days: int
    min_abs_uplift: float
    min_rel_uplift: float
    winsor_lower_q: float
    winsor_upper_q: float
    scan_specs: Tuple[Tuple[str, str], ...]



def load_config() -> Config:
    high_q = _env_float("REGIME_HIGH_Q", 0.70)
    low_q = _env_float("REGIME_LOW_Q", 0.30)
    if not (0.0 < low_q < high_q < 1.0):
        raise RuntimeError(
            f"Invalid regime quantiles: low={low_q} high={high_q}. Expected 0 < low < high < 1"
        )
    return Config(
        feature_file=Path(_env_str("FEATURE_FILE", str(FEATURE_FILE_DEFAULT))),
        out_dir=Path(_env_str("REGIME_OUT_DIR", str(OUT_DIR_DEFAULT))),
        start=_env_str("START", "2023-01-01"),
        end=_env_str("END", "2026-02-28"),
        horizons=tuple(int(x) for x in _env_str("EDGE_HORIZONS", "1,2,3,5").split(",") if str(x).strip()),
        high_q=high_q,
        low_q=low_q,
        min_cross_section=_env_int("MIN_CROSS_SECTION", 20),
        min_ic_days=_env_int("MIN_IC_DAYS", 30),
        min_abs_uplift=_env_float("MIN_ABS_UPLIFT", 0.0020),
        min_rel_uplift=_env_float("MIN_REL_UPLIFT", 0.15),
        winsor_lower_q=_env_float("WINSOR_LOWER_Q", 0.02),
        winsor_upper_q=_env_float("WINSOR_UPPER_Q", 0.98),
        scan_specs=_parse_scan_specs(_env_str("REGIME_SCAN_SPECS", "rev1:cs_liq_z;rev1:rel_volume_20;mom3:vol_spike_5_20;gap_ret:rel_volume_20;pressure:liq;wick_imbalance:range_comp_5_20;dist_high_20:rel_volume_20;dist_low_20:rel_volume_20")),
    )


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def _safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("float64")



def _winsorize_series(x: pd.Series, lower_q: float, upper_q: float) -> pd.Series:
    s = _safe_numeric(x)
    valid = s.dropna()
    if valid.empty:
        return s
    lo = float(valid.quantile(lower_q))
    hi = float(valid.quantile(upper_q))
    return s.clip(lower=lo, upper=hi)



def _robust_zscore(x: pd.Series) -> pd.Series:
    s = _safe_numeric(x)
    valid = s.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=s.index, dtype="float64")
    med = float(valid.median())
    mad = float((valid - med).abs().median())
    if mad > EPS:
        return (s - med) / (1.4826 * mad)
    mean = float(valid.mean())
    std = float(valid.std(ddof=0))
    if std > EPS:
        return (s - mean) / std
    return pd.Series(0.0, index=s.index, dtype="float64")



def _rank_pct(x: pd.Series) -> pd.Series:
    return _safe_numeric(x).rank(method="average", pct=True)



def _resolve_column(df: pd.DataFrame, name: str) -> str:
    if name in df.columns:
        return name
    aliases = FACTOR_ALIASES.get(name, (name,))
    for alias in aliases:
        if alias in df.columns:
            return alias
        z_alias = f"z_{alias}"
        if z_alias in df.columns:
            return z_alias
    raise RuntimeError(
        f"Could not resolve column '{name}'. Available aliases tried: {list(aliases)}"
    )


def _ensure_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def has_any(names: Sequence[str]) -> bool:
        return any((c in out.columns) for c in names)

    if "ret_1d_simple" not in out.columns and "close" in out.columns:
        out["ret_1d_simple"] = pd.to_numeric(out.groupby("symbol", sort=False)["close"].pct_change(), errors="coerce")

    if "gap_ret" not in out.columns and all(c in out.columns for c in ["open", "close"]):
        prev_close = out.groupby("symbol", sort=False)["close"].shift(1)
        out["gap_ret"] = (pd.to_numeric(out["open"], errors="coerce") / (pd.to_numeric(prev_close, errors="coerce") + EPS)) - 1.0

    if "hl_range_pct" not in out.columns and all(c in out.columns for c in ["high", "low", "close"]):
        out["hl_range_pct"] = (
            (pd.to_numeric(out["high"], errors="coerce") - pd.to_numeric(out["low"], errors="coerce"))
            / (pd.to_numeric(out["close"], errors="coerce") + EPS)
        )

    if "oc_body_pct" not in out.columns and all(c in out.columns for c in ["open", "close"]):
        out["oc_body_pct"] = (
            (pd.to_numeric(out["close"], errors="coerce") - pd.to_numeric(out["open"], errors="coerce"))
            / (pd.to_numeric(out["open"], errors="coerce") + EPS)
        )

    if "dollar_vol" not in out.columns and all(c in out.columns for c in ["close", "volume"]):
        out["dollar_vol"] = pd.to_numeric(out["close"], errors="coerce") * pd.to_numeric(out["volume"], errors="coerce")

    if "liq" not in out.columns and "dollar_vol" in out.columns:
        out["liq"] = np.log1p(pd.to_numeric(out["dollar_vol"], errors="coerce").clip(lower=0.0))

    if "mom_3d" not in out.columns:
        if "close" in out.columns:
            out["mom_3d"] = pd.to_numeric(out.groupby("symbol", sort=False)["close"].pct_change(3), errors="coerce")
        elif "ret_1d_simple" in out.columns:
            out["mom_3d"] = pd.to_numeric(
                out.groupby("symbol", sort=False)["ret_1d_simple"].rolling(3, min_periods=3).sum().reset_index(level=0, drop=True),
                errors="coerce",
            )

    if "rv_10d" not in out.columns and has_any(["ret_1d_simple"]):
        out["rv_10d"] = pd.to_numeric(
            out.groupby("symbol", sort=False)["ret_1d_simple"].rolling(10, min_periods=5).std().reset_index(level=0, drop=True),
            errors="coerce",
        )

    if "rv_5d" not in out.columns and has_any(["ret_1d_simple"]):
        out["rv_5d"] = pd.to_numeric(
            out.groupby("symbol", sort=False)["ret_1d_simple"].rolling(5, min_periods=3).std().reset_index(level=0, drop=True),
            errors="coerce",
        )

    if "vol_spike_5_20" not in out.columns:
        if "rv_5d" not in out.columns and has_any(["ret_1d_simple"]):
            out["rv_5d"] = pd.to_numeric(
                out.groupby("symbol", sort=False)["ret_1d_simple"].rolling(5, min_periods=3).std().reset_index(level=0, drop=True),
                errors="coerce",
            )
        if "rv_20d" not in out.columns and has_any(["ret_1d_simple"]):
            out["rv_20d"] = pd.to_numeric(
                out.groupby("symbol", sort=False)["ret_1d_simple"].rolling(20, min_periods=10).std().reset_index(level=0, drop=True),
                errors="coerce",
            )
        if all(c in out.columns for c in ["rv_5d", "rv_20d"]):
            out["vol_spike_5_20"] = pd.to_numeric(out["rv_5d"], errors="coerce") / (pd.to_numeric(out["rv_20d"], errors="coerce") + EPS)

    if "atr_14d" not in out.columns and all(c in out.columns for c in ["high", "low", "close"]):
        prev_close = out.groupby("symbol", sort=False)["close"].shift(1)
        tr = pd.concat([
            (pd.to_numeric(out["high"], errors="coerce") - pd.to_numeric(out["low"], errors="coerce")).abs(),
            (pd.to_numeric(out["high"], errors="coerce") - pd.to_numeric(prev_close, errors="coerce")).abs(),
            (pd.to_numeric(out["low"], errors="coerce") - pd.to_numeric(prev_close, errors="coerce")).abs(),
        ], axis=1).max(axis=1)
        out["atr_14d"] = pd.to_numeric(
            tr.groupby(out["symbol"], sort=False).rolling(14, min_periods=7).mean().reset_index(level=0, drop=True),
            errors="coerce",
        )

    if "compression_10d" not in out.columns and has_any(["hl_range_pct"]):
        r10 = pd.to_numeric(
            out.groupby("symbol", sort=False)["hl_range_pct"].rolling(10, min_periods=5).mean().reset_index(level=0, drop=True),
            errors="coerce",
        )
        r20 = pd.to_numeric(
            out.groupby("symbol", sort=False)["hl_range_pct"].rolling(20, min_periods=10).mean().reset_index(level=0, drop=True),
            errors="coerce",
        )
        out["compression_10d"] = r10 / (r20 + EPS)

    if "range_comp_5_20" not in out.columns and has_any(["hl_range_pct"]):
        r5 = pd.to_numeric(
            out.groupby("symbol", sort=False)["hl_range_pct"].rolling(5, min_periods=3).mean().reset_index(level=0, drop=True),
            errors="coerce",
        )
        r20 = pd.to_numeric(
            out.groupby("symbol", sort=False)["hl_range_pct"].rolling(20, min_periods=10).mean().reset_index(level=0, drop=True),
            errors="coerce",
        )
        out["range_comp_5_20"] = r5 / (r20 + EPS)

    if "wick_imbalance" not in out.columns and all(c in out.columns for c in ["open", "high", "low", "close"]):
        upper = (pd.to_numeric(out["high"], errors="coerce") - pd.concat([pd.to_numeric(out["open"], errors="coerce"), pd.to_numeric(out["close"], errors="coerce")], axis=1).max(axis=1)) / (pd.to_numeric(out["close"], errors="coerce") + EPS)
        lower = (pd.concat([pd.to_numeric(out["open"], errors="coerce"), pd.to_numeric(out["close"], errors="coerce")], axis=1).min(axis=1) - pd.to_numeric(out["low"], errors="coerce")) / (pd.to_numeric(out["close"], errors="coerce") + EPS)
        out["wick_imbalance"] = upper - lower

    if "dist_high_20" not in out.columns and "close" in out.columns:
        roll_max_20 = out.groupby("symbol", sort=False)["close"].rolling(20, min_periods=10).max().reset_index(level=0, drop=True)
        out["dist_high_20"] = (pd.to_numeric(out["close"], errors="coerce") / (pd.to_numeric(roll_max_20, errors="coerce") + EPS)) - 1.0

    if "dist_low_20" not in out.columns and "close" in out.columns:
        roll_min_20 = out.groupby("symbol", sort=False)["close"].rolling(20, min_periods=10).min().reset_index(level=0, drop=True)
        out["dist_low_20"] = (pd.to_numeric(out["close"], errors="coerce") / (pd.to_numeric(roll_min_20, errors="coerce") + EPS)) - 1.0

    if "volume_shock" not in out.columns and "volume" in out.columns:
        vol_mean_20 = out.groupby("symbol", sort=False)["volume"].rolling(20, min_periods=10).mean().reset_index(level=0, drop=True)
        out["volume_shock"] = pd.to_numeric(out["volume"], errors="coerce") / (pd.to_numeric(vol_mean_20, errors="coerce") + EPS)

    if "rel_volume_20" not in out.columns and "volume_shock" in out.columns:
        out["rel_volume_20"] = pd.to_numeric(out["volume_shock"], errors="coerce")

    if "z_liq" not in out.columns and "liq" in out.columns:
        out["z_liq"] = pd.to_numeric(out.groupby("date", sort=False)["liq"].transform(_robust_zscore), errors="coerce")

    return out


# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------

def load_feature_matrix(cfg: Config) -> pd.DataFrame:
    if not cfg.feature_file.exists():
        raise FileNotFoundError(f"Feature file not found: {cfg.feature_file}")
    df = pd.read_parquet(cfg.feature_file)
    if df.empty:
        raise RuntimeError("Feature matrix is empty")
    if "symbol" not in df.columns:
        raise RuntimeError("Feature matrix missing symbol column")
    if "date" not in df.columns:
        raise RuntimeError("Feature matrix missing date column")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    if cfg.start:
        df = df.loc[df["date"] >= pd.Timestamp(cfg.start)].copy()
    if cfg.end:
        df = df.loc[df["date"] <= pd.Timestamp(cfg.end)].copy()
    if df.empty:
        raise RuntimeError("Feature matrix empty after START/END filter")
    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)

    required_targets = [f"target_fwd_ret_{h}d" for h in cfg.horizons]
    missing = [c for c in required_targets if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing target columns in feature matrix: {missing}")
    return df


# ------------------------------------------------------------
# FEATURE CONSTRUCTION
# ------------------------------------------------------------

def build_regime_conditioned_features(
    df: pd.DataFrame,
    factor_col: str,
    regime_col: str,
    cfg: Config,
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["raw"] = np.nan
    out["high"] = np.nan
    out["low"] = np.nan
    out["rank"] = np.nan
    out["z"] = np.nan
    out["dbg_mask_high"] = np.nan
    out["dbg_mask_low"] = np.nan

    for _, idx in df.groupby("date", sort=False).groups.items():
        factor_base = _winsorize_series(df.loc[idx, factor_col], cfg.winsor_lower_q, cfg.winsor_upper_q)
        factor_norm = _robust_zscore(factor_base)

        regime_base = _winsorize_series(df.loc[idx, regime_col], cfg.winsor_lower_q, cfg.winsor_upper_q)
        regime_rank = _rank_pct(regime_base)
        regime_rank_centered = regime_rank - 0.5
        regime_z = _robust_zscore(regime_base)
        mask_high = (regime_rank >= cfg.high_q).astype("float64")
        mask_low = (regime_rank <= cfg.low_q).astype("float64")

        out.loc[idx, "raw"] = factor_norm.to_numpy()
        out.loc[idx, "high"] = (factor_norm * mask_high).to_numpy()
        out.loc[idx, "low"] = (factor_norm * mask_low).to_numpy()
        out.loc[idx, "rank"] = (factor_norm * regime_rank_centered).to_numpy()
        out.loc[idx, "z"] = (factor_norm * regime_z).to_numpy()
        out.loc[idx, "dbg_mask_high"] = mask_high.to_numpy()
        out.loc[idx, "dbg_mask_low"] = mask_low.to_numpy()

    return out


# ------------------------------------------------------------
# IC EVAL
# ------------------------------------------------------------

def factor_ic(frame: pd.DataFrame, factor_col: str, target_col: str, min_cross_section: int) -> Tuple[float, int, Dict[str, int]]:
    vals: List[float] = []
    dbg = {
        "dates_total": 0,
        "dates_too_small": 0,
        "dates_const_factor": 0,
        "dates_const_target": 0,
        "dates_ok": 0,
    }
    for _, g in frame.groupby("date", sort=False):
        dbg["dates_total"] += 1
        x = g[[factor_col, target_col]].dropna()
        if len(x) < min_cross_section:
            dbg["dates_too_small"] += 1
            continue
        if x[factor_col].nunique(dropna=True) <= 1 or float(x[factor_col].std(ddof=0)) <= EPS:
            dbg["dates_const_factor"] += 1
            continue
        if x[target_col].nunique(dropna=True) <= 1 or float(x[target_col].std(ddof=0)) <= EPS:
            dbg["dates_const_target"] += 1
            continue
        ic = x[factor_col].corr(x[target_col], method="spearman")
        if pd.notna(ic):
            vals.append(float(ic))
            dbg["dates_ok"] += 1
    if not vals:
        return float("nan"), 0, dbg
    return float(np.nanmean(vals)), int(len(vals)), dbg


# ------------------------------------------------------------
# SCAN
# ------------------------------------------------------------

def scan_one_pair(
    df: pd.DataFrame,
    factor_name: str,
    regime_name: str,
    cfg: Config,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    factor_col = _resolve_column(df, factor_name)
    regime_col = _resolve_column(df, regime_name)
    built = build_regime_conditioned_features(df, factor_col, regime_col, cfg)

    work = df[["date", "symbol"] + [f"target_fwd_ret_{h}d" for h in cfg.horizons]].copy()
    col_map = {
        "raw": f"rg__{factor_name}__raw",
        "high": f"rg__{factor_name}__in__{regime_name}__high",
        "low": f"rg__{factor_name}__in__{regime_name}__low",
        "rank": f"rg__{factor_name}__x__rank__{regime_name}",
        "z": f"rg__{factor_name}__x__z__{regime_name}",
    }
    for key, out_col in col_map.items():
        work[out_col] = built[key]

    variant_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    for h in cfg.horizons:
        target_col = f"target_fwd_ret_{h}d"
        by_variant: Dict[str, Dict[str, object]] = {}
        for key, col in col_map.items():
            ic, n_days, dbg = factor_ic(work, col, target_col, cfg.min_cross_section)
            row = {
                "factor": factor_name,
                "factor_col": factor_col,
                "regime": regime_name,
                "regime_col": regime_col,
                "variant": key,
                "feature_name": col,
                "horizon": target_col,
                "ic": ic,
                "abs_ic": abs(ic) if pd.notna(ic) else np.nan,
                "n_days": n_days,
                "dates_total": dbg["dates_total"],
                "dates_too_small": dbg["dates_too_small"],
                "dates_const_factor": dbg["dates_const_factor"],
                "dates_const_target": dbg["dates_const_target"],
                "dates_ok": dbg["dates_ok"],
            }
            variant_rows.append(row)
            by_variant[key] = row

        raw_row = by_variant["raw"]
        candidates = [by_variant["high"], by_variant["low"], by_variant["rank"], by_variant["z"]]
        best = max(candidates, key=lambda r: (-1.0 if pd.isna(r["abs_ic"]) else float(r["abs_ic"])))

        raw_abs_ic = float(raw_row["abs_ic"]) if pd.notna(raw_row["abs_ic"]) else np.nan
        best_abs_ic = float(best["abs_ic"]) if pd.notna(best["abs_ic"]) else np.nan
        abs_uplift = best_abs_ic - raw_abs_ic if pd.notna(raw_abs_ic) and pd.notna(best_abs_ic) else np.nan
        rel_uplift = abs_uplift / (raw_abs_ic + EPS) if pd.notna(abs_uplift) and pd.notna(raw_abs_ic) else np.nan
        passed = bool(
            pd.notna(abs_uplift)
            and pd.notna(rel_uplift)
            and int(best["n_days"]) >= cfg.min_ic_days
            and abs_uplift >= cfg.min_abs_uplift
            and rel_uplift >= cfg.min_rel_uplift
        )

        summary_rows.append({
            "factor": factor_name,
            "factor_col": factor_col,
            "regime": regime_name,
            "regime_col": regime_col,
            "horizon": target_col,
            "raw_feature_name": raw_row["feature_name"],
            "raw_ic": raw_row["ic"],
            "raw_abs_ic": raw_abs_ic,
            "raw_n_days": raw_row["n_days"],
            "best_variant": best["variant"],
            "best_feature_name": best["feature_name"],
            "best_ic": best["ic"],
            "best_abs_ic": best_abs_ic,
            "best_n_days": best["n_days"],
            "abs_uplift": abs_uplift,
            "rel_uplift": rel_uplift,
            "pass_regime_uplift": int(passed),
            "threshold_abs_uplift": cfg.min_abs_uplift,
            "threshold_rel_uplift": cfg.min_rel_uplift,
            "threshold_min_ic_days": cfg.min_ic_days,
        })

    return pd.DataFrame(variant_rows), pd.DataFrame(summary_rows)


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main() -> int:
    _enable_line_buffering()
    cfg = load_config()

    print(f"[CFG] feature_file={cfg.feature_file}")
    print(f"[CFG] out_dir={cfg.out_dir}")
    print(f"[CFG] start={cfg.start} end={cfg.end}")
    print(f"[CFG] horizons={cfg.horizons}")
    print(
        "[CFG] regime_quantiles="
        f"low<={cfg.low_q:.2f} high>={cfg.high_q:.2f} "
        f"min_cross_section={cfg.min_cross_section} min_ic_days={cfg.min_ic_days}"
    )
    print(
        "[CFG] uplift_filters="
        f"abs>={cfg.min_abs_uplift:.4f} rel>={cfg.min_rel_uplift:.2f} "
        f"winsor=({cfg.winsor_lower_q:.2f},{cfg.winsor_upper_q:.2f})"
    )
    print(f"[CFG] scan_specs={'; '.join(f'{a}:{b}' for a, b in cfg.scan_specs)}")

    df = load_feature_matrix(cfg)
    df = _ensure_derived_columns(df)
    print(f"[DATA] rows={len(df)} dates={df['date'].nunique()} symbols={df['symbol'].nunique()}")

    all_variant_tables: List[pd.DataFrame] = []
    all_summary_tables: List[pd.DataFrame] = []
    resolved_pairs: List[Dict[str, str]] = []

    for factor_name, regime_name in cfg.scan_specs:
        factor_col = _resolve_column(df, factor_name)
        regime_col = _resolve_column(df, regime_name)
        print(f"[SCAN] factor={factor_name}->{factor_col} regime={regime_name}->{regime_col}")
        variants_df, summary_df = scan_one_pair(df, factor_name, regime_name, cfg)
        all_variant_tables.append(variants_df)
        all_summary_tables.append(summary_df)
        resolved_pairs.append({
            "factor": factor_name,
            "factor_col": factor_col,
            "regime": regime_name,
            "regime_col": regime_col,
        })

    variants = pd.concat(all_variant_tables, ignore_index=True) if all_variant_tables else pd.DataFrame()
    summary = pd.concat(all_summary_tables, ignore_index=True) if all_summary_tables else pd.DataFrame()
    if not summary.empty:
        summary = summary.sort_values(
            ["pass_regime_uplift", "best_abs_ic", "abs_uplift", "rel_uplift"],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)
    kept = summary.loc[summary["pass_regime_uplift"] == 1].copy().reset_index(drop=True) if not summary.empty else pd.DataFrame()

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    p_variants = cfg.out_dir / "regime_variant_ic_table.csv"
    p_summary = cfg.out_dir / "regime_summary_table.csv"
    p_kept = cfg.out_dir / "regime_retained_candidates.csv"
    p_meta = cfg.out_dir / "regime_scan_meta.json"

    variants.to_csv(p_variants, index=False)
    summary.to_csv(p_summary, index=False)
    kept.to_csv(p_kept, index=False)

    meta = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "feature_file": str(cfg.feature_file),
        "out_dir": str(cfg.out_dir),
        "start": cfg.start,
        "end": cfg.end,
        "horizons": list(cfg.horizons),
        "high_q": cfg.high_q,
        "low_q": cfg.low_q,
        "min_cross_section": cfg.min_cross_section,
        "min_ic_days": cfg.min_ic_days,
        "min_abs_uplift": cfg.min_abs_uplift,
        "min_rel_uplift": cfg.min_rel_uplift,
        "winsor_lower_q": cfg.winsor_lower_q,
        "winsor_upper_q": cfg.winsor_upper_q,
        "scan_specs": [{"factor": a, "regime": b} for a, b in cfg.scan_specs],
        "resolved_pairs": resolved_pairs,
        "rows": int(len(df)),
        "symbols": int(df['symbol'].nunique()),
        "dates": int(df['date'].nunique()),
        "variants_rows": int(len(variants)),
        "summary_rows": int(len(summary)),
        "kept_rows": int(len(kept)),
    }
    p_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[SUMMARY] TOP REGIME RESULTS")
    if summary.empty:
        print("No summary rows produced")
    else:
        cols = [
            "factor", "regime", "horizon", "raw_ic", "best_variant", "best_ic",
            "abs_uplift", "rel_uplift", "best_n_days", "pass_regime_uplift",
        ]
        print(summary[cols].head(100).to_string(index=False))

    print("\n[KEPT] REGIME-RETAINED CANDIDATES")
    if kept.empty:
        print("No candidates passed regime uplift filters")
    else:
        cols = [
            "factor", "regime", "horizon", "raw_ic", "best_variant", "best_ic",
            "abs_uplift", "rel_uplift", "best_n_days",
        ]
        print(kept[cols].head(100).to_string(index=False))

    print(f"\n[ARTIFACT] {p_variants}")
    print(f"[ARTIFACT] {p_summary}")
    print(f"[ARTIFACT] {p_kept}")
    print(f"[ARTIFACT] {p_meta}")
    print(f"[FINAL] scan_specs={len(cfg.scan_specs)} summary_rows={len(summary)} kept_rows={len(kept)}")
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 1
    _press_enter_exit(rc)
