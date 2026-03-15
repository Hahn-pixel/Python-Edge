from __future__ import annotations

import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
for _p in [ROOT, SRC_DIR]:
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

import numpy as np
import pandas as pd

from python_edge.model.conditional_factors import CONDITIONAL_FEATURE_COLS, add_conditional_factors
from python_edge.model.cs_normalize import cs_zscore

EPS = 1e-12
FEATURE_FILE = Path(os.getenv("FEATURE_FILE", r"data\features\feature_matrix_v1.parquet"))
OUT_DIR = Path(os.getenv("ALPHA_LIB_OUT_DIR", r"data\alpha_library"))
OUT_PARQUET = OUT_DIR / "alpha_library_v1.parquet"
OUT_META = OUT_DIR / "alpha_library_v1.meta.json"
OUT_CSV = OUT_DIR / "alpha_library_v1__head.csv"
PAUSE_ON_EXIT_ENV = str(os.getenv("PAUSE_ON_EXIT", "auto")).strip().lower()

REGIME_HIGH_Q = float(os.getenv("REGIME_HIGH_Q", "0.70"))
REGIME_LOW_Q = float(os.getenv("REGIME_LOW_Q", "0.30"))
START = str(os.getenv("START", "2023-01-01")).strip()
END = str(os.getenv("END", "2026-02-28")).strip()
CSV_MAX_ROWS = int(os.getenv("ALPHA_LIB_CSV_MAX_ROWS", "200000"))


@dataclass(frozen=True)
class AlphaSpec:
    name: str
    base_col: str
    regime_col: str


BASE_REGIME_SPECS: tuple[AlphaSpec, ...] = (
    AlphaSpec("rev1__hi_rvol", "rev1_base", "rel_volume_20"),
    AlphaSpec("rev1__lo_rvol", "rev1_base", "rel_volume_20"),
    AlphaSpec("rev1__z_rvol", "rev1_base", "rel_volume_20"),
    AlphaSpec("rev1__rank_rvol", "rev1_base", "rel_volume_20"),
    AlphaSpec("gap__hi_rvol", "gap_base", "rel_volume_20"),
    AlphaSpec("gap__z_rvol", "gap_base", "rel_volume_20"),
    AlphaSpec("gap__rank_rvol", "gap_base", "rel_volume_20"),
    AlphaSpec("pressure__hi_liq", "pressure_base", "liq"),
    AlphaSpec("pressure__z_liq", "pressure_base", "liq"),
    AlphaSpec("pressure__rank_liq", "pressure_base", "liq"),
    AlphaSpec("mom3__hi_rvol", "mom3_base", "rel_volume_20"),
    AlphaSpec("mom3__z_rvol", "mom3_base", "rel_volume_20"),
    AlphaSpec("dist_low_20__hi_rvol", "dist_low_20_base", "rel_volume_20"),
    AlphaSpec("dist_low_20__z_rvol", "dist_low_20_base", "rel_volume_20"),
    AlphaSpec("dist_high_20__hi_rvol", "dist_high_20_base", "rel_volume_20"),
    AlphaSpec("wick_imbalance__hi_range_comp", "wick_imbalance_base", "range_comp_5_20"),
    AlphaSpec("wick_imbalance__z_range_comp", "wick_imbalance_base", "range_comp_5_20"),
    AlphaSpec("rev1__hi_liq", "rev1_base", "liq"),
    AlphaSpec("gap__hi_liq", "gap_base", "liq"),
    AlphaSpec("mom3__hi_liq", "mom3_base", "liq"),
)


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
    if PAUSE_ON_EXIT_ENV in {"0", "false", "no", "off"}:
        return False
    if PAUSE_ON_EXIT_ENV in {"1", "true", "yes", "on"}:
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


def _safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("float64")


def _cs_rank_pct(df: pd.DataFrame, col: str) -> pd.Series:
    return _safe_numeric(df[col]).groupby(df["date"], sort=False).rank(method="average", pct=True)


def _cs_zscore_one(df: pd.DataFrame, col: str) -> pd.Series:
    x = _safe_numeric(df[col])
    grp = x.groupby(df["date"], sort=False)
    mean = grp.transform("mean")
    std = grp.transform("std").replace(0.0, np.nan)
    out = (x - mean) / std
    return out.replace([np.inf, -np.inf], np.nan)


def _series_or_zero(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return _safe_numeric(df[col]).fillna(0.0)
    return pd.Series(0.0, index=df.index, dtype="float64")


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "symbol" not in out.columns:
        if "ticker" in out.columns:
            out = out.rename(columns={"ticker": "symbol"})
        elif "sym" in out.columns:
            out = out.rename(columns={"sym": "symbol"})
    required = ["date", "symbol", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise RuntimeError(f"build_multi_alpha_library: missing required columns: {missing}")
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    if START:
        out = out.loc[out["date"] >= pd.Timestamp(START)].copy()
    if END:
        out = out.loc[out["date"] <= pd.Timestamp(END)].copy()
    if out.empty:
        raise RuntimeError("build_multi_alpha_library: empty after START/END filter")
    out = out.sort_values(["date", "symbol"]).reset_index(drop=True)
    return out


def _derive_base_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    prev_close = _safe_numeric(out.groupby("symbol", sort=False)["close"].shift(1))

    out["ret_1d_simple"] = _safe_numeric(out.groupby("symbol", sort=False)["close"].pct_change())
    out["mom3_raw"] = _safe_numeric(out.groupby("symbol", sort=False)["close"].pct_change(3))
    out["gap_ret"] = (_safe_numeric(out["open"]) / (prev_close + EPS)) - 1.0
    out["hl_range_pct"] = (_safe_numeric(out["high"]) - _safe_numeric(out["low"])) / (_safe_numeric(out["close"]) + EPS)
    out["oc_body_pct"] = (_safe_numeric(out["close"]) - _safe_numeric(out["open"])) / (_safe_numeric(out["open"]) + EPS)
    out["dollar_vol"] = _safe_numeric(out["close"]) * _safe_numeric(out["volume"])
    out["liq"] = np.log1p(_safe_numeric(out["dollar_vol"]).clip(lower=0.0))

    vol_mean_20 = out.groupby("symbol", sort=False)["volume"].rolling(20, min_periods=10).mean().reset_index(level=0, drop=True)
    out["volume_shock"] = _safe_numeric(out["volume"]) / (_safe_numeric(vol_mean_20) + EPS)
    out["rel_volume_20"] = _safe_numeric(out["volume_shock"])

    range_mean_5 = out.groupby("symbol", sort=False)["hl_range_pct"].rolling(5, min_periods=3).mean().reset_index(level=0, drop=True)
    range_mean_20 = out.groupby("symbol", sort=False)["hl_range_pct"].rolling(20, min_periods=10).mean().reset_index(level=0, drop=True)
    out["range_comp_5_20"] = _safe_numeric(range_mean_5) / (_safe_numeric(range_mean_20) + EPS)

    roll_max_20 = out.groupby("symbol", sort=False)["close"].rolling(20, min_periods=10).max().reset_index(level=0, drop=True)
    roll_min_20 = out.groupby("symbol", sort=False)["close"].rolling(20, min_periods=10).min().reset_index(level=0, drop=True)
    out["dist_high_20"] = (_safe_numeric(out["close"]) / (_safe_numeric(roll_max_20) + EPS)) - 1.0
    out["dist_low_20"] = (_safe_numeric(out["close"]) / (_safe_numeric(roll_min_20) + EPS)) - 1.0

    upper = (_safe_numeric(out["high"]) - pd.concat([_safe_numeric(out["open"]), _safe_numeric(out["close"] )], axis=1).max(axis=1)) / (_safe_numeric(out["close"]) + EPS)
    lower = (pd.concat([_safe_numeric(out["open"]), _safe_numeric(out["close"] )], axis=1).min(axis=1) - _safe_numeric(out["low"])) / (_safe_numeric(out["close"]) + EPS)
    out["wick_imbalance"] = upper - lower

    out["market_breadth"] = out.groupby("date", sort=False)["ret_1d_simple"].transform(lambda s: pd.to_numeric(s, errors="coerce").gt(0).mean())
    out["liq_rank"] = _cs_rank_pct(out, "liq")

    out["intraday_rs"] = -_cs_zscore_one(out, "ret_1d_simple")
    out["intraday_pressure"] = _cs_zscore_one(out, "oc_body_pct")

    out["momentum_20d"] = _safe_numeric(out.groupby("symbol", sort=False)["close"].pct_change(20))
    out["str_3d"] = -_safe_numeric(out.groupby("symbol", sort=False)["close"].pct_change(3))
    out["overnight_drift_20d"] = _safe_numeric(out.groupby("symbol", sort=False)["gap_ret"].rolling(20, min_periods=10).mean().reset_index(level=0, drop=True))
    out["ivol_20d"] = _safe_numeric(out.groupby("symbol", sort=False)["ret_1d_simple"].rolling(20, min_periods=10).std().reset_index(level=0, drop=True))
    out["vol_compression"] = 1.0 / (1.0 + _safe_numeric(out["range_comp_5_20"]).abs())

    out["rev1_base"] = -_cs_zscore_one(out, "ret_1d_simple")
    out["gap_base"] = _cs_zscore_one(out, "gap_ret")
    out["pressure_base"] = _cs_zscore_one(out, "oc_body_pct")
    out["mom3_base"] = _cs_zscore_one(out, "mom3_raw")
    out["dist_low_20_base"] = _cs_zscore_one(out, "dist_low_20")
    out["dist_high_20_base"] = _cs_zscore_one(out, "dist_high_20")
    out["wick_imbalance_base"] = _cs_zscore_one(out, "wick_imbalance")
    return out


def _build_regime_features(df: pd.DataFrame, specs: Sequence[AlphaSpec]) -> pd.DataFrame:
    out = df.copy()
    for spec in specs:
        if spec.base_col not in out.columns:
            raise RuntimeError(f"Missing base column for alpha {spec.name}: {spec.base_col}")
        if spec.regime_col not in out.columns:
            raise RuntimeError(f"Missing regime column for alpha {spec.name}: {spec.regime_col}")

        regime_rank = _cs_rank_pct(out, spec.regime_col)
        regime_z = _cs_zscore_one(out, spec.regime_col)
        base = _safe_numeric(out[spec.base_col])

        if "__hi_" in spec.name:
            out[f"alpha_{spec.name}"] = base * (regime_rank >= REGIME_HIGH_Q).astype("float64")
        elif "__lo_" in spec.name:
            out[f"alpha_{spec.name}"] = base * (regime_rank <= REGIME_LOW_Q).astype("float64")
        elif "__z_" in spec.name:
            out[f"alpha_{spec.name}"] = base * regime_z
        elif "__rank_" in spec.name:
            out[f"alpha_{spec.name}"] = base * (regime_rank - 0.5)
        else:
            raise RuntimeError(f"Unsupported alpha spec naming convention: {spec.name}")
    return out


def _add_conditional_block(df: pd.DataFrame) -> pd.DataFrame:
    out = add_conditional_factors(df)
    cond_cols = [c for c in CONDITIONAL_FEATURE_COLS if c in out.columns]
    if cond_cols:
        out = cs_zscore(out, cond_cols)
        for col in cond_cols:
            zcol = f"z_{col}"
            out[f"alpha_{col}"] = _safe_numeric(out[zcol])
    return out


def _target_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in ["target_fwd_ret_1d", "target_fwd_ret_2d", "target_fwd_ret_3d", "target_fwd_ret_5d"] if c in df.columns]


def _alpha_columns(df: pd.DataFrame) -> List[str]:
    return sorted([c for c in df.columns if c.startswith("alpha_")])


def _daily_ic(df: pd.DataFrame, factor_col: str, target_col: str, min_cs: int = 20) -> tuple[float, int]:
    vals: List[float] = []
    for _, g in df.groupby("date", sort=False):
        x = g[[factor_col, target_col]].dropna()
        if len(x) < min_cs:
            continue
        if x[factor_col].nunique(dropna=True) <= 1:
            continue
        if x[target_col].nunique(dropna=True) <= 1:
            continue
        ic = x[factor_col].corr(x[target_col], method="spearman")
        if pd.notna(ic):
            vals.append(float(ic))
    if not vals:
        return float("nan"), 0
    return float(np.nanmean(vals)), int(len(vals))


def _build_meta(df: pd.DataFrame) -> Dict[str, object]:
    alpha_cols = _alpha_columns(df)
    target_cols = _target_columns(df)
    ic_rows: List[Dict[str, object]] = []
    for col in alpha_cols:
        row: Dict[str, object] = {"alpha": col}
        for tgt in target_cols:
            ic, n_days = _daily_ic(df, col, tgt)
            row[f"{tgt}__ic"] = ic
            row[f"{tgt}__n_days"] = n_days
        ic_rows.append(row)
    ic_df = pd.DataFrame(ic_rows)
    if not ic_df.empty and "target_fwd_ret_1d__ic" in ic_df.columns:
        ic_df = ic_df.sort_values(["target_fwd_ret_1d__ic"], ascending=[False], na_position="last").reset_index(drop=True)

    top_ic_records = ic_df.head(20).to_dict(orient="records") if not ic_df.empty else []
    return {
        "rows": int(len(df)),
        "dates": int(df["date"].nunique()),
        "symbols": int(df["symbol"].nunique()),
        "alpha_count": int(len(alpha_cols)),
        "conditional_alpha_count": int(sum(1 for c in alpha_cols if c.startswith("alpha_cond_"))),
        "regime_alpha_count": int(sum(1 for c in alpha_cols if not c.startswith("alpha_cond_"))),
        "alpha_cols": alpha_cols,
        "target_cols": target_cols,
        "top_target_1d_ic": top_ic_records,
        "dbg_non_na_intraday_rs": float(_series_or_zero(df, "intraday_rs").ne(0.0).sum()),
        "dbg_non_na_intraday_pressure": float(_series_or_zero(df, "intraday_pressure").ne(0.0).sum()),
        "dbg_non_na_market_breadth": float(_series_or_zero(df, "market_breadth").ne(0.0).sum()),
    }


def main() -> int:
    _enable_line_buffering()
    print(f"[CFG] feature_file={FEATURE_FILE}")
    print(f"[CFG] out_dir={OUT_DIR}")
    print(f"[CFG] start={START} end={END}")
    print(f"[CFG] regime_high_q={REGIME_HIGH_Q} regime_low_q={REGIME_LOW_Q}")

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(f"Feature file not found: {FEATURE_FILE}")

    df = pd.read_parquet(FEATURE_FILE)
    if df.empty:
        raise RuntimeError("feature matrix is empty")

    df = _ensure_columns(df)
    df = _derive_base_features(df)
    df = _build_regime_features(df, BASE_REGIME_SPECS)
    df = _add_conditional_block(df)

    keep_cols = ["date", "symbol", "open", "high", "low", "close", "volume"]
    keep_cols += [c for c in ["market_breadth", "liq", "liq_rank", "rel_volume_20", "intraday_rs", "intraday_pressure"] if c in df.columns]
    keep_cols += _target_columns(df)
    keep_cols += _alpha_columns(df)
    keep_cols = [c for c in keep_cols if c in df.columns]
    out = df[keep_cols].copy()

    meta = _build_meta(out)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PARQUET, index=False)
    out.head(min(len(out), CSV_MAX_ROWS)).to_csv(OUT_CSV, index=False)
    OUT_META.write_text(pd.Series(meta).to_json(force_ascii=False, indent=2), encoding="utf-8")

    print(f"[DATA] rows={meta['rows']} dates={meta['dates']} symbols={meta['symbols']}")
    print(f"[ALPHAS] total={meta['alpha_count']} regime={meta['regime_alpha_count']} conditional={meta['conditional_alpha_count']}")
    print(f"[ARTIFACT] {OUT_PARQUET}")
    print(f"[ARTIFACT] {OUT_CSV}")
    print(f"[ARTIFACT] {OUT_META}")
    print("[FINAL] multi-alpha library build complete")
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 1
    _press_enter_exit(rc)
