# scripts/run_production_regime_alpha_pack.py
# Build a small production alpha pack from confirmed regime winners.
# Double-click runnable. Never auto-closes.
#
# This file intentionally does NOT rescan hundreds of candidates.
# It takes the confirmed winners from the regime scan stage and materializes
# production-ready alpha columns on top of the real feature_matrix_v1.parquet.
#
# Confirmed core used here:
# - rev1 / intraday_rs in high relative-volume regime
# - gap / intraday_pressure style alpha in z(volume_shock) regime
# - pressure in high-liquidity regime
#
# Outputs:
# - alpha_pack_production_v1.parquet
# - alpha_pack_production_v1.csv
# - alpha_pack_meta.json

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

FEATURE_FILE = Path("data") / "features" / "feature_matrix_v1.parquet"
OUT_DIR = Path("artifacts") / "production_alpha_pack"
OUT_PARQUET = OUT_DIR / "alpha_pack_production_v1.parquet"
OUT_CSV = OUT_DIR / "alpha_pack_production_v1.csv"
OUT_META = OUT_DIR / "alpha_pack_meta.json"

EPS = 1e-12


# ------------------------------------------------------------
# RUNTIME / ENV
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


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    feature_file: Path
    out_dir: Path
    start: str
    end: str
    target_col: str
    high_q: float
    low_q: float
    alpha_weight_rev_hi_rvol: float
    alpha_weight_gap_z_rvol: float
    alpha_weight_pressure_hi_liq: float
    output_keep_debug: bool
    min_cross_section: int



def load_config() -> Config:
    high_q = _env_float("REGIME_HIGH_Q", 0.70)
    low_q = _env_float("REGIME_LOW_Q", 0.30)
    if not (0.0 < low_q < high_q < 1.0):
        raise RuntimeError(f"Invalid regime quantiles: low={low_q} high={high_q}")
    return Config(
        feature_file=Path(_env_str("FEATURE_FILE", str(FEATURE_FILE))),
        out_dir=Path(_env_str("PROD_ALPHA_OUT_DIR", str(OUT_DIR))),
        start=_env_str("START", ""),
        end=_env_str("END", ""),
        target_col=_env_str("REGIME_TARGET_COL", "target_fwd_ret_1d"),
        high_q=high_q,
        low_q=low_q,
        alpha_weight_rev_hi_rvol=_env_float("ALPHA_W_REV_HI_RVOL", 1.00),
        alpha_weight_gap_z_rvol=_env_float("ALPHA_W_GAP_Z_RVOL", 1.00),
        alpha_weight_pressure_hi_liq=_env_float("ALPHA_W_PRESSURE_HI_LIQ", 0.60),
        output_keep_debug=str(_env_str("OUTPUT_KEEP_DEBUG", "1")).strip() == "1",
        min_cross_section=_env_int("MIN_CROSS_SECTION", 20),
    )


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def _safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("float64")



def _cross_section_rank_centered(df: pd.DataFrame, col: str) -> pd.Series:
    x = _safe_numeric(df[col])
    r = x.groupby(df["date"], sort=False).rank(method="average", pct=True)
    return r - 0.5



def _cross_section_zscore(df: pd.DataFrame, col: str) -> pd.Series:
    x = _safe_numeric(df[col])
    grp = x.groupby(df["date"], sort=False)
    mean = grp.transform("mean")
    std = grp.transform("std").replace(0.0, np.nan)
    out = (x - mean) / std
    return out.replace([np.inf, -np.inf], np.nan)



def _cross_section_high_mask(df: pd.DataFrame, col: str, q: float) -> pd.Series:
    x = _safe_numeric(df[col])
    r = x.groupby(df["date"], sort=False).rank(method="average", pct=True)
    return (r >= q).astype("float64")



def _cross_section_low_mask(df: pd.DataFrame, col: str, q: float) -> pd.Series:
    x = _safe_numeric(df[col])
    r = x.groupby(df["date"], sort=False).rank(method="average", pct=True)
    return (r <= q).astype("float64")



def _normalize_factor(df: pd.DataFrame, col: str) -> pd.Series:
    zcol = f"z_{col}"
    if zcol in df.columns:
        return _safe_numeric(df[zcol]).replace([np.inf, -np.inf], np.nan)
    return _cross_section_zscore(df, col)



def _signed_rank_compress(df: pd.DataFrame, col: str) -> pd.Series:
    z = _normalize_factor(df, col)
    out = np.tanh(z)
    return pd.Series(out, index=df.index, dtype="float64")



def _validate_required_columns(df: pd.DataFrame, cfg: Config) -> None:
    required = ["date", "symbol", cfg.target_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required base columns: {missing}")

    candidate_groups = [
        ["intraday_rs", "rev1"],
        ["volume_shock", "rel_volume_20"],
        ["intraday_pressure", "gap_ret", "pressure"],
        ["liq_rank", "liq", "cs_liq_z"],
    ]
    unresolved: List[str] = []
    for group in candidate_groups:
        if not any((c in df.columns) or (f"z_{c}" in df.columns) for c in group):
            unresolved.append(" OR ".join(group))
    if unresolved:
        raise RuntimeError(
            "Missing required factor families for production alpha pack: "
            + "; ".join(unresolved)
        )



def _resolve_first_existing(df: pd.DataFrame, names: Sequence[str]) -> str:
    for name in names:
        if name in df.columns or f"z_{name}" in df.columns:
            return name
    raise RuntimeError(f"Could not resolve any of the columns: {list(names)}")



def _daily_ic(frame: pd.DataFrame, factor_col: str, target_col: str, min_cross_section: int) -> Tuple[float, int, Dict[str, int]]:
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
# LOAD / FILTER
# ------------------------------------------------------------

def load_feature_matrix(cfg: Config) -> pd.DataFrame:
    if not cfg.feature_file.exists():
        raise FileNotFoundError(f"Feature file not found: {cfg.feature_file}")
    df = pd.read_parquet(cfg.feature_file)
    if df.empty:
        raise RuntimeError("Loaded feature matrix is empty")

    if "symbol" not in df.columns:
        if "ticker" in df.columns:
            df = df.rename(columns={"ticker": "symbol"})
        elif "sym" in df.columns:
            df = df.rename(columns={"sym": "symbol"})
    if "date" not in df.columns or "symbol" not in df.columns:
        raise RuntimeError("Feature matrix must contain date and symbol")

    df["date"] = pd.to_datetime(df["date"])
    if cfg.start:
        df = df.loc[df["date"] >= pd.Timestamp(cfg.start)].copy()
    if cfg.end:
        df = df.loc[df["date"] <= pd.Timestamp(cfg.end)].copy()
    if df.empty:
        raise RuntimeError("Feature matrix is empty after START/END filter")

    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)
    _validate_required_columns(df, cfg)
    return df


# ------------------------------------------------------------
# ALPHA PACK
# ------------------------------------------------------------

def build_production_alpha_pack(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, Dict[str, str]]:
    rev_col = _resolve_first_existing(df, ["intraday_rs", "rev1"])
    rvol_col = _resolve_first_existing(df, ["volume_shock", "rel_volume_20"])
    gap_pressure_col = _resolve_first_existing(df, ["gap_ret", "intraday_pressure", "pressure"])
    pressure_col = _resolve_first_existing(df, ["intraday_pressure", "pressure", "gap_ret"])
    liq_col = _resolve_first_existing(df, ["liq_rank", "liq", "cs_liq_z"])

    mapping = {
        "rev_col": rev_col,
        "rvol_col": rvol_col,
        "gap_pressure_col": gap_pressure_col,
        "pressure_col": pressure_col,
        "liq_col": liq_col,
    }

    out = df[["date", "symbol", cfg.target_col]].copy()

    rev_norm = _signed_rank_compress(df, rev_col)
    gap_pressure_norm = _signed_rank_compress(df, gap_pressure_col)
    pressure_norm = _signed_rank_compress(df, pressure_col)

    rvol_z = _cross_section_zscore(df, rvol_col)
    rvol_hi = _cross_section_high_mask(df, rvol_col, cfg.high_q)
    rvol_lo = _cross_section_low_mask(df, rvol_col, cfg.low_q)

    liq_hi = _cross_section_high_mask(df, liq_col, cfg.high_q)
    liq_rank_centered = _cross_section_rank_centered(df, liq_col)

    # Confirmed winner 1:
    # reversal / relative-strength factor activated in high relative-volume regime.
    out["alpha_rev_hi_rvol"] = rev_norm * rvol_hi

    # Confirmed winner 2:
    # gap/pressure-style signal scaled by z(relative volume regime).
    out["alpha_gap_z_rvol"] = gap_pressure_norm * rvol_z

    # Tier-2 helper:
    # pressure-like signal in high-liquidity regime.
    out["alpha_pressure_hi_liq"] = pressure_norm * liq_hi

    # Optional soft variants retained only as debug features, not in final weighted score.
    out["dbg_alpha_rev_lo_rvol"] = rev_norm * rvol_lo
    out["dbg_alpha_pressure_x_liq_rank"] = pressure_norm * liq_rank_centered

    score = (
        cfg.alpha_weight_rev_hi_rvol * out["alpha_rev_hi_rvol"].fillna(0.0)
        + cfg.alpha_weight_gap_z_rvol * out["alpha_gap_z_rvol"].fillna(0.0)
        + cfg.alpha_weight_pressure_hi_liq * out["alpha_pressure_hi_liq"].fillna(0.0)
    )
    out["alpha_score_regime_pack_raw"] = score

    grp = score.groupby(df["date"], sort=False)
    mean = grp.transform("mean")
    std = grp.transform("std").replace(0.0, np.nan)
    out["alpha_score_regime_pack_z"] = ((score - mean) / std).replace([np.inf, -np.inf], np.nan)

    out["alpha_score_regime_pack_rank"] = (
        out["alpha_score_regime_pack_raw"].groupby(df["date"], sort=False).rank(method="average", pct=True) - 0.5
    )

    # Final production-facing score: compressed to limit tail explosions.
    out["alpha_score_regime_pack_final"] = np.tanh(out["alpha_score_regime_pack_z"].fillna(0.0))

    # Debug counters / masks for explicit no-silent-bypass behavior.
    out["dbg_mask_rvol_hi"] = rvol_hi
    out["dbg_mask_liq_hi"] = liq_hi
    out["dbg_has_rev_input"] = rev_norm.notna().astype("int64")
    out["dbg_has_gap_input"] = gap_pressure_norm.notna().astype("int64")
    out["dbg_has_pressure_input"] = pressure_norm.notna().astype("int64")
    out["dbg_has_rvol_z"] = rvol_z.notna().astype("int64")
    out["dbg_has_score_final"] = out["alpha_score_regime_pack_final"].notna().astype("int64")

    return out, mapping


# ------------------------------------------------------------
# EVALUATION
# ------------------------------------------------------------

def build_eval_table(alpha_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    eval_cols = [
        "alpha_rev_hi_rvol",
        "alpha_gap_z_rvol",
        "alpha_pressure_hi_liq",
        "alpha_score_regime_pack_raw",
        "alpha_score_regime_pack_z",
        "alpha_score_regime_pack_rank",
        "alpha_score_regime_pack_final",
    ]
    rows: List[Dict[str, object]] = []
    for col in eval_cols:
        ic, n_days, dbg = _daily_ic(alpha_df[["date", cfg.target_col, col]].copy(), col, cfg.target_col, cfg.min_cross_section)
        rows.append({
            "feature": col,
            "ic": ic,
            "abs_ic": abs(ic) if pd.notna(ic) else np.nan,
            "n_days": n_days,
            **dbg,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["abs_ic", "n_days"], ascending=[False, False]).reset_index(drop=True)
    return out


# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

def save_outputs(alpha_df: pd.DataFrame, eval_df: pd.DataFrame, mapping: Dict[str, str], cfg: Config) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.output_keep_debug:
        alpha_out = alpha_df.copy()
    else:
        keep_cols = [
            "date",
            "symbol",
            cfg.target_col,
            "alpha_rev_hi_rvol",
            "alpha_gap_z_rvol",
            "alpha_pressure_hi_liq",
            "alpha_score_regime_pack_raw",
            "alpha_score_regime_pack_z",
            "alpha_score_regime_pack_rank",
            "alpha_score_regime_pack_final",
        ]
        alpha_out = alpha_df[keep_cols].copy()

    alpha_out.to_parquet(OUT_PARQUET, index=False)
    eval_df.to_csv(OUT_CSV, index=False)

    mask_summary = {
        "rows_total": int(len(alpha_df)),
        "rows_rvol_hi": int(alpha_df["dbg_mask_rvol_hi"].fillna(0.0).sum()),
        "rows_liq_hi": int(alpha_df["dbg_mask_liq_hi"].fillna(0.0).sum()),
        "rows_has_final": int(alpha_df["dbg_has_score_final"].fillna(0).sum()),
        "dates": int(alpha_df["date"].nunique()),
        "symbols": int(alpha_df["symbol"].nunique()),
    }

    meta = {
        "feature_file": str(cfg.feature_file),
        "out_dir": str(cfg.out_dir),
        "start": cfg.start,
        "end": cfg.end,
        "target_col": cfg.target_col,
        "high_q": cfg.high_q,
        "low_q": cfg.low_q,
        "alpha_weight_rev_hi_rvol": cfg.alpha_weight_rev_hi_rvol,
        "alpha_weight_gap_z_rvol": cfg.alpha_weight_gap_z_rvol,
        "alpha_weight_pressure_hi_liq": cfg.alpha_weight_pressure_hi_liq,
        "min_cross_section": cfg.min_cross_section,
        "resolved_columns": mapping,
        "mask_summary": mask_summary,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    OUT_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main() -> int:
    _enable_line_buffering()
    cfg = load_config()

    print(f"[CFG] feature_file={cfg.feature_file}")
    print(f"[CFG] out_dir={cfg.out_dir}")
    print(f"[CFG] start={cfg.start or 'FULL'} end={cfg.end or 'FULL'}")
    print(f"[CFG] target_col={cfg.target_col}")
    print(f"[CFG] regime_low_q={cfg.low_q:.2f} regime_high_q={cfg.high_q:.2f}")
    print(
        "[CFG] alpha_weights="
        f"rev_hi_rvol={cfg.alpha_weight_rev_hi_rvol:.3f} "
        f"gap_z_rvol={cfg.alpha_weight_gap_z_rvol:.3f} "
        f"pressure_hi_liq={cfg.alpha_weight_pressure_hi_liq:.3f}"
    )

    df = load_feature_matrix(cfg)
    print(f"[DATA] rows={len(df)} dates={df['date'].nunique()} symbols={df['symbol'].nunique()}")

    alpha_df, mapping = build_production_alpha_pack(df, cfg)
    print(
        "[RESOLVED] "
        f"rev={mapping['rev_col']} "
        f"rvol={mapping['rvol_col']} "
        f"gap_pressure={mapping['gap_pressure_col']} "
        f"pressure={mapping['pressure_col']} "
        f"liq={mapping['liq_col']}"
    )

    eval_df = build_eval_table(alpha_df, cfg)

    print("\n[IC] ALPHA PACK EVAL")
    if eval_df.empty:
        print("No eval rows produced")
    else:
        print(eval_df.to_string(index=False))

    print("\n[DEBUG] MASK COUNTS")
    print(
        f"rvol_hi={int(alpha_df['dbg_mask_rvol_hi'].fillna(0.0).sum())} "
        f"liq_hi={int(alpha_df['dbg_mask_liq_hi'].fillna(0.0).sum())} "
        f"has_final={int(alpha_df['dbg_has_score_final'].fillna(0).sum())}"
    )

    save_outputs(alpha_df, eval_df, mapping, cfg)

    print(f"\n[ARTIFACT] {OUT_PARQUET}")
    print(f"[ARTIFACT] {OUT_CSV}")
    print(f"[ARTIFACT] {OUT_META}")
    print(f"[FINAL] rows={len(alpha_df)} eval_rows={len(eval_df)}")
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 1
    _press_enter_exit(rc)
