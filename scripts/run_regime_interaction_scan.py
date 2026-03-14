from __future__ import annotations

import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np
import pandas as pd

try:
    from python_edge.features.add_intraday_pressure import add_intraday_pressure
    from python_edge.features.add_intraday_rs import add_intraday_rs
    from python_edge.model.conditional_factors import add_conditional_factors
    from python_edge.model.cs_normalize import cs_zscore
except Exception as import_exc:
    raise RuntimeError(
        "Failed to import project modules. Run this from repo root with src on PYTHONPATH. "
        f"repo_root={_REPO_ROOT} src_dir={_SRC_DIR} original_error={import_exc}"
    )

_ARTIFACT_DIR = _REPO_ROOT / "artifacts" / "regime_interaction_scan"
EPS = 1e-12
NY_TZ = "America/New_York"

DEFAULT_SCAN_SPECS: Tuple[Tuple[str, str], ...] = (
    ("intraday_rs", "liq_rank"),
    ("intraday_rs", "volume_shock"),
    ("intraday_pressure", "volume_shock"),
    ("intraday_pressure", "liq_rank"),
    ("cond_str_weak_breadth", "market_breadth"),
    ("cond_overnight_trend_follow", "market_breadth"),
    ("cond_vol_compression_liq_breakout", "liq_rank"),
    ("cond_momentum_liq_trend", "volume_shock"),
    ("rev1", "liq_rank"),
    ("rev1", "volume_shock"),
    ("gap_ret", "volume_shock"),
)


@dataclass(frozen=True)
class Config:
    dataset_root: Path
    artifact_dir: Path
    start: str
    end: str
    target_col: str
    target_mode: str
    use_rth_only: bool
    min_cross_section: int
    min_ic_days: int
    high_q: float
    low_q: float
    min_abs_uplift: float
    min_rel_uplift: float
    use_abs_ic: bool
    min_symbol_days: int
    scan_specs: Tuple[Tuple[str, str], ...]


def _enable_line_buffering() -> None:
    for name in ["stdout", "stderr"]:
        stream = getattr(sys, name, None)
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


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


def _safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("float64")


def _cross_section_zscore(df: pd.DataFrame, col: str, by_col: str = "date") -> pd.Series:
    x = _safe_numeric(df[col])
    grp = x.groupby(df[by_col], sort=False)
    mean = grp.transform("mean")
    std = grp.transform("std").replace(0.0, np.nan)
    z = (x - mean) / std
    return z.replace([np.inf, -np.inf], np.nan)


def _safe_rank_pct(df: pd.DataFrame, col: str, by_col: str = "date") -> pd.Series:
    return _safe_numeric(df[col]).groupby(df[by_col], sort=False).rank(method="average", pct=True)


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _stage_start(name: str) -> float:
    _log(f"[STAGE][START] {name}")
    return time.perf_counter()


def _stage_done(name: str, t0: float, extra: str = "") -> None:
    dt = time.perf_counter() - t0
    suffix = f" {extra}" if extra else ""
    _log(f"[STAGE][DONE] {name} dt={dt:.2f}s{suffix}")


def _parse_scan_specs(raw: str) -> Tuple[Tuple[str, str], ...]:
    txt = raw.strip()
    if not txt:
        return DEFAULT_SCAN_SPECS
    out: List[Tuple[str, str]] = []
    for token in txt.split(";"):
        piece = token.strip()
        if not piece:
            continue
        if ":" not in piece:
            raise RuntimeError(
                "Invalid REGIME_SCAN_SPECS token. Use 'factor:regime;factor:regime'. "
                f"Bad token: {piece}"
            )
        left, right = piece.split(":", 1)
        factor_col = left.strip()
        regime_col = right.strip()
        if not factor_col or not regime_col:
            raise RuntimeError(f"Invalid REGIME_SCAN_SPECS token: {piece}")
        out.append((factor_col, regime_col))
    if not out:
        raise RuntimeError("REGIME_SCAN_SPECS resolved to empty list")
    return tuple(out)


def load_config() -> Config:
    high_q = _env_float("REGIME_HIGH_Q", 0.70)
    low_q = _env_float("REGIME_LOW_Q", 0.30)
    if not (0.0 < low_q < high_q < 1.0):
        raise RuntimeError(f"Invalid regime quantiles: low={low_q} high={high_q}")
    return Config(
        dataset_root=Path(_env_str("DATASET_ROOT", r"D:\massive_dataset")),
        artifact_dir=Path(_env_str("REGIME_ARTIFACT_DIR", str(_ARTIFACT_DIR))),
        start=_env_str("START", "2023-01-01"),
        end=_env_str("END", "2026-02-28"),
        target_col=_env_str("REGIME_TARGET_COL", "target_fwd_ret_1d"),
        target_mode=_env_str("REGIME_TARGET_MODE", "next_open_to_next_close").lower(),
        use_rth_only=_env_bool("USE_RTH_ONLY", True),
        min_cross_section=_env_int("MIN_CROSS_SECTION", 20),
        min_ic_days=_env_int("MIN_IC_DAYS", 30),
        high_q=high_q,
        low_q=low_q,
        min_abs_uplift=_env_float("MIN_ABS_UPLIFT", 0.0020),
        min_rel_uplift=_env_float("MIN_REL_UPLIFT", 0.15),
        use_abs_ic=_env_bool("REGIME_USE_ABS_IC", True),
        min_symbol_days=_env_int("MIN_SYMBOL_DAYS", 80),
        scan_specs=_parse_scan_specs(_env_str("REGIME_SCAN_SPECS", "")),
    )


def _find_15m_files(root: Path) -> List[Tuple[str, Path]]:
    out: List[Tuple[str, Path]] = []
    if not root.exists():
        raise RuntimeError(f"dataset_root not found: {root}")
    for sym_dir in sorted(root.iterdir()):
        if not sym_dir.is_dir():
            continue
        symbol = sym_dir.name.upper()
        files = sorted(sym_dir.glob("aggs_15m_*__FULL.json"))
        if not files:
            continue
        for path in files:
            out.append((symbol, path))
    if not out:
        raise RuntimeError(f"No aggs_15m_*__FULL.json files found under {root}")
    return out


def _load_15m_file(symbol: str, path: Path, cfg: Config) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    rows = payload.get("results") or []
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "t" not in df.columns:
        return pd.DataFrame()
    rename_map = {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
    df = df.rename(columns=rename_map)
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            raise RuntimeError(f"{path}: missing raw column {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    ts_utc = pd.to_datetime(df["t"], unit="ms", utc=True)
    ts_local = ts_utc.dt.tz_convert(NY_TZ)
    df["ts_local"] = ts_local
    df["session_date"] = ts_local.dt.date.astype(str)
    df["date"] = pd.to_datetime(df["session_date"])
    df["symbol"] = symbol
    df["hour"] = ts_local.dt.hour
    df["minute"] = ts_local.dt.minute
    df["is_rth"] = (((df["hour"] > 9) | ((df["hour"] == 9) & (df["minute"] >= 30))) & ((df["hour"] < 16) | ((df["hour"] == 16) & (df["minute"] == 0)))).astype(bool)
    if cfg.use_rth_only:
        before = len(df)
        df = df.loc[df["is_rth"]].copy()
        after = len(df)
        _log(f"[RTH] symbol={symbol} file={path.name} kept={after} dropped={before - after}")
    if df.empty:
        return pd.DataFrame()
    return df[["date", "session_date", "symbol", "ts_local", "open", "high", "low", "close", "volume"]]


def load_raw_15m(cfg: Config) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    files = _find_15m_files(cfg.dataset_root)
    _log(f"[LOAD] found_15m_files={len(files)}")
    for idx, item in enumerate(files, start=1):
        symbol, path = item
        _log(f"[LOAD][FILE] {idx}/{len(files)} symbol={symbol} file={path.name}")
        block = _load_15m_file(symbol, path, cfg)
        if not block.empty:
            parts.append(block)
    if not parts:
        raise RuntimeError("No non-empty 15m data blocks loaded")
    df15 = pd.concat(parts, ignore_index=True)
    df15 = df15.loc[(df15["date"] >= pd.Timestamp(cfg.start)) & (df15["date"] <= pd.Timestamp(cfg.end))].copy()
    if df15.empty:
        raise RuntimeError("15m dataset empty after START/END filter")
    df15 = df15.sort_values(["symbol", "ts_local"]).reset_index(drop=True)
    return df15


def aggregate_15m_to_daily(df15: pd.DataFrame) -> pd.DataFrame:
    daily = df15.groupby(["symbol", "session_date"], as_index=False).agg(
        date=("date", "first"),
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        bars=("close", "size"),
    )
    daily["meta_price"] = pd.to_numeric(daily["close"], errors="coerce")
    daily["meta_dollar_volume"] = pd.to_numeric(daily["close"], errors="coerce") * pd.to_numeric(daily["volume"], errors="coerce")
    return daily.sort_values(["symbol", "date"]).reset_index(drop=True)


def add_base_daily_features(df1d: pd.DataFrame, target_mode: str) -> pd.DataFrame:
    out = df1d.copy()
    grp_close = out.groupby("symbol", sort=False)["close"]
    grp_open = out.groupby("symbol", sort=False)["open"]
    prev_close = grp_close.shift(1)

    out["ret_1d"] = grp_close.pct_change(fill_method=None)
    out["ret_3d"] = grp_close.pct_change(3, fill_method=None)
    out["ret_5d"] = grp_close.pct_change(5, fill_method=None)
    out["ret_20d"] = grp_close.pct_change(20, fill_method=None)
    out["rev1"] = -out["ret_1d"]
    out["gap_ret"] = (out["open"] / (prev_close + EPS)) - 1.0
    out["intraday_ret_1d"] = (out["close"] / (out["open"] + EPS)) - 1.0
    out["overnight_ret_1d"] = (out["open"] / (prev_close + EPS)) - 1.0
    out["range_pct"] = (out["high"] - out["low"]) / (out["close"] + EPS)

    out["volume_mean_20"] = out.groupby("symbol", sort=False)["volume"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    out["volume_shock"] = out["volume"] / (out["volume_mean_20"] + EPS)
    out["liq_rank"] = _safe_rank_pct(out, "meta_dollar_volume", by_col="date")

    out["momentum_20d"] = out["ret_20d"]
    out["str_3d"] = -out["ret_3d"]
    out["overnight_drift_20d"] = out.groupby("symbol", sort=False)["overnight_ret_1d"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    out["ivol_20d"] = out.groupby("symbol", sort=False)["ret_1d"].transform(lambda s: s.rolling(20, min_periods=10).std())
    out["rv_5d"] = out.groupby("symbol", sort=False)["ret_1d"].transform(lambda s: s.rolling(5, min_periods=3).std())
    out["rv_20d"] = out.groupby("symbol", sort=False)["ret_1d"].transform(lambda s: s.rolling(20, min_periods=10).std())
    out["vol_compression"] = out["rv_5d"] / (out["rv_20d"] + EPS)

    adv_breadth = out.groupby("date", sort=False)["intraday_ret_1d"].transform(lambda s: (s > 0).mean())
    out["market_breadth"] = adv_breadth.astype("float64")

    next_open = grp_open.shift(-1)
    next_close = grp_close.shift(-1)
    if target_mode == "next_close_to_next_close":
        out["target_fwd_ret_1d"] = grp_close.shift(-1) / (out["close"] + EPS) - 1.0
    else:
        out["target_fwd_ret_1d"] = next_close / (next_open + EPS) - 1.0

    out["target_fwd_ret_cc_1d"] = grp_close.shift(-1) / (out["close"] + EPS) - 1.0
    out["target_fwd_ret_oc_1d"] = next_close / (next_open + EPS) - 1.0
    return out


def build_feature_frame(df15: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    t0 = _stage_start("aggregate_15m_to_daily")
    daily = aggregate_15m_to_daily(df15)
    _stage_done("aggregate_15m_to_daily", t0, extra=f"rows_1d={len(daily)}")

    t1 = _stage_start("add_base_daily_features")
    daily = add_base_daily_features(daily, cfg.target_mode)
    _stage_done("add_base_daily_features", t1, extra=f"rows_1d={len(daily)}")

    merged_parts: List[pd.DataFrame] = []
    groups = list(df15.groupby("symbol", sort=False))
    _log(f"[FEATURE] symbol_groups={len(groups)}")
    for idx, item in enumerate(groups, start=1):
        symbol, g15 = item
        ts = time.perf_counter()
        g1d = daily.loc[daily["symbol"] == symbol].copy()
        g15 = g15.sort_values("ts_local").reset_index(drop=True)
        g1d = g1d.sort_values("date").reset_index(drop=True)
        _log(f"[FEATURE][SYMBOL][START] {idx}/{len(groups)} symbol={symbol} rows15={len(g15)} rows1d={len(g1d)}")
        g1d = add_intraday_rs(g15, g1d)
        g1d = add_intraday_pressure(g15, g1d)
        merged_parts.append(g1d)
        _log(f"[FEATURE][SYMBOL][DONE] {idx}/{len(groups)} symbol={symbol} rows1d={len(g1d)} dt={time.perf_counter() - ts:.2f}s")

    t2 = _stage_start("concat_feature_blocks")
    out = pd.concat(merged_parts, ignore_index=True)
    _stage_done("concat_feature_blocks", t2, extra=f"rows={len(out)}")

    out["intraday_rs_x_volume_shock"] = out["intraday_rs"] * out["volume_shock"]
    out["intraday_pressure_x_volume_shock"] = out["intraday_pressure"] * out["volume_shock"]
    out["liq_rank_x_intraday_rs"] = out["liq_rank"] * out["intraday_rs"]

    t3 = _stage_start("add_conditional_factors")
    out = add_conditional_factors(out)
    _stage_done("add_conditional_factors", t3, extra=f"rows={len(out)} cols={len(out.columns)}")

    feature_cols_to_z = sorted({
        "intraday_rs",
        "volume_shock",
        "intraday_pressure",
        "intraday_rs_x_volume_shock",
        "intraday_pressure_x_volume_shock",
        "liq_rank_x_intraday_rs",
        "market_breadth",
        "liq_rank",
        "momentum_20d",
        "str_3d",
        "overnight_drift_20d",
        "ivol_20d",
        "vol_compression",
        *[a for a, _ in cfg.scan_specs],
        *[b for _, b in cfg.scan_specs],
    })
    feature_cols_to_z = [c for c in feature_cols_to_z if c in out.columns]

    t4 = _stage_start("cs_zscore")
    out = cs_zscore(out, feature_cols_to_z)
    _stage_done("cs_zscore", t4, extra=f"ncols={len(feature_cols_to_z)}")

    out = out.sort_values(["date", "symbol"]).reset_index(drop=True)
    return out


def _normalize_factor(df: pd.DataFrame, col: str) -> pd.Series:
    zcol = f"z_{col}"
    if zcol in df.columns:
        return _safe_numeric(df[zcol])
    return _cross_section_zscore(df, col)


def _normalize_regime(df: pd.DataFrame, col: str, cfg: Config) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    if col not in df.columns:
        raise RuntimeError(f"_normalize_regime: missing regime column {col}")
    x = _safe_numeric(df[col])
    rank = x.groupby(df["date"], sort=False).rank(method="average", pct=True)
    z = _cross_section_zscore(df, col)
    high = (rank >= cfg.high_q).astype("float64")
    low = (rank <= cfg.low_q).astype("float64")
    return rank - 0.5, z, high, low


def build_variant_frame(df: pd.DataFrame, factor_col: str, regime_col: str, cfg: Config) -> pd.DataFrame:
    out = df[["date", "symbol", cfg.target_col]].copy()
    factor_norm = _normalize_factor(df, factor_col)
    regime_rank_centered, regime_z, regime_high, regime_low = _normalize_regime(df, regime_col, cfg)
    out["variant_raw"] = factor_norm
    out["variant_high"] = factor_norm * regime_high
    out["variant_low"] = factor_norm * regime_low
    out["variant_rank"] = factor_norm * regime_rank_centered
    out["variant_z"] = factor_norm * regime_z
    return out


def _ic_by_date(frame: pd.DataFrame, factor_col: str, target_col: str, min_cross_section: int) -> Tuple[float, int, Dict[str, int]]:
    rows: List[float] = []
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
            rows.append(float(ic))
            dbg["dates_ok"] += 1
    if not rows:
        return float("nan"), 0, dbg
    return float(np.nanmean(rows)), int(len(rows)), dbg


def scan_pair(df: pd.DataFrame, factor_col: str, regime_col: str, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    work = build_variant_frame(df, factor_col, regime_col, cfg)
    variant_cols = {
        "raw": "variant_raw",
        "high": "variant_high",
        "low": "variant_low",
        "rank": "variant_rank",
        "z": "variant_z",
    }
    variant_rows: List[Dict[str, object]] = []
    by_variant: Dict[str, Dict[str, object]] = {}
    for variant_name, col_name in variant_cols.items():
        ic, n_days, dbg = _ic_by_date(work, col_name, cfg.target_col, cfg.min_cross_section)
        abs_ic = abs(ic) if pd.notna(ic) else np.nan
        row = {
            "factor": factor_col,
            "regime": regime_col,
            "variant": variant_name,
            "feature_name": f"rg__{factor_col}__{variant_name}__{regime_col}",
            "ic": ic,
            "abs_ic": abs_ic,
            "n_days": n_days,
            **dbg,
        }
        variant_rows.append(row)
        by_variant[variant_name] = row

    raw_row = by_variant["raw"]
    candidates = [by_variant["high"], by_variant["low"], by_variant["rank"], by_variant["z"]]
    best_row = max(candidates, key=lambda r: (-1.0 if pd.isna(r["abs_ic"]) else float(r["abs_ic"])))

    metric_raw = raw_row["abs_ic"] if cfg.use_abs_ic else raw_row["ic"]
    metric_best = best_row["abs_ic"] if cfg.use_abs_ic else best_row["ic"]
    abs_uplift = best_row["abs_ic"] - raw_row["abs_ic"] if pd.notna(best_row["abs_ic"]) and pd.notna(raw_row["abs_ic"]) else np.nan
    rel_uplift = abs_uplift / (raw_row["abs_ic"] + EPS) if pd.notna(abs_uplift) and pd.notna(raw_row["abs_ic"]) else np.nan
    passed = bool(
        pd.notna(abs_uplift)
        and pd.notna(rel_uplift)
        and int(best_row["n_days"]) >= cfg.min_ic_days
        and abs_uplift >= cfg.min_abs_uplift
        and rel_uplift >= cfg.min_rel_uplift
    )

    summary = pd.DataFrame([{
        "factor": factor_col,
        "regime": regime_col,
        "target_col": cfg.target_col,
        "raw_ic": raw_row["ic"],
        "raw_abs_ic": raw_row["abs_ic"],
        "best_variant": best_row["variant"],
        "best_ic": best_row["ic"],
        "best_abs_ic": best_row["abs_ic"],
        "raw_n_days": raw_row["n_days"],
        "best_n_days": best_row["n_days"],
        "abs_uplift": abs_uplift,
        "rel_uplift": rel_uplift,
        "pass_regime_uplift": int(passed),
        "metric_mode": "abs_ic" if cfg.use_abs_ic else "signed_ic",
        "metric_raw": metric_raw,
        "metric_best": metric_best,
    }])
    return pd.DataFrame(variant_rows), summary


def _filter_feature_frame(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = df.copy()
    counts = out.groupby("symbol", sort=False)["date"].transform("count")
    before = len(out)
    out = out.loc[counts >= cfg.min_symbol_days].copy()
    dropped = before - len(out)
    _log(f"[FILTER] min_symbol_days={cfg.min_symbol_days} dropped_rows={dropped}")
    if out.empty:
        raise RuntimeError("All rows dropped by MIN_SYMBOL_DAYS")
    return out


def main() -> int:
    _enable_line_buffering()
    cfg = load_config()

    _log(f"[CFG] dataset_root={cfg.dataset_root}")
    _log(f"[CFG] repo_root={_REPO_ROOT} src_dir={_SRC_DIR}")
    _log(f"[CFG] start={cfg.start} end={cfg.end} use_rth_only={int(cfg.use_rth_only)}")
    _log(f"[CFG] target_col={cfg.target_col} target_mode={cfg.target_mode}")
    _log(f"[CFG] min_cross_section={cfg.min_cross_section} min_ic_days={cfg.min_ic_days} min_symbol_days={cfg.min_symbol_days}")
    _log(f"[CFG] regime_low_q={cfg.low_q:.2f} regime_high_q={cfg.high_q:.2f}")
    _log(f"[CFG] min_abs_uplift={cfg.min_abs_uplift:.4f} min_rel_uplift={cfg.min_rel_uplift:.2f}")
    _log(f"[CFG] use_abs_ic={int(cfg.use_abs_ic)}")
    _log(f"[CFG] scan_specs={'; '.join(f'{a}:{b}' for a, b in cfg.scan_specs)}")

    t0 = _stage_start("load_raw_15m")
    df15 = load_raw_15m(cfg)
    _stage_done("load_raw_15m", t0, extra=f"rows={len(df15)} dates={df15['date'].nunique()} symbols={df15['symbol'].nunique()}")
    _log(f"[DATA15] rows={len(df15)} dates={df15['date'].nunique()} symbols={df15['symbol'].nunique()}")

    t1 = _stage_start("build_feature_frame")
    df = build_feature_frame(df15, cfg)
    _stage_done("build_feature_frame", t1, extra=f"rows={len(df)} cols={len(df.columns)}")

    t2 = _stage_start("filter_feature_frame")
    df = _filter_feature_frame(df, cfg)
    _stage_done("filter_feature_frame", t2, extra=f"rows={len(df)} dates={df['date'].nunique()} symbols={df['symbol'].nunique()}")
    _log(f"[DATA1D] rows={len(df)} dates={df['date'].nunique()} symbols={df['symbol'].nunique()}")

    missing_base = [c for c in ["date", "symbol", cfg.target_col] if c not in df.columns]
    if missing_base:
        raise RuntimeError(f"Missing required base columns: {missing_base}")

    missing_cols: List[str] = []
    for factor_col, regime_col in cfg.scan_specs:
        if factor_col not in df.columns and f"z_{factor_col}" not in df.columns:
            missing_cols.append(factor_col)
        if regime_col not in df.columns:
            missing_cols.append(regime_col)
    if missing_cols:
        raise RuntimeError(f"Missing required factor/regime columns: {sorted(set(missing_cols))}")

    all_variants: List[pd.DataFrame] = []
    all_summary: List[pd.DataFrame] = []
    for idx, item in enumerate(cfg.scan_specs, start=1):
        factor_col, regime_col = item
        ts = _stage_start(f"scan_pair {idx}/{len(cfg.scan_specs)} factor={factor_col} regime={regime_col}")
        variants_df, summary_df = scan_pair(df, factor_col, regime_col, cfg)
        all_variants.append(variants_df)
        all_summary.append(summary_df)
        _stage_done(
            f"scan_pair {idx}/{len(cfg.scan_specs)} factor={factor_col} regime={regime_col}",
            ts,
            extra=f"variant_rows={len(variants_df)} summary_rows={len(summary_df)}",
        )

    variants = pd.concat(all_variants, ignore_index=True) if all_variants else pd.DataFrame()
    summary = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
    if not summary.empty:
        summary = summary.sort_values(
            ["pass_regime_uplift", "best_abs_ic", "abs_uplift", "rel_uplift"],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)
    kept = summary.loc[summary["pass_regime_uplift"] == 1].copy().reset_index(drop=True) if not summary.empty else pd.DataFrame()

    cfg.artifact_dir.mkdir(parents=True, exist_ok=True)
    p_variants = cfg.artifact_dir / "regime_variant_ic_table.csv"
    p_summary = cfg.artifact_dir / "regime_summary_table.csv"
    p_kept = cfg.artifact_dir / "regime_retained_candidates.csv"
    p_meta = cfg.artifact_dir / "regime_scan_meta.json"
    p_feature_snapshot = cfg.artifact_dir / "regime_feature_snapshot.parquet"

    tw = _stage_start("write_artifacts")
    variants.to_csv(p_variants, index=False)
    summary.to_csv(p_summary, index=False)
    kept.to_csv(p_kept, index=False)
    df.to_parquet(p_feature_snapshot, index=False)

    meta = {
        "dataset_root": str(cfg.dataset_root),
        "target_col": cfg.target_col,
        "target_mode": cfg.target_mode,
        "use_rth_only": cfg.use_rth_only,
        "start": cfg.start,
        "end": cfg.end,
        "min_cross_section": cfg.min_cross_section,
        "min_ic_days": cfg.min_ic_days,
        "min_symbol_days": cfg.min_symbol_days,
        "high_q": cfg.high_q,
        "low_q": cfg.low_q,
        "min_abs_uplift": cfg.min_abs_uplift,
        "min_rel_uplift": cfg.min_rel_uplift,
        "use_abs_ic": cfg.use_abs_ic,
        "scan_specs": [{"factor": a, "regime": b} for a, b in cfg.scan_specs],
        "rows": int(len(df)),
        "dates": int(df["date"].nunique()),
        "symbols": int(df["symbol"].nunique()),
        "variant_rows": int(len(variants)),
        "summary_rows": int(len(summary)),
        "kept_rows": int(len(kept)),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    p_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    _stage_done("write_artifacts", tw, extra=f"summary_rows={len(summary)} kept_rows={len(kept)}")

    print("\n[SUMMARY] TOP REGIME RESULTS")
    if summary.empty:
        print("No summary rows produced")
    else:
        cols = [
            "factor",
            "regime",
            "raw_ic",
            "best_variant",
            "best_ic",
            "abs_uplift",
            "rel_uplift",
            "best_n_days",
            "pass_regime_uplift",
        ]
        print(summary[cols].head(100).to_string(index=False))

    print("\n[KEPT] REGIME-RETAINED CANDIDATES")
    if kept.empty:
        print("No candidates passed regime uplift filters")
    else:
        cols = [
            "factor",
            "regime",
            "raw_ic",
            "best_variant",
            "best_ic",
            "abs_uplift",
            "rel_uplift",
            "best_n_days",
        ]
        print(kept[cols].head(100).to_string(index=False))

    print(f"\n[ARTIFACT] {p_variants}")
    print(f"[ARTIFACT] {p_summary}")
    print(f"[ARTIFACT] {p_kept}")
    print(f"[ARTIFACT] {p_meta}")
    print(f"[ARTIFACT] {p_feature_snapshot}")
    print(f"[FINAL] scan_specs={len(cfg.scan_specs)} summary_rows={len(summary)} kept_rows={len(kept)}")
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 1
    _press_enter_exit(rc)

