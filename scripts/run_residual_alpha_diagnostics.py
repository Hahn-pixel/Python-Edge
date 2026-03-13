# scripts/run_residual_alpha_diagnostics.py
# Residual alpha diagnostics for Python-Edge / massive dataset
# Double-click runnable. Never auto-closes (always waits for Enter).
#
# Diagnostics:
# - decile / quantile return curve
# - horizon decay
# - rank IC (Spearman/Pearson proxy)
# - entry-mode comparison

from __future__ import annotations

import json
import os
import random
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None or not str(v).strip():
        return default
    try:
        return float(str(v).strip())
    except Exception:
        return default


def _safe_series(x: pd.Series) -> pd.Series:
    return pd.to_numeric(x, errors="coerce").astype("float64")


def _winsorize_series(x: pd.Series, lower_q: float, upper_q: float) -> pd.Series:
    s = _safe_series(x)
    valid = s.dropna()
    if valid.empty:
        return s
    lo = float(valid.quantile(lower_q))
    hi = float(valid.quantile(upper_q))
    return s.clip(lower=lo, upper=hi)


def _robust_zscore(x: pd.Series) -> pd.Series:
    s = _safe_series(x)
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


@dataclass(frozen=True)
class Config:
    dataset_root: Path
    start: str
    end: str
    residual_direction: str
    seed: int
    debug: bool
    quantiles: Tuple[float, ...]
    horizons: Tuple[int, ...]
    entry_modes: Tuple[str, ...]
    deciles: int


def load_config() -> Config:
    quantiles = tuple(float(x) for x in _env_str("EDGE_QUANTILES", "0.01,0.02,0.05,0.10,0.20").split(",") if str(x).strip())
    horizons = tuple(int(x) for x in _env_str("EDGE_HORIZONS", "1,2,3,4,5").split(",") if str(x).strip())
    entry_modes = tuple(str(x).strip() for x in _env_str("EDGE_ENTRY_MODES", "next_open,next_close,same_close").split(",") if str(x).strip())
    return Config(
        dataset_root=Path(_env_str("DATASET_ROOT", r"D:\massive_dataset")),
        start=_env_str("START", "2023-01-01"),
        end=_env_str("END", "2026-02-28"),
        residual_direction=_env_str("RESIDUAL_DIRECTION", "long_high_short_low"),
        seed=_env_int("SEED", 7),
        debug=_env_str("DEBUG_RESIDUAL", "1") in ("1", "true", "True"),
        quantiles=quantiles,
        horizons=horizons,
        entry_modes=entry_modes,
        deciles=_env_int("DECILES", 10),
    )


def _find_aggs_1d_files(dataset_root: Path) -> List[Tuple[str, Path]]:
    out: List[Tuple[str, Path]] = []
    if not dataset_root.exists():
        raise RuntimeError(f"dataset_root not found: {dataset_root}")
    for sym_dir in sorted([p for p in dataset_root.iterdir() if p.is_dir()]):
        sym = sym_dir.name.strip().upper()
        candidates = sorted(sym_dir.glob("aggs_1d_*.json"))
        if not candidates:
            continue
        best = max(candidates, key=lambda p: p.stat().st_size)
        out.append((sym, best))
    return out


def _load_aggs_1d(sym: str, path: Path) -> pd.DataFrame:
    js = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    rows = js.get("results") or []
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "t" not in df.columns:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.date.astype(str)
    df["symbol"] = sym
    for c in ("o", "h", "l", "c", "v"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["date", "symbol", "o", "h", "l", "c", "v"]].dropna().sort_values(["date"]).reset_index(drop=True)


def load_dataset(cfg: Config) -> pd.DataFrame:
    pairs = _find_aggs_1d_files(cfg.dataset_root)
    dfs: List[pd.DataFrame] = []
    for sym, fp in pairs:
        d = _load_aggs_1d(sym, fp)
        if not d.empty:
            dfs.append(d)
    if not dfs:
        raise RuntimeError("No aggs_1d data loaded")
    df = pd.concat(dfs, ignore_index=True)
    df = df[(df["date"] >= cfg.start) & (df["date"] <= cfg.end)].copy()
    return df.sort_values(["date", "symbol"]).reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ("o", "h", "l", "c", "v"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["ret_1d"] = out.groupby("symbol")["c"].pct_change(1, fill_method=None)
    out["ret_3d"] = out.groupby("symbol")["c"].pct_change(3, fill_method=None)
    out["ret_5d"] = out.groupby("symbol")["c"].pct_change(5, fill_method=None)
    out["rv_10"] = out.groupby("symbol")["ret_1d"].rolling(10, min_periods=5).std().reset_index(level=0, drop=True)
    out["mom_1d"] = out["ret_1d"]
    out["mom_3d"] = out["ret_3d"]

    def _ema(s: pd.Series, span: int) -> pd.Series:
        return s.ewm(span=span, adjust=False, min_periods=max(5, span // 3)).mean()

    out["ema10"] = out.groupby("symbol")["c"].transform(lambda s: _ema(s, 10))
    out["ema20"] = out.groupby("symbol")["c"].transform(lambda s: _ema(s, 20))
    out["ema50"] = out.groupby("symbol")["c"].transform(lambda s: _ema(s, 50))
    out["ema_dist"] = (out["c"] - out["ema20"]) / out["ema20"].replace(0, np.nan)
    out["ema_fast_slope"] = out.groupby("symbol")["ema10"].pct_change(1, fill_method=None)
    out["ema_slow_slope"] = out.groupby("symbol")["ema50"].pct_change(1, fill_method=None)
    out["mkt_ret_1d"] = out.groupby("date")["ret_1d"].transform("mean")
    out["sector_bucket"] = out["symbol"].str[0].fillna("_")
    out["sector_ret_1d"] = out.groupby(["date", "sector_bucket"])["ret_1d"].transform("mean")

    out["beta_20d"] = np.nan
    for _, idx in out.groupby("symbol", sort=False).groups.items():
        g = out.loc[idx, ["ret_1d", "mkt_ret_1d"]].copy()
        cov20 = g["ret_1d"].rolling(20, min_periods=10).cov(g["mkt_ret_1d"])
        var20 = g["mkt_ret_1d"].rolling(20, min_periods=10).var()
        beta20 = (cov20 / var20.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        out.loc[idx, "beta_20d"] = beta20.to_numpy()
    return out


def build_residual_score(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = df.copy()
    ret_realized = _safe_series(out["ret_1d"]).fillna(0.0)
    market = _safe_series(out["mkt_ret_1d"]).fillna(0.0)
    sector = _safe_series(out["sector_ret_1d"]).fillna(0.0)
    beta = _safe_series(out["beta_20d"]).fillna(1.0)
    factor_blend = (
        0.35 * _safe_series(out["mom_1d"]).fillna(0.0)
        + 0.20 * _safe_series(out["mom_3d"]).fillna(0.0)
        + 0.15 * _safe_series(out["rv_10"]).fillna(0.0)
        + 0.10 * _safe_series(out["ema_dist"]).fillna(0.0)
        + 0.10 * _safe_series(out["ema_fast_slope"]).fillna(0.0)
        + 0.10 * _safe_series(out["ema_slow_slope"]).fillna(0.0)
    )
    anchor_ret = beta * market + sector
    residual_raw = ret_realized - anchor_ret - factor_blend
    residual_signal_raw = residual_raw.copy() if cfg.residual_direction == "long_high_short_low" else (-residual_raw).copy()
    out["residual_raw"] = residual_raw
    out["residual_signal_raw"] = residual_signal_raw
    out["score"] = 0.0

    for _, idx in out.groupby("date", sort=False).groups.items():
        raw = _winsorize_series(out.loc[idx, "residual_signal_raw"], 0.02, 0.98)
        z = _robust_zscore(raw).clip(lower=-6.0, upper=6.0)
        out.loc[idx, "score"] = z.fillna(0.0)
    return out


def _validate_entry_mode(entry_mode: str) -> str:
    mode = str(entry_mode).strip().lower()
    allowed = {"next_open", "next_close", "same_close"}
    if mode not in allowed:
        raise RuntimeError(f"Unsupported entry mode={entry_mode!r}. Allowed: {sorted(allowed)}")
    return mode


def add_forward_returns(df: pd.DataFrame, entry_mode: str, delay_days: int, horizons: Sequence[int]) -> pd.DataFrame:
    out = df.copy()
    mode = _validate_entry_mode(entry_mode)
    if mode == "same_close":
        entry_shift = delay_days
        out["entry_px"] = out.groupby("symbol")["c"].shift(-entry_shift)
        entry_day_idx = entry_shift
    elif mode == "next_close":
        entry_shift = delay_days + 1
        out["entry_px"] = out.groupby("symbol")["c"].shift(-entry_shift)
        entry_day_idx = entry_shift
    else:
        entry_shift = delay_days + 1
        out["entry_px"] = out.groupby("symbol")["o"].shift(-entry_shift)
        entry_day_idx = entry_shift

    for h in horizons:
        exit_shift = entry_day_idx + (h - 1)
        exit_px = out.groupby("symbol")["c"].shift(-exit_shift)
        out[f"entry_ret_{h}d"] = (exit_px / out["entry_px"]) - 1.0
    return out


def build_quantile_curve(scored: pd.DataFrame, cfg: Config, entry_mode: str, delay_days: int) -> pd.DataFrame:
    base = add_forward_returns(scored.copy(), entry_mode=entry_mode, delay_days=delay_days, horizons=cfg.horizons)
    base = base.dropna(subset=["date", "symbol", "score", "o", "c"]).copy()
    base["score_rank_pct"] = base.groupby("date")["score"].rank(method="average", pct=True)
    rows: List[Dict[str, object]] = []
    for q in cfg.quantiles:
        long_mask = base["score_rank_pct"] >= (1.0 - q)
        short_mask = base["score_rank_pct"] <= q
        for h in cfg.horizons:
            col = f"entry_ret_{h}d"
            long_vals = pd.to_numeric(base.loc[long_mask, col], errors="coerce").dropna()
            short_vals = pd.to_numeric(base.loc[short_mask, col], errors="coerce").dropna()
            rows.append({
                "entry_mode": entry_mode,
                "delay_days": int(delay_days),
                "quantile": float(q),
                "horizon_days": int(h),
                "long_mean": float(long_vals.mean()) if len(long_vals) else float("nan"),
                "short_mean": float(short_vals.mean()) if len(short_vals) else float("nan"),
                "spread_mean": float(long_vals.mean() - short_vals.mean()) if len(long_vals) and len(short_vals) else float("nan"),
                "long_n": int(len(long_vals)),
                "short_n": int(len(short_vals)),
            })
    return pd.DataFrame(rows)


def build_decile_curve(scored: pd.DataFrame, cfg: Config, entry_mode: str, delay_days: int) -> pd.DataFrame:
    base = add_forward_returns(scored.copy(), entry_mode=entry_mode, delay_days=delay_days, horizons=cfg.horizons)
    base = base.dropna(subset=["date", "symbol", "score", "o", "c"]).copy()
    base["decile"] = base.groupby("date")["score"].transform(lambda s: pd.qcut(s.rank(method="first"), q=cfg.deciles, labels=False, duplicates="drop"))
    rows: List[Dict[str, object]] = []
    for h in cfg.horizons:
        col = f"entry_ret_{h}d"
        for dec, g in base.groupby("decile", dropna=True):
            vals = pd.to_numeric(g[col], errors="coerce").dropna()
            rows.append({
                "entry_mode": entry_mode,
                "delay_days": int(delay_days),
                "horizon_days": int(h),
                "decile": int(dec) + 1,
                "mean_ret": float(vals.mean()) if len(vals) else float("nan"),
                "n_obs": int(len(vals)),
            })
    return pd.DataFrame(rows)


def build_ic_table(scored: pd.DataFrame, cfg: Config, entry_mode: str, delay_days: int) -> pd.DataFrame:
    base = add_forward_returns(scored.copy(), entry_mode=entry_mode, delay_days=delay_days, horizons=cfg.horizons)
    base = base.dropna(subset=["date", "score", "symbol"]).copy()
    rows: List[Dict[str, object]] = []
    for h in cfg.horizons:
        col = f"entry_ret_{h}d"
        daily_rows: List[Tuple[float, float]] = []
        for d, g in base.groupby("date", sort=False):
            gg = g[["score", col]].dropna().copy()
            if len(gg) < 5:
                continue
            pear = float(gg["score"].corr(gg[col], method="pearson"))
            spear = float(gg["score"].corr(gg[col], method="spearman"))
            daily_rows.append((pear, spear))
        pear_mean = float(np.nanmean([x[0] for x in daily_rows])) if daily_rows else float("nan")
        spear_mean = float(np.nanmean([x[1] for x in daily_rows])) if daily_rows else float("nan")
        rows.append({
            "entry_mode": entry_mode,
            "delay_days": int(delay_days),
            "horizon_days": int(h),
            "ic_pearson_mean": pear_mean,
            "ic_spearman_mean": spear_mean,
            "n_days": int(len(daily_rows)),
        })
    return pd.DataFrame(rows)


def save_csv(df: pd.DataFrame, name: str) -> Path:
    _ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = _ARTIFACT_DIR / name
    df.to_csv(path, index=False, encoding="utf-8")
    return path


def main() -> int:
    cfg = load_config()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    print(f"[CFG] dataset_root={cfg.dataset_root}")
    print(f"[CFG] start={cfg.start} end={cfg.end}")
    print(f"[CFG] residual_direction={cfg.residual_direction}")
    print(f"[CFG] quantiles={cfg.quantiles}")
    print(f"[CFG] horizons={cfg.horizons}")
    print(f"[CFG] entry_modes={cfg.entry_modes}")
    print(f"[CFG] deciles={cfg.deciles}")

    raw = load_dataset(cfg)
    feat = add_features(raw)
    feat = feat.dropna(subset=["ret_1d", "mom_1d", "mom_3d", "rv_10", "ema_dist", "ema_fast_slope", "ema_slow_slope", "mkt_ret_1d", "sector_ret_1d", "beta_20d"]).copy()
    scored = build_residual_score(feat, cfg)

    if cfg.debug:
        print(f"[DATA] rows={len(scored)} dates={scored['date'].nunique()} symbols={scored['symbol'].nunique()}")

    q_frames: List[pd.DataFrame] = []
    d_frames: List[pd.DataFrame] = []
    ic_frames: List[pd.DataFrame] = []
    for entry_mode in cfg.entry_modes:
        for delay in (0, 1, 2):
            q_frames.append(build_quantile_curve(scored, cfg, entry_mode, delay))
            d_frames.append(build_decile_curve(scored, cfg, entry_mode, delay))
            ic_frames.append(build_ic_table(scored, cfg, entry_mode, delay))

    quantile_curve = pd.concat(q_frames, ignore_index=True)
    decile_curve = pd.concat(d_frames, ignore_index=True)
    ic_table = pd.concat(ic_frames, ignore_index=True)

    horizon_decay = (
        quantile_curve.groupby(["entry_mode", "delay_days", "horizon_days"], as_index=False)["spread_mean"]
        .mean()
        .sort_values(["entry_mode", "delay_days", "horizon_days"])
        .reset_index(drop=True)
    )

    print("\n[ALPHA] quantile curve head")
    print(quantile_curve.head(40).to_string(index=False))

    print("\n[ALPHA] decile curve")
    print(decile_curve.to_string(index=False))

    print("\n[ALPHA] horizon decay")
    print(horizon_decay.to_string(index=False))

    print("\n[ALPHA] IC table")
    print(ic_table.to_string(index=False))

    p1 = save_csv(quantile_curve, "quantile_curve.csv")
    p2 = save_csv(decile_curve, "decile_curve.csv")
    p3 = save_csv(horizon_decay, "horizon_decay.csv")
    p4 = save_csv(ic_table, "ic_table.csv")
    print(f"\n[ARTIFACT] saved={p1}")
    print(f"[ARTIFACT] saved={p2}")
    print(f"[ARTIFACT] saved={p3}")
    print(f"[ARTIFACT] saved={p4}")

    _log("[DONE] Residual alpha diagnostics completed.")
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        _log("[FATAL] unhandled exception")
        traceback.print_exc()
        rc = 1
    _press_enter_exit(int(rc))
