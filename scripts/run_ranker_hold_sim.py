# scripts/run_ranker_hold_sim.py
# Python-Edge / massive dataset
# Residual Stat-Arb runner
# Double-click runnable. Never auto-closes (always waits for Enter).

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
_SRC_ROOT = _REPO_ROOT / "src"
for _p in (str(_REPO_ROOT), str(_SRC_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from python_edge.model.cross_sectional_signal import build_cross_sectional_signal
from python_edge.portfolio.construct import build_long_short_portfolio
from python_edge.portfolio.signal_sizing import apply_signal_strength_sizing
from python_edge.portfolio.exit_rules import apply_residual_exit_stack


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


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


@dataclass(frozen=True)
class Config:
    dataset_root: Path
    start: str
    end: str
    hold_days: int
    signal_delay_days: int
    entry_mode: str
    alpha_horizons: Tuple[int, ...]
    alpha_quantiles: Tuple[float, ...]
    top_pct: float
    bottom_pct: float
    min_abs_rank_pct: float
    require_fresh_dislocation: bool
    sizing_preset: str
    max_names_per_side: int
    seed: int
    debug: bool


def load_config() -> Config:
    return Config(
        dataset_root=Path(_env_str("DATASET_ROOT", r"D:\massive_dataset")),
        start=_env_str("START", "2023-01-01"),
        end=_env_str("END", "2026-02-28"),
        hold_days=_env_int("HOLD_DAYS", 2),
        signal_delay_days=max(0, _env_int("SIGNAL_DELAY_DAYS", 0)),
        entry_mode=_env_str("ENTRY_MODE", "next_open").lower(),
        alpha_horizons=tuple(int(x) for x in _env_str("ALPHA_HORIZONS", "1,2,3").split(",") if str(x).strip()),
        alpha_quantiles=tuple(float(x) for x in _env_str("ALPHA_QUANTILES", "0.02,0.05,0.10").split(",") if str(x).strip()),
        top_pct=_env_float("TOP_PCT", 0.10),
        bottom_pct=_env_float("BOTTOM_PCT", 0.10),
        min_abs_rank_pct=_env_float("MIN_ABS_RANK_PCT", 0.80),
        require_fresh_dislocation=_env_bool("REQUIRE_FRESH_DISLOCATION", False),
        sizing_preset=_env_str("SIZING_PRESET", "residual_stat_arb"),
        max_names_per_side=_env_int("MAX_NAMES_PER_SIDE", 10),
        seed=_env_int("SEED", 7),
        debug=_env_bool("DEBUG_RESIDUAL", True),
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
    df = df[["date", "symbol", "o", "h", "l", "c", "v"]].dropna().sort_values(["date"]).reset_index(drop=True)
    return df


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
    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)
    return df


def add_daily_features(df: pd.DataFrame) -> pd.DataFrame:
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

    for c in ("ret_1d", "ret_3d", "ret_5d", "rv_10", "mom_1d", "mom_3d", "ema_dist", "ema_fast_slope", "ema_slow_slope", "mkt_ret_1d", "sector_ret_1d", "beta_20d"):
        out[c] = pd.to_numeric(out[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return out


def _validate_entry_mode(entry_mode: str) -> str:
    mode = str(entry_mode).strip().lower()
    allowed = {"next_open", "next_close", "same_close"}
    if mode not in allowed:
        raise RuntimeError(f"Unsupported ENTRY_MODE={entry_mode!r}. Allowed: {sorted(allowed)}")
    return mode


def add_delay_forward_returns(df: pd.DataFrame, delay_days: int, entry_mode: str, horizons: Tuple[int, ...]) -> pd.DataFrame:
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


def build_delay_surface(scored_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = scored_df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date", "symbol", "score", "c", "o"]).copy()
    out = add_delay_forward_returns(out, cfg.signal_delay_days, cfg.entry_mode, cfg.alpha_horizons)
    out["score_rank_pct"] = out.groupby("date")["score"].rank(method="average", pct=True)
    rows: List[Dict[str, object]] = []
    for q in cfg.alpha_quantiles:
        long_mask = out["score_rank_pct"] >= (1.0 - q)
        short_mask = out["score_rank_pct"] <= q
        for h in cfg.alpha_horizons:
            col = f"entry_ret_{h}d"
            long_vals = pd.to_numeric(out.loc[long_mask, col], errors="coerce").dropna()
            short_vals = pd.to_numeric(out.loc[short_mask, col], errors="coerce").dropna()
            long_mean = float(long_vals.mean()) if len(long_vals) else float("nan")
            short_mean = float(short_vals.mean()) if len(short_vals) else float("nan")
            spread_mean = long_mean - short_mean if pd.notna(long_mean) and pd.notna(short_mean) else float("nan")
            rows.append({"bucket": f"top_{int(round(q * 100.0))}pct", "side": "long", "delay_days": cfg.signal_delay_days, "entry_mode": cfg.entry_mode, "horizon_days": h, "mean_ret": long_mean, "win_rate": float((long_vals > 0.0).mean()) if len(long_vals) else float("nan"), "n_obs": int(len(long_vals))})
            rows.append({"bucket": f"bottom_{int(round(q * 100.0))}pct", "side": "short", "delay_days": cfg.signal_delay_days, "entry_mode": cfg.entry_mode, "horizon_days": h, "mean_ret": short_mean, "win_rate": float((short_vals < 0.0).mean()) if len(short_vals) else float("nan"), "n_obs": int(len(short_vals))})
            rows.append({"bucket": f"spread_{int(round(q * 100.0))}pct", "side": "long_short", "delay_days": cfg.signal_delay_days, "entry_mode": cfg.entry_mode, "horizon_days": h, "mean_ret": spread_mean, "win_rate": float("nan"), "n_obs": int(min(len(long_vals), len(short_vals)))})
    return pd.DataFrame(rows)


def print_delay_surface_summary(surface_df: pd.DataFrame) -> None:
    if surface_df.empty:
        print("[ALPHA] delay surface empty")
        return
    cols = ["entry_mode", "delay_days", "bucket", "side", "horizon_days", "mean_ret", "win_rate", "n_obs"]
    print("[ALPHA] entry-mode / delay / horizon surface")
    print(surface_df[cols].to_string(index=False))


def _select_book(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = build_long_short_portfolio(df, top_pct=cfg.top_pct, bottom_pct=cfg.bottom_pct, score_col="score", date_col="date", min_abs_rank_pct=cfg.min_abs_rank_pct, require_fresh_dislocation=cfg.require_fresh_dislocation, fresh_flag_col="fresh_dislocation_flag", abs_rank_col="score_abs_rank_pct")
    out = apply_signal_strength_sizing(out, side_col="side", score_col="score", out_col="side_sized", preset_name=cfg.sizing_preset)
    out["hold_age_days"] = 0
    out = apply_residual_exit_stack(out, age_col="hold_age_days", side_col="side", score_col="score", max_hold_days=cfg.hold_days, abs_exit_threshold=0.35, out_col="exit_flag")
    return out


def summarize_book(book: pd.DataFrame) -> None:
    if book.empty:
        print("[BOOK] empty")
        return
    longs = int((pd.to_numeric(book["side"], errors="coerce").fillna(0.0) > 0).sum())
    shorts = int((pd.to_numeric(book["side"], errors="coerce").fillna(0.0) < 0).sum())
    fresh = int(pd.to_numeric(book.get("fresh_dislocation_flag", 0), errors="coerce").fillna(0).sum())
    qflag = int(pd.to_numeric(book.get("cs_signal_quality_flag", 0), errors="coerce").fillna(0).sum())
    print(f"[BOOK] rows={len(book)} longs={longs} shorts={shorts} fresh_dislocations={fresh} quality_flag_rows={qflag}")


def main() -> int:
    cfg = load_config()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    _validate_entry_mode(cfg.entry_mode)

    print(f"[CFG] dataset_root={cfg.dataset_root}")
    print(f"[CFG] start={cfg.start} end={cfg.end}")
    print(f"[CFG] entry_mode={cfg.entry_mode} signal_delay_days={cfg.signal_delay_days} hold_days={cfg.hold_days}")
    print(f"[CFG] top_pct={cfg.top_pct} bottom_pct={cfg.bottom_pct} min_abs_rank_pct={cfg.min_abs_rank_pct} require_fresh_dislocation={cfg.require_fresh_dislocation}")
    print(f"[CFG] alpha_horizons={cfg.alpha_horizons} alpha_quantiles={cfg.alpha_quantiles} sizing_preset={cfg.sizing_preset}")

    raw = load_dataset(cfg)
    feat = add_daily_features(raw)
    feat = feat.dropna(subset=["ret_1d", "mom_1d", "mom_3d", "rv_10", "ema_dist", "ema_fast_slope", "ema_slow_slope", "mkt_ret_1d", "sector_ret_1d", "beta_20d"]).copy()

    scored = build_cross_sectional_signal(feat, signal_mode="residual_stat_arb", realized_ret_col="ret_1d", market_col="mkt_ret_1d", sector_col="sector_ret_1d", beta_col="beta_20d", factor_cols=("mom_1d", "mom_3d", "rv_10", "ema_dist", "ema_fast_slope", "ema_slow_slope"), factor_weights=(0.35, 0.20, 0.15, 0.10, 0.10, 0.10), invert_residual=True, residual_quality_floor=0.20)
    scored["score"] = pd.to_numeric(scored["score_final"], errors="coerce").fillna(0.0)

    if cfg.debug:
        print(f"[DATA] rows={len(scored)} dates={scored['date'].nunique()} symbols={scored['symbol'].nunique()}")
        print(f"[DATA] cs_quality_flag_rate={pd.to_numeric(scored['cs_signal_quality_flag'], errors='coerce').fillna(0).mean():.4f}")
        print(f"[DATA] fresh_dislocation_rate={pd.to_numeric(scored['fresh_dislocation_flag'], errors='coerce').fillna(0).mean():.4f}")

    surface = build_delay_surface(scored, cfg)
    print_delay_surface_summary(surface)

    latest_date = str(pd.Series(scored["date"]).dropna().astype(str).max())
    book = scored[scored["date"].astype(str) == latest_date].copy()
    book = _select_book(book, cfg)
    summarize_book(book)

    if not book.empty:
        show_cols = [c for c in ["date", "symbol", "score", "side", "side_sized", "conviction_bucket", "fresh_dislocation_flag", "score_abs_rank_pct", "ret_realized", "anchor_ret", "residual_raw"] if c in book.columns]
        print("[BOOK] latest date selection")
        print(book[show_cols].sort_values(["score"], ascending=False).head(20).to_string(index=False))

    _log("[DONE] Residual Stat-Arb runner completed.")
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        _log("[FATAL] unhandled exception")
        traceback.print_exc()
        rc = 1
    _press_enter_exit(int(rc))
