# scripts/run_residual_portfolio_sim.py
# Production-style residual portfolio simulator
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
from typing import List, Tuple

import numpy as np
import pandas as pd

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
for _p in (str(_REPO_ROOT), str(_SRC_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from python_edge.portfolio.construct import build_long_short_portfolio
from python_edge.portfolio.signal_sizing import apply_signal_strength_sizing, normalize_side_weights

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
    top_pct: float
    bottom_pct: float
    min_abs_rank_pct: float
    max_names_per_side: int
    hold_days: int
    entry_mode: str
    delay_days: int
    gross_target: float
    side_target: float
    max_weight_per_name: float
    tc_bps_one_way: float
    sizing_preset: str
    seed: int


def load_config() -> Config:
    return Config(
        dataset_root=Path(_env_str("DATASET_ROOT", r"D:\massive_dataset")),
        start=_env_str("START", "2023-01-01"),
        end=_env_str("END", "2026-02-28"),
        residual_direction=_env_str("RESIDUAL_DIRECTION", "long_high_short_low"),
        top_pct=_env_float("TOP_PCT", 0.10),
        bottom_pct=_env_float("BOTTOM_PCT", 0.10),
        min_abs_rank_pct=_env_float("MIN_ABS_RANK_PCT", 0.80),
        max_names_per_side=_env_int("MAX_NAMES_PER_SIDE", 10),
        hold_days=_env_int("HOLD_DAYS", 3),
        entry_mode=_env_str("ENTRY_MODE", "next_open"),
        delay_days=max(0, _env_int("SIGNAL_DELAY_DAYS", 0)),
        gross_target=_env_float("GROSS_TARGET", 1.0),
        side_target=_env_float("SIDE_TARGET", 0.5),
        max_weight_per_name=_env_float("MAX_WEIGHT_PER_NAME", 0.10),
        tc_bps_one_way=_env_float("TC_BPS_ONE_WAY", 5.0),
        sizing_preset=_env_str("SIZING_PRESET", "production_residual"),
        seed=_env_int("SEED", 7),
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
    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ("o", "h", "l", "c", "v"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["ret_1d"] = out.groupby("symbol")["c"].pct_change(1, fill_method=None)
    out["ret_3d"] = out.groupby("symbol")["c"].pct_change(3, fill_method=None)
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
    if str(cfg.residual_direction).strip().lower() == "long_high_short_low":
        residual_signal_raw = residual_raw.copy()
    else:
        residual_signal_raw = (-residual_raw).copy()

    out["ret_realized"] = ret_realized
    out["anchor_ret"] = anchor_ret
    out["residual_raw"] = residual_raw
    out["residual_signal_raw"] = residual_signal_raw
    out["score"] = 0.0
    out["score_abs_rank_pct"] = 0.5
    out["fresh_dislocation_flag"] = 0

    for _, idx in out.groupby("date", sort=False).groups.items():
        raw = _winsorize_series(out.loc[idx, "residual_signal_raw"], 0.02, 0.98)
        z = _robust_zscore(raw).clip(lower=-6.0, upper=6.0)
        out.loc[idx, "score"] = z.fillna(0.0)
        abs_rank = z.abs().rank(method="average", pct=True).reindex(z.index).fillna(0.5)
        out.loc[idx, "score_abs_rank_pct"] = abs_rank.astype("float64")
        fresh = ((z.abs() >= 1.5) & (abs_rank >= 0.90)).astype(int)
        out.loc[idx, "fresh_dislocation_flag"] = fresh.astype("int64")
    return out


def _entry_return_frame(df: pd.DataFrame, entry_mode: str, delay_days: int, hold_days: int) -> pd.DataFrame:
    out = df.copy()
    mode = str(entry_mode).strip().lower()
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
    exit_shift = entry_day_idx + (hold_days - 1)
    out["exit_px"] = out.groupby("symbol")["c"].shift(-exit_shift)
    out["trade_ret"] = (out["exit_px"] / out["entry_px"]) - 1.0
    return out


def build_daily_targets(scored: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    port = build_long_short_portfolio(
        scored,
        top_pct=cfg.top_pct,
        bottom_pct=cfg.bottom_pct,
        score_col="score",
        date_col="date",
        min_abs_rank_pct=cfg.min_abs_rank_pct,
        require_fresh_dislocation=False,
        fresh_flag_col="fresh_dislocation_flag",
        abs_rank_col="score_abs_rank_pct",
        max_names_per_side=cfg.max_names_per_side,
    )
    port = apply_signal_strength_sizing(port, side_col="side", score_col="score", out_col="side_sized", preset_name=cfg.sizing_preset)
    port = normalize_side_weights(
        port,
        sized_col="side_sized",
        out_col="target_weight",
        gross_target=cfg.gross_target,
        side_target=cfg.side_target,
        max_weight_per_name=cfg.max_weight_per_name,
    )
    return port


def simulate_portfolio(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    scored = build_residual_score(add_features(df), cfg)
    scored = scored.dropna(subset=["ret_1d", "score", "o", "c"]).copy()
    frame = _entry_return_frame(scored, cfg.entry_mode, cfg.delay_days, cfg.hold_days)
    frame = frame.dropna(subset=["trade_ret"]).copy()
    targets = build_daily_targets(frame, cfg)
    active = targets[pd.to_numeric(targets["target_weight"], errors="coerce").fillna(0.0) != 0.0].copy()
    if active.empty:
        raise RuntimeError("No active positions after target construction")

    daily = (
        active.groupby("date", as_index=False)
        .apply(lambda g: pd.Series({
            "gross_exposure": float(pd.to_numeric(g["target_weight"], errors="coerce").abs().sum()),
            "net_exposure": float(pd.to_numeric(g["target_weight"], errors="coerce").sum()),
            "n_positions": int((pd.to_numeric(g["target_weight"], errors="coerce").abs() > 0).sum()),
            "day_ret_gross": float((pd.to_numeric(g["target_weight"], errors="coerce") * pd.to_numeric(g["trade_ret"], errors="coerce")).sum()),
        }))
        .reset_index(drop=True)
    )

    daily = daily.sort_values("date").reset_index(drop=True)
    turnover = daily["gross_exposure"].diff().abs().fillna(daily["gross_exposure"])
    tc = turnover * (cfg.tc_bps_one_way / 10000.0)
    daily["turnover"] = turnover
    daily["tc_cost"] = tc
    daily["day_ret_net"] = daily["day_ret_gross"] - daily["tc_cost"]
    daily["equity"] = (1.0 + daily["day_ret_net"]).cumprod()
    return daily, active


def main() -> int:
    cfg = load_config()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    print(f"[CFG] dataset_root={cfg.dataset_root}")
    print(f"[CFG] residual_direction={cfg.residual_direction}")
    print(f"[CFG] top_pct={cfg.top_pct} bottom_pct={cfg.bottom_pct} max_names_per_side={cfg.max_names_per_side}")
    print(f"[CFG] hold_days={cfg.hold_days} entry_mode={cfg.entry_mode} delay_days={cfg.delay_days}")
    print(f"[CFG] gross_target={cfg.gross_target} side_target={cfg.side_target} max_weight_per_name={cfg.max_weight_per_name}")
    print(f"[CFG] tc_bps_one_way={cfg.tc_bps_one_way} sizing_preset={cfg.sizing_preset}")

    raw = load_dataset(cfg)
    daily, active = simulate_portfolio(raw, cfg)

    ann_ret_like = float(daily["day_ret_net"].mean() * 252.0)
    ann_vol_like = float(daily["day_ret_net"].std(ddof=1) * np.sqrt(252.0)) if len(daily) > 1 else float("nan")
    sharpe_like = float(ann_ret_like / ann_vol_like) if ann_vol_like and ann_vol_like > 0 else float("nan")

    print(f"[SIM] days={len(daily)} active_rows={len(active)}")
    print(f"[SIM] equity_end={daily['equity'].iloc[-1]:.4f} ann_ret~={ann_ret_like:+0.4f} ann_vol~={ann_vol_like:+0.4f} sharpe~={sharpe_like:+0.4f}")
    print(f"[SIM] avg_gross={daily['gross_exposure'].mean():0.4f} avg_net={daily['net_exposure'].mean():+0.4f} avg_turnover={daily['turnover'].mean():0.4f}")
    print(f"[SIM] avg_positions={daily['n_positions'].mean():0.2f} avg_tc_cost={daily['tc_cost'].mean():0.6f}")

    print("\n[SIM] daily head")
    print(daily.head(20).to_string(index=False))

    print("\n[SIM] active positions head")
    show_cols = [c for c in ["date", "symbol", "score", "side", "side_sized", "target_weight", "trade_ret", "ret_realized", "anchor_ret", "residual_raw"] if c in active.columns]
    print(active[show_cols].head(30).to_string(index=False))

    _log("[DONE] Residual production portfolio simulator completed.")
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        _log("[FATAL] unhandled exception")
        traceback.print_exc()
        rc = 1
    _press_enter_exit(int(rc))
