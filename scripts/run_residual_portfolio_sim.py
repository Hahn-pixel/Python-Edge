# scripts/run_residual_portfolio_sim.py
# Production-style residual portfolio simulator
# Double-click runnable. Never auto-closes (always waits for Enter).
#
# Key production changes in this version:
# - staggered / overlapping book
# - sleeve-based holding for HOLD_DAYS
# - per-day turnover from sleeve replacement
# - normalized gross exposure across active sleeves

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
    residual_signal_raw = residual_raw.copy() if str(cfg.residual_direction).strip().lower() == "long_high_short_low" else (-residual_raw).copy()

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


def _activation_offset(entry_mode: str, delay_days: int) -> int:
    mode = str(entry_mode).strip().lower()
    if mode == "same_close":
        return delay_days
    if mode in {"next_open", "next_close"}:
        return delay_days + 1
    raise RuntimeError(f"Unsupported ENTRY_MODE={entry_mode!r}")


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
    # Each sleeve gets 1 / HOLD_DAYS of portfolio risk budget.
    port = normalize_side_weights(
        port,
        sized_col="side_sized",
        out_col="target_weight",
        gross_target=cfg.gross_target / max(1, cfg.hold_days),
        side_target=cfg.side_target / max(1, cfg.hold_days),
        max_weight_per_name=cfg.max_weight_per_name / max(1, cfg.hold_days),
    )
    return port


def _schedule_targets(targets: pd.DataFrame, dates: List[str], cfg: Config) -> Dict[str, pd.DataFrame]:
    offset = _activation_offset(cfg.entry_mode, cfg.delay_days)
    date_to_idx = {d: i for i, d in enumerate(dates)}
    scheduled: Dict[str, List[pd.DataFrame]] = {}
    for d, g in targets.groupby("date", sort=False):
        d_str = str(d)
        if d_str not in date_to_idx:
            continue
        act_idx = date_to_idx[d_str] + offset
        if act_idx >= len(dates):
            continue
        act_date = dates[act_idx]
        gg = g.copy()
        gg["signal_date"] = d_str
        gg["activation_date"] = act_date
        scheduled.setdefault(act_date, []).append(gg)
    merged: Dict[str, pd.DataFrame] = {}
    for d, parts in scheduled.items():
        merged[d] = pd.concat(parts, ignore_index=False)
    return merged


def _frame_to_weights(g: pd.DataFrame) -> Dict[str, float]:
    x = g[["symbol", "target_weight"]].copy()
    x["target_weight"] = pd.to_numeric(x["target_weight"], errors="coerce").fillna(0.0)
    x = x.groupby("symbol", as_index=False)["target_weight"].sum()
    x = x[x["target_weight"].abs() > 0.0]
    return {str(r.symbol): float(r.target_weight) for r in x.itertuples(index=False)}


def _gross(weights: Dict[str, float]) -> float:
    return float(sum(abs(v) for v in weights.values()))


def _net(weights: Dict[str, float]) -> float:
    return float(sum(weights.values()))


def _merge_sleeves(sleeves: List[Dict[str, float]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for sl in sleeves:
        for sym, w in sl.items():
            out[sym] = out.get(sym, 0.0) + float(w)
    out = {k: v for k, v in out.items() if abs(v) > 1e-12}
    return out


def _turnover_abs(old: Dict[str, float], new: Dict[str, float]) -> float:
    syms = set(old.keys()) | set(new.keys())
    return float(sum(abs(new.get(s, 0.0) - old.get(s, 0.0)) for s in syms))


def simulate_portfolio(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    feat = add_features(df)
    scored = build_residual_score(feat, cfg)
    scored = scored.dropna(subset=["ret_1d", "score", "c"]).copy()
    dates = sorted(pd.Series(scored["date"]).astype(str).unique().tolist())
    if len(dates) < cfg.hold_days + 5:
        raise RuntimeError("Not enough dates for staggered simulation")

    targets = build_daily_targets(scored, cfg)
    scheduled = _schedule_targets(targets, dates, cfg)
    ret_map_by_date: Dict[str, Dict[str, float]] = {}
    for d, g in scored.groupby("date", sort=False):
        g2 = g[["symbol", "ret_1d"]].copy()
        g2["ret_1d"] = pd.to_numeric(g2["ret_1d"], errors="coerce").fillna(0.0)
        ret_map_by_date[str(d)] = {str(r.symbol): float(r.ret_1d) for r in g2.itertuples(index=False)}

    sleeves: List[Dict[str, float]] = [dict() for _ in range(max(1, cfg.hold_days))]
    daily_rows: List[Dict[str, float]] = []
    active_rows: List[pd.DataFrame] = []

    for di, d in enumerate(dates):
        # 1) PnL from currently active sleeves using today's close-to-close returns.
        combined_before = _merge_sleeves(sleeves)
        ret_map = ret_map_by_date.get(d, {})
        day_ret_gross = float(sum(w * ret_map.get(sym, 0.0) for sym, w in combined_before.items()))
        gross_before = _gross(combined_before)
        net_before = _net(combined_before)
        npos_before = int(sum(1 for _, w in combined_before.items() if abs(w) > 0.0))

        # 2) Replace one sleeve with newly activated targets for this date.
        sleeve_idx = di % max(1, cfg.hold_days)
        old_sleeve = sleeves[sleeve_idx]
        new_sleeve_df = scheduled.get(d)
        new_sleeve = _frame_to_weights(new_sleeve_df) if new_sleeve_df is not None else {}
        turnover = _turnover_abs(old_sleeve, new_sleeve)
        tc_cost = turnover * (cfg.tc_bps_one_way / 10000.0)
        sleeves[sleeve_idx] = new_sleeve

        combined_after = _merge_sleeves(sleeves)
        gross_after = _gross(combined_after)
        net_after = _net(combined_after)
        npos_after = int(sum(1 for _, w in combined_after.items() if abs(w) > 0.0))

        daily_rows.append({
            "date": d,
            "gross_exposure_before": gross_before,
            "net_exposure_before": net_before,
            "n_positions_before": npos_before,
            "day_ret_gross": day_ret_gross,
            "turnover": turnover,
            "tc_cost": tc_cost,
            "day_ret_net": day_ret_gross - tc_cost,
            "gross_exposure_after": gross_after,
            "net_exposure_after": net_after,
            "n_positions_after": npos_after,
        })

        if new_sleeve_df is not None and not new_sleeve_df.empty:
            act = new_sleeve_df.copy()
            act["sleeve_idx"] = sleeve_idx
            act["sim_date"] = d
            active_rows.append(act)

    daily = pd.DataFrame(daily_rows)
    daily["equity"] = (1.0 + pd.to_numeric(daily["day_ret_net"], errors="coerce").fillna(0.0)).cumprod()
    active = pd.concat(active_rows, ignore_index=True) if active_rows else pd.DataFrame()
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

    ann_ret_like = float(pd.to_numeric(daily["day_ret_net"], errors="coerce").fillna(0.0).mean() * 252.0)
    ann_vol_like = float(pd.to_numeric(daily["day_ret_net"], errors="coerce").fillna(0.0).std(ddof=1) * np.sqrt(252.0)) if len(daily) > 1 else float("nan")
    sharpe_like = float(ann_ret_like / ann_vol_like) if ann_vol_like and ann_vol_like > 0 else float("nan")

    print(f"[SIM] days={len(daily)} active_rows={len(active)}")
    print(f"[SIM] equity_end={daily['equity'].iloc[-1]:.4f} ann_ret~={ann_ret_like:+0.4f} ann_vol~={ann_vol_like:+0.4f} sharpe~={sharpe_like:+0.4f}")
    print(f"[SIM] avg_gross_before={daily['gross_exposure_before'].mean():0.4f} avg_net_before={daily['net_exposure_before'].mean():+0.4f} avg_turnover={daily['turnover'].mean():0.4f}")
    print(f"[SIM] avg_positions_before={daily['n_positions_before'].mean():0.2f} avg_tc_cost={daily['tc_cost'].mean():0.6f}")

    print("\n[SIM] daily head")
    print(daily.head(20).to_string(index=False))

    if not active.empty:
        print("\n[SIM] activated sleeve positions head")
        show_cols = [c for c in ["sim_date", "signal_date", "activation_date", "sleeve_idx", "date", "symbol", "score", "side", "side_sized", "target_weight", "ret_realized", "anchor_ret", "residual_raw"] if c in active.columns]
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
