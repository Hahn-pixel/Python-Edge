from __future__ import annotations

import math
from typing import Iterable, Sequence

import pandas as pd

EPS = 1e-12


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
        z = (s - med) / (1.4826 * mad)
        return z.fillna(0.0)
    mean = float(valid.mean())
    std = float(valid.std())
    if std > EPS:
        return ((s - mean) / std).fillna(0.0)
    return pd.Series(0.0, index=s.index, dtype="float64")


def _ensure_float_column(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return _safe_series(df[col]).fillna(default)


def _build_anchor_return(out: pd.DataFrame, market_col: str, sector_col: str, beta_col: str) -> pd.Series:
    market = _ensure_float_column(out, market_col, 0.0)
    sector = _ensure_float_column(out, sector_col, 0.0)
    beta = _ensure_float_column(out, beta_col, 1.0)
    return beta * market + sector


def _build_factor_blend(
    out: pd.DataFrame,
    factor_cols: Sequence[str],
    factor_weights: Sequence[float],
) -> pd.Series:
    if len(factor_cols) != len(factor_weights):
        raise ValueError("factor_cols and factor_weights must have the same length")
    blend = pd.Series(0.0, index=out.index, dtype="float64")
    for col, w in zip(factor_cols, factor_weights):
        if abs(float(w)) <= EPS:
            continue
        blend = blend + float(w) * _ensure_float_column(out, col, 0.0)
    return blend


def _build_residual_signal(
    out: pd.DataFrame,
    realized_ret_col: str,
    market_col: str,
    sector_col: str,
    beta_col: str,
    factor_cols: Sequence[str],
    factor_weights: Sequence[float],
    invert_residual: bool,
) -> pd.DataFrame:
    ret_realized = _ensure_float_column(out, realized_ret_col, 0.0)
    anchor_ret = _build_anchor_return(out, market_col=market_col, sector_col=sector_col, beta_col=beta_col)
    factor_blend = _build_factor_blend(out, factor_cols=factor_cols, factor_weights=factor_weights)
    residual_raw = ret_realized - anchor_ret - factor_blend
    if invert_residual:
        residual_signal_raw = -residual_raw
    else:
        residual_signal_raw = residual_raw
    out["ret_realized"] = ret_realized
    out["anchor_ret"] = anchor_ret
    out["factor_blend"] = factor_blend
    out["residual_raw"] = residual_raw
    out["residual_signal_raw"] = residual_signal_raw
    return out


def build_cross_sectional_signal(
    df: pd.DataFrame,
    score_col: str = "score",
    date_col: str = "date",
    winsor_lower_q: float = 0.02,
    winsor_upper_q: float = 0.98,
    conf_floor: float = 0.35,
    conf_cap: float = 1.25,
    final_score_cap: float = 6.0,
    signal_mode: str = "residual_stat_arb",
    realized_ret_col: str = "ret_1d",
    market_col: str = "mkt_ret_1d",
    sector_col: str = "sector_ret_1d",
    beta_col: str = "beta_20d",
    factor_cols: Sequence[str] = ("mom_1d", "mom_3d", "rv_10", "ema_dist", "ema_fast_slope", "ema_slow_slope"),
    factor_weights: Sequence[float] = (0.35, 0.20, 0.15, 0.10, 0.10, 0.10),
    invert_residual: bool = True,
    residual_quality_floor: float = 0.20,
) -> pd.DataFrame:
    out = df.copy()
    required = [date_col]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise RuntimeError(f"build_cross_sectional_signal: missing columns: {missing}")
    if not (0.0 <= winsor_lower_q < winsor_upper_q <= 1.0):
        raise ValueError("winsor quantiles must satisfy 0 <= lower < upper <= 1")
    if conf_floor <= 0.0:
        raise ValueError("conf_floor must be > 0")
    if conf_cap <= 0.0:
        raise ValueError("conf_cap must be > 0")
    if final_score_cap <= 0.0:
        raise ValueError("final_score_cap must be > 0")
    if residual_quality_floor <= 0.0:
        raise ValueError("residual_quality_floor must be > 0")

    mode = str(signal_mode).strip().lower()
    if mode not in {"raw_score", "residual_stat_arb"}:
        raise RuntimeError(f"build_cross_sectional_signal: unsupported signal_mode={signal_mode!r}")

    if mode == "raw_score":
        if score_col not in out.columns:
            raise RuntimeError(f"build_cross_sectional_signal: missing score_col={score_col!r}")
        out["score_input"] = _safe_series(out[score_col])
        out["signal_mode"] = "raw_score"
        out["ret_realized"] = 0.0
        out["anchor_ret"] = 0.0
        out["factor_blend"] = 0.0
        out["residual_raw"] = 0.0
        out["residual_signal_raw"] = 0.0
    else:
        out = _build_residual_signal(
            out,
            realized_ret_col=realized_ret_col,
            market_col=market_col,
            sector_col=sector_col,
            beta_col=beta_col,
            factor_cols=factor_cols,
            factor_weights=factor_weights,
            invert_residual=invert_residual,
        )
        out["score_input"] = _safe_series(out["residual_signal_raw"])
        out["signal_mode"] = "residual_stat_arb"

    out["score_raw"] = _safe_series(out["score_input"])
    out["score_winsor"] = 0.0
    out["score_cs"] = 0.0
    out["score_conf"] = 1.0
    out["score_final"] = 0.0
    out["cs_dispersion"] = 0.0
    out["cs_top_bottom_spread"] = 0.0
    out["cs_signal_breadth"] = 0.0
    out["cs_nonzero_frac"] = 0.0
    out["cs_signal_count"] = 0
    out["cs_signal_quality_flag"] = 0
    out["score_rank_pct"] = 0.5
    out["score_abs_rank_pct"] = 0.5
    out["residual_abs_rank_pct"] = 0.5
    out["residual_quality"] = 0.0
    out["fresh_dislocation_flag"] = 0

    for _, idx in out.groupby(date_col).groups.items():
        raw = _safe_series(out.loc[idx, "score_raw"])
        wins = _winsorize_series(raw, winsor_lower_q, winsor_upper_q)
        z = _robust_zscore(wins).clip(lower=-final_score_cap, upper=final_score_cap)
        valid = z.dropna()
        if valid.empty:
            out.loc[idx, "score_winsor"] = wins.fillna(0.0)
            out.loc[idx, "score_cs"] = 0.0
            out.loc[idx, "score_conf"] = conf_floor
            out.loc[idx, "score_final"] = 0.0
            out.loc[idx, "cs_signal_quality_flag"] = 1
            continue

        dispersion = float(valid.std())
        top_q = float(valid.quantile(0.90))
        bot_q = float(valid.quantile(0.10))
        top_bottom_spread = float(top_q - bot_q)
        signal_breadth = float((valid.abs() >= 1.0).mean())
        nonzero_frac = float((valid.abs() > 0.0).mean())
        signal_count = int(valid.shape[0])

        conf_raw = 0.50 * min(2.0, dispersion) + 0.50 * min(2.0, top_bottom_spread / 2.0)
        conf = max(conf_floor, min(conf_cap, conf_raw))
        quality_flag = int(signal_count < 20 or top_bottom_spread < 0.25 or dispersion < residual_quality_floor)
        if quality_flag == 1:
            conf = max(conf_floor, min(conf, 0.75))

        rank_pct = valid.rank(method="average", pct=True).reindex(z.index).fillna(0.5)
        abs_rank_pct = valid.abs().rank(method="average", pct=True).reindex(z.index).fillna(0.5)
        final_score = (z * conf).clip(lower=-final_score_cap, upper=final_score_cap)
        fresh_flag = ((z.abs() >= 1.5) & (abs_rank_pct >= 0.90)).astype(int)

        out.loc[idx, "score_winsor"] = wins.fillna(0.0)
        out.loc[idx, "score_cs"] = z.fillna(0.0)
        out.loc[idx, "score_conf"] = conf
        out.loc[idx, "score_final"] = final_score.fillna(0.0)
        out.loc[idx, "cs_dispersion"] = dispersion
        out.loc[idx, "cs_top_bottom_spread"] = top_bottom_spread
        out.loc[idx, "cs_signal_breadth"] = signal_breadth
        out.loc[idx, "cs_nonzero_frac"] = nonzero_frac
        out.loc[idx, "cs_signal_count"] = signal_count
        out.loc[idx, "cs_signal_quality_flag"] = quality_flag
        out.loc[idx, "score_rank_pct"] = rank_pct.astype("float64")
        out.loc[idx, "score_abs_rank_pct"] = abs_rank_pct.astype("float64")
        out.loc[idx, "residual_abs_rank_pct"] = abs_rank_pct.astype("float64")
        out.loc[idx, "residual_quality"] = conf
        out.loc[idx, "fresh_dislocation_flag"] = fresh_flag.astype("int64")

    return out