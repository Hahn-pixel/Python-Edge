from __future__ import annotations

import math

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



def build_cross_sectional_signal(
    df: pd.DataFrame,
    score_col: str = "score",
    date_col: str = "date",
    winsor_lower_q: float = 0.02,
    winsor_upper_q: float = 0.98,
    conf_floor: float = 0.35,
    conf_cap: float = 1.25,
    final_score_cap: float = 6.0,
) -> pd.DataFrame:
    out = df.copy()
    required = [date_col, score_col]
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

    out["score_raw"] = _safe_series(out[score_col])
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
        quality_flag = int(signal_count < 20 or top_bottom_spread < 0.25 or dispersion < 0.20)
        if quality_flag == 1:
            conf = max(conf_floor, min(conf, 0.75))

        rank_pct = valid.rank(method="average", pct=True).reindex(z.index).fillna(0.5)
        abs_rank_pct = valid.abs().rank(method="average", pct=True).reindex(z.index).fillna(0.5)
        final_score = (z * conf).clip(lower=-final_score_cap, upper=final_score_cap)

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

    return out