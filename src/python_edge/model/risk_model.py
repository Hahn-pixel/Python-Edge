from __future__ import annotations

import pandas as pd


EPS = 1e-12



def _safe_num(x: pd.Series) -> pd.Series:
    return pd.to_numeric(x, errors="coerce").astype("float64")



def _cross_section_rank01(x: pd.Series) -> pd.Series:
    s = _safe_num(x)
    valid = s.dropna()
    if valid.empty:
        return pd.Series(0.5, index=s.index, dtype="float64")
    return s.rank(method="average", pct=True).fillna(0.5)



def _ensure_vol_proxy(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "risk_vol_proxy" in out.columns:
        out["risk_vol_proxy"] = _safe_num(out["risk_vol_proxy"]).abs().fillna(0.0)
        return out

    candidates = [
        "ivol_20d",
        "idiosyncratic_vol",
        "realized_vol_20d",
        "rv_20d",
        "volatility_20d",
        "vol_20d",
        "atr_pct_14",
        "atr_pct",
    ]
    for col in candidates:
        if col in out.columns:
            out["risk_vol_proxy"] = _safe_num(out[col]).abs().fillna(0.0)
            return out

    out["risk_vol_proxy"] = 0.0
    return out



def build_risk_model(
    df: pd.DataFrame,
    score_col: str = "score_final",
    date_col: str = "date",
    symbol_col: str = "symbol",
    beta_penalty: float = 0.35,
    liq_penalty: float = 0.35,
    vol_penalty: float = 0.45,
    market_regime_penalty: float = 0.15,
    risk_floor: float = 0.35,
    risk_cap: float = 3.50,
) -> pd.DataFrame:
    out = df.copy()
    required = [date_col, symbol_col, score_col, "beta_proxy_60d", "liq_rank", "meta_dollar_volume", "meta_price"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise RuntimeError(f"build_risk_model: missing columns: {missing}")

    out = _ensure_vol_proxy(out)

    out["score_final"] = _safe_num(out[score_col]).fillna(0.0)
    out["beta_proxy_60d"] = _safe_num(out["beta_proxy_60d"]).fillna(0.0)
    out["liq_rank"] = _safe_num(out["liq_rank"]).fillna(0.5)
    out["meta_dollar_volume"] = _safe_num(out["meta_dollar_volume"]).fillna(0.0)
    out["meta_price"] = _safe_num(out["meta_price"]).fillna(0.0)
    out["risk_vol_proxy"] = _safe_num(out["risk_vol_proxy"]).abs().fillna(0.0)

    out["risk_beta_rank"] = 0.5
    out["risk_vol_rank"] = 0.5
    out["risk_liq_penalty"] = 0.0
    out["risk_market_penalty"] = 0.0
    out["risk_unit"] = 1.0
    out["score_risk_adj"] = 0.0
    out["score_alpha_to_risk"] = 0.0
    out["risk_penalty_rate"] = 0.0
    out["risk_quality_flag"] = 0

    for _, idx in out.groupby(date_col).groups.items():
        beta_abs = out.loc[idx, "beta_proxy_60d"].abs()
        beta_rank = _cross_section_rank01(beta_abs)
        vol_rank = _cross_section_rank01(out.loc[idx, "risk_vol_proxy"])

        liq_penalty_series = 1.0 - out.loc[idx, "liq_rank"].clip(lower=0.0, upper=1.0)
        liq_penalty_series = liq_penalty_series.clip(lower=0.0, upper=1.0)

        if "market_breadth" in out.columns:
            breadth_val = float(_safe_num(out.loc[idx, "market_breadth"]).mean())
            market_penalty_val = min(1.0, max(0.0, abs(breadth_val - 0.50) * 2.0))
        else:
            market_penalty_val = 0.0
        market_penalty = pd.Series(market_penalty_val, index=idx, dtype="float64")

        risk_unit = (
            1.0
            + beta_penalty * beta_rank
            + vol_penalty * vol_rank
            + liq_penalty * liq_penalty_series
            + market_regime_penalty * market_penalty
        )
        risk_unit = risk_unit.clip(lower=risk_floor, upper=risk_cap)

        score_final = out.loc[idx, "score_final"]
        score_risk_adj = score_final / risk_unit.replace(0.0, EPS)
        alpha_to_risk = score_final.abs() / risk_unit.replace(0.0, EPS)
        penalty_rate = 1.0 - (score_risk_adj.abs() / score_final.abs().replace(0.0, EPS))
        penalty_rate = penalty_rate.clip(lower=0.0, upper=1.0)

        quality_flag = ((risk_unit >= risk_cap * 0.98) | (out.loc[idx, "meta_dollar_volume"] <= 0.0)).astype("int64")

        out.loc[idx, "risk_beta_rank"] = beta_rank.astype("float64")
        out.loc[idx, "risk_vol_rank"] = vol_rank.astype("float64")
        out.loc[idx, "risk_liq_penalty"] = liq_penalty_series.astype("float64")
        out.loc[idx, "risk_market_penalty"] = market_penalty.astype("float64")
        out.loc[idx, "risk_unit"] = risk_unit.astype("float64")
        out.loc[idx, "score_risk_adj"] = score_risk_adj.astype("float64")
        out.loc[idx, "score_alpha_to_risk"] = alpha_to_risk.astype("float64")
        out.loc[idx, "risk_penalty_rate"] = penalty_rate.astype("float64")
        out.loc[idx, "risk_quality_flag"] = quality_flag.astype("int64")

    return out