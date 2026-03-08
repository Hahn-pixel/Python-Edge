from __future__ import annotations

import numpy as np
import pandas as pd



def add_beta_proxy(df: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    out = df.copy()
    required = ["symbol", "date", "close", "market_ret_mean"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise RuntimeError(f"add_beta_proxy: missing columns: {missing}")

    out = out.sort_values(["symbol", "date"], ascending=[True, True]).reset_index(drop=True)
    out["ret_1d_for_beta"] = out.groupby("symbol")["close"].pct_change()

    frames: list[pd.DataFrame] = []
    for sym, g in out.groupby("symbol", sort=False):
        gg = g.copy()
        x = pd.to_numeric(gg["market_ret_mean"], errors="coerce")
        y = pd.to_numeric(gg["ret_1d_for_beta"], errors="coerce")
        cov = y.rolling(lookback, min_periods=lookback).cov(x)
        var = x.rolling(lookback, min_periods=lookback).var()
        gg["beta_proxy_60d"] = cov / var.replace(0.0, pd.NA)
        frames.append(gg)

    if not frames:
        raise RuntimeError("add_beta_proxy: no symbol frames produced")

    out = pd.concat(frames, axis=0, ignore_index=True)
    if "symbol" not in out.columns:
        raise RuntimeError("add_beta_proxy: symbol disappeared after beta computation")
    return out



def neutralize_score_cross_section(
    df: pd.DataFrame,
    score_col: str = "score",
    exposure_cols: list[str] | None = None,
    out_col: str = "score_neutral",
) -> pd.DataFrame:
    out = df.copy()
    if "date" not in out.columns:
        raise RuntimeError("neutralize_score_cross_section: missing date")
    if score_col not in out.columns:
        raise RuntimeError(f"neutralize_score_cross_section: missing score_col={score_col}")

    if exposure_cols is None:
        exposure_cols = ["liq_rank", "beta_proxy_60d"]

    missing = [c for c in exposure_cols if c not in out.columns]
    if missing:
        raise RuntimeError(f"neutralize_score_cross_section: missing exposure columns: {missing}")

    out[out_col] = pd.NA

    for dt, idx in out.groupby("date").groups.items():
        block = out.loc[idx, [score_col] + exposure_cols].copy()
        block = block.dropna()
        if len(block) < max(20, len(exposure_cols) + 5):
            out.loc[idx, out_col] = pd.to_numeric(out.loc[idx, score_col], errors="coerce")
            continue

        y = pd.to_numeric(block[score_col], errors="coerce")
        x = block[exposure_cols].copy().apply(pd.to_numeric, errors="coerce")
        x = x.fillna(x.mean())
        x["intercept"] = 1.0

        try:
            beta = x.to_numpy(dtype="float64")
            yy = y.to_numpy(dtype="float64")
            coef = np.linalg.lstsq(beta, yy, rcond=None)[0]
            fitted = beta @ coef
            resid = yy - fitted
            out.loc[block.index, out_col] = resid

            missing_idx = out.loc[idx].index.difference(block.index)
            if len(missing_idx) > 0:
                out.loc[missing_idx, out_col] = pd.to_numeric(out.loc[missing_idx, score_col], errors="coerce")
        except Exception:
            out.loc[idx, out_col] = pd.to_numeric(out.loc[idx, score_col], errors="coerce")

    out[out_col] = pd.to_numeric(out[out_col], errors="coerce")
    return out