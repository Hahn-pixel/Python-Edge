from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class LinearRankerFit:
    zcols: list[str]
    weights: dict[str, float]
    target_col: str



def fit_corr_weights(train_df: pd.DataFrame, zcols: list[str], target_col: str) -> LinearRankerFit:
    if train_df.empty:
        raise RuntimeError("fit_corr_weights: empty train_df")
    if target_col not in train_df.columns:
        raise RuntimeError(f"fit_corr_weights: missing target_col={target_col}")

    weights: dict[str, float] = {}
    for col in zcols:
        if col not in train_df.columns:
            raise RuntimeError(f"fit_corr_weights: missing feature column {col}")
        pair = train_df[[col, target_col]].dropna()
        if len(pair) < 20:
            weights[col] = 0.0
            continue
        x = pair[col]
        y = pair[target_col]
        if float(x.std()) == 0.0 or float(y.std()) == 0.0:
            weights[col] = 0.0
            continue
        weights[col] = float(x.corr(y))

    return LinearRankerFit(zcols=list(zcols), weights=weights, target_col=target_col)



def apply_linear_score(df: pd.DataFrame, fit: LinearRankerFit, out_col: str = "score") -> pd.DataFrame:
    out = df.copy()
    score = pd.Series(0.0, index=out.index, dtype="float64")

    for col in fit.zcols:
        if col not in out.columns:
            raise RuntimeError(f"apply_linear_score: missing column {col}")
        w = float(fit.weights.get(col, 0.0))
        score = score + pd.to_numeric(out[col], errors="coerce").fillna(0.0) * w

    out[out_col] = score
    return out



def print_fit_summary(fit: LinearRankerFit) -> None:
    print(f"[RANKER] target_col={fit.target_col}")
    print(f"[RANKER] zcols={fit.zcols}")
    for col in fit.zcols:
        print(f"[RANKER][WEIGHT] {col}={fit.weights.get(col, 0.0):.6f}")