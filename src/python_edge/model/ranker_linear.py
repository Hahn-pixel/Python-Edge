from __future__ import annotations

import pandas as pd


def linear_rank_score(df: pd.DataFrame, zcols: list[str], weights: dict[str, float]) -> pd.DataFrame:

    out = df.copy()

    score = 0

    for col in zcols:

        if col not in out.columns:
            raise RuntimeError(f"ranker: missing column {col}")

        w = weights.get(col, 0.0)

        score = score + out[col] * w

    out["score"] = score

    return out