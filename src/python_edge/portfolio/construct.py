from __future__ import annotations

import pandas as pd


def build_long_short_portfolio(df: pd.DataFrame, top_pct: float = 0.1) -> pd.DataFrame:

    if "date" not in df.columns:
        raise RuntimeError("portfolio: missing date")

    if "score" not in df.columns:
        raise RuntimeError("portfolio: missing score")

    out = df.copy()

    out["rank"] = out.groupby("date")["score"].rank(pct=True)

    out["side"] = 0

    out.loc[out["rank"] >= 1 - top_pct, "side"] = 1
    out.loc[out["rank"] <= top_pct, "side"] = -1

    return out