from __future__ import annotations

import pandas as pd


def build_long_short_portfolio(
    df: pd.DataFrame,
    top_pct: float = 0.10,
    bottom_pct: float | None = None,
    score_col: str = "score",
    date_col: str = "date",
    min_abs_rank_pct: float = 0.80,
    require_fresh_dislocation: bool = False,
    fresh_flag_col: str = "fresh_dislocation_flag",
    abs_rank_col: str = "score_abs_rank_pct",
    max_names_per_side: int | None = None,
) -> pd.DataFrame:
    if date_col not in df.columns:
        raise RuntimeError("portfolio: missing date")
    if score_col not in df.columns:
        raise RuntimeError("portfolio: missing score")
    if bottom_pct is None:
        bottom_pct = top_pct
    if not (0.0 < float(top_pct) < 0.5):
        raise ValueError("top_pct must be in (0, 0.5)")
    if not (0.0 < float(bottom_pct) < 0.5):
        raise ValueError("bottom_pct must be in (0, 0.5)")

    out = df.copy()
    out["rank"] = out.groupby(date_col)[score_col].rank(method="average", pct=True)
    out["side"] = 0

    abs_rank = pd.to_numeric(out.get(abs_rank_col, 0.5), errors="coerce").fillna(0.5)
    fresh_flag = pd.to_numeric(out.get(fresh_flag_col, 0), errors="coerce").fillna(0).astype(int)

    long_mask = out["rank"] >= (1.0 - float(top_pct))
    short_mask = out["rank"] <= float(bottom_pct)
    quality_mask = abs_rank >= float(min_abs_rank_pct)
    if require_fresh_dislocation:
        quality_mask = quality_mask & (fresh_flag == 1)

    out.loc[long_mask & quality_mask, "side"] = 1
    out.loc[short_mask & quality_mask, "side"] = -1

    if max_names_per_side is not None and int(max_names_per_side) > 0:
        keep_idx: list[int] = []
        for _, g in out.groupby(date_col, sort=False):
            longs = g[g["side"] > 0].sort_values(score_col, ascending=False).head(int(max_names_per_side))
            shorts = g[g["side"] < 0].sort_values(score_col, ascending=True).head(int(max_names_per_side))
            keep_idx.extend(longs.index.tolist())
            keep_idx.extend(shorts.index.tolist())
        keep_set = set(keep_idx)
        out.loc[~out.index.isin(keep_set), "side"] = 0

    return out