from __future__ import annotations

import pandas as pd



def add_position_age(
    df: pd.DataFrame,
    symbol_col: str = "symbol",
    date_col: str = "date",
    side_col: str = "side",
) -> pd.DataFrame:
    out = df.copy()
    required = [symbol_col, date_col, side_col]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise RuntimeError(f"add_position_age: missing columns: {missing}")

    out = out.sort_values([symbol_col, date_col]).reset_index(drop=True)

    ages: list[int] = []
    prev_symbol = None
    prev_side = 0.0
    current_age = 0

    for row in out.itertuples(index=False):
        sym = getattr(row, symbol_col)
        side = float(getattr(row, side_col))

        if sym != prev_symbol:
            current_age = 0
            prev_side = 0.0

        if side == 0.0:
            current_age = 0
        elif prev_side == side:
            current_age += 1
        else:
            current_age = 1

        ages.append(current_age)
        prev_symbol = sym
        prev_side = side

    out["position_age"] = ages
    return out



def add_rank_pct(
    df: pd.DataFrame,
    score_col: str = "score",
    date_col: str = "date",
) -> pd.DataFrame:
    out = df.copy()
    required = [score_col, date_col]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise RuntimeError(f"add_rank_pct: missing columns: {missing}")
    out["rank_pct"] = out.groupby(date_col)[score_col].rank(method="average", pct=True)
    return out



def apply_adaptive_exit_rules(
    df: pd.DataFrame,
    side_col: str = "side",
    score_col: str = "score",
    long_max_days_strong: int = 20,
    long_max_days_regular: int = 12,
    short_max_days: int = 3,
    long_strong_rank: float = 0.98,
    long_keep_rank: float = 0.82,
    long_exit_rank: float = 0.70,
    short_keep_rank: float = 0.12,
    short_exit_rank: float = 0.25,
) -> pd.DataFrame:
    out = df.copy()
    required = ["symbol", "date", side_col, score_col]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise RuntimeError(f"apply_adaptive_exit_rules: missing columns: {missing}")

    out = add_rank_pct(out, score_col=score_col, date_col="date")
    out = add_position_age(out, symbol_col="symbol", date_col="date", side_col=side_col)

    side = pd.to_numeric(out[side_col], errors="coerce").fillna(0.0)
    rank_pct = pd.to_numeric(out["rank_pct"], errors="coerce")
    age = pd.to_numeric(out["position_age"], errors="coerce").fillna(0)

    strong_long = (side > 0.0) & (rank_pct >= long_strong_rank)
    long_time_stop = ((side > 0.0) & strong_long & (age > long_max_days_strong)) | ((side > 0.0) & (~strong_long) & (age > long_max_days_regular))
    short_time_stop = (side < 0.0) & (age > short_max_days)

    long_rank_decay = (side > 0.0) & ((rank_pct < long_keep_rank) & (age >= 2))
    long_hard_exit = (side > 0.0) & (rank_pct < long_exit_rank)

    short_rank_decay = (side < 0.0) & ((rank_pct > short_keep_rank) & (age >= 1))
    short_hard_exit = (side < 0.0) & (rank_pct > short_exit_rank)

    exit_mask = long_time_stop | short_time_stop | long_rank_decay | long_hard_exit | short_rank_decay | short_hard_exit

    out["exit_time_stop_long"] = long_time_stop.astype(int)
    out["exit_time_stop_short"] = short_time_stop.astype(int)
    out["exit_rank_decay_long"] = (long_rank_decay | long_hard_exit).astype(int)
    out["exit_rank_decay_short"] = (short_rank_decay | short_hard_exit).astype(int)
    out["exit_any"] = exit_mask.astype(int)

    out.loc[exit_mask, side_col] = 0.0
    return out