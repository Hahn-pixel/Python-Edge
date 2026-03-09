from __future__ import annotations

import pandas as pd


# -------------------------------------------------
# 1. POSITION AGE TRACKING
# -------------------------------------------------


def add_position_age(
    df: pd.DataFrame,
    symbol_col: str = "symbol",
    date_col: str = "date",
    side_col: str = "side",
) -> pd.DataFrame:

    out = df.copy()
    out = out.sort_values([symbol_col, date_col]).reset_index(drop=True)

    age = []

    last_symbol = None
    last_side = 0
    current_age = 0

    for row in out.itertuples(index=False):
        sym = getattr(row, symbol_col)
        side = getattr(row, side_col)

        if sym != last_symbol:
            current_age = 0

        if side == 0:
            current_age = 0
        else:
            if last_side == side:
                current_age += 1
            else:
                current_age = 1

        age.append(current_age)

        last_symbol = sym
        last_side = side

    out["position_age"] = age
    return out


# -------------------------------------------------
# 2. TIME STOP
# -------------------------------------------------


def apply_time_stop(
    df: pd.DataFrame,
    long_max_days: int = 15,
    short_max_days: int = 3,
    side_col: str = "side",
) -> pd.DataFrame:

    out = df.copy()

    if "position_age" not in out.columns:
        raise RuntimeError("apply_time_stop: position_age missing")

    long_mask = (out[side_col] > 0) & (out["position_age"] > long_max_days)
    short_mask = (out[side_col] < 0) & (out["position_age"] > short_max_days)

    exit_mask = long_mask | short_mask

    out.loc[exit_mask, side_col] = 0.0

    out["exit_time_stop"] = exit_mask.astype(int)

    return out


# -------------------------------------------------
# 3. RANK DECAY EXIT
# -------------------------------------------------


def apply_rank_decay_exit(
    df: pd.DataFrame,
    rank_col: str = "rank_pct",
    side_col: str = "side",
    long_exit_threshold: float = 0.30,
    short_exit_threshold: float = 0.70,
) -> pd.DataFrame:

    out = df.copy()

    if rank_col not in out.columns:
        raise RuntimeError("apply_rank_decay_exit: rank_pct missing")

    long_decay = (out[side_col] > 0) & (out[rank_col] < (1.0 - long_exit_threshold))
    short_decay = (out[side_col] < 0) & (out[rank_col] > short_exit_threshold)

    exit_mask = long_decay | short_decay

    out.loc[exit_mask, side_col] = 0.0

    out["exit_rank_decay"] = exit_mask.astype(int)

    return out


# -------------------------------------------------
# 4. FULL EXIT PIPELINE
# -------------------------------------------------


def apply_exit_rules(
    df: pd.DataFrame,
    long_max_days: int = 15,
    short_max_days: int = 3,
) -> pd.DataFrame:

    out = df.copy()

    out = add_position_age(out)

    out = apply_time_stop(
        out,
        long_max_days=long_max_days,
        short_max_days=short_max_days,
    )

    out = apply_rank_decay_exit(out)

    return out