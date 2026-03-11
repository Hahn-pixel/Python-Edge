from __future__ import annotations

import math

import pandas as pd


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


def _side_strength(side: float, rank_pct: float, score: float) -> float:
    if side > 0.0:
        return float(rank_pct)
    if side < 0.0:
        return float(1.0 - rank_pct)
    return 0.0


def apply_adaptive_exit_rules(
    df: pd.DataFrame,
    side_col: str = "side",
    score_col: str = "score",
    long_max_days_strong: int = 20,
    long_max_days_regular: int = 12,
    short_max_days: int = 5,
    long_strong_rank: float = 0.98,
    long_keep_rank: float = 0.82,
    long_exit_rank: float = 0.70,
    short_keep_rank: float = 0.12,
    short_exit_rank: float = 0.25,
    peak_trail_drop_long: float = 0.10,
    peak_trail_drop_short: float = 0.10,
    peak_trail_min_age_long: int = 2,
    peak_trail_min_age_short: int = 1,
    score_peak_reentry_buffer: float = 0.01,
) -> pd.DataFrame:
    out = df.copy()
    required = ["symbol", "date", side_col, score_col]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise RuntimeError(f"apply_adaptive_exit_rules: missing columns: {missing}")

    out = add_rank_pct(out, score_col=score_col, date_col="date")
    out = out.sort_values(["symbol", "date"]).reset_index(drop=True)

    final_side: list[float] = []
    position_age: list[int] = []
    peak_strength_seen: list[float] = []
    peak_score_seen: list[float] = []
    trail_distance_seen: list[float] = []
    exit_time_stop_long: list[int] = []
    exit_time_stop_short: list[int] = []
    exit_rank_decay_long: list[int] = []
    exit_rank_decay_short: list[int] = []
    exit_score_peak_long: list[int] = []
    exit_score_peak_short: list[int] = []
    exit_signal_flip: list[int] = []
    exit_signal_flat: list[int] = []
    exit_any: list[int] = []

    for _, sdf in out.groupby("symbol", sort=False):
        live_side = 0.0
        live_age = 0
        live_peak_strength = 0.0
        live_peak_score = 0.0

        for row in sdf.itertuples(index=False):
            signal_side = float(getattr(row, side_col))
            rank_pct = float(getattr(row, "rank_pct"))
            score = float(getattr(row, score_col))

            flag_time_long = 0
            flag_time_short = 0
            flag_rank_long = 0
            flag_rank_short = 0
            flag_peak_long = 0
            flag_peak_short = 0
            flag_flip = 0
            flag_flat = 0

            if signal_side == 0.0:
                if live_side != 0.0:
                    flag_flat = 1
                live_side = 0.0
                live_age = 0
                live_peak_strength = 0.0
                live_peak_score = 0.0
                final_side.append(0.0)
                position_age.append(0)
                peak_strength_seen.append(0.0)
                peak_score_seen.append(0.0)
                trail_distance_seen.append(0.0)
                exit_time_stop_long.append(flag_time_long)
                exit_time_stop_short.append(flag_time_short)
                exit_rank_decay_long.append(flag_rank_long)
                exit_rank_decay_short.append(flag_rank_short)
                exit_score_peak_long.append(flag_peak_long)
                exit_score_peak_short.append(flag_peak_short)
                exit_signal_flip.append(flag_flip)
                exit_signal_flat.append(flag_flat)
                exit_any.append(int(flag_flat == 1))
                continue

            if live_side != 0.0 and math.copysign(1.0, live_side) != math.copysign(1.0, signal_side):
                flag_flip = 1
                live_side = 0.0
                live_age = 0
                live_peak_strength = 0.0
                live_peak_score = 0.0

            entering_new = live_side == 0.0
            trial_age = 1 if entering_new else live_age + 1
            curr_strength = _side_strength(signal_side, rank_pct, score)
            trial_peak_strength = curr_strength if entering_new else max(live_peak_strength, curr_strength)
            trial_peak_score = score if entering_new else (max(live_peak_score, score) if signal_side > 0.0 else min(live_peak_score, score))
            trail_distance = max(0.0, trial_peak_strength - curr_strength)

            long_time_stop = False
            short_time_stop = False
            long_rank_decay = False
            short_rank_decay = False
            long_peak_exit = False
            short_peak_exit = False

            if signal_side > 0.0:
                is_strong_long = rank_pct >= long_strong_rank
                max_days = long_max_days_strong if is_strong_long else long_max_days_regular
                long_time_stop = trial_age > max_days
                long_rank_decay = rank_pct < long_exit_rank or (rank_pct < long_keep_rank and trial_age >= 2)
                long_peak_exit = trial_age >= peak_trail_min_age_long and trail_distance >= peak_trail_drop_long and rank_pct < max(long_keep_rank, trial_peak_strength - score_peak_reentry_buffer)
            else:
                short_time_stop = trial_age > short_max_days
                short_rank_decay = rank_pct > short_exit_rank or (rank_pct > short_keep_rank and trial_age >= 1)
                short_peak_exit = trial_age >= peak_trail_min_age_short and trail_distance >= peak_trail_drop_short and rank_pct > min(short_keep_rank, 1.0 - trial_peak_strength + score_peak_reentry_buffer)

            should_exit = long_time_stop or short_time_stop or long_rank_decay or short_rank_decay or long_peak_exit or short_peak_exit

            if should_exit:
                if long_time_stop:
                    flag_time_long = 1
                if short_time_stop:
                    flag_time_short = 1
                if long_rank_decay:
                    flag_rank_long = 1
                if short_rank_decay:
                    flag_rank_short = 1
                if long_peak_exit:
                    flag_peak_long = 1
                if short_peak_exit:
                    flag_peak_short = 1
                live_side = 0.0
                live_age = 0
                live_peak_strength = 0.0
                live_peak_score = 0.0
                final_side.append(0.0)
                position_age.append(0)
                peak_strength_seen.append(trial_peak_strength)
                peak_score_seen.append(trial_peak_score)
                trail_distance_seen.append(trail_distance)
                exit_time_stop_long.append(flag_time_long)
                exit_time_stop_short.append(flag_time_short)
                exit_rank_decay_long.append(flag_rank_long)
                exit_rank_decay_short.append(flag_rank_short)
                exit_score_peak_long.append(flag_peak_long)
                exit_score_peak_short.append(flag_peak_short)
                exit_signal_flip.append(flag_flip)
                exit_signal_flat.append(flag_flat)
                exit_any.append(1)
                continue

            live_side = signal_side
            live_age = trial_age
            live_peak_strength = trial_peak_strength
            live_peak_score = trial_peak_score
            final_side.append(live_side)
            position_age.append(live_age)
            peak_strength_seen.append(live_peak_strength)
            peak_score_seen.append(live_peak_score)
            trail_distance_seen.append(trail_distance)
            exit_time_stop_long.append(flag_time_long)
            exit_time_stop_short.append(flag_time_short)
            exit_rank_decay_long.append(flag_rank_long)
            exit_rank_decay_short.append(flag_rank_short)
            exit_score_peak_long.append(flag_peak_long)
            exit_score_peak_short.append(flag_peak_short)
            exit_signal_flip.append(flag_flip)
            exit_signal_flat.append(flag_flat)
            exit_any.append(int(flag_flip == 1))

    out[side_col] = final_side
    out["position_age"] = position_age
    out["peak_strength_seen"] = peak_strength_seen
    out["peak_score_seen"] = peak_score_seen
    out["trail_distance_seen"] = trail_distance_seen
    out["exit_time_stop_long"] = exit_time_stop_long
    out["exit_time_stop_short"] = exit_time_stop_short
    out["exit_rank_decay_long"] = exit_rank_decay_long
    out["exit_rank_decay_short"] = exit_rank_decay_short
    out["exit_score_peak_long"] = exit_score_peak_long
    out["exit_score_peak_short"] = exit_score_peak_short
    out["exit_signal_flip"] = exit_signal_flip
    out["exit_signal_flat"] = exit_signal_flat
    out["exit_any"] = exit_any
    return out