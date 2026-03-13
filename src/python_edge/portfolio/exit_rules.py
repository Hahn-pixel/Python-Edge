from __future__ import annotations

import pandas as pd


def apply_time_exit(
    df: pd.DataFrame,
    age_col: str = "hold_age_days",
    side_col: str = "side",
    max_hold_days: int = 2,
    out_col: str = "exit_flag",
) -> pd.DataFrame:
    out = df.copy()
    age = pd.to_numeric(out.get(age_col, 0), errors="coerce").fillna(0).astype(int)
    side = pd.to_numeric(out.get(side_col, 0), errors="coerce").fillna(0.0)
    out[out_col] = ((side != 0.0) & (age >= int(max_hold_days))).astype(int)
    return out


def apply_score_reversion_exit(
    df: pd.DataFrame,
    score_col: str = "score",
    side_col: str = "side",
    abs_exit_threshold: float = 0.35,
    out_col: str = "exit_score_reversion_flag",
) -> pd.DataFrame:
    out = df.copy()
    score = pd.to_numeric(out.get(score_col, 0.0), errors="coerce").fillna(0.0)
    side = pd.to_numeric(out.get(side_col, 0.0), errors="coerce").fillna(0.0)
    out[out_col] = ((side != 0.0) & (score.abs() <= float(abs_exit_threshold))).astype(int)
    return out


def apply_residual_exit_stack(
    df: pd.DataFrame,
    age_col: str = "hold_age_days",
    side_col: str = "side",
    score_col: str = "score",
    max_hold_days: int = 2,
    abs_exit_threshold: float = 0.35,
    out_col: str = "exit_flag",
) -> pd.DataFrame:
    out = apply_time_exit(df, age_col=age_col, side_col=side_col, max_hold_days=max_hold_days, out_col="_exit_time_flag")
    out = apply_score_reversion_exit(out, score_col=score_col, side_col=side_col, abs_exit_threshold=abs_exit_threshold, out_col="_exit_revert_flag")
    out[out_col] = ((pd.to_numeric(out["_exit_time_flag"], errors="coerce").fillna(0) == 1) | (pd.to_numeric(out["_exit_revert_flag"], errors="coerce").fillna(0) == 1)).astype(int)
    return out