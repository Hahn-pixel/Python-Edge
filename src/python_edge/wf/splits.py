from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class WFSplit:
    fold_id: int
    train_start: object
    train_end: object
    test_start: object
    test_end: object



def build_walkforward_splits(dates: pd.Series, train_days: int, test_days: int, step_days: int | None = None) -> list[WFSplit]:
    if train_days <= 0:
        raise ValueError("train_days must be > 0")
    if test_days <= 0:
        raise ValueError("test_days must be > 0")
    if step_days is None:
        step_days = test_days
    if step_days <= 0:
        raise ValueError("step_days must be > 0")

    uniq = sorted(pd.Series(dates).dropna().unique())
    if len(uniq) < train_days + test_days:
        raise RuntimeError("Not enough unique dates for requested walk-forward configuration")

    out: list[WFSplit] = []
    start_i = 0
    fold_id = 1
    while True:
        train_start_i = start_i
        train_end_i = train_start_i + train_days - 1
        test_start_i = train_end_i + 1
        test_end_i = test_start_i + test_days - 1
        if test_end_i >= len(uniq):
            break
        out.append(
            WFSplit(
                fold_id=fold_id,
                train_start=uniq[train_start_i],
                train_end=uniq[train_end_i],
                test_start=uniq[test_start_i],
                test_end=uniq[test_end_i],
            )
        )
        fold_id += 1
        start_i += step_days

    if not out:
        raise RuntimeError("No walk-forward splits were generated")
    return out



def print_split_summary(splits: list[WFSplit]) -> None:
    print(f"[WF] folds={len(splits)}")
    for sp in splits:
        print(
            f"[WF][FOLD {sp.fold_id}] "
            f"train={sp.train_start}..{sp.train_end} "
            f"test={sp.test_start}..{sp.test_end}"
        )