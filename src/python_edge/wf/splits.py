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
    purge_start: object | None = None
    purge_end: object | None = None
    embargo_start: object | None = None
    embargo_end: object | None = None
    purge_days: int = 0
    embargo_days: int = 0


def _normalize_unique_dates(dates: pd.Series) -> list[object]:
    uniq = pd.Series(dates).dropna().unique().tolist()
    uniq = sorted(uniq)
    if not uniq:
        raise RuntimeError("No valid dates supplied")
    return uniq


def build_walkforward_splits(
    dates: pd.Series,
    train_days: int,
    test_days: int,
    step_days: int | None = None,
    purge_days: int = 0,
    embargo_days: int = 0,
) -> list[WFSplit]:
    if train_days <= 0:
        raise ValueError("train_days must be > 0")
    if test_days <= 0:
        raise ValueError("test_days must be > 0")
    if step_days is None:
        step_days = test_days
    if step_days <= 0:
        raise ValueError("step_days must be > 0")
    if purge_days < 0:
        raise ValueError("purge_days must be >= 0")
    if embargo_days < 0:
        raise ValueError("embargo_days must be >= 0")

    uniq = _normalize_unique_dates(dates)
    min_needed = train_days + purge_days + test_days + embargo_days
    if len(uniq) < min_needed:
        raise RuntimeError("Not enough unique dates for requested walk-forward configuration")

    out: list[WFSplit] = []
    fold_id = 1
    train_start_i = 0

    while True:
        train_end_i = train_start_i + train_days - 1
        purge_start_i = train_end_i + 1 if purge_days > 0 else None
        purge_end_i = train_end_i + purge_days if purge_days > 0 else None
        test_start_i = train_end_i + purge_days + 1
        test_end_i = test_start_i + test_days - 1
        embargo_start_i = test_end_i + 1 if embargo_days > 0 else None
        embargo_end_i = test_end_i + embargo_days if embargo_days > 0 else None
        if embargo_end_i is not None:
            if embargo_end_i >= len(uniq):
                break
        elif test_end_i >= len(uniq):
            break

        out.append(
            WFSplit(
                fold_id=fold_id,
                train_start=uniq[train_start_i],
                train_end=uniq[train_end_i],
                test_start=uniq[test_start_i],
                test_end=uniq[test_end_i],
                purge_start=uniq[purge_start_i] if purge_start_i is not None else None,
                purge_end=uniq[purge_end_i] if purge_end_i is not None else None,
                embargo_start=uniq[embargo_start_i] if embargo_start_i is not None else None,
                embargo_end=uniq[embargo_end_i] if embargo_end_i is not None else None,
                purge_days=purge_days,
                embargo_days=embargo_days,
            )
        )

        fold_id += 1
        next_train_start_i = train_start_i + step_days
        if next_train_start_i + train_days + purge_days + test_days + embargo_days - 1 >= len(uniq):
            break
        train_start_i = next_train_start_i

    if not out:
        raise RuntimeError("No walk-forward splits were generated")
    return out


def print_split_summary(splits: list[WFSplit]) -> None:
    print(f"[WF] folds={len(splits)}")
    for sp in splits:
        base = (
            f"[WF][FOLD {sp.fold_id}] "
            f"train={sp.train_start}..{sp.train_end} "
            f"test={sp.test_start}..{sp.test_end}"
        )
        if sp.purge_days > 0:
            base += f" purge={sp.purge_start}..{sp.purge_end}"
        if sp.embargo_days > 0:
            base += f" embargo={sp.embargo_start}..{sp.embargo_end}"
        print(base)