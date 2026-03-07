from __future__ import annotations

import pandas as pd


def walkforward_splits(dates: pd.Series, train_days: int, test_days: int):

    uniq = sorted(dates.unique())

    i = 0

    while True:

        train_start = i
        train_end = train_start + train_days

        test_start = train_end
        test_end = test_start + test_days

        if test_end >= len(uniq):
            break

        yield (
            uniq[train_start],
            uniq[train_end],
            uniq[test_start],
            uniq[test_end],
        )

        i += test_days