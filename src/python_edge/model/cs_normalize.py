from __future__ import annotations

import pandas as pd


def cs_zscore(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Cross‑sectional z-score normalization per date.
    """

    out = df.copy()

    if "date" not in out.columns:
        raise RuntimeError("cs_zscore: missing 'date' column")

    for col in feature_cols:
        if col not in out.columns:
            raise RuntimeError(f"cs_zscore: missing feature {col}")

        grp = out.groupby("date")[col]

        mean = grp.transform("mean")
        std = grp.transform("std").replace(0, pd.NA)

        out[f"z_{col}"] = (out[col] - mean) / std

    return out