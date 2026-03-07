from __future__ import annotations

import pandas as pd


def print_feature_matrix_summary(df: pd.DataFrame) -> None:
    print(f"[FEATURE] rows={len(df)}")
    print(f"[FEATURE] cols={len(df.columns)}")
    if "symbol" in df.columns:
        print(f"[FEATURE] symbols={df['symbol'].nunique()}")
    if "date" in df.columns and len(df) > 0:
        print(f"[FEATURE] dates={df['date'].nunique()}")
        print(f"[FEATURE] min_date={df['date'].min()}")
        print(f"[FEATURE] max_date={df['date'].max()}")


def print_feature_coverage(df: pd.DataFrame, feature_cols: list[str]) -> None:
    total = max(len(df), 1)
    for col in feature_cols:
        if col not in df.columns:
            print(f"[FEATURE][WARN] missing_feature_col={col}")
            continue
        non_na = int(df[col].notna().sum())
        cov = non_na / total
        print(f"[FEATURE][COVERAGE] {col} non_na={non_na} coverage={cov:.4f}")


def print_feature_warnings(df: pd.DataFrame, feature_cols: list[str], near_constant_unique_max: int = 3, nan_warn_threshold: float = 0.25) -> None:
    total = max(len(df), 1)
    for col in feature_cols:
        if col not in df.columns:
            continue
        s = df[col]
        nan_ratio = float(s.isna().sum()) / total
        nunique = int(s.nunique(dropna=True))
        if nan_ratio >= nan_warn_threshold:
            print(f"[FEATURE][WARN] high_nan_ratio col={col} nan_ratio={nan_ratio:.4f}")
        if nunique <= near_constant_unique_max:
            print(f"[FEATURE][WARN] near_constant col={col} nunique={nunique}")