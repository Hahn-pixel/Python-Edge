from __future__ import annotations

import pandas as pd



def apply_position_filters(
    df: pd.DataFrame,
    min_price: float = 5.0,
    min_dollar_volume: float = 1_000_000.0,
) -> pd.DataFrame:
    out = df.copy()
    required = ["meta_price", "meta_dollar_volume", "side"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise RuntimeError(f"apply_position_filters: missing columns: {missing}")

    bad_price = pd.to_numeric(out["meta_price"], errors="coerce") < min_price
    bad_dv = pd.to_numeric(out["meta_dollar_volume"], errors="coerce") < min_dollar_volume
    drop_mask = bad_price | bad_dv
    out.loc[drop_mask, "side"] = 0.0
    out["blocked_low_price"] = bad_price.astype("int64")
    out["blocked_low_dv"] = bad_dv.astype("int64")
    return out



def normalize_gross_exposure(df: pd.DataFrame, gross_target: float = 1.0) -> pd.DataFrame:
    out = df.copy()
    if "date" not in out.columns or "side" not in out.columns:
        raise RuntimeError("normalize_gross_exposure: missing date/side")

    out["weight"] = 0.0
    for dt, idx in out.groupby("date").groups.items():
        gross = float(pd.to_numeric(out.loc[idx, "side"], errors="coerce").abs().sum())
        if gross <= 0.0:
            continue
        out.loc[idx, "weight"] = pd.to_numeric(out.loc[idx, "side"], errors="coerce").fillna(0.0) * (gross_target / gross)
    return out