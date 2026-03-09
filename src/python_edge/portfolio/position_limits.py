from __future__ import annotations

import pandas as pd



def apply_position_filters(
    df: pd.DataFrame,
    side_col: str = "side",
    min_price: float = 5.0,
    min_dollar_volume: float = 1_000_000.0,
) -> pd.DataFrame:
    out = df.copy()
    required = ["meta_price", "meta_dollar_volume", side_col]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise RuntimeError(f"apply_position_filters: missing columns: {missing}")

    bad_price = pd.to_numeric(out["meta_price"], errors="coerce") < min_price
    bad_dv = pd.to_numeric(out["meta_dollar_volume"], errors="coerce") < min_dollar_volume
    drop_mask = bad_price | bad_dv

    out.loc[drop_mask, side_col] = 0.0
    out["blocked_low_price"] = bad_price.astype("int64")
    out["blocked_low_dv"] = bad_dv.astype("int64")
    return out



def normalize_gross_exposure(
    df: pd.DataFrame,
    side_col: str = "side",
    gross_target: float = 1.0,
    out_col: str = "weight",
) -> pd.DataFrame:
    out = df.copy()
    if "date" not in out.columns or side_col not in out.columns:
        raise RuntimeError("normalize_gross_exposure: missing date/side")

    out[out_col] = 0.0
    raw_side = pd.to_numeric(out[side_col], errors="coerce").fillna(0.0)

    for dt, idx in out.groupby("date").groups.items():
        gross = float(raw_side.loc[idx].abs().sum())
        if gross <= 0.0:
            continue
        out.loc[idx, out_col] = raw_side.loc[idx] * (gross_target / gross)

    return out



def cap_final_weight(
    df: pd.DataFrame,
    weight_col: str = "weight",
    cap_abs_weight: float = 0.08,
) -> pd.DataFrame:
    out = df.copy()
    if weight_col not in out.columns:
        raise RuntimeError(f"cap_final_weight: missing weight_col={weight_col}")

    w = pd.to_numeric(out[weight_col], errors="coerce").fillna(0.0)
    capped = w.clip(lower=-cap_abs_weight, upper=cap_abs_weight)
    out[weight_col] = capped
    out["weight_capped_flag"] = (w != capped).astype("int64")
    return out



def renormalize_after_caps(
    df: pd.DataFrame,
    weight_col: str = "weight",
    gross_target: float = 1.0,
) -> pd.DataFrame:
    out = df.copy()
    if "date" not in out.columns or weight_col not in out.columns:
        raise RuntimeError("renormalize_after_caps: missing date/weight")

    out[weight_col] = pd.to_numeric(out[weight_col], errors="coerce").fillna(0.0)

    for dt, idx in out.groupby("date").groups.items():
        gross = float(out.loc[idx, weight_col].abs().sum())
        if gross <= 0.0:
            continue
        if gross <= gross_target:
            continue
        out.loc[idx, weight_col] = out.loc[idx, weight_col] * (gross_target / gross)

    return out