from __future__ import annotations

import pandas as pd



def attach_prev_side(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "symbol" not in out.columns or "date" not in out.columns or "side" not in out.columns:
        raise RuntimeError("attach_prev_side: missing symbol/date/side")
    out = out.sort_values(["symbol", "date"], ascending=[True, True]).reset_index(drop=True)
    out["prev_side"] = out.groupby("symbol")["side"].shift(1).fillna(0.0)
    return out



def attach_turnover_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = attach_prev_side(df)
    out["turnover_unit"] = (pd.to_numeric(out["side"], errors="coerce").fillna(0.0) - pd.to_numeric(out["prev_side"], errors="coerce").fillna(0.0)).abs()
    return out



def dampen_turnover(df: pd.DataFrame, max_turnover_unit: float = 1.0) -> pd.DataFrame:
    out = attach_turnover_metrics(df)
    required = ["date", "turnover_unit", "side"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise RuntimeError(f"dampen_turnover: missing columns: {missing}")

    out["side_after_turnover"] = pd.to_numeric(out["side"], errors="coerce").fillna(0.0)

    for dt, idx in out.groupby("date").groups.items():
        day_turnover = float(out.loc[idx, "turnover_unit"].sum())
        if day_turnover <= max_turnover_unit or day_turnover <= 0.0:
            continue
        scale = max_turnover_unit / day_turnover
        prev_side = pd.to_numeric(out.loc[idx, "prev_side"], errors="coerce").fillna(0.0)
        delta = pd.to_numeric(out.loc[idx, "side"], errors="coerce").fillna(0.0) - prev_side
        out.loc[idx, "side_after_turnover"] = prev_side + delta * scale

    out["turnover_unit_after"] = (pd.to_numeric(out["side_after_turnover"], errors="coerce").fillna(0.0) - pd.to_numeric(out["prev_side"], errors="coerce").fillna(0.0)).abs()
    out["side"] = out["side_after_turnover"]
    return out