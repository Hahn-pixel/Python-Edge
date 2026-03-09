from __future__ import annotations

import pandas as pd



def attach_prev_weights(df: pd.DataFrame, weight_col: str = "weight") -> pd.DataFrame:
    out = df.copy()
    required = ["symbol", "date", weight_col]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise RuntimeError(f"attach_prev_weights: missing columns: {missing}")

    out = out.sort_values(["symbol", "date"], ascending=[True, True]).reset_index(drop=True)
    out["prev_weight"] = out.groupby("symbol")[weight_col].shift(1).fillna(0.0)
    out["trade_delta"] = (pd.to_numeric(out[weight_col], errors="coerce").fillna(0.0) - pd.to_numeric(out["prev_weight"], errors="coerce").fillna(0.0))
    out["trade_abs"] = out["trade_delta"].abs()
    return out



def cap_daily_turnover(df: pd.DataFrame, weight_col: str = "weight", max_daily_turnover: float = 0.60) -> pd.DataFrame:
    out = attach_prev_weights(df, weight_col=weight_col)
    out["weight_after_turnover"] = pd.to_numeric(out[weight_col], errors="coerce").fillna(0.0)

    for dt, idx in out.groupby("date").groups.items():
        day_trade_abs = float(pd.to_numeric(out.loc[idx, "trade_abs"], errors="coerce").fillna(0.0).sum())
        if day_trade_abs <= max_daily_turnover or day_trade_abs <= 0.0:
            continue

        scale = max_daily_turnover / day_trade_abs
        prev_w = pd.to_numeric(out.loc[idx, "prev_weight"], errors="coerce").fillna(0.0)
        delta = pd.to_numeric(out.loc[idx, "trade_delta"], errors="coerce").fillna(0.0)
        out.loc[idx, "weight_after_turnover"] = prev_w + delta * scale

    out["trade_delta_after"] = pd.to_numeric(out["weight_after_turnover"], errors="coerce").fillna(0.0) - pd.to_numeric(out["prev_weight"], errors="coerce").fillna(0.0)
    out["trade_abs_after"] = out["trade_delta_after"].abs()
    out[weight_col] = pd.to_numeric(out["weight_after_turnover"], errors="coerce").fillna(0.0)
    return out