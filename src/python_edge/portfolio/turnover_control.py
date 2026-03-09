from __future__ import annotations

import pandas as pd


def cap_daily_turnover(
    df: pd.DataFrame,
    weight_col: str = "weight",
    max_daily_turnover: float = 0.60,
) -> pd.DataFrame:
    out = df.copy()

    if "date" not in out.columns:
        raise RuntimeError("cap_daily_turnover: missing date column")

    if weight_col not in out.columns:
        raise RuntimeError(f"cap_daily_turnover: missing {weight_col}")

    out = out.sort_values(["symbol", "date"]).reset_index(drop=True)

    out["prev_weight"] = out.groupby("symbol")[weight_col].shift(1).fillna(0.0)
    out["trade_delta"] = out[weight_col] - out["prev_weight"]
    out["trade_abs_raw"] = out["trade_delta"].abs()

    capped_weights = []

    for dt, sdf in out.groupby("date", sort=False):
        sdf = sdf.copy()

        raw_turnover = float(sdf["trade_abs_raw"].sum())

        if raw_turnover > max_daily_turnover and raw_turnover > 0:
            scale = max_daily_turnover / raw_turnover
            sdf["trade_delta"] = sdf["trade_delta"] * scale
            sdf["cap_hit"] = 1
        else:
            sdf["cap_hit"] = 0

        sdf[weight_col] = sdf["prev_weight"] + sdf["trade_delta"]
        sdf["trade_abs_after"] = sdf["trade_delta"].abs()

        capped_weights.append(sdf)

    out = pd.concat(capped_weights, axis=0, ignore_index=True)

    out["raw_turnover"] = out.groupby("date")["trade_abs_raw"].transform("sum")
    out["capped_turnover"] = out.groupby("date")["trade_abs_after"].transform("sum")

    return out