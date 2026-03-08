from __future__ import annotations

import pandas as pd



def apply_holding_inertia(
    df: pd.DataFrame,
    enter_pct: float = 0.10,
    exit_pct: float = 0.20,
) -> pd.DataFrame:
    """
    Hysteresis portfolio rule.

    New positions are opened only in the strongest names (enter threshold),
    but existing positions are allowed to persist until they deteriorate below
    a weaker exit threshold. This reduces churn materially.
    """

    if enter_pct <= 0.0 or enter_pct >= 0.5:
        raise ValueError("enter_pct must be in (0, 0.5)")
    if exit_pct < enter_pct or exit_pct >= 0.5:
        raise ValueError("exit_pct must be in [enter_pct, 0.5)")

    required = ["date", "symbol", "score"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"apply_holding_inertia: missing columns: {missing}")

    out = df.copy()
    out = out.sort_values(["date", "symbol"], ascending=[True, True]).reset_index(drop=True)
    out["rank_pct"] = out.groupby("date")["score"].rank(method="average", pct=True)
    out["side"] = 0.0

    prev_side_by_symbol: dict[str, float] = {}

    for dt in sorted(out["date"].dropna().unique()):
        day_mask = out["date"] == dt
        day_idx = out.index[day_mask]
        day = out.loc[day_idx, ["symbol", "rank_pct"]].copy()

        next_side = pd.Series(0.0, index=day.index, dtype="float64")

        long_enter = day["rank_pct"] >= (1.0 - enter_pct)
        short_enter = day["rank_pct"] <= enter_pct
        long_keep = day["rank_pct"] >= (1.0 - exit_pct)
        short_keep = day["rank_pct"] <= exit_pct

        for idx, row in day.iterrows():
            sym = str(row["symbol"])
            prev_side = float(prev_side_by_symbol.get(sym, 0.0))
            rp = float(row["rank_pct"])

            if rp >= (1.0 - enter_pct):
                curr_side = 1.0
            elif rp <= enter_pct:
                curr_side = -1.0
            elif prev_side > 0.0 and rp >= (1.0 - exit_pct):
                curr_side = 1.0
            elif prev_side < 0.0 and rp <= exit_pct:
                curr_side = -1.0
            else:
                curr_side = 0.0

            next_side.loc[idx] = curr_side
            prev_side_by_symbol[sym] = curr_side

        out.loc[day_idx, "side"] = next_side.values

    return out