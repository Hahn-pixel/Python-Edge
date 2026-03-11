from __future__ import annotations

import math

import pandas as pd



def _infer_strength(row: pd.Series, weight_col: str) -> float:
    target_weight = float(pd.to_numeric(row.get(weight_col, 0.0), errors="coerce"))
    rank_pct = pd.to_numeric(row.get("rank_pct", None), errors="coerce")
    score = pd.to_numeric(row.get("score", None), errors="coerce")
    if pd.notna(rank_pct):
        if target_weight > 0.0:
            return float(rank_pct)
        if target_weight < 0.0:
            return float(1.0 - rank_pct)
    if pd.notna(score):
        return float(abs(score))
    return float(abs(target_weight))



def _trade_priority_components(prev_weight: float, target_weight: float) -> list[tuple[str, float]]:
    pieces: list[tuple[str, float]] = []
    if prev_weight == 0.0 and target_weight == 0.0:
        return pieces
    if prev_weight != 0.0 and target_weight == 0.0:
        pieces.append(("mandatory_exit", -prev_weight))
        return pieces
    if prev_weight == 0.0 and target_weight != 0.0:
        pieces.append(("new_entry", target_weight))
        return pieces
    if math.copysign(1.0, prev_weight) != math.copysign(1.0, target_weight):
        pieces.append(("mandatory_exit", -prev_weight))
        pieces.append(("new_entry", target_weight))
        return pieces
    if abs(target_weight) < abs(prev_weight):
        pieces.append(("risk_trim", target_weight - prev_weight))
        return pieces
    if abs(target_weight) > abs(prev_weight):
        if prev_weight != 0.0:
            pieces.append(("keep_add", target_weight - prev_weight))
        else:
            pieces.append(("new_entry", target_weight))
        return pieces
    return pieces



def cap_daily_turnover(
    df: pd.DataFrame,
    weight_col: str = "weight",
    max_daily_turnover: float = 0.60,
) -> pd.DataFrame:
    out = df.copy()
    required = ["date", "symbol", weight_col]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise RuntimeError(f"cap_daily_turnover: missing columns: {missing}")
    if max_daily_turnover <= 0.0:
        raise ValueError("max_daily_turnover must be > 0")

    out = out.sort_values(["date", "symbol"]).reset_index(drop=True)
    actual_rows: list[pd.DataFrame] = []
    prev_actual_by_symbol: dict[str, float] = {}

    for _, sdf in out.groupby("date", sort=False):
        day = sdf.copy()
        prev_weight_list: list[float] = []
        target_weight_list: list[float] = []
        target_trade_abs_list: list[float] = []
        actual_trade_list: list[float] = []
        actual_weight_list: list[float] = []
        raw_priority_list: list[str] = []
        strength_list: list[float] = []
        unfilled_abs_list: list[float] = []
        budget_left = float(max_daily_turnover)

        plan_rows: list[dict[str, object]] = []
        for idx, row in day.iterrows():
            symbol = str(row["symbol"])
            prev_weight = float(prev_actual_by_symbol.get(symbol, 0.0))
            target_weight = float(pd.to_numeric(row[weight_col], errors="coerce"))
            strength = _infer_strength(row, weight_col)
            parts = _trade_priority_components(prev_weight=prev_weight, target_weight=target_weight)
            for bucket, delta in parts:
                plan_rows.append(
                    {
                        "idx": idx,
                        "symbol": symbol,
                        "bucket": bucket,
                        "delta": float(delta),
                        "trade_abs": abs(float(delta)),
                        "strength": strength,
                    }
                )
            prev_weight_list.append(prev_weight)
            target_weight_list.append(target_weight)
            target_trade_abs_list.append(abs(target_weight - prev_weight))
            actual_trade_list.append(0.0)
            actual_weight_list.append(prev_weight)
            raw_priority_list.append("hold")
            strength_list.append(strength)
            unfilled_abs_list.append(abs(target_weight - prev_weight))

        bucket_order = {
            "mandatory_exit": 0,
            "risk_trim": 1,
            "keep_add": 2,
            "new_entry": 3,
        }
        plan_rows = sorted(
            plan_rows,
            key=lambda x: (
                bucket_order.get(str(x["bucket"]), 99),
                -float(x["strength"]),
                -float(x["trade_abs"]),
                str(x["symbol"]),
            ),
        )

        executed_by_idx: dict[int, float] = {}
        top_bucket_by_idx: dict[int, str] = {}
        for item in plan_rows:
            idx = int(item["idx"])
            delta = float(item["delta"])
            need = abs(delta)
            if need <= 0.0:
                continue
            if budget_left <= 0.0:
                continue
            exec_abs = min(need, budget_left)
            exec_delta = math.copysign(exec_abs, delta)
            executed_by_idx[idx] = float(executed_by_idx.get(idx, 0.0)) + exec_delta
            if idx not in top_bucket_by_idx:
                top_bucket_by_idx[idx] = str(item["bucket"])
            budget_left -= exec_abs

        for pos, idx in enumerate(day.index.tolist()):
            prev_weight = prev_weight_list[pos]
            target_weight = target_weight_list[pos]
            executed_delta = float(executed_by_idx.get(idx, 0.0))
            actual_weight = prev_weight + executed_delta
            remaining = abs(target_weight - actual_weight)
            actual_trade_abs = abs(executed_delta)
            actual_trade_list[pos] = executed_delta
            actual_weight_list[pos] = actual_weight
            raw_priority_list[pos] = str(top_bucket_by_idx.get(idx, "hold"))
            unfilled_abs_list[pos] = remaining
            prev_actual_by_symbol[str(day.loc[idx, "symbol"])] = actual_weight

        day["prev_weight"] = prev_weight_list
        day["target_weight"] = target_weight_list
        day["trade_delta_target"] = day["target_weight"] - day["prev_weight"]
        day["trade_abs_raw"] = target_trade_abs_list
        day["trade_delta"] = actual_trade_list
        day[weight_col] = actual_weight_list
        day["trade_abs_after"] = [abs(x) for x in actual_trade_list]
        day["trade_priority"] = raw_priority_list
        day["trade_strength"] = strength_list
        day["trade_abs_unfilled"] = unfilled_abs_list
        day["raw_turnover"] = float(sum(target_trade_abs_list))
        day["capped_turnover"] = float(sum(abs(x) for x in actual_trade_list))
        day["cap_hit"] = int(day["raw_turnover"].iloc[0] > max_daily_turnover)
        day["turnover_budget_left"] = max(0.0, budget_left)
        actual_rows.append(day)

    out = pd.concat(actual_rows, axis=0, ignore_index=True)
    return out