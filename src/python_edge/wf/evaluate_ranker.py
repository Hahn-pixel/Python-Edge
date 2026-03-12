from __future__ import annotations

import pandas as pd


def evaluate_long_short(df: pd.DataFrame, target_col: str = "target_fwd_ret_1d") -> pd.DataFrame:
    out = df.copy()
    required = ["date", "weight", target_col]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise RuntimeError(f"evaluate_long_short: missing columns: {missing}")

    out = out.loc[out[target_col].notna()].copy()
    weight = pd.to_numeric(out["weight"], errors="coerce").fillna(0.0)
    fwd = pd.to_numeric(out[target_col], errors="coerce").fillna(0.0)
    out["gross_ret_contrib"] = weight * fwd

    if "cost_trading" not in out.columns:
        out["cost_trading"] = 0.0
    if "cost_borrow" not in out.columns:
        out["cost_borrow"] = 0.0
    if "cost_total" not in out.columns:
        out["cost_total"] = pd.to_numeric(out["cost_trading"], errors="coerce").fillna(0.0) + pd.to_numeric(out["cost_borrow"], errors="coerce").fillna(0.0)

    out["net_ret_contrib"] = out["gross_ret_contrib"] - pd.to_numeric(out["cost_total"], errors="coerce").fillna(0.0)
    out["long_gross_contrib"] = 0.0
    out.loc[weight > 0.0, "long_gross_contrib"] = out.loc[weight > 0.0, "gross_ret_contrib"]
    out["short_gross_contrib"] = 0.0
    out.loc[weight < 0.0, "short_gross_contrib"] = out.loc[weight < 0.0, "gross_ret_contrib"]
    out["long_cost_contrib"] = 0.0
    out.loc[weight > 0.0, "long_cost_contrib"] = pd.to_numeric(out.loc[weight > 0.0, "cost_total"], errors="coerce").fillna(0.0)
    out["short_cost_contrib"] = 0.0
    out.loc[weight < 0.0, "short_cost_contrib"] = pd.to_numeric(out.loc[weight < 0.0, "cost_total"], errors="coerce").fillna(0.0)

    agg_map = {
        "portfolio_ret": ("net_ret_contrib", "sum"),
        "gross_ret": ("gross_ret_contrib", "sum"),
        "trading_costs": ("cost_trading", "sum"),
        "borrow_costs": ("cost_borrow", "sum"),
        "costs": ("cost_total", "sum"),
        "long_gross_ret": ("long_gross_contrib", "sum"),
        "short_gross_ret": ("short_gross_contrib", "sum"),
        "long_costs": ("long_cost_contrib", "sum"),
        "short_costs": ("short_cost_contrib", "sum"),
        "gross_long_exposure": ("weight", lambda s: float(pd.to_numeric(s, errors="coerce").clip(lower=0.0).sum())),
        "gross_short_exposure": ("weight", lambda s: float((-pd.to_numeric(s, errors="coerce").clip(upper=0.0)).sum())),
        "positions": ("weight", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0.0) != 0.0).sum())),
    }

    optional_mean_cols = [
        "raw_turnover",
        "capped_turnover",
        "cap_hit",
        "cash_weight",
        "deployed_gross",
        "turnover_budget_left",
        "execution_participation",
        "execution_participation_flag",
        "long_budget",
        "short_budget",
        "position_age",
        "exit_any",
    ]
    for col in optional_mean_cols:
        if col in out.columns:
            agg_map[col] = (col, "mean")

    daily = out.groupby("date", as_index=False).agg(**agg_map)
    if "cap_hit" in daily.columns:
        daily = daily.rename(columns={"cap_hit": "cap_hit_rate"})
    if "execution_participation_flag" in daily.columns:
        daily = daily.rename(columns={"execution_participation_flag": "participation_limit_hit_rate"})
    if "position_age" in daily.columns:
        daily = daily.rename(columns={"position_age": "avg_hold_days"})
    if "exit_any" in daily.columns:
        daily = daily.rename(columns={"exit_any": "exit_rate"})
    return daily


def summarize_daily_returns(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {}
    s = pd.to_numeric(df["portfolio_ret"], errors="coerce").dropna()
    if s.empty:
        return {}
    out: dict[str, float] = {}
    out["days"] = float(len(s))
    out["avg_daily_ret"] = float(s.mean())
    out["std_daily_ret"] = float(s.std())
    out["win_rate_days"] = float((s > 0).mean())
    out["cum_ret"] = float((1.0 + s).prod() - 1.0)

    for col, key in [
        ("raw_turnover", "avg_raw_turnover"),
        ("capped_turnover", "avg_turnover"),
        ("cap_hit_rate", "cap_hit_rate"),
        ("gross_ret", "avg_gross_ret"),
        ("trading_costs", "avg_trading_costs"),
        ("borrow_costs", "avg_borrow_costs"),
        ("costs", "avg_costs"),
        ("long_gross_ret", "avg_long_gross_ret"),
        ("short_gross_ret", "avg_short_gross_ret"),
        ("long_costs", "avg_long_costs"),
        ("short_costs", "avg_short_costs"),
        ("gross_long_exposure", "avg_gross_long_exposure"),
        ("gross_short_exposure", "avg_gross_short_exposure"),
        ("cash_weight", "avg_cash_weight"),
        ("deployed_gross", "avg_deployed_gross"),
        ("turnover_budget_left", "avg_turnover_budget_left"),
        ("execution_participation", "avg_execution_participation"),
        ("participation_limit_hit_rate", "participation_limit_hit_rate"),
        ("long_budget", "avg_long_budget"),
        ("short_budget", "avg_short_budget"),
        ("avg_hold_days", "avg_hold_days"),
        ("exit_rate", "exit_rate"),
    ]:
        if col in df.columns:
            out[key] = float(pd.to_numeric(df[col], errors="coerce").fillna(0.0).mean())
    return out


def print_summary(tag: str, summary: dict[str, float]) -> None:
    if not summary:
        print(f"{tag} EMPTY")
        return
    ordered_keys = [
        "days",
        "avg_daily_ret",
        "std_daily_ret",
        "win_rate_days",
        "cum_ret",
        "avg_raw_turnover",
        "avg_turnover",
        "cap_hit_rate",
        "avg_turnover_budget_left",
        "avg_gross_ret",
        "avg_trading_costs",
        "avg_borrow_costs",
        "avg_costs",
        "avg_long_gross_ret",
        "avg_short_gross_ret",
        "avg_long_costs",
        "avg_short_costs",
        "avg_gross_long_exposure",
        "avg_gross_short_exposure",
        "avg_cash_weight",
        "avg_deployed_gross",
        "avg_execution_participation",
        "participation_limit_hit_rate",
        "avg_long_budget",
        "avg_short_budget",
        "avg_hold_days",
        "exit_rate",
    ]
    parts: list[str] = [tag]
    for key in ordered_keys:
        if key not in summary:
            continue
        val = summary[key]
        if key == "days":
            parts.append(f"{key}={int(val)}")
        else:
            parts.append(f"{key}={val:.6f}")
    print(" ".join(parts))