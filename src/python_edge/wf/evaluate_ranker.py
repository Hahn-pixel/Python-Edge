from __future__ import annotations

import pandas as pd



def evaluate_long_short(df: pd.DataFrame, target_col: str = "target_fwd_ret_1d") -> pd.DataFrame:
    if target_col not in df.columns:
        raise RuntimeError(f"evaluate_long_short: missing target_col={target_col}")
    if "date" not in df.columns:
        raise RuntimeError("evaluate_long_short: missing date")
    if "weight" not in df.columns:
        raise RuntimeError("evaluate_long_short: missing weight")

    out = df.copy()
    out = out.loc[out[target_col].notna()].copy()

    weight = pd.to_numeric(out["weight"], errors="coerce").fillna(0.0)
    fwd = pd.to_numeric(out[target_col], errors="coerce").fillna(0.0)
    out["gross_pnl"] = weight * fwd

    if "cost_trading" not in out.columns:
        out["cost_trading"] = 0.0
    if "cost_borrow" not in out.columns:
        out["cost_borrow"] = 0.0
    if "cost_total" not in out.columns:
        out["cost_total"] = pd.to_numeric(out["cost_trading"], errors="coerce").fillna(0.0) + pd.to_numeric(out["cost_borrow"], errors="coerce").fillna(0.0)

    out["net_pnl"] = out["gross_pnl"] - pd.to_numeric(out["cost_total"], errors="coerce").fillna(0.0)

    turnover_col = "trade_abs_after" if "trade_abs_after" in out.columns else None

    agg_map = {
        "portfolio_ret": ("net_pnl", "sum"),
        "gross_ret": ("gross_pnl", "sum"),
        "trading_costs": ("cost_trading", "sum"),
        "borrow_costs": ("cost_borrow", "sum"),
        "costs": ("cost_total", "sum"),
        "positions": ("weight", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0.0) != 0.0).sum())),
    }
    if turnover_col is not None:
        agg_map["turnover"] = (turnover_col, "sum")

    res = out.groupby("date", as_index=False).agg(**agg_map)
    return res



def summarize_daily_returns(daily_df: pd.DataFrame) -> dict[str, float]:
    if daily_df.empty:
        raise RuntimeError("summarize_daily_returns: empty daily_df")
    if "portfolio_ret" not in daily_df.columns:
        raise RuntimeError("summarize_daily_returns: missing portfolio_ret")

    s = pd.to_numeric(daily_df["portfolio_ret"], errors="coerce").dropna()
    if s.empty:
        raise RuntimeError("summarize_daily_returns: no valid portfolio_ret values")

    out = {
        "days": float(len(s)),
        "avg_daily_ret": float(s.mean()),
        "std_daily_ret": float(s.std()),
        "win_rate_days": float((s > 0).mean()),
        "cum_ret": float((1.0 + s).prod() - 1.0),
    }
    for col, key in [
        ("turnover", "avg_turnover"),
        ("gross_ret", "avg_gross_ret"),
        ("trading_costs", "avg_trading_costs"),
        ("borrow_costs", "avg_borrow_costs"),
        ("costs", "avg_costs"),
    ]:
        if col in daily_df.columns:
            out[key] = float(pd.to_numeric(daily_df[col], errors="coerce").fillna(0.0).mean())
    return out



def print_summary(tag: str, summary: dict[str, float]) -> None:
    extra_parts = []
    for key in ["avg_turnover", "avg_gross_ret", "avg_trading_costs", "avg_borrow_costs", "avg_costs"]:
        if key in summary:
            extra_parts.append(f"{key}={summary[key]:.6f}")
    extra = " " + " ".join(extra_parts) if extra_parts else ""
    print(
        f"{tag} days={int(summary['days'])} "
        f"avg_daily_ret={summary['avg_daily_ret']:.6f} "
        f"std_daily_ret={summary['std_daily_ret']:.6f} "
        f"win_rate_days={summary['win_rate_days']:.4f} "
        f"cum_ret={summary['cum_ret']:.6f}"
        f"{extra}"
    )