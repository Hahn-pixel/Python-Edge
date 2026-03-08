from __future__ import annotations

import pandas as pd



def evaluate_long_short(df: pd.DataFrame, target_col: str = "target_fwd_ret_1d") -> pd.DataFrame:
    if "side" not in df.columns:
        raise RuntimeError("evaluate_long_short: missing side")
    if target_col not in df.columns:
        raise RuntimeError(f"evaluate_long_short: missing target_col={target_col}")
    if "date" not in df.columns:
        raise RuntimeError("evaluate_long_short: missing date")

    out = df.copy()
    out = out.loc[out[target_col].notna()].copy()

    if "weight" in out.columns:
        exposure = pd.to_numeric(out["weight"], errors="coerce").fillna(0.0)
    else:
        exposure = pd.to_numeric(out["side"], errors="coerce").fillna(0.0)

    out["gross_pnl"] = exposure * pd.to_numeric(out[target_col], errors="coerce").fillna(0.0)

    if "cost_total" in out.columns:
        out["net_pnl"] = out["gross_pnl"] - pd.to_numeric(out["cost_total"], errors="coerce").fillna(0.0) * exposure.abs()
    else:
        out["net_pnl"] = out["gross_pnl"]

    res = out.groupby("date", as_index=False).agg(
        portfolio_ret=("net_pnl", "sum"),
        gross_ret=("gross_pnl", "sum"),
        positions=("side", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0.0) != 0).sum())),
        turnover=("turnover_unit_after", "sum") if "turnover_unit_after" in out.columns else ("side", lambda s: 0.0),
        costs=("cost_total", "sum") if "cost_total" in out.columns else ("side", lambda s: 0.0),
    )
    return res



def summarize_daily_returns(daily_df: pd.DataFrame) -> dict[str, float]:
    if daily_df.empty:
        raise RuntimeError("summarize_daily_returns: empty daily_df")
    if "portfolio_ret" not in daily_df.columns:
        raise RuntimeError("summarize_daily_returns: missing portfolio_ret")

    s = pd.to_numeric(daily_df["portfolio_ret"], errors="coerce").dropna()
    if s.empty:
        raise RuntimeError("summarize_daily_returns: no valid portfolio_ret values")

    cum = float((1.0 + s).prod() - 1.0)
    win_rate = float((s > 0).mean())
    out = {
        "days": float(len(s)),
        "avg_daily_ret": float(s.mean()),
        "std_daily_ret": float(s.std()),
        "win_rate_days": win_rate,
        "cum_ret": cum,
    }
    if "turnover" in daily_df.columns:
        out["avg_turnover"] = float(pd.to_numeric(daily_df["turnover"], errors="coerce").fillna(0.0).mean())
    if "costs" in daily_df.columns:
        out["avg_costs"] = float(pd.to_numeric(daily_df["costs"], errors="coerce").fillna(0.0).mean())
    return out



def print_summary(tag: str, summary: dict[str, float]) -> None:
    extra = ""
    if "avg_turnover" in summary:
        extra += f" avg_turnover={summary['avg_turnover']:.6f}"
    if "avg_costs" in summary:
        extra += f" avg_costs={summary['avg_costs']:.6f}"
    print(
        f"{tag} days={int(summary['days'])} "
        f"avg_daily_ret={summary['avg_daily_ret']:.6f} "
        f"std_daily_ret={summary['std_daily_ret']:.6f} "
        f"win_rate_days={summary['win_rate_days']:.4f} "
        f"cum_ret={summary['cum_ret']:.6f}"
        f"{extra}"
    )