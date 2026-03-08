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
    out["pnl"] = out["side"] * out[target_col]

    res = out.groupby("date", as_index=False).agg(
        portfolio_ret=("pnl", "mean"),
        positions=("side", lambda s: int((s != 0).sum())),
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
    return {
        "days": float(len(s)),
        "avg_daily_ret": float(s.mean()),
        "std_daily_ret": float(s.std()),
        "win_rate_days": win_rate,
        "cum_ret": cum,
    }



def print_summary(tag: str, summary: dict[str, float]) -> None:
    print(
        f"{tag} days={int(summary['days'])} "
        f"avg_daily_ret={summary['avg_daily_ret']:.6f} "
        f"std_daily_ret={summary['std_daily_ret']:.6f} "
        f"win_rate_days={summary['win_rate_days']:.4f} "
        f"cum_ret={summary['cum_ret']:.6f}"
    )