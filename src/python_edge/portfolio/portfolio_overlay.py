from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from python_edge.rules.rule_miner import Rule, RuleScore, apply_rules_signals


@dataclass(frozen=True)
class PortfolioConfig:
    max_positions: int = 5
    max_long: int = 3
    max_short: int = 2
    cost_bps: float = 10.0
    hold_days: int = 5  # holds for target horizon (MVP)
    rebalance: str = "D"  # daily


def _cost_per_trade(cost_bps: float) -> float:
    return float(cost_bps) * 1e-4


def build_portfolio_oos(
    df_oos: pd.DataFrame,
    rules: List[Rule],
    scores: Dict[str, RuleScore],
    target_col: str,
    cfg: PortfolioConfig,
) -> pd.DataFrame:
    """
    Very simple MVP portfolio:
    - each day, evaluate all (date,symbol) rows
    - for each rule, if it fires, produce a candidate signal with score = rule_score.mean - 0.5*abs(es5)
    - select up to max_long / max_short with highest scores
    - realized return uses forward target_col (already forward return over hold horizon)
    - cost applied per position opened
    """
    if df_oos.empty:
        return pd.DataFrame()

    df = df_oos.sort_values(["date", "symbol"]).reset_index(drop=True)

    sigs = apply_rules_signals(df, rules)

    # build candidate list per row: best rule firing per direction
    best_long = np.full(len(df), -np.inf, dtype=float)
    best_short = np.full(len(df), -np.inf, dtype=float)

    y = df[target_col].to_numpy(dtype=float)

    for r in rules:
        sc = scores.get(r.rule_id)
        if sc is None:
            continue
        # quality score (MVP)
        q = sc.mean - 0.5 * abs(sc.es_5 if np.isfinite(sc.es_5) else 0.0)
        m = sigs[r.rule_id].to_numpy()
        if not m.any():
            continue
        if r.direction == "long":
            best_long[m] = np.maximum(best_long[m], q)
        else:
            best_short[m] = np.maximum(best_short[m], q)

    df["best_long_score"] = best_long
    df["best_short_score"] = best_short

    # per-date selection
    dates = sorted(df["date"].unique().tolist())
    rows = []
    cost = _cost_per_trade(cfg.cost_bps)

    for d in dates:
        day = df[df["date"] == d].copy()
        # long picks
        longs = day[np.isfinite(day["best_long_score"]) & (day["best_long_score"] > -np.inf)].copy()
        shorts = day[np.isfinite(day["best_short_score"]) & (day["best_short_score"] > -np.inf)].copy()

        longs = longs.sort_values("best_long_score", ascending=False).head(cfg.max_long)
        shorts = shorts.sort_values("best_short_score", ascending=False).head(cfg.max_short)

        picks = []
        for _, r in longs.iterrows():
            picks.append(("long", r["symbol"], float(r["best_long_score"]), float(r[target_col])))
        for _, r in shorts.iterrows():
            # short realized return = -forward return
            picks.append(("short", r["symbol"], float(r["best_short_score"]), float(-r[target_col])))

        if not picks:
            rows.append({"date": d, "n_pos": 0, "ret_gross": 0.0, "ret_net": 0.0})
            continue

        # equal weight across positions
        rets = np.array([p[3] for p in picks], dtype=float)
        ret_gross = float(np.mean(rets))
        # apply cost per position opened (MVP: open&close cost collapsed to one hit; conservative users can double this)
        ret_net = ret_gross - cost * len(picks)

        rows.append({"date": d, "n_pos": int(len(picks)), "ret_gross": ret_gross, "ret_net": ret_net})

    out = pd.DataFrame(rows)
    out["equity_gross"] = (1.0 + out["ret_gross"].fillna(0.0)).cumprod()
    out["equity_net"] = (1.0 + out["ret_net"].fillna(0.0)).cumprod()
    return out