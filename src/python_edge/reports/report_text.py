from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


def _max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    return float(np.min(dd))


def summarize_portfolio(df: pd.DataFrame, label: str) -> str:
    if df.empty:
        return f"[{label}] empty\n"

    ret = df["ret_net"].to_numpy(dtype=float)
    eq = df["equity_net"].to_numpy(dtype=float)

    mu = float(np.mean(ret))
    sd = float(np.std(ret, ddof=1)) if ret.size > 1 else 0.0
    sharpe = (mu / sd) * np.sqrt(252.0) if sd > 0 else float("nan")
    mdd = _max_drawdown(eq)

    cagr = float(eq[-1] ** (252.0 / max(1, len(eq))) - 1.0)

    q05 = float(np.quantile(ret, 0.05)) if ret.size else 0.0
    es5 = float(np.mean(ret[ret <= q05])) if ret.size else 0.0

    avg_pos = float(df["n_pos"].mean())

    lines = []
    lines.append(f"[{label}] n_days={len(df)} avg_pos={avg_pos:.2f}")
    lines.append(f"[{label}] equity_end={eq[-1]:.4f} CAGR~={cagr:.3%} Sharpe~={sharpe:.2f} MaxDD={mdd:.2%}")
    lines.append(f"[{label}] ES(5%)={es5:.4%} q05={q05:.4%} mean_day={mu:.4%} sd_day={sd:.4%}")
    return "\n".join(lines) + "\n"