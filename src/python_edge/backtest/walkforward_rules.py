from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from python_edge.rules.rule_miner import Rule, RuleScore, apply_rules_signals, mine_rules


@dataclass(frozen=True)
class WFConfig:
    n_folds: int = 6
    train_days: int = 420  # ~1.6y trading days
    test_days: int = 90    # ~1q
    purge_days: int = 10
    max_rules: int = 250
    min_trades: int = 60
    seed: int = 7


@dataclass(frozen=True)
class FoldResult:
    fold_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    n_rules: int
    oos_mean: float
    oos_es5: float
    oos_n: int


def _es_5(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    q = np.quantile(x, 0.05)
    tail = x[x <= q]
    return float(np.mean(tail)) if tail.size else float(q)


def _make_folds(dates: List[str], cfg: WFConfig) -> List[Tuple[int, int, int, int]]:
    """
    Returns index slices (train_start, train_end_excl, test_start, test_end_excl)
    with purge gap between train and test.
    """
    n = len(dates)
    folds = []
    # rolling forward by test_days
    # last fold must fit train + purge + test
    step = cfg.test_days
    # start train window so that we have enough room
    start_train_end = cfg.train_days
    fold_id = 0
    t_end = start_train_end
    while True:
        train_end = t_end
        test_start = train_end + cfg.purge_days
        test_end = test_start + cfg.test_days
        train_start = max(0, train_end - cfg.train_days)
        if test_end > n:
            break
        folds.append((train_start, train_end, test_start, test_end))
        fold_id += 1
        t_end += step
        if fold_id >= cfg.n_folds:
            break
    return folds


def walkforward_rule_selection(
    df_all: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    cfg: WFConfig,
) -> Tuple[List[FoldResult], Dict[int, List[Rule]], Dict[int, Dict[str, RuleScore]]]:
    """
    df_all: panel flattened for one symbol OR pooled cross-section. Must include 'date' sorted.
    In MVP we pool cross-section per-date by stacking (date, symbol) rows.
    """
    df_all = df_all.sort_values(["date", "symbol"]).reset_index(drop=True)
    dates = sorted(df_all["date"].unique().tolist())
    folds = _make_folds(dates, cfg)

    fold_results: List[FoldResult] = []
    fold_rules: Dict[int, List[Rule]] = {}
    fold_scores: Dict[int, Dict[str, RuleScore]] = {}

    for k, (i0, i1, j0, j1) in enumerate(folds, 1):
        train_dates = dates[i0:i1]
        test_dates = dates[j0:j1]
        if not train_dates or not test_dates:
            continue

        df_tr = df_all[df_all["date"].isin(train_dates)].copy()
        df_te = df_all[df_all["date"].isin(test_dates)].copy()

        rules, scores = mine_rules(
            df_train=df_tr,
            feature_cols=feature_cols,
            target_col=target_col,
            min_trades=cfg.min_trades,
            max_rules=cfg.max_rules,
            seed=cfg.seed + k,
        )

        # Evaluate OOS aggregate (simple): take all rule firings equally (direction-adjusted)
        y_te = df_te[target_col].to_numpy(dtype=float)
        if rules:
            sigs = apply_rules_signals(df_te, rules)
            fired = np.zeros(len(df_te), dtype=bool)
            signed = np.zeros(len(df_te), dtype=float)

            for r in rules:
                m = sigs[r.rule_id].to_numpy()
                if not m.any():
                    continue
                # for each row, add signed return contribution
                # (MVP: average across rules that fired)
                contrib = y_te.copy()
                if r.direction == "short":
                    contrib = -contrib
                signed[m] += contrib[m]
                fired |= m

            # normalize by number of rules fired per row to avoid overweighting heavy-fire dates
            # (MVP: per-row average)
            # count fires:
            cnt = np.zeros(len(df_te), dtype=float)
            for r in rules:
                m = sigs[r.rule_id].to_numpy()
                cnt[m] += 1.0
            cnt[cnt == 0] = np.nan
            eff = signed / cnt
            eff = eff[np.isfinite(eff)]
        else:
            eff = np.array([], dtype=float)

        oos_mean = float(np.mean(eff)) if eff.size else 0.0
        oos_es5 = float(_es_5(eff)) if eff.size else 0.0

        fold_results.append(FoldResult(
            fold_id=k,
            train_start=train_dates[0],
            train_end=train_dates[-1],
            test_start=test_dates[0],
            test_end=test_dates[-1],
            n_rules=len(rules),
            oos_mean=oos_mean,
            oos_es5=oos_es5,
            oos_n=int(eff.size),
        ))
        fold_rules[k] = rules
        fold_scores[k] = scores

    return fold_results, fold_rules, fold_scores