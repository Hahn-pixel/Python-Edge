from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Rule:
    rule_id: str
    direction: str  # "long" or "short"
    # list of (feature, op, threshold_name) where threshold_name maps to numeric value in metadata
    conds: List[Tuple[str, str, str]]
    meta: Dict[str, float]


@dataclass(frozen=True)
class RuleScore:
    rule_id: str
    direction: str
    n: int
    mean: float
    median: float
    es_5: float
    sharpe: float


def _es_5(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    q = np.quantile(x, 0.05)
    tail = x[x <= q]
    return float(np.mean(tail)) if tail.size else float(q)


def _sharpe(x: np.ndarray) -> float:
    if x.size < 10:
        return float("nan")
    mu = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    if sd <= 0:
        return float("nan")
    return mu / sd


def _quantile_thresholds(s: pd.Series) -> Dict[str, float]:
    s2 = s.replace([np.inf, -np.inf], np.nan).dropna()
    if len(s2) < 100:
        # fallback
        return {"q20": float(s2.quantile(0.2)) if len(s2) else 0.0,
                "q50": float(s2.quantile(0.5)) if len(s2) else 0.0,
                "q80": float(s2.quantile(0.8)) if len(s2) else 0.0}
    return {
        "q20": float(s2.quantile(0.2)),
        "q50": float(s2.quantile(0.5)),
        "q80": float(s2.quantile(0.8)),
    }


def _apply_rule_mask(df: pd.DataFrame, rule: Rule) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for feat, op, thr_name in rule.conds:
        thr = rule.meta[thr_name]
        if op == ">":
            mask &= (df[feat] > thr)
        elif op == "<":
            mask &= (df[feat] < thr)
        else:
            raise ValueError(f"Unsupported op: {op}")
    return mask


def mine_rules(
    df_train: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    min_trades: int = 80,
    max_rules: int = 400,
    seed: int = 7,
) -> Tuple[List[Rule], Dict[str, RuleScore]]:
    """
    Produces a library of simple conjunction rules using quantile thresholds.
    - long rules target positive forward returns
    - short rules target negative forward returns
    """
    rng = np.random.default_rng(seed)

    # thresholds per feature
    thrs: Dict[str, Dict[str, float]] = {}
    for c in feature_cols:
        thrs[c] = _quantile_thresholds(df_train[c])

    candidates: List[Rule] = []

    # generate rules: 2-3 conditions, randomized combinations (fast + broad)
    feat_pool = [c for c in feature_cols]
    if len(feat_pool) < 4:
        return [], {}

    def make_rule(rule_id: str, direction: str, feats: List[str]) -> Rule:
        conds = []
        meta: Dict[str, float] = {}
        for f in feats:
            # choose op + threshold name based on direction (slightly biased but still broad)
            if direction == "long":
                op = rng.choice([">", "<"], p=[0.7, 0.3])
                thr_name = rng.choice(["q50", "q80"] if op == ">" else ["q20", "q50"])
            else:
                op = rng.choice(["<", ">"], p=[0.7, 0.3])
                thr_name = rng.choice(["q20", "q50"] if op == "<" else ["q50", "q80"])

            meta_key = f"{f}__{thr_name}"
            meta[meta_key] = thrs[f][thr_name]
            conds.append((f, op, meta_key))
        return Rule(rule_id=rule_id, direction=direction, conds=conds, meta=meta)

    # create candidates
    n_try = max_rules * 6
    for k in range(n_try):
        direction = "long" if (k % 2 == 0) else "short"
        k_conds = int(rng.choice([2, 3], p=[0.7, 0.3]))
        feats = list(rng.choice(feat_pool, size=k_conds, replace=False))
        rid = f"R{direction[0].upper()}{k:05d}"
        candidates.append(make_rule(rid, direction, feats))

    # score candidates
    scores: Dict[str, RuleScore] = {}
    kept: List[Rule] = []

    y = df_train[target_col].to_numpy(dtype=float)

    for r in candidates:
        m = _apply_rule_mask(df_train, r).to_numpy()
        x = y[m]
        if x.size < min_trades:
            continue
        # direction consistency: for short rules, flip sign for scoring (so "good" is positive)
        x_eff = x if r.direction == "long" else (-x)

        mean = float(np.mean(x_eff))
        med = float(np.median(x_eff))
        es5 = _es_5(x_eff)
        sh = _sharpe(x_eff)

        # basic viability filter: must have positive mean and non-horrific ES
        if not np.isfinite(mean) or mean <= 0:
            continue
        if np.isfinite(es5) and es5 < -0.05:  # very rough tail sanity
            continue

        scores[r.rule_id] = RuleScore(
            rule_id=r.rule_id,
            direction=r.direction,
            n=int(x.size),
            mean=mean,
            median=med,
            es_5=float(es5),
            sharpe=float(sh) if np.isfinite(sh) else float("nan"),
        )
        kept.append(r)

    # select top rules by (mean - 0.5*abs(es5)) with size penalty
    def rank_key(r: Rule) -> float:
        sc = scores[r.rule_id]
        es_pen = 0.5 * abs(sc.es_5) if np.isfinite(sc.es_5) else 0.0
        size_pen = 0.0001 * max(0, (min_trades - sc.n))
        return sc.mean - es_pen - size_pen

    kept = sorted(kept, key=rank_key, reverse=True)[:max_rules]
    scores = {r.rule_id: scores[r.rule_id] for r in kept}

    return kept, scores


def apply_rules_signals(df: pd.DataFrame, rules: List[Rule]) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    for r in rules:
        out[r.rule_id] = _apply_rule_mask(df, r)
    return out