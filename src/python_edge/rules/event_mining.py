from __future__ import annotations

"""python_edge.rules.event_mining

Restored full version with:
- q10..q90 quantile grid
- unary rules + band rules
- per-symbol rolling sigma in label_events()
- permutation sanity / gate
- backward compatibility with old caller passing max_rules_try
- optional directional budgets:
    max_rules_try_long
    max_rules_try_short

Compatibility contract
----------------------
run_ranker_mvp.py may still instantiate EventMiningConfig(max_rules_try=...).
This file preserves that API while also allowing separate long/short budgets.
If max_rules_try_long / max_rules_try_short are not provided, they are derived
from max_rules_try.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# Config / structs
# ============================================================


@dataclass(frozen=True)
class EventMiningConfig:
    # event definition
    fwd_days: int = 5
    sigma_lookback: int = 60
    k_sigma: float = 1.5

    # backward-compatible total budget
    max_rules_try: int = 6000

    # optional directional budgets; if None, derived from max_rules_try
    max_rules_try_long: Optional[int] = None
    max_rules_try_short: Optional[int] = None

    max_rules_keep: int = 800
    min_support: int = 200
    min_event_hits: int = 30
    max_conds: int = 3
    seed: int = 7

    # permutation sanity / gate
    perm_trials: int = 50
    perm_topk: int = 20
    perm_gate_enabled: bool = True
    perm_gate_margin: float = 0.15

    def resolved_try_long(self) -> int:
        if self.max_rules_try_long is not None:
            return int(self.max_rules_try_long)
        total = int(self.max_rules_try)
        return total // 2

    def resolved_try_short(self) -> int:
        if self.max_rules_try_short is not None:
            return int(self.max_rules_try_short)
        total = int(self.max_rules_try)
        return total - (total // 2)


@dataclass(frozen=True)
class Rule:
    rule_id: str
    direction: str  # "long" predicts UP events, "short" predicts DOWN events
    # cond: (feature, op, meta_key) where op in {">","<"}, meta_key stores threshold
    conds: List[Tuple[str, str, str]]
    meta: Dict[str, float]


@dataclass(frozen=True)
class RuleStats:
    rule_id: str
    direction: str
    support: int
    event_hits: int
    base_rate: float
    precision: float
    lift: float
    mean_signed: float
    median_signed: float
    p_pos_signed: float
    es5_signed: float
    signature: str


# ============================================================
# Internals
# ============================================================


_QTAGS: Tuple[str, ...] = ("q10", "q20", "q30", "q40", "q50", "q60", "q70", "q80", "q90")


def _es5(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    q = float(np.quantile(x, 0.05))
    tail = x[x <= q]
    return float(np.mean(tail)) if tail.size else q


def _quantile_thresholds(s: pd.Series) -> Dict[str, float]:
    s2 = s.replace([np.inf, -np.inf], np.nan).dropna()
    if len(s2) == 0:
        return {t: 0.0 for t in _QTAGS}

    out: Dict[str, float] = {}
    for t in _QTAGS:
        q = int(t[1:]) / 100.0
        out[t] = float(s2.quantile(q))
    return out


def _apply_rule_mask(df: pd.DataFrame, rule: Rule) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for feat, op, meta_key in rule.conds:
        thr = float(rule.meta[meta_key])
        x = pd.to_numeric(df[feat], errors="coerce")
        if op == ">":
            mask &= x > thr
        elif op == "<":
            mask &= x < thr
        else:
            raise ValueError(f"Unsupported op: {op}")
    return mask.fillna(False)


def _canonical_signature(direction: str, conds: List[Tuple[str, str, str]]) -> str:
    """Signature uses quantile tags (q10..q90) not numeric thresholds."""
    parts: List[str] = []
    for feat, op, meta_key in conds:
        tag = meta_key.split("__")[-1] if "__" in meta_key else meta_key
        parts.append(f"{feat}{op}{tag}")
    parts.sort()
    return direction + "|" + "|".join(parts)


def _pick_tail_tag(rng: np.random.Generator, side: str) -> str:
    if side == "high":
        tags = ["q60", "q70", "q80", "q90"]
        p = np.array([0.20, 0.25, 0.35, 0.20], dtype=float)
    else:
        tags = ["q10", "q20", "q30", "q40"]
        p = np.array([0.20, 0.35, 0.25, 0.20], dtype=float)
    p = p / p.sum()
    return str(rng.choice(tags, p=p))


def _pick_band(rng: np.random.Generator) -> Tuple[str, str]:
    bands = [
        ("q40", "q60"),
        ("q30", "q70"),
        ("q20", "q80"),
        ("q10", "q90"),
        ("q20", "q60"),
        ("q40", "q80"),
    ]
    lo, hi = bands[int(rng.integers(0, len(bands)))]
    return lo, hi


def _make_rule_diverse(
    rng: np.random.Generator,
    rule_id: str,
    direction: str,
    feature_cols: List[str],
    thrs: Dict[str, Dict[str, float]],
    max_conds: int,
) -> Rule:
    """Generate a diverse rule.

    - Allows unary rules.
    - Adds band constraints (2 inequalities on same feature) ~30% when budget allows.
    - Allows some "paradox" op choice (not forced by direction).
    """
    kmax = int(max(1, max_conds))

    choices = [c for c in (1, 2, 3) if c <= kmax]
    base_p = {1: 0.35, 2: 0.45, 3: 0.20}
    probs = np.array([base_p[c] for c in choices], dtype=float)
    probs = probs / probs.sum()
    target = int(rng.choice(choices, p=probs))

    conds: List[Tuple[str, str, str]] = []
    meta: Dict[str, float] = {}
    used: set[str] = set()

    while len(conds) < target:
        remaining = target - len(conds)
        avail = [f for f in feature_cols if f not in used]
        if not avail:
            break
        feat = str(rng.choice(avail))
        used.add(feat)

        do_band = (remaining >= 2) and (rng.random() < 0.30)
        if do_band:
            lo_tag, hi_tag = _pick_band(rng)
            lo_key = f"{feat}__{lo_tag}"
            hi_key = f"{feat}__{hi_tag}"
            meta[lo_key] = float(thrs[feat][lo_tag])
            meta[hi_key] = float(thrs[feat][hi_tag])
            conds.append((feat, ">", lo_key))
            conds.append((feat, "<", hi_key))
            continue

        if direction == "long":
            op = str(rng.choice([">", "<"], p=[0.55, 0.45]))
        else:
            op = str(rng.choice(["<", ">"], p=[0.55, 0.45]))

        tag = _pick_tail_tag(rng, "high" if op == ">" else "low")
        meta_key = f"{feat}__{tag}"
        meta[meta_key] = float(thrs[feat][tag])
        conds.append((feat, op, meta_key))

    return Rule(rule_id=rule_id, direction=direction, conds=conds, meta=meta)


def _score_rule_event(df: pd.DataFrame, rule: Rule, event_col: str, fwd_col: str) -> Optional[RuleStats]:
    m = _apply_rule_mask(df, rule).to_numpy(dtype=bool)
    support = int(np.sum(m))
    if support <= 0:
        return None

    ev = pd.to_numeric(df[event_col], errors="coerce").fillna(0).to_numpy(dtype=int)
    hits = int(np.sum(ev[m] == 1))

    base_rate = float(np.mean(ev == 1)) if ev.size else 0.0
    precision = float(hits / support) if support > 0 else 0.0
    lift = float(precision / base_rate) if base_rate > 0 else float("nan")

    fwd = pd.to_numeric(df[fwd_col], errors="coerce").to_numpy(dtype=float)
    signed = fwd if rule.direction == "long" else (-fwd)
    xs = signed[m]
    xs = xs[np.isfinite(xs)]

    mean_s = float(np.mean(xs)) if xs.size else float("nan")
    med_s = float(np.median(xs)) if xs.size else float("nan")
    ppos = float(np.mean(xs > 0.0)) if xs.size else float("nan")
    es5 = _es5(xs.astype(float)) if xs.size else float("nan")

    sig = _canonical_signature(rule.direction, rule.conds)

    return RuleStats(
        rule_id=rule.rule_id,
        direction=rule.direction,
        support=support,
        event_hits=hits,
        base_rate=base_rate,
        precision=precision,
        lift=lift,
        mean_signed=mean_s,
        median_signed=med_s,
        p_pos_signed=ppos,
        es5_signed=es5,
        signature=sig,
    )


def _permutation_sanity(df_train: pd.DataFrame, rules: List[Rule], cfg: EventMiningConfig) -> Dict[str, float]:
    if (not rules) or int(cfg.perm_trials) <= 0:
        return {
            "perm_trials": float(cfg.perm_trials),
            "perm_topk_mean": float("nan"),
            "perm_topk_p95": float("nan"),
        }

    rng = np.random.default_rng(int(cfg.seed) + 100)
    masks: Dict[str, np.ndarray] = {r.rule_id: _apply_rule_mask(df_train, r).to_numpy(dtype=bool) for r in rules}

    ev_up = pd.to_numeric(df_train["event_up"], errors="coerce").fillna(0).to_numpy(dtype=int)
    ev_dn = pd.to_numeric(df_train["event_dn"], errors="coerce").fillna(0).to_numpy(dtype=int)

    lifts_topk: List[float] = []
    for _ in range(int(cfg.perm_trials)):
        perm_up = ev_up.copy()
        perm_dn = ev_dn.copy()
        rng.shuffle(perm_up)
        rng.shuffle(perm_dn)

        base_up = float(np.mean(perm_up == 1)) if perm_up.size else 0.0
        base_dn = float(np.mean(perm_dn == 1)) if perm_dn.size else 0.0

        lifts: List[float] = []
        for r in rules:
            m = masks[r.rule_id]
            support = int(np.sum(m))
            if support < int(cfg.min_support):
                continue

            if r.direction == "long":
                if base_up <= 0:
                    continue
                hits = int(np.sum(perm_up[m] == 1))
                precision = float(hits / support)
                lift = float(precision / base_up)
            else:
                if base_dn <= 0:
                    continue
                hits = int(np.sum(perm_dn[m] == 1))
                precision = float(hits / support)
                lift = float(precision / base_dn)

            if np.isfinite(lift):
                lifts.append(lift)

        if lifts:
            lifts.sort(reverse=True)
            topk = lifts[: max(1, int(cfg.perm_topk))]
            lifts_topk.append(float(np.mean(topk)))
        else:
            lifts_topk.append(1.0)

    arr = np.array(lifts_topk, dtype=float)
    if arr.size == 0:
        return {
            "perm_trials": float(cfg.perm_trials),
            "perm_topk_mean": float("nan"),
            "perm_topk_p95": float("nan"),
        }

    return {
        "perm_trials": float(cfg.perm_trials),
        "perm_topk_mean": float(np.mean(arr)),
        "perm_topk_p95": float(np.quantile(arr, 0.95)),
    }


# ============================================================
# Public API
# ============================================================


def label_events(df: pd.DataFrame, cfg: EventMiningConfig) -> pd.DataFrame:
    out = df.copy()

    if "ret_1d" not in out.columns:
        raise ValueError("label_events: missing required column ret_1d")

    fwd_col = f"fwd_{int(cfg.fwd_days)}d_ret"
    if fwd_col not in out.columns:
        raise ValueError(f"label_events: missing required column {fwd_col}")

    lookback = int(cfg.sigma_lookback)

    if "symbol" in out.columns:
        sig = (
            out.groupby("symbol", sort=False)["ret_1d"]
            .rolling(lookback)
            .std()
            .reset_index(level=0, drop=True)
        )
    else:
        sig = out["ret_1d"].rolling(lookback).std()

    thr = float(cfg.k_sigma) * sig * float(np.sqrt(float(cfg.fwd_days)))

    out["sigma_roll"] = sig
    out["event_thr"] = thr

    fwd = pd.to_numeric(out[fwd_col], errors="coerce")
    out["event_up"] = (fwd > thr).astype("Int64")
    out["event_dn"] = (fwd < -thr).astype("Int64")
    return out


def mine_event_rules(
    df_train: pd.DataFrame,
    feature_cols: List[str],
    cfg: EventMiningConfig,
    direction: Optional[str] = None,
) -> Tuple[List[Rule], Dict[str, RuleStats], Dict[str, float]]:
    rng = np.random.default_rng(int(cfg.seed))

    thrs: Dict[str, Dict[str, float]] = {c: _quantile_thresholds(df_train[c]) for c in feature_cols}
    fwd_col = f"fwd_{int(cfg.fwd_days)}d_ret"

    candidates: List[Rule] = []

    try_long = int(cfg.resolved_try_long())
    try_short = int(cfg.resolved_try_short())

    if direction in (None, "long"):
        for k in range(try_long):
            rid = f"EL{k:05d}"
            candidates.append(_make_rule_diverse(rng, rid, "long", feature_cols, thrs, int(cfg.max_conds)))

    if direction in (None, "short"):
        for k in range(try_short):
            rid = f"ES{k:05d}"
            candidates.append(_make_rule_diverse(rng, rid, "short", feature_cols, thrs, int(cfg.max_conds)))

    # keep best 2 variants per signature
    best_by_sig: Dict[str, List[Tuple[Rule, RuleStats]]] = {}

    def _score_key(st: RuleStats) -> Tuple[float, float, int, float]:
        return (float(st.lift), float(st.precision), int(st.support), float(st.mean_signed))

    for r in candidates:
        event_col = "event_up" if r.direction == "long" else "event_dn"
        st = _score_rule_event(df_train, r, event_col=event_col, fwd_col=fwd_col)
        if st is None:
            continue
        if st.support < int(cfg.min_support):
            continue
        if st.event_hits < int(cfg.min_event_hits):
            continue
        if (not np.isfinite(st.lift)) or st.lift <= 1.05:
            continue
        if (not np.isfinite(st.mean_signed)) or st.mean_signed <= 0.0:
            continue

        bucket = best_by_sig.setdefault(st.signature, [])
        bucket.append((r, st))
        bucket.sort(key=lambda rs: _score_key(rs[1]), reverse=True)
        if len(bucket) > 2:
            del bucket[2:]

    rules: List[Rule] = []
    stats: Dict[str, RuleStats] = {}
    for bucket in best_by_sig.values():
        for r, st in bucket:
            rules.append(r)
            stats[r.rule_id] = st

    perm = _permutation_sanity(df_train, rules, cfg)
    perm_p95 = float(perm.get("perm_topk_p95", float("nan")))

    if bool(cfg.perm_gate_enabled) and np.isfinite(perm_p95):
        gate_thr = float(perm_p95) + float(cfg.perm_gate_margin)
        gated_rules: List[Rule] = []
        gated_stats: Dict[str, RuleStats] = {}
        for r in rules:
            st = stats[r.rule_id]
            if st.lift > gate_thr:
                gated_rules.append(r)
                gated_stats[r.rule_id] = st
        rules, stats = gated_rules, gated_stats

    def _rank_key(rr: Rule) -> float:
        st = stats[rr.rule_id]
        return (
            (st.lift - 1.0) * 2.0
            + st.precision * 1.0
            + st.mean_signed * 5.0
            + float(np.log1p(st.event_hits)) * 0.2
        )

    rules = sorted(rules, key=_rank_key, reverse=True)[: int(cfg.max_rules_keep)]
    stats = {r.rule_id: stats[r.rule_id] for r in rules}

    return rules, stats, perm


def evaluate_rules_oos(df_oos: pd.DataFrame, rules: List[Rule], cfg: EventMiningConfig) -> pd.DataFrame:
    if df_oos.empty or not rules:
        return pd.DataFrame()

    fwd_col = f"fwd_{int(cfg.fwd_days)}d_ret"
    rows: List[Dict[str, object]] = []

    for r in rules:
        event_col = "event_up" if r.direction == "long" else "event_dn"
        st = _score_rule_event(df_oos, r, event_col=event_col, fwd_col=fwd_col)
        if st is None:
            continue
        rows.append(
            {
                "rule_id": st.rule_id,
                "direction": st.direction,
                "signature": st.signature,
                "support": st.support,
                "event_hits": st.event_hits,
                "base_rate": st.base_rate,
                "precision": st.precision,
                "lift": st.lift,
                "mean_signed": st.mean_signed,
                "median_signed": st.median_signed,
                "p_pos_signed": st.p_pos_signed,
                "es5_signed": st.es5_signed,
            }
        )

    return pd.DataFrame(rows)
