# scripts/run_ranker_mvp.py
# Double-click runnable. Never auto-closes (always waits for Enter).
#
# MVP runner:
# massive daily ETF-first dataset -> build features -> label events -> mine rules -> OOS filter
# -> (optional) recency/stability selection -> (optional) HEALTH filter -> rank top-K -> report.
#
# Output is intentionally compact. No silent fail-open.

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# =========================
# Runtime / UX
# =========================

def _press_enter_exit(code: int) -> None:
    # Always wait for user input (double-click runnable).
    try:
        print(f"\n[EXIT] code={code}")
        input("Press Enter to exit...")
    except Exception:
        pass
    raise SystemExit(code)


def _add_src_to_syspath() -> Path:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if src.exists():
        sys.path.insert(0, str(src))
    return root


def _env_str(name: str, default: str = "") -> str:
    v = os.environ.get(name, default)
    return (v if v is not None else default).strip()


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or not str(v).strip():
        return default
    try:
        return int(float(str(v).strip()))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None or not str(v).strip():
        return default
    try:
        return float(str(v).strip())
    except Exception:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None or not str(v).strip():
        return default
    s = str(v).strip().lower()
    return s in ("1", "true", "yes", "y", "on")


def _fmt(x: float, nd: int = 4) -> str:
    if x != x:
        return "nan"
    return f"{x:.{nd}f}"


# =========================
# Config
# =========================


@dataclass(frozen=True)
class HealthConfig:
    enabled: bool = False
    win: int = 60
    min_n: int = 50
    med_min: float = 0.0
    es5_min: float = -0.08
    max_print: int = 5


@dataclass(frozen=True)
class RecencyConfig:
    must_pass_latest: bool = False
    latest_only: bool = False


@dataclass(frozen=True)
class FoldConfig:
    # Defaults match your earlier 3-fold setup.
    train_days: int = 420
    test_days: int = 90
    purge_days: int = 10
    max_folds: int = 3


# =========================
# Helpers
# =========================


def _load_universe(root: Path) -> List[str]:
    p = root / "data" / "universe_etf_first_30.txt"
    if not p.exists():
        raise RuntimeError(f"Missing universe file: {p}")
    txt = p.read_text(encoding="utf-8", errors="ignore")
    out: List[str] = []
    for line in txt.splitlines():
        s = line.strip()
        if (not s) or s.startswith("#"):
            continue
        out.append(s.upper())
    return out


def _make_folds(
    dates: Sequence[str],
    cfg: FoldConfig,
) -> List[Tuple[int, int, int, int, int]]:
    """Returns list of (fold_id, train_start_i, train_end_i, test_start_i, test_end_i)."""
    folds: List[Tuple[int, int, int, int, int]] = []
    t_end = cfg.train_days
    fid = 0
    while True:
        train_end = t_end
        test_start = train_end + cfg.purge_days
        test_end = test_start + cfg.test_days
        train_start = max(0, train_end - cfg.train_days)
        if test_end > len(dates):
            break
        fid += 1
        folds.append((fid, train_start, train_end, test_start, test_end))
        t_end += cfg.test_days
        if fid >= cfg.max_folds:
            break
    return folds


def _apply_rule_mask(df: pd.DataFrame, rule) -> pd.Series:
    # Compatible with python_edge.rules.event_mining.Rule
    mask = pd.Series(True, index=df.index)
    for feat, op, meta_key in getattr(rule, "conds", []):
        thr = rule.meta[meta_key]
        if op == ">":
            mask &= (df[feat] > thr)
        elif op == "<":
            mask &= (df[feat] < thr)
        else:
            raise ValueError(f"Unsupported op: {op}")
    return mask


def _weight_from_metrics(avg_lift: float, fold_count: int, avg_es5: float, oos_lift_min: float) -> float:
    # Simple, monotone scheme: edge above threshold, reward fold_count, penalize nasty tails.
    edge = max(0.0, avg_lift - oos_lift_min)
    stability = 1.25 if fold_count >= 3 else (1.10 if fold_count >= 2 else 1.0)
    tail_pen = max(0.0, (-avg_es5) - 0.10)  # penalty only if ES worse than -10%
    w = stability * edge / (1.0 + 5.0 * tail_pen)
    return float(w)


def _health_filter_rules(
    df_train: pd.DataFrame,
    rules: List[object],
    direction: str,
    cfg_event,
    cfg_health: HealthConfig,
) -> Tuple[List[object], List[Tuple[str, int, float, float]]]:
    """Evaluate each rule on LAST cfg_health.win train dates and retire if failing.

    Returns (kept_rules, retired_info)
    retired_info items: (signature, support, median_signed, es5_signed)
    """

    if (not cfg_health.enabled) or (not rules) or df_train.empty:
        return rules, []

    from python_edge.rules.event_mining import score_rule_event

    dates = sorted(df_train["date"].unique().tolist())
    use_dates = dates[-cfg_health.win :] if len(dates) > cfg_health.win else dates
    df_h = df_train[df_train["date"].isin(use_dates)].copy()

    fwd_col = f"fwd_{cfg_event.fwd_days}d_ret"
    event_col = "event_up" if direction == "long" else "event_dn"

    kept: List[object] = []
    retired: List[Tuple[str, int, float, float]] = []

    for r in rules:
        try:
            st = score_rule_event(df_h, r, event_col=event_col, fwd_col=fwd_col)
        except Exception:
            st = None

        if st is None:
            retired.append(("", 0, float("nan"), float("nan")))
            continue

        support = int(st.support)
        med = float(st.median_signed)
        es5 = float(st.es5_signed)

        ok = True
        if support < cfg_health.min_n:
            ok = False
        if np.isfinite(med) and med < cfg_health.med_min:
            ok = False
        if np.isfinite(es5) and es5 < cfg_health.es5_min:
            ok = False

        if ok:
            kept.append(r)
        else:
            retired.append((str(st.signature), support, med, es5))

    return kept, retired


def _topk_perf(df_fold: pd.DataFrame, net: np.ndarray, k: int, fwd_col: str) -> Tuple[float, float, float]:
    # For each date: pick top-k by net score; collect signed fwd ret.
    if df_fold.empty:
        return 0.0, 0.0, 0.0
    df = df_fold[["date", fwd_col]].copy()
    df["net"] = net
    xs: List[float] = []
    for d, g in df.groupby("date", sort=True):
        gg = g.sort_values("net", ascending=False)
        top = gg.head(k)
        vals = top[fwd_col].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        xs.extend(vals.tolist())
    if not xs:
        return 0.0, 0.0, 0.0
    arr = np.array(xs, dtype=float)
    mean = float(np.mean(arr))
    med = float(np.median(arr))
    ppos = float(np.mean(arr > 0))
    return mean, med, ppos


# =========================
# Main
# =========================


def main() -> int:
    root = _add_src_to_syspath()

    from python_edge.data.ingest_aggs import load_aggs, to_daily_index
    from python_edge.features.build_features_daily import DailyFeatureConfig, build_daily_features
    from python_edge.rules.event_mining import (
        EventMiningConfig,
        evaluate_rules_oos,
        label_events,
        mine_event_rules,
    )

    # ---------- ENV ----------
    start = _env_str("DATA_START", "")
    end = _env_str("DATA_END", "")
    if not start or not end:
        raise RuntimeError("Missing DATA_START/DATA_END. Set env vars before running.")

    dataset_root = Path(_env_str("DATASET_ROOT", str(root / "data" / "raw" / "massive_dataset")))

    # Event / mining config
    fwd_days = _env_int("FWD_DAYS", 5)
    k_sigma = _env_float("K_SIGMA", 1.5)
    sigma_lookback = _env_int("SIGMA_LOOKBACK", 60)

    rules_try = _env_int("RULES_TRY", 6000)
    rules_keep = _env_int("RULES_KEEP", 800)
    min_support = _env_int("MIN_SUPPORT", 200)
    min_event_hits = _env_int("MIN_EVENT_HITS", 30)

    perm_trials = _env_int("PERM_TRIALS", 50)
    perm_topk = _env_int("PERM_TOPK", 20)
    perm_gate = _env_bool("PERM_GATE", True)
    perm_margin = _env_float("PERM_MARGIN", 0.15)

    oos_lift_min = _env_float("OOS_LIFT_MIN", 1.35)

    # Rank / report
    k_top = _env_int("RANK_K", 3)

    # HEALTH / RECENCY
    health = HealthConfig(
        enabled=_env_bool("HEALTH", False),
        win=_env_int("HEALTH_WIN", 60),
        min_n=_env_int("HEALTH_MIN_N", 50),
        med_min=_env_float("HEALTH_MED_MIN", 0.0),
        es5_min=_env_float("HEALTH_ES5_MIN", -0.08),
        max_print=_env_int("HEALTH_MAX_PRINT", 5),
    )

    recency = RecencyConfig(
        must_pass_latest=_env_bool("MUST_PASS_LATEST", False),
        latest_only=_env_bool("LATEST_ONLY", False),
    )

    folds_cfg = FoldConfig(
        train_days=_env_int("TRAIN_DAYS", 420),
        test_days=_env_int("TEST_DAYS", 90),
        purge_days=_env_int("PURGE_DAYS", 10),
        max_folds=_env_int("MAX_FOLDS", 3),
    )

    # ---------- CFG print ----------
    universe = _load_universe(root)
    print(f"[CFG] vendor=massive dataset_root={dataset_root}")
    print(f"[CFG] universe={len(universe)} start={start} end={end}")
    print(f"[CFG] event: fwd_days={fwd_days} k_sigma={k_sigma} sigma_lookback={sigma_lookback}")
    print(f"[CFG] rules: try={rules_try} keep={rules_keep} min_support={min_support} min_event_hits={min_event_hits}")
    print(f"[CFG] perm: trials={perm_trials} topk={perm_topk} gate={int(perm_gate)} margin={perm_margin}")
    print(f"[CFG] OOS filter: support>={min_support} lift>={oos_lift_min}")
    print(f"[CFG] rank: K={k_top}")
    print(
        f"[CFG] health: enabled={int(health.enabled)} win={health.win} min_n={health.min_n} med_min={_fmt(health.med_min)} es5_min={_fmt(health.es5_min)}"
    )
    print(
        f"[CFG] recency: MUST_PASS_LATEST={int(recency.must_pass_latest)} LATEST_ONLY={int(recency.latest_only)}"
    )

    # ---------- Load data ----------
    rows_all: List[pd.DataFrame] = []
    for sym in universe:
        r = load_aggs(dataset_root=dataset_root, symbol=sym, tf="1D", start=start, end=end, prefer_full=True)
        d = to_daily_index(r.df)
        if d.empty:
            continue
        d = d[[c for c in ["date", "o", "h", "l", "c", "v"] if c in d.columns]].copy()
        d["symbol"] = sym
        # build features per symbol (expects o/h/l/c/v)
        feats_cfg = DailyFeatureConfig(fwd_days=fwd_days)
        d = build_daily_features(d, feats_cfg)
        # unify naming expected by event_mining
        d = d.rename(columns={"c": "close"})
        # label events using event_mining thresholds
        cfg_event = EventMiningConfig(
            fwd_days=fwd_days,
            sigma_lookback=sigma_lookback,
            k_sigma=k_sigma,
            max_rules_try=rules_try,
            max_rules_keep=rules_keep,
            min_support=min_support,
            min_event_hits=min_event_hits,
            seed=7,
            perm_trials=perm_trials,
            perm_topk=perm_topk,
            perm_gate_enabled=perm_gate,
            perm_gate_margin=perm_margin,
        )
        d = label_events(d, cfg_event)
        rows_all.append(d)

    if not rows_all:
        raise RuntimeError("No data loaded.")

    pooled = pd.concat(rows_all, axis=0, ignore_index=True)
    pooled = pooled.dropna(subset=["date", "symbol"]).copy()

    # Ensure key columns exist
    fwd_col = f"fwd_{fwd_days}d_ret"
    need_cols = ["date", "symbol", "close", "ret_1d", fwd_col, "event_up", "event_dn"]
    missing = [c for c in need_cols if c not in pooled.columns]
    if missing:
        raise RuntimeError(f"Pooled missing required columns: {missing}")

    # Feature columns for mining
    blacklist = {
        "date",
        "symbol",
        "dt_utc",
        "t",
        "o",
        "h",
        "l",
        "c",
        "close",
        "v",
        "vw",
        "n",
        "ret_1d",
        fwd_col,
        "sigma_roll",
        "event_thr",
        "event_up",
        "event_dn",
        "impulse_up",
        "impulse_dn",
    }
    feature_cols = [c for c in pooled.columns if c not in blacklist and pooled[c].dtype != "O"]
    feature_cols = sorted(feature_cols)

    dates = sorted(pooled["date"].unique().tolist())
    base_up = float(pooled["event_up"].fillna(0).astype(int).mean())
    base_dn = float(pooled["event_dn"].fillna(0).astype(int).mean())
    print(f"[DATA] pooled rows={len(pooled)} dates={len(dates)} symbols={pooled['symbol'].nunique()}")
    print(f"[DATA] base_rate up={_fmt(100.0*base_up,4)}% dn={_fmt(100.0*base_dn,4)}%")

    folds = _make_folds(dates, folds_cfg)
    if not folds:
        raise RuntimeError("No folds could be constructed (not enough dates for train/test config).")

    # Per fold storage
    pass_rows: List[dict] = []
    rules_by_fold_sig: Dict[int, Dict[str, object]] = {}  # signature -> Rule
    latest_fold_id = max(fid for fid, *_ in folds)

    # ---------- LIB (per fold) ----------
    for (fid, tr0, tr1, te0, te1) in folds:
        tr_dates = set(dates[tr0:tr1])
        te_dates = set(dates[te0:te1])
        df_tr = pooled[pooled["date"].isin(tr_dates)].copy()
        df_te = pooled[pooled["date"].isin(te_dates)].copy()

        tr_from, tr_to = dates[tr0], dates[tr1 - 1]
        te_from, te_to = dates[te0], dates[te1 - 1]

        print("\n" + "=" * 80)
        print(f"[LIB/FOLD {fid}] train={tr_from}..{tr_to}  test={te_from}..{te_to}")
        print(f"[LIB/FOLD {fid}] train rows={len(df_tr)} test rows={len(df_te)}")

        # mine (both directions)
        rules, stats_tr, perm = mine_event_rules(df_tr, feature_cols, cfg_event)

        perm_p95 = perm.get("perm_topk_p95", float("nan"))
        perm_status = "ok" if np.isfinite(perm_p95) else "unavailable"

        print(f"[LIB/FOLD {fid}] mined_rules={len(rules)} perm_p95={_fmt(float(perm_p95),3) if np.isfinite(perm_p95) else 'NA'} status={perm_status}")

        # Build signature->rule map for this fold
        sig_map: Dict[str, object] = {}
        for r in rules:
            sig = getattr(stats_tr.get(getattr(r, "rule_id")), "signature", None)
            # safer: canonical signature lives in score_rule_event, but mine_event_rules already computed stats w/ signature
            if sig is None:
                # fallback: call score_rule_event quickly is expensive; skip
                continue
            # include direction in signature string already (direction|...)
            sig_map[str(sig)] = r
        rules_by_fold_sig[fid] = sig_map

        # OOS eval and pass set
        oos = evaluate_rules_oos(df_te, rules, cfg_event)
        if oos.empty:
            print(f"[LIB/FOLD {fid}] OOS eval=0 pass(lift>={oos_lift_min})=0")
            continue

        # filter
        oos = oos[oos["support"] >= min_support].copy()
        oos = oos[oos["lift"] >= oos_lift_min].copy()

        print(f"[LIB/FOLD {fid}] OOS eval={len(oos)} pass(lift>={oos_lift_min})={len(oos)}")

        for _, r in oos.iterrows():
            pass_rows.append(
                {
                    "fold_id": fid,
                    "direction": str(r["direction"]),
                    "signature": str(r["signature"]),
                    "support": int(r["support"]),
                    "lift": float(r["lift"]),
                    "median_signed": float(r["median_signed"]),
                    "es5_signed": float(r["es5_signed"]),
                }
            )

    print("\n" + "=" * 80)
    print("[GLOBAL] Rule Library (OOS-filtered across folds)")

    if not pass_rows:
        print("[GLOBAL] Fold pass counts: " + " | ".join([f"F{fid}: 0/0" for fid, *_ in folds]))
        print("[GLOBAL] library empty (after stability/recency filters).")
        if health.enabled:
            print("[HEALTH] enabled=1 but no rules survived GLOBAL selection -> nothing to filter.")
        return 0

    pass_df = pd.DataFrame.from_records(pass_rows)
    # fold pass counts for display
    parts = []
    for fid, *_ in folds:
        parts.append(f"F{fid}: {int((pass_df['fold_id']==fid).sum())}")
    print("[GLOBAL] Fold pass counts: " + " | ".join(parts))

    # Aggregate by signature+direction
    agg = (
        pass_df.groupby(["direction", "signature"], as_index=False)
        .agg(
            fold_count=("fold_id", "nunique"),
            avg_lift=("lift", "mean"),
            avg_es5=("es5_signed", "mean"),
            avg_support=("support", "mean"),
        )
        .copy()
    )

    # recency filters
    latest_pass = pass_df[pass_df["fold_id"] == latest_fold_id].copy()
    latest_sigs = set(latest_pass["signature"].tolist())

    if recency.latest_only:
        # Use only latest fold passes
        agg = agg[agg["signature"].isin(latest_sigs)].copy()
        print(f"[GLOBAL][RECENCY] LATEST_ONLY=1 -> using latest fold rules only (fold={latest_fold_id})")

    if recency.must_pass_latest:
        before = len(agg)
        agg = agg[agg["signature"].isin(latest_sigs)].copy()
        after = len(agg)
        print(f"[GLOBAL][RECENCY] MUST_PASS_LATEST=1 -> kept {after}/{before} rules")

    # stability requirement: default fold_count>=2 (Phase-1 baseline)
    min_fc = 2
    agg_stable = agg[agg["fold_count"] >= min_fc].copy()

    if recency.must_pass_latest and agg_stable.empty:
        # Explicit failsafe: allow min_fc=1 but keep only latest-fold passes.
        print("[GLOBAL][FAILSAFE] library empty under MUST_PASS_LATEST -> using latest-fold pass rules only (explicit)")
        agg_stable = agg.copy()
        min_fc = 1

    if agg_stable.empty:
        print("[GLOBAL] library empty (after stability/recency filters).")
        if health.enabled:
            print("[HEALTH] enabled=1 but no rules survived GLOBAL selection -> nothing to filter.")
        return 0

    # Weight
    weights: List[float] = []
    for _, r in agg_stable.iterrows():
        weights.append(_weight_from_metrics(float(r["avg_lift"]), int(r["fold_count"]), float(r["avg_es5"]), oos_lift_min))
    agg_stable["weight"] = weights

    # Split long/short library (kept)
    long_lib = agg_stable[agg_stable["direction"] == "long"].copy()
    short_lib = agg_stable[agg_stable["direction"] == "short"].copy()

    print(f"[GLOBAL] library_size={len(agg_stable)} (min_fc={min_fc})")
    print(f"[GLOBAL] long_kept={len(long_lib)} short_kept={len(short_lib)}")

    # Choose representative Rule objects: prefer latest fold, else any fold.
    def pick_rule(sig: str) -> Optional[object]:
        m = rules_by_fold_sig.get(latest_fold_id, {})
        if sig in m:
            return m[sig]
        for fid, sm in rules_by_fold_sig.items():
            if sig in sm:
                return sm[sig]
        return None

    # ---------- Rank per fold ----------
    for (fid, tr0, tr1, te0, te1) in folds:
        tr_dates = set(dates[tr0:tr1])
        te_dates = set(dates[te0:te1])
        df_tr = pooled[pooled["date"].isin(tr_dates)].copy()
        df_te = pooled[pooled["date"].isin(te_dates)].copy()

        te_from, te_to = dates[te0], dates[te1 - 1]
        print("\n" + "=" * 80)
        print(f"[RANK/FOLD {fid}] OOS window={te_from}..{te_to}  rows={len(df_te)}")

        # Build fold rules from global library (representative Rule objects)
        fold_long_rules: List[object] = []
        fold_short_rules: List[object] = []

        for _, r in long_lib.iterrows():
            sig = str(r["signature"])
            rr = pick_rule(sig)
            if rr is not None:
                fold_long_rules.append(rr)

        for _, r in short_lib.iterrows():
            sig = str(r["signature"])
            rr = pick_rule(sig)
            if rr is not None:
                fold_short_rules.append(rr)

        # HEALTH
        if health.enabled:
            print(
                f"[HEALTH] enabled=1 win={health.win} min_n={health.min_n} med_min={_fmt(health.med_min)} es5_min={_fmt(health.es5_min)}"
            )
            before_l = len(fold_long_rules)
            fold_long_rules, retired_l = _health_filter_rules(df_tr, fold_long_rules, "long", cfg_event, health)
            after_l = len(fold_long_rules)
            print(f"[HEALTH] long: before={before_l} after={after_l} retired={len(retired_l)}")
            for (sig, support, med, es5) in retired_l[: health.max_print]:
                if not sig:
                    continue
                print(f"[HEALTH] retired_long sig={sig} support={support} med={_fmt(med,4)} es5={_fmt(es5,4)}")

            before_s = len(fold_short_rules)
            fold_short_rules, retired_s = _health_filter_rules(df_tr, fold_short_rules, "short", cfg_event, health)
            after_s = len(fold_short_rules)
            print(f"[HEALTH] short: before={before_s} after={after_s} retired={len(retired_s)}")
            for (sig, support, med, es5) in retired_s[: health.max_print]:
                if not sig:
                    continue
                print(f"[HEALTH] retired_short sig={sig} support={support} med={_fmt(med,4)} es5={_fmt(es5,4)}")

        # Create weights lookup by signature
        w_long: Dict[str, float] = {str(r["signature"]): float(r["weight"]) for _, r in long_lib.iterrows()}
        w_short: Dict[str, float] = {str(r["signature"]): float(r["weight"]) for _, r in short_lib.iterrows()}

        # Score rows
        long_score = np.zeros(len(df_te), dtype=float)
        short_score = np.zeros(len(df_te), dtype=float)
        fires_long = 0
        fires_short = 0

        # We need rule signatures to lookup weight. Compute once via score_rule_event on small slice?
        # Faster: rebuild signature from rule.conds + direction is already canonical in event_mining.
        # The canonical signature is direction|sorted(feat{op}tag). It uses meta_key tags (q20/q50/q80).
        def rule_signature(r) -> str:
            direction = getattr(r, "direction")
            parts = []
            for feat, op, meta_key in getattr(r, "conds", []):
                tag = meta_key.split("__")[-1] if "__" in meta_key else meta_key
                parts.append(f"{feat}{op}{tag}")
            parts = sorted(parts)
            return str(direction) + "|" + "|".join(parts)

        for r in fold_long_rules:
            sig = rule_signature(r)
            w = w_long.get(sig, 0.0)
            if w <= 0:
                continue
            m = _apply_rule_mask(df_te, r).to_numpy(dtype=bool)
            if m.any():
                long_score[m] += w
                fires_long += int(np.sum(m))

        for r in fold_short_rules:
            sig = rule_signature(r)
            w = w_short.get(sig, 0.0)
            if w <= 0:
                continue
            m = _apply_rule_mask(df_te, r).to_numpy(dtype=bool)
            if m.any():
                short_score[m] += w
                fires_short += int(np.sum(m))

        net = long_score - short_score
        scored_rows = int(np.sum(net != 0.0))

        print(f"[RANK/FOLD {fid}] fold_rules kept: long={len(fold_long_rules)} short={len(fold_short_rules)}")
        print(
            f"[RANK/FOLD {fid}] scored_rows(nonzero)={scored_rows}/{len(df_te)} fires_long={fires_long} fires_short={fires_short}"
        )

        mean, med, ppos = _topk_perf(df_te, net, k_top, fwd_col=fwd_col)
        print(f"[RANK/FOLD {fid}] LONG top-{k_top} fwd_{fwd_days}d: mean={_fmt(mean)} med={_fmt(med)} p>0={_fmt(100.0*ppos,2)}%")

    print("\n[DONE] Ranker MVP completed.")
    return 0


if __name__ == "__main__":
    try:
        rc = main()
        _press_enter_exit(int(rc))
    except SystemExit:
        raise
    except Exception as e:
        print("\n[ERROR] Unhandled exception:")
        print(str(e))
        import traceback

        traceback.print_exc()
        _press_enter_exit(1)
