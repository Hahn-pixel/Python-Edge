# scripts/run_ranker_mvp.py
# Double-click runnable. Never auto-closes (always waits for Enter).
#
# Pipeline: load massive 1D -> build features -> label events -> WF mine rules -> OOS filter ->
# GLOBAL library (stability/recency) -> HEALTH runtime filter -> rank -> evaluate top-K.

from __future__ import annotations

import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# =========================
# CLI / env helpers
# =========================

def _press_enter_exit(code: int) -> None:
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


def _fmt(x: float, nd: int = 4) -> str:
    if x != x:
        return "nan"
    return f"{x:.{nd}f}"


def _fmt_pct(x: float) -> str:
    if x != x:
        return "nan"
    return f"{x * 100:.2f}%"


# =========================
# Core utilities
# =========================

def _load_universe(root: Path) -> List[str]:
    p = root / "data" / "universe_etf_first_30.txt"
    if not p.exists():
        raise RuntimeError(f"Missing universe file: {p}")
    out: List[str] = []
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s.upper())
    return out


def _make_folds(
    dates: Sequence[str],
    train_days: int,
    test_days: int,
    purge_days: int,
    max_folds: int,
) -> List[Tuple[int, int, int, int, int]]:
    """(fold_id, train_start_i, train_end_i, test_start_i, test_end_i)"""
    folds: List[Tuple[int, int, int, int, int]] = []
    t_end = train_days
    fid = 0
    step = test_days
    while True:
        train_end = t_end
        test_start = train_end + purge_days
        test_end = test_start + test_days
        train_start = max(0, train_end - train_days)
        if test_end > len(dates):
            break
        fid += 1
        folds.append((fid, train_start, train_end, test_start, test_end))
        t_end += step
        if fid >= max_folds:
            break
    return folds


def _forward_return(df: pd.DataFrame, fwd_days: int) -> pd.Series:
    return df.groupby("symbol", sort=False)["close"].shift(-fwd_days) / df["close"] - 1.0


def _weight_from_metrics(avg_lift: float, fold_count: int, avg_es5: float, oos_lift_min: float) -> float:
    # scale to be stable and non-tiny
    # edge: lift above threshold; stability: fold count bonus; tail: penalize ugly ES
    edge = max(0.0, avg_lift - oos_lift_min)
    stability = 1.35 if fold_count >= 3 else 1.0
    tail_pen = max(0.0, (-avg_es5) - 0.10)  # penalize ES worse than -10%
    w = stability * edge / (1.0 + 5.0 * tail_pen)
    # fixed gain so net scores aren’t microscopic
    return float(50.0 * w)


# =========================
# HEALTH (Phase 2 MVP)
# =========================

@dataclass(frozen=True)
class HealthConfig:
    enabled: bool = False
    win: int = 60
    min_n: int = 50
    med_min: float = 0.0
    es5_min: float = -0.08
    max_print: int = 5


def _unwrap_rule(obj: object) -> object:
    # safety: accept Rule, tuple(rule, stats), dict with "rule", etc.
    if hasattr(obj, "conds"):
        return obj
    if isinstance(obj, tuple) and obj:
        return obj[0]
    if isinstance(obj, dict) and "rule" in obj:
        return obj["rule"]
    return obj


def _health_filter_rules(
    df_train: pd.DataFrame,
    rules: List[object],
    direction: str,
    cfg_event,
    cfg_health: HealthConfig,
) -> Tuple[List[object], List[Tuple[str, int, float, float]]]:
    """Evaluate each rule on LAST health.win train dates.

    Retire if failing any threshold.
    Returns (kept_rules, retired_info(signature,support,median,es5)).
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

    for obj in rules:
        r = _unwrap_rule(obj)
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
            kept.append(obj)
        else:
            retired.append((str(st.signature), support, med, es5))

    return kept, retired


# =========================
# Mining wrapper
# =========================

def _mine_dir(df_tr: pd.DataFrame, feature_cols: List[str], cfg_event, direction: str):
    """Single canonical call path.

    IMPORTANT: mine_event_rules is direction-aware (see src/python_edge/rules/event_mining.py).
    """
    from python_edge.rules.event_mining import mine_event_rules

    rules, stats, perm = mine_event_rules(df_tr, feature_cols=feature_cols, cfg=cfg_event, direction=direction)
    return rules, stats, perm


# =========================
# Main
# =========================

def main() -> int:
    root = _add_src_to_syspath()

    # Imports AFTER sys.path injection
    from python_edge.data.ingest_aggs import load_aggs, to_daily_index
    from python_edge.features.build_features_daily import DailyFeatureConfig, build_daily_features
    from python_edge.rules.event_mining import EventMiningConfig, evaluate_rules_oos, label_events

    # show module paths (prevents "imported old installed package" bugs)
    import python_edge.features.build_features_daily as _bfd
    import python_edge.rules.event_mining as _em

    print(f"[DBG] python_edge.features.build_features_daily={Path(_bfd.__file__).resolve()}")
    print(f"[DBG] python_edge.rules.event_mining={Path(_em.__file__).resolve()}")

    start = _env_str("DATA_START", "")
    end = _env_str("DATA_END", "")
    if not start or not end:
        raise RuntimeError("Missing DATA_START/DATA_END. Example: 2023-01-01 / 2026-02-28")

    dataset_root = Path(_env_str("DATA_OUT_DIR", str(root / "data" / "raw" / "massive_dataset")))
    tickers = _load_universe(root)

    # ---- configs ----
    cfg = EventMiningConfig(
        fwd_days=_env_int("LAB_FWD_DAYS", 5),
        k_sigma=_env_float("LAB_K_SIGMA", 1.5),
        sigma_lookback=_env_int("LAB_SIGMA_LOOKBACK", 60),
        max_rules_try=_env_int("LAB_MAX_RULES_TRY", 6000),
        max_rules_keep=_env_int("LAB_MAX_RULES_KEEP", 800),
        min_support=_env_int("LAB_MIN_SUPPORT", 200),
        min_event_hits=_env_int("LAB_MIN_EVENT_HITS", 30),
        max_conds=_env_int("LAB_MAX_CONDS", 3),
        seed=_env_int("LAB_SEED", 7),
        perm_trials=_env_int("LAB_PERM_TRIALS", 50),
        perm_topk=_env_int("LAB_PERM_TOPK", 20),
        perm_gate_enabled=(_env_int("LAB_PERM_GATE", 1) == 1),
        perm_gate_margin=_env_float("LAB_PERM_GATE_MARGIN", 0.15),
    )

    OOS_LIFT_MIN = _env_float("LAB_OOS_LIFT_MIN", 1.35)
    K = _env_int("RANK_K", 3)

    SHORT_GATE_MEAN_MIN = _env_float("SHORT_GATE_MEAN", 0.0)
    SHORT_GATE_PPOS_MIN = _env_float("SHORT_GATE_PPOS", 0.50)

    train_days = _env_int("WF_TRAIN_DAYS", 420)
    test_days = _env_int("WF_TEST_DAYS", 90)
    purge_days = _env_int("WF_PURGE_DAYS", 10)
    max_folds = _env_int("WF_MAX_FOLDS", 3)

    TOPN_LONG = _env_int("RUNTIME_TOPN_LONG", 40)
    TOPN_SHORT = _env_int("RUNTIME_TOPN_SHORT", 30)

    cfg_health = HealthConfig(
        enabled=(_env_int("HEALTH", 0) == 1),
        win=_env_int("HEALTH_WIN", 60),
        min_n=_env_int("HEALTH_MIN_N", 50),
        med_min=_env_float("HEALTH_MED_MIN", 0.0),
        es5_min=_env_float("HEALTH_ES5_MIN", -0.08),
        max_print=_env_int("HEALTH_MAX_PRINT", 5),
    )

    MUST_PASS_LATEST = (_env_int("MUST_PASS_LATEST", 0) == 1)
    LATEST_ONLY = (_env_int("LATEST_ONLY", 0) == 1)

    # ---- prints ----
    print(f"[CFG] vendor=massive dataset_root={dataset_root}")
    print(f"[CFG] universe={len(tickers)} start={start} end={end}")
    print(f"[CFG] event: fwd_days={cfg.fwd_days} k_sigma={cfg.k_sigma} sigma_lookback={cfg.sigma_lookback}")
    print(f"[CFG] rules: try={cfg.max_rules_try} keep={cfg.max_rules_keep} min_support={cfg.min_support} min_event_hits={cfg.min_event_hits}")
    print(f"[CFG] perm: trials={cfg.perm_trials} topk={cfg.perm_topk} gate={int(cfg.perm_gate_enabled)} margin={cfg.perm_gate_margin}")
    print(f"[CFG] OOS filter: support>={cfg.min_support} lift>={OOS_LIFT_MIN}")
    print(f"[CFG] short payoff-gate: mean>{SHORT_GATE_MEAN_MIN} p>0>{SHORT_GATE_PPOS_MIN}")
    print(f"[CFG] rank: K={K}")
    print(
        f"[CFG] health: enabled={int(cfg_health.enabled)} win={cfg_health.win} min_n={cfg_health.min_n} "
        f"med_min={_fmt(cfg_health.med_min,4)} es5_min={_fmt(cfg_health.es5_min,4)}"
    )
    print(f"[CFG] recency: MUST_PASS_LATEST={int(MUST_PASS_LATEST)} LATEST_ONLY={int(LATEST_ONLY)}")

    # ---- load data ----
    panels: List[pd.DataFrame] = []

    for t in tickers:
        r = load_aggs(dataset_root=dataset_root, symbol=t, tf="1d", start=start, end=end, prefer_full=True)
        df = to_daily_index(r.df)
        if df.empty:
            continue

        d = df[["date", "o", "h", "l", "c", "v"]].copy()
        d = d.rename(columns={"c": "c"})
        d["symbol"] = t
        feat = build_daily_features(d.rename(columns={"c": "c"}), DailyFeatureConfig(fwd_days=cfg.fwd_days))

        # unify expected col names
        feat = feat.rename(columns={"c": "close"})

        panels.append(feat[["date", "symbol", "close", "ret_1d", f"fwd_{cfg.fwd_days}d_ret", "mom_1d", "mom_3d", "mom_5d", "rv_10", "atr_pct", "ema_slow_slope", "compression"]])

    if not panels:
        raise RuntimeError("No data loaded.")

    all_df = pd.concat(panels, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    all_df = label_events(all_df, cfg)

    feature_cols = [
        "mom_1d",
        "mom_3d",
        "mom_5d",
        "rv_10",
        "atr_pct",
        "ema_slow_slope",
        "compression",
    ]

    have = [c for c in feature_cols if c in all_df.columns]
    if len(have) != len(feature_cols):
        missing = [c for c in feature_cols if c not in all_df.columns]
        raise RuntimeError(f"Missing feature columns: {missing}. Check build_features_daily.py module path above.")

    need = feature_cols + [f"fwd_{cfg.fwd_days}d_ret", "event_up", "event_dn"]
    before = len(all_df)
    all_df = all_df.dropna(subset=need).copy()
    after = len(all_df)

    print(f"[DATA] pooled rows={after} (dropped={before - after}) dates={all_df['date'].nunique()} symbols={all_df['symbol'].nunique()}")
    print(f"[DATA] base_rate up={float(all_df['event_up'].mean()):.4%} dn={float(all_df['event_dn'].mean()):.4%}")

    dates = sorted(all_df["date"].unique().tolist())
    folds = _make_folds(dates, train_days=train_days, test_days=test_days, purge_days=purge_days, max_folds=max_folds)
    if not folds:
        raise RuntimeError("No folds produced. Check WF_* env.")

    # ---- library building ----
    fold_rule_rows: List[pd.DataFrame] = []
    fold_rules_long: Dict[int, List[object]] = {}
    fold_rules_short: Dict[int, List[object]] = {}

    last_fold_id = folds[-1][0]

    for (fid, tr_s, tr_e, te_s, te_e) in folds:
        tr_dates = dates[tr_s:tr_e]
        te_dates = dates[te_s:te_e]

        df_tr = all_df[all_df["date"].isin(tr_dates)].copy()
        df_te = all_df[all_df["date"].isin(te_dates)].copy()

        print("\n" + "=" * 80)
        print(f"[LIB/FOLD {fid}] train={tr_dates[0]}..{tr_dates[-1]}  test={te_dates[0]}..{te_dates[-1]}")
        print(f"[LIB/FOLD {fid}] train rows={len(df_tr)} test rows={len(df_te)}")

        long_rules, long_stats, perm_l = _mine_dir(df_tr, feature_cols, cfg, direction="long")
        short_rules, short_stats, perm_s = _mine_dir(df_tr, feature_cols, cfg, direction="short")

        fold_rules_long[fid] = long_rules
        fold_rules_short[fid] = short_rules

        perm_p95 = float(np.nanmax([perm_l.get("perm_topk_p95", np.nan), perm_s.get("perm_topk_p95", np.nan)]))
        perm_status = "ok" if np.isfinite(perm_p95) else "unavailable"
        print(f"[LIB/FOLD {fid}] mined_rules={len(long_rules) + len(short_rules)} perm_p95={_fmt(perm_p95,3) if np.isfinite(perm_p95) else 'NA'} status={perm_status}")

        # OOS evaluation for selection
        oos_long = evaluate_rules_oos(df_te, long_rules, cfg)
        oos_short = evaluate_rules_oos(df_te, short_rules, cfg)
        oos = pd.concat([oos_long, oos_short], ignore_index=True) if (not oos_long.empty or not oos_short.empty) else pd.DataFrame()

        if oos.empty:
            print(f"[LIB/FOLD {fid}] OOS eval=0 pass(lift>={OOS_LIFT_MIN})=0")
            continue

        oos["fold_id"] = fid
        oos = oos[oos["support"] >= cfg.min_support].copy()
        oos = oos[oos["lift"] >= OOS_LIFT_MIN].copy()

        print(f"[LIB/FOLD {fid}] OOS eval={len(oos_long) + len(oos_short)} pass(lift>={OOS_LIFT_MIN})={len(oos)}")
        fold_rule_rows.append(oos)

    print("\n" + "=" * 80)
    print("[GLOBAL] Rule Library (OOS-filtered across folds)")

    if not fold_rule_rows:
        print("[GLOBAL] no OOS-passing rules in any fold.")
        print("[GLOBAL] library empty (after stability/recency filters).")
        if cfg_health.enabled:
            print("[HEALTH] enabled=1 but no rules survived GLOBAL selection -> nothing to filter.")
        return 0

    lib = pd.concat(fold_rule_rows, ignore_index=True)

    # fold_count / aggregates by signature
    agg = (
        lib.groupby(["direction", "signature"], as_index=False)
        .agg(
            fold_count=("fold_id", "nunique"),
            avg_lift=("lift", "mean"),
            avg_es5=("es5_signed", "mean"),
            avg_med=("median_signed", "mean"),
            avg_support=("support", "mean"),
        )
        .copy()
    )

    # stability filter
    min_fc = 2
    if MUST_PASS_LATEST:
        min_fc = 2
    if LATEST_ONLY:
        min_fc = 1

    if LATEST_ONLY:
        lib_latest = lib[lib["fold_id"] == last_fold_id].copy()
        if lib_latest.empty:
            print(f"[GLOBAL][RECENCY] LATEST_ONLY=1 -> latest fold has no passing rules (fold={last_fold_id})")
            agg = agg.iloc[0:0]
        else:
            agg = (
                lib_latest.groupby(["direction", "signature"], as_index=False)
                .agg(
                    fold_count=("fold_id", "nunique"),
                    avg_lift=("lift", "mean"),
                    avg_es5=("es5_signed", "mean"),
                    avg_med=("median_signed", "mean"),
                    avg_support=("support", "mean"),
                )
                .copy()
            )
            print(f"[GLOBAL][RECENCY] LATEST_ONLY=1 -> using latest fold rules only (fold={last_fold_id})")

    agg = agg[agg["fold_count"] >= min_fc].copy()

    if agg.empty and MUST_PASS_LATEST and (not LATEST_ONLY):
        # explicit fail-safe
        lib_latest = lib[lib["fold_id"] == last_fold_id].copy()
        if not lib_latest.empty:
            print("[GLOBAL][FAILSAFE] library empty under MUST_PASS_LATEST -> using latest-fold pass rules only (explicit)")
            agg = (
                lib_latest.groupby(["direction", "signature"], as_index=False)
                .agg(
                    fold_count=("fold_id", "nunique"),
                    avg_lift=("lift", "mean"),
                    avg_es5=("es5_signed", "mean"),
                    avg_med=("median_signed", "mean"),
                    avg_support=("support", "mean"),
                )
                .copy()
            )
            min_fc = 1

    if agg.empty:
        print("[GLOBAL] library empty (after stability/recency filters).")
        if cfg_health.enabled:
            print("[HEALTH] enabled=1 but no rules survived GLOBAL selection -> nothing to filter.")
        return 0

    # weights
    weights: List[float] = []
    for _, r in agg.iterrows():
        weights.append(_weight_from_metrics(float(r["avg_lift"]), int(r["fold_count"]), float(r["avg_es5"]), OOS_LIFT_MIN))
    agg["weight"] = weights

    # map signature -> weight
    w_map: Dict[Tuple[str, str], float] = {}
    for _, r in agg.iterrows():
        w_map[(str(r["direction"]), str(r["signature"]))] = float(r["weight"])

    # build runtime rule sets by fold
    def _select_fold_rules(fid: int, direction: str) -> List[object]:
        rules = fold_rules_long[fid] if direction == "long" else fold_rules_short[fid]
        out: List[object] = []
        for rr in rules:
            sig = getattr(rr, "meta", {}).get("signature") if hasattr(rr, "meta") else None
            if sig is None and hasattr(rr, "conds"):
                # fallback: compute with helper in event_mining via scoring on 1 row (cheap)
                try:
                    from python_edge.rules.event_mining import score_rule_event
                    tmp = all_df.iloc[0:1]
                    event_col = "event_up" if direction == "long" else "event_dn"
                    st = score_rule_event(tmp, rr, event_col=event_col, fwd_col=f"fwd_{cfg.fwd_days}d_ret")
                    sig = st.signature if st else None
                except Exception:
                    sig = None
            if sig is None:
                continue
            if (direction, str(sig)) in w_map:
                out.append(rr)
        return out

    print(f"[GLOBAL] library_size={len(agg)} (min_fc={min_fc})")
    print(f"[GLOBAL] long_kept={(agg[agg['direction']=='long']).shape[0]} short_kept(after payoff-gate)={(agg[agg['direction']=='short']).shape[0]}")

    # ---- Rank / Sim (hold-based proxy): top-K each day by score ----
    for (fid, tr_s, tr_e, te_s, te_e) in folds:
        te_dates = dates[te_s:te_e]
        df_te = all_df[all_df["date"].isin(te_dates)].copy().reset_index(drop=True)
        df_tr = all_df[all_df["date"].isin(dates[tr_s:tr_e])].copy()

        fold_long = _select_fold_rules(fid, "long")
        fold_short = _select_fold_rules(fid, "short")

        # payoff gate for shorts
        if fold_short:
            from python_edge.rules.event_mining import score_rule_event
            fwd_col = f"fwd_{cfg.fwd_days}d_ret"
            kept_s: List[object] = []
            for r in fold_short:
                st = score_rule_event(df_tr, r, event_col="event_dn", fwd_col=fwd_col)
                if st is None:
                    continue
                if (st.mean_signed > SHORT_GATE_MEAN_MIN) and (st.p_pos_signed > SHORT_GATE_PPOS_MIN):
                    kept_s.append(r)
            fold_short = kept_s

        # HEALTH filter
        print("\n" + "=" * 80)
        print(f"[RANK/FOLD {fid}] OOS window={te_dates[0]}..{te_dates[-1]}  rows={len(df_te)}")
        if cfg_health.enabled:
            print(
                f"[HEALTH] enabled=1 win={cfg_health.win} min_n={cfg_health.min_n} "
                f"med_min={_fmt(cfg_health.med_min,4)} es5_min={_fmt(cfg_health.es5_min,4)}"
            )

        before_l = len(fold_long)
        before_s = len(fold_short)

        fold_long, retired_l = _health_filter_rules(df_tr, fold_long, direction="long", cfg_event=cfg, cfg_health=cfg_health)
        fold_short, retired_s = _health_filter_rules(df_tr, fold_short, direction="short", cfg_event=cfg, cfg_health=cfg_health)

        if cfg_health.enabled:
            print(f"[HEALTH] long: before={before_l} after={len(fold_long)} retired={len(retired_l)}")
            for i, (sig, sup, med, es5) in enumerate(retired_l[: cfg_health.max_print]):
                if not sig:
                    continue
                print(f"[HEALTH] retired_long sig={sig} support={sup} med={_fmt(med,4)} es5={_fmt(es5,4)}")
            print(f"[HEALTH] short: before={before_s} after={len(fold_short)} retired={len(retired_s)}")
            for i, (sig, sup, med, es5) in enumerate(retired_s[: cfg_health.max_print]):
                if not sig:
                    continue
                print(f"[HEALTH] retired_short sig={sig} support={sup} med={_fmt(med,4)} es5={_fmt(es5,4)}")

        # runtime caps
        fold_long = fold_long[:TOPN_LONG]
        fold_short = fold_short[:TOPN_SHORT]

        print(f"[RANK/FOLD {fid}] fold_rules kept: long={len(fold_long)} short={len(fold_short)}")

        if (not fold_long) and (not fold_short):
            print(f"[RANK/FOLD {fid}] scored_rows(nonzero)=0/{len(df_te)} fires_long=0 fires_short=0")
            print(f"[RANK/FOLD {fid}] LONG top-{K} fwd_{cfg.fwd_days}d: mean=0.0000 med=0.0000 p>0=0.00%")
            continue

        # scoring: sum weights per firing rule
        from python_edge.rules.event_mining import score_rule_event, _apply_rule_mask  # type: ignore

        long_score = np.zeros(len(df_te), dtype=float)
        short_score = np.zeros(len(df_te), dtype=float)
        fires_l = 0
        fires_s = 0

        # precompute signature via score_rule_event on a tiny slice (cheap, stable)
        fwd_col = f"fwd_{cfg.fwd_days}d_ret"

        for r in fold_long:
            st = score_rule_event(df_tr.iloc[0:1], r, event_col="event_up", fwd_col=fwd_col)
            if st is None:
                continue
            w = w_map.get(("long", str(st.signature)), 0.0)
            if w == 0.0:
                continue
            m = _apply_rule_mask(df_te, r).to_numpy(dtype=bool)
            fires_l += int(np.sum(m))
            long_score[m] += w

        for r in fold_short:
            st = score_rule_event(df_tr.iloc[0:1], r, event_col="event_dn", fwd_col=fwd_col)
            if st is None:
                continue
            w = w_map.get(("short", str(st.signature)), 0.0)
            if w == 0.0:
                continue
            m = _apply_rule_mask(df_te, r).to_numpy(dtype=bool)
            fires_s += int(np.sum(m))
            short_score[m] += w

        net = long_score - short_score
        nonzero = int(np.sum(net != 0.0))
        print(f"[RANK/FOLD {fid}] scored_rows(nonzero)={nonzero}/{len(df_te)} fires_long={fires_l} fires_short={fires_s}")

        # daily top-K evaluation on fwd returns
        # (MVP: choose top-K by net each date and average their forward returns)
        daily = []
        for d, g in df_te.groupby("date", sort=False):
            idx = g.index.to_numpy()
            scores = net[idx]
            if scores.size == 0:
                continue
            top_idx = idx[np.argsort(scores)[::-1][:K]]
            rets = df_te.loc[top_idx, fwd_col].to_numpy(dtype=float)
            rets = rets[np.isfinite(rets)]
            if rets.size:
                daily.append(float(np.mean(rets)))

        if not daily:
            print(f"[RANK/FOLD {fid}] LONG top-{K} fwd_{cfg.fwd_days}d: mean=0.0000 med=0.0000 p>0=0.00%")
            continue

        arr = np.array(daily, dtype=float)
        mean = float(np.mean(arr))
        med = float(np.median(arr))
        ppos = float(np.mean(arr > 0.0))
        print(f"[RANK/FOLD {fid}] LONG top-{K} fwd_{cfg.fwd_days}d: mean={_fmt(mean,4)} med={_fmt(med,4)} p>0={_fmt_pct(ppos)}")

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
        traceback.print_exc()
        _press_enter_exit(1)
