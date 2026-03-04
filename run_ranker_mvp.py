# scripts/run_ranker_mvp.py
# Double-click runnable. Never auto-closes (always waits for Enter).
#
# NOTE: This file is intentionally verbose in *CFG/DBG* (compact elsewhere),
# because most failures are configuration / import / feature-availability issues.

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
# Utils
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


def _load_universe(root: Path) -> List[str]:
    p = root / "data" / "universe_etf_first_30.txt"
    if not p.exists():
        raise RuntimeError(f"Missing universe file: {p}")
    txt = p.read_text(encoding="utf-8", errors="ignore")
    out: List[str] = []
    for line in txt.splitlines():
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
    max_folds: int = 6,
) -> List[Tuple[int, int, int, int, int]]:
    """Returns list of (fold_id, train_start_i, train_end_i, test_start_i, test_end_i)."""
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


def _weight_from_metrics(avg_lift: float, fold_count: int, avg_es5: float, oos_lift_min: float) -> float:
    edge = max(0.0, avg_lift - oos_lift_min)
    stability = 1.25 if fold_count >= 3 else 1.0
    tail_pen = max(0.0, (-avg_es5) - 0.10)  # penalize ES worse than -10%
    w = stability * edge / (1.0 + 5.0 * tail_pen)
    return float(w)


def _forward_return(df: pd.DataFrame, fwd_days: int) -> pd.Series:
    return df.groupby("symbol", sort=False)["close"].shift(-fwd_days) / df["close"] - 1.0


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


def _health_filter_rules(
    df_train: pd.DataFrame,
    rules: List[object],
    direction: str,
    cfg_event,
    cfg_health: HealthConfig,
) -> Tuple[List[object], List[Tuple[str, int, float, float]]]:
    """Evaluate each rule on LAST HEALTH_WIN train dates and retire if failing.

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
            retired.append(("<unknown>", 0, float("nan"), float("nan")))
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


# =========================
# Main
# =========================


def main() -> int:
    root = _add_src_to_syspath()

    # Imports AFTER sys.path injection to ensure we use local src/.
    from python_edge.data.ingest_aggs import load_aggs, to_daily_index
    from python_edge.features import build_features_daily as _bfd
    from python_edge.features.build_features_daily import FeatureConfig, build_features_daily
    from python_edge.rules import event_mining as _em
    from python_edge.rules.event_mining import EventMiningConfig, label_events, mine_event_rules, evaluate_rules_oos

    # --- Sanity: show module paths (prevents “imported old installed package” bugs) ---
    print(f"[DBG] python_edge.features.build_features_daily={Path(_bfd.__file__).resolve()}")
    print(f"[DBG] python_edge.rules.event_mining={Path(_em.__file__).resolve()}")

    start = _env_str("DATA_START", "")
    end = _env_str("DATA_END", "")
    if not start or not end:
        raise RuntimeError("Missing DATA_START/DATA_END. Example: 2023-01-01 / 2026-02-28")

    dataset_root = Path(_env_str("DATA_OUT_DIR", str(root / "data" / "raw" / "massive_dataset")))
    tickers = _load_universe(root)

    # --- Event mining config ---
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
        # creative mix knobs
        mix_random=_env_float("LAB_MIX_RANDOM", 0.55),
        mix_midband=_env_float("LAB_MIX_MIDBAND", 0.25),
        mix_prototype=_env_float("LAB_MIX_PROTO", 0.20),
        proto_top_days=_env_int("LAB_PROTO_TOP_DAYS", 180),
        proto_rules=_env_int("LAB_PROTO_RULES", 800),
        midband_prob=_env_float("LAB_MIDBAND_PROB", 0.65),
    )

    OOS_LIFT_MIN = _env_float("LAB_OOS_LIFT_MIN", 1.35)
    K = _env_int("RANK_K", 3)

    SHORT_GATE_MEAN_MIN = _env_float("SHORT_GATE_MEAN", 0.0)
    SHORT_GATE_PPOS_MIN = _env_float("SHORT_GATE_PPOS", 0.50)

    train_days = _env_int("WF_TRAIN_DAYS", 420)
    test_days = _env_int("WF_TEST_DAYS", 90)
    purge_days = _env_int("WF_PURGE_DAYS", 10)
    max_folds = _env_int("WF_MAX_FOLDS", 6)

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

    # ---- Load data ----
    panels: List[pd.DataFrame] = []
    for t in tickers:
        r = load_aggs(dataset_root=dataset_root, symbol=t, tf="1d", start=start, end=end, prefer_full=True)
        df = to_daily_index(r.df)
        if df.empty:
            continue

        d = df[["date", "c"]].copy()
        d = d.rename(columns={"c": "close"})
        d["symbol"] = t

        d = build_features_daily(d, FeatureConfig())
        d[f"fwd_{cfg.fwd_days}d_ret"] = _forward_return(d, cfg.fwd_days)
        panels.append(d)

    if not panels:
        raise RuntimeError("No data loaded.")

    all_df = pd.concat(panels, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    all_df = label_events(all_df, cfg)

    # Candidate features for mining.
    wanted = [
        "mom_1d__pct",
        "mom_3d__pct",
        "mom_5d__pct",
        "rv_10__pct",
        "atr_pct__pct",
        "ema_slow_slope__pct",
        "compression__pct",
        "mom_5d_minus_mom_1d__pct",
        "rel_rv_to_atr__pct",
        "is_high_mom_5d",
        "is_high_rv10",
    ]
    feature_cols = [c for c in wanted if c in all_df.columns]

    print(f"[DBG] feature_cols_n={len(feature_cols)}")
    if len(feature_cols) == 0:
        raise RuntimeError(
            "No feature columns found. This usually means you are NOT using the updated build_features_daily.py. "
            "Check [DBG] module path above and overwrite src/python_edge/features/build_features_daily.py from canvas."
        )

    need = feature_cols + [f"fwd_{cfg.fwd_days}d_ret", "event_up", "event_dn", "event_thr"]
    before = len(all_df)
    all_df = all_df.dropna(subset=need).copy()
    after = len(all_df)
    print(f"[DATA] pooled rows={after} (dropped={before - after}) dates={all_df['date'].nunique()} symbols={all_df['symbol'].nunique()}")
    print(f"[DATA] base_rate up={float(all_df['event_up'].mean()):.4%} dn={float(all_df['event_dn'].mean()):.4%}")

    dates = sorted(all_df["date"].unique().tolist())
    folds = _make_folds(dates, train_days=train_days, test_days=test_days, purge_days=purge_days, max_folds=max_folds)
    if not folds:
        raise RuntimeError("No folds produced. Need more date coverage.")

    # ---- Pass 1: GLOBAL library ----
    lib: Dict[str, Dict[str, object]] = {}

    def _acc(sig: str, direction: str, row: pd.Series, fold_id: int) -> None:
        d = lib.get(sig)
        if d is None:
            d = {"signature": sig, "direction": direction, "folds": set(), "n": 0,
                 "sum_support": 0.0, "sum_lift": 0.0, "sum_prec": 0.0, "sum_mean": 0.0,
                 "sum_median": 0.0, "sum_ppos": 0.0, "sum_es5": 0.0}
            lib[sig] = d
        d["folds"].add(fold_id)
        d["n"] += 1
        d["sum_support"] += float(row["support"])
        d["sum_lift"] += float(row["lift"])
        d["sum_prec"] += float(row["precision"])
        d["sum_mean"] += float(row["mean_signed"])
        d["sum_median"] += float(row["median_signed"])
        d["sum_ppos"] += float(row["p_pos_signed"])
        d["sum_es5"] += float(row["es5_signed"])

    per_fold_counts: List[Tuple[int, int, int]] = []
    latest_fold_pass: List[Dict[str, object]] = []
    latest_fold_id: Optional[int] = None

    for (fold_id, i0, i1, j0, j1) in folds:
        latest_fold_id = fold_id
        tr_dates = dates[i0:i1]
        te_dates = dates[j0:j1]
        df_tr = all_df[all_df["date"].isin(tr_dates)].copy()
        df_te = all_df[all_df["date"].isin(te_dates)].copy()

        print("\n" + "=" * 80)
        print(f"[LIB/FOLD {fold_id}] train={tr_dates[0]}..{tr_dates[-1]}  test={te_dates[0]}..{te_dates[-1]}")
        print(f"[LIB/FOLD {fold_id}] train rows={len(df_tr)} test rows={len(df_te)}")

        rules, _stats_map, perm = mine_event_rules(df_tr, feature_cols, cfg)
        perm_status = str(perm.get("perm_status", ""))
        perm_p95 = perm.get("perm_topk_p95", float("nan"))
        if perm_status == "ok":
            perm_str = _fmt(float(perm_p95), 3)
        else:
            perm_str = "NA"
        print(f"[LIB/FOLD {fold_id}] mined_rules={len(rules)} perm_p95={perm_str} status={perm_status}")

        if not rules:
            per_fold_counts.append((fold_id, 0, 0))
            continue

        oos = evaluate_rules_oos(df_te, rules, cfg)
        if oos.empty:
            per_fold_counts.append((fold_id, 0, 0))
            continue

        big_eval = int((oos["support"] >= cfg.min_support).sum())
        oos_f = oos[(oos["support"] >= cfg.min_support) & (oos["lift"].fillna(0.0) >= OOS_LIFT_MIN)].copy()
        passed = int(len(oos_f))
        per_fold_counts.append((fold_id, int(len(oos)), passed))
        print(f"[LIB/FOLD {fold_id}] OOS eval={len(oos)} pass(lift>={OOS_LIFT_MIN})={passed}")

        # store for recency/failsafe
        latest_fold_pass = []
        for _, r in oos_f.iterrows():
            latest_fold_pass.append(
                {
                    "signature": str(r["signature"]),
                    "direction": str(r["direction"]),
                    "fold_count": 1,
                    "n_obs": 1,
                    "avg_support": float(r["support"]),
                    "avg_lift": float(r["lift"]),
                    "avg_precision": float(r["precision"]),
                    "avg_mean": float(r["mean_signed"]),
                    "avg_median": float(r["median_signed"]),
                    "avg_p_pos": float(r["p_pos_signed"]),
                    "avg_es5": float(r["es5_signed"]),
                    "folds": str(fold_id),
                }
            )

        for _, r in oos_f.iterrows():
            _acc(str(r["signature"]), str(r["direction"]), r, fold_id)

    print("\n" + "=" * 80)
    print("[GLOBAL] Rule Library (OOS-filtered across folds)")
    print("[GLOBAL] Fold pass counts: " + " | ".join([f"F{fid}: {p}/{e}" for (fid, e, p) in per_fold_counts]))

    rows: List[Dict[str, object]] = []
    for sig, d in lib.items():
        n = int(d["n"])
        folds_set = d["folds"]
        fold_count = len(folds_set)
        rows.append(
            {
                "signature": sig,
                "direction": d["direction"],
                "fold_count": fold_count,
                "n_obs": n,
                "avg_support": float(d["sum_support"]) / n,
                "avg_lift": float(d["sum_lift"]) / n,
                "avg_precision": float(d["sum_prec"]) / n,
                "avg_mean": float(d["sum_mean"]) / n,
                "avg_median": float(d["sum_median"]) / n,
                "avg_p_pos": float(d["sum_ppos"]) / n,
                "avg_es5": float(d["sum_es5"]) / n,
                "folds": ",".join([str(x) for x in sorted(list(folds_set))]),
            }
        )

    df_lib = pd.DataFrame(rows)
    df_stable = df_lib[df_lib["fold_count"] >= 2].copy() if not df_lib.empty else pd.DataFrame()

    if LATEST_ONLY:
        df_use = pd.DataFrame(latest_fold_pass)
        print(f"[GLOBAL][RECENCY] LATEST_ONLY=1 -> using latest fold rules only (fold={latest_fold_id})")
    elif (df_stable.empty) and MUST_PASS_LATEST:
        df_use = pd.DataFrame(latest_fold_pass)
        print("[GLOBAL][FAILSAFE] library empty under MUST_PASS_LATEST -> using latest-fold pass rules only (explicit)")
    else:
        df_use = df_stable

    if df_use.empty:
        print("[GLOBAL] library empty (after stability/recency filters).")
        # HEALTH is enabled but there is nothing to apply it to.
        if cfg_health.enabled:
            print("[HEALTH] enabled=1 but no rules survived GLOBAL selection -> nothing to filter.")
        return 0

    # split and gates
    df_long = df_use[df_use["direction"] == "long"].copy()
    df_short = df_use[df_use["direction"] == "short"].copy()
    df_short_g = df_short[(df_short["avg_mean"] > SHORT_GATE_MEAN_MIN) & (df_short["avg_p_pos"] > SHORT_GATE_PPOS_MIN)].copy()

    # weights
    if not df_long.empty:
        df_long["weight"] = [
            _weight_from_metrics(float(r["avg_lift"]), int(r["fold_count"]), float(r["avg_es5"]), OOS_LIFT_MIN) for _, r in df_long.iterrows()
        ]
    else:
        df_long["weight"] = []

    if not df_short_g.empty:
        df_short_g["weight"] = [
            _weight_from_metrics(float(r["avg_lift"]), int(r["fold_count"]), float(r["avg_es5"]), OOS_LIFT_MIN) for _, r in df_short_g.iterrows()
        ]
    else:
        df_short_g["weight"] = []

    df_long = df_long.sort_values(["fold_count", "weight", "avg_lift"], ascending=[False, False, False]).reset_index(drop=True)
    df_short_g = df_short_g.sort_values(["fold_count", "weight", "avg_lift"], ascending=[False, False, False]).reset_index(drop=True)

    print(f"[GLOBAL] library_size={len(df_use)} (min_fc={int(df_use['fold_count'].min())})")
    print(f"[GLOBAL] long_kept={len(df_long)} short_kept(after payoff-gate)={len(df_short_g)}")

    long_sigs = df_long.head(TOPN_LONG)["signature"].tolist() if not df_long.empty else []
    short_sigs = df_short_g.head(TOPN_SHORT)["signature"].tolist() if not df_short_g.empty else []

    w_by_sig: Dict[str, float] = {}
    for _, r in df_long.head(TOPN_LONG).iterrows():
        w_by_sig[str(r["signature"])] = float(r["weight"])
    for _, r in df_short_g.head(TOPN_SHORT).iterrows():
        w_by_sig[str(r["signature"])] = float(r["weight"])

    print(f"[RUNTIME] use signatures: long={len(long_sigs)} short={len(short_sigs)}")

    # ---- Pass 2: fold-specific ranking (health applies here) ----
    from python_edge.rules.event_mining import _canonical_signature

    for (fold_id, i0, i1, j0, j1) in folds:
        tr_dates = dates[i0:i1]
        te_dates = dates[j0:j1]
        df_tr = all_df[all_df["date"].isin(tr_dates)].copy()
        df_te = all_df[all_df["date"].isin(te_dates)].copy()

        print("\n" + "=" * 80)
        print(f"[RANK/FOLD {fold_id}] OOS window={te_dates[0]}..{te_dates[-1]}  rows={len(df_te)}")
        if cfg_health.enabled:
            print(
                f"[HEALTH] enabled=1 win={cfg_health.win} min_n={cfg_health.min_n} "
                f"med_min={_fmt(cfg_health.med_min,4)} es5_min={_fmt(cfg_health.es5_min,4)}"
            )

        rules, _stats_map, _perm = mine_event_rules(df_tr, feature_cols, cfg)
        if (not rules) or df_te.empty:
            print(f"[RANK/FOLD {fold_id}] fold_rules kept: long=0 short=0")
            print(f"[RANK/FOLD {fold_id}] scored_rows(nonzero)=0/{len(df_te)} fires_long=0 fires_short=0")
            print(f"[RANK/FOLD {fold_id}] LONG top-{K} fwd_{cfg.fwd_days}d: mean=0.0000 med=0.0000 p>0=0.00%")
            continue

        fold_long_rules: List[object] = []
        fold_short_rules: List[object] = []
        for r in rules:
            sig = _canonical_signature(r.direction, r.conds)
            if r.direction == "long" and sig in long_sigs:
                fold_long_rules.append(r)
            if r.direction == "short" and sig in short_sigs:
                fold_short_rules.append(r)

        if cfg_health.enabled:
            before_l = len(fold_long_rules)
            before_s = len(fold_short_rules)
            fold_long_rules, retired_l = _health_filter_rules(df_tr, fold_long_rules, "long", cfg, cfg_health)
            fold_short_rules, retired_s = _health_filter_rules(df_tr, fold_short_rules, "short", cfg, cfg_health)
            print(f"[HEALTH] long: before={before_l} after={len(fold_long_rules)} retired={len(retired_l)}")
            for sig, sup, med, es5 in retired_l[: cfg_health.max_print]:
                print(f"[HEALTH] retired_long sig={sig} support={sup} med={_fmt(med,4)} es5={_fmt(es5,4)}")
            print(f"[HEALTH] short: before={before_s} after={len(fold_short_rules)} retired={len(retired_s)}")
            for sig, sup, med, es5 in retired_s[: cfg_health.max_print]:
                print(f"[HEALTH] retired_short sig={sig} support={sup} med={_fmt(med,4)} es5={_fmt(es5,4)}")

        print(f"[RANK/FOLD {fold_id}] fold_rules kept: long={len(fold_long_rules)} short={len(fold_short_rules)}")

        if (len(fold_long_rules) == 0) and (len(fold_short_rules) == 0):
            print(f"[RANK/FOLD {fold_id}] scored_rows(nonzero)=0/{len(df_te)} fires_long=0 fires_short=0")
            print(f"[RANK/FOLD {fold_id}] LONG top-{K} fwd_{cfg.fwd_days}d: mean=0.0000 med=0.0000 p>0=0.00%")
            continue

        long_score = np.zeros(len(df_te), dtype=float)
        short_score = np.zeros(len(df_te), dtype=float)

        # Score by weighted sum of fired rules
        for r in fold_long_rules:
            try:
                m = pd.Series(True, index=df_te.index)
                for feat, op, meta_key in r.conds:
                    thr = float(r.meta[meta_key])
                    if op == ">":
                        m &= (df_te[feat] > thr)
                    else:
                        m &= (df_te[feat] < thr)
                sig = _canonical_signature(r.direction, r.conds)
                w = float(w_by_sig.get(sig, 0.0))
                long_score[m.to_numpy()] += w
            except Exception:
                continue

        for r in fold_short_rules:
            try:
                m = pd.Series(True, index=df_te.index)
                for feat, op, meta_key in r.conds:
                    thr = float(r.meta[meta_key])
                    if op == ">":
                        m &= (df_te[feat] > thr)
                    else:
                        m &= (df_te[feat] < thr)
                sig = _canonical_signature(r.direction, r.conds)
                w = float(w_by_sig.get(sig, 0.0))
                short_score[m.to_numpy()] += w
            except Exception:
                continue

        fires_long = int(np.sum(long_score > 0))
        fires_short = int(np.sum(short_score > 0))
        nonzero = int(np.sum((long_score + short_score) > 0))
        print(f"[RANK/FOLD {fold_id}] scored_rows(nonzero)={nonzero}/{len(df_te)} fires_long={fires_long} fires_short={fires_short}")

        fwd_col = f"fwd_{cfg.fwd_days}d_ret"
        df_te = df_te.copy()
        df_te["_ls"] = long_score

        picks: List[float] = []
        for d in sorted(df_te["date"].unique().tolist()):
            day = df_te[df_te["date"] == d]
            if day.empty:
                continue
            top = day.sort_values("_ls", ascending=False).head(K)
            top = top[top["_ls"] > 0]
            if top.empty:
                continue
            picks.extend(top[fwd_col].to_numpy(dtype=float).tolist())

        if not picks:
            print(f"[RANK/FOLD {fold_id}] LONG top-{K} fwd_{cfg.fwd_days}d: mean=0.0000 med=0.0000 p>0=0.00%")
        else:
            arr = np.array(picks, dtype=float)
            mean = float(np.mean(arr))
            med = float(np.median(arr))
            ppos = float(np.mean(arr > 0))
            print(f"[RANK/FOLD {fold_id}] LONG top-{K} fwd_{cfg.fwd_days}d: mean={_fmt(mean,4)} med={_fmt(med,4)} p>0={_fmt_pct(ppos)}")

    print("\n[DONE] Ranker MVP completed.")
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except SystemExit:
        raise
    except Exception:
        print("\n[ERROR] Unhandled exception:")
        traceback.print_exc()
        rc = 1

    _press_enter_exit(int(rc))
