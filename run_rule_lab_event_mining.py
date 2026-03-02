from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def _press_enter_exit(code: int) -> None:
    try:
        print(f"\n[EXIT] code={code}")
        input("Press Enter to exit...")
    except Exception:
        pass
    raise SystemExit(code)


def _add_src_to_syspath() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if src.exists():
        sys.path.insert(0, str(src))


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


def _load_universe(root: Path) -> List[str]:
    p = root / "data" / "universe_etf_first_30.txt"
    txt = p.read_text(encoding="utf-8", errors="ignore")
    out = []
    for line in txt.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s.upper())
    return out


def _fmt_pct(x: float) -> str:
    if x != x:
        return "nan"
    return f"{x*100:.2f}%"


def _fmt(x: float, nd: int = 3) -> str:
    if x != x:
        return "nan"
    return f"{x:.{nd}f}"


def main() -> int:
    _add_src_to_syspath()

    from python_edge.data.ingest_aggs import load_aggs, to_daily_index
    from python_edge.features.build_features_daily import DailyFeatureConfig, build_daily_features
    from python_edge.rules.event_mining import EventMiningConfig, label_events, mine_event_rules, evaluate_rules_oos

    root = Path(__file__).resolve().parents[1]
    start = _env_str("DATA_START", "")
    end = _env_str("DATA_END", "")
    if not start or not end:
        raise RuntimeError("Missing DATA_START/DATA_END (must match FULL file names).")

    dataset_root = Path(_env_str("DATA_OUT_DIR", str(root / "data" / "raw" / "massive_dataset")))
    tickers = _load_universe(root)

    cfg = EventMiningConfig(
        fwd_days=5,
        k_sigma=_env_float("LAB_K_SIGMA", 1.5),
        max_rules_try=_env_int("LAB_MAX_RULES_TRY", 6000),
        max_rules_keep=_env_int("LAB_MAX_RULES_KEEP", 800),
        min_support=_env_int("LAB_MIN_SUPPORT", 200),
        min_event_hits=_env_int("LAB_MIN_EVENT_HITS", 30),
        max_conds=3,
        seed=7,
        perm_trials=_env_int("LAB_PERM_TRIALS", 50),
        perm_topk=_env_int("LAB_PERM_TOPK", 20),
        perm_gate_enabled=(_env_int("LAB_PERM_GATE", 1) == 1),
        perm_gate_margin=_env_float("LAB_PERM_GATE_MARGIN", 0.15),
    )

    OOS_LIFT_MIN = _env_float("LAB_OOS_LIFT_MIN", 1.35)  # your choice
    OOS_MIN_SUPPORT = cfg.min_support

    print(f"[CFG] vendor=massive dataset_root={dataset_root}")
    print(f"[CFG] universe={len(tickers)} start={start} end={end}")
    print(f"[CFG] event: fwd_days={cfg.fwd_days} k_sigma={cfg.k_sigma} sigma_lookback={cfg.sigma_lookback}")
    print(f"[CFG] rules: try={cfg.max_rules_try} keep={cfg.max_rules_keep} min_support={cfg.min_support} min_event_hits={cfg.min_event_hits}")
    print(f"[CFG] perm: trials={cfg.perm_trials} topk={cfg.perm_topk} gate={cfg.perm_gate_enabled} margin={cfg.perm_gate_margin}")
    print(f"[CFG] OOS filter: support>={OOS_MIN_SUPPORT} lift>={OOS_LIFT_MIN}")

    # build pooled panel
    panels = []
    for t in tickers:
        r = load_aggs(dataset_root=dataset_root, symbol=t, tf="1d", start=start, end=end, prefer_full=True)
        df = to_daily_index(r.df)
        if df.empty:
            print(f"[WARN] {t} empty 1d")
            continue
        df = df[["date", "o", "h", "l", "c", "v"]].copy()
        df["symbol"] = t
        df = build_daily_features(df, DailyFeatureConfig(fwd_days=cfg.fwd_days))
        panels.append(df)
    if not panels:
        raise RuntimeError("No data loaded.")

    all_df = pd.concat(panels, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    all_df = label_events(all_df, cfg)

    feature_cols = [
        "mom_1d", "mom_3d", "mom_5d", "mom_10d", "mom_20d",
        "ema_dist", "ema_fast_slope", "ema_slow_slope",
        "atr_pct", "rv_10", "compression",
    ]
    feature_cols = [c for c in feature_cols if c in all_df.columns]

    need = feature_cols + [f"fwd_{cfg.fwd_days}d_ret", "ret_1d", "event_up", "event_dn", "event_thr"]
    all_df = all_df.dropna(subset=need).copy()

    print(f"[DATA] pooled rows={len(all_df)} dates={all_df['date'].nunique()} symbols={all_df['symbol'].nunique()}")
    print(f"[DATA] base_rate up={float(all_df['event_up'].mean()):.4%} dn={float(all_df['event_dn'].mean()):.4%}")

    # WF folds
    dates = sorted(all_df["date"].unique().tolist())
    train_days = 420
    test_days = 90
    purge_days = 10
    step = test_days

    folds: List[Tuple[int, int, int, int, int]] = []
    t_end = train_days
    fid = 0
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
        if fid >= 6:
            break
    if not folds:
        raise RuntimeError("No folds produced. Need more date coverage.")

    # Global library accumulator keyed by signature
    # store per-fold OOS metrics
    lib: Dict[str, Dict[str, object]] = {}

    def _acc(sig: str, direction: str, row: pd.Series, fold_id: int) -> None:
        d = lib.get(sig)
        if d is None:
            d = {
                "signature": sig,
                "direction": direction,
                "folds": set(),
                "n": 0,
                "sum_support": 0.0,
                "sum_lift": 0.0,
                "sum_prec": 0.0,
                "sum_mean": 0.0,
                "sum_median": 0.0,
                "sum_ppos": 0.0,
                "sum_es5": 0.0,
            }
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

    per_fold_counts = []

    for (fold_id, i0, i1, j0, j1) in folds:
        tr_dates = dates[i0:i1]
        te_dates = dates[j0:j1]
        df_tr = all_df[all_df["date"].isin(tr_dates)].copy()
        df_te = all_df[all_df["date"].isin(te_dates)].copy()

        print("\n" + "=" * 80)
        print(f"[FOLD {fold_id}] train={tr_dates[0]}..{tr_dates[-1]}  test={te_dates[0]}..{te_dates[-1]}")
        print(f"[FOLD {fold_id}] train rows={len(df_tr)} test rows={len(df_te)}")

        rules, stats_map, perm = mine_event_rules(df_tr, feature_cols, cfg)
        print(f"[FOLD {fold_id}] mined_rules={len(rules)} perm_p95={_fmt(float(perm.get('perm_topk_p95', float('nan'))), 3)}")

        if not rules:
            print(f"[FOLD {fold_id}] no rules passed filters/gates.")
            per_fold_counts.append((fold_id, 0, 0))
            continue

        # OOS evaluate
        oos = evaluate_rules_oos(df_te, rules, cfg)
        if oos.empty:
            print(f"[FOLD {fold_id}] OOS: no evaluations.")
            per_fold_counts.append((fold_id, 0, 0))
            continue

        # Apply OOS filters (the ones we trust)
        oos_big = oos[(oos["support"] >= OOS_MIN_SUPPORT) & (oos["lift"].fillna(0.0) >= OOS_LIFT_MIN)].copy()
        n_eval_big = int((oos["support"] >= OOS_MIN_SUPPORT).sum())
        n_pass = int(len(oos_big))

        per_fold_counts.append((fold_id, n_eval_big, n_pass))
        print(f"[FOLD {fold_id}] OOS big-eval={n_eval_big}  OOS pass(lift>={OOS_LIFT_MIN})={n_pass}")

        # Print TOP OOS among passers (minimal)
        if not oos_big.empty:
            top = oos_big.sort_values(["lift", "precision", "support"], ascending=[False, False, False]).head(8)
            print(f"[FOLD {fold_id}] TOP OOS (filtered):")
            for _, r in top.iterrows():
                print(
                    f"  {r['rule_id']} {r['direction']:5s} supp={int(r['support']):4d} "
                    f"prec={float(r['precision']):.3%} lift={float(r['lift']):.2f} "
                    f"mean={_fmt(float(r['mean_signed']), 4)} med={_fmt(float(r['median_signed']), 4)} "
                    f"p>0={_fmt_pct(float(r['p_pos_signed']))} es5={_fmt(float(r['es5_signed']), 4)}"
                )

            # Accumulate library by signature
            for _, r in oos_big.iterrows():
                _acc(str(r["signature"]), str(r["direction"]), r, fold_id)

    # ===== Global Rule Library Report (console) =====
    print("\n" + "=" * 80)
    print("[GLOBAL] Rule Library (OOS-filtered across folds)")
    print(f"[GLOBAL] Fold pass counts: " + " | ".join([f"F{fid}: {npass}/{neval}" for (fid, neval, npass) in per_fold_counts]))

    if not lib:
        print("[GLOBAL] No rules passed OOS filters across folds.")
        return 0

    rows = []
    for sig, d in lib.items():
        n = int(d["n"])
        folds_set = d["folds"]
        fold_count = len(folds_set)
        rows.append({
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
        })

    df_lib = pd.DataFrame(rows)
    # Require multi-fold appearance (>=2) for "library"
    df_lib = df_lib[df_lib["fold_count"] >= 2].copy()
    if df_lib.empty:
        print("[GLOBAL] Rules exist but none appear in >=2 folds (after OOS filters).")
        return 0

    df_lib = df_lib.sort_values(["fold_count", "avg_lift", "avg_precision", "avg_support"], ascending=[False, False, False, False]).reset_index(drop=True)

    print(f"[GLOBAL] library_size={len(df_lib)} (fold_count>=2 & OOS filtered)")
    print("[GLOBAL] TOP 25 signatures:")
    topn = min(25, len(df_lib))
    for i in range(topn):
        r = df_lib.iloc[i]
        print(
            f"  #{i+1:02d} {r['direction']:5s} folds={int(r['fold_count'])} ({r['folds']}) "
            f"lift={_fmt(float(r['avg_lift']),2)} prec={float(r['avg_precision']):.3%} supp={int(r['avg_support']):4d} "
            f"mean={_fmt(float(r['avg_mean']),4)} med={_fmt(float(r['avg_median']),4)} "
            f"p>0={_fmt_pct(float(r['avg_p_pos']))} es5={_fmt(float(r['avg_es5']),4)}"
        )
        # print the condition string itself (signature) on next line to keep it readable
        print(f"       sig={r['signature']}")

    print("\n[DONE] Rule Lab completed.")
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 1
    _press_enter_exit(int(rc))