# scripts/run_ranker_mvp.py
# Double-click runnable. Never auto-closes.
#
# Compatible with python_edge.rules.event_mining having:
# - label_events(df, cfg)
# - mine_event_rules(df_train, feature_cols, cfg) -> (rules, stats, perm_dict)
# - evaluate_rules_oos(df_oos, rules, cfg) -> DataFrame
# - score_rule_event(df, rule, event_col, fwd_col)
# - _apply_rule_mask(df, rule) -> Series[bool]

from __future__ import annotations

import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# =========================
# UX: never auto-close
# =========================
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


# =========================
# env helpers
# =========================
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
    try:
        if x != x:
            return "nan"
    except Exception:
        return "nan"
    return f"{x:.{nd}f}"


def _fmt_pct(x: float) -> str:
    try:
        if x != x:
            return "nan"
    except Exception:
        return "nan"
    return f"{x * 100:.2f}%"


# =========================
# universe / folds
# =========================
def _load_universe(root: Path) -> List[str]:
    p = root / "data" / "universe_etf_first_30.txt"
    txt = p.read_text(encoding="utf-8", errors="ignore")
    out: List[str] = []
    for line in txt.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s.upper())
    return out


def _make_folds(
    dates: List[str],
    *,
    train_days: int,
    test_days: int,
    purge_days: int,
    max_folds: int,
) -> List[Tuple[int, int, int, int, int]]:
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


# =========================
# rules: signature + weights
# =========================
def _rule_signature(rule: Any) -> str:
    sig = getattr(rule, "signature", None)
    if isinstance(sig, str) and sig:
        return sig

    # EventMining Rule in your file has 'direction' and 'conds'
    direction = getattr(rule, "direction", "unk")
    conds = getattr(rule, "conds", [])
    parts: List[str] = []
    try:
        for feat, op, meta_key in conds:
            tag = meta_key.split("__")[-1] if "__" in meta_key else meta_key
            parts.append(f"{feat}{op}{tag}")
        parts.sort()
        return str(direction) + "|" + "|".join(parts)
    except Exception:
        return str(rule)


def _weight_from_metrics(avg_lift: float, fold_count: int, avg_es5: float, oos_lift_min: float) -> float:
    edge = max(0.0, avg_lift - oos_lift_min)
    stability = 1.25 if fold_count >= 3 else 1.0
    tail_pen = max(0.0, (-avg_es5) - 0.10)
    w = stability * edge / (1.0 + 5.0 * tail_pen)
    return float(w)


def _rescale_weights_to_target(df_rules: pd.DataFrame, target_p95_weight: float) -> pd.DataFrame:
    """
    Deterministic scaling for stability:
    scale weights so p95(weight) ~= target_p95_weight.
    Disable by target_p95_weight<=0.
    """
    df = df_rules.copy()
    if target_p95_weight <= 0 or "weight" not in df.columns or df.empty:
        return df
    w = df["weight"].astype(float).to_numpy()
    w = w[np.isfinite(w)]
    if w.size == 0:
        return df
    p95 = float(np.quantile(w, 0.95))
    if p95 <= 0:
        return df
    scale = float(target_p95_weight) / p95
    df["weight"] = df["weight"].astype(float) * scale
    df["weight_scale"] = float(scale)
    return df


# =========================
# Phase 2: health filter
# =========================
@dataclass(frozen=True)
class RetiredRule:
    signature: str
    support: int
    median_signed: float
    es5_signed: float


def _health_filter_rules(
    df_tr: pd.DataFrame,
    rules_in: Sequence[Any],
    *,
    direction: str,
    cfg: Any,
    health_win: int,
    health_min_n: int,
    health_med_min: float,
    health_es5_min: float,
) -> Tuple[List[Any], List[RetiredRule]]:
    if not rules_in:
        return list(rules_in), []

    if "date" not in df_tr.columns:
        return list(rules_in), []

    dts = sorted(df_tr["date"].unique().tolist())
    if not dts:
        return list(rules_in), []

    win = max(1, int(health_win))
    df_h = df_tr[df_tr["date"].isin(dts[-win:])].copy()

    fwd_col = f"fwd_{cfg.fwd_days}d_ret"
    event_col = "event_up" if direction == "long" else "event_dn"

    try:
        from python_edge.rules.event_mining import score_rule_event  # type: ignore
    except Exception:
        return list(rules_in), []

    kept: List[Any] = []
    retired: List[RetiredRule] = []

    for r in rules_in:
        if getattr(r, "direction", None) != direction:
            kept.append(r)
            continue

        st = score_rule_event(df_h, r, event_col=event_col, fwd_col=fwd_col)
        if st is None:
            kept.append(r)
            continue

        try:
            sup = int(st.support)
            med = float(st.median_signed)
            es5 = float(st.es5_signed)
            sig = str(getattr(st, "signature", _rule_signature(r)))
        except Exception:
            kept.append(r)
            continue

        if sup < int(health_min_n):
            kept.append(r)
            continue

        if (med <= float(health_med_min)) or (es5 <= float(health_es5_min)):
            retired.append(RetiredRule(sig, sup, med, es5))
        else:
            kept.append(r)

    retired.sort(key=lambda x: (x.median_signed, x.es5_signed))
    return kept, retired


# =========================
# main
# =========================
def main() -> int:
    _add_src_to_syspath()

    from python_edge.data.ingest_aggs import load_aggs, to_daily_index  # type: ignore
    from python_edge.features.build_features_daily import DailyFeatureConfig, build_daily_features  # type: ignore
    from python_edge.rules.event_mining import EventMiningConfig, label_events, mine_event_rules, evaluate_rules_oos  # type: ignore
    from python_edge.rules.event_mining import _apply_rule_mask  # type: ignore

    root = Path(__file__).resolve().parents[1]

    start = _env_str("DATA_START", "")
    end = _env_str("DATA_END", "")
    if not start or not end:
        raise RuntimeError("Missing DATA_START/DATA_END. Example: 2023-01-01 / 2026-02-28")

    dataset_root = Path(_env_str("DATA_OUT_DIR", str(root / "data" / "raw" / "massive_dataset")))
    tickers = _load_universe(root)

    cfg = EventMiningConfig(
        fwd_days=_env_int("LAB_FWD_DAYS", 5),
        sigma_lookback=_env_int("LAB_SIGMA_LB", 60),
        k_sigma=_env_float("LAB_K_SIGMA", 1.5),
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

    HEALTH_ENABLE = (_env_int("HEALTH_ENABLE", 0) == 1)
    HEALTH_WIN = _env_int("HEALTH_WIN", 60)
    HEALTH_MIN_N = _env_int("HEALTH_MIN_N", 50)
    HEALTH_MED_MIN = _env_float("HEALTH_MED_MIN", 0.0)
    HEALTH_ES5_MIN = _env_float("HEALTH_ES5_MIN", -0.01)

    WEIGHT_P95_TARGET = _env_float("WEIGHT_P95_TARGET", 0.0)  # 0 disables

    train_days = _env_int("WF_TRAIN_DAYS", 420)
    test_days = _env_int("WF_TEST_DAYS", 90)
    purge_days = _env_int("WF_PURGE_DAYS", 10)
    max_folds = _env_int("WF_MAX_FOLDS", 6)

    print(f"[CFG] vendor=massive dataset_root={dataset_root}")
    print(f"[CFG] universe={len(tickers)} start={start} end={end}")
    print(f"[CFG] event: fwd_days={cfg.fwd_days} k_sigma={cfg.k_sigma} sigma_lookback={cfg.sigma_lookback}")
    print(f"[CFG] rules: try={cfg.max_rules_try} keep={cfg.max_rules_keep} min_support={cfg.min_support} min_event_hits={cfg.min_event_hits}")
    print(f"[CFG] perm: trials={cfg.perm_trials} topk={cfg.perm_topk} gate={int(cfg.perm_gate_enabled)} margin={cfg.perm_gate_margin}")
    print(f"[CFG] OOS filter: support>={cfg.min_support} lift>={OOS_LIFT_MIN}")
    print(f"[CFG] short payoff-gate: mean>{SHORT_GATE_MEAN_MIN} p>0>{SHORT_GATE_PPOS_MIN}")
    print(f"[CFG] rank: K={K}")
    if HEALTH_ENABLE:
        print(f"[CFG] health: win={HEALTH_WIN} min_n={HEALTH_MIN_N} med_min={_fmt(HEALTH_MED_MIN,4)} es5_min={_fmt(HEALTH_ES5_MIN,4)}")

    # ---------- data ----------
    panels: List[pd.DataFrame] = []
    for t in tickers:
        r = load_aggs(dataset_root=dataset_root, symbol=t, tf="1d", start=start, end=end, prefer_full=True)
        df = to_daily_index(r.df)
        if df.empty:
            continue
        df = df[["date", "o", "h", "l", "c", "v"]].copy()
        df["symbol"] = t
        df = build_daily_features(df, DailyFeatureConfig(fwd_days=cfg.fwd_days))
        panels.append(df)

    if not panels:
        raise RuntimeError("No data loaded (panels empty).")

    all_df = pd.concat(panels, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    all_df = label_events(all_df, cfg)

    # keep only columns that exist
    feature_cols = [
        "mom_1d", "mom_3d", "mom_5d", "mom_10d", "mom_20d",
        "ema_dist", "ema_fast_slope", "ema_slow_slope",
        "atr_pct", "rv_10", "compression",
    ]
    feature_cols = [c for c in feature_cols if c in all_df.columns]

    need = feature_cols + [f"fwd_{cfg.fwd_days}d_ret", "ret_1d", "event_up", "event_dn", "event_thr"]
    all_df = all_df.replace([np.inf, -np.inf], np.nan).dropna(subset=need).copy()

    print(f"[DATA] pooled rows={len(all_df)} dates={all_df['date'].nunique()} symbols={all_df['symbol'].nunique()}")
    print(f"[DATA] base_rate up={float(all_df['event_up'].mean()):.4%} dn={float(all_df['event_dn'].mean()):.4%}")

    dates = sorted(all_df["date"].unique().tolist())
    folds = _make_folds(dates, train_days=train_days, test_days=test_days, purge_days=purge_days, max_folds=max_folds)
    if not folds:
        raise RuntimeError("No folds produced. Check WF_* params vs available dates.")

    # ---------- per-fold mining + OOS ----------
    fold_oos_rows: List[pd.DataFrame] = []
    fold_rules_by_id: Dict[int, List[Any]] = {}

    for (fold_id, tr_s, tr_e, te_s, te_e) in folds:
        tr_dates = set(dates[tr_s:tr_e])
        te_dates = set(dates[te_s:te_e])

        df_tr = all_df[all_df["date"].isin(tr_dates)].copy()
        df_te = all_df[all_df["date"].isin(te_dates)].copy()

        print("\n" + "=" * 80)
        print(f"[LIB/FOLD {fold_id}] train={min(tr_dates)}..{max(tr_dates)}  test={min(te_dates)}..{max(te_dates)}")
        print(f"[LIB/FOLD {fold_id}] train rows={len(df_tr)} test rows={len(df_te)}")

        # mine (single call, returns long+short)
        rules, stats, perm = mine_event_rules(df_tr, feature_cols, cfg)
        fold_rules_by_id[fold_id] = list(rules)

        perm_p95 = float(perm.get("perm_topk_p95", float("nan"))) if isinstance(perm, dict) else float("nan")
        print(f"[LIB/FOLD {fold_id}] mined_rules={len(rules)} perm_p95={_fmt(perm_p95,3)}")

        df_oos = evaluate_rules_oos(df_te, list(rules), cfg)
        if df_oos is None or len(df_oos) == 0:
            print(f"[LIB/FOLD {fold_id}] OOS eval=0 pass=0")
            continue

        df_oos = df_oos.copy()
        df_oos["fold_id"] = fold_id

        before = len(df_oos)
        df_oos_f = df_oos[(df_oos["support"] >= cfg.min_support) & (df_oos["lift"] >= OOS_LIFT_MIN)].copy()
        print(f"[LIB/FOLD {fold_id}] OOS eval={before} pass(lift>={OOS_LIFT_MIN})={len(df_oos_f)}")

        fold_oos_rows.append(df_oos_f)

    # ---------- GLOBAL library ----------
    print("\n" + "=" * 80)
    print("[GLOBAL] Rule Library (OOS-filtered across folds)")

    if not fold_oos_rows:
        raise RuntimeError("No OOS rows collected after filtering.")

    df_oos_all = pd.concat(fold_oos_rows, ignore_index=True)

    agg = (
        df_oos_all.groupby(["direction", "signature"], as_index=False)
        .agg(
            fold_count=("fold_id", "nunique"),
            avg_lift=("lift", "mean"),
            avg_es5=("es5_signed", "mean"),
            avg_mean=("mean_signed", "mean"),
            avg_median=("median_signed", "mean"),
            avg_ppos=("p_pos_signed", "mean"),
            support=("support", "mean"),
        )
        .copy()
    )

    agg = agg[agg["fold_count"] >= 2].copy()
    agg = agg[agg["avg_lift"] >= OOS_LIFT_MIN].copy()

    # short payoff gate
    is_short = agg["direction"] == "short"
    if is_short.any():
        agg_short = agg[is_short].copy()
        agg_long = agg[~is_short].copy()
        agg_short = agg_short[
            (agg_short["avg_mean"] > SHORT_GATE_MEAN_MIN) & (agg_short["avg_ppos"] > SHORT_GATE_PPOS_MIN)
        ].copy()
        agg = pd.concat([agg_long, agg_short], ignore_index=True)

    agg["weight"] = [
        _weight_from_metrics(float(r["avg_lift"]), int(r["fold_count"]), float(r["avg_es5"]), float(OOS_LIFT_MIN))
        for _, r in agg.iterrows()
    ]

    agg = _rescale_weights_to_target(agg, WEIGHT_P95_TARGET)

    print(f"[GLOBAL] library_size={len(agg)} (fold_count>=2)")
    print(f"[GLOBAL] long_kept={int((agg['direction']=='long').sum())} short_kept(after payoff-gate)={int((agg['direction']=='short').sum())}")

    lib_w: Dict[str, float] = {str(r["signature"]): float(r["weight"]) for _, r in agg.iterrows()}

    # ---------- RANK per fold ----------
    fwd_col = f"fwd_{cfg.fwd_days}d_ret"

    for (fold_id, tr_s, tr_e, te_s, te_e) in folds:
        te_dates = set(dates[te_s:te_e])
        tr_dates = set(dates[tr_s:tr_e])

        df_tr = all_df[all_df["date"].isin(tr_dates)].copy()
        df_te = all_df[all_df["date"].isin(te_dates)].copy()

        rules_fold = fold_rules_by_id.get(fold_id, [])
        # keep only rules present in global library signatures
        rules_fold = [r for r in rules_fold if getattr(r, "signature", None) in lib_w]

        # split by direction
        fold_long_rules = [r for r in rules_fold if getattr(r, "direction", None) == "long"]
        fold_short_rules = [r for r in rules_fold if getattr(r, "direction", None) == "short"]

        print("\n" + "=" * 80)
        print(f"[RANK/FOLD {fold_id}] OOS window={min(te_dates)}..{max(te_dates)}  rows={len(df_te)}")

        if HEALTH_ENABLE and fold_long_rules:
            print(f"[HEALTH] enabled=1 win={HEALTH_WIN} min_n={HEALTH_MIN_N} med_min={_fmt(HEALTH_MED_MIN,4)} es5_min={_fmt(HEALTH_ES5_MIN,4)}")
            n0 = len(fold_long_rules)
            fold_long_rules, retired_l = _health_filter_rules(
                df_tr,
                fold_long_rules,
                direction="long",
                cfg=cfg,
                health_win=HEALTH_WIN,
                health_min_n=HEALTH_MIN_N,
                health_med_min=HEALTH_MED_MIN,
                health_es5_min=HEALTH_ES5_MIN,
            )
            print(f"[HEALTH] long: before={n0} after={len(fold_long_rules)} retired={len(retired_l)}")
            for rr in retired_l[:5]:
                print(f"[HEALTH] retired_long sig={rr.signature} support={rr.support} med={_fmt(rr.median_signed,4)} es5={_fmt(rr.es5_signed,4)}")

        print(f"[RANK/FOLD {fold_id}] fold_rules kept: long={len(fold_long_rules)} short={len(fold_short_rules)}")

        long_score = np.zeros(len(df_te), dtype=float)
        short_score = np.zeros(len(df_te), dtype=float)
        fires_long = 0
        fires_short = 0

        for r in fold_long_rules:
            w = float(lib_w.get(r.signature, 0.0))
            if w <= 0:
                continue
            m = _apply_rule_mask(df_te, r).to_numpy(dtype=bool, copy=False)
            c = int(m.sum())
            fires_long += c
            if c:
                long_score[m] += w

        for r in fold_short_rules:
            w = float(lib_w.get(r.signature, 0.0))
            if w <= 0:
                continue
            m = _apply_rule_mask(df_te, r).to_numpy(dtype=bool, copy=False)
            c = int(m.sum())
            fires_short += c
            if c:
                short_score[m] += w

        net_score = long_score - short_score
        scored_rows = int(np.sum(net_score != 0.0))
        print(f"[RANK/FOLD {fold_id}] scored_rows(nonzero)={scored_rows}/{len(df_te)} fires_long={fires_long} fires_short={fires_short}")

        df_te2 = df_te.copy()
        df_te2["score"] = net_score

        picked: List[pd.DataFrame] = []
        for d, g in df_te2.groupby("date", sort=True):
            gg = g[g["score"] != 0.0].copy()
            if gg.empty:
                continue
            top = gg.sort_values("score", ascending=False).head(K)
            picked.append(top)

        if picked:
            df_p = pd.concat(picked, ignore_index=True)
            arr = df_p[fwd_col].astype(float).to_numpy()
            mean_fwd = float(np.mean(arr)) if arr.size else 0.0
            med = float(np.median(arr)) if arr.size else 0.0
            ppos = float(np.mean(arr > 0)) if arr.size else 0.0
            print(f"[RANK/FOLD {fold_id}] LONG top-{K} fwd_{cfg.fwd_days}d: mean={_fmt(mean_fwd,4)} med={_fmt(med,4)} p>0={_fmt_pct(ppos)}")
        else:
            print(f"[RANK/FOLD {fold_id}] LONG top-{K} fwd_{cfg.fwd_days}d: mean=0.0000 med=0.0000 p>0=0.00%")

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
        print("".join(traceback.format_exception(type(e), e, e.__traceback__)))
        _press_enter_exit(1)
