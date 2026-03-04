# scripts/run_ranker_mvp.py
# Python-Edge: mining -> library -> ranker (WF) with RT freshness controls.
#
# Goals for this version:
# - No indentation fragility (single-file, deterministic control flow)
# - MUST_PASS_LATEST mode that can trade even if fold_count>=2 collapses (explicit FAILSAFE)
# - Phase-2 HEALTH filter that retires rules only when BOTH median and ES5 are below thresholds
# - Strict direction handling (no short leakage when short_kept=0)
# - Perm gate reporting is nan-safe (no RuntimeWarning; explicit status)
# - Compact output by default (no per-day spam)
# - Double-click runnable (always waits for Enter)
#
# Expected functions from python_edge.rules.event_mining:
# - label_events(df, cfg)
# - mine_event_rules(df_train, feature_cols, cfg, direction=...) -> (rules, stats, perm_dict)
# - evaluate_rules_oos(df_oos, rules, cfg) -> DataFrame with at least:
#     direction, signature, support, lift, mean_signed, median_signed, p_pos_signed, es5_signed
# - score_rule_event(df, rule, event_col, fwd_col) -> stats object with support/median_signed/es5_signed/signature
# - _apply_rule_mask(df, rule) -> Series[bool]
#
# NOTE: if your mine_event_rules signature differs, adapt the call in _mine_dir().

from __future__ import annotations

import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# -------------------------
# UX: never auto-close
# -------------------------

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


# -------------------------
# env helpers
# -------------------------

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


# -------------------------
# universe / folds
# -------------------------

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
    dates: List[Any],
    *,
    train_days: int,
    test_days: int,
    purge_days: int,
    max_folds: int,
) -> List[Tuple[int, int, int, int, int]]:
    folds: List[Tuple[int, int, int, int, int]] = []
    t_end = int(train_days)
    fid = 0
    step = int(test_days)

    while True:
        train_end = t_end
        test_start = train_end + int(purge_days)
        test_end = test_start + int(test_days)
        train_start = max(0, train_end - int(train_days))

        if test_end > len(dates):
            break

        fid += 1
        folds.append((fid, train_start, train_end, test_start, test_end))

        t_end += step
        if fid >= int(max_folds):
            break

    return folds


# -------------------------
# rules: signature + weights
# -------------------------

def _rule_direction(rule: Any, default: str = "long") -> str:
    d = getattr(rule, "direction", None)
    if isinstance(d, str) and d.strip():
        dd = d.strip().lower()
        if dd in ("long", "short"):
            return dd
    return default


def _rule_signature(rule: Any) -> str:
    # Prefer rule.signature if present
    sig = getattr(rule, "signature", None)
    if isinstance(sig, str) and sig.strip():
        s = sig.strip()
        # enforce direction prefix if absent
        d = _rule_direction(rule, default="long")
        if not (s.startswith("long|") or s.startswith("short|")):
            return f"{d}|{s}"
        return s

    d = _rule_direction(rule, default="long")
    conds = getattr(rule, "conds", [])
    parts: List[str] = []
    try:
        for feat, op, meta_key in conds:
            tag = meta_key.split("__")[-1] if isinstance(meta_key, str) and "__" in meta_key else str(meta_key)
            parts.append(f"{feat}{op}{tag}")
        parts.sort()
        return d + "|" + "|".join(parts)
    except Exception:
        return d + "|" + str(rule)


def _weight_from_metrics(lift: float, fold_count: int, es5: float, oos_lift_min: float) -> float:
    # deterministic, monotone in lift, penalize bad tails
    edge = max(0.0, float(lift) - float(oos_lift_min))
    stability = 1.25 if int(fold_count) >= 3 else 1.0
    tail_pen = max(0.0, (-float(es5)) - 0.10)
    w = stability * edge / (1.0 + 5.0 * tail_pen)
    return float(w)


def _rescale_weights_to_target(df_rules: pd.DataFrame, target_p95_weight: float) -> pd.DataFrame:
    df = df_rules.copy()
    if target_p95_weight <= 0 or df.empty or "weight" not in df.columns:
        return df
    w = pd.to_numeric(df["weight"], errors="coerce").to_numpy(dtype=float)
    w = w[np.isfinite(w)]
    if w.size == 0:
        return df
    p95 = float(np.quantile(w, 0.95))
    if p95 <= 0:
        return df
    scale = float(target_p95_weight) / p95
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce") * scale
    df["weight_scale"] = float(scale)
    return df


# -------------------------
# perm p95: nan-safe
# -------------------------

def _safe_perm_p95(perm_obj: Any) -> float:
    if perm_obj is None:
        return float("nan")
    if isinstance(perm_obj, dict):
        for k in ("perm_p95", "p95", "lift_p95", "p95_lift"):
            if k in perm_obj:
                try:
                    return float(perm_obj[k])
                except Exception:
                    continue
    try:
        return float(getattr(perm_obj, "perm_p95"))
    except Exception:
        return float("nan")


def _perm_p95_merge(a: Any, b: Any) -> Tuple[Optional[float], str]:
    x = _safe_perm_p95(a)
    y = _safe_perm_p95(b)
    if (not np.isfinite(x)) and (not np.isfinite(y)):
        return None, "unavailable"
    if np.isfinite(x) and np.isfinite(y):
        return float(max(x, y)), "ok"
    if np.isfinite(x):
        return float(x), "partial"
    return float(y), "partial"


# -------------------------
# Phase 2: health filter
# -------------------------

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

    from python_edge.rules.event_mining import score_rule_event  # type: ignore

    kept: List[Any] = []
    retired: List[RetiredRule] = []

    for r in rules_in:
        if _rule_direction(r, default=direction) != direction:
            kept.append(r)
            continue

        st = score_rule_event(df_h, r, event_col=event_col, fwd_col=fwd_col)
        if st is None:
            kept.append(r)
            continue

        try:
            sup = int(getattr(st, "support"))
            med = float(getattr(st, "median_signed"))
            es5 = float(getattr(st, "es5_signed"))
            sig = str(getattr(st, "signature", _rule_signature(r)))
        except Exception:
            kept.append(r)
            continue

        if sup < int(health_min_n):
            kept.append(r)
            continue

        # retire only when BOTH are bad
        if (med <= float(health_med_min)) and (es5 <= float(health_es5_min)):
            retired.append(RetiredRule(sig, sup, med, es5))
        else:
            kept.append(r)

    retired.sort(key=lambda x: (x.median_signed, x.es5_signed))
    return kept, retired


# -------------------------
# ranking eval
# -------------------------

def _topk_daily_scores(df: pd.DataFrame, score_col: str, k: int, fwd_col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["date", "n", "mean_fwd", "median_fwd", "p_pos"])

    out_rows: List[Dict[str, Any]] = []
    for d, g in df.groupby("date", sort=True):
        gg = g.sort_values(score_col, ascending=False).head(int(k))
        x = pd.to_numeric(gg[fwd_col], errors="coerce").to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            out_rows.append({"date": d, "n": 0, "mean_fwd": 0.0, "median_fwd": 0.0, "p_pos": 0.0})
            continue
        out_rows.append(
            {
                "date": d,
                "n": int(x.size),
                "mean_fwd": float(np.mean(x)),
                "median_fwd": float(np.median(x)),
                "p_pos": float(np.mean(x > 0.0)),
            }
        )

    return pd.DataFrame(out_rows)


# -------------------------
# mining wrapper (direction-safe)
# -------------------------

def _mine_dir(df_tr: pd.DataFrame, feature_cols: List[str], cfg: Any, direction: str) -> Tuple[List[Any], Any, Any]:
    from python_edge.rules.event_mining import mine_event_rules  # type: ignore

    # Prefer keyword direction if supported; fallback to positional if not.
    try:
        rules, stats, perm = mine_event_rules(df_tr, feature_cols, cfg, direction=direction)
        return list(rules), stats, perm
    except TypeError:
        # legacy signature: mine_event_rules(df_tr, cfg, feature_cols=..., direction=...)
        try:
            rules, stats, perm = mine_event_rules(df_tr, cfg, feature_cols=feature_cols, direction=direction)
            return list(rules), stats, perm
        except TypeError:
            # last resort: mine_event_rules(df_tr, feature_cols, cfg) and then filter by rule.direction
            rules, stats, perm = mine_event_rules(df_tr, feature_cols, cfg)
            rr = [r for r in list(rules) if _rule_direction(r, default=direction) == direction]
            return rr, stats, perm


# -------------------------
# main
# -------------------------

def main() -> int:
    root = _add_src_to_syspath()

    from python_edge.data.ingest_aggs import load_aggs, to_daily_index  # type: ignore
    from python_edge.features.build_features_daily import DailyFeatureConfig, build_daily_features  # type: ignore
    from python_edge.rules.event_mining import EventMiningConfig, label_events, evaluate_rules_oos, _apply_rule_mask  # type: ignore

    start = _env_str("DATA_START", "")
    end = _env_str("DATA_END", "")
    if not start or not end:
        raise RuntimeError('Missing DATA_START/DATA_END. Example: 2023-01-01 / 2026-02-28')

    dataset_root = Path(_env_str("DATA_OUT_DIR", str(root / "data" / "raw" / "massive_dataset")))
    tickers = _load_universe(root)

    # --- config ---
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

    # Phase-2 Health
    HEALTH_ENABLE = (_env_int("HEALTH_ENABLE", 0) == 1)
    HEALTH_WIN = _env_int("HEALTH_WIN", 60)
    HEALTH_MIN_N = _env_int("HEALTH_MIN_N", 50)
    HEALTH_MED_MIN = _env_float("HEALTH_MED_MIN", 0.0)
    HEALTH_ES5_MIN = _env_float("HEALTH_ES5_MIN", -0.08)

    # weights scaling
    WEIGHT_P95_TARGET = _env_float("WEIGHT_P95_TARGET", 0.0)  # 0 disables

    # walk-forward
    train_days = _env_int("WF_TRAIN_DAYS", 420)
    test_days = _env_int("WF_TEST_DAYS", 90)
    purge_days = _env_int("WF_PURGE_DAYS", 10)
    max_folds = _env_int("WF_MAX_FOLDS", 6)

    # RT freshness
    RECENCY_ENABLE = (_env_int("RECENCY_ENABLE", 1) == 1)
    RECENCY_DECAY = _env_float("RECENCY_DECAY", 0.85)
    MUST_PASS_LATEST = (_env_int("MUST_PASS_LATEST", 0) == 1)
    MIN_FOLD_COUNT_RT = _env_int("MIN_FOLD_COUNT_RT", 1)

    LATEST_LIFT_MIN = _env_float("LATEST_LIFT_MIN", 1.00)
    LATEST_MED_MIN = _env_float("LATEST_MED_MIN", 0.0)
    LATEST_ES5_MIN = _env_float("LATEST_ES5_MIN", -0.08)

    RUNTIME_LONG_TOPN = _env_int("RUNTIME_LONG_TOPN", 40)
    RUNTIME_SHORT_TOPN = _env_int("RUNTIME_SHORT_TOPN", 20)

    # output
    COMPACT = (_env_int("COMPACT", 1) == 1)
    PRINT_DAYS = _env_int("PRINT_DAYS", 0)

    # --- prints ---
    print(f"[CFG] vendor=massive dataset_root={dataset_root}")
    print(f"[CFG] universe={len(tickers)} start={start} end={end}")
    print(f"[CFG] event: fwd_days={cfg.fwd_days} k_sigma={cfg.k_sigma} sigma_lookback={cfg.sigma_lookback}")
    print(
        f"[CFG] rules: try={cfg.max_rules_try} keep={cfg.max_rules_keep} "
        f"min_support={cfg.min_support} min_event_hits={cfg.min_event_hits}"
    )
    print(f"[CFG] perm: trials={cfg.perm_trials} topk={cfg.perm_topk} gate={int(cfg.perm_gate_enabled)} margin={cfg.perm_gate_margin}")
    print(f"[CFG] OOS filter: support>={cfg.min_support} lift>={OOS_LIFT_MIN}")
    print(f"[CFG] short payoff-gate: mean>{SHORT_GATE_MEAN_MIN} p>0>{SHORT_GATE_PPOS_MIN}")
    print(f"[CFG] rank: K={K}")
    if HEALTH_ENABLE:
        print(f"[CFG] health: win={HEALTH_WIN} min_n={HEALTH_MIN_N} med_min={_fmt(HEALTH_MED_MIN,4)} es5_min={_fmt(HEALTH_ES5_MIN,4)}")
    if RECENCY_ENABLE or MUST_PASS_LATEST:
        print(
            f"[CFG] recency: enable={int(RECENCY_ENABLE)} decay={_fmt(RECENCY_DECAY,3)} "
            f"must_pass_latest={int(MUST_PASS_LATEST)} min_fc_rt={MIN_FOLD_COUNT_RT} "
            f"latest_lift_min={_fmt(LATEST_LIFT_MIN,3)} latest_med_min={_fmt(LATEST_MED_MIN,4)} latest_es5_min={_fmt(LATEST_ES5_MIN,4)}"
        )
        print(f"[CFG] runtime caps: long_topn={RUNTIME_LONG_TOPN} short_topn={RUNTIME_SHORT_TOPN}")
    if WEIGHT_P95_TARGET > 0:
        print(f"[CFG] weights: p95_target={_fmt(WEIGHT_P95_TARGET,4)}")

    # --- load data ---
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

    feature_cols = [
        "mom_1d",
        "mom_3d",
        "mom_5d",
        "mom_10d",
        "mom_20d",
        "ema_dist",
        "ema_fast_slope",
        "ema_slow_slope",
        "atr_pct",
        "rv_10",
        "compression",
    ]
    feature_cols = [c for c in feature_cols if c in all_df.columns]

    fwd_col = f"fwd_{cfg.fwd_days}d_ret"
    need_cols = feature_cols + [fwd_col, "ret_1d", "event_up", "event_dn", "event_thr"]
    all_df = all_df.replace([np.inf, -np.inf], np.nan).dropna(subset=need_cols).copy()

    print(f"[DATA] pooled rows={len(all_df)} dates={all_df['date'].nunique()} symbols={all_df['symbol'].nunique()}")
    print(f"[DATA] base_rate up={float(all_df['event_up'].mean()):.4%} dn={float(all_df['event_dn'].mean()):.4%}")

    dates = sorted(all_df["date"].unique().tolist())
    folds = _make_folds(dates, train_days=train_days, test_days=test_days, purge_days=purge_days, max_folds=max_folds)
    if not folds:
        raise RuntimeError("No folds produced. Check WF_* settings.")

    # --- per fold mining + OOS eval ---
    df_oos_all: List[pd.DataFrame] = []
    fold_rule_bank: Dict[int, Dict[str, List[Any]]] = {}

    for fid, tr_s, tr_e, te_s, te_e in folds:
        tr0 = dates[tr_s]
        tr1 = dates[tr_e - 1]
        te0 = dates[te_s]
        te1 = dates[te_e - 1]

        print("\n" + "=" * 80)
        print(f"[LIB/FOLD {fid}] train={tr0}..{tr1}  test={te0}..{te1}")

        df_tr = all_df[(all_df["date"] >= tr0) & (all_df["date"] <= tr1)].copy()
        df_te = all_df[(all_df["date"] >= te0) & (all_df["date"] <= te1)].copy()

        print(f"[LIB/FOLD {fid}] train rows={len(df_tr)} test rows={len(df_te)}")

        rules_long, _stats_l, perm_l = _mine_dir(df_tr, feature_cols, cfg, "long")
        rules_short, _stats_s, perm_s = _mine_dir(df_tr, feature_cols, cfg, "short")

        # enforce direction safety
        rules_long = [r for r in rules_long if _rule_direction(r, default="long") == "long"]
        rules_short = [r for r in rules_short if _rule_direction(r, default="short") == "short"]

        fold_rule_bank[fid] = {"long": list(rules_long), "short": list(rules_short)}

        perm_p95, perm_status = _perm_p95_merge(perm_l, perm_s)
        if perm_p95 is None:
            print(f"[LIB/FOLD {fid}] mined_rules={len(rules_long)} perm_p95=NA status={perm_status}")
        else:
            print(f"[LIB/FOLD {fid}] mined_rules={len(rules_long)} perm_p95={_fmt(float(perm_p95),3)} status={perm_status}")

        df_oos_l = evaluate_rules_oos(df_te, rules_long, cfg)
        df_oos_s = evaluate_rules_oos(df_te, rules_short, cfg)
        df_oos = pd.concat([df_oos_l, df_oos_s], ignore_index=True)

        # normalize expected columns
        if "direction" not in df_oos.columns:
            df_oos["direction"] = "long"
        df_oos["direction"] = df_oos["direction"].astype(str).str.lower().str.strip()
        df_oos.loc[~df_oos["direction"].isin(["long", "short"]), "direction"] = "long"

        if "signature" not in df_oos.columns:
            df_oos["signature"] = ""

        # enforce direction prefix in signature for stable joins
        def _fix_sig(row: pd.Series) -> str:
            d = str(row.get("direction", "long")).strip().lower()
            s = str(row.get("signature", "")).strip()
            if not s:
                return ""
            if s.startswith("long|") or s.startswith("short|"):
                return s
            return f"{d}|{s}"

        df_oos["signature"] = df_oos.apply(_fix_sig, axis=1)

        for col in ("support", "lift", "mean_signed", "median_signed", "p_pos_signed", "es5_signed"):
            if col not in df_oos.columns:
                df_oos[col] = np.nan

        df_oos["fold_id"] = int(fid)
        df_oos["train_start"] = tr0
        df_oos["train_end"] = tr1
        df_oos["test_start"] = te0
        df_oos["test_end"] = te1

        df_oos["support"] = pd.to_numeric(df_oos["support"], errors="coerce").fillna(0).astype(int)
        df_oos["lift"] = pd.to_numeric(df_oos["lift"], errors="coerce")
        df_oos["pass_oos"] = (df_oos["support"] >= int(cfg.min_support)) & (df_oos["lift"] >= float(OOS_LIFT_MIN))

        n_eval = int(len(df_oos))
        n_pass = int(df_oos["pass_oos"].sum())
        print(f"[LIB/FOLD {fid}] OOS eval={n_eval} pass(lift>={OOS_LIFT_MIN})={n_pass}")

        df_oos_all.append(df_oos)

    # --- GLOBAL library ---
    print("\n" + "=" * 80)
    print("[GLOBAL] Rule Library (OOS-filtered across folds)")

    df_fold = pd.concat(df_oos_all, ignore_index=True)
    df_fold = df_fold[df_fold["pass_oos"]].copy()

    lib_w: Dict[str, float] = {}
    lib_dir: Dict[str, str] = {}

    if df_fold.empty:
        print("[GLOBAL] library_size=0 (no rules passed OOS in any fold)")
        print("[GLOBAL] long_kept=0 short_kept(after payoff-gate)=0")
    else:
        last_fold_id = int(df_fold["fold_id"].max())

        if RECENCY_ENABLE:
            # recency weight by fold distance
            df_fold["recency_w"] = np.exp(-float(RECENCY_DECAY) * (last_fold_id - df_fold["fold_id"].astype(int)))
        else:
            df_fold["recency_w"] = 1.0

        df_last = df_fold[df_fold["fold_id"] == last_fold_id].copy()

        last_metrics = (
            df_last.groupby(["direction", "signature"], as_index=False)
            .agg(
                last_lift=("lift", "mean"),
                last_es5=("es5_signed", "mean"),
                last_mean=("mean_signed", "mean"),
                last_median=("median_signed", "mean"),
                last_ppos=("p_pos_signed", "mean"),
                last_support=("support", "mean"),
            )
            .copy()
        )

        def _wavg(g: pd.DataFrame, col: str) -> float:
            w = pd.to_numeric(g["recency_w"], errors="coerce").to_numpy(dtype=float)
            x = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float)
            m = np.isfinite(w) & np.isfinite(x) & (w > 0)
            if not np.any(m):
                return float("nan")
            return float(np.sum(w[m] * x[m]) / np.sum(w[m]))

        agg_rows: List[Dict[str, Any]] = []
        for (direction, signature), g in df_fold.groupby(["direction", "signature"], sort=False):
            agg_rows.append(
                {
                    "direction": direction,
                    "signature": signature,
                    "fold_count": int(g["fold_id"].nunique()),
                    "w_lift": _wavg(g, "lift"),
                    "w_es5": _wavg(g, "es5_signed"),
                    "w_mean": _wavg(g, "mean_signed"),
                    "w_median": _wavg(g, "median_signed"),
                    "w_ppos": _wavg(g, "p_pos_signed"),
                    "w_support": _wavg(g, "support"),
                }
            )

        agg = pd.DataFrame(agg_rows).merge(last_metrics, on=["direction", "signature"], how="left")

        # fold-count threshold
        min_fc = 2
        if MUST_PASS_LATEST:
            min_fc = int(MIN_FOLD_COUNT_RT)
        agg = agg[agg["fold_count"] >= int(min_fc)].copy()

        # weighted OOS
        agg = agg[pd.to_numeric(agg["w_lift"], errors="coerce") >= float(OOS_LIFT_MIN)].copy()

        # must-pass-latest gates
        if MUST_PASS_LATEST:
            agg = agg[agg["last_lift"].notna()].copy()
            agg = agg[pd.to_numeric(agg["last_lift"], errors="coerce") >= float(LATEST_LIFT_MIN)].copy()
            agg = agg[pd.to_numeric(agg["last_median"], errors="coerce") >= float(LATEST_MED_MIN)].copy()
            agg = agg[pd.to_numeric(agg["last_es5"], errors="coerce") >= float(LATEST_ES5_MIN)].copy()

        # short payoff-gate
        if (agg["direction"] == "short").any():
            agg_short = agg[agg["direction"] == "short"].copy()
            agg_long = agg[agg["direction"] == "long"].copy()
            agg_short = agg_short[
                (pd.to_numeric(agg_short["w_mean"], errors="coerce") > float(SHORT_GATE_MEAN_MIN))
                & (pd.to_numeric(agg_short["w_ppos"], errors="coerce") > float(SHORT_GATE_PPOS_MIN))
            ].copy()
            agg = pd.concat([agg_long, agg_short], ignore_index=True)

        # weights
        agg["weight"] = [
            _weight_from_metrics(float(r["w_lift"]), int(r["fold_count"]), float(r["w_es5"]), float(OOS_LIFT_MIN))
            for _, r in agg.iterrows()
        ]

        if WEIGHT_P95_TARGET > 0:
            agg = _rescale_weights_to_target(agg, WEIGHT_P95_TARGET)

        # runtime caps: strictly by direction
        agg = agg.sort_values(["direction", "weight", "last_lift", "w_lift"], ascending=[True, False, False, False]).copy()
        use_long = agg[agg["direction"] == "long"].head(int(RUNTIME_LONG_TOPN))
        use_short = agg[agg["direction"] == "short"].head(int(RUNTIME_SHORT_TOPN))
        use = pd.concat([use_long, use_short], ignore_index=True)

        for _, r in use.iterrows():
            sig = str(r["signature"]).strip()
            if not sig:
                continue
            lib_w[sig] = float(r["weight"])
            lib_dir[sig] = str(r["direction"]).strip().lower()

        # FAILSAFE (explicit): if empty under MUST_PASS_LATEST, use latest fold pass rules only
        if (len(lib_w) == 0) and MUST_PASS_LATEST:
            print("[GLOBAL][FAILSAFE] library empty under MUST_PASS_LATEST -> using latest-fold pass rules only (explicit)")
            if df_last.empty:
                print("[GLOBAL][FAILSAFE] latest fold has 0 OOS-pass rows; staying empty (explicit)")
            else:
                df_last_pass = df_last[df_last["lift"] >= float(OOS_LIFT_MIN)].copy()
                if df_last_pass.empty:
                    print("[GLOBAL][FAILSAFE] latest fold has 0 rules passing OOS_LIFT_MIN; staying empty (explicit)")
                else:
                    df_last_pass["weight"] = [
                        _weight_from_metrics(float(l), 1, float(e), float(OOS_LIFT_MIN))
                        for l, e in zip(
                            pd.to_numeric(df_last_pass["lift"], errors="coerce").to_numpy(dtype=float),
                            pd.to_numeric(df_last_pass["es5_signed"], errors="coerce").to_numpy(dtype=float),
                        )
                    ]
                    df_last_pass = df_last_pass.sort_values(["direction", "weight", "lift"], ascending=[True, False, False]).copy()
                    df_use_long = df_last_pass[df_last_pass["direction"] == "long"].head(int(RUNTIME_LONG_TOPN))
                    df_use_short = df_last_pass[df_last_pass["direction"] == "short"].head(int(RUNTIME_SHORT_TOPN))
                    df_use2 = pd.concat([df_use_long, df_use_short], ignore_index=True)
                    for _, rr in df_use2.iterrows():
                        sig = str(rr["signature"]).strip()
                        if not sig:
                            continue
                        lib_w[sig] = float(rr["weight"])
                        lib_dir[sig] = str(rr["direction"]).strip().lower()

        long_kept = int(sum(1 for s, d in lib_dir.items() if d == "long"))
        short_kept = int(sum(1 for s, d in lib_dir.items() if d == "short"))
        print(f"[GLOBAL] library_size={len(lib_w)} (min_fc={min_fc})")
        print(f"[GLOBAL] long_kept={long_kept} short_kept(after payoff-gate)={short_kept}")

    # --- rank per fold ---
    for fid, tr_s, tr_e, te_s, te_e in folds:
        tr0 = dates[tr_s]
        tr1 = dates[tr_e - 1]
        te0 = dates[te_s]
        te1 = dates[te_e - 1]

        df_tr = all_df[(all_df["date"] >= tr0) & (all_df["date"] <= tr1)].copy()
        df_te = all_df[(all_df["date"] >= te0) & (all_df["date"] <= te1)].copy()

        print("\n" + "=" * 80)
        print(f"[RANK/FOLD {fid}] OOS window={te0}..{te1}  rows={len(df_te)}")

        if HEALTH_ENABLE:
            print(f"[HEALTH] enabled=1 win={HEALTH_WIN} min_n={HEALTH_MIN_N} med_min={_fmt(HEALTH_MED_MIN,4)} es5_min={_fmt(HEALTH_ES5_MIN,4)}")

        bank = fold_rule_bank.get(fid, {"long": [], "short": []})

        # strict direction selection: signature must exist in lib and direction must match lib_dir
        fold_long_rules: List[Any] = []
        fold_short_rules: List[Any] = []

        for r in bank.get("long", []):
            sig = _rule_signature(r)
            if sig in lib_w and lib_dir.get(sig, "long") == "long":
                fold_long_rules.append(r)

        for r in bank.get("short", []):
            sig = _rule_signature(r)
            if sig in lib_w and lib_dir.get(sig, "short") == "short":
                fold_short_rules.append(r)

        # HEALTH apply
        retired_l: List[RetiredRule] = []
        retired_s: List[RetiredRule] = []

        if HEALTH_ENABLE and fold_long_rules:
            before = len(fold_long_rules)
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
            print(f"[HEALTH] long: before={before} after={len(fold_long_rules)} retired={len(retired_l)}")
            for rr in retired_l[:5]:
                print(f"[HEALTH] retired_long sig={rr.signature} support={rr.support} med={_fmt(rr.median_signed,4)} es5={_fmt(rr.es5_signed,4)}")

        if HEALTH_ENABLE and fold_short_rules:
            before = len(fold_short_rules)
            fold_short_rules, retired_s = _health_filter_rules(
                df_tr,
                fold_short_rules,
                direction="short",
                cfg=cfg,
                health_win=HEALTH_WIN,
                health_min_n=HEALTH_MIN_N,
                health_med_min=HEALTH_MED_MIN,
                health_es5_min=HEALTH_ES5_MIN,
            )
            print(f"[HEALTH] short: before={before} after={len(fold_short_rules)} retired={len(retired_s)}")

        print(f"[RANK/FOLD {fid}] fold_rules kept: long={len(fold_long_rules)} short={len(fold_short_rules)}")

        # score rows
        long_score = np.zeros(len(df_te), dtype=float)
        short_score = np.zeros(len(df_te), dtype=float)
        fires_long = 0
        fires_short = 0

        for r in fold_long_rules:
            sig = _rule_signature(r)
            w = float(lib_w.get(sig, 0.0))
            if w <= 0:
                continue
            m = _apply_rule_mask(df_te, r)
            mm = m.to_numpy(dtype=bool)
            if mm.size != long_score.size:
                continue
            fires_long += int(mm.sum())
            long_score[mm] += w

        for r in fold_short_rules:
            sig = _rule_signature(r)
            w = float(lib_w.get(sig, 0.0))
            if w <= 0:
                continue
            m = _apply_rule_mask(df_te, r)
            mm = m.to_numpy(dtype=bool)
            if mm.size != short_score.size:
                continue
            fires_short += int(mm.sum())
            short_score[mm] += w

        df_te = df_te.copy()
        df_te["score_long"] = long_score
        df_te["score_short"] = short_score
        df_te["score_net"] = long_score - short_score

        scored_rows = int((np.abs(df_te["score_net"].to_numpy(dtype=float)) > 0).sum())
        print(f"[RANK/FOLD {fid}] scored_rows(nonzero)={scored_rows}/{len(df_te)} fires_long={fires_long} fires_short={fires_short}")

        df_daily = _topk_daily_scores(df_te, "score_net", int(K), fwd_col)
        if df_daily.empty:
            print(f"[RANK/FOLD {fid}] LONG top-{K} fwd_{cfg.fwd_days}d: mean=0.0000 med=0.0000 p>0=0.00%")
            continue

        mean_fwd = float(df_daily["mean_fwd"].mean())
        med_fwd = float(np.median(df_daily["median_fwd"].to_numpy(dtype=float)))
        ppos = float(df_daily["p_pos"].mean())

        print(f"[RANK/FOLD {fid}] LONG top-{K} fwd_{cfg.fwd_days}d: mean={_fmt(mean_fwd,4)} med={_fmt(med_fwd,4)} p>0={_fmt_pct(ppos)}")

        if (not COMPACT) and PRINT_DAYS > 0:
            dlist = sorted(df_te["date"].unique().tolist())[: int(PRINT_DAYS)]
            for d in dlist:
                g = df_te[df_te["date"] == d].sort_values("score_net", ascending=False).head(int(K))
                print(f"\n[RANK {d}] LONG top-{K}:")
                for _, row in g.iterrows():
                    print(f"  {row['symbol']:<6} net={_fmt(float(row['score_net']),4)}")

    print("\n[DONE] Ranker MVP completed.")
    return 0


if __name__ == "__main__":
    try:
        rc = main()
        _press_enter_exit(int(rc))
    except SystemExit:
        raise
    except Exception:
        print("\n[ERROR] Unhandled exception:\n")
        print(traceback.format_exc())
        _press_enter_exit(1)
