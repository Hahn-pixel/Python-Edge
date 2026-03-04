from __future__ import annotations

"""Python-Edge :: Ranker MVP (daily)

Goals:
- One-command runnable script that mines event rules per fold, applies OOS filters,
  builds a GLOBAL library, applies optional RECENCY and HEALTH filters, then
  runs a simple rank-by-weight score on each OOS fold.

Design constraints (project conventions):
- No silent fail-open: any fallback is printed explicitly.
- Compact, deterministic console output.
- Robust imports: works whether modules live in the python_edge package or as
  repo-root modules.

Notes on recent breakages you saw:
- mine_event_rules() signature is (df_train, feature_cols, cfg, direction=None)
  in event_mining.py. Passing cfg as positional and feature_cols as kw leads to
  "multiple values for argument 'feature_cols'".
- Some local copies expect OHLC columns like close/high/low; the repo version
  uses c/h/l/o. We normalize both ways.
"""

import os
import sys
import math
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# -------------------------
# Helpers: printing
# -------------------------

def _p(msg: str) -> None:
    print(msg, flush=True)


def _banner() -> None:
    _p("=" * 80)


# -------------------------
# Robust imports (package or repo-root)
# -------------------------


def _import_ingest() -> Tuple[Any, Any]:
    # Prefer packaged path.
    try:
        from python_edge.data.ingest_aggs import load_aggs, to_daily_index  # type: ignore

        return load_aggs, to_daily_index
    except Exception:
        pass

    # Fallback to repo-root module.
    try:
        from ingest_aggs import load_aggs, to_daily_index  # type: ignore

        _p("[DBG] import ingest_aggs from repo-root")
        return load_aggs, to_daily_index
    except Exception as e:
        raise ImportError("Cannot import ingest_aggs (package or root)") from e


def _import_features() -> Tuple[Any, Any]:
    # Variant A: repo-root build_features_daily.py
    try:
        from python_edge.features.build_features_daily import DailyFeatureConfig, build_daily_features  # type: ignore

        return DailyFeatureConfig, build_daily_features
    except Exception:
        pass

    try:
        from build_features_daily import DailyFeatureConfig, build_daily_features  # type: ignore

        _p("[DBG] import build_features_daily from repo-root")
        return DailyFeatureConfig, build_daily_features
    except Exception:
        pass

    # Variant B: older naming
    try:
        from python_edge.features.build_features_daily import FeatureConfig, build_features_daily  # type: ignore

        return FeatureConfig, build_features_daily
    except Exception:
        pass

    try:
        from build_features_daily import FeatureConfig, build_features_daily  # type: ignore

        _p("[DBG] import build_features_daily(legacy) from repo-root")
        return FeatureConfig, build_features_daily
    except Exception as e:
        raise ImportError(
            "Cannot import feature builder. Expected DailyFeatureConfig/build_daily_features "
            "or FeatureConfig/build_features_daily."
        ) from e


def _import_event_mining() -> Any:
    try:
        import python_edge.rules.event_mining as em  # type: ignore

        return em
    except Exception:
        pass

    try:
        import event_mining as em  # type: ignore

        _p("[DBG] import event_mining from repo-root")
        return em
    except Exception as e:
        raise ImportError("Cannot import event_mining (package or root)") from e


# -------------------------
# Config
# -------------------------


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return int(default)
    return int(float(v))


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return float(default)
    return float(v)


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return bool(default)
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return bool(default)


@dataclass(frozen=True)
class HealthCfg:
    enabled: bool = False
    win: int = 60
    min_n: int = 50
    med_min: float = 0.0
    es5_min: float = -0.08


@dataclass(frozen=True)
class RecencyCfg:
    must_pass_latest: bool = False
    latest_only: bool = False


@dataclass(frozen=True)
class RunCfg:
    vendor: str = "massive"
    dataset_root: Path = Path("data/raw/massive_dataset")
    start: str = "2023-01-01"
    end: str = "2026-02-28"
    tf: str = "1D"

    # folds
    fold_count: int = 3

    # OOS filter
    oos_lift_min: float = 1.35
    oos_support_min: int = 200

    # rank
    K: int = 3

    # permutation display
    perm_gate: bool = True

    health: HealthCfg = HealthCfg()
    recency: RecencyCfg = RecencyCfg()


def load_cfg() -> RunCfg:
    root = Path(__file__).resolve().parents[1]
    ds = Path(os.getenv("DATASET_ROOT", str(root / "data" / "raw" / "massive_dataset")))

    start = os.getenv("DATA_START", "2023-01-01")
    end = os.getenv("DATA_END", "2026-02-28")

    health = HealthCfg(
        enabled=_env_bool("HEALTH", False),
        win=_env_int("HEALTH_WIN", 60),
        min_n=_env_int("HEALTH_MIN_N", 50),
        med_min=_env_float("HEALTH_MED_MIN", 0.0),
        es5_min=_env_float("HEALTH_ES5_MIN", -0.08),
    )

    rec = RecencyCfg(
        must_pass_latest=_env_bool("MUST_PASS_LATEST", False),
        latest_only=_env_bool("LATEST_ONLY", False),
    )

    return RunCfg(
        vendor="massive",
        dataset_root=ds,
        start=start,
        end=end,
        tf=os.getenv("TF", "1D"),
        fold_count=_env_int("FOLD_COUNT", 3),
        oos_lift_min=_env_float("OOS_LIFT_MIN", 1.35),
        oos_support_min=_env_int("OOS_SUPPORT_MIN", 200),
        K=_env_int("RANK_K", 3),
        perm_gate=_env_bool("PERM_GATE", True),
        health=health,
        recency=rec,
    )


# -------------------------
# Data normalization
# -------------------------


def _ensure_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the dataframe has o/h/l/c/v columns (float/int).

    Some local variants use open/high/low/close/volume.
    """
    if df.empty:
        return df

    out = df.copy()
    rename: Dict[str, str] = {}

    if "close" in out.columns and "c" not in out.columns:
        rename["close"] = "c"
    if "open" in out.columns and "o" not in out.columns:
        rename["open"] = "o"
    if "high" in out.columns and "h" not in out.columns:
        rename["high"] = "h"
    if "low" in out.columns and "l" not in out.columns:
        rename["low"] = "l"
    if "volume" in out.columns and "v" not in out.columns:
        rename["volume"] = "v"

    if rename:
        out = out.rename(columns=rename)

    needed = ["date", "o", "h", "l", "c", "v"]
    missing = [c for c in needed if c not in out.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns after normalization: {missing}. Have={list(out.columns)}")

    for c in ["o", "h", "l", "c", "v"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=["date", "c"]).copy()
    out["date"] = out["date"].astype(str)
    out = out.sort_values(["date"]).reset_index(drop=True)
    return out


# -------------------------
# Folds
# -------------------------


def _make_folds(dates: List[str], n_folds: int) -> List[Tuple[str, str, str, str]]:
    """Return list of (tr_start, tr_end, te_start, te_end) using expanding windows.

    We keep folds simple and deterministic:
    - Split the date index into (train_end, test_start..test_end) blocks.
    - Each fold advances forward; fold N is the newest.
    """
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")

    dates_sorted = sorted(set(dates))
    n = len(dates_sorted)
    if n < 200:
        raise ValueError(f"Not enough dates for folds: n={n}")

    # Test block size ~ 1/(n_folds+1) of the data, with guardrails.
    te_size = max(60, int(n / (n_folds + 2)))
    te_size = min(te_size, max(60, int(0.35 * n)))

    folds: List[Tuple[str, str, str, str]] = []
    # last fold ends at last available date
    for k in range(n_folds):
        te_end_idx = n - 1 - (n_folds - 1 - k) * te_size
        te_start_idx = max(0, te_end_idx - te_size + 1)
        tr_end_idx = max(0, te_start_idx - 1)
        # expanding train starts at 0
        tr_start_idx = 0

        # guard: require some train history
        if tr_end_idx < 120:
            continue

        tr_start = dates_sorted[tr_start_idx]
        tr_end = dates_sorted[tr_end_idx]
        te_start = dates_sorted[te_start_idx]
        te_end = dates_sorted[te_end_idx]
        folds.append((tr_start, tr_end, te_start, te_end))

    if not folds:
        raise ValueError("Could not construct any folds (insufficient history after guards)")
    return folds


# -------------------------
# Rule weighting + HEALTH
# -------------------------


def _weight_from_metrics(lift: float, fold_count: int, es5: float, oos_lift_min: float) -> float:
    """Stable weight scale.

    Intuition:
    - Lift above threshold: primary driver.
    - Fold count provides mild stability boost.
    - ES5 penalizes ugly left tail.

    Output is roughly in [0..10] for typical metrics.
    """
    if not np.isfinite(lift):
        return 0.0

    base = max(0.0, lift - oos_lift_min)
    stab = 1.0 + 0.15 * max(0, int(fold_count) - 1)

    # ES5 is typically negative. Penalize only when it is "too" negative.
    tail_pen = 1.0
    if np.isfinite(es5):
        tail_pen = float(np.clip(1.0 + (es5 / 0.10), 0.2, 1.2))  # es5=-0.10 -> ~0.0 clipped to 0.2

    w = base * 20.0 * stab * tail_pen
    return float(max(0.0, w))


def _health_filter_rules(
    df_train: pd.DataFrame,
    rules: Sequence[Any],
    em: Any,
    cfg_event: Any,
    direction: str,
    health: HealthCfg,
) -> Tuple[List[Any], List[Dict[str, Any]]]:
    """Evaluate rules on LAST health.win train dates and retire weak ones.

    Returns: (kept_rules, retired_rows)

    retired_rows fields include signature/support/median/es5 for easy debugging.
    """
    rules_list = list(rules) if rules is not None else []
    if (not health.enabled) or (not rules_list) or df_train.empty:
        return rules_list, []

    # window slice
    df_h = df_train.sort_values(["date", "symbol"]).copy() if "symbol" in df_train.columns else df_train.sort_values(["date"]).copy()
    last_dates = sorted(df_h["date"].unique())[-int(health.win) :]
    df_h = df_h[df_h["date"].isin(last_dates)].copy()

    if df_h.empty:
        return rules_list, []

    fwd_col = f"fwd_{int(cfg_event.fwd_days)}d_ret" if hasattr(cfg_event, "fwd_days") else "fwd_5d_ret"
    event_col = "event_up" if direction == "long" else "event_dn"

    kept: List[Any] = []
    retired: List[Dict[str, Any]] = []

    for r in rules_list:
        # wrapper-safe: accept Rule, tuple, dict
        rr = r
        if isinstance(r, tuple) and len(r) >= 1:
            rr = r[0]
        if isinstance(r, dict) and ("rule" in r):
            rr = r["rule"]

        st = em.score_rule_event(df_h, rr, event_col=event_col, fwd_col=fwd_col)
        if st is None:
            retired.append({"direction": direction, "signature": getattr(rr, "rule_id", "?"), "support": 0, "median": float("nan"), "es5": float("nan")})
            continue

        support = int(getattr(st, "support", 0))
        med = float(getattr(st, "median_signed", float("nan")))
        es5 = float(getattr(st, "es5_signed", float("nan")))
        sig = str(getattr(st, "signature", getattr(rr, "rule_id", "?")))

        ok_n = support >= int(health.min_n)
        ok_med = (not np.isfinite(med)) or (med >= float(health.med_min))
        ok_es5 = (not np.isfinite(es5)) or (es5 >= float(health.es5_min))

        if ok_n and ok_med and ok_es5:
            kept.append(rr)
        else:
            retired.append({"direction": direction, "signature": sig, "support": support, "median": med, "es5": es5})

    return kept, retired


# -------------------------
# Main
# -------------------------


def main() -> int:
    cfg = load_cfg()

    # ensure repo root on path for root-module fallbacks
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    src = root / "src"
    if src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))

    load_aggs, to_daily_index = _import_ingest()
    FeatureCfgCls, build_features_fn = _import_features()
    em = _import_event_mining()

    # event config from event_mining
    cfg_event = em.EventMiningConfig(
        fwd_days=_env_int("FWD_DAYS", 5),
        k_sigma=_env_float("K_SIGMA", 1.5),
        sigma_lookback=_env_int("SIGMA_LOOKBACK", 60),
        max_rules_try=_env_int("RULES_TRY", 6000),
        max_rules_keep=_env_int("RULES_KEEP", 800),
        min_support=_env_int("MIN_SUPPORT", 200),
        min_event_hits=_env_int("MIN_EVENT_HITS", 30),
        perm_trials=_env_int("PERM_TRIALS", 50),
        perm_topk=_env_int("PERM_TOPK", 20),
        perm_gate_enabled=_env_bool("PERM_GATE", True),
        perm_gate_margin=_env_float("PERM_MARGIN", 0.15),
    )

    # ---- CFG prints (compact) ----
    _p(f"[CFG] vendor={cfg.vendor} dataset_root={cfg.dataset_root}")
    _p(f"[CFG] universe=auto start={cfg.start} end={cfg.end}")
    _p(f"[CFG] event: fwd_days={cfg_event.fwd_days} k_sigma={cfg_event.k_sigma} sigma_lookback={cfg_event.sigma_lookback}")
    _p(
        f"[CFG] rules: try={cfg_event.max_rules_try} keep={cfg_event.max_rules_keep} "
        f"min_support={cfg_event.min_support} min_event_hits={cfg_event.min_event_hits}"
    )
    _p(
        f"[CFG] perm: trials={cfg_event.perm_trials} topk={cfg_event.perm_topk} "
        f"gate={int(bool(cfg_event.perm_gate_enabled))} margin={cfg_event.perm_gate_margin}"
    )
    _p(f"[CFG] OOS filter: support>={cfg.oos_support_min} lift>={cfg.oos_lift_min}")
    _p(f"[CFG] rank: K={cfg.K}")
    _p(
        f"[CFG] health: enabled={int(cfg.health.enabled)} win={cfg.health.win} min_n={cfg.health.min_n} "
        f"med_min={cfg.health.med_min:.4f} es5_min={cfg.health.es5_min:.4f}"
    )
    _p(
        f"[CFG] recency: MUST_PASS_LATEST={int(cfg.recency.must_pass_latest)} "
        f"LATEST_ONLY={int(cfg.recency.latest_only)}"
    )

    # ---- Load universe ----
    if not cfg.dataset_root.exists():
        raise FileNotFoundError(f"Missing dataset_root: {cfg.dataset_root}")

    symbols = sorted([p.name for p in cfg.dataset_root.iterdir() if p.is_dir()])
    if not symbols:
        raise RuntimeError(f"No symbols found under {cfg.dataset_root}")

    # pooled daily df
    rows: List[pd.DataFrame] = []
    for sym in symbols:
        res = load_aggs(cfg.dataset_root, sym, cfg.tf, cfg.start, cfg.end)
        d0 = to_daily_index(res.df)
        # Expect Polygon-style columns: t,o,h,l,c,v
        d0["symbol"] = sym
        # keep only daily bars columns + date
        d = d0[["date", "o", "h", "l", "c", "v", "symbol"]].copy() if "c" in d0.columns else d0.copy()
        d = _ensure_ohlc(d)
        rows.append(d)

    df = pd.concat(rows, axis=0, ignore_index=True)
    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)

    # features
    # Some builders need cfg passed as instance.
    feat_cfg = FeatureCfgCls()  # type: ignore

    out_frames: List[pd.DataFrame] = []
    for sym, d in df.groupby("symbol", sort=True):
        dd = d.sort_values("date").reset_index(drop=True)
        # Provide both naming schemes to feature builders that expect close/open...
        if "close" not in dd.columns:
            dd["close"] = dd["c"]
        if "open" not in dd.columns:
            dd["open"] = dd["o"]
        if "high" not in dd.columns:
            dd["high"] = dd["h"]
        if "low" not in dd.columns:
            dd["low"] = dd["l"]
        if "volume" not in dd.columns:
            dd["volume"] = dd["v"]

        ff = build_features_fn(dd, feat_cfg)
        # Normalize back to expected event_mining columns
        if "c" not in ff.columns and "close" in ff.columns:
            ff["c"] = ff["close"]
        if "date" not in ff.columns:
            ff["date"] = dd["date"].astype(str)
        if "symbol" not in ff.columns:
            ff["symbol"] = sym
        out_frames.append(ff)

    df_feat = pd.concat(out_frames, axis=0, ignore_index=True)
    df_feat = df_feat.sort_values(["date", "symbol"]).reset_index(drop=True)

    # ensure required forward return column exists
    fwd_col = f"fwd_{int(cfg_event.fwd_days)}d_ret"
    if fwd_col not in df_feat.columns:
        raise ValueError(f"Missing required fwd column: {fwd_col}. Have={list(df_feat.columns)[:30]}...")

    # label events using event_mining (adds event_up/dn)
    df_feat = em.label_events(df_feat, cfg_event)

    # base rates
    base_up = float(pd.to_numeric(df_feat["event_up"], errors="coerce").fillna(0).mean())
    base_dn = float(pd.to_numeric(df_feat["event_dn"], errors="coerce").fillna(0).mean())

    dates = sorted(df_feat["date"].unique())
    _p(f"[DATA] pooled rows={len(df_feat)} dates={len(dates)} symbols={len(symbols)}")
    _p(f"[DATA] base_rate up={base_up*100:.4f}% dn={base_dn*100:.4f}%")

    # Feature columns for mining: numeric, exclude obvious non-features
    drop_cols = {"date", "symbol", "t", "dt_utc", "event_up", "event_dn", "sigma_roll", "event_thr"}
    drop_cols.update({fwd_col})
    feature_cols = [c for c in df_feat.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df_feat[c])]
    # remove raw ohlc if present (too trivial)
    for c in ["o", "h", "l", "c", "v", "open", "high", "low", "close", "volume"]:
        if c in feature_cols:
            feature_cols.remove(c)

    _p(f"[DBG] feature_cols_n={len(feature_cols)}")

    folds = _make_folds(dates, cfg.fold_count)

    # Per-fold mining + OOS eval
    fold_pass_rows: List[pd.DataFrame] = []
    fold_pass_rules: Dict[int, Dict[str, Any]] = {}

    for i, (tr_s, tr_e, te_s, te_e) in enumerate(folds, start=1):
        _banner()
        _p(f"[LIB/FOLD {i}] train={tr_s}..{tr_e}  test={te_s}..{te_e}")

        d_tr = df_feat[(df_feat["date"] >= tr_s) & (df_feat["date"] <= tr_e)].copy()
        d_te = df_feat[(df_feat["date"] >= te_s) & (df_feat["date"] <= te_e)].copy()
        _p(f"[LIB/FOLD {i}] train rows={len(d_tr)} test rows={len(d_te)}")

        # Mine rules (IMPORTANT: signature is (df_train, feature_cols, cfg, direction=None))
        rules, stats, perm = em.mine_event_rules(d_tr, feature_cols, cfg_event, direction=None)

        perm_p95 = perm.get("perm_topk_p95", float("nan"))
        status = "ok" if np.isfinite(perm_p95) else "unavailable"

        _p(f"[LIB/FOLD {i}] mined_rules={len(rules)} perm_p95={(perm_p95 if np.isfinite(perm_p95) else 'NA')} status={status}")

        # OOS eval on test
        oos = em.evaluate_rules_oos(d_te, rules, cfg_event)
        if oos.empty:
            _p(f"[LIB/FOLD {i}] OOS eval=0 pass(lift>={cfg.oos_lift_min})=0")
            fold_pass_rows.append(pd.DataFrame())
            fold_pass_rules[i] = {"rules": [], "oos": pd.DataFrame(), "train": d_tr, "test": d_te}
            continue

        # OOS filter
        oos = oos.copy()
        oos["fold"] = i
        oos = oos[pd.to_numeric(oos["support"], errors="coerce").fillna(0) >= int(cfg.oos_support_min)].copy()
        oos = oos[pd.to_numeric(oos["lift"], errors="coerce").fillna(0) >= float(cfg.oos_lift_min)].copy()

        _p(f"[LIB/FOLD {i}] OOS eval={len(rules)} pass(lift>={cfg.oos_lift_min})={len(oos)}")

        fold_pass_rows.append(oos)

        # Keep pass rules list by rule_id
        pass_ids = set(oos["rule_id"].astype(str).tolist())
        pass_rules = [r for r in rules if getattr(r, "rule_id", "") in pass_ids]
        fold_pass_rules[i] = {"rules": pass_rules, "oos": oos, "train": d_tr, "test": d_te}

    _banner()
    _p("[GLOBAL] Rule Library (OOS-filtered across folds)")

    # Aggregate across folds by canonical signature
    all_oos = pd.concat([x for x in fold_pass_rows if x is not None and not x.empty], axis=0, ignore_index=True) if fold_pass_rows else pd.DataFrame()

    if all_oos.empty:
        _p("[GLOBAL] Fold pass counts: " + " | ".join([f"F{i}: 0/0" for i in range(1, len(folds) + 1)]))
        _p("[GLOBAL] library empty (after stability/recency filters).")
        if cfg.health.enabled:
            _p("[HEALTH] enabled=1 but no rules survived GLOBAL selection -> nothing to filter.")
        return 0

    # fold pass counts
    counts = []
    for i in range(1, len(folds) + 1):
        total_i = len(fold_pass_rules.get(i, {}).get("rules", []))
        counts.append(f"F{i}: {total_i}/{total_i}")
    _p("[GLOBAL] Fold pass counts: " + " | ".join(counts))

    # group by signature
    g = all_oos.groupby(["signature", "direction"], as_index=False).agg(
        fold_count=("fold", "nunique"),
        avg_lift=("lift", "mean"),
        avg_es5=("es5_signed", "mean"),
        avg_med=("median_signed", "mean"),
        avg_support=("support", "mean"),
    )

    latest_fold = max(range(1, len(folds) + 1))

    # recency requirements
    if cfg.recency.latest_only:
        # keep only signatures that appear in latest fold pass set
        latest_oos = all_oos[all_oos["fold"] == latest_fold].copy()
        latest_sig = set(latest_oos["signature"].astype(str).tolist())
        g = g[g["signature"].astype(str).isin(latest_sig)].copy()
        _p(f"[GLOBAL][RECENCY] LATEST_ONLY=1 -> using latest fold rules only (fold={latest_fold})")

    min_fc = 2
    if cfg.recency.must_pass_latest:
        # keep fold_count>=2, but if empty, use latest fold only explicitly
        g2 = g[g["fold_count"] >= 2].copy()
        if g2.empty:
            _p("[GLOBAL][FAILSAFE] library empty under MUST_PASS_LATEST -> using latest-fold pass rules only (explicit)")
            min_fc = 1
        else:
            g = g2

    if min_fc > 1:
        g = g[g["fold_count"] >= min_fc].copy()

    if g.empty:
        _p("[GLOBAL] library empty (after stability/recency filters).")
        if cfg.health.enabled:
            _p("[HEALTH] enabled=1 but no rules survived GLOBAL selection -> nothing to filter.")
        return 0

    # pick representative rule per signature from latest fold if available else best lift
    # Build a mapping signature->Rule by scanning fold rules
    sig_to_rule: Dict[Tuple[str, str], Any] = {}
    # Prefer latest fold rules
    latest_rules: List[Any] = fold_pass_rules.get(latest_fold, {}).get("rules", [])
    # helper: canonical signature is stored in OOS rows
    latest_oos = all_oos[all_oos["fold"] == latest_fold].copy()
    rid_to_sig_dir = {
        str(rid): (str(sig), str(d))
        for rid, sig, d in zip(latest_oos["rule_id"].astype(str), latest_oos["signature"].astype(str), latest_oos["direction"].astype(str))
    }
    for r in latest_rules:
        rid = str(getattr(r, "rule_id", ""))
        if rid in rid_to_sig_dir:
            sig, d = rid_to_sig_dir[rid]
            key = (sig, d)
            if key not in sig_to_rule:
                sig_to_rule[key] = r

    # Fallback: scan all folds
    if len(sig_to_rule) < len(g):
        for fi in range(1, len(folds) + 1):
            o = fold_pass_rules.get(fi, {}).get("oos")
            rs = fold_pass_rules.get(fi, {}).get("rules", [])
            if o is None or o.empty:
                continue
            rid_to_sig_dir2 = {
                str(rid): (str(sig), str(d))
                for rid, sig, d in zip(o["rule_id"].astype(str), o["signature"].astype(str), o["direction"].astype(str))
            }
            for r in rs:
                rid = str(getattr(r, "rule_id", ""))
                if rid in rid_to_sig_dir2:
                    sig, d = rid_to_sig_dir2[rid]
                    key = (sig, d)
                    if key not in sig_to_rule:
                        sig_to_rule[key] = r

    # Create library table with weights
    lib_rows: List[Dict[str, Any]] = []
    for _, row in g.iterrows():
        sig = str(row["signature"])
        d = str(row["direction"])
        fc = int(row["fold_count"])
        lift = float(row["avg_lift"])
        es5 = float(row["avg_es5"]) if np.isfinite(row["avg_es5"]) else float("nan")
        w = _weight_from_metrics(lift, fc, es5, cfg.oos_lift_min)
        lib_rows.append({"signature": sig, "direction": d, "fold_count": fc, "avg_lift": lift, "avg_es5": es5, "weight": w})

    lib = pd.DataFrame(lib_rows)

    # Split
    lib_long = lib[lib["direction"] == "long"].copy()
    lib_short = lib[lib["direction"] == "short"].copy()

    _p(f"[GLOBAL] library_size={len(lib)} (min_fc={min_fc})")
    _p(f"[GLOBAL] long_kept={len(lib_long)} short_kept={len(lib_short)}")

    # --------------------
    # Rank per fold
    # --------------------

    for i, (tr_s, tr_e, te_s, te_e) in enumerate(folds, start=1):
        _banner()
        _p(f"[RANK/FOLD {i}] OOS window={te_s}..{te_e}  rows={len(fold_pass_rules[i]['test'])}")

        df_tr = fold_pass_rules[i]["train"].copy()
        df_te = fold_pass_rules[i]["test"].copy()

        # Build fold rules from global library: choose rules by signature mapping
        fold_long_rules: List[Any] = []
        for _, r in lib_long.iterrows():
            key = (str(r["signature"]), "long")
            rr = sig_to_rule.get(key)
            if rr is not None:
                fold_long_rules.append(rr)

        fold_short_rules: List[Any] = []
        for _, r in lib_short.iterrows():
            key = (str(r["signature"]), "short")
            rr = sig_to_rule.get(key)
            if rr is not None:
                fold_short_rules.append(rr)

        # HEALTH
        _p(
            f"[HEALTH] enabled={int(cfg.health.enabled)} win={cfg.health.win} min_n={cfg.health.min_n} "
            f"med_min={cfg.health.med_min:.4f} es5_min={cfg.health.es5_min:.4f}"
        )

        retired_print_max = 5

        fold_long_rules_kept, retired_l = _health_filter_rules(df_tr, fold_long_rules, em, cfg_event, "long", cfg.health)
        fold_short_rules_kept, retired_s = _health_filter_rules(df_tr, fold_short_rules, em, cfg_event, "short", cfg.health)

        _p(f"[HEALTH] long: before={len(fold_long_rules)} after={len(fold_long_rules_kept)} retired={len(retired_l)}")
        for rr in retired_l[:retired_print_max]:
            _p(
                f"[HEALTH] retired_long sig={rr['signature']} support={rr['support']} "
                f"med={rr['median']:.4f} es5={rr['es5']:.4f}"
            )

        _p(f"[HEALTH] short: before={len(fold_short_rules)} after={len(fold_short_rules_kept)} retired={len(retired_s)}")
        for rr in retired_s[:retired_print_max]:
            _p(
                f"[HEALTH] retired_short sig={rr['signature']} support={rr['support']} "
                f"med={rr['median']:.4f} es5={rr['es5']:.4f}"
            )

        fold_long_rules = fold_long_rules_kept
        fold_short_rules = fold_short_rules_kept

        _p(f"[RANK/FOLD {i}] fold_rules kept: long={len(fold_long_rules)} short={len(fold_short_rules)}")

        if df_te.empty or (not fold_long_rules and not fold_short_rules):
            _p(f"[RANK/FOLD {i}] scored_rows(nonzero)=0/{len(df_te)} fires_long=0 fires_short=0")
            _p(f"[RANK/FOLD {i}] LONG top-{cfg.K} fwd_{cfg_event.fwd_days}d: mean=0.0000 med=0.0000 p>0=0.00%")
            continue

        # Score each row by sum(weights of firing rules)
        # Build weight maps by canonical signature
        w_long = {str(r["signature"]): float(r["weight"]) for _, r in lib_long.iterrows()}
        w_short = {str(r["signature"]): float(r["weight"]) for _, r in lib_short.iterrows()}

        long_score = np.zeros(len(df_te), dtype=float)
        short_score = np.zeros(len(df_te), dtype=float)
        fires_long = 0
        fires_short = 0

        # Precompute masks for speed
        for rr in fold_long_rules:
            st = em.score_rule_event(df_te, rr, event_col="event_up", fwd_col=fwd_col)
            sig = str(getattr(st, "signature", getattr(rr, "rule_id", ""))) if st is not None else ""
            w = float(w_long.get(sig, 0.0))
            if w <= 0.0:
                continue
            m = em._apply_rule_mask(df_te, rr).to_numpy(dtype=bool)  # type: ignore
            fires_long += int(np.sum(m))
            long_score[m] += w

        for rr in fold_short_rules:
            st = em.score_rule_event(df_te, rr, event_col="event_dn", fwd_col=fwd_col)
            sig = str(getattr(st, "signature", getattr(rr, "rule_id", ""))) if st is not None else ""
            w = float(w_short.get(sig, 0.0))
            if w <= 0.0:
                continue
            m = em._apply_rule_mask(df_te, rr).to_numpy(dtype=bool)  # type: ignore
            fires_short += int(np.sum(m))
            short_score[m] += w

        net_score = long_score - short_score
        nonzero = int(np.sum(net_score != 0.0))
        _p(f"[RANK/FOLD {i}] scored_rows(nonzero)={nonzero}/{len(df_te)} fires_long={fires_long} fires_short={fires_short}")

        # Evaluate top-K each date
        fwd = pd.to_numeric(df_te[fwd_col], errors="coerce").to_numpy(dtype=float)
        df_te2 = df_te[["date", "symbol"]].copy()
        df_te2["net"] = net_score
        df_te2["fwd"] = fwd

        per_date = []
        for d, g2 in df_te2.groupby("date", sort=True):
            g2 = g2.sort_values("net", ascending=False)
            top = g2.head(int(cfg.K))
            # average forward return across selected symbols
            xs = pd.to_numeric(top["fwd"], errors="coerce").dropna().to_numpy(dtype=float)
            if xs.size:
                per_date.append(float(np.mean(xs)))

        arr = np.array(per_date, dtype=float)
        mean_v = float(np.mean(arr)) if arr.size else 0.0
        med_v = float(np.median(arr)) if arr.size else 0.0
        ppos_v = float(np.mean(arr > 0.0)) if arr.size else 0.0

        _p(
            f"[RANK/FOLD {i}] LONG top-{cfg.K} fwd_{cfg_event.fwd_days}d: "
            f"mean={mean_v:.4f} med={med_v:.4f} p>0={ppos_v*100:.2f}%"
        )

    _banner()
    _p("[DONE] Ranker MVP completed.")
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except Exception as e:
        _p("[ERROR] Unhandled exception:")
        _p(str(e))
        _p(traceback.format_exc())
        rc = 1
    _p(f"\n[EXIT] code={rc}")
    # Double-click runnable: always wait.
    try:
        input("Press Enter to exit...")
    except Exception:
        pass
    raise SystemExit(rc)
