from __future__ import annotations

"""Python-Edge :: Ranker MVP (daily)

This script is intentionally self-contained and robust to module layout.
It mines quantile-threshold event rules per walk-forward fold, applies OOS filters,
builds a GLOBAL library with optional RECENCY gating, applies optional HEALTH
(retire) filter, then scores each OOS fold by summing rule weights.

Project constraints:
- No silent fail-open: any fallback is printed explicitly.
- Compact output by default.
- Double-click runnable: waits for input() at the end.

Environment variables (key ones):
  DATASET_ROOT   path to massive dataset root
  DATA_START     YYYY-MM-DD
  DATA_END       YYYY-MM-DD
  TF             timeframe (default 1D)
  FOLD_COUNT     number of folds (default 3)

  # Event mining
  FWD_DAYS       default 5
  SIGMA_LOOKBACK default 60
  K_SIGMA        default 1.5
  RULES_TRY      default 6000
  RULES_KEEP     default 800
  MIN_SUPPORT    default 200
  MIN_EVENT_HITS default 30

  # OOS filters
  OOS_LIFT_MIN      default 1.35
  OOS_SUPPORT_MIN   default 200

  # Rank
  RANK_K         default 3

  # Permutation display (event_mining handles gate internally)
  PERM_TRIALS    default 50
  PERM_TOPK      default 20
  PERM_GATE      default 1
  PERM_MARGIN    default 0.15

  # HEALTH (Phase 2)
  HEALTH         0/1
  HEALTH_WIN     default 60
  HEALTH_MIN_N   default 50
  HEALTH_MED_MIN default 0.0
  HEALTH_ES5_MIN default -0.08

  # RECENCY
  MUST_PASS_LATEST 0/1   (rule must pass OOS in latest fold)
  LATEST_ONLY      0/1   (use only latest fold rules)

  # Output
  DEBUG_DAILY    0/1   (print first few daily top-K lines)

Exit codes:
  0 success
  1 failure
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


# =========================
# Printing helpers
# =========================

def _p(msg: str) -> None:
    print(msg, flush=True)


def _banner() -> None:
    _p("=" * 80)


# =========================
# Env helpers
# =========================

def _env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return default
    return str(v)


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


# =========================
# Robust imports
# =========================

def _import_ingest() -> Tuple[Any, Any]:
    try:
        from python_edge.data.ingest_aggs import load_aggs, to_daily_index  # type: ignore

        return load_aggs, to_daily_index
    except Exception:
        pass

    try:
        from ingest_aggs import load_aggs, to_daily_index  # type: ignore

        _p("[DBG] import ingest_aggs from repo-root")
        return load_aggs, to_daily_index
    except Exception as e:
        raise ImportError("Cannot import ingest_aggs (package or repo-root)") from e


def _import_features() -> Tuple[Any, Any]:
    # Repo-root (current) naming:
    try:
        from python_edge.features.build_features_daily import FeatureConfig, build_features_daily  # type: ignore

        return FeatureConfig, build_features_daily
    except Exception:
        pass

    try:
        from build_features_daily import FeatureConfig, build_features_daily  # type: ignore

        _p("[DBG] import build_features_daily from repo-root")
        return FeatureConfig, build_features_daily
    except Exception as e:
        raise ImportError("Cannot import build_features_daily FeatureConfig/build_features_daily") from e


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
        raise ImportError("Cannot import event_mining (package or repo-root)") from e


# =========================
# Config
# =========================


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
    vendor: str
    dataset_root: Path
    start: str
    end: str
    tf: str

    # folds
    fold_count: int

    # event config
    fwd_days: int
    sigma_lookback: int
    k_sigma: float

    # mining
    rules_try: int
    rules_keep: int
    min_support: int
    min_event_hits: int

    # perm
    perm_trials: int
    perm_topk: int
    perm_gate: bool
    perm_margin: float

    # OOS filter
    oos_lift_min: float
    oos_support_min: int

    # short payoff-gate
    short_mean_min: float
    short_p_pos_min: float

    # rank
    K: int

    health: HealthCfg
    recency: RecencyCfg

    debug_daily: bool


def load_cfg() -> RunCfg:
    root = Path(__file__).resolve().parents[1]
    ds = Path(_env_str("DATASET_ROOT", str(root / "data" / "raw" / "massive_dataset")))

    start = _env_str("DATA_START", "2023-01-01")
    end = _env_str("DATA_END", "2026-02-28")
    tf = _env_str("TF", "1D")

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
        tf=tf,
        fold_count=_env_int("FOLD_COUNT", 3),
        fwd_days=_env_int("FWD_DAYS", 5),
        sigma_lookback=_env_int("SIGMA_LOOKBACK", 60),
        k_sigma=_env_float("K_SIGMA", 1.5),
        rules_try=_env_int("RULES_TRY", 6000),
        rules_keep=_env_int("RULES_KEEP", 800),
        min_support=_env_int("MIN_SUPPORT", 200),
        min_event_hits=_env_int("MIN_EVENT_HITS", 30),
        perm_trials=_env_int("PERM_TRIALS", 50),
        perm_topk=_env_int("PERM_TOPK", 20),
        perm_gate=_env_bool("PERM_GATE", True),
        perm_margin=_env_float("PERM_MARGIN", 0.15),
        oos_lift_min=_env_float("OOS_LIFT_MIN", 1.35),
        oos_support_min=_env_int("OOS_SUPPORT_MIN", 200),
        short_mean_min=_env_float("SHORT_MEAN_MIN", 0.0),
        short_p_pos_min=_env_float("SHORT_PPOS_MIN", 0.5),
        K=_env_int("RANK_K", 3),
        health=health,
        recency=rec,
        debug_daily=_env_bool("DEBUG_DAILY", False),
    )


# =========================
# Data normalization
# =========================

def _ensure_close_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure we have both Polygon-style (o/h/l/c/v) and human-style (open/high/low/close/volume)."""
    if df.empty:
        return df
    out = df.copy()
    # Map to c/o/h/l/v if needed
    rename_to_chl: Dict[str, str] = {}
    if "close" in out.columns and "c" not in out.columns:
        rename_to_chl["close"] = "c"
    if "open" in out.columns and "o" not in out.columns:
        rename_to_chl["open"] = "o"
    if "high" in out.columns and "h" not in out.columns:
        rename_to_chl["high"] = "h"
    if "low" in out.columns and "l" not in out.columns:
        rename_to_chl["low"] = "l"
    if "volume" in out.columns and "v" not in out.columns:
        rename_to_chl["volume"] = "v"
    if rename_to_chl:
        out = out.rename(columns=rename_to_chl)

    need = ["date", "symbol", "c"]
    missing = [c for c in need if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns after normalization: {missing}. Have={list(out.columns)}")

    # Create human-friendly aliases expected by build_features_daily
    if "close" not in out.columns:
        out["close"] = out["c"]
    if "open" not in out.columns and "o" in out.columns:
        out["open"] = out["o"]
    if "high" not in out.columns and "h" in out.columns:
        out["high"] = out["h"]
    if "low" not in out.columns and "l" in out.columns:
        out["low"] = out["l"]
    if "volume" not in out.columns and "v" in out.columns:
        out["volume"] = out["v"]

    out["date"] = out["date"].astype(str)
    out["symbol"] = out["symbol"].astype(str)
    out = out.sort_values(["symbol", "date"]).reset_index(drop=True)
    return out


def _add_fwd_returns(df: pd.DataFrame, fwd_days: int) -> pd.DataFrame:
    out = df.copy()
    out[f"fwd_{fwd_days}d_ret"] = (
        out.groupby("symbol", sort=False)["close"].shift(-fwd_days) / out["close"] - 1.0
    )
    return out


# =========================
# Folds
# =========================

def _make_folds(dates: List[str], n_folds: int) -> List[Tuple[str, str, str, str]]:
    """(train_start, train_end, test_start, test_end) with expanding train and rolling test blocks.

    Fold N is the newest fold.
    """
    if n_folds < 1:
        raise ValueError("FOLD_COUNT must be >= 1")

    ds = sorted(set(dates))
    n = len(ds)
    if n < 250:
        raise ValueError(f"Not enough dates for folds: n={n}")

    te_size = max(60, int(n / (n_folds + 2)))
    te_size = min(te_size, max(60, int(0.35 * n)))

    folds: List[Tuple[str, str, str, str]] = []
    for k in range(n_folds):
        te_end_idx = n - 1 - (n_folds - 1 - k) * te_size
        te_start_idx = max(0, te_end_idx - te_size + 1)
        tr_end_idx = max(0, te_start_idx - 1)
        if tr_end_idx < 120:
            continue
        folds.append((ds[0], ds[tr_end_idx], ds[te_start_idx], ds[te_end_idx]))

    if not folds:
        raise ValueError("Could not construct any folds (insufficient history after guards)")
    return folds


# =========================
# Weights + HEALTH
# =========================

def _weight_from_metrics(lift: float, fold_count: int, es5: float, oos_lift_min: float) -> float:
    """Stable weight scale (roughly 0..10).

    - base: lift above threshold
    - stab: mild fold_count boost
    - tail_pen: penalize bad ES5
    """
    if not np.isfinite(lift):
        return 0.0
    base = max(0.0, lift - oos_lift_min)
    stab = 1.0 + 0.15 * max(0, int(fold_count) - 1)
    tail_pen = 1.0
    if np.isfinite(es5):
        tail_pen = float(np.clip(1.0 + (es5 / 0.10), 0.2, 1.2))
    return float(max(0.0, base * 20.0 * stab * tail_pen))


def _health_filter_rules(
    df_train: pd.DataFrame,
    rules: Sequence[Any],
    em: Any,
    cfg_event: Any,
    direction: str,
    health: HealthCfg,
) -> Tuple[List[Any], List[Dict[str, Any]]]:
    if (not health.enabled) or (not rules) or df_train.empty:
        return list(rules), []

    # last window of dates (freshness)
    dates = sorted(set(df_train["date"].astype(str).tolist()))
    win = int(max(10, health.win))
    keep_dates = set(dates[-win:])
    df_h = df_train[df_train["date"].astype(str).isin(keep_dates)].copy()

    event_col = "event_up" if direction == "long" else "event_dn"
    fwd_col = f"fwd_{cfg_event.fwd_days}d_ret"

    kept: List[Any] = []
    retired: List[Dict[str, Any]] = []

    for r in rules:
        st = em.score_rule_event(df_h, r, event_col=event_col, fwd_col=fwd_col)
        if st is None:
            retired.append({"signature": getattr(r, "rule_id", "?"), "support": 0, "median": float("nan"), "es5": float("nan")})
            continue

        if int(st.support) < int(health.min_n):
            retired.append({"signature": st.signature, "support": int(st.support), "median": float(st.median_signed), "es5": float(st.es5_signed)})
            continue
        if np.isfinite(st.median_signed) and float(st.median_signed) < float(health.med_min):
            retired.append({"signature": st.signature, "support": int(st.support), "median": float(st.median_signed), "es5": float(st.es5_signed)})
            continue
        if np.isfinite(st.es5_signed) and float(st.es5_signed) < float(health.es5_min):
            retired.append({"signature": st.signature, "support": int(st.support), "median": float(st.median_signed), "es5": float(st.es5_signed)})
            continue

        kept.append(r)

    return kept, retired


# =========================
# Ranking
# =========================

def _score_fold(
    df_te: pd.DataFrame,
    long_rules: Sequence[Any],
    short_rules: Sequence[Any],
    weights_by_sig: Dict[str, float],
    em: Any,
    K: int,
) -> Tuple[float, float, float, int, int, int]:
    """Return (mean, median, ppos, scored_rows, fires_long, fires_short) for top-K daily picks."""

    if df_te.empty:
        return 0.0, 0.0, 0.0, 0, 0, 0

    n = len(df_te)
    long_score = np.zeros(n, dtype=float)
    short_score = np.zeros(n, dtype=float)

    fires_long = 0
    fires_short = 0

    # precompute for speed
    for r in long_rules:
        sig = em._canonical_signature(r.direction, r.conds)  # type: ignore
        w = float(weights_by_sig.get(sig, 0.0))
        if w <= 0.0:
            continue
        m = em._apply_rule_mask(df_te, r).to_numpy(dtype=bool)  # type: ignore
        if m.any():
            long_score[m] += w
            fires_long += int(m.sum())

    for r in short_rules:
        sig = em._canonical_signature(r.direction, r.conds)  # type: ignore
        w = float(weights_by_sig.get(sig, 0.0))
        if w <= 0.0:
            continue
        m = em._apply_rule_mask(df_te, r).to_numpy(dtype=bool)  # type: ignore
        if m.any():
            short_score[m] += w
            fires_short += int(m.sum())

    net = long_score - short_score

    # rank per date across symbols
    df_tmp = df_te[["date", "symbol", f"fwd_{_CFG.fwd_days}d_ret"]].copy()  # noqa: F821
    df_tmp["net"] = net

    scored_rows = int(np.sum(net != 0.0))

    # pick top-K per date
    picks = (
        df_tmp.sort_values(["date", "net"], ascending=[True, False])
        .groupby("date", sort=False)
        .head(int(max(1, K)))
    )

    xs = pd.to_numeric(picks[f"fwd_{_CFG.fwd_days}d_ret"], errors="coerce").to_numpy(dtype=float)  # noqa: F821
    xs = xs[np.isfinite(xs)]
    if xs.size == 0:
        return 0.0, 0.0, 0.0, scored_rows, fires_long, fires_short

    mean = float(np.mean(xs))
    med = float(np.median(xs))
    ppos = float(np.mean(xs > 0.0))
    return mean, med, ppos, scored_rows, fires_long, fires_short


# =========================
# Main
# =========================

# global cfg for a couple of helper lines (kept minimal)
_CFG: RunCfg


def main() -> int:
    global _CFG

    cfg = load_cfg()
    _CFG = cfg

    load_aggs, to_daily_index = _import_ingest()
    FeatureConfig, build_features_daily = _import_features()
    em = _import_event_mining()

    _p(f"[CFG] vendor={cfg.vendor} dataset_root={cfg.dataset_root}")
    _p(f"[CFG] start={cfg.start} end={cfg.end} tf={cfg.tf} folds={cfg.fold_count}")
    _p(f"[CFG] event: fwd_days={cfg.fwd_days} k_sigma={cfg.k_sigma} sigma_lookback={cfg.sigma_lookback}")
    _p(
        "[CFG] rules: try={t} keep={k} min_support={ms} min_event_hits={meh}".format(
            t=cfg.rules_try, k=cfg.rules_keep, ms=cfg.min_support, meh=cfg.min_event_hits
        )
    )
    _p(
        "[CFG] perm: trials={t} topk={k} gate={g} margin={m}".format(
            t=cfg.perm_trials, k=cfg.perm_topk, g=int(cfg.perm_gate), m=cfg.perm_margin
        )
    )
    _p(f"[CFG] OOS filter: support>={cfg.oos_support_min} lift>={cfg.oos_lift_min}")
    _p(f"[CFG] short payoff-gate: mean>{cfg.short_mean_min} p>0>{cfg.short_p_pos_min}")
    _p(f"[CFG] rank: K={cfg.K}")
    _p(
        "[CFG] health: enabled={e} win={w} min_n={n} med_min={mm:.4f} es5_min={es:.4f}".format(
            e=int(cfg.health.enabled), w=cfg.health.win, n=cfg.health.min_n, mm=cfg.health.med_min, es=cfg.health.es5_min
        )
    )
    _p(
        "[CFG] recency: MUST_PASS_LATEST={a} LATEST_ONLY={b}".format(
            a=int(cfg.recency.must_pass_latest), b=int(cfg.recency.latest_only)
        )
    )

    # Universe from folders
    if not cfg.dataset_root.exists():
        raise FileNotFoundError(f"Missing dataset_root: {cfg.dataset_root}")
    symbols = sorted([p.name for p in cfg.dataset_root.iterdir() if p.is_dir()])
    if not symbols:
        raise RuntimeError(f"No symbol folders under {cfg.dataset_root}")

    dfs: List[pd.DataFrame] = []
    for sym in symbols:
        r = load_aggs(cfg.dataset_root, sym, cfg.tf, cfg.start, cfg.end)
        d = to_daily_index(r.df)
        if d.empty:
            continue
        # keep essential columns
        d = d[["date", "o", "h", "l", "c", "v"]].copy()
        d["symbol"] = sym
        dfs.append(d)

    if not dfs:
        raise RuntimeError("No data loaded (all symbols empty)")

    df = pd.concat(dfs, axis=0, ignore_index=True)
    df = _ensure_close_cols(df)

    # add forward returns
    df = _add_fwd_returns(df, cfg.fwd_days)

    # build features (ret_1d + percentiles)
    df = build_features_daily(df, FeatureConfig())

    # label events (needs ret_1d and fwd_{N}d_ret)
    cfg_event = em.EventMiningConfig(
        fwd_days=cfg.fwd_days,
        sigma_lookback=cfg.sigma_lookback,
        k_sigma=cfg.k_sigma,
        max_rules_try=cfg.rules_try,
        max_rules_keep=cfg.rules_keep,
        min_support=cfg.min_support,
        min_event_hits=cfg.min_event_hits,
        perm_trials=cfg.perm_trials,
        perm_topk=cfg.perm_topk,
        perm_gate_enabled=bool(cfg.perm_gate),
        perm_gate_margin=cfg.perm_margin,
    )
    df = em.label_events(df, cfg_event)

    # Drop rows that cannot participate
    need_cols = ["date", "symbol", "ret_1d", f"fwd_{cfg.fwd_days}d_ret", "event_up", "event_dn"]
    df0 = df.dropna(subset=[c for c in need_cols if c in df.columns]).copy()

    dates = sorted(set(df0["date"].astype(str).tolist()))

    _p(f"[DATA] pooled rows={len(df0)} (dropped={len(df)-len(df0)}) dates={len(dates)} symbols={len(symbols)}")
    base_up = float(pd.to_numeric(df0["event_up"], errors="coerce").fillna(0).mean())
    base_dn = float(pd.to_numeric(df0["event_dn"], errors="coerce").fillna(0).mean())
    _p(f"[DATA] base_rate up={base_up*100:.4f}% dn={base_dn*100:.4f}%")

    folds = _make_folds(dates, cfg.fold_count)

    # feature columns for mining
    feature_cols = [c for c in df0.columns if c.endswith("__pct")]
    _p(f"[DBG] feature_cols_n={len(feature_cols)}")

    # Per-fold mining + OOS eval store
    fold_pass_sigs: List[Dict[str, Any]] = []
    latest_pass_sigs: set[str] = set()

    for i, (tr_s, tr_e, te_s, te_e) in enumerate(folds, start=1):
        _banner()
        _p(f"[LIB/FOLD {i}] train={tr_s}..{tr_e}  test={te_s}..{te_e}")

        d_tr = df0[(df0["date"] >= tr_s) & (df0["date"] <= tr_e)].copy()
        d_te = df0[(df0["date"] >= te_s) & (df0["date"] <= te_e)].copy()
        _p(f"[LIB/FOLD {i}] train rows={len(d_tr)} test rows={len(d_te)}")

        # mine
        mined_long, _, perm_long = em.mine_event_rules(d_tr, feature_cols, cfg_event, direction="long")
        mined_short, _, perm_short = em.mine_event_rules(d_tr, feature_cols, cfg_event, direction="short")

        # perm display
        def _perm_status(perm: Dict[str, float]) -> Tuple[str, str]:
            p95 = perm.get("perm_topk_p95", float("nan"))
            if not np.isfinite(p95):
                return "NA", "unavailable"
            return f"{float(p95):.3f}", "ok"

        p95_l, st_l = _perm_status(perm_long)
        p95_s, st_s = _perm_status(perm_short)
        perm_status = st_l if st_l == "ok" else st_s
        perm_p95 = p95_l if st_l == "ok" else p95_s

        mined_total = len(mined_long) + len(mined_short)
        _p(f"[LIB/FOLD {i}] mined_rules={mined_total} perm_p95={perm_p95} status={perm_status}")

        # OOS evaluation for mined rules
        oos_long = em.evaluate_rules_oos(d_te, mined_long, cfg_event)
        oos_short = em.evaluate_rules_oos(d_te, mined_short, cfg_event)
        oos = pd.concat([oos_long, oos_short], axis=0, ignore_index=True) if (not oos_long.empty or not oos_short.empty) else pd.DataFrame()

        if oos.empty:
            _p(f"[LIB/FOLD {i}] OOS eval=0 pass(lift>={cfg.oos_lift_min})=0")
            fold_pass_sigs.append({"fold": i, "df": pd.DataFrame()})
            continue

        # strict OOS filter
        oos = oos.copy()
        oos = oos[np.isfinite(pd.to_numeric(oos["lift"], errors="coerce"))]
        oos = oos[pd.to_numeric(oos["support"], errors="coerce").fillna(0).astype(int) >= int(cfg.oos_support_min)]
        oos = oos[pd.to_numeric(oos["lift"], errors="coerce").astype(float) >= float(cfg.oos_lift_min)]

        _p(f"[LIB/FOLD {i}] OOS eval={len(oos_long)+len(oos_short)} pass(lift>={cfg.oos_lift_min})={len(oos)}")

        if i == len(folds):
            latest_pass_sigs = set(oos["signature"].astype(str).tolist())

        fold_pass_sigs.append({"fold": i, "df": oos})

    _banner()
    _p("[GLOBAL] Rule Library (OOS-filtered across folds)")

    # Combine and fold-count by signature
    if not fold_pass_sigs:
        _p("[GLOBAL] no folds -> nothing to do")
        return 0

    dfs_pass = [x["df"] for x in fold_pass_sigs if isinstance(x.get("df"), pd.DataFrame) and not x["df"].empty]
    if not dfs_pass:
        _p("[GLOBAL] library empty (no fold produced pass rules)")
        if cfg.health.enabled:
            _p("[HEALTH] enabled=1 but no rules survived GLOBAL selection -> nothing to filter.")
        return 0

    all_pass = pd.concat(dfs_pass, axis=0, ignore_index=True)

    # Recency gates
    if cfg.recency.latest_only:
        latest_fold = len(folds)
        df_latest = next((x["df"] for x in fold_pass_sigs if x["fold"] == latest_fold), pd.DataFrame())
        if df_latest.empty:
            _p(f"[GLOBAL][RECENCY] LATEST_ONLY=1 -> latest fold empty, library empty")
            if cfg.health.enabled:
                _p("[HEALTH] enabled=1 but no rules survived GLOBAL selection -> nothing to filter.")
            return 0
        all_pass = df_latest.copy()
        _p(f"[GLOBAL][RECENCY] LATEST_ONLY=1 -> using latest fold rules only (fold={latest_fold})")

    if cfg.recency.must_pass_latest:
        if not latest_pass_sigs:
            _p("[GLOBAL][RECENCY] MUST_PASS_LATEST=1 but latest fold has no pass rules -> library empty")
            if cfg.health.enabled:
                _p("[HEALTH] enabled=1 but no rules survived GLOBAL selection -> nothing to filter.")
            return 0
        before = len(all_pass)
        all_pass = all_pass[all_pass["signature"].astype(str).isin(latest_pass_sigs)].copy()
        _p(f"[GLOBAL][RECENCY] MUST_PASS_LATEST=1 -> kept {len(all_pass)}/{before} rows by signature")

    if all_pass.empty:
        _p("[GLOBAL] library empty (after stability/recency filters).")
        if cfg.health.enabled:
            _p("[HEALTH] enabled=1 but no rules survived GLOBAL selection -> nothing to filter.")
        return 0

    # Aggregate by signature + direction
    agg = (
        all_pass.groupby(["signature", "direction"], as_index=False)
        .agg(
            fold_count=("signature", "size"),
            avg_lift=("lift", "mean"),
            avg_es5=("es5_signed", "mean"),
            avg_med=("median_signed", "mean"),
            avg_mean=("mean_signed", "mean"),
            avg_p_pos=("p_pos_signed", "mean"),
        )
        .copy()
    )

    # OOS filters already applied, but keep safe
    agg = agg[pd.to_numeric(agg["avg_lift"], errors="coerce").astype(float) >= float(cfg.oos_lift_min)].copy()

    # Short payoff-gate
    if not agg.empty:
        is_short = agg["direction"].astype(str) == "short"
        if is_short.any():
            s = agg[is_short].copy()
            s = s[(pd.to_numeric(s["avg_mean"], errors="coerce") > float(cfg.short_mean_min)) & (pd.to_numeric(s["avg_p_pos"], errors="coerce") > float(cfg.short_p_pos_min))]
            agg = pd.concat([agg[~is_short], s], axis=0, ignore_index=True)

    # weights by signature
    weights_by_sig: Dict[str, float] = {}
    for _, r in agg.iterrows():
        sig = str(r["signature"])
        w = _weight_from_metrics(float(r["avg_lift"]), int(r["fold_count"]), float(r["avg_es5"]), float(cfg.oos_lift_min))
        weights_by_sig[sig] = float(w)

    long_sigs = set(agg[agg["direction"].astype(str) == "long"]["signature"].astype(str).tolist())
    short_sigs = set(agg[agg["direction"].astype(str) == "short"]["signature"].astype(str).tolist())

    _p(f"[GLOBAL] library_size={len(agg)}")
    _p(f"[GLOBAL] long_kept={len(long_sigs)} short_kept(after payoff-gate)={len(short_sigs)}")

    # For each fold: build fold rules by re-mining mapping signature->Rule from that fold's mined set
    # We need Rule objects to apply masks.

    for i, (tr_s, tr_e, te_s, te_e) in enumerate(folds, start=1):
        _banner()
        _p(f"[RANK/FOLD {i}] OOS window={te_s}..{te_e}")

        d_tr = df0[(df0["date"] >= tr_s) & (df0["date"] <= tr_e)].copy()
        d_te = df0[(df0["date"] >= te_s) & (df0["date"] <= te_e)].copy()

        # Re-mine (cheap relative to full pipeline; keeps code simple and consistent)
        mined_long, _, _ = em.mine_event_rules(d_tr, feature_cols, cfg_event, direction="long")
        mined_short, _, _ = em.mine_event_rules(d_tr, feature_cols, cfg_event, direction="short")

        fold_long = [r for r in mined_long if em._canonical_signature(r.direction, r.conds) in long_sigs]  # type: ignore
        fold_short = [r for r in mined_short if em._canonical_signature(r.direction, r.conds) in short_sigs]  # type: ignore

        if cfg.health.enabled:
            _p(
                "[HEALTH] enabled=1 win={w} min_n={n} med_min={mm:.4f} es5_min={es:.4f}".format(
                    w=cfg.health.win, n=cfg.health.min_n, mm=cfg.health.med_min, es=cfg.health.es5_min
                )
            )

            fold_long_kept, retired_l = _health_filter_rules(d_tr, fold_long, em, cfg_event, "long", cfg.health)
            fold_short_kept, retired_s = _health_filter_rules(d_tr, fold_short, em, cfg_event, "short", cfg.health)

            _p(f"[HEALTH] long: before={len(fold_long)} after={len(fold_long_kept)} retired={len(retired_l)}")
            for rr in retired_l[:5]:
                _p(
                    "[HEALTH] retired_long sig={sig} support={sup} med={med:.4f} es5={es:.4f}".format(
                        sig=str(rr.get("signature")),
                        sup=int(rr.get("support", 0)),
                        med=float(rr.get("median", float("nan"))),
                        es=float(rr.get("es5", float("nan"))),
                    )
                )

            _p(f"[HEALTH] short: before={len(fold_short)} after={len(fold_short_kept)} retired={len(retired_s)}")
            for rr in retired_s[:5]:
                _p(
                    "[HEALTH] retired_short sig={sig} support={sup} med={med:.4f} es5={es:.4f}".format(
                        sig=str(rr.get("signature")),
                        sup=int(rr.get("support", 0)),
                        med=float(rr.get("median", float("nan"))),
                        es=float(rr.get("es5", float("nan"))),
                    )
                )

            fold_long, fold_short = fold_long_kept, fold_short_kept

        _p(f"[RANK/FOLD {i}] fold_rules kept: long={len(fold_long)} short={len(fold_short)}")

        if len(fold_long) == 0 and len(fold_short) == 0:
            _p(f"[RANK/FOLD {i}] scored_rows(nonzero)=0/{len(d_te)} fires_long=0 fires_short=0")
            _p(f"[RANK/FOLD {i}] LONG top-{cfg.K} fwd_{cfg.fwd_days}d: mean=0.0000 med=0.0000 p>0=0.00%")
            continue

        mean, med, ppos, scored_rows, fires_long, fires_short = _score_fold(
            d_te, fold_long, fold_short, weights_by_sig, em, cfg.K
        )

        _p(
            "[RANK/FOLD {i}] scored_rows(nonzero)={sr}/{n} fires_long={fl} fires_short={fs}".format(
                i=i, sr=scored_rows, n=len(d_te), fl=fires_long, fs=fires_short
            )
        )
        _p(
            "[RANK/FOLD {i}] LONG top-{K} fwd_{fd}d: mean={m:.4f} med={md:.4f} p>0={pp:.2f}%".format(
                i=i,
                K=cfg.K,
                fd=cfg.fwd_days,
                m=mean,
                md=med,
                pp=ppos * 100.0,
            )
        )

    _banner()
    _p("[DONE] Ranker MVP completed.")
    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except KeyboardInterrupt:
        _p("[EXIT] KeyboardInterrupt")
        rc = 130
    except Exception as e:
        _p("[ERROR] Unhandled exception:")
        _p(str(e))
        traceback.print_exc()
        rc = 1
    finally:
        _p("")
        try:
            input("Press Enter to exit...")
        except Exception:
            pass
    raise SystemExit(rc)
