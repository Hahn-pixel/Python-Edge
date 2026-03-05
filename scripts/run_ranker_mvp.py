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

DATASET_ROOT  path to massive dataset root
DATA_START    YYYY-MM-DD
DATA_END      YYYY-MM-DD
TF            timeframe (default 1D)
FOLD_COUNT    number of folds (default 3)

# Event mining
FWD_DAYS        default 5
SIGMA_LOOKBACK  default 60
K_SIGMA         default 1.5
RULES_TRY       default 6000
RULES_KEEP      default 800
MIN_SUPPORT     default 200
MIN_EVENT_HITS  default 30

# OOS filters
OOS_LIFT_MIN     default 1.35
OOS_SUPPORT_MIN  default 200

# Rank
RANK_K default 3

# Permutation display (event_mining handles gate internally)
PERM_TRIALS  default 50
PERM_TOPK    default 20
PERM_GATE    default 1
PERM_MARGIN  default 0.15

# HEALTH (Phase 2)
HEALTH         0/1
HEALTH_WIN     default 60
HEALTH_MIN_N   default 50
HEALTH_MED_MIN default 0.0
HEALTH_ES5_MIN default -0.08

# RECENCY
MUST_PASS_LATEST 0/1 (rule must pass OOS in latest fold)
LATEST_ONLY      0/1 (use only latest fold rules)

# Output
DEBUG_DAILY 0/1 (print first few daily top-K lines)

Exit codes:
0 success
1 failure
"""

import os
import sys
import re
import math
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Ensure repo/src is on sys.path (robust import behavior)
ROOT = Path(__file__).resolve().parents[1]  # ...\Python-Edge
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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

def _import_ingest():
    # 1) Package path (expected)
    try:
        from python_edge.data.ingest_aggs import load_aggs, to_daily_index  # type: ignore

        return load_aggs, to_daily_index
    except Exception:
        pass

    # 2) Legacy alt path (if you ever had python_edge.data.ingest_aggs elsewhere)
    try:
        from python_edge.data.ingest_aggs import load_aggs, to_daily_index  # type: ignore

        return load_aggs, to_daily_index
    except Exception:
        pass

    # 3) Repo-root fallback (only if file exists in root)
    try:
        from ingest_aggs import load_aggs, to_daily_index  # type: ignore

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
    short_gate: bool
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
        short_gate=_env_bool("SHORT_GATE", False),
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
        raise ValueError(f"Missing required columns after normalization: {missing}.\nHave={list(out.columns)}")

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

    We want weights to be monotonic in lift, and mildly punish tail risk via es5.
    """
    if not np.isfinite(lift):
        return 0.0

    # Scale lift above threshold
    x = max(0.0, float(lift) - float(oos_lift_min))

    # Fold-count boost for stability (small)
    fc = max(1, int(fold_count))
    fc_boost = 1.0 + 0.05 * float(min(5, fc - 1))

    # Tail risk adjustment: more negative es5 => smaller weight
    tail_adj = 1.0
    if np.isfinite(es5):
        tail_adj = 1.0 / (1.0 + max(0.0, -float(es5)) * 10.0)

    w = (x * 10.0) * fc_boost * tail_adj
    return float(max(0.0, min(10.0, w)))


def _health_filter(
    df: pd.DataFrame,
    date_col: str,
    ret_col: str,
    win: int,
    min_n: int,
    med_min: float,
    es5_min: float,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Retire rows/dates whose recent performance looks bad.

    This is a coarse filter intended for Phase 2; off by default.
    """
    if df.empty:
        return df, {"health_rows_in": 0.0, "health_rows_out": 0.0}

    out = df.copy()
    x = pd.to_numeric(out[ret_col], errors="coerce")

    # Rolling window stats
    rmed = x.rolling(int(win)).median()
    # ES5
    def _roll_es5(v: pd.Series) -> float:
        a = pd.to_numeric(v, errors="coerce").dropna().to_numpy(dtype=float)
        if a.size < int(min_n):
            return float("nan")
        q = float(np.quantile(a, 0.05))
        tail = a[a <= q]
        return float(np.mean(tail)) if tail.size else q

    res5 = x.rolling(int(win)).apply(_roll_es5, raw=False)

    good = (
        (x.rolling(int(win)).count() >= int(min_n))
        & (rmed >= float(med_min))
        & (res5 >= float(es5_min))
    )

    n_in = int(len(out))
    out = out[good.fillna(False)].copy()
    n_out = int(len(out))

    return out, {"health_rows_in": float(n_in), "health_rows_out": float(n_out)}


# =========================
# Rule library aggregation
# =========================

def _apply_short_payoff_gate(df_oos: pd.DataFrame, cfg: RunCfg, em: Any) -> pd.DataFrame:
    """Optional gate for short rules by signed payoff stats."""
    if df_oos.empty:
        return df_oos
    if not bool(cfg.short_gate):
        return df_oos

    # We interpret mean_signed and p_pos_signed as already signed (short rules use -fwd)
    out = df_oos.copy()
    out = out[(out["mean_signed"] > float(cfg.short_mean_min)) & (out["p_pos_signed"] > float(cfg.short_p_pos_min))]
    return out


def _oos_filter(df_oos: pd.DataFrame, cfg: RunCfg) -> pd.DataFrame:
    if df_oos.empty:
        return df_oos

    out = df_oos.copy()
    out = out[(out["support"] >= int(cfg.oos_support_min)) & (out["lift"] >= float(cfg.oos_lift_min))]
    return out


def _pool_fold_rules(
    per_fold: List[pd.DataFrame],
    cfg: RunCfg,
) -> pd.DataFrame:
    """Build GLOBAL library from per-fold OOS survivors."""
    if not per_fold:
        return pd.DataFrame()

    df = pd.concat([x for x in per_fold if x is not None and not x.empty], axis=0, ignore_index=True)
    if df.empty:
        return df

    # RECENCY gating
    if cfg.recency.latest_only:
        # Keep only rules that appear in latest fold
        latest = per_fold[-1]
        if latest is None or latest.empty:
            _p("[GLOBAL][RECENCY] LATEST_ONLY=1 -> latest fold empty, library empty")
            return pd.DataFrame()
        sigs = set(latest["signature"].astype(str))
        df = df[df["signature"].astype(str).isin(sigs)].copy()

    if cfg.recency.must_pass_latest:
        latest = per_fold[-1]
        if latest is None or latest.empty:
            _p("[GLOBAL][RECENCY] MUST_PASS_LATEST=1 -> latest fold empty, library empty")
            return pd.DataFrame()
        sigs = set(latest["signature"].astype(str))
        df = df[df["signature"].astype(str).isin(sigs)].copy()

    # Dedup by signature + direction: keep best by lift then support
    df["_key"] = (
        df["signature"].astype(str)
        + "|"
        + df["direction"].astype(str)
    )

    df = df.sort_values(["lift", "support"], ascending=[False, False])
    df = df.drop_duplicates(subset=["_key"], keep="first").copy()
    df = df.drop(columns=["_key"])

    return df.reset_index(drop=True)


# =========================
# Scoring
# =========================

def _score_fold(
    df_oos: pd.DataFrame,
    rules_global: pd.DataFrame,
    cfg: RunCfg,
) -> Tuple[pd.Series, Dict[str, int]]:
    """Return net score per row and debug counts."""
    dbg = {
        "fires_long": 0,
        "fires_short": 0,
        "rows_scored": 0,
    }

    if df_oos.empty or rules_global.empty:
        return pd.Series(0.0, index=df_oos.index), dbg

    # Precompute masks per rule
    # Each rule row has: direction, signature, and condition columns depend on event_mining implementation.
    # Here we use the stored 'signature' and rely on event_mining Rule representation reconstruction.

    # For MVP, we approximate by re-parsing signature like: dir|feat>q80|feat2<q20
    # This keeps scripts self-contained.

    def _sig_to_conds(signature: str) -> List[Tuple[str, str, str]]:
        parts = str(signature).split("|")
        # parts[0] is direction
        conds: List[Tuple[str, str, str]] = []
        for p in parts[1:]:
            m = re.match(r"^(.*?)([<>])(q\d\d)$", p)
            if not m:
                continue
            feat, op, qtag = m.group(1), m.group(2), m.group(3)
            conds.append((feat, op, qtag))
        return conds

    # Quantile thresholds in OOS fold for each feature
    feat_cols = sorted({c.split("|")[0] for c in []})

    # But we can compute thresholds on demand per feature per qtag
    thr_cache: Dict[Tuple[str, str], float] = {}

    def _thr(feat: str, qtag: str) -> float:
        k = (feat, qtag)
        if k in thr_cache:
            return thr_cache[k]
        s = pd.to_numeric(df_oos[feat], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            thr_cache[k] = 0.0
            return 0.0
        q = int(qtag[1:]) / 100.0
        v = float(s.quantile(q))
        thr_cache[k] = v
        return v

    # Score accumulation
    long_score = pd.Series(0.0, index=df_oos.index)
    short_score = pd.Series(0.0, index=df_oos.index)

    for _, rr in rules_global.iterrows():
        sig = str(rr["signature"])
        direction = str(rr["direction"])
        lift = float(rr["lift"])
        es5 = float(rr.get("es5_signed", np.nan))

        w = _weight_from_metrics(lift=lift, fold_count=int(cfg.fold_count), es5=es5, oos_lift_min=float(cfg.oos_lift_min))
        if w <= 0:
            continue

        conds = _sig_to_conds(sig)
        if not conds:
            continue

        mask = pd.Series(True, index=df_oos.index)
        for feat, op, qtag in conds:
            if feat not in df_oos.columns:
                mask &= False
                continue
            x = pd.to_numeric(df_oos[feat], errors="coerce")
            t = _thr(feat, qtag)
            if op == ">":
                mask &= x > t
            else:
                mask &= x < t

        mask = mask.fillna(False)
        if not mask.any():
            continue

        if direction == "long":
            dbg["fires_long"] += int(mask.sum())
            long_score[mask] += w
        else:
            dbg["fires_short"] += int(mask.sum())
            short_score[mask] += w

    net = long_score - short_score
    dbg["rows_scored"] = int((net != 0.0).sum())

    return net, dbg


def _topk_eval(df_oos: pd.DataFrame, score: pd.Series, cfg: RunCfg) -> Dict[str, float]:
    if df_oos.empty:
        return {"mean": 0.0, "med": 0.0, "p_pos": 0.0}

    fwd_col = f"fwd_{int(cfg.fwd_days)}d_ret"
    if fwd_col not in df_oos.columns:
        return {"mean": 0.0, "med": 0.0, "p_pos": 0.0}

    df = df_oos.copy()
    df["_score"] = pd.to_numeric(score, errors="coerce").fillna(0.0)
    df = df.sort_values("_score", ascending=False).head(int(cfg.K))

    fwd = pd.to_numeric(df[fwd_col], errors="coerce").fillna(0.0)
    mean = float(fwd.mean())
    med = float(fwd.median())
    ppos = float((fwd > 0).mean())

    return {"mean": mean, "med": med, "p_pos": ppos}


# =========================
# Main
# =========================

def main() -> int:
    cfg = load_cfg()

    _p(f"[CFG] vendor={cfg.vendor} dataset_root={cfg.dataset_root}")
    _p(f"[CFG] start={cfg.start} end={cfg.end} tf={cfg.tf} folds={cfg.fold_count}")
    _p(
        f"[CFG] event: fwd_days={cfg.fwd_days} k_sigma={cfg.k_sigma} sigma_lookback={cfg.sigma_lookback}"
    )
    _p(
        f"[CFG] rules: try={cfg.rules_try} keep={cfg.rules_keep} min_support={cfg.min_support} min_event_hits={cfg.min_event_hits}"
    )
    _p(
        f"[CFG] perm: trials={cfg.perm_trials} topk={cfg.perm_topk} gate={int(cfg.perm_gate)} margin={cfg.perm_margin}"
    )
    _p(f"[CFG] OOS filter: support>={cfg.oos_support_min} lift>={cfg.oos_lift_min}")
    _p(
        f"[CFG] short payoff-gate: mean>{cfg.short_mean_min} p>0>{cfg.short_p_pos_min}"
        if cfg.short_gate
        else "[CFG] short payoff-gate: disabled"
    )
    _p(f"[CFG] rank: K={cfg.K}")
    _p(
        f"[CFG] health: enabled={int(cfg.health.enabled)} win={cfg.health.win} min_n={cfg.health.min_n} med_min={cfg.health.med_min:.4f} es5_min={cfg.health.es5_min:.4f}"
    )
    _p(
        f"[CFG] recency: MUST_PASS_LATEST={int(cfg.recency.must_pass_latest)} LATEST_ONLY={int(cfg.recency.latest_only)}"
    )

    load_aggs, to_daily_index = _import_ingest()
    FeatureConfig, build_features_daily = _import_features()
    em = _import_event_mining()

    # Load universe symbols
    tf_token = cfg.tf.lower()
    if tf_token in ("1d", "d", "day", "daily"):
        tf_token = "1d"

    # Accept either 1D.parquet OR aggs_1d_*.json structure.
    ds_root = Path(cfg.dataset_root)
    if not ds_root.exists():
        raise RuntimeError(f"DATASET_ROOT does not exist: {ds_root}")

    # Determine universe by scanning symbol subfolders
    sym_dirs = [p for p in ds_root.iterdir() if p.is_dir()]
    symbols = sorted([p.name for p in sym_dirs])

    # Load data for each symbol
    loaded: List[pd.DataFrame] = []
    loaded_ok = 0
    loaded_empty = 0
    loaded_fail = 0

    for sym in symbols:
        try:
            r = load_aggs(cfg.dataset_root, sym, cfg.tf, cfg.start, cfg.end)

            # load_aggs may return either a DataFrame OR an AggsLoadResult-like object with .df
            rdf = getattr(r, "df", r)
            if rdf is None:
                loaded_empty += 1
                continue

            d = to_daily_index(rdf)
        except Exception as e:
            loaded_fail += 1
            _p(f"[DATA][SKIP] sym={sym} reason=load_fail err={type(e).__name__}: {e}")
            continue

        if d is None or getattr(d, "empty", True):
            loaded_empty += 1
            continue

        # Ensure required columns
        d = d.copy()
        if "symbol" not in d.columns:
            d["symbol"] = sym

        loaded.append(d)
        loaded_ok += 1

    _p(f"[DATA] universe_symbols={len(symbols)} tf={cfg.tf}")
    _p(f"[DATA] loaded_ok={loaded_ok} loaded_empty={loaded_empty} loaded_fail={loaded_fail}")

    if not loaded:
        raise RuntimeError("No data loaded (all symbols empty)")

    df = pd.concat(loaded, axis=0, ignore_index=True)
    df = _ensure_close_cols(df)

    # Add returns and forward returns
    df["ret_1d"] = df.groupby("symbol", sort=False)["close"].pct_change()
    df = _add_fwd_returns(df, cfg.fwd_days)

    # Build features
    fcfg = FeatureConfig()
    df_feat = build_features_daily(df, fcfg)

    # Drop rows without forward return
    fwd_col = f"fwd_{int(cfg.fwd_days)}d_ret"
    before = len(df_feat)
    df_feat = df_feat[np.isfinite(pd.to_numeric(df_feat[fwd_col], errors="coerce"))].copy()
    dropped = before - len(df_feat)

    dates = sorted(df_feat["date"].astype(str).unique().tolist())
    syms = sorted(df_feat["symbol"].astype(str).unique().tolist())

    _p(f"[DATA] pooled rows={len(df_feat)} (dropped={dropped}) dates={len(dates)} symbols={len(syms)}")

    # Base rates
    tmp = em.label_events(df_feat, em.EventMiningConfig(fwd_days=cfg.fwd_days, sigma_lookback=cfg.sigma_lookback, k_sigma=cfg.k_sigma))
    up = float(pd.to_numeric(tmp["event_up"], errors="coerce").fillna(0).mean()) * 100.0
    dn = float(pd.to_numeric(tmp["event_dn"], errors="coerce").fillna(0).mean()) * 100.0
    _p(f"[DATA] base_rate up={up:.4f}% dn={dn:.4f}%")

    # Feature columns used for mining
    feature_cols = [c for c in df_feat.columns if c.startswith("feat_")]
    if not feature_cols:
        # fallback: accept common names if build_features_daily does not prefix
        feature_cols = [c for c in df_feat.columns if c not in ("date", "symbol", "open", "high", "low", "close", "volume", "o", "h", "l", "c", "v", "ret_1d", fwd_col)]

    _p(f"[DBG] feature_cols_n={len(feature_cols)}")

    folds = _make_folds(dates, cfg.fold_count)

    # Mine rules per fold
    per_fold_oos: List[pd.DataFrame] = []

    _banner()
    for i, (tr0, tr1, te0, te1) in enumerate(folds, start=1):
        _p(f"[LIB/FOLD {i}] train={tr0}..{tr1}  test={te0}..{te1}")

        df_tr = df_feat[(df_feat["date"] >= tr0) & (df_feat["date"] <= tr1)].copy()
        df_te = df_feat[(df_feat["date"] >= te0) & (df_feat["date"] <= te1)].copy()

        _p(f"[LIB/FOLD {i}] train rows={len(df_tr)} test rows={len(df_te)}")

        emcfg = em.EventMiningConfig(
            fwd_days=cfg.fwd_days,
            sigma_lookback=cfg.sigma_lookback,
            k_sigma=cfg.k_sigma,
            max_rules_try=cfg.rules_try,
            max_rules_keep=cfg.rules_keep,
            min_support=cfg.min_support,
            min_event_hits=cfg.min_event_hits,
            max_conds=3,
            seed=7 + i,
            perm_trials=cfg.perm_trials,
            perm_topk=cfg.perm_topk,
            perm_gate_enabled=cfg.perm_gate,
            perm_gate_margin=cfg.perm_margin,
        )

        df_tr_l = em.label_events(df_tr, emcfg)
        df_te_l = em.label_events(df_te, emcfg)

        rules, stats_by_id, perm_stats = em.mine_event_rules(df_tr_l, feature_cols, emcfg, direction=None)

        perm_p95 = float(perm_stats.get("perm_topk_p95", float("nan")))
        status = "ok" if (len(rules) > 0) else "empty"
        _p(f"[LIB/FOLD {i}] mined_rules={len(rules)} perm_p95={perm_p95:.3f} status={status}")

        df_oos = em.evaluate_rules_oos(df_te_l, rules, emcfg)
        _p(f"[LIB/FOLD {i}] OOS eval={len(df_oos)} pass(lift>={cfg.oos_lift_min})={(df_oos['lift'] >= cfg.oos_lift_min).sum() if not df_oos.empty else 0}")

        # Apply OOS filters
        df_oos = _oos_filter(df_oos, cfg)
        # Optional short payoff gate
        df_oos = _apply_short_payoff_gate(df_oos, cfg, em)

        if not df_oos.empty:
            n_long = int((df_oos["direction"] == "long").sum())
            n_short = int((df_oos["direction"] == "short").sum())
            _p(f"[LIB/FOLD {i}] pass_breakdown long={n_long} short={n_short}")
        else:
            _p(f"[LIB/FOLD {i}] pass_breakdown long=0 short=0")

        per_fold_oos.append(df_oos)
        _banner()

    # GLOBAL library
    _p("[GLOBAL] Rule Library (OOS-filtered across folds)")
    lib = _pool_fold_rules(per_fold_oos, cfg)
    _p(f"[GLOBAL] library_size={len(lib)}")
    _p(f"[GLOBAL] long_kept={(lib['direction'] == 'long').sum() if not lib.empty else 0} short_kept(after payoff-gate)={(lib['direction'] == 'short').sum() if not lib.empty else 0}")

    _banner()

    # Rank per fold
    for i, (tr0, tr1, te0, te1) in enumerate(folds, start=1):
        df_te = df_feat[(df_feat["date"] >= te0) & (df_feat["date"] <= te1)].copy()

        # HEALTH filter (optional)
        if cfg.health.enabled:
            df_te, hstats = _health_filter(
                df_te,
                date_col="date",
                ret_col=fwd_col,
                win=cfg.health.win,
                min_n=cfg.health.min_n,
                med_min=cfg.health.med_min,
                es5_min=cfg.health.es5_min,
            )
            _p(f"[HEALTH] fold={i} rows_in={int(hstats['health_rows_in'])} rows_out={int(hstats['health_rows_out'])}")

        score, dbg = _score_fold(df_te, lib, cfg)

        _p(f"[RANK/FOLD {i}] OOS window={te0}..{te1}")
        _p(
            f"[RANK/FOLD {i}] fold_rules kept: long={(lib['direction'] == 'long').sum() if not lib.empty else 0} short={(lib['direction'] == 'short').sum() if not lib.empty else 0}"
        )
        _p(f"[RANK/FOLD {i}] scored_rows(nonzero)={dbg['rows_scored']}/{len(df_te)} fires_long={dbg['fires_long']} fires_short={dbg['fires_short']}")

        top = _topk_eval(df_te, score, cfg)
        _p(
            f"[RANK/FOLD {i}] LONG top-{cfg.K} fwd_{cfg.fwd_days}d: mean={top['mean']:.4f} med={top['med']:.4f} p>0={top['p_pos']*100.0:.2f}%"
        )

        if cfg.debug_daily:
            # Print first few daily top-k
            dd = df_te.copy()
            dd["_score"] = score
            dd = dd.sort_values(["date", "_score"], ascending=[True, False])
            for d in sorted(dd["date"].astype(str).unique().tolist())[:5]:
                x = dd[dd["date"] == d].head(int(cfg.K))
                rr = float(pd.to_numeric(x[fwd_col], errors="coerce").fillna(0.0).mean()) if not x.empty else 0.0
                _p(f"[DBG][daily] date={d} topK_mean_fwd={rr:.4f}")

        _banner()

    # Done
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        _p("[ERROR] Unhandled exception:")
        traceback.print_exc()
        rc = 1

    # Double-click runnable behavior
    try:
        input("\n[END] Press Enter to exit...")
    except Exception:
        pass

    raise SystemExit(rc)
