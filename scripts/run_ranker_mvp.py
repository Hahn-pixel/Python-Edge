from __future__ import annotations

"""Python-Edge :: Ranker MVP (daily)

This script:
1) Loads daily bars from the massive dataset
2) Builds daily features
3) Labels events (event_up/event_dn)
4) Mines event rules per fold
5) Evaluates rules OOS and applies strict OOS filters
6) Builds a GLOBAL rule library
7) Scores each OOS fold by net score = long_score - short_score
8) Evaluates top-K forward returns and compares vs a RANDOM baseline distribution

Notes
-----
- No algorithmic behavior is changed by diagnostics/baselines; they only add evaluation.
- Double-click runnable: always waits for input() at the end (even on success).

Env vars
--------
DATASET_ROOT      path to massive dataset root
DATA_START        YYYY-MM-DD
DATA_END          YYYY-MM-DD
TF                timeframe (default 1D)
FOLD_COUNT        number of folds (default 3)

# Event mining
FWD_DAYS          default 5
SIGMA_LOOKBACK    default 60
K_SIGMA           default 1.5
RULES_TRY         default 6000
RULES_KEEP        default 800
MIN_SUPPORT       default 200
MIN_EVENT_HITS    default 30

# Permutation gate
PERM_TRIALS       default 50
PERM_TOPK         default 20
PERM_GATE         default 1
PERM_MARGIN       default 0.15

# OOS filters
OOS_LIFT_MIN      default 1.35
OOS_SUPPORT_MIN   default 200

# Short payoff gate (optional)
SHORT_GATE        0/1
SHORT_MEAN_MIN    default 0.0
SHORT_PPOS_MIN    default 0.5

# Rank
RANK_K            default 3

# Health
HEALTH            0/1
HEALTH_WIN        default 60
HEALTH_MIN_N      default 50
HEALTH_MED_MIN    default 0.0
HEALTH_ES5_MIN    default -0.08

# Recency
MUST_PASS_LATEST  0/1
LATEST_ONLY       0/1

# Debug
DEBUG_IMPORTS     0/1
DEBUG_OOS_DIAG    0/1  (default 1)

# Baseline
BASELINE_SEED     default 1337 (used for the single baseline sample line)
BASELINE_TRIALS   default 2000 (distribution trials)

Exit codes
----------
0 success
1 failure
"""

import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


# ============================================================
# sys.path bootstrap (repo/src)
# ============================================================

ROOT = Path(__file__).resolve().parents[1]  # ...\Python-Edge
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================
# Printing helpers
# ============================================================


def _p(msg: str) -> None:
    print(msg, flush=True)


def _banner() -> None:
    _p("=" * 80)


# ============================================================
# Env helpers
# ============================================================


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


# ============================================================
# Robust imports
# ============================================================


def _import_ingest():
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
        raise ImportError("Cannot import build_features_daily") from e


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
        raise ImportError("Cannot import event_mining") from e


# ============================================================
# Config
# ============================================================


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

    fold_count: int

    fwd_days: int
    sigma_lookback: int
    k_sigma: float

    rules_try: int
    rules_keep: int
    min_support: int
    min_event_hits: int

    perm_trials: int
    perm_topk: int
    perm_gate: bool
    perm_margin: float

    oos_lift_min: float
    oos_support_min: int

    short_gate: bool
    short_mean_min: float
    short_p_pos_min: float

    K: int

    health: HealthCfg
    recency: RecencyCfg

    debug_imports: bool
    debug_oos_diag: bool

    baseline_seed: int
    baseline_trials: int


def load_cfg() -> RunCfg:
    ds = Path(_env_str("DATASET_ROOT", str(ROOT / "data" / "raw" / "massive_dataset")))
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
        debug_imports=_env_bool("DEBUG_IMPORTS", False),
        debug_oos_diag=_env_bool("DEBUG_OOS_DIAG", True),
        baseline_seed=_env_int("BASELINE_SEED", 1337),
        baseline_trials=_env_int("BASELINE_TRIALS", 2000),
    )


# ============================================================
# Data normalization
# ============================================================


def _ensure_close_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()

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
    out[f"fwd_{fwd_days}d_ret"] = out.groupby("symbol", sort=False)["close"].shift(-fwd_days) / out["close"] - 1.0
    out["ret_1d"] = out.groupby("symbol", sort=False)["close"].pct_change()
    return out


# ============================================================
# Folds
# ============================================================


def _make_folds(dates: List[str], n_folds: int) -> List[Tuple[str, str, str, str]]:
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


# ============================================================
# Global library helpers
# ============================================================


def _oos_filter(df_oos: pd.DataFrame, cfg: RunCfg) -> pd.DataFrame:
    if df_oos.empty:
        return df_oos
    out = df_oos.copy()
    out = out[(out["support"] >= int(cfg.oos_support_min)) & (out["lift"] >= float(cfg.oos_lift_min))]
    return out


def _apply_short_payoff_gate(df_oos: pd.DataFrame, cfg: RunCfg) -> pd.DataFrame:
    if df_oos.empty:
        return df_oos
    if not bool(cfg.short_gate):
        return df_oos

    out = df_oos.copy()
    out = out[(out["mean_signed"] > float(cfg.short_mean_min)) & (out["p_pos_signed"] > float(cfg.short_p_pos_min))]
    return out


def _pool_fold_rules(per_fold: List[pd.DataFrame], cfg: RunCfg) -> pd.DataFrame:
    if not per_fold:
        return pd.DataFrame()

    df = pd.concat([x for x in per_fold if x is not None and not x.empty], axis=0, ignore_index=True)
    if df.empty:
        return df

    # RECENCY gating
    if cfg.recency.latest_only:
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

    # Dedup by signature+direction
    df["_key"] = df["signature"].astype(str) + "|" + df["direction"].astype(str)
    df = df.sort_values(["lift", "support"], ascending=[False, False]).drop_duplicates(subset=["_key"], keep="first")
    df = df.drop(columns=["_key"]).reset_index(drop=True)
    return df


# ============================================================
# OOS diagnostics
# ============================================================


def _oos_diag_summary(df_oos_all: pd.DataFrame, cfg: RunCfg) -> str:
    if df_oos_all is None or df_oos_all.empty:
        return "oos_all=0"

    x = df_oos_all.copy()
    x["lift"] = pd.to_numeric(x.get("lift"), errors="coerce")
    x["support"] = pd.to_numeric(x.get("support"), errors="coerce")

    lift = x["lift"].replace([np.inf, -np.inf], np.nan).dropna()
    sup = x["support"].replace([np.inf, -np.inf], np.nan).dropna()

    best_lift = float(lift.max()) if not lift.empty else float("nan")
    med_lift = float(lift.median()) if not lift.empty else float("nan")
    p90_lift = float(lift.quantile(0.90)) if not lift.empty else float("nan")

    best_sup = int(sup.max()) if not sup.empty else -1
    med_sup = float(sup.median()) if not sup.empty else float("nan")

    def _cnt(lift_thr: float, sup_thr: int) -> int:
        y = x[(x["lift"] >= float(lift_thr)) & (x["support"] >= int(sup_thr))]
        return int(len(y))

    c_thr = _cnt(float(cfg.oos_lift_min), int(cfg.oos_support_min))
    c130 = _cnt(1.30, int(cfg.oos_support_min))
    c125 = _cnt(1.25, int(cfg.oos_support_min))

    top = x.sort_values(["lift", "support"], ascending=[False, False]).head(3)
    top_s = []
    for _, r in top.iterrows():
        top_s.append(
            f"{str(r.get('direction','?'))}:{float(r['lift']):.3f}@{int(r['support'])}:{str(r.get('signature',''))[:50]}"
        )

    s = (
        f"oos_all={len(x)} lift_best={best_lift:.3f} lift_med={med_lift:.3f} lift_p90={p90_lift:.3f} "
        f"sup_best={best_sup} sup_med={med_sup:.1f} "
        f"cnt(thr)={c_thr} cnt(>=1.30)={c130} cnt(>=1.25)={c125}"
    )
    if top_s:
        s += " | top3=" + " | ".join(top_s)
    return s


# ============================================================
# Scoring
# ============================================================


def _weight_from_metrics(lift: float, fold_count: int, es5: float, oos_lift_min: float) -> float:
    if not np.isfinite(lift):
        return 0.0

    x = max(0.0, float(lift) - float(oos_lift_min))

    # stronger separation above threshold
    lift_term = (x ** 2) * 25.0

    # mild stability boost by fold count
    fc = max(1, int(fold_count))
    fc_boost = 1.0 + 0.05 * float(min(5, fc - 1))

    # tail penalty
    tail_adj = 1.0
    if np.isfinite(es5):
        tail_adj = 1.0 / (1.0 + max(0.0, -float(es5)) * 10.0)

    w = lift_term * fc_boost * tail_adj
    return float(max(0.0, min(10.0, w)))


def _score_fold(df_oos: pd.DataFrame, rules_global: pd.DataFrame, cfg: RunCfg) -> Tuple[pd.Series, Dict[str, int]]:
    dbg = {"fires_long": 0, "fires_short": 0, "rows_scored": 0}
    if df_oos.empty or rules_global.empty:
        return pd.Series(0.0, index=df_oos.index), dbg

    def _sig_to_conds(signature: str) -> List[Tuple[str, str, str]]:
        parts = str(signature).split("|")
        conds: List[Tuple[str, str, str]] = []
        for p in parts[1:]:
            if len(p) < 4:
                continue
            op = ">" if ">" in p else ("<" if "<" in p else "")
            if not op:
                continue
            feat, qtag = p.split(op, 1)
            if not qtag.startswith("q"):
                continue
            conds.append((feat, op, qtag))
        return conds

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


def _topk_eval(df_oos: pd.DataFrame, score: pd.Series, cfg: RunCfg) -> Tuple[float, float, float]:
    if df_oos.empty:
        return 0.0, 0.0, 0.0

    fwd_col = f"fwd_{int(cfg.fwd_days)}d_ret"
    if fwd_col not in df_oos.columns:
        return 0.0, 0.0, 0.0

    df = df_oos.copy()
    df["_score"] = pd.to_numeric(score, errors="coerce").fillna(0.0)
    df = df.sort_values("_score", ascending=False).head(int(cfg.K))

    fwd = pd.to_numeric(df[fwd_col], errors="coerce").fillna(0.0)
    mean = float(fwd.mean())
    med = float(fwd.median())
    ppos = float((fwd > 0).mean())
    return mean, med, ppos


def _baseline_one(df_oos: pd.DataFrame, cfg: RunCfg) -> Tuple[float, float, float, int]:
    if df_oos.empty:
        return 0.0, 0.0, 0.0, int(cfg.baseline_seed)

    fwd_col = f"fwd_{int(cfg.fwd_days)}d_ret"
    if fwd_col not in df_oos.columns:
        return 0.0, 0.0, 0.0, int(cfg.baseline_seed)

    seed = int(cfg.baseline_seed)
    rs = np.random.RandomState(seed)

    n = int(len(df_oos))
    k = int(min(max(1, int(cfg.K)), n))
    idx = rs.choice(n, size=k, replace=False)

    fwd = pd.to_numeric(df_oos.iloc[idx][fwd_col], errors="coerce").fillna(0.0)
    mean = float(fwd.mean())
    med = float(fwd.median())
    ppos = float((fwd > 0).mean())
    return mean, med, ppos, seed


def _baseline_distribution(df_oos: pd.DataFrame, cfg: RunCfg) -> Dict[str, float]:
    """Baseline distribution over random top-K selections.

    Returns mean/p50/p95 of the *mean forward return* across trials,
    plus percentile of NET vs baseline (computed outside).
    """
    out = {
        "trials": float(cfg.baseline_trials),
        "mean": 0.0,
        "p50": 0.0,
        "p95": 0.0,
    }

    if df_oos.empty or cfg.baseline_trials <= 0:
        return out

    fwd_col = f"fwd_{int(cfg.fwd_days)}d_ret"
    if fwd_col not in df_oos.columns:
        return out

    n = int(len(df_oos))
    k = int(min(max(1, int(cfg.K)), n))

    fwd = pd.to_numeric(df_oos[fwd_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if fwd.size != n:
        fwd = fwd[:n]

    rs = np.random.RandomState(int(cfg.baseline_seed))

    means: List[float] = []
    t = int(cfg.baseline_trials)

    # vectorization is possible, but keep it simple/robust.
    for _ in range(t):
        idx = rs.choice(n, size=k, replace=False)
        means.append(float(np.mean(fwd[idx])))

    arr = np.asarray(means, dtype=float)
    out["mean"] = float(np.mean(arr))
    out["p50"] = float(np.quantile(arr, 0.50))
    out["p95"] = float(np.quantile(arr, 0.95))
    return out


def _percentile_of_value(samples: np.ndarray, value: float) -> float:
    """Percent of samples <= value (in [0,1])."""
    if samples.size == 0:
        return 0.0
    return float(np.mean(samples <= float(value)))


# ============================================================
# Main
# ============================================================


def main() -> int:
    cfg = load_cfg()

    _p(f"[CFG] vendor={cfg.vendor} dataset_root={cfg.dataset_root}")
    _p(f"[CFG] start={cfg.start} end={cfg.end} tf={cfg.tf} folds={cfg.fold_count}")
    _p(f"[CFG] event: fwd_days={cfg.fwd_days} k_sigma={cfg.k_sigma} sigma_lookback={cfg.sigma_lookback}")
    _p(f"[CFG] rules: try={cfg.rules_try} keep={cfg.rules_keep} min_support={cfg.min_support} min_event_hits={cfg.min_event_hits}")
    _p(f"[CFG] perm: trials={cfg.perm_trials} topk={cfg.perm_topk} gate={int(cfg.perm_gate)} margin={cfg.perm_margin}")
    _p(f"[CFG] OOS filter: support>={cfg.oos_support_min} lift>={cfg.oos_lift_min}")
    _p(f"[CFG] short payoff-gate: mean>{cfg.short_mean_min} p>0>{cfg.short_p_pos_min}")
    _p(f"[CFG] rank: K={cfg.K}")
    _p(
        f"[CFG] health: enabled={int(cfg.health.enabled)} win={cfg.health.win} min_n={cfg.health.min_n} "
        f"med_min={cfg.health.med_min:.4f} es5_min={cfg.health.es5_min:.4f}"
    )
    _p(f"[CFG] recency: MUST_PASS_LATEST={int(cfg.recency.must_pass_latest)} LATEST_ONLY={int(cfg.recency.latest_only)}")
    _p(f"[CFG] baseline: seed={cfg.baseline_seed} trials={cfg.baseline_trials}")

    load_aggs, to_daily_index = _import_ingest()
    FeatureConfig, build_features_daily = _import_features()
    em = _import_event_mining()

    if cfg.debug_imports:
        _p(f"[DBG][imports] event_mining={getattr(em, '__file__', '<?>')}")

    if not cfg.dataset_root.exists():
        raise FileNotFoundError(f"Missing dataset_root: {cfg.dataset_root}")

    symbols = sorted([p.name for p in cfg.dataset_root.iterdir() if p.is_dir()])
    if not symbols:
        raise RuntimeError(f"No symbol folders under {cfg.dataset_root}")

    dfs: List[pd.DataFrame] = []
    loaded_ok = 0
    loaded_empty = 0
    loaded_fail = 0

    for sym in symbols:
        try:
            r = load_aggs(cfg.dataset_root, sym, cfg.tf, cfg.start, cfg.end)
            rdf = getattr(r, "df", r)  # AggsLoadResult or DataFrame
            if rdf is None:
                loaded_empty += 1
                continue

            d = to_daily_index(rdf)
            if d is None or getattr(d, "empty", True):
                loaded_empty += 1
                continue

            d = d[["date", "o", "h", "l", "c", "v"]].copy()
            d["symbol"] = sym
            dfs.append(d)
            loaded_ok += 1
        except Exception as e:
            loaded_fail += 1
            _p(f"[DATA][SKIP] sym={sym} reason=load_fail err={type(e).__name__}: {e}")

    _p(f"[DATA] universe_symbols={len(symbols)} tf={cfg.tf}")
    _p(f"[DATA] loaded_ok={loaded_ok} loaded_empty={loaded_empty} loaded_fail={loaded_fail}")

    if not dfs:
        raise RuntimeError("No data loaded (all symbols empty)")

    df = pd.concat(dfs, axis=0, ignore_index=True)
    df = _ensure_close_cols(df)

    df = _add_fwd_returns(df, cfg.fwd_days)

    df = build_features_daily(df, FeatureConfig())

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

    fwd_col = f"fwd_{cfg.fwd_days}d_ret"
    df0 = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["date", "symbol", "ret_1d", fwd_col]).copy()

    dates = sorted(set(df0["date"].astype(str).tolist()))
    _p(f"[DATA] pooled rows={len(df0)} (dropped={len(df)-len(df0)}) dates={len(dates)} symbols={len(symbols)}")

    base_up = float(pd.to_numeric(df0["event_up"], errors="coerce").fillna(0).mean())
    base_dn = float(pd.to_numeric(df0["event_dn"], errors="coerce").fillna(0).mean())
    _p(f"[DATA] base_rate up={base_up*100:.4f}% dn={base_dn*100:.4f}%")

    folds = _make_folds(dates, cfg.fold_count)

    feature_cols = [c for c in df0.columns if c.endswith("__pct")]
    _p(f"[DBG] feature_cols_n={len(feature_cols)}")

    per_fold_oos: List[pd.DataFrame] = []

    _banner()
    for i, (tr0, tr1, te0, te1) in enumerate(folds, start=1):
        _p(f"[LIB/FOLD {i}] train={tr0}..{tr1}  test={te0}..{te1}")

        d_tr = df0[(df0["date"] >= tr0) & (df0["date"] <= tr1)].copy()
        d_te = df0[(df0["date"] >= te0) & (df0["date"] <= te1)].copy()

        _p(f"[LIB/FOLD {i}] train rows={len(d_tr)} test rows={len(d_te)}")

        cfg_event_fold = em.EventMiningConfig(
            fwd_days=cfg_event.fwd_days,
            sigma_lookback=cfg_event.sigma_lookback,
            k_sigma=cfg_event.k_sigma,
            max_rules_try=cfg_event.max_rules_try,
            max_rules_keep=cfg_event.max_rules_keep,
            min_support=cfg_event.min_support,
            min_event_hits=cfg_event.min_event_hits,
            max_conds=getattr(cfg_event, "max_conds", 3),
            seed=getattr(cfg_event, "seed", 7) + i,
            perm_trials=cfg_event.perm_trials,
            perm_topk=cfg_event.perm_topk,
            perm_gate_enabled=getattr(cfg_event, "perm_gate_enabled", True),
            perm_gate_margin=getattr(cfg_event, "perm_gate_margin", 0.15),
        )

        rules, _stats_by_id, perm_stats = em.mine_event_rules(d_tr, feature_cols, cfg_event_fold, direction=None)

        perm_p95 = float(perm_stats.get("perm_topk_p95", float("nan")))
        status = "ok" if len(rules) > 0 else "empty"
        _p(f"[LIB/FOLD {i}] mined_rules={len(rules)} perm_p95={perm_p95:.3f} status={status}")

        oos_all = em.evaluate_rules_oos(d_te, rules, cfg_event_fold)
        if cfg.debug_oos_diag:
            _p(f"[DIAG/FOLD {i}] " + _oos_diag_summary(oos_all, cfg))

        _p(
            f"[LIB/FOLD {i}] OOS eval={len(oos_all)} pass(lift>={cfg.oos_lift_min})={(oos_all['lift'] >= cfg.oos_lift_min).sum() if not oos_all.empty else 0}"
        )

        oos = _oos_filter(oos_all, cfg)
        oos = _apply_short_payoff_gate(oos, cfg)

        if not oos.empty:
            n_long = int((oos["direction"] == "long").sum())
            n_short = int((oos["direction"] == "short").sum())
            _p(f"[LIB/FOLD {i}] pass_breakdown long={n_long} short={n_short}")
        else:
            _p(f"[LIB/FOLD {i}] pass_breakdown long=0 short=0")

        per_fold_oos.append(oos)
        _banner()

    _p("[GLOBAL] Rule Library (OOS-filtered across folds)")
    lib = _pool_fold_rules(per_fold_oos, cfg)
    _p(f"[GLOBAL] library_size={len(lib)}")
    _p(f"[GLOBAL] long_kept={(lib['direction']=='long').sum() if not lib.empty else 0} short_kept(after payoff-gate)={(lib['direction']=='short').sum() if not lib.empty else 0}")

    _banner()

    for i, (_tr0, _tr1, te0, te1) in enumerate(folds, start=1):
        d_te = df0[(df0["date"] >= te0) & (df0["date"] <= te1)].copy()

        net, dbg = _score_fold(d_te, lib, cfg)

        # If library contains only short rules, flip so TOP-K picks strongest short setups.
        if not lib.empty:
            n_long = int((lib["direction"] == "long").sum())
            n_short = int((lib["direction"] == "short").sum())
            if n_long == 0 and n_short > 0:
                net = -net

        scored_rows = int((net != 0.0).sum())
        mean, med, ppos = _topk_eval(d_te, net, cfg)

        _p(f"[RANK/FOLD {i}] OOS window={te0}..{te1}")
        _p(f"[RANK/FOLD {i}] fold_rules kept: long={(lib['direction']=='long').sum() if not lib.empty else 0} short={(lib['direction']=='short').sum() if not lib.empty else 0}")
        _p(f"[RANK/FOLD {i}] scored_rows(nonzero)={scored_rows}/{len(d_te)} fires_long={dbg['fires_long']} fires_short={dbg['fires_short']}")
        _p(f"[RANK/FOLD {i}] NET  top-{cfg.K} fwd_{cfg.fwd_days}d: mean={mean:.4f} med={med:.4f} p>0={ppos*100.0:.2f}%")

        # single baseline sample (kept for visual intuition)
        b_mean, b_med, b_ppos, b_seed = _baseline_one(d_te, cfg)
        _p(f"[RANK/FOLD {i}] BASE top-{cfg.K} fwd_{cfg.fwd_days}d: mean={b_mean:.4f} med={b_med:.4f} p>0={b_ppos*100.0:.2f}% seed={b_seed}")

        # baseline distribution for robustness
        if cfg.baseline_trials > 0:
            # build distribution of random-K means using the SAME RNG seed stream
            fwd_col = f"fwd_{int(cfg.fwd_days)}d_ret"
            fwd = pd.to_numeric(d_te[fwd_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            n = int(len(fwd))
            k = int(min(max(1, int(cfg.K)), n))
            rs = np.random.RandomState(int(cfg.baseline_seed))
            t = int(cfg.baseline_trials)
            means_arr = np.empty(t, dtype=float)
            for ti in range(t):
                idx = rs.choice(n, size=k, replace=False)
                means_arr[ti] = float(np.mean(fwd[idx]))

            base_mean = float(np.mean(means_arr))
            base_p50 = float(np.quantile(means_arr, 0.50))
            base_p95 = float(np.quantile(means_arr, 0.95))
            net_pct = _percentile_of_value(means_arr, mean)
            _p(
                f"[RANK/FOLD {i}] BASE_DIST trials={t} mean={base_mean:.4f} p50={base_p50:.4f} p95={base_p95:.4f} | "
                f"NET_percentile={net_pct*100.0:.2f}% (higher=better)"
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
