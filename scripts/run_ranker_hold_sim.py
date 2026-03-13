# scripts/run_ranker_hold_sim.py
# Python-Edge / massive dataset
# Double-click runnable. Never auto-closes (always waits for Enter).
#
# Purpose:
# - Mine cross-sectional "rules" on DAILY bars.
# - Walk-forward CV: mine on train, evaluate on OOS with NO leakage.
# - Try global rule library first.
# - If global library is empty or explicitly disabled, fall back to FOLD-LOCAL runtime.
# - Print alpha-delay surface so we can test the same-day dislocation thesis.

from __future__ import annotations

import json
import math
import os
import random
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _press_enter_exit(code: int) -> None:
    try:
        print(f"\n[EXIT] code={code}")
        input("Press Enter to exit...")
    except Exception:
        pass
    raise SystemExit(code)


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
    if v is None:
        return default
    s = str(v).strip().lower()
    return s in ("1", "true", "yes", "y", "on")


@dataclass(frozen=True)
class Config:
    dataset_root: Path
    start: str
    end: str
    fwd_days: int
    k_sigma: float
    sigma_lookback: int
    rules_try: int
    rules_keep: int
    min_support: int
    min_event_hits: int
    perm_trials: int
    perm_topk: int
    gate: bool
    gate_margin: float
    oos_min_support: int
    oos_min_lift: float
    runtime_topn_long: int
    rank_k: int
    hold_days: int
    max_pos: int
    tc_bps_in: float
    tc_bps_out: float
    rebalance_band: float
    cash_floor: float
    clamp_cash: bool
    cap_vol: int
    cap_mom: int
    cap_trend: int
    cap_other: int
    cap_pair: int
    slices: int
    fold2_only_slices: bool
    seed: int
    alpha_decay_debug: bool
    delay_days: int
    entry_mode: str
    alpha_horizons: Tuple[int, ...]
    alpha_quantiles: Tuple[float, ...]
    use_fold_local_runtime: bool
    fold_local_allow_oos_big_fallback: bool


def load_config() -> Config:
    return Config(
        dataset_root=Path(_env_str("DATASET_ROOT", r"D:\massive_dataset")),
        start=_env_str("START", "2023-01-01"),
        end=_env_str("END", "2026-02-28"),
        fwd_days=_env_int("FWD_DAYS", 5),
        k_sigma=_env_float("K_SIGMA", 1.5),
        sigma_lookback=_env_int("SIGMA_LOOKBACK", 60),
        rules_try=_env_int("RULES_TRY", 6000),
        rules_keep=_env_int("RULES_KEEP", 800),
        min_support=_env_int("MIN_SUPPORT", 200),
        min_event_hits=_env_int("MIN_EVENT_HITS", 30),
        perm_trials=_env_int("PERM_TRIALS", 50),
        perm_topk=_env_int("PERM_TOPK", 20),
        gate=_env_bool("GATE", True),
        gate_margin=_env_float("GATE_MARGIN", 0.15),
        oos_min_support=_env_int("OOS_MIN_SUPPORT", 200),
        oos_min_lift=_env_float("OOS_MIN_LIFT", 1.35),
        runtime_topn_long=_env_int("RUNTIME_TOPN_LONG", 40),
        rank_k=_env_int("RANK_K", 3),
        hold_days=_env_int("HOLD_DAYS", 5),
        max_pos=_env_int("MAX_POS", 3),
        tc_bps_in=_env_float("TC_BPS_IN", 4.0),
        tc_bps_out=_env_float("TC_BPS_OUT", 4.0),
        rebalance_band=_env_float("REBALANCE_BAND", 0.10),
        cash_floor=_env_float("CASH_FLOOR", 0.00),
        clamp_cash=_env_bool("CLAMP_CASH", True),
        cap_vol=_env_int("CAP_VOL", 20),
        cap_mom=_env_int("CAP_MOM", 15),
        cap_trend=_env_int("CAP_TREND", 16),
        cap_other=_env_int("CAP_OTHER", 40),
        cap_pair=_env_int("CAP_PAIR", 5),
        slices=_env_int("SLICES", 3),
        fold2_only_slices=_env_bool("FOLD2_ONLY", True),
        seed=_env_int("SEED", 7),
        alpha_decay_debug=_env_bool("ALPHA_DECAY_DEBUG", True),
        delay_days=max(0, _env_int("SIGNAL_DELAY_DAYS", 0)),
        entry_mode=_env_str("ENTRY_MODE", "next_open").lower(),
        alpha_horizons=tuple(int(x) for x in _env_str("ALPHA_HORIZONS", "1,2,3").split(",") if str(x).strip()),
        alpha_quantiles=tuple(float(x) for x in _env_str("ALPHA_QUANTILES", "0.02,0.05,0.10").split(",") if str(x).strip()),
        use_fold_local_runtime=_env_bool("USE_FOLD_LOCAL_RUNTIME", True),
        fold_local_allow_oos_big_fallback=_env_bool("FOLD_LOCAL_ALLOW_OOS_BIG_FALLBACK", True),
    )


def _find_aggs_1d_files(dataset_root: Path) -> List[Tuple[str, Path]]:
    out: List[Tuple[str, Path]] = []
    if not dataset_root.exists():
        raise RuntimeError(f"dataset_root not found: {dataset_root}")
    for sym_dir in sorted([p for p in dataset_root.iterdir() if p.is_dir()]):
        sym = sym_dir.name.strip().upper()
        candidates = sorted(sym_dir.glob("aggs_1d_*.json"))
        if not candidates:
            continue
        best = max(candidates, key=lambda p: p.stat().st_size)
        out.append((sym, best))
    return out


def _load_aggs_1d(sym: str, path: Path) -> pd.DataFrame:
    js = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    rows = js.get("results") or []
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "t" not in df.columns:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.date.astype(str)
    df["symbol"] = sym
    for c in ("o", "h", "l", "c", "v"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[["date", "symbol", "o", "h", "l", "c", "v"]].dropna()
    df = df.sort_values(["date"]).reset_index(drop=True)
    return df


def load_dataset(cfg: Config) -> pd.DataFrame:
    pairs = _find_aggs_1d_files(cfg.dataset_root)
    if not pairs:
        raise RuntimeError(f"No aggs_1d_*.json found under {cfg.dataset_root}")
    dfs: List[pd.DataFrame] = []
    for sym, fp in pairs:
        d = _load_aggs_1d(sym, fp)
        if not d.empty:
            dfs.append(d)
    if not dfs:
        raise RuntimeError("All aggs files loaded empty")
    df = pd.concat(dfs, ignore_index=True)
    df = df[(df["date"] >= cfg.start) & (df["date"] <= cfg.end)].copy()
    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)
    return df


def _winsorize_series(x: pd.Series, p: float = 0.01) -> pd.Series:
    lo = x.quantile(p)
    hi = x.quantile(1.0 - p)
    return x.clip(lower=lo, upper=hi)


def add_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()
    for c in ("c", "o", "h", "l", "v"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ret_1d"] = df.groupby("symbol")["c"].pct_change(1)
    df["ret_3d"] = df.groupby("symbol")["c"].pct_change(3)
    df["rng"] = (df["h"] - df["l"]) / df["c"].replace(0, np.nan)
    df["rv"] = df.groupby("symbol")["ret_1d"].rolling(cfg.sigma_lookback, min_periods=max(10, cfg.sigma_lookback // 3)).std().reset_index(level=0, drop=True)
    df["rng_ma20"] = df.groupby("symbol")["rng"].rolling(20, min_periods=10).mean().reset_index(level=0, drop=True)
    df["compression_raw"] = (df["rng_ma20"] / df["rng"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    def _ema(s: pd.Series, span: int) -> pd.Series:
        return s.ewm(span=span, adjust=False, min_periods=max(5, span // 3)).mean()

    df["ema20"] = df.groupby("symbol")["c"].apply(lambda s: _ema(s, 20)).reset_index(level=0, drop=True)
    df["ema10"] = df.groupby("symbol")["c"].apply(lambda s: _ema(s, 10)).reset_index(level=0, drop=True)
    df["ema50"] = df.groupby("symbol")["c"].apply(lambda s: _ema(s, 50)).reset_index(level=0, drop=True)
    df["ema_dist_raw"] = (df["c"] - df["ema20"]) / df["ema20"].replace(0, np.nan)
    df["ema_fast_slope_raw"] = df.groupby("symbol")["ema10"].diff(1) / df.groupby("symbol")["ema10"].shift(1).replace(0, np.nan)
    df["ema_slow_slope_raw"] = df.groupby("symbol")["ema50"].diff(1) / df.groupby("symbol")["ema50"].shift(1).replace(0, np.nan)
    for c in ("ret_1d", "ret_3d", "rng", "rv", "compression_raw", "ema_dist_raw", "ema_fast_slope_raw", "ema_slow_slope_raw"):
        df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)

    def _cs_pct_series(df_: pd.DataFrame, col: str, winsor_p: float = 0.01) -> pd.Series:
        out = pd.Series(index=df_.index, dtype=float)
        for _, g in df_.groupby("date", sort=False):
            x = pd.to_numeric(g[col], errors="coerce")
            if x.notna().sum() == 0:
                out.loc[g.index] = np.nan
                continue
            if winsor_p > 0.0 and x.notna().sum() >= 5:
                x = _winsorize_series(x, p=winsor_p)
            out.loc[g.index] = x.rank(method="average", pct=True)
        return out

    df["atr_pct"] = _cs_pct_series(df, "rng")
    df["rv_10"] = _cs_pct_series(df, "rv")
    df["mom_1d"] = _cs_pct_series(df, "ret_1d")
    df["mom_3d"] = _cs_pct_series(df, "ret_3d")
    df["compression"] = _cs_pct_series(df, "compression_raw")
    df["ema_dist"] = _cs_pct_series(df, "ema_dist_raw")
    df["ema_fast_slope"] = _cs_pct_series(df, "ema_fast_slope_raw")
    df["ema_slow_slope"] = _cs_pct_series(df, "ema_slow_slope_raw")
    return df


def add_event_labels(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()
    df["fwd_ret"] = df.groupby("symbol")["c"].shift(-cfg.fwd_days) / df["c"] - 1.0
    sigma_term = df["rv"] * math.sqrt(float(max(1, cfg.fwd_days)))
    thresh = cfg.k_sigma * sigma_term
    df["event_up"] = (df["fwd_ret"] > thresh).astype(float)
    return df


@dataclass(frozen=True)
class Rule:
    cluster: str
    a: str
    op: str
    b: Optional[str] = None
    thr: Optional[float] = None

    def sig(self) -> str:
        if self.b is not None:
            return f"{self.cluster}:{self.a}{self.op}{self.b}"
        return f"{self.cluster}:{self.a}{self.op}{self.thr:.2f}"


@dataclass
class RuleStats:
    sig: str
    rule: Rule
    support: int
    hits: int
    prec: float
    lift: float
    mean: float
    med: float
    p_pos: float
    es5: float
    w: float


FEATURES = ["atr_pct", "rv_10", "mom_1d", "mom_3d", "compression", "ema_dist", "ema_fast_slope", "ema_slow_slope"]


def compute_base_rate(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    return float(pd.to_numeric(df["event_up"], errors="coerce").fillna(0.0).mean())


def _rule_cluster(a: str, b: Optional[str], thr: Optional[float]) -> str:
    s = {a, b} if b is not None else {a}
    if {"atr_pct", "rv_10"} & s:
        return "VOL"
    if {"mom_1d", "mom_3d"} & s:
        return "MOM"
    if {"ema_dist", "ema_fast_slope", "ema_slow_slope"} & s:
        return "TREND"
    if b is not None:
        return "PAIR"
    return "OTHER"


def random_rule(rng: random.Random) -> Rule:
    if rng.random() < 0.45:
        a = rng.choice(FEATURES)
        thr = rng.choice([0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90])
        op = rng.choice([">=", "<="])
        return Rule(cluster=_rule_cluster(a, None, thr), a=a, op=op, thr=float(thr))
    a, b = rng.sample(FEATURES, 2)
    op = rng.choice([">=", "<="])
    return Rule(cluster=_rule_cluster(a, b, None), a=a, op=op, b=b)


def fire_rule(df: pd.DataFrame, rule: Rule) -> pd.Series:
    a = pd.to_numeric(df[rule.a], errors="coerce")
    if rule.b is not None:
        b = pd.to_numeric(df[rule.b], errors="coerce")
        if rule.op == ">=":
            return (a >= b) & a.notna() & b.notna()
        return (a <= b) & a.notna() & b.notna()
    thr = float(rule.thr if rule.thr is not None else 0.5)
    if rule.op == ">=":
        return (a >= thr) & a.notna()
    return (a <= thr) & a.notna()


def eval_rule(df: pd.DataFrame, rule: Rule, base_rate: float) -> RuleStats:
    m = fire_rule(df, rule)
    sub = df.loc[m].copy()
    support = int(len(sub))
    if support <= 0:
        return RuleStats(rule.sig(), rule, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, float("nan"), 0.0)
    evt = pd.to_numeric(sub["event_up"], errors="coerce").fillna(0.0)
    fwd = pd.to_numeric(sub["fwd_ret"], errors="coerce")
    prec = float(evt.mean()) if support > 0 else 0.0
    hits = int(evt.sum())
    lift = float(prec / base_rate) if base_rate > 1e-12 else 0.0
    mean = float(fwd.mean()) if len(fwd) else 0.0
    med = float(fwd.median()) if len(fwd) else 0.0
    p_pos = float((fwd > 0).mean()) if len(fwd) else 0.0
    es5 = float(fwd[fwd <= fwd.quantile(0.05)].mean()) if len(fwd) >= 20 else float("nan")
    w = max(0.0, lift - 1.0) * max(0.0, mean)
    return RuleStats(rule.sig(), rule, support, hits, prec, lift, mean, med, p_pos, es5, w)


def mine_rules(df: pd.DataFrame, cfg: Config, rng: random.Random) -> Tuple[List[RuleStats], float, List[RuleStats]]:
    br = compute_base_rate(df)
    seen = set()
    mined: List[RuleStats] = []
    tries = 0
    while tries < cfg.rules_try:
        tries += 1
        r = random_rule(rng)
        s = r.sig()
        if s in seen:
            continue
        seen.add(s)
        st = eval_rule(df, r, br)
        if st.support < cfg.min_support:
            continue
        if st.hits < cfg.min_event_hits:
            continue
        mined.append(st)
    mined.sort(key=lambda x: (x.w, x.lift, x.support), reverse=True)
    kept = mined[: cfg.rules_keep]
    perm_best: List[float] = []
    if cfg.perm_trials > 0 and len(df) > 0:
        evt = df["event_up"].to_numpy(copy=True)
        for _ in range(cfg.perm_trials):
            evt2 = evt.copy()
            rng.shuffle(evt2)
            d2 = df.copy()
            d2["event_up"] = evt2
            br2 = float(np.mean(evt2)) if len(evt2) else 0.0
            best_lifts = []
            sub_rules = [random_rule(rng) for _ in range(max(20, cfg.perm_topk))]
            for r in sub_rules:
                st = eval_rule(d2, r, br2)
                if st.support >= cfg.min_support and st.hits >= cfg.min_event_hits:
                    best_lifts.append(st.lift)
            perm_best.append(float(max(best_lifts)) if best_lifts else 1.0)
        perm_p95 = float(np.quantile(np.array(perm_best, dtype=float), 0.95)) if perm_best else 1.0
    else:
        perm_p95 = 1.0
    if cfg.gate:
        gate_thr = perm_p95 + cfg.gate_margin
        kept = [x for x in kept if x.lift >= gate_thr]
    return kept, perm_p95, mined


@dataclass(frozen=True)
class Fold:
    k: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str


def build_folds(dates: List[str]) -> List[Fold]:
    if len(dates) < 120:
        raise RuntimeError("Not enough dates to build folds")
    n = len(dates)
    i1 = int(n * 0.60)
    j1 = int(n * 0.80)
    i2 = int(n * 0.70)
    j2 = int(n * 0.85)
    i3 = int(n * 0.80)
    j3 = n
    return [
        Fold(1, dates[0], dates[i1 - 1], dates[i1], dates[j1 - 1]),
        Fold(2, dates[0], dates[i2 - 1], dates[i2], dates[j2 - 1]),
        Fold(3, dates[0], dates[i3 - 1], dates[i3], dates[j3 - 1]),
    ]


def select_with_cluster_caps(rules: List[RuleStats], cfg: Config, want_n: int) -> Tuple[List[RuleStats], Dict[str, int]]:
    caps = {"VOL": cfg.cap_vol, "MOM": cfg.cap_mom, "TREND": cfg.cap_trend, "OTHER": cfg.cap_other, "PAIR": cfg.cap_pair}
    used = {k: 0 for k in caps.keys()}
    out: List[RuleStats] = []
    for rs in rules:
        cl = rs.rule.cluster if rs.rule.cluster in caps else "OTHER"
        if used[cl] >= caps[cl]:
            continue
        out.append(rs)
        used[cl] += 1
        if len(out) >= want_n:
            break
    return out, used


def _validate_entry_mode(entry_mode: str) -> str:
    mode = str(entry_mode).strip().lower()
    allowed = {"next_open", "next_close", "same_close"}
    if mode not in allowed:
        raise RuntimeError(f"Unsupported ENTRY_MODE={entry_mode!r}. Allowed: {sorted(allowed)}")
    return mode


def _score_rank_pct_by_date(df: pd.DataFrame, score_col: str = "score") -> pd.Series:
    return df.groupby("date")[score_col].rank(method="average", pct=True)


def add_delay_entry_forward_returns(df: pd.DataFrame, delay_days: int, entry_mode: str, horizons: Tuple[int, ...]) -> pd.DataFrame:
    out = df.copy()
    mode = _validate_entry_mode(entry_mode)
    if mode == "same_close":
        entry_shift = delay_days
        out["entry_px"] = out.groupby("symbol")["c"].shift(-entry_shift)
        entry_day_idx = entry_shift
    elif mode == "next_close":
        entry_shift = delay_days + 1
        out["entry_px"] = out.groupby("symbol")["c"].shift(-entry_shift)
        entry_day_idx = entry_shift
    else:
        entry_shift = delay_days + 1
        out["entry_px"] = out.groupby("symbol")["o"].shift(-entry_shift)
        entry_day_idx = entry_shift
    for h in horizons:
        exit_shift = entry_day_idx + (h - 1)
        exit_px = out.groupby("symbol")["c"].shift(-exit_shift)
        out[f"entry_ret_{h}d"] = (exit_px / out["entry_px"]) - 1.0
    return out


def build_delay_surface(scored_df: pd.DataFrame, delay_days: int, entry_mode: str, quantiles: Tuple[float, ...], horizons: Tuple[int, ...]) -> pd.DataFrame:
    out = scored_df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date", "symbol", "score", "c", "o"]).copy()
    out = add_delay_entry_forward_returns(out, delay_days=delay_days, entry_mode=entry_mode, horizons=horizons)
    out["score_rank_pct"] = _score_rank_pct_by_date(out, score_col="score")
    rows: List[Dict[str, object]] = []
    for q in quantiles:
        long_mask = out["score_rank_pct"] >= (1.0 - q)
        short_mask = out["score_rank_pct"] <= q
        for h in horizons:
            col = f"entry_ret_{h}d"
            long_vals = pd.to_numeric(out.loc[long_mask, col], errors="coerce").dropna()
            short_vals = pd.to_numeric(out.loc[short_mask, col], errors="coerce").dropna()
            long_mean = float(long_vals.mean()) if len(long_vals) else float("nan")
            short_mean = float(short_vals.mean()) if len(short_vals) else float("nan")
            spread_mean = long_mean - short_mean if pd.notna(long_mean) and pd.notna(short_mean) else float("nan")
            rows.append({"bucket": f"top_{int(round(q * 100.0))}pct", "side": "long", "delay_days": delay_days, "entry_mode": entry_mode, "horizon_days": h, "mean_ret": long_mean, "win_rate": float((long_vals > 0).mean()) if len(long_vals) else float("nan"), "n_obs": int(len(long_vals))})
            rows.append({"bucket": f"bottom_{int(round(q * 100.0))}pct", "side": "short", "delay_days": delay_days, "entry_mode": entry_mode, "horizon_days": h, "mean_ret": short_mean, "win_rate": float((short_vals < 0).mean()) if len(short_vals) else float("nan"), "n_obs": int(len(short_vals))})
            rows.append({"bucket": f"spread_{int(round(q * 100.0))}pct", "side": "long_short", "delay_days": delay_days, "entry_mode": entry_mode, "horizon_days": h, "mean_ret": spread_mean, "win_rate": float("nan"), "n_obs": int(min(len(long_vals), len(short_vals)))})
    return pd.DataFrame(rows)


def score_oos(oos_df: pd.DataFrame, rules: List[RuleStats]) -> Tuple[pd.DataFrame, int]:
    df = oos_df.copy()
    score = np.zeros(len(df), dtype=float)
    fires_total = 0
    for rs in rules:
        m = fire_rule(df, rs.rule).to_numpy(dtype=bool)
        fires_total += int(m.sum())
        if rs.w > 0:
            score += rs.w * m.astype(float)
    df["score"] = score
    return df, fires_total


def run_fold_library(full_df: pd.DataFrame, fold: Fold, cfg: Config, rng: random.Random) -> Dict[str, object]:
    train = full_df[(full_df["date"] >= fold.train_start) & (full_df["date"] <= fold.train_end)].copy()
    test = full_df[(full_df["date"] >= fold.test_start) & (full_df["date"] <= fold.test_end)].copy()
    br_train = compute_base_rate(train)
    br_test = compute_base_rate(test)
    mined, perm_p95, _ = mine_rules(train, cfg, rng)
    oos_stats: List[RuleStats] = []
    for st in mined:
        s2 = eval_rule(test, st.rule, br_test)
        s2.w = st.w
        oos_stats.append(s2)
    oos_big = [s for s in oos_stats if s.support >= cfg.oos_min_support]
    oos_pass = [s for s in oos_big if s.lift >= cfg.oos_min_lift]
    return {"fold": fold, "train_rows": int(train.shape[0]), "test_rows": int(test.shape[0]), "br_train": float(br_train), "br_test": float(br_test), "perm_p95": float(perm_p95), "mined_train": mined, "oos_big": oos_big, "oos_pass": oos_pass, "test_df": test}


def global_library_from_folds(fold_runs: List[Dict[str, object]]) -> Tuple[List[RuleStats], Dict[str, List[int]]]:
    sig_to_stats: Dict[str, List[RuleStats]] = {}
    sig_to_folds: Dict[str, List[int]] = {}
    for fr in fold_runs:
        fold: Fold = fr["fold"]  # type: ignore
        for st in fr["oos_pass"]:  # type: ignore
            sig_to_stats.setdefault(st.sig, []).append(st)
            sig_to_folds.setdefault(st.sig, []).append(fold.k)
    library: List[RuleStats] = []
    for sig, lst in sig_to_stats.items():
        folds = sig_to_folds.get(sig, [])
        if len(set(folds)) < 2:
            continue
        library.append(RuleStats(sig=sig, rule=lst[0].rule, support=int(np.mean([x.support for x in lst])), hits=int(np.mean([x.hits for x in lst])), prec=float(np.mean([x.prec for x in lst])), lift=float(np.mean([x.lift for x in lst])), mean=float(np.mean([x.mean for x in lst])), med=float(np.mean([x.med for x in lst])), p_pos=float(np.mean([x.p_pos for x in lst])), es5=float(np.mean([x.es5 for x in lst if not math.isnan(x.es5)])) if any(not math.isnan(x.es5) for x in lst) else float("nan"), w=float(np.mean([x.w for x in lst]))))
    library.sort(key=lambda s: (s.w, s.lift, s.support), reverse=True)
    return library, sig_to_folds


def build_runtime_selection(candidates: List[RuleStats], cfg: Config) -> Tuple[List[RuleStats], Dict[str, int]]:
    candidates = sorted(candidates, key=lambda s: (s.w, s.lift, s.support), reverse=True)
    return select_with_cluster_caps(candidates, cfg, cfg.runtime_topn_long)


def _print_delay_surface(surface_df: pd.DataFrame, fold_k: int, runtime_label: str) -> None:
    if surface_df.empty:
        print(f"[ALPHA/FOLD {fold_k}][{runtime_label}] delay surface empty")
        return
    print(f"[ALPHA/FOLD {fold_k}][{runtime_label}] entry-mode / delay / horizon surface")
    print(surface_df[["entry_mode", "delay_days", "bucket", "side", "horizon_days", "mean_ret", "win_rate", "n_obs"]].to_string(index=False))


def main() -> int:
    cfg = load_config()
    _validate_entry_mode(cfg.entry_mode)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng = random.Random(cfg.seed)

    print(f"[CFG] vendor=massive dataset_root={cfg.dataset_root}")
    print(f"[CFG] alpha_decay: debug={cfg.alpha_decay_debug} entry_mode={cfg.entry_mode} delay_days={cfg.delay_days} horizons={cfg.alpha_horizons} quantiles={cfg.alpha_quantiles}")
    print(f"[CFG] runtime_mode: USE_FOLD_LOCAL_RUNTIME={cfg.use_fold_local_runtime} FOLD_LOCAL_ALLOW_OOS_BIG_FALLBACK={cfg.fold_local_allow_oos_big_fallback}")

    raw = load_dataset(cfg)
    feat = add_features(raw, cfg)
    feat = add_event_labels(feat, cfg)
    feat = feat.dropna(subset=["atr_pct", "rv_10", "mom_1d", "mom_3d", "compression", "ema_dist", "ema_fast_slope", "ema_slow_slope", "fwd_ret", "event_up"]).copy()
    feat["event_up"] = feat["event_up"].astype(int)

    all_dates = sorted(feat["date"].unique().tolist())
    print(f"[DATA] pooled rows={len(feat)} dates={len(all_dates)} symbols={feat['symbol'].nunique()}")

    folds = build_folds(all_dates)
    fold_runs: List[Dict[str, object]] = []
    for fold in folds:
        print("\n" + "=" * 80)
        print(f"[LIB/FOLD {fold.k}] train={fold.train_start}..{fold.train_end}  test={fold.test_start}..{fold.test_end}")
        fr = run_fold_library(feat, fold, cfg, rng)
        print(f"[LIB/FOLD {fold.k}] train rows={fr['train_rows']} test rows={fr['test_rows']}")
        print(f"[LIB/FOLD {fold.k}] base_rate train={fr['br_train']:.4%} test={fr['br_test']:.4%} perm_p95={fr['perm_p95']:.3f}")
        print(f"[LIB/FOLD {fold.k}] mined_rules(train_kept)={len(fr['mined_train'])} OOS big-eval={len(fr['oos_big'])} pass(lift>={cfg.oos_min_lift})={len(fr['oos_pass'])}")
        fold_runs.append(fr)

    print("\n" + "=" * 80)
    print("[GLOBAL] Long Rule Library (OOS-filtered across folds)")
    pass_counts = []
    for fr in fold_runs:
        fold: Fold = fr["fold"]  # type: ignore
        pass_counts.append((fold.k, len(fr["oos_pass"]), len(fr["oos_big"])))  # type: ignore
    print("[GLOBAL] Fold pass counts: " + " | ".join([f"F{k}: {p}/{b}" for k, p, b in pass_counts]))

    library, _ = global_library_from_folds(fold_runs)
    print(f"[GLOBAL] library_size={len(library)} (fold_count>=2 & OOS filtered)")
    if not library:
        print("[GLOBAL] library empty -> global reusable library not available in this run.")

    for fr in fold_runs:
        fold: Fold = fr["fold"]  # type: ignore
        oos_df: pd.DataFrame = fr["test_df"]  # type: ignore
        source: List[RuleStats] = list(fr["oos_pass"])  # type: ignore
        runtime_label = "FOLD_LOCAL_OOS_PASS"
        if not source and cfg.fold_local_allow_oos_big_fallback:
            source = list(fr["oos_big"])  # type: ignore
            runtime_label = "FOLD_LOCAL_OOS_BIG"
        runtime_sel, caps_used = build_runtime_selection(source, cfg)
        print(f"[RUNTIME/FOLD {fold.k}] source={runtime_label} source_n={len(source)} selected={len(runtime_sel)}/{cfg.runtime_topn_long} caps_used={caps_used}")
        if not runtime_sel:
            print(f"[SIM/FOLD {fold.k}][{runtime_label}] runtime selection empty -> skip")
            continue
        scored, fires_total = score_oos(oos_df, runtime_sel)
        print(f"[SIM/FOLD {fold.k}][{runtime_label}] runtime_rules={len(runtime_sel)} scored_rows(score>0)={(scored['score'] > 0).sum()}/{len(scored)} fires_total={fires_total}")
        if cfg.alpha_decay_debug:
            try:
                surface = build_delay_surface(scored, cfg.delay_days, cfg.entry_mode, cfg.alpha_quantiles, cfg.alpha_horizons)
                _print_delay_surface(surface, fold.k, runtime_label)
            except Exception as e:
                print(f"[ALPHA/FOLD {fold.k}][{runtime_label}] delay surface failed: {e}")

    _log("[DONE] Ranker hold-sim completed.")
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        _log("[FATAL] unhandled exception")
        traceback.print_exc()
        rc = 1
    _press_enter_exit(rc)
