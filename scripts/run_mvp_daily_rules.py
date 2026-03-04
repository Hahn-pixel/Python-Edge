# scripts/run_mvp_daily_rules.py
# Double-click runnable. Never auto-closes.
#
# MVP:
# 1) Load massive 1D FULL files for ETF-first universe
# 2) Build daily features + forward labels (5d)
# 3) Mine hundreds of rules on pooled cross-section
# 4) Purged walk-forward (6 folds)
# 5) Portfolio overlay (max 5 positions, long/short)
# 6) Print report
#
# Env flags:
#   DATA_START / DATA_END (must match your downloaded FULL file names)
#   DATA_OUT_DIR (optional): defaults to data/raw/massive_dataset
#   MVP_COST_BPS (optional) default 10
#   MVP_MAX_POS (optional) default 5
#   MVP_MAX_LONG (optional) default 3
#   MVP_MAX_SHORT (optional) default 2

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import List

import pandas as pd


def _press_enter_exit(code: int) -> None:
    try:
        print(f"\n[EXIT] code={code}")
        input("Press Enter to exit...")
    except Exception:
        pass
    raise SystemExit(code)


def _add_src_to_syspath() -> None:
    # Make imports work when running as a script without installing the package.
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
    if not p.exists():
        raise FileNotFoundError(f"Missing universe file: {p}")
    txt = p.read_text(encoding="utf-8", errors="ignore")
    out = []
    for line in txt.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s.upper())
    return out


def main() -> int:
    _add_src_to_syspath()

    # imports after sys.path fix
    from python_edge.data.ingest_aggs import load_aggs, to_daily_index
    from python_edge.features.build_features_daily import DailyFeatureConfig, build_daily_features
    from python_edge.backtest.walkforward_rules import WFConfig, walkforward_rule_selection
    from python_edge.portfolio.portfolio_overlay import PortfolioConfig, build_portfolio_oos
    from python_edge.reports.report_text import summarize_portfolio

    root = Path(__file__).resolve().parents[1]
    start = _env_str("DATA_START", "")
    end = _env_str("DATA_END", "")
    if not start or not end:
        raise RuntimeError(
            "Missing DATA_START/DATA_END (must match your downloaded FULL file names). "
            "Example: DATA_START=2023-01-01 DATA_END=2026-02-28"
        )

    dataset_root = Path(_env_str("DATA_OUT_DIR", str(root / "data" / "raw" / "massive_dataset")))

    tickers = _load_universe(root)
    print(f"[CFG] vendor=massive dataset_root={dataset_root}")
    print(f"[CFG] universe={len(tickers)} start={start} end={end}")

    panels = []
    for t in tickers:
        r = load_aggs(dataset_root=dataset_root, symbol=t, tf="1d", start=start, end=end, prefer_full=True)
        df = to_daily_index(r.df)
        if df.empty:
            print(f"[WARN] {t} empty 1d")
            continue

        keep = ["date", "o", "h", "l", "c", "v"]
        df = df[keep].copy()
        df["symbol"] = t

        df = build_daily_features(df, DailyFeatureConfig())
        panels.append(df)

    if not panels:
        raise RuntimeError("No data panels loaded. Check DATA_OUT_DIR and FULL files naming.")

    all_df = pd.concat(panels, axis=0, ignore_index=True)
    all_df = all_df.sort_values(["date", "symbol"]).reset_index(drop=True)

    target_col = "fwd_5d_ret"
    feature_cols = [
        "mom_1d", "mom_3d", "mom_5d", "mom_10d", "mom_20d",
        "ema_dist", "ema_fast_slope", "ema_slow_slope",
        "atr_pct", "rv_10", "compression",
    ]
    feature_cols = [c for c in feature_cols if c in all_df.columns]

    all_df = all_df.dropna(subset=feature_cols + [target_col]).copy()
    print(f"[DATA] pooled rows={len(all_df)} dates={all_df['date'].nunique()} symbols={all_df['symbol'].nunique()}")

    wf_cfg = WFConfig(
        n_folds=6,
        train_days=420,
        test_days=90,
        purge_days=10,
        max_rules=250,
        min_trades=60,
        seed=7,
    )

    fold_res, fold_rules, fold_scores = walkforward_rule_selection(
        df_all=all_df,
        feature_cols=feature_cols,
        target_col=target_col,
        cfg=wf_cfg,
    )

    print("\n=== WALK-FORWARD SUMMARY (rule library quality proxy) ===")
    if not fold_res:
        print("[WARN] No folds produced (insufficient dates). Check DATA_START/DATA_END and coverage.")
    for fr in fold_res:
        print(
            f"[FOLD {fr.fold_id}] "
            f"train={fr.train_start}..{fr.train_end} "
            f"test={fr.test_start}..{fr.test_end} "
            f"rules={fr.n_rules} oos_mean={fr.oos_mean:.6f} oos_es5={fr.oos_es5:.6f} n={fr.oos_n}"
        )

    cost_bps = _env_float("MVP_COST_BPS", 10.0)
    max_pos = _env_int("MVP_MAX_POS", 5)
    max_long = _env_int("MVP_MAX_LONG", 3)
    max_short = _env_int("MVP_MAX_SHORT", 2)

    print("\n=== PORTFOLIO OOS (per fold) ===")
    for fr in fold_res:
        rules = fold_rules.get(fr.fold_id, [])
        scores = fold_scores.get(fr.fold_id, {})
        if not rules:
            print(f"[FOLD {fr.fold_id}] no rules -> skip")
            continue

        df_oos = all_df[(all_df["date"] >= fr.test_start) & (all_df["date"] <= fr.test_end)].copy()
        pf = build_portfolio_oos(
            df_oos=df_oos,
            rules=rules,
            scores=scores,
            target_col=target_col,
            cfg=PortfolioConfig(
                max_positions=max_pos,
                max_long=max_long,
                max_short=max_short,
                cost_bps=cost_bps,
                hold_days=5,
            ),
        )
        print(summarize_portfolio(pf, f"FOLD {fr.fold_id}"))

    print("[DONE] MVP run completed.")
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 1
    _press_enter_exit(int(rc))