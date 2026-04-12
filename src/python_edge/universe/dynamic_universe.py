from __future__ import annotations

"""
python_edge.universe.dynamic_universe
======================================
Stage 2 — Dynamic tradable universe selection.

Reads universe_snapshot.parquet (output of universe_builder / run_universe_builder),
applies eligibility policy (from eligibility.py), ranks by liquidity,
and produces explicit artifacts:
    dynamic_universe.parquet       — selected symbols with ranks
    dynamic_universe_dropped.csv   — dropped symbols with reason
    dynamic_universe_selected.csv  — selected symbols (human-readable)
    dynamic_universe_summary.json  — counters + config snapshot

Design rules:
  - No silent assumptions: every dropped name is logged with reason.
  - top_n is explicit and enforced.
  - Ranking is deterministic (sort by median_dollar_volume_20d DESC, symbol ASC).
  - Portfolio state symbols are surfaced in diagnostics but do NOT force-include.
  - This module has no broker / execution dependency.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from python_edge.universe.eligibility import (
    UniverseEligibilityPolicy,
    apply_eligibility_policy,
    log_eligibility_counters,
    save_eligibility_report,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DynamicUniverseConfig:
    """
    Configuration for dynamic universe selection.

    Args:
        snapshot_path       : path to universe_snapshot.parquet
        output_dir          : directory for output artifacts
        top_n               : number of symbols to select
        rank_by             : primary sort column for ranking
        rank_by_fallback    : fallback sort column if primary missing
        policy              : eligibility policy to apply
        pinned_symbols      : symbols always included if they pass eligibility
                              (e.g. current portfolio names for continuity)
        min_pin_pass_policy : if True, pinned symbols still must pass policy
        portfolio_state_path: optional path to portfolio_state.json for diagnostics
    """
    snapshot_path: Path
    output_dir: Path
    top_n: int = 150
    rank_by: str = "median_dollar_volume_20d"
    rank_by_fallback: str = "dollar_volume_1d"
    policy: UniverseEligibilityPolicy = field(default_factory=UniverseEligibilityPolicy)
    pinned_symbols: Tuple[str, ...] = tuple()
    min_pin_pass_policy: bool = True
    portfolio_state_path: Optional[Path] = None


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class DynamicUniverseResult:
    selected: pd.DataFrame          # top_n eligible rows, ranked
    dropped: pd.DataFrame           # ineligible rows with drop_reason
    counters: Dict[str, int]        # eligibility + selection counters
    summary: Dict                   # full JSON-serializable summary
    selected_symbols: List[str]     # convenience list


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_portfolio_symbols(path: Optional[Path]) -> List[str]:
    """Load current position symbols from portfolio_state.json."""
    if path is None or not Path(path).exists():
        return []
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        positions = data.get("positions", {})
        return [str(s).upper() for s in positions.keys() if str(s).strip()]
    except Exception:
        return []


def _load_snapshot(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"[DYN_UNIVERSE] snapshot not found: {path}")
    df = pd.read_parquet(path)
    if df.empty:
        raise ValueError(f"[DYN_UNIVERSE] snapshot is empty: {path}")

    # normalize symbol column
    if "symbol" not in df.columns and "ticker" in df.columns:
        df = df.rename(columns={"ticker": "symbol"})
    if "symbol" not in df.columns:
        raise ValueError(f"[DYN_UNIVERSE] snapshot missing 'symbol' column. cols={list(df.columns)}")
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()

    # ensure required eligibility columns exist with safe defaults
    defaults = {
        "active": True,
        "ticker_type": "CS",
        "primary_exchange": "",
        "close": 0.0,
        "median_dollar_volume_20d": 0.0,
        "dollar_volume_1d": 0.0,
        "history_days": 0,
        "nan_ratio": 1.0,
        "missing_days_20d": 999,
    }
    for col, default in defaults.items():
        if col not in df.columns:
            print(f"[DYN_UNIVERSE][WARN] snapshot missing column '{col}', defaulting to {default!r}")
            df[col] = default
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default) if col not in ("active", "ticker_type", "primary_exchange") else df[col]

    return df.drop_duplicates(subset=["symbol"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def build_dynamic_universe(cfg: DynamicUniverseConfig) -> DynamicUniverseResult:
    """
    Main entry point: load snapshot, apply policy, rank, select top_n.

    Returns DynamicUniverseResult with all artifacts ready to save.
    """
    print(f"[DYN_UNIVERSE] snapshot={cfg.snapshot_path}")
    print(f"[DYN_UNIVERSE] top_n={cfg.top_n} rank_by={cfg.rank_by}")
    print(f"[DYN_UNIVERSE] pinned_symbols={list(cfg.pinned_symbols)}")

    # 1. load
    df = _load_snapshot(cfg.snapshot_path)
    print(f"[DYN_UNIVERSE] loaded rows={len(df)}")

    # 2. load portfolio symbols for diagnostics
    portfolio_symbols = _load_portfolio_symbols(cfg.portfolio_state_path)
    if portfolio_symbols:
        print(f"[DYN_UNIVERSE] portfolio_symbols={portfolio_symbols}")

    # 3. apply eligibility
    df_flagged, counters = apply_eligibility_policy(df, cfg.policy)
    log_eligibility_counters(counters, prefix="dynamic_universe")

    eligible = df_flagged.loc[df_flagged["eligible"]].copy()
    dropped = df_flagged.loc[~df_flagged["eligible"]].copy()

    # 4. rank eligible by liquidity
    rank_col = cfg.rank_by if cfg.rank_by in eligible.columns else cfg.rank_by_fallback
    if rank_col not in eligible.columns:
        rank_col = "close"  # last resort
        print(f"[DYN_UNIVERSE][WARN] rank_by column missing, falling back to 'close'")

    eligible = eligible.sort_values(
        [rank_col, "symbol"],
        ascending=[False, True],
    ).reset_index(drop=True)
    eligible["liquidity_rank"] = range(1, len(eligible) + 1)

    # 5. handle pinned symbols
    pinned = [s.upper() for s in cfg.pinned_symbols if s.strip()]
    if pinned:
        pinned_eligible = eligible.loc[eligible["symbol"].isin(pinned)].copy()
        pinned_missing = [s for s in pinned if s not in eligible["symbol"].values]
        if pinned_missing:
            print(f"[DYN_UNIVERSE][WARN] pinned symbols not in eligible: {pinned_missing}")
        if not pinned_eligible.empty:
            print(f"[DYN_UNIVERSE] pinned eligible: {pinned_eligible['symbol'].tolist()}")

    # 6. select top_n
    selected = eligible.head(max(1, cfg.top_n)).copy()
    selected["selected_rank"] = range(1, len(selected) + 1)

    # 7. diagnostics: portfolio symbols coverage
    portfolio_in_selected = [s for s in portfolio_symbols if s in selected["symbol"].values]
    portfolio_not_in_selected = [s for s in portfolio_symbols if s not in selected["symbol"].values]

    counters_ext = dict(counters)
    counters_ext["top_n_requested"] = cfg.top_n
    counters_ext["selected_total"] = int(len(selected))
    counters_ext["dropped_total"] = int(len(dropped))
    counters_ext["portfolio_symbols_total"] = int(len(portfolio_symbols))
    counters_ext["portfolio_in_selected"] = int(len(portfolio_in_selected))
    counters_ext["portfolio_not_in_selected"] = int(len(portfolio_not_in_selected))

    if portfolio_not_in_selected:
        print(f"[DYN_UNIVERSE][DIAG] portfolio symbols NOT in selected: {portfolio_not_in_selected}")

    # 8. build summary
    latest_date = ""
    for col in ("trade_date", "as_of_date", "date", "session_date"):
        if col in selected.columns and not selected.empty:
            latest_date = str(selected[col].iloc[0])
            break

    summary = {
        "ts_utc": _utc_now_iso(),
        "snapshot_path": str(cfg.snapshot_path),
        "output_dir": str(cfg.output_dir),
        "top_n": cfg.top_n,
        "rank_by": rank_col,
        "latest_date": latest_date,
        "pinned_symbols": list(cfg.pinned_symbols),
        "min_pin_pass_policy": cfg.min_pin_pass_policy,
        "portfolio_symbols": portfolio_symbols,
        "portfolio_in_selected": portfolio_in_selected,
        "portfolio_not_in_selected": portfolio_not_in_selected,
        "policy": asdict(cfg.policy),
        **counters_ext,
    }

    selected_symbols = selected["symbol"].tolist()
    print(f"[DYN_UNIVERSE] selected={len(selected_symbols)} symbols")

    return DynamicUniverseResult(
        selected=selected,
        dropped=dropped,
        counters=counters_ext,
        summary=summary,
        selected_symbols=selected_symbols,
    )


# ---------------------------------------------------------------------------
# Save artifacts
# ---------------------------------------------------------------------------

def save_dynamic_universe(result: DynamicUniverseResult, output_dir: Path) -> Dict[str, Path]:
    """
    Save all artifacts to output_dir.

    Returns dict of written paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / "dynamic_universe.parquet"
    selected_csv_path = output_dir / "dynamic_universe_selected.csv"
    dropped_csv_path = output_dir / "dynamic_universe_dropped.csv"
    summary_path = output_dir / "dynamic_universe_summary.json"

    result.selected.to_parquet(parquet_path, index=False)

    # human-readable selected CSV
    selected_cols = [
        "symbol", "selected_rank", "liquidity_rank",
        "median_dollar_volume_20d", "dollar_volume_1d",
        "close", "ticker_type", "primary_exchange",
        "history_days", "nan_ratio", "missing_days_20d",
    ]
    export_selected_cols = [c for c in selected_cols if c in result.selected.columns]
    result.selected[export_selected_cols].to_csv(selected_csv_path, index=False)

    # human-readable dropped CSV
    dropped_cols = [
        "symbol", "drop_reason",
        "ticker_type", "primary_exchange",
        "close", "median_dollar_volume_20d",
        "history_days", "nan_ratio", "missing_days_20d", "active",
    ]
    export_dropped_cols = [c for c in dropped_cols if c in result.dropped.columns]
    result.dropped[export_dropped_cols].to_csv(dropped_csv_path, index=False)

    summary_path.write_text(
        json.dumps(result.summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "parquet": parquet_path,
        "selected_csv": selected_csv_path,
        "dropped_csv": dropped_csv_path,
        "summary": summary_path,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "DynamicUniverseConfig",
    "DynamicUniverseResult",
    "build_dynamic_universe",
    "save_dynamic_universe",
]
