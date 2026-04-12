from __future__ import annotations

"""
python_edge.universe.eligibility
================================
Standalone eligibility policy layer.

Import from anywhere:
    from python_edge.universe.eligibility import (
        UniverseEligibilityPolicy,
        EligibilityProfile,
        load_policy_from_profile,
        apply_eligibility_policy,
        save_eligibility_report,
    )

Does NOT import universe_builder — zero circular dependency risk.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Policy dataclass (canonical definition — universe_builder re-imports this)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UniverseEligibilityPolicy:
    """
    All eligibility rules in one immutable record.

    Rules applied in apply_eligibility_policy():
      - require_active        : ticker must be active
      - allowed_ticker_types  : e.g. ("CS",) for common stock only
      - allowed_primary_exchanges : e.g. ("NYSE","NASDAQ") — empty = allow all
      - allowed_exchanges     : secondary exchange filter — empty = allow all
      - min_price             : minimum close price
      - min_median_dollar_vol_20d : minimum 20-day median dollar volume
      - min_history_days      : minimum trading history rows
      - max_nan_ratio         : max fraction of NaN in critical columns (20d window)
      - max_missing_days_20d  : max calendar-business-day gaps in last 20 sessions
    """
    min_price: float = 7.50
    min_median_dollar_vol_20d: float = 20_000_000.0
    min_history_days: int = 25
    max_nan_ratio: float = 0.02
    max_missing_days_20d: int = 1
    allowed_ticker_types: Tuple[str, ...] = ("CS",)
    allowed_primary_exchanges: Tuple[str, ...] = tuple()
    allowed_exchanges: Tuple[str, ...] = tuple()
    require_active: bool = True


# ---------------------------------------------------------------------------
# Profile registry
# ---------------------------------------------------------------------------

class EligibilityProfile:
    """
    Named policy presets.

    Usage:
        policy = EligibilityProfile.get("us_core")
    """

    _PROFILES: Dict[str, UniverseEligibilityPolicy] = {}

    @classmethod
    def register(cls, name: str, policy: UniverseEligibilityPolicy) -> None:
        cls._PROFILES[name.lower().strip()] = policy

    @classmethod
    def get(cls, name: str) -> UniverseEligibilityPolicy:
        key = name.lower().strip()
        if key not in cls._PROFILES:
            known = sorted(cls._PROFILES.keys())
            raise ValueError(f"Unknown eligibility profile: {name!r}. Known: {known}")
        return cls._PROFILES[key]

    @classmethod
    def list_profiles(cls) -> List[str]:
        return sorted(cls._PROFILES.keys())


# ------------------------------------------------------------------
# Built-in profile definitions
# ------------------------------------------------------------------

# us_core — default production profile
# CS only, $7.50+, $20M+ daily dollar volume, strict history
EligibilityProfile.register(
    "us_core",
    UniverseEligibilityPolicy(
        min_price=7.50,
        min_median_dollar_vol_20d=20_000_000.0,
        min_history_days=25,
        max_nan_ratio=0.02,
        max_missing_days_20d=1,
        allowed_ticker_types=("CS",),
        allowed_primary_exchanges=tuple(),
        allowed_exchanges=tuple(),
        require_active=True,
    ),
)

# us_extended — broader universe, lower liquidity bar, still CS only
EligibilityProfile.register(
    "us_extended",
    UniverseEligibilityPolicy(
        min_price=5.00,
        min_median_dollar_vol_20d=5_000_000.0,
        min_history_days=20,
        max_nan_ratio=0.05,
        max_missing_days_20d=2,
        allowed_ticker_types=("CS",),
        allowed_primary_exchanges=tuple(),
        allowed_exchanges=tuple(),
        require_active=True,
    ),
)

# etf_proxy — for ETF context features only, not for alpha / execution
# lower price floor, high liquidity requirement, ETF type
EligibilityProfile.register(
    "etf_proxy",
    UniverseEligibilityPolicy(
        min_price=5.00,
        min_median_dollar_vol_20d=50_000_000.0,
        min_history_days=25,
        max_nan_ratio=0.02,
        max_missing_days_20d=1,
        allowed_ticker_types=("ETF",),
        allowed_primary_exchanges=tuple(),
        allowed_exchanges=tuple(),
        require_active=True,
    ),
)

# adr_proxy — ADR instruments as international context features
EligibilityProfile.register(
    "adr_proxy",
    UniverseEligibilityPolicy(
        min_price=5.00,
        min_median_dollar_vol_20d=10_000_000.0,
        min_history_days=20,
        max_nan_ratio=0.05,
        max_missing_days_20d=2,
        allowed_ticker_types=("ADRC",),
        allowed_primary_exchanges=tuple(),
        allowed_exchanges=tuple(),
        require_active=True,
    ),
)

# commodity_etf — commodity ETF proxies (GLD, SLV, USO, etc.)
EligibilityProfile.register(
    "commodity_etf",
    UniverseEligibilityPolicy(
        min_price=5.00,
        min_median_dollar_vol_20d=20_000_000.0,
        min_history_days=25,
        max_nan_ratio=0.02,
        max_missing_days_20d=1,
        allowed_ticker_types=("ETF",),
        allowed_primary_exchanges=tuple(),
        allowed_exchanges=tuple(),
        require_active=True,
    ),
)


# ---------------------------------------------------------------------------
# Profile loader (env-aware factory)
# ---------------------------------------------------------------------------

def load_policy_from_profile(profile_name: str) -> UniverseEligibilityPolicy:
    """
    Load a named policy preset.

    Args:
        profile_name: one of EligibilityProfile.list_profiles()

    Returns:
        UniverseEligibilityPolicy (frozen dataclass)

    Raises:
        ValueError if profile_name is unknown.
    """
    return EligibilityProfile.get(profile_name)


# ---------------------------------------------------------------------------
# Core filter logic (no universe_builder dependency)
# ---------------------------------------------------------------------------

def _passes_allowed(value: str, allowed: Iterable[str]) -> bool:
    allowed_norm = [str(x).strip().upper() for x in allowed if str(x).strip()]
    if not allowed_norm:
        return True
    return str(value or "").strip().upper() in set(allowed_norm)


def apply_eligibility_policy(
    df: pd.DataFrame,
    policy: UniverseEligibilityPolicy,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Apply eligibility policy to a DataFrame.

    Required columns (at minimum):
        symbol, active, ticker_type, primary_exchange,
        close, median_dollar_volume_20d, history_days,
        nan_ratio, missing_days_20d

    Returns:
        (df_with_flags, counters_dict)

    df_with_flags adds boolean columns:
        passes_active, passes_ticker_type, passes_primary_exchange,
        passes_exchange, passes_price, passes_liquidity,
        passes_history, passes_nan_ratio, passes_missing_days,
        eligible (all must be True), drop_reason (str, empty if eligible)

    counters_dict keys:
        candidates_total, eligible_total,
        dropped_active, dropped_ticker_type, dropped_primary_exchange,
        dropped_exchange, dropped_price, dropped_liquidity,
        dropped_history, dropped_nan_ratio, dropped_missing_days
    """
    out = df.copy()

    # --- individual rule flags ---
    if policy.require_active:
        out["passes_active"] = out["active"].fillna(False).astype(bool)
    else:
        out["passes_active"] = True

    out["passes_ticker_type"] = (
        out["ticker_type"].astype(str).str.upper()
        .map(lambda x: _passes_allowed(x, policy.allowed_ticker_types))
    )

    out["passes_primary_exchange"] = (
        out["primary_exchange"].astype(str).str.upper()
        .map(lambda x: _passes_allowed(x, policy.allowed_primary_exchanges))
    )

    exchange_col = (
        out["primary_exchange"]
        if "primary_exchange" in out.columns
        else pd.Series([""] * len(out), index=out.index)
    )
    out["passes_exchange"] = (
        exchange_col.astype(str).str.upper()
        .map(lambda x: _passes_allowed(x, policy.allowed_exchanges))
    )

    out["passes_price"] = (
        pd.to_numeric(out["close"], errors="coerce").fillna(0.0)
        >= float(policy.min_price)
    )

    out["passes_liquidity"] = (
        pd.to_numeric(out["median_dollar_volume_20d"], errors="coerce").fillna(0.0)
        >= float(policy.min_median_dollar_vol_20d)
    )

    out["passes_history"] = (
        pd.to_numeric(out["history_days"], errors="coerce").fillna(0).astype(int)
        >= int(policy.min_history_days)
    )

    out["passes_nan_ratio"] = (
        pd.to_numeric(out["nan_ratio"], errors="coerce").fillna(1.0)
        <= float(policy.max_nan_ratio)
    )

    out["passes_missing_days"] = (
        pd.to_numeric(out["missing_days_20d"], errors="coerce").fillna(999).astype(int)
        <= int(policy.max_missing_days_20d)
    )

    rule_cols = [
        "passes_active",
        "passes_ticker_type",
        "passes_primary_exchange",
        "passes_exchange",
        "passes_price",
        "passes_liquidity",
        "passes_history",
        "passes_nan_ratio",
        "passes_missing_days",
    ]
    out["eligible"] = out[rule_cols].all(axis=1)

    def _drop_reason(row: pd.Series) -> str:
        if bool(row.get("eligible", False)):
            return ""
        for key in rule_cols:
            if not bool(row.get(key, False)):
                return key.replace("passes_", "")
        return "unknown"

    out["drop_reason"] = out.apply(_drop_reason, axis=1)

    counters: Dict[str, int] = {
        "candidates_total": int(len(out)),
        "eligible_total": int(out["eligible"].sum()),
        "dropped_active": int((~out["passes_active"]).sum()),
        "dropped_ticker_type": int((~out["passes_ticker_type"]).sum()),
        "dropped_primary_exchange": int((~out["passes_primary_exchange"]).sum()),
        "dropped_exchange": int((~out["passes_exchange"]).sum()),
        "dropped_price": int((~out["passes_price"]).sum()),
        "dropped_liquidity": int((~out["passes_liquidity"]).sum()),
        "dropped_history": int((~out["passes_history"]).sum()),
        "dropped_nan_ratio": int((~out["passes_nan_ratio"]).sum()),
        "dropped_missing_days": int((~out["passes_missing_days"]).sum()),
    }

    return out, counters


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------

def save_eligibility_report(
    df_with_flags: pd.DataFrame,
    counters: Dict[str, int],
    output_dir: Path,
    policy: Optional[UniverseEligibilityPolicy] = None,
    prefix: str = "universe",
) -> Dict[str, Path]:
    """
    Save human-readable eligibility artifacts:
        {prefix}_eligibility_dropped.csv   — dropped rows with drop_reason
        {prefix}_eligibility_eligible.csv  — eligible rows
        {prefix}_eligibility_counters.json — summary counters + policy snapshot

    Returns dict of written paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_cols = ["symbol", "drop_reason", "ticker_type", "primary_exchange",
                 "close", "median_dollar_volume_20d", "history_days",
                 "nan_ratio", "missing_days_20d", "active"]
    export_cols = [c for c in base_cols if c in df_with_flags.columns]

    dropped = df_with_flags.loc[~df_with_flags["eligible"], export_cols].copy()
    eligible = df_with_flags.loc[df_with_flags["eligible"], export_cols].copy()

    dropped_path = output_dir / f"{prefix}_eligibility_dropped.csv"
    eligible_path = output_dir / f"{prefix}_eligibility_eligible.csv"
    counters_path = output_dir / f"{prefix}_eligibility_counters.json"

    dropped.to_csv(dropped_path, index=False)
    eligible.to_csv(eligible_path, index=False)

    payload: Dict = dict(counters)
    if policy is not None:
        payload["policy"] = asdict(policy)
    counters_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "dropped": dropped_path,
        "eligible": eligible_path,
        "counters": counters_path,
    }


def log_eligibility_counters(counters: Dict[str, int], prefix: str = "") -> None:
    """
    Print eligibility counters in a structured [ELIG] format.
    Useful for any script that runs apply_eligibility_policy().
    """
    tag = f"[ELIG][{prefix}]" if prefix else "[ELIG]"
    total = counters.get("candidates_total", 0)
    eligible = counters.get("eligible_total", 0)
    dropped = total - eligible
    print(f"{tag} candidates={total} eligible={eligible} dropped={dropped}")
    drop_keys = [k for k in counters if k.startswith("dropped_") and counters[k] > 0]
    for k in drop_keys:
        print(f"{tag}   {k}={counters[k]}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "UniverseEligibilityPolicy",
    "EligibilityProfile",
    "load_policy_from_profile",
    "apply_eligibility_policy",
    "save_eligibility_report",
    "log_eligibility_counters",
]
