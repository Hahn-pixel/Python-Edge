from __future__ import annotations

"""
scripts/run_dynamic_universe.py
================================
Stage 2 — Dynamic universe selection runner.

Reads universe_snapshot.parquet, applies eligibility policy,
ranks by liquidity, outputs top-N selected symbols with full diagnostics.

Env vars:
    UNIVERSE_SNAPSHOT_PATH          path to universe_snapshot.parquet
                                    default: artifacts/daily_cycle/universe/universe_snapshot.parquet
    UNIVERSE_OUTPUT_DIR             output dir for dynamic universe artifacts
                                    default: artifacts/daily_cycle/universe
    DYNAMIC_UNIVERSE_TOP_N          number of symbols to select (default: 150)
    DYNAMIC_UNIVERSE_RANK_BY        rank column (default: median_dollar_volume_20d)
    PORTFOLIO_STATE_PATH            optional: path to portfolio_state.json for diagnostics
                                    default: artifacts/execution_loop/optimal/portfolio_state.json
    DYNAMIC_UNIVERSE_PINNED         pipe-separated pinned symbols, e.g. AAPL|MSFT

    Eligibility env vars (same as universe_builder):
    UNIVERSE_MIN_PRICE              (default: 7.50)
    UNIVERSE_MIN_MEDIAN_DOLLAR_VOL_20D (default: 20000000)
    UNIVERSE_MIN_HISTORY_DAYS       (default: 25)
    UNIVERSE_MAX_NAN_RATIO          (default: 0.02)
    UNIVERSE_MAX_MISSING_DAYS_20D   (default: 1)
    UNIVERSE_ALLOWED_TICKER_TYPES   pipe-separated, default: CS
    UNIVERSE_ALLOWED_PRIMARY_EXCHANGES pipe-separated, default: empty (allow all)
    UNIVERSE_REQUIRE_ACTIVE         0/1 (default: 1)

    PAUSE_ON_EXIT                   auto/0/1
"""

import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
for p in [ROOT, SRC_DIR]:
    ps = str(p)
    if ps not in sys.path:
        sys.path.insert(0, ps)

from python_edge.universe.eligibility import UniverseEligibilityPolicy
from python_edge.universe.dynamic_universe import (
    DynamicUniverseConfig,
    build_dynamic_universe,
    save_dynamic_universe,
)

PAUSE_ON_EXIT = str(os.getenv("PAUSE_ON_EXIT", "auto")).strip().lower()

# ---------------------------------------------------------------------------
# Pause / exit helpers
# ---------------------------------------------------------------------------

def _enable_line_buffering() -> None:
    for stream_name in ["stdout", "stderr"]:
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(line_buffering=True)
            except Exception:
                pass


def _should_pause() -> bool:
    if PAUSE_ON_EXIT in {"0", "false", "no", "off"}:
        return False
    if PAUSE_ON_EXIT in {"1", "true", "yes", "on"}:
        return True
    stdin_obj = getattr(sys, "stdin", None)
    stdout_obj = getattr(sys, "stdout", None)
    return bool(
        stdin_obj
        and stdout_obj
        and hasattr(stdin_obj, "isatty")
        and hasattr(stdout_obj, "isatty")
        and stdin_obj.isatty()
        and stdout_obj.isatty()
    )


def _safe_exit(code: int) -> None:
    if _should_pause():
        try:
            print("")
            print(f"[EXIT] code={code}")
            input("Press Enter to exit...")
        except Exception:
            pass
    raise SystemExit(code)


# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------

def _env_float(name: str, default: float) -> float:
    raw = str(os.getenv(name, str(default))).strip()
    try:
        return float(raw)
    except Exception:
        raise RuntimeError(f"Invalid float env {name}={raw!r}")


def _env_int(name: str, default: int) -> int:
    raw = str(os.getenv(name, str(default))).strip()
    try:
        return int(raw)
    except Exception:
        raise RuntimeError(f"Invalid int env {name}={raw!r}")


def _env_flag(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, "1" if default else "0")).strip().lower()
    return raw not in {"0", "false", "no", "off", ""}


def _env_tuple(name: str, default: tuple) -> tuple:
    raw = str(os.getenv(name, "")).strip()
    if not raw:
        return default
    items = [x.strip().upper() for x in raw.split("|") if x.strip()]
    return tuple(items)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config_from_env() -> DynamicUniverseConfig:
    snapshot_path = Path(
        str(os.getenv(
            "UNIVERSE_SNAPSHOT_PATH",
            str(ROOT / "artifacts" / "daily_cycle" / "universe" / "universe_snapshot.parquet"),
        )).strip()
    )
    output_dir = Path(
        str(os.getenv(
            "UNIVERSE_OUTPUT_DIR",
            str(ROOT / "artifacts" / "daily_cycle" / "universe"),
        )).strip()
    )
    top_n = _env_int("DYNAMIC_UNIVERSE_TOP_N", 150)
    rank_by = str(os.getenv("DYNAMIC_UNIVERSE_RANK_BY", "median_dollar_volume_20d")).strip()

    portfolio_state_raw = str(os.getenv(
        "PORTFOLIO_STATE_PATH",
        str(ROOT / "artifacts" / "execution_loop" / "optimal" / "portfolio_state.json"),
    )).strip()
    portfolio_state_path = Path(portfolio_state_raw) if portfolio_state_raw else None

    pinned_raw = str(os.getenv("DYNAMIC_UNIVERSE_PINNED", "")).strip()
    pinned_symbols = tuple(
        x.strip().upper() for x in pinned_raw.split("|") if x.strip()
    ) if pinned_raw else tuple()

    policy = UniverseEligibilityPolicy(
        min_price=_env_float("UNIVERSE_MIN_PRICE", 7.50),
        min_median_dollar_vol_20d=_env_float("UNIVERSE_MIN_MEDIAN_DOLLAR_VOL_20D", 20_000_000.0),
        min_history_days=_env_int("UNIVERSE_MIN_HISTORY_DAYS", 25),
        max_nan_ratio=_env_float("UNIVERSE_MAX_NAN_RATIO", 0.02),
        max_missing_days_20d=_env_int("UNIVERSE_MAX_MISSING_DAYS_20D", 1),
        allowed_ticker_types=_env_tuple("UNIVERSE_ALLOWED_TICKER_TYPES", ("CS",)),
        allowed_primary_exchanges=_env_tuple("UNIVERSE_ALLOWED_PRIMARY_EXCHANGES", tuple()),
        allowed_exchanges=_env_tuple("UNIVERSE_ALLOWED_EXCHANGES", tuple()),
        require_active=_env_flag("UNIVERSE_REQUIRE_ACTIVE", True),
    )

    return DynamicUniverseConfig(
        snapshot_path=snapshot_path,
        output_dir=output_dir,
        top_n=top_n,
        rank_by=rank_by,
        policy=policy,
        pinned_symbols=pinned_symbols,
        portfolio_state_path=portfolio_state_path,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    _enable_line_buffering()

    cfg = load_config_from_env()

    print(f"[CFG] snapshot_path={cfg.snapshot_path}")
    print(f"[CFG] output_dir={cfg.output_dir}")
    print(f"[CFG] top_n={cfg.top_n} rank_by={cfg.rank_by}")
    print(f"[CFG] portfolio_state_path={cfg.portfolio_state_path}")
    print(f"[CFG] pinned_symbols={list(cfg.pinned_symbols)}")
    print(f"[CFG] policy.min_price={cfg.policy.min_price}")
    print(f"[CFG] policy.min_median_dollar_vol_20d={cfg.policy.min_median_dollar_vol_20d:,.0f}")
    print(f"[CFG] policy.min_history_days={cfg.policy.min_history_days}")
    print(f"[CFG] policy.allowed_ticker_types={cfg.policy.allowed_ticker_types}")

    if not cfg.snapshot_path.exists():
        print(f"[ERROR] snapshot not found: {cfg.snapshot_path}")
        print("[ERROR] Run run_universe_builder.py first.")
        return 1

    result = build_dynamic_universe(cfg)
    paths = save_dynamic_universe(result, cfg.output_dir)

    print(f"[OK] parquet={paths['parquet']}")
    print(f"[OK] selected_csv={paths['selected_csv']}")
    print(f"[OK] dropped_csv={paths['dropped_csv']}")
    print(f"[OK] summary={paths['summary']}")
    print(f"[FINAL] dynamic_universe complete selected={result.counters['selected_total']} dropped={result.counters['dropped_total']}")

    # print top-10 selected for quick visual check
    top10 = result.selected.head(10)
    if not top10.empty:
        print("[TOP10]")
        for _, row in top10.iterrows():
            dvol = float(row.get("median_dollar_volume_20d", 0))
            print(f"  rank={int(row.get('selected_rank',0))} symbol={row['symbol']} dvol_20d={dvol:,.0f} close={row.get('close', 0):.2f}")

    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 1
    _safe_exit(rc)
