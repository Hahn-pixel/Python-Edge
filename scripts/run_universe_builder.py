from __future__ import annotations

import json
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

from python_edge.universe.universe_builder import (
    build_and_save_universe_snapshot,
    load_config_from_env,
)
from python_edge.universe.eligibility import (
    log_eligibility_counters,
    save_eligibility_report,
    apply_eligibility_policy,
)

PAUSE_ON_EXIT = str(os.getenv("PAUSE_ON_EXIT", "auto")).strip().lower()


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


def main() -> int:
    _enable_line_buffering()

    config = load_config_from_env(ROOT)
    output_dir = config.output_dir

    print(f"[CFG] output_dir={output_dir}")
    print(f"[CFG] universe_profile={config.universe_profile}")
    print(f"[CFG] target_size={config.target_size} top_n={config.top_n}")
    print(f"[CFG] history_lookback_days={config.history_lookback_days}")
    print(f"[CFG] rebalance_freq={config.rebalance_freq} reuse_last={config.reuse_last}")
    print(f"[CFG] eligibility.min_price={config.eligibility.min_price}")
    print(f"[CFG] eligibility.min_median_dollar_vol_20d={config.eligibility.min_median_dollar_vol_20d:,.0f}")
    print(f"[CFG] eligibility.min_history_days={config.eligibility.min_history_days}")
    print(f"[CFG] eligibility.allowed_ticker_types={config.eligibility.allowed_ticker_types}")

    # --- build universe ---
    from python_edge.universe.universe_builder import build_universe_snapshot

    selected, summary, eligible_df = build_universe_snapshot(config)

    # --- re-apply policy to get fresh counters for logging ---
    # eligible_df already has all flags from build_universe_snapshot
    # We re-derive counters from its columns
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

    existing_rule_cols = [c for c in rule_cols if c in eligible_df.columns]
    if existing_rule_cols:
        counters = {
            "candidates_total": int(len(eligible_df)),
            "eligible_total": int(eligible_df["eligible"].sum()) if "eligible" in eligible_df.columns else 0,
        }
        for col in rule_cols:
            key = "dropped_" + col.replace("passes_", "")
            if col in eligible_df.columns:
                counters[key] = int((~eligible_df[col]).sum())
            else:
                counters[key] = 0
    else:
        # fallback: re-apply policy from scratch
        _, counters = apply_eligibility_policy(eligible_df, config.eligibility)

    # --- log counters ---
    log_eligibility_counters(counters, prefix="universe_builder")

    # --- save outputs ---
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = output_dir / "universe_snapshot.parquet"
    eligible_path = output_dir / "universe_eligibility_debug.parquet"
    summary_path = output_dir / "universe_summary.json"

    selected.to_parquet(snapshot_path, index=False)
    eligible_df.to_parquet(eligible_path, index=False)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # --- save human-readable eligibility report ---
    report_paths = save_eligibility_report(
        df_with_flags=eligible_df,
        counters=counters,
        output_dir=output_dir,
        policy=config.eligibility,
        prefix="universe",
    )

    print(f"[OK] snapshot={snapshot_path} rows={len(selected)}")
    print(f"[OK] eligibility_debug={eligible_path}")
    print(f"[OK] summary={summary_path}")
    print(f"[OK] eligibility_dropped={report_paths['dropped']}")
    print(f"[OK] eligibility_eligible={report_paths['eligible']}")
    print(f"[OK] eligibility_counters={report_paths['counters']}")
    print(f"[FINAL] universe builder complete snapshot={snapshot_path} summary={summary_path}")

    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 1
    _safe_exit(rc)
