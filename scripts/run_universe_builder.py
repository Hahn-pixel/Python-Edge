from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from python_edge.universe.universe_builder import build_universe_snapshot
from python_edge.universe.universe_builder import load_config_from_env
from python_edge.universe.universe_builder import save_universe_outputs

PAUSE_ON_EXIT = str(os.getenv("PAUSE_ON_EXIT", "auto")).strip().lower()



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
    config = load_config_from_env(ROOT)
    print(f"[CFG] base_url={config.base_url}")
    print(f"[CFG] locale={config.locale} market={config.market} ticker_type={config.ticker_type}")
    print(f"[CFG] min_price={config.min_price:.2f} min_dollar_volume={config.min_dollar_volume:.2f} top_n={config.top_n}")
    print(f"[CFG] output_dir={config.output_dir}")
    snapshot_df, summary = build_universe_snapshot(config)
    parquet_path, summary_path = save_universe_outputs(snapshot_df, summary, config.output_dir)
    counters = summary.get("counters", {})
    print(f"[DATA] as_of_date={summary.get('as_of_date')}")
    print(f"[UNIVERSE] candidates_total={counters.get('candidates_total', 0)}")
    print(f"[UNIVERSE] missing_grouped_daily_bar={counters.get('missing_grouped_daily_bar', 0)}")
    print(f"[UNIVERSE] dropped_price={counters.get('dropped_price', 0)}")
    print(f"[UNIVERSE] dropped_liquidity={counters.get('dropped_liquidity', 0)}")
    print(f"[UNIVERSE] dropped_history={counters.get('dropped_history', 0)}")
    print(f"[UNIVERSE] eligible_total={counters.get('eligible_total', 0)}")
    print(f"[UNIVERSE] selected_total={counters.get('selected_total', 0)}")
    print(f"[OK] snapshot={parquet_path}")
    print(f"[OK] summary={summary_path}")
    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 1
    _safe_exit(rc)