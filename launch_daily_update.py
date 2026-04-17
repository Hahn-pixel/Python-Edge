"""
launch_daily_update.py — щоденне оновлення даних

Запускає послідовно:
  1. run_universe_builder.py  )
  2. run_live_alpha_snapshot.py) через run_daily_cycle.py
  3. run_freeze_runner.py      )

Запускати щодня перед launch_full_cycle_cpapi.py
Час запуску: ~15:30-15:40 ET

Подвійний клік — вікно не закривається до натискання Enter.
"""

from __future__ import annotations

import os
import subprocess
import sys
import traceback
from pathlib import Path

# ── Налаштування ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent

MASSIVE_API_KEY  = os.getenv("MASSIVE_API_KEY", "")
MASSIVE_BASE_URL = "https://api.massive.com"

# ──────────────────────────────────────────────────────────────

def _run(script: str, env: dict) -> int:
    path = ROOT / script
    if not path.exists():
        print(f"[ERROR] script not found: {path}")
        return 1
    print(f"\n{'='*60}")
    print(f"[RUN] {script}")
    print(f"{'='*60}")
    result = subprocess.run(
        [sys.executable, str(path)],
        cwd=str(ROOT),
        env=env,
    )
    return result.returncode


def main() -> int:
    if not MASSIVE_API_KEY:
        print("[WARN] MASSIVE_API_KEY not set in environment")
        print("       Set it via: $env:MASSIVE_API_KEY = 'your_key'")
        print("       або додайте в System Environment Variables")

    env = os.environ.copy()
    env.update({
        # Daily cycle
        "FREEZE_ROOT":                          "artifacts/freeze_runner",
        "CONFIG_NAMES":                         "optimal|aggressive",
        "REQUIRE_ANY_LIVE_ACTIVE_NAMES":        "1",
        "REQUIRE_FRESH_FREEZE_DATE_MATCH":      "1",
        "REQUIRE_FREEZE_UNIVERSE_FILTER":       "1",
        "MIN_FREEZE_UNIVERSE_SURVIVAL_RATIO":   "0.90",
        "REQUIRE_FREEZE_LIVE_GATE_PASSED":      "0",
        "REQUIRE_ANY_REPLAY_GATE_PASS":         "1",
        # Вимикаємо broker steps — вони в launch_full_cycle_cpapi.py
        "RUN_BROKER_HANDOFF":                   "0",
        "RUN_BROKER_ADAPTER":                   "0",
        "RUN_BROKER_RECONCILE":                 "0",
        # Live alpha snapshot
        "UNIVERSE_SNAPSHOT_FILE":               "artifacts/daily_cycle/universe/universe_snapshot.parquet",
        "LIVE_ALPHA_OUT_DIR":                   "artifacts/live_alpha",
        "LIVE_ALPHA_LOOKBACK_DAYS":             "260",
        "LIVE_ALPHA_MAX_SYMBOLS":               "500",
        "LIVE_ALPHA_SCOPE":                     "registry",
        "LIVE_ALPHA_SURVIVOR_TOP_N":            "6",
        "LIVE_ALPHA_PROXY_ENABLE":              "1",
        "LIVE_ALPHA_INTERACTION_ENABLE":        "1",
        "LIVE_ALPHA_INTERACTION_TOP_K":         "24",
        "LIVE_ALPHA_INTERACTION_GATES":         "oil_up|dollar_up|macro_risk_off",
        # Massive
        "MASSIVE_API_KEY":                      MASSIVE_API_KEY,
        "MASSIVE_BASE_URL":                     MASSIVE_BASE_URL,
        "MASSIVE_TIMEOUT_SEC":                  "20.0",
        # Misc
        "CHILD_PAUSE_ON_EXIT":                  "0",
        "PAUSE_ON_EXIT":                        "0",
    })

    print("=" * 60)
    print("  Python-Edge — Daily Data Update")
    print("=" * 60)
    print(f"  ROOT: {ROOT}")
    print(f"  MASSIVE_API_KEY: {'SET' if MASSIVE_API_KEY else 'NOT SET'}")
    print()

    rc = _run("scripts/run_daily_cycle.py", env)

    print()
    if rc == 0:
        print("[OK] Daily update complete")
        print("     Run launch_full_cycle_cpapi.py to execute trades")
    else:
        print(f"[FAILED] run_daily_cycle.py exited with code {rc}")
        print("         Check logs above for details")

    return rc


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        print("\n[CRASHED]")
    finally:
        print()
        input("Press Enter to exit...")
    sys.exit(rc)
