"""
launch_cpapi_fill_sync.py — запуск fill sync після закриття

Запускати о ~16:05 ET (через ~5 хвилин після закриття біржі)
щоб синхронізувати fills з IBKR Gateway в broker_log.json і fills.csv.

Подвійний клік — вікно не закривається до натискання Enter.
"""

from __future__ import annotations

import os
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent

IB_ACCOUNT_CODE = "DUP561175"
CPAPI_BASE_URL  = "https://localhost:5000"

SCRIPT_FILL_SYNC = ROOT / "scripts" / "run_cpapi_fill_sync.py"


def main() -> int:
    print("=" * 60)
    print("  Python-Edge — Fill Sync")
    print("=" * 60)
    print(f"  ROOT:    {ROOT}")
    print(f"  ACCOUNT: {IB_ACCOUNT_CODE}")
    print()

    if not SCRIPT_FILL_SYNC.exists():
        print(f"[ERROR] script not found: {SCRIPT_FILL_SYNC}")
        return 1

    env = os.environ.copy()
    env.update({
        "EXECUTION_ROOT":    "artifacts/execution_loop",
        "CONFIG_NAMES":      "optimal|aggressive",
        "BROKER_ACCOUNT_ID": IB_ACCOUNT_CODE,
        "CPAPI_BASE_URL":    CPAPI_BASE_URL,
        "CPAPI_VERIFY_SSL":  "0",
        "CPAPI_TIMEOUT_SEC": "10.0",
        "PAUSE_ON_EXIT":     "0",
    })

    print(f"[RUN] {SCRIPT_FILL_SYNC.name}")
    result = subprocess.run(
        [sys.executable, str(SCRIPT_FILL_SYNC)],
        cwd=str(ROOT),
        env=env,
    )

    if result.returncode == 0:
        print("\n[OK] Fill sync complete")
    else:
        print(f"\n[FAILED] exit={result.returncode}")

    return result.returncode


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
