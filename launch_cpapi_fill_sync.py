"""
launch_cpapi_fill_sync.py — запуск fill sync після закриття

Запускати о ~16:05 ET (21:05 Київ).
Якщо запускаєте після опівночі UTC — передайте дату явно:
  set SYNC_DATE=2026-04-21 && python launch_cpapi_fill_sync.py

Подвійний клік — вікно не закривається до натискання Enter.
"""

from __future__ import annotations

import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent

IB_ACCOUNT_CODE = "DUP561175"
CPAPI_BASE_URL  = "https://localhost:5000"

# Залишити порожнім — використовує сьогоднішню дату UTC автоматично.
# Встановити явно якщо запускаєте після опівночі UTC.
SYNC_DATE = ""

SCRIPT_FILL_SYNC = ROOT / "scripts" / "run_cpapi_fill_sync.py"


def main() -> int:
    date = SYNC_DATE or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    print("=" * 60)
    print("  Python-Edge — Fill Sync")
    print("=" * 60)
    print(f"  ROOT:    {ROOT}")
    print(f"  ACCOUNT: {IB_ACCOUNT_CODE}")
    print(f"  DATE:    {date}")
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
        "SYNC_DATE":         date,
        "PAUSE_ON_EXIT":     "0",
    })

    result = subprocess.run([sys.executable, str(SCRIPT_FILL_SYNC)], cwd=str(ROOT), env=env)
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
