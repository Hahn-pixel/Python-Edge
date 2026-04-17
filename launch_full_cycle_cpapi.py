"""
launch_full_cycle_cpapi.py — виконання trades (CPAPI)

Запускає run_full_cycle_cpapi.ps1 через PowerShell 7.

Передумови:
  1. Gateway запущений на https://localhost:5000 і авторизований
  2. launch_daily_update.py вже виконано сьогодні
  3. MASSIVE_API_KEY встановлено в середовищі

Час запуску: ~15:45 ET

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

IB_ACCOUNT_CODE  = "DUP561175"
NAV_OPTIMAL      = "978809.6875"   # оновлювати щодня або автоматизувати
NAV_AGGRESSIVE   = "978809.6875"

MASSIVE_API_KEY  = os.getenv("MASSIVE_API_KEY", "")
MASSIVE_BASE_URL = "https://api.massive.com"
CPAPI_BASE_URL   = "https://localhost:5000"

# Exit policy (порожнє = використовувати дефолти з exit_policy.py)
EXIT_OPTIMAL_STOP_LOSS_PCT     = ""   # default: 0.08
EXIT_OPTIMAL_TAKE_PROFIT_PCT   = ""   # default: 0.25
EXIT_OPTIMAL_TRAIL_K           = ""   # default: 1.5
EXIT_OPTIMAL_TRAIL_MIN         = ""   # default: 0.03
EXIT_OPTIMAL_TRAIL_MAX         = ""   # default: 0.20
EXIT_OPTIMAL_MAX_HOLD_DAYS     = ""   # default: 30

EXIT_AGGRESSIVE_STOP_LOSS_PCT     = ""   # default: 0.10
EXIT_AGGRESSIVE_TAKE_PROFIT_PCT   = ""   # default: 0.30
EXIT_AGGRESSIVE_TRAIL_K           = ""   # default: 1.5
EXIT_AGGRESSIVE_TRAIL_MIN         = ""   # default: 0.03
EXIT_AGGRESSIVE_TRAIL_MAX         = ""   # default: 0.20
EXIT_AGGRESSIVE_MAX_HOLD_DAYS     = ""   # default: 20

# ──────────────────────────────────────────────────────────────

def _check_gateway() -> bool:
    """Перевірити чи Gateway доступний."""
    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        import requests
        resp = requests.get(
            f"{CPAPI_BASE_URL}/v1/api/iserver/auth/status",
            verify=False,
            timeout=5,
        )
        data = resp.json()
        authenticated = bool(data.get("authenticated", False))
        print(f"[GATEWAY] authenticated={authenticated} connected={data.get('connected')}")
        return authenticated
    except Exception as exc:
        print(f"[GATEWAY] unreachable: {exc}")
        return False


def _fetch_nav() -> tuple[str, str]:
    """Отримати актуальний NAV з CPAPI."""
    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        import requests
        s = requests.Session()
        s.verify = False
        # Отримати account list
        resp = s.get(f"{CPAPI_BASE_URL}/v1/api/iserver/accounts", timeout=10)
        accounts = resp.json().get("accounts", [])
        if not accounts:
            return NAV_OPTIMAL, NAV_AGGRESSIVE
        acct = accounts[0]
        resp2 = s.get(f"{CPAPI_BASE_URL}/v1/api/portfolio/{acct}/summary", timeout=10)
        data = resp2.json()
        nl = data.get("netliquidation", {})
        nav = str(nl.get("amount", NAV_OPTIMAL)) if isinstance(nl, dict) else str(nl)
        print(f"[NAV] fetched from CPAPI: {nav}")
        return nav, nav
    except Exception as exc:
        print(f"[NAV] fetch failed: {exc} — using hardcoded values")
        return NAV_OPTIMAL, NAV_AGGRESSIVE


def main() -> int:
    print("=" * 60)
    print("  Python-Edge — Full Cycle CPAPI (Execution)")
    print("=" * 60)
    print(f"  ROOT:    {ROOT}")
    print(f"  ACCOUNT: {IB_ACCOUNT_CODE}")
    print(f"  CPAPI:   {CPAPI_BASE_URL}")
    print()

    # Перевірка Gateway
    print("[CHECK] Gateway status...")
    if not _check_gateway():
        print("[ERROR] Gateway not authenticated")
        print("        Відкрийте браузер: https://localhost:5000")
        print("        та авторизуйтесь через IBKR")
        return 1

    # Отримати актуальний NAV
    print("[NAV] Fetching current NAV...")
    nav_optimal, nav_aggressive = _fetch_nav()

    # Перевірка MASSIVE_API_KEY
    if not MASSIVE_API_KEY:
        print("[WARN] MASSIVE_API_KEY not set — massive fallback disabled")

    # Побудувати env для ps1
    env = os.environ.copy()
    env.update({
        "IB_ACCOUNT_CODE":   IB_ACCOUNT_CODE,
        "NAV_OPTIMAL":       nav_optimal,
        "NAV_AGGRESSIVE":    nav_aggressive,
        "MASSIVE_API_KEY":   MASSIVE_API_KEY,
        "MASSIVE_BASE_URL":  MASSIVE_BASE_URL,
        "CPAPI_BASE_URL":    CPAPI_BASE_URL,
    })

    # Exit policy env
    policy_vars = {
        "EXIT_OPTIMAL_STOP_LOSS_PCT":        EXIT_OPTIMAL_STOP_LOSS_PCT,
        "EXIT_OPTIMAL_TAKE_PROFIT_PCT":      EXIT_OPTIMAL_TAKE_PROFIT_PCT,
        "EXIT_OPTIMAL_TRAIL_K":              EXIT_OPTIMAL_TRAIL_K,
        "EXIT_OPTIMAL_TRAIL_MIN":            EXIT_OPTIMAL_TRAIL_MIN,
        "EXIT_OPTIMAL_TRAIL_MAX":            EXIT_OPTIMAL_TRAIL_MAX,
        "EXIT_OPTIMAL_MAX_HOLD_DAYS":        EXIT_OPTIMAL_MAX_HOLD_DAYS,
        "EXIT_AGGRESSIVE_STOP_LOSS_PCT":     EXIT_AGGRESSIVE_STOP_LOSS_PCT,
        "EXIT_AGGRESSIVE_TAKE_PROFIT_PCT":   EXIT_AGGRESSIVE_TAKE_PROFIT_PCT,
        "EXIT_AGGRESSIVE_TRAIL_K":           EXIT_AGGRESSIVE_TRAIL_K,
        "EXIT_AGGRESSIVE_TRAIL_MIN":         EXIT_AGGRESSIVE_TRAIL_MIN,
        "EXIT_AGGRESSIVE_TRAIL_MAX":         EXIT_AGGRESSIVE_TRAIL_MAX,
        "EXIT_AGGRESSIVE_MAX_HOLD_DAYS":     EXIT_AGGRESSIVE_MAX_HOLD_DAYS,
    }
    for k, v in policy_vars.items():
        if v:
            env[k] = v

    # Запустити ps1 через pwsh
    ps1_path = ROOT / "run_full_cycle_cpapi.ps1"
    if not ps1_path.exists():
        print(f"[ERROR] run_full_cycle_cpapi.ps1 not found: {ps1_path}")
        return 1

    # Підставити IB_ACCOUNT_CODE і NAV в ps1 через -Command з env
    # Використовуємо pwsh (PowerShell 7)
    cmd = [
        "pwsh", "-NoProfile", "-NonInteractive",
        "-Command",
        # Встановити змінні і запустити ps1
        f"$env:IB_ACCOUNT_CODE='{IB_ACCOUNT_CODE}'; "
        f"$env:NAV_OPTIMAL='{nav_optimal}'; "
        f"$env:NAV_AGGRESSIVE='{nav_aggressive}'; "
        f". '{ps1_path}'",
    ]

    # Альтернатива: пропатчити ps1 inline і запустити
    # Простіше — передати через окремий wrapper ps1
    ps1_wrapper = ROOT / "_launch_wrapper.ps1"
    ps1_wrapper.write_text(
        f"""Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$IB_ACCOUNT_CODE  = "{IB_ACCOUNT_CODE}"
$NAV_OPTIMAL      = "{nav_optimal}"
$NAV_AGGRESSIVE   = "{nav_aggressive}"
$MASSIVE_API_KEY  = $env:MASSIVE_API_KEY
. "{ps1_path}"
""",
        encoding="utf-8",
    )

    print(f"\n[RUN] run_full_cycle_cpapi.ps1")
    print(f"      NAV_OPTIMAL={nav_optimal}  NAV_AGGRESSIVE={nav_aggressive}")
    print()

    result = subprocess.run(
        ["pwsh", "-NoProfile", "-NonInteractive", "-File", str(ps1_wrapper)],
        cwd=str(ROOT),
        env=env,
    )

    # Видалити тимчасовий wrapper
    try:
        ps1_wrapper.unlink()
    except Exception:
        pass

    print()
    if result.returncode == 0:
        print("[OK] Pipeline complete")
    else:
        print(f"[FAILED] Pipeline exited with code {result.returncode}")

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
