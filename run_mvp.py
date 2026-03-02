# scripts/run_mvp.py
# Double-click runnable. Never auto-closes.
from __future__ import annotations
import traceback
from pathlib import Path

def _press_enter_exit(code: int) -> None:
    try:
        print(f\"\\n[EXIT] code={code}\")
        input(\"Press Enter to exit...\")
    except Exception:
        pass
    raise SystemExit(code)

def main() -> int:
    root = Path(__file__).resolve().parents[1]
    print(f\"[Python-Edge] root={root}\")
    print(\"MVP pipeline placeholder: ingest -> QA -> features -> rule_mining -> WF -> portfolio -> report\")
    return 0

if __name__ == \"__main__\":
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 1
    _press_enter_exit(int(rc))