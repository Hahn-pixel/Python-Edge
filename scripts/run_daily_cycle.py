from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
PAUSE_ON_EXIT = str(os.getenv("PAUSE_ON_EXIT", "auto")).strip().lower()
FREEZE_ROOT = Path(os.getenv("FREEZE_ROOT", "artifacts/freeze_runner"))
CONFIG_NAMES = [x.strip() for x in str(os.getenv("CONFIG_NAMES", "optimal|aggressive")).split("|") if x.strip()]
REQUIRE_ANY_LIVE_ACTIVE_NAMES = str(os.getenv("REQUIRE_ANY_LIVE_ACTIVE_NAMES", "1")).strip().lower() not in {"0", "false", "no", "off"}
REQUIRE_FRESH_FREEZE_DATE_MATCH = str(os.getenv("REQUIRE_FRESH_FREEZE_DATE_MATCH", "1")).strip().lower() not in {"0", "false", "no", "off"}


def _should_pause() -> bool:
    if PAUSE_ON_EXIT in {"0", "false", "no", "off"}:
        return False
    if PAUSE_ON_EXIT in {"1", "true", "yes", "on"}:
        return True
    stdin_obj = getattr(sys, "stdin", None)
    stdout_obj = getattr(sys, "stdout", None)
    return bool(stdin_obj and stdout_obj and hasattr(stdin_obj, "isatty") and hasattr(stdout_obj, "isatty") and stdin_obj.isatty() and stdout_obj.isatty())


def _safe_exit(code: int) -> None:
    if _should_pause():
        try:
            print("")
            print(f"[EXIT] code={code}")
            input("Press Enter to exit...")
        except Exception:
            pass
    raise SystemExit(code)


def _run_step(script_rel_path: str) -> None:
    script_path = ROOT / script_rel_path
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")
    cmd = [sys.executable, str(script_path)]
    print(f"[RUN] {' '.join(cmd)}")
    completed = subprocess.run(cmd, cwd=str(ROOT))
    if completed.returncode != 0:
        raise RuntimeError(f"Step failed: {script_rel_path} rc={completed.returncode}")


def _load_freeze_summaries() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for name in CONFIG_NAMES:
        summary_path = FREEZE_ROOT / name / "freeze_current_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Freeze current summary not found for config={name}: {summary_path}")
        with summary_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        payload = dict(payload)
        payload["config"] = name
        rows.append(payload)
    return rows


def _freeze_gate_allows_execution() -> bool:
    summaries = _load_freeze_summaries()
    print("[GATE] freeze summaries")
    for row in summaries:
        print(
            "[GATE] "
            f"config={row.get('config')} "
            f"live_current_date={row.get('live_current_date')} "
            f"freeze_live_active_names={row.get('live_active_names')} "
            f"live_gross_exposure_current_day={row.get('live_gross_exposure_current_day')} "
            f"mr_enabled_effective={row.get('mr_enabled_effective')}"
        )

    any_live_names = any(int(row.get("live_active_names", 0) or 0) > 0 for row in summaries)
    live_dates = {str(row.get("live_current_date", "")).strip() for row in summaries if str(row.get("live_current_date", "")).strip()}
    dates_match = len(live_dates) == 1

    if REQUIRE_ANY_LIVE_ACTIVE_NAMES and not any_live_names:
        print("[GATE][BLOCK] all freeze configs have live_active_names=0 -> execution loop skipped")
        return False
    if REQUIRE_FRESH_FREEZE_DATE_MATCH and not dates_match:
        print(f"[GATE][BLOCK] freeze live_current_date mismatch across configs: {sorted(live_dates)}")
        return False

    print("[GATE][PASS] execution loop allowed")
    return True


def main() -> int:
    print(f"[ROOT] {ROOT}")
    print("[STEP] universe builder")
    _run_step("scripts/run_universe_builder.py")
    print("[STEP] live alpha snapshot")
    _run_step("scripts/run_live_alpha_snapshot.py")
    print("[STEP] freeze runner")
    _run_step("scripts/run_freeze_runner.py")
    if _freeze_gate_allows_execution():
        print("[STEP] execution loop")
        _run_step("scripts/run_execution_loop.py")
    else:
        print("[STEP] execution loop skipped by gate")
    print("[FINAL] daily cycle complete")
    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 1
    _safe_exit(rc)