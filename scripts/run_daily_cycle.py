from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAUSE_ON_EXIT = str(os.getenv("PAUSE_ON_EXIT", "auto")).strip().lower()
FREEZE_ROOT = Path(os.getenv("FREEZE_ROOT", "artifacts/freeze_runner"))
CONFIG_NAMES = [x.strip() for x in str(os.getenv("CONFIG_NAMES", "optimal|aggressive")).split("|") if x.strip()]
REQUIRE_ANY_LIVE_ACTIVE_NAMES = str(os.getenv("REQUIRE_ANY_LIVE_ACTIVE_NAMES", "1")).strip().lower() not in {"0", "false", "no", "off"}
REQUIRE_FRESH_FREEZE_DATE_MATCH = str(os.getenv("REQUIRE_FRESH_FREEZE_DATE_MATCH", "1")).strip().lower() not in {"0", "false", "no", "off"}
RUN_BROKER_HANDOFF = str(os.getenv("RUN_BROKER_HANDOFF", "1")).strip().lower() not in {"0", "false", "no", "off"}

UNIVERSE_SNAPSHOT_FILE = Path(os.getenv("UNIVERSE_SNAPSHOT_FILE", "artifacts/daily_cycle/universe/universe_snapshot.parquet"))
UNIVERSE_SUMMARY_FILE = Path(os.getenv("UNIVERSE_SUMMARY_FILE", "artifacts/daily_cycle/universe/universe_summary.json"))
LIVE_ALPHA_SNAPSHOT_FILE = Path(os.getenv("LIVE_ALPHA_SNAPSHOT_FILE", "artifacts/live_alpha/live_alpha_snapshot.parquet"))
FREEZE_ALL_SUMMARY_FILE = Path(os.getenv("FREEZE_ALL_SUMMARY_FILE", "artifacts/freeze_runner/freeze_all_configs_summary.json"))
TOP_MISSING_SYMBOLS = int(os.getenv("TOP_MISSING_SYMBOLS", "25"))

SYMBOL_COLUMN_CANDIDATES = [
    "symbol",
    "ticker",
    "sym",
    "Symbol",
    "Ticker",
    "SYM",
]
DATE_COLUMN_CANDIDATES = [
    "date",
    "as_of_date",
    "session_date",
    "trade_date",
    "Date",
    "DATE",
]


def _enable_line_buffering() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        try:
            stream.reconfigure(line_buffering=True)
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


def _resolve(rel_or_abs: Path) -> Path:
    return rel_or_abs if rel_or_abs.is_absolute() else (ROOT / rel_or_abs)


def _must_exist(path_like: Path, label: str) -> Path:
    path = _resolve(path_like)
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _run_step(script_rel_path: str) -> None:
    script_path = ROOT / script_rel_path
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")
    cmd = [sys.executable, str(script_path)]
    print(f"[RUN] {' '.join(cmd)}")
    completed = subprocess.run(cmd, cwd=str(ROOT))
    if completed.returncode != 0:
        raise RuntimeError(f"Step failed: {script_rel_path} rc={completed.returncode}")


def _read_json(path_like: Path, label: str) -> Dict[str, Any]:
    path = _must_exist(path_like, label)
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} is not a JSON object: {path}")
    return payload


def _normalize_symbol_set(values: Iterable[object]) -> Set[str]:
    out: Set[str] = set()
    for value in values:
        sym = str(value).strip().upper()
        if sym and sym != "NAN":
            out.add(sym)
    return out


def _find_first_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def _find_symbol_column(df: pd.DataFrame, label: str) -> str:
    column = _find_first_existing_column(df, SYMBOL_COLUMN_CANDIDATES)
    if column is not None:
        return column

    index_names = [name for name in df.index.names if name is not None]
    for name in index_names:
        if name in SYMBOL_COLUMN_CANDIDATES:
            df.reset_index(inplace=True)
            return str(name)

    single_index_name = df.index.name
    if single_index_name in SYMBOL_COLUMN_CANDIDATES:
        df.reset_index(inplace=True)
        return str(single_index_name)

    raise RuntimeError(
        f"{label} missing symbol-like column. Available columns={list(df.columns)} index_names={list(df.index.names)}"
    )


def _find_date_column(df: pd.DataFrame) -> str | None:
    column = _find_first_existing_column(df, DATE_COLUMN_CANDIDATES)
    if column is not None:
        return column

    index_names = [name for name in df.index.names if name is not None]
    for name in index_names:
        if name in DATE_COLUMN_CANDIDATES:
            df.reset_index(inplace=True)
            return str(name)

    single_index_name = df.index.name
    if single_index_name in DATE_COLUMN_CANDIDATES:
        df.reset_index(inplace=True)
        return str(single_index_name)

    return None


def _extract_normalized_date_series(df: pd.DataFrame, date_col: str | None) -> pd.Series:
    if date_col is None:
        return pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    return pd.to_datetime(df[date_col], errors="coerce").dt.normalize()


def _load_universe_snapshot() -> Tuple[pd.DataFrame, Set[str], pd.Timestamp | None, Path, str, str | None]:
    path = _must_exist(UNIVERSE_SNAPSHOT_FILE, "Universe snapshot")
    df = pd.read_parquet(path)
    if df.empty:
        raise RuntimeError(f"Universe snapshot is empty: {path}")

    symbol_col = _find_symbol_column(df, "Universe snapshot")
    date_col = _find_date_column(df)
    date_series = _extract_normalized_date_series(df, date_col)

    latest_date = None
    non_na = date_series.dropna()
    if len(non_na):
        latest_date = pd.Timestamp(non_na.max()).normalize()
        df = df.loc[date_series == latest_date].copy()

    df[symbol_col] = df[symbol_col].astype(str).str.strip().str.upper()
    df = df.loc[df[symbol_col].ne("") & df[symbol_col].ne("NAN")].copy()
    df["symbol"] = df[symbol_col]
    if date_col is not None:
        df["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()

    symbols = _normalize_symbol_set(df["symbol"])
    return df.reset_index(drop=True), symbols, latest_date, path, symbol_col, date_col


def _load_live_alpha_snapshot() -> Tuple[pd.DataFrame, Set[str], pd.Timestamp | None, int, Path]:
    path = _must_exist(LIVE_ALPHA_SNAPSHOT_FILE, "Live alpha snapshot")
    df = pd.read_parquet(path)
    if df.empty:
        raise RuntimeError(f"Live alpha snapshot is empty: {path}")
    if "symbol" not in df.columns:
        raise RuntimeError(f"Live alpha snapshot missing 'symbol' column: {path}")
    if "date" not in df.columns:
        raise RuntimeError(f"Live alpha snapshot missing 'date' column: {path}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    non_na = df["date"].dropna()
    if non_na.empty:
        raise RuntimeError(f"Live alpha snapshot has no valid dates: {path}")
    latest_date = pd.Timestamp(non_na.max()).normalize()
    df = df.loc[df["date"] == latest_date].copy()
    df["symbol"] = df["symbol"].astype(str).str.upper()
    alpha_cols = [str(c) for c in df.columns if str(c).startswith("alpha_")]
    symbols = _normalize_symbol_set(df["symbol"])
    return df.reset_index(drop=True), symbols, latest_date, len(alpha_cols), path


def _safe_int(payload: Dict[str, Any], key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _print_symbol_preview(tag: str, symbols: Sequence[str]) -> None:
    if not symbols:
        print(f"{tag} none")
        return
    preview = ", ".join(symbols[:TOP_MISSING_SYMBOLS])
    more = max(0, len(symbols) - TOP_MISSING_SYMBOLS)
    suffix = "" if more == 0 else f" ... (+{more} more)"
    print(f"{tag} {preview}{suffix}")


def _diagnose_universe_stage() -> Dict[str, Any]:
    summary = _read_json(UNIVERSE_SUMMARY_FILE, "Universe summary")
    universe_df, universe_symbols, universe_date, universe_path, symbol_col, date_col = _load_universe_snapshot()
    print(
        "[DIAG][UNIVERSE] "
        f"snapshot={universe_path} "
        f"summary={_resolve(UNIVERSE_SUMMARY_FILE)} "
        f"rows_current={len(universe_df)} symbols_current={len(universe_symbols)} "
        f"symbol_col={symbol_col} "
        f"date_col={(date_col if date_col is not None else 'NA')} "
        f"current_date={(universe_date.date().isoformat() if universe_date is not None else 'NA')}"
    )

    summary_target = _safe_int(summary, "selected_total")
    summary_eligible = _safe_int(summary, "eligible_total")
    summary_candidates = _safe_int(summary, "candidates_total")
    print(
        "[DIAG][UNIVERSE][SUMMARY] "
        f"candidates_total={summary_candidates} "
        f"eligible_total={summary_eligible} "
        f"selected_total={summary_target} "
        f"overview_missing_sic_total={summary.get('overview_missing_sic_total')} "
        f"sector_unknown_selected={summary.get('sector_unknown_selected')} "
        f"industry_unknown_selected={summary.get('industry_unknown_selected')}"
    )

    if summary_target is not None and summary_target != len(universe_symbols):
        print(
            "[DIAG][UNIVERSE][WARN] "
            f"summary selected_total={summary_target} but current snapshot symbols={len(universe_symbols)}"
        )

    return {
        "summary": summary,
        "df": universe_df,
        "symbols": universe_symbols,
        "date": universe_date,
        "symbol_col": symbol_col,
        "date_col": date_col,
    }


def _diagnose_live_alpha_against_universe(universe_diag: Dict[str, Any]) -> Dict[str, Any]:
    live_df, live_symbols, live_date, alpha_col_count, live_path = _load_live_alpha_snapshot()
    universe_symbols = set(universe_diag["symbols"])
    universe_date = universe_diag["date"]

    present = sorted(universe_symbols & live_symbols)
    missing = sorted(universe_symbols - live_symbols)
    extra = sorted(live_symbols - universe_symbols)

    requested = len(universe_symbols)
    present_count = len(present)
    missing_count = len(missing)
    extra_count = len(extra)
    survival_ratio = (present_count / requested) if requested > 0 else 0.0

    print(
        "[DIAG][LIVE_ALPHA] "
        f"snapshot={live_path} current_date={(live_date.date().isoformat() if live_date is not None else 'NA')} "
        f"rows_current={len(live_df)} symbols_current={len(live_symbols)} alpha_cols={alpha_col_count}"
    )
    print(
        "[DIAG][SURVIVAL] "
        f"universe_requested={requested} "
        f"live_symbols_current={len(live_symbols)} "
        f"present_in_live_alpha={present_count} "
        f"missing_in_live_alpha={missing_count} "
        f"extra_live_alpha_symbols={extra_count} "
        f"survival_ratio={survival_ratio:.4f}"
    )

    if universe_date is not None and live_date is not None and universe_date != live_date:
        print(
            "[DIAG][SURVIVAL][WARN] "
            f"date mismatch universe_date={universe_date.date().isoformat()} "
            f"live_alpha_date={live_date.date().isoformat()}"
        )

    _print_symbol_preview("[DIAG][SURVIVAL][MISSING_TOP]", missing)
    _print_symbol_preview("[DIAG][SURVIVAL][EXTRA_TOP]", extra)

    payload = {
        "universe_requested": requested,
        "live_symbols_current": len(live_symbols),
        "present_in_live_alpha": present_count,
        "missing_in_live_alpha": missing_count,
        "extra_live_alpha_symbols": extra_count,
        "survival_ratio": survival_ratio,
        "universe_current_date": universe_date.date().isoformat() if universe_date is not None else None,
        "live_alpha_current_date": live_date.date().isoformat() if live_date is not None else None,
        "universe_symbol_col": str(universe_diag.get("symbol_col")),
        "universe_date_col": universe_diag.get("date_col"),
        "alpha_col_count": alpha_col_count,
        "missing_symbols_top": missing[:TOP_MISSING_SYMBOLS],
        "extra_symbols_top": extra[:TOP_MISSING_SYMBOLS],
    }
    out_path = _resolve(Path("artifacts/daily_cycle") / "universe_live_alpha_diagnostics.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ARTIFACT] {out_path}")
    return payload


def _load_freeze_summaries() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name in CONFIG_NAMES:
        summary_path = FREEZE_ROOT / name / "freeze_current_summary.json"
        path = _must_exist(summary_path, f"Freeze current summary for config={name}")
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if not isinstance(payload, dict):
            raise RuntimeError(f"Freeze current summary is not a JSON object: {path}")
        row = dict(payload)
        row["config"] = name
        row["summary_path"] = str(path)
        rows.append(row)
    return rows


def _diagnose_freeze_stage(live_diag: Dict[str, Any]) -> Dict[str, Any]:
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
    live_dates = {
        str(row.get("live_current_date", "")).strip()
        for row in summaries
        if str(row.get("live_current_date", "")).strip()
    }
    dates_match = len(live_dates) == 1

    freeze_diag = {
        "config_count": len(summaries),
        "any_live_names": bool(any_live_names),
        "live_dates": sorted(live_dates),
        "dates_match": bool(dates_match),
        "configs": summaries,
        "upstream_live_alpha_symbols": int(live_diag.get("live_symbols_current", 0)),
        "upstream_present_in_live_alpha": int(live_diag.get("present_in_live_alpha", 0)),
    }

    merged_payload = {
        "live_alpha": live_diag,
        "freeze": freeze_diag,
    }
    out_path = _resolve(Path("artifacts/daily_cycle") / "freeze_gate_diagnostics.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ARTIFACT] {out_path}")
    return freeze_diag


def _freeze_gate_allows_execution_from_diag(freeze_diag: Dict[str, Any]) -> bool:
    any_live_names = bool(freeze_diag.get("any_live_names", False))
    dates_match = bool(freeze_diag.get("dates_match", False))
    live_dates = list(freeze_diag.get("live_dates", []))

    if REQUIRE_ANY_LIVE_ACTIVE_NAMES and not any_live_names:
        print("[GATE][BLOCK] all freeze configs have live_active_names=0 -> execution loop skipped")
        return False
    if REQUIRE_FRESH_FREEZE_DATE_MATCH and not dates_match:
        print(f"[GATE][BLOCK] freeze live_current_date mismatch across configs: {sorted(live_dates)}")
        return False
    print("[GATE][PASS] execution loop allowed")
    return True


def main() -> int:
    _enable_line_buffering()
    print(f"[ROOT] {ROOT}")

    print("[STEP] universe builder")
    _run_step("scripts/run_universe_builder.py")
    universe_diag = _diagnose_universe_stage()

    print("[STEP] live alpha snapshot")
    _run_step("scripts/run_live_alpha_snapshot.py")
    live_diag = _diagnose_live_alpha_against_universe(universe_diag)

    print("[STEP] freeze runner")
    _run_step("scripts/run_freeze_runner.py")
    freeze_diag = _diagnose_freeze_stage(live_diag)

    if _freeze_gate_allows_execution_from_diag(freeze_diag):
        print("[STEP] execution loop")
        _run_step("scripts/run_execution_loop.py")
        if RUN_BROKER_HANDOFF:
            print("[STEP] broker handoff")
            _run_step("scripts/run_broker_handoff.py")
        else:
            print("[STEP] broker handoff skipped by config")
    else:
        print("[STEP] execution loop skipped by gate")

    if _resolve(FREEZE_ALL_SUMMARY_FILE).exists():
        print(f"[ARTIFACT] {_resolve(FREEZE_ALL_SUMMARY_FILE)}")

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
