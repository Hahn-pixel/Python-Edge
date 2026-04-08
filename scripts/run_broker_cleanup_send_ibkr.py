from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
for p in [ROOT, SRC_DIR]:
    ps = str(p)
    if ps not in sys.path:
        sys.path.insert(0, ps)

PAUSE_ON_EXIT = str(os.getenv("PAUSE_ON_EXIT", "auto")).strip().lower()
EXECUTION_ROOT = Path(os.getenv("EXECUTION_ROOT", "artifacts/execution_loop"))
CONFIG_NAMES = [x.strip() for x in str(os.getenv("CONFIG_NAMES", "optimal|aggressive")).split("|") if x.strip()]
CLEANUP_SEND_PREVIEW_ONLY = str(os.getenv("CLEANUP_SEND_PREVIEW_ONLY", "1")).strip().lower() not in {"0", "false", "no", "off"}
CLEANUP_SEND_ONLY_SYMBOLS = {x.strip().upper() for x in str(os.getenv("CLEANUP_SEND_ONLY_SYMBOLS", "")).split("|") if x.strip()}
CLEANUP_SEND_EXCLUDE_SYMBOLS = {x.strip().upper() for x in str(os.getenv("CLEANUP_SEND_EXCLUDE_SYMBOLS", "")).split("|") if x.strip()}
CLEANUP_SEND_MAX_ROWS = int(os.getenv("CLEANUP_SEND_MAX_ROWS", "1000000"))
CLEANUP_SEND_REQUIRE_EMPTY_PENDING_REFS = str(os.getenv("CLEANUP_SEND_REQUIRE_EMPTY_PENDING_REFS", "1")).strip().lower() not in {"0", "false", "no", "off"}
CLEANUP_SEND_REQUIRE_SIDE = str(os.getenv("CLEANUP_SEND_REQUIRE_SIDE", "BUY|SELL")).strip().upper()
CLEANUP_SEND_REASON_TAG = str(os.getenv("CLEANUP_SEND_REASON_TAG", "broker_cleanup_send")).strip() or "broker_cleanup_send"
TOPK_PRINT = int(os.getenv("TOPK_PRINT", "50"))


@dataclass(frozen=True)
class ConfigPaths:
    name: str
    execution_dir: Path
    cleanup_orders_csv: Path
    cleanup_emit_summary_json: Path
    orders_csv: Path
    orders_backup_csv: Path
    cleanup_send_plan_csv: Path
    cleanup_send_summary_json: Path


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


def _must_exist(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def _norm_symbol(value: object) -> str:
    return str(value or "").strip().upper()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _cfg_paths(config_name: str) -> ConfigPaths:
    base = EXECUTION_ROOT / config_name
    return ConfigPaths(
        name=config_name,
        execution_dir=base,
        cleanup_orders_csv=base / "broker_cleanup_orders.csv",
        cleanup_emit_summary_json=base / "broker_cleanup_emit_summary.json",
        orders_csv=base / "orders.csv",
        orders_backup_csv=base / "orders_pre_cleanup_backup.csv",
        cleanup_send_plan_csv=base / "broker_cleanup_send_plan.csv",
        cleanup_send_summary_json=base / "broker_cleanup_send_summary.json",
    )


def _load_cleanup_orders(paths: ConfigPaths) -> pd.DataFrame:
    _must_exist(paths.cleanup_orders_csv, f"cleanup orders for {paths.name}")
    df = pd.read_csv(paths.cleanup_orders_csv)
    if df.empty:
        return pd.DataFrame(columns=["symbol", "order_side", "delta_shares", "cleanup_reason", "pending_order_refs"])
    required = ["symbol", "order_side", "delta_shares"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"cleanup orders missing required columns {missing}: {paths.cleanup_orders_csv}")
    out = df.copy()
    out["symbol"] = out["symbol"].astype(str).map(_norm_symbol)
    out["order_side"] = out["order_side"].astype(str).str.strip().str.upper()
    out["delta_shares"] = pd.to_numeric(out["delta_shares"], errors="coerce").fillna(0.0)
    if "pending_order_refs" not in out.columns:
        out["pending_order_refs"] = ""
    out["pending_order_refs"] = out["pending_order_refs"].astype(str).fillna("")
    if "cleanup_reason" not in out.columns:
        out["cleanup_reason"] = ""
    out["cleanup_reason"] = out["cleanup_reason"].astype(str).fillna("")
    return out.sort_values(["delta_shares", "symbol"], ascending=[False, True]).reset_index(drop=True)


def _passes_send_filters(row: pd.Series) -> tuple[bool, str]:
    symbol = _norm_symbol(row.get("symbol", ""))
    side = str(row.get("order_side", "") or "").strip().upper()
    qty = abs(float(pd.to_numeric(pd.Series([row.get("delta_shares", 0.0)]), errors="coerce").iloc[0]))
    pending_refs = str(row.get("pending_order_refs", "") or "").strip()

    if not symbol:
        return False, "empty_symbol"
    if CLEANUP_SEND_ONLY_SYMBOLS and symbol not in CLEANUP_SEND_ONLY_SYMBOLS:
        return False, "not_in_only_symbols"
    if CLEANUP_SEND_EXCLUDE_SYMBOLS and symbol in CLEANUP_SEND_EXCLUDE_SYMBOLS:
        return False, "excluded_symbol"
    if qty <= 0.0:
        return False, "non_positive_qty"
    allowed_sides = {x.strip().upper() for x in CLEANUP_SEND_REQUIRE_SIDE.split("|") if x.strip()}
    if side not in allowed_sides:
        return False, "side_not_allowed"
    if CLEANUP_SEND_REQUIRE_EMPTY_PENDING_REFS and pending_refs:
        return False, "pending_refs_present"
    return True, "emit"


def _build_send_plan(paths: ConfigPaths, cleanup_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows: List[Dict[str, object]] = []
    counters: Dict[str, int] = {
        "cleanup_rows": int(len(cleanup_df)),
        "emitted_rows": 0,
        "blocked_empty_symbol": 0,
        "blocked_not_in_only_symbols": 0,
        "blocked_excluded_symbol": 0,
        "blocked_non_positive_qty": 0,
        "blocked_side_not_allowed": 0,
        "blocked_pending_refs_present": 0,
        "blocked_max_rows": 0,
    }

    emitted = 0
    for _, row in cleanup_df.iterrows():
        ok, reason = _passes_send_filters(row)
        if not ok:
            counters[f"blocked_{reason}"] = int(counters.get(f"blocked_{reason}", 0)) + 1
            continue
        if emitted >= int(CLEANUP_SEND_MAX_ROWS):
            counters["blocked_max_rows"] += 1
            continue
        emitted += 1
        counters["emitted_rows"] = emitted
        symbol = _norm_symbol(row["symbol"])
        side = str(row["order_side"] or "").strip().upper()
        qty = abs(float(pd.to_numeric(pd.Series([row["delta_shares"]]), errors="coerce").iloc[0]))
        cleanup_reason = str(row.get("cleanup_reason", "") or "")
        broker_avg_cost = float(pd.to_numeric(pd.Series([row.get("broker_avg_cost", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
        quote_ts = _utc_now_iso()
        price_hint_source = "cleanup_broker_avg_cost" if broker_avg_cost > 0.0 else "cleanup_no_price"
        rows.append(
            {
                "date": "",
                "symbol": symbol,
                "order_side": side,
                "delta_shares": qty,
                "price": broker_avg_cost if broker_avg_cost > 0.0 else pd.NA,
                "price_source": "cleanup_send_plan",
                "price_hint_source": price_hint_source,
                "quote_ts": quote_ts,
                "quote_provider": "cleanup_preview",
                "quote_timeframe": "N/A",
                "model_price_reference": broker_avg_cost if broker_avg_cost > 0.0 else 0.0,
                "bid": 0.0,
                "ask": 0.0,
                "mid": 0.0,
                "last": 0.0,
                "close_price": 0.0,
                "spread_bps": 0.0,
                "price_deviation_vs_model": 0.0,
                "fallback_reason": cleanup_reason,
                "is_priced": 1 if broker_avg_cost > 0.0 else 0,
                "target_weight": 0.0,
                "target_notional": 0.0,
                "target_shares_raw": 0.0,
                "current_shares": float(pd.to_numeric(pd.Series([row.get("broker_position", 0.0)]), errors="coerce").iloc[0]),
                "target_shares": float(pd.to_numeric(pd.Series([row.get("expected_shares", 0.0)]), errors="coerce").iloc[0]),
                "raw_delta_shares": qty if side == "BUY" else -qty,
                "order_notional": 0.0,
                "order_notional_abs": 0.0,
                "estimated_commission": 0.0,
                "estimated_slippage": 0.0,
                "estimated_total_cost": 0.0,
                "skip_reason": "",
                "cleanup_reason": cleanup_reason,
                "cleanup_tag": CLEANUP_SEND_REASON_TAG,
                "pending_order_refs": str(row.get("pending_order_refs", "") or ""),
            }
        )

    plan_df = pd.DataFrame(rows)
    if not plan_df.empty:
        plan_df = plan_df.sort_values(["delta_shares", "symbol"], ascending=[False, True]).reset_index(drop=True)
    summary = {
        "config": paths.name,
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "preview_only": int(CLEANUP_SEND_PREVIEW_ONLY),
        "cleanup_orders_csv": str(paths.cleanup_orders_csv),
        "send_plan_csv": str(paths.cleanup_send_plan_csv),
        "orders_csv": str(paths.orders_csv),
        "orders_backup_csv": str(paths.orders_backup_csv),
        "filters": {
            "cleanup_send_only_symbols": sorted(CLEANUP_SEND_ONLY_SYMBOLS),
            "cleanup_send_exclude_symbols": sorted(CLEANUP_SEND_EXCLUDE_SYMBOLS),
            "cleanup_send_max_rows": int(CLEANUP_SEND_MAX_ROWS),
            "cleanup_send_require_empty_pending_refs": int(CLEANUP_SEND_REQUIRE_EMPTY_PENDING_REFS),
            "cleanup_send_require_side": CLEANUP_SEND_REQUIRE_SIDE,
            "cleanup_send_reason_tag": CLEANUP_SEND_REASON_TAG,
        },
        "counters": counters,
    }
    return plan_df, summary


def _write_plan(paths: ConfigPaths, plan_df: pd.DataFrame, summary: dict) -> None:
    paths.execution_dir.mkdir(parents=True, exist_ok=True)
    plan_df.to_csv(paths.cleanup_send_plan_csv, index=False)
    paths.cleanup_send_summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def _replace_orders_with_cleanup_plan(paths: ConfigPaths, plan_df: pd.DataFrame) -> None:
    if paths.orders_csv.exists():
        shutil.copy2(paths.orders_csv, paths.orders_backup_csv)
    plan_df.to_csv(paths.orders_csv, index=False)


def _restore_orders_backup(paths: ConfigPaths) -> None:
    if paths.orders_backup_csv.exists():
        shutil.copy2(paths.orders_backup_csv, paths.orders_csv)


def _run_adapter_for_config(config_name: str) -> None:
    script_path = ROOT / "scripts" / "run_broker_adapter_ibkr.py"
    _must_exist(script_path, "run_broker_adapter_ibkr.py")
    env = os.environ.copy()
    env["CONFIG_NAMES"] = config_name
    env["PAUSE_ON_EXIT"] = "0"
    cmd = [sys.executable, str(script_path)]
    print(f"[CLEANUP_SEND][RUN] {' '.join(cmd)} config={config_name}")
    completed = subprocess.run(cmd, cwd=str(ROOT), env=env)
    if completed.returncode != 0:
        raise RuntimeError(f"run_broker_adapter_ibkr.py failed for config={config_name} rc={completed.returncode}")


def _run_one_config(paths: ConfigPaths) -> None:
    cleanup_df = _load_cleanup_orders(paths)
    plan_df, summary = _build_send_plan(paths, cleanup_df)
    _write_plan(paths, plan_df, summary)
    counters = summary["counters"]
    print(
        f"[CLEANUP_SEND][{paths.name}][SUMMARY] cleanup_rows={counters['cleanup_rows']} emitted_rows={counters['emitted_rows']} "
        f"preview_only={summary['preview_only']} blocked_pending_refs_present={counters.get('blocked_pending_refs_present', 0)} "
        f"blocked_not_in_only_symbols={counters.get('blocked_not_in_only_symbols', 0)} blocked_excluded_symbol={counters.get('blocked_excluded_symbol', 0)} "
        f"blocked_side_not_allowed={counters.get('blocked_side_not_allowed', 0)} blocked_max_rows={counters.get('blocked_max_rows', 0)}"
    )
    if not plan_df.empty:
        print(f"[CLEANUP_SEND][{paths.name}][PLAN_TOP]")
        print(plan_df.head(min(TOPK_PRINT, len(plan_df))).to_string(index=False))
    else:
        print(f"[CLEANUP_SEND][{paths.name}][PLAN_TOP] none")
    print(f"[ARTIFACT] {paths.cleanup_send_plan_csv}")
    print(f"[ARTIFACT] {paths.cleanup_send_summary_json}")

    if CLEANUP_SEND_PREVIEW_ONLY:
        print(f"[CLEANUP_SEND][{paths.name}] preview-only mode, adapter not called")
        return
    if plan_df.empty:
        print(f"[CLEANUP_SEND][{paths.name}] no cleanup rows after filters, adapter not called")
        return

    _replace_orders_with_cleanup_plan(paths, plan_df)
    try:
        _run_adapter_for_config(paths.name)
    finally:
        _restore_orders_backup(paths)
        print(f"[CLEANUP_SEND][{paths.name}] original orders.csv restored from backup")


def main() -> int:
    _enable_line_buffering()
    print(f"[CFG] execution_root={EXECUTION_ROOT}")
    print(f"[CFG] configs={CONFIG_NAMES}")
    print(f"[CFG] cleanup_send_preview_only={int(CLEANUP_SEND_PREVIEW_ONLY)}")
    print(f"[CFG] cleanup_send_only_symbols={sorted(CLEANUP_SEND_ONLY_SYMBOLS)}")
    print(f"[CFG] cleanup_send_exclude_symbols={sorted(CLEANUP_SEND_EXCLUDE_SYMBOLS)}")
    print(
        f"[CFG] cleanup_send_max_rows={CLEANUP_SEND_MAX_ROWS} cleanup_send_require_empty_pending_refs={int(CLEANUP_SEND_REQUIRE_EMPTY_PENDING_REFS)} "
        f"cleanup_send_require_side={CLEANUP_SEND_REQUIRE_SIDE}"
    )
    for config_name in CONFIG_NAMES:
        _run_one_config(_cfg_paths(config_name))
    print("[FINAL] broker cleanup send complete")
    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 1
    _safe_exit(rc)
