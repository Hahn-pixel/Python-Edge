from __future__ import annotations

import json
import math
import os
import sys
import traceback
from dataclasses import dataclass
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
CLEANUP_ONLY_SYMBOLS = {x.strip().upper() for x in str(os.getenv("CLEANUP_ONLY_SYMBOLS", "")).split("|") if x.strip()}
CLEANUP_EXCLUDE_SYMBOLS = {x.strip().upper() for x in str(os.getenv("CLEANUP_EXCLUDE_SYMBOLS", "")).split("|") if x.strip()}
CLEANUP_UNEXPECTED_ONLY = str(os.getenv("CLEANUP_UNEXPECTED_ONLY", "0")).strip().lower() not in {"0", "false", "no", "off"}
CLEANUP_INCLUDE_DRIFT = str(os.getenv("CLEANUP_INCLUDE_DRIFT", "1")).strip().lower() not in {"0", "false", "no", "off"}
CLEANUP_MIN_ABS_SHARES = float(os.getenv("CLEANUP_MIN_ABS_SHARES", "1.0"))
CLEANUP_MAX_ROWS = int(os.getenv("CLEANUP_MAX_ROWS", "1000000"))
CLEANUP_REQUIRE_NO_PENDING = str(os.getenv("CLEANUP_REQUIRE_NO_PENDING", "1")).strip().lower() not in {"0", "false", "no", "off"}
CLEANUP_REQUIRE_BROKER_PRICE = str(os.getenv("CLEANUP_REQUIRE_BROKER_PRICE", "0")).strip().lower() not in {"0", "false", "no", "off"}
TOPK_PRINT = int(os.getenv("TOPK_PRINT", "50"))
EMIT_REASON_TAG = str(os.getenv("EMIT_REASON_TAG", "broker_cleanup_emit")).strip() or "broker_cleanup_emit"


@dataclass(frozen=True)
class ConfigPaths:
    name: str
    execution_dir: Path
    cleanup_preview_csv: Path
    cleanup_orders_csv: Path
    cleanup_emit_summary_json: Path


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


def _to_float(value: object) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def _cfg_paths(config_name: str) -> ConfigPaths:
    base = EXECUTION_ROOT / config_name
    return ConfigPaths(
        name=config_name,
        execution_dir=base,
        cleanup_preview_csv=base / "broker_cleanup_preview.csv",
        cleanup_orders_csv=base / "broker_cleanup_orders.csv",
        cleanup_emit_summary_json=base / "broker_cleanup_emit_summary.json",
    )


def _load_preview(paths: ConfigPaths) -> pd.DataFrame:
    _must_exist(paths.cleanup_preview_csv, f"cleanup preview for {paths.name}")
    df = pd.read_csv(paths.cleanup_preview_csv)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "cleanup_side",
                "cleanup_qty",
                "cleanup_reason",
                "broker_avg_cost",
                "broker_position",
                "expected_shares",
                "pending_side",
                "pending_qty",
                "pending_status",
                "pending_order_refs",
            ]
        )
    required = ["symbol", "cleanup_side", "cleanup_qty", "cleanup_reason"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"cleanup preview missing required columns {missing}: {paths.cleanup_preview_csv}")
    out = df.copy()
    out["symbol"] = out["symbol"].astype(str).map(_norm_symbol)
    out["cleanup_side"] = out["cleanup_side"].astype(str).str.strip().str.upper()
    out["cleanup_qty"] = pd.to_numeric(out["cleanup_qty"], errors="coerce").fillna(0.0)
    if "broker_avg_cost" in out.columns:
        out["broker_avg_cost"] = pd.to_numeric(out["broker_avg_cost"], errors="coerce")
    else:
        out["broker_avg_cost"] = float("nan")
    if "broker_position" in out.columns:
        out["broker_position"] = pd.to_numeric(out["broker_position"], errors="coerce").fillna(0.0)
    else:
        out["broker_position"] = 0.0
    if "expected_shares" in out.columns:
        out["expected_shares"] = pd.to_numeric(out["expected_shares"], errors="coerce").fillna(0.0)
    else:
        out["expected_shares"] = 0.0
    if "pending_qty" in out.columns:
        out["pending_qty"] = pd.to_numeric(out["pending_qty"], errors="coerce").fillna(0.0)
    else:
        out["pending_qty"] = 0.0
    for col in ["pending_side", "pending_status", "pending_order_refs"]:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].astype(str).fillna("")
    return out.sort_values(["cleanup_qty", "symbol"], ascending=[False, True]).reset_index(drop=True)


def _passes_filters(row: pd.Series) -> tuple[bool, str]:
    symbol = _norm_symbol(row.get("symbol", ""))
    reason = str(row.get("cleanup_reason", "") or "").strip()
    qty = abs(_to_float(row.get("cleanup_qty", 0.0)))
    pending_qty = abs(_to_float(row.get("pending_qty", 0.0)))
    broker_avg_cost = _to_float(row.get("broker_avg_cost", 0.0))

    if not symbol:
        return False, "empty_symbol"
    if CLEANUP_ONLY_SYMBOLS and symbol not in CLEANUP_ONLY_SYMBOLS:
        return False, "not_in_only_symbols"
    if CLEANUP_EXCLUDE_SYMBOLS and symbol in CLEANUP_EXCLUDE_SYMBOLS:
        return False, "excluded_symbol"
    if qty < float(CLEANUP_MIN_ABS_SHARES):
        return False, "below_min_abs_shares"
    if CLEANUP_UNEXPECTED_ONLY and reason != "unexpected_at_broker":
        return False, "unexpected_only_filter"
    if not CLEANUP_INCLUDE_DRIFT and reason == "drift":
        return False, "drift_disabled"
    if CLEANUP_REQUIRE_NO_PENDING and pending_qty > 0.0:
        return False, "pending_exists"
    if CLEANUP_REQUIRE_BROKER_PRICE and not (math.isfinite(broker_avg_cost) and broker_avg_cost > 0.0):
        return False, "missing_broker_price"
    side = str(row.get("cleanup_side", "") or "").strip().upper()
    if side not in {"BUY", "SELL"}:
        return False, "invalid_cleanup_side"
    return True, "emit"


def _emit_orders(paths: ConfigPaths, preview_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows: List[Dict[str, object]] = []
    counters: Dict[str, int] = {
        "preview_rows": int(len(preview_df)),
        "emitted_rows": 0,
        "blocked_empty_symbol": 0,
        "blocked_not_in_only_symbols": 0,
        "blocked_excluded_symbol": 0,
        "blocked_below_min_abs_shares": 0,
        "blocked_unexpected_only_filter": 0,
        "blocked_drift_disabled": 0,
        "blocked_pending_exists": 0,
        "blocked_missing_broker_price": 0,
        "blocked_invalid_cleanup_side": 0,
        "blocked_max_rows": 0,
    }

    emitted_count = 0
    for _, row in preview_df.iterrows():
        ok, reason = _passes_filters(row)
        if not ok:
            key = f"blocked_{reason}"
            counters[key] = int(counters.get(key, 0)) + 1
            continue
        if emitted_count >= int(CLEANUP_MAX_ROWS):
            counters["blocked_max_rows"] += 1
            continue
        symbol = _norm_symbol(row["symbol"])
        cleanup_side = str(row["cleanup_side"] or "").strip().upper()
        cleanup_qty = abs(_to_float(row["cleanup_qty"]))
        broker_position = _to_float(row.get("broker_position", 0.0))
        expected_shares = _to_float(row.get("expected_shares", 0.0))
        broker_avg_cost = _to_float(row.get("broker_avg_cost", 0.0))
        pending_qty = _to_float(row.get("pending_qty", 0.0))
        emitted_count += 1
        counters["emitted_rows"] = emitted_count
        rows.append(
            {
                "date": "",
                "symbol": symbol,
                "order_side": cleanup_side,
                "delta_shares": cleanup_qty,
                "price": pd.NA,
                "price_source": "cleanup_preview",
                "is_priced": 0,
                "target_weight": 0.0,
                "target_notional": 0.0,
                "target_shares_raw": 0.0,
                "current_shares": broker_position,
                "target_shares": expected_shares,
                "raw_delta_shares": cleanup_qty if cleanup_side == "BUY" else -cleanup_qty,
                "order_notional": 0.0,
                "order_notional_abs": 0.0,
                "estimated_commission": 0.0,
                "estimated_slippage": 0.0,
                "estimated_total_cost": 0.0,
                "skip_reason": "",
                "cleanup_reason": str(row.get("cleanup_reason", "") or ""),
                "cleanup_tag": EMIT_REASON_TAG,
                "broker_avg_cost": broker_avg_cost,
                "broker_position": broker_position,
                "expected_shares": expected_shares,
                "pending_side": str(row.get("pending_side", "") or ""),
                "pending_qty": pending_qty,
                "pending_status": str(row.get("pending_status", "") or ""),
                "pending_order_refs": str(row.get("pending_order_refs", "") or ""),
            }
        )

    orders_df = pd.DataFrame(rows)
    if not orders_df.empty:
        orders_df = orders_df.sort_values(["order_notional_abs", "delta_shares", "symbol"], ascending=[False, False, True]).reset_index(drop=True)
    summary = {
        "config": paths.name,
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "preview_csv": str(paths.cleanup_preview_csv),
        "orders_csv": str(paths.cleanup_orders_csv),
        "filters": {
            "cleanup_only_symbols": sorted(CLEANUP_ONLY_SYMBOLS),
            "cleanup_exclude_symbols": sorted(CLEANUP_EXCLUDE_SYMBOLS),
            "cleanup_unexpected_only": int(CLEANUP_UNEXPECTED_ONLY),
            "cleanup_include_drift": int(CLEANUP_INCLUDE_DRIFT),
            "cleanup_min_abs_shares": float(CLEANUP_MIN_ABS_SHARES),
            "cleanup_max_rows": int(CLEANUP_MAX_ROWS),
            "cleanup_require_no_pending": int(CLEANUP_REQUIRE_NO_PENDING),
            "cleanup_require_broker_price": int(CLEANUP_REQUIRE_BROKER_PRICE),
            "emit_reason_tag": EMIT_REASON_TAG,
        },
        "counters": counters,
    }
    return orders_df, summary


def _write_outputs(paths: ConfigPaths, orders_df: pd.DataFrame, summary: dict) -> None:
    paths.execution_dir.mkdir(parents=True, exist_ok=True)
    orders_df.to_csv(paths.cleanup_orders_csv, index=False)
    paths.cleanup_emit_summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def _run_one_config(paths: ConfigPaths) -> None:
    preview_df = _load_preview(paths)
    orders_df, summary = _emit_orders(paths, preview_df)
    _write_outputs(paths, orders_df, summary)
    counters = summary["counters"]
    print(
        f"[CLEANUP_EMIT][{paths.name}][SUMMARY] preview_rows={counters['preview_rows']} emitted_rows={counters['emitted_rows']} "
        f"blocked_pending_exists={counters.get('blocked_pending_exists', 0)} blocked_not_in_only_symbols={counters.get('blocked_not_in_only_symbols', 0)} "
        f"blocked_excluded_symbol={counters.get('blocked_excluded_symbol', 0)} blocked_below_min_abs_shares={counters.get('blocked_below_min_abs_shares', 0)} "
        f"blocked_unexpected_only_filter={counters.get('blocked_unexpected_only_filter', 0)} blocked_drift_disabled={counters.get('blocked_drift_disabled', 0)} "
        f"blocked_missing_broker_price={counters.get('blocked_missing_broker_price', 0)} blocked_max_rows={counters.get('blocked_max_rows', 0)}"
    )
    if not orders_df.empty:
        print(f"[CLEANUP_EMIT][{paths.name}][ORDERS_TOP]")
        print(orders_df.head(min(TOPK_PRINT, len(orders_df))).to_string(index=False))
    else:
        print(f"[CLEANUP_EMIT][{paths.name}][ORDERS_TOP] none")
    print(f"[ARTIFACT] {paths.cleanup_orders_csv}")
    print(f"[ARTIFACT] {paths.cleanup_emit_summary_json}")


def main() -> int:
    _enable_line_buffering()
    print(f"[CFG] execution_root={EXECUTION_ROOT}")
    print(f"[CFG] configs={CONFIG_NAMES}")
    print(f"[CFG] cleanup_only_symbols={sorted(CLEANUP_ONLY_SYMBOLS)}")
    print(f"[CFG] cleanup_exclude_symbols={sorted(CLEANUP_EXCLUDE_SYMBOLS)}")
    print(
        f"[CFG] cleanup_unexpected_only={int(CLEANUP_UNEXPECTED_ONLY)} cleanup_include_drift={int(CLEANUP_INCLUDE_DRIFT)} "
        f"cleanup_min_abs_shares={CLEANUP_MIN_ABS_SHARES:.8f} cleanup_max_rows={CLEANUP_MAX_ROWS} "
        f"cleanup_require_no_pending={int(CLEANUP_REQUIRE_NO_PENDING)} cleanup_require_broker_price={int(CLEANUP_REQUIRE_BROKER_PRICE)}"
    )
    for config_name in CONFIG_NAMES:
        _run_one_config(_cfg_paths(config_name))
    print("[FINAL] broker cleanup emit complete")
    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 1
    _safe_exit(rc)
