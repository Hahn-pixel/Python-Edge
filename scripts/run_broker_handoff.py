from __future__ import annotations

import json
import math
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
for p in [ROOT, SRC_DIR]:
    ps = str(p)
    if ps not in sys.path:
        sys.path.insert(0, ps)

PAUSE_ON_EXIT = str(os.getenv("PAUSE_ON_EXIT", "auto")).strip().lower()
EXECUTION_ROOT = Path(os.getenv("EXECUTION_ROOT", "artifacts/execution_loop"))
BROKER_HANDOFF_ROOT = Path(os.getenv("BROKER_HANDOFF_ROOT", "artifacts/broker_handoff"))
CONFIG_NAMES = [x.strip() for x in str(os.getenv("CONFIG_NAMES", "optimal|aggressive")).split("|") if x.strip()]
BROKER_NAME = str(os.getenv("BROKER_NAME", "MEXEM")).strip() or "MEXEM"
BROKER_PLATFORM = str(os.getenv("BROKER_PLATFORM", "IBKR")).strip() or "IBKR"
BROKER_ACCOUNT_ID = str(os.getenv("BROKER_ACCOUNT_ID", "")).strip()
HANDOFF_MODE = str(os.getenv("HANDOFF_MODE", "preview")).strip().lower()
MIN_ORDER_NOTIONAL = float(os.getenv("MIN_ORDER_NOTIONAL", "25.0"))
MAX_ORDER_NOTIONAL = float(os.getenv("MAX_ORDER_NOTIONAL", "25000.0"))
MAX_TOTAL_ORDERS = int(os.getenv("MAX_TOTAL_ORDERS", "100"))
MAX_ORDERS_PER_CONFIG = int(os.getenv("MAX_ORDERS_PER_CONFIG", "50"))
MAX_POSITION_NOTIONAL = float(os.getenv("MAX_POSITION_NOTIONAL", "10000.0"))
ALLOW_SHORTS = str(os.getenv("ALLOW_SHORTS", "1")).strip().lower() not in {"0", "false", "no", "off"}
ALLOW_FRACTIONAL_BROKER = str(os.getenv("ALLOW_FRACTIONAL_BROKER", "1")).strip().lower() not in {"0", "false", "no", "off"}
ROUND_FRACTIONAL_FOR_BROKER = str(os.getenv("ROUND_FRACTIONAL_FOR_BROKER", "0")).strip().lower() not in {"0", "false", "no", "off"}
TOPK_PRINT = int(os.getenv("TOPK_PRINT", "50"))
COMMISSION_BPS = float(os.getenv("COMMISSION_BPS", "0.50"))
COMMISSION_MIN_PER_ORDER = float(os.getenv("COMMISSION_MIN_PER_ORDER", "0.35"))
SLIPPAGE_BPS = float(os.getenv("SLIPPAGE_BPS", "1.50"))


@dataclass(frozen=True)
class ConfigArtifacts:
    name: str
    execution_dir: Path
    orders_csv: Path
    target_book_csv: Path
    execution_log_csv: Path
    state_json: Path


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


def _must_exist(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("float64")


def _estimate_commission(order_notional_abs: float) -> float:
    if order_notional_abs <= 0.0:
        return 0.0
    pct_fee = float(COMMISSION_BPS) / 10000.0 * float(order_notional_abs)
    return float(max(float(COMMISSION_MIN_PER_ORDER), pct_fee))


def _estimate_slippage(order_notional_abs: float) -> float:
    if order_notional_abs <= 0.0:
        return 0.0
    return float(SLIPPAGE_BPS) / 10000.0 * float(order_notional_abs)


def _cfg_paths(name: str) -> ConfigArtifacts:
    execution_dir = EXECUTION_ROOT / name
    return ConfigArtifacts(
        name=name,
        execution_dir=execution_dir,
        orders_csv=execution_dir / "orders.csv",
        target_book_csv=execution_dir / "target_book.csv",
        execution_log_csv=execution_dir / "execution_log.csv",
        state_json=execution_dir / "portfolio_state.json",
    )


def _load_execution_artifacts(cfg: ConfigArtifacts) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    _must_exist(cfg.orders_csv, f"Orders csv for {cfg.name}")
    _must_exist(cfg.target_book_csv, f"Target book csv for {cfg.name}")
    _must_exist(cfg.execution_log_csv, f"Execution log csv for {cfg.name}")
    _must_exist(cfg.state_json, f"Portfolio state json for {cfg.name}")

    orders = pd.read_csv(cfg.orders_csv)
    target = pd.read_csv(cfg.target_book_csv)
    exec_log = pd.read_csv(cfg.execution_log_csv)
    with cfg.state_json.open("r", encoding="utf-8") as fh:
        state = json.load(fh)

    if orders.empty:
        orders = pd.DataFrame(columns=[
            "date", "symbol", "price", "target_weight", "target_notional", "target_shares_raw",
            "current_shares", "target_shares", "delta_shares", "order_side", "order_notional",
            "order_notional_abs", "estimated_commission", "estimated_slippage", "estimated_total_cost", "skip_reason"
        ])
    if target.empty:
        target = pd.DataFrame(columns=["symbol", "weight", "target_notional", "target_shares"])
    if exec_log.empty:
        raise RuntimeError(f"Execution log is empty for {cfg.name}")

    return orders, target, exec_log, state


def _normalize_broker_quantity(qty: float) -> Tuple[float, str]:
    if ALLOW_FRACTIONAL_BROKER and not ROUND_FRACTIONAL_FOR_BROKER:
        return float(qty), "fractional"
    rounded = float(math.floor(qty + 1e-12))
    return rounded, "integer"


def _build_handoff_rows(cfg: ConfigArtifacts, orders: pd.DataFrame, target: pd.DataFrame, exec_log: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    latest_log = exec_log.tail(1).copy()
    exec_date = str(latest_log.iloc[0]["date"])
    target_map = target.set_index("symbol", drop=False).to_dict(orient="index") if len(target) else {}

    rows: List[Dict[str, object]] = []
    counters = {
        "rows_total": 0,
        "rows_hold": 0,
        "rows_skipped_min_notional": 0,
        "rows_skipped_short_not_allowed": 0,
        "rows_skipped_max_notional": 0,
        "rows_skipped_zero_after_rounding": 0,
        "rows_ready": 0,
    }

    for _, row in orders.iterrows():
        counters["rows_total"] += 1
        symbol = str(row.get("symbol", "")).upper()
        side = str(row.get("order_side", "HOLD")).upper()
        skip_reason = str(row.get("skip_reason", "") or "")
        price = float(row.get("price", 0.0) or 0.0)
        delta_shares = float(row.get("delta_shares", 0.0) or 0.0)
        target_weight = float(row.get("target_weight", 0.0) or 0.0)
        target_notional = float(row.get("target_notional", 0.0) or 0.0)
        raw_order_notional = float(row.get("order_notional", 0.0) or 0.0)
        exec_estimated_commission = float(row.get("estimated_commission", 0.0) or 0.0)
        exec_estimated_slippage = float(row.get("estimated_slippage", 0.0) or 0.0)
        exec_estimated_total_cost = float(row.get("estimated_total_cost", 0.0) or 0.0)

        if side == "HOLD":
            counters["rows_hold"] += 1
            if skip_reason == "below_min_notional":
                counters["rows_skipped_min_notional"] += 1
            continue

        broker_qty, broker_qty_mode = _normalize_broker_quantity(abs(delta_shares))
        broker_order_notional = float(broker_qty * price)
        broker_side = side
        broker_skip_reason = ""

        if broker_side == "SELL" and not ALLOW_SHORTS:
            current_shares = float(row.get("current_shares", 0.0) or 0.0)
            if abs(delta_shares) > abs(current_shares) + 1e-12:
                broker_skip_reason = "shorts_not_allowed"
                counters["rows_skipped_short_not_allowed"] += 1

        if broker_order_notional > float(MAX_ORDER_NOTIONAL):
            broker_skip_reason = "max_order_notional_exceeded"
            counters["rows_skipped_max_notional"] += 1

        if broker_qty <= 0.0:
            broker_skip_reason = "zero_after_rounding"
            counters["rows_skipped_zero_after_rounding"] += 1

        symbol_target = target_map.get(symbol, {})
        target_position_notional = float(symbol_target.get("target_notional", target_notional) or target_notional)
        capped_target_position_notional = max(-float(MAX_POSITION_NOTIONAL), min(float(MAX_POSITION_NOTIONAL), target_position_notional))

        broker_estimated_commission = _estimate_commission(abs(broker_order_notional))
        broker_estimated_slippage = _estimate_slippage(abs(broker_order_notional))
        broker_estimated_total_cost = float(broker_estimated_commission + broker_estimated_slippage)

        ready = broker_skip_reason == ""
        if ready:
            counters["rows_ready"] += 1

        rows.append(
            {
                "date": exec_date,
                "config": cfg.name,
                "broker_name": BROKER_NAME,
                "broker_platform": BROKER_PLATFORM,
                "broker_account_id": BROKER_ACCOUNT_ID,
                "handoff_mode": HANDOFF_MODE,
                "symbol": symbol,
                "broker_side": broker_side,
                "broker_qty": broker_qty,
                "broker_qty_mode": broker_qty_mode,
                "price_reference": price,
                "broker_order_notional": broker_order_notional,
                "raw_delta_shares": delta_shares,
                "raw_order_notional": raw_order_notional,
                "target_weight": target_weight,
                "target_position_notional": target_position_notional,
                "capped_target_position_notional": capped_target_position_notional,
                "exec_estimated_commission": exec_estimated_commission,
                "exec_estimated_slippage": exec_estimated_slippage,
                "exec_estimated_total_cost": exec_estimated_total_cost,
                "broker_estimated_commission": broker_estimated_commission,
                "broker_estimated_slippage": broker_estimated_slippage,
                "broker_estimated_total_cost": broker_estimated_total_cost,
                "ready": int(ready),
                "skip_reason": broker_skip_reason,
            }
        )

    handoff = pd.DataFrame(rows)
    if len(handoff):
        handoff = handoff.sort_values(["ready", "broker_order_notional", "symbol"], ascending=[False, False, True]).reset_index(drop=True)
    return handoff, counters


def _run_preflight(handoff: pd.DataFrame) -> Dict[str, object]:
    if handoff.empty:
        return {
            "ready_orders": 0,
            "blocked_orders": 0,
            "total_broker_notional": 0.0,
            "max_broker_order_notional": 0.0,
            "total_broker_estimated_commission": 0.0,
            "total_broker_estimated_slippage": 0.0,
            "total_broker_estimated_cost": 0.0,
            "configs_present": [],
            "passes": False,
            "reasons": ["empty_handoff"],
        }

    ready = handoff.loc[handoff["ready"] == 1].copy()
    blocked = handoff.loc[handoff["ready"] != 1].copy()
    reasons: List[str] = []

    if len(ready) == 0:
        reasons.append("no_ready_orders")
    if len(ready) > int(MAX_TOTAL_ORDERS):
        reasons.append("max_total_orders_exceeded")
    per_cfg = ready.groupby("config", sort=False).size().to_dict() if len(ready) else {}
    for cfg_name, cnt in per_cfg.items():
        if int(cnt) > int(MAX_ORDERS_PER_CONFIG):
            reasons.append(f"max_orders_per_config_exceeded:{cfg_name}")
    if len(ready) and float(_num(ready["broker_order_notional"]).max()) > float(MAX_ORDER_NOTIONAL):
        reasons.append("ready_order_above_max_order_notional")

    return {
        "ready_orders": int(len(ready)),
        "blocked_orders": int(len(blocked)),
        "total_broker_notional": float(_num(ready["broker_order_notional"]).sum()) if len(ready) else 0.0,
        "max_broker_order_notional": float(_num(ready["broker_order_notional"]).max()) if len(ready) else 0.0,
        "total_broker_estimated_commission": float(_num(ready["broker_estimated_commission"]).sum()) if len(ready) else 0.0,
        "total_broker_estimated_slippage": float(_num(ready["broker_estimated_slippage"]).sum()) if len(ready) else 0.0,
        "total_broker_estimated_cost": float(_num(ready["broker_estimated_total_cost"]).sum()) if len(ready) else 0.0,
        "configs_present": sorted(ready["config"].astype(str).unique().tolist()) if len(ready) else [],
        "passes": len(reasons) == 0,
        "reasons": reasons,
    }


def main() -> int:
    _enable_line_buffering()
    print(f"[CFG] execution_root={EXECUTION_ROOT}")
    print(f"[CFG] broker_handoff_root={BROKER_HANDOFF_ROOT}")
    print(f"[CFG] broker_name={BROKER_NAME} broker_platform={BROKER_PLATFORM} handoff_mode={HANDOFF_MODE}")
    print(f"[CFG] configs={CONFIG_NAMES}")
    print(
        f"[CFG] allow_fractional_broker={int(ALLOW_FRACTIONAL_BROKER)} round_fractional_for_broker={int(ROUND_FRACTIONAL_FOR_BROKER)} "
        f"allow_shorts={int(ALLOW_SHORTS)} min_order_notional={MIN_ORDER_NOTIONAL} max_order_notional={MAX_ORDER_NOTIONAL}"
    )
    print(
        f"[CFG] max_total_orders={MAX_TOTAL_ORDERS} max_orders_per_config={MAX_ORDERS_PER_CONFIG} "
        f"max_position_notional={MAX_POSITION_NOTIONAL}"
    )
    print(
        f"[CFG] commission_bps={COMMISSION_BPS} commission_min_per_order={COMMISSION_MIN_PER_ORDER} slippage_bps={SLIPPAGE_BPS}"
    )

    all_rows: List[pd.DataFrame] = []
    cfg_summaries: Dict[str, object] = {}

    for name in CONFIG_NAMES:
        cfg = _cfg_paths(name)
        orders, target, exec_log, state = _load_execution_artifacts(cfg)
        handoff_df, counters = _build_handoff_rows(cfg, orders, target, exec_log)
        cfg_summaries[name] = {
            "rows_total": int(counters["rows_total"]),
            "rows_hold": int(counters["rows_hold"]),
            "rows_ready": int(counters["rows_ready"]),
            "rows_skipped_min_notional": int(counters["rows_skipped_min_notional"]),
            "rows_skipped_short_not_allowed": int(counters["rows_skipped_short_not_allowed"]),
            "rows_skipped_max_notional": int(counters["rows_skipped_max_notional"]),
            "rows_skipped_zero_after_rounding": int(counters["rows_skipped_zero_after_rounding"]),
            "state_nav": float(state.get("nav", 0.0) or 0.0),
            "state_cash": float(state.get("cash", 0.0) or 0.0),
            "broker_estimated_commission_total": float(_num(handoff_df.get("broker_estimated_commission", pd.Series(dtype="float64"))).sum()) if len(handoff_df) else 0.0,
            "broker_estimated_slippage_total": float(_num(handoff_df.get("broker_estimated_slippage", pd.Series(dtype="float64"))).sum()) if len(handoff_df) else 0.0,
            "broker_estimated_total_cost": float(_num(handoff_df.get("broker_estimated_total_cost", pd.Series(dtype="float64"))).sum()) if len(handoff_df) else 0.0,
        }
        print(
            f"[HANDOFF][{name}] rows_total={counters['rows_total']} rows_ready={counters['rows_ready']} rows_hold={counters['rows_hold']} "
            f"skip_min_notional={counters['rows_skipped_min_notional']} skip_shorts={counters['rows_skipped_short_not_allowed']} "
            f"skip_max_notional={counters['rows_skipped_max_notional']} skip_zero_after_rounding={counters['rows_skipped_zero_after_rounding']}"
        )
        print(
            f"[HANDOFF][{name}][COST] broker_estimated_commission_total={cfg_summaries[name]['broker_estimated_commission_total']:.4f} "
            f"broker_estimated_slippage_total={cfg_summaries[name]['broker_estimated_slippage_total']:.4f} "
            f"broker_estimated_total_cost={cfg_summaries[name]['broker_estimated_total_cost']:.4f}"
        )
        if len(handoff_df):
            print(f"[HANDOFF][{name}][TOP]")
            print(handoff_df.head(min(TOPK_PRINT, len(handoff_df))).to_string(index=False))
        all_rows.append(handoff_df)

    combined = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    preflight = _run_preflight(combined)

    BROKER_HANDOFF_ROOT.mkdir(parents=True, exist_ok=True)
    blotter_csv = BROKER_HANDOFF_ROOT / "broker_order_blotter.csv"
    ready_csv = BROKER_HANDOFF_ROOT / "broker_order_blotter_ready.csv"
    blocked_csv = BROKER_HANDOFF_ROOT / "broker_order_blotter_blocked.csv"
    preflight_json = BROKER_HANDOFF_ROOT / "broker_preflight_summary.json"

    combined.to_csv(blotter_csv, index=False)
    if len(combined):
        combined.loc[combined["ready"] == 1].to_csv(ready_csv, index=False)
        combined.loc[combined["ready"] != 1].to_csv(blocked_csv, index=False)
    else:
        pd.DataFrame().to_csv(ready_csv, index=False)
        pd.DataFrame().to_csv(blocked_csv, index=False)

    summary_payload = {
        "broker_name": BROKER_NAME,
        "broker_platform": BROKER_PLATFORM,
        "broker_account_id": BROKER_ACCOUNT_ID,
        "handoff_mode": HANDOFF_MODE,
        "config_summaries": cfg_summaries,
        "preflight": preflight,
    }
    preflight_json.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"[PREFLIGHT] passes={int(preflight['passes'])} ready_orders={preflight['ready_orders']} blocked_orders={preflight['blocked_orders']} "
        f"total_broker_notional={preflight['total_broker_notional']:.2f} max_broker_order_notional={preflight['max_broker_order_notional']:.2f}"
    )
    print(
        f"[PREFLIGHT][COST] total_broker_estimated_commission={preflight['total_broker_estimated_commission']:.4f} "
        f"total_broker_estimated_slippage={preflight['total_broker_estimated_slippage']:.4f} "
        f"total_broker_estimated_cost={preflight['total_broker_estimated_cost']:.4f}"
    )
    if preflight["reasons"]:
        print(f"[PREFLIGHT][REASONS] {preflight['reasons']}")

    print(f"[ARTIFACT] {blotter_csv}")
    print(f"[ARTIFACT] {ready_csv}")
    print(f"[ARTIFACT] {blocked_csv}")
    print(f"[ARTIFACT] {preflight_json}")
    print("[FINAL] broker handoff complete")
    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 1
    _safe_exit(rc)