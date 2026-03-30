from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from python_edge.broker.ibkr_models import ConfigPaths, PreparedOrder

FINAL_DUPLICATE_STATUSES = {
    "presubmitted",
    "submitted",
    "pending_submit",
    "pending_cancel",
    "api_pending",
    "api_cancelled",
    "partially_filled",
    "filled",
}


def load_broker_log(path: Path, config_name: str, broker_name: str, broker_platform: str, broker_account_id: str, utc_now_iso, reset: bool) -> dict:
    if reset or not path.exists():
        return {
            "config": config_name,
            "broker_name": broker_name,
            "broker_platform": broker_platform,
            "broker_account_id": broker_account_id,
            "created_at_utc": utc_now_iso(),
            "updated_at_utc": utc_now_iso(),
            "orders": {},
        }
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise RuntimeError(f"broker_log.json must contain a JSON object: {path}")
    payload.setdefault("config", config_name)
    payload.setdefault("broker_name", broker_name)
    payload.setdefault("broker_platform", broker_platform)
    payload.setdefault("broker_account_id", broker_account_id)
    payload.setdefault("created_at_utc", utc_now_iso())
    payload.setdefault("updated_at_utc", utc_now_iso())
    payload.setdefault("orders", {})
    if not isinstance(payload["orders"], dict):
        raise RuntimeError(f"broker_log.json orders must be an object: {path}")
    return payload


def save_broker_log(path: Path, broker_log: dict, utc_now_iso) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    broker_log["updated_at_utc"] = utc_now_iso()
    path.write_text(json.dumps(broker_log, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def existing_duplicate_status(broker_log: dict, idempotency_key: str) -> Optional[str]:
    entry = broker_log.get("orders", {}).get(idempotency_key)
    if not isinstance(entry, dict):
        return None
    status = str(entry.get("status", "")).strip().lower()
    if status in FINAL_DUPLICATE_STATUSES:
        return status
    return None


def upsert_broker_log_entry(broker_log: dict, entry: dict, utc_now_iso) -> None:
    broker_log.setdefault("orders", {})[entry["idempotency_key"]] = {
        "status": entry["status"],
        "client_order_id": entry["client_order_id"],
        "broker_order_id": entry["broker_order_id"],
        "perm_id": entry.get("perm_id", 0),
        "config": entry["config"],
        "date": entry["date"],
        "symbol": entry["symbol"],
        "broker_symbol": entry["broker_symbol"],
        "side": entry["side"],
        "qty": entry["qty"],
        "filled_qty": entry["filled_qty"],
        "remaining_qty": entry.get("remaining_qty", 0.0),
        "filled_avg_price": entry["filled_avg_price"],
        "order_notional": entry["order_notional"],
        "fill_notional": entry["fill_notional"],
        "submitted_at": entry["submitted_at"],
        "filled_at": entry["filled_at"],
        "mode": entry["mode"],
        "source_order_path": entry["source_order_path"],
        "request": entry.get("request", {}),
        "response": entry.get("response", {}),
        "updated_at_utc": utc_now_iso(),
    }


def append_or_replace_fills(fills_csv: Path, entries: List[dict]) -> None:
    cols = [
        "idempotency_key", "client_order_id", "broker_order_id", "perm_id", "config", "source_order_path",
        "date", "symbol", "broker_symbol", "side", "qty", "filled_qty", "remaining_qty", "price_hint",
        "filled_avg_price", "order_notional", "fill_notional", "status", "submitted_at", "filled_at", "mode",
    ]
    new_df = pd.DataFrame(entries)
    if new_df.empty:
        if not fills_csv.exists():
            pd.DataFrame(columns=cols).to_csv(fills_csv, index=False)
        return
    new_df = new_df[cols].copy()
    if fills_csv.exists():
        prev = pd.read_csv(fills_csv)
        for col in cols:
            if col not in prev.columns:
                prev[col] = ""
        prev = prev[cols].copy()
        out = pd.concat([prev, new_df], ignore_index=True).drop_duplicates(subset=["idempotency_key"], keep="last")
    else:
        out = new_df
    out.sort_values(["date", "symbol", "side", "idempotency_key"]).reset_index(drop=True).to_csv(fills_csv, index=False)


def duplicate_fill_entry(prepared: PreparedOrder, duplicate_status: str, source_path: Path, broker_log: dict) -> dict:
    existing = broker_log.get("orders", {}).get(prepared.idempotency_key, {})
    return {
        "idempotency_key": prepared.idempotency_key,
        "client_order_id": str(existing.get("client_order_id", prepared.client_tag)),
        "broker_order_id": str(existing.get("broker_order_id", "")),
        "perm_id": int(existing.get("perm_id", 0) or 0),
        "config": prepared.config,
        "source_order_path": str(source_path),
        "date": prepared.order_date,
        "symbol": prepared.symbol,
        "broker_symbol": prepared.broker_symbol,
        "side": prepared.order_side,
        "qty": float(prepared.qty),
        "filled_qty": float(existing.get("filled_qty", 0.0) or 0.0),
        "remaining_qty": float(existing.get("remaining_qty", 0.0) or 0.0),
        "price_hint": float(prepared.price),
        "filled_avg_price": float(existing.get("filled_avg_price", prepared.price) or prepared.price),
        "order_notional": float(prepared.order_notional),
        "fill_notional": float(existing.get("fill_notional", 0.0) or 0.0),
        "status": f"duplicate_skipped:{duplicate_status}",
        "submitted_at": str(existing.get("submitted_at", "")),
        "filled_at": str(existing.get("filled_at", "")),
        "mode": "ibkr_gateway",
    }