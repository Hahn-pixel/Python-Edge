from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from python_edge.broker.cpapi_models import ExecState, ExecutionIntent, FillResult

# ---------------------------------------------------------------------------
# Statuses that prevent re-submission (idempotency guard)
# ---------------------------------------------------------------------------

FINAL_DUPLICATE_STATUSES = frozenset({
    "presubmitted",
    "submitted",
    "pending_submit",
    "pending_cancel",
    "api_pending",
    "api_cancelled",
    "partially_filled",
    "filled",
    "done",           # CPAPI-specific terminal state
})

# fills.csv column order — matches ibkr_storage exactly so downstream
# consumers (run_execution_loop, reports) see the same schema
_FILLS_COLS = [
    "idempotency_key", "client_order_id", "broker_order_id", "perm_id",
    "config", "source_order_path",
    "date", "symbol", "broker_symbol", "side",
    "qty", "filled_qty", "remaining_qty",
    "price_hint", "filled_avg_price",
    "order_notional", "fill_notional",
    "status", "submitted_at", "filled_at", "mode",
]

_MODE = "cpapi_gateway"


# ---------------------------------------------------------------------------
# broker_log load / save  (identical contract to ibkr_storage)
# ---------------------------------------------------------------------------

def load_broker_log(
    path: Path,
    config_name: str,
    broker_name: str,
    broker_platform: str,
    broker_account_id: str,
    utc_now_iso,
    reset: bool = False,
) -> dict:
    if reset or not path.exists():
        return {
            "config":            config_name,
            "broker_name":       broker_name,
            "broker_platform":   broker_platform,
            "broker_account_id": broker_account_id,
            "created_at_utc":    utc_now_iso(),
            "updated_at_utc":    utc_now_iso(),
            "orders":            {},
        }
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise RuntimeError(f"broker_log.json must contain a JSON object: {path}")
    payload.setdefault("config",            config_name)
    payload.setdefault("broker_name",       broker_name)
    payload.setdefault("broker_platform",   broker_platform)
    payload.setdefault("broker_account_id", broker_account_id)
    payload.setdefault("created_at_utc",    utc_now_iso())
    payload.setdefault("updated_at_utc",    utc_now_iso())
    payload.setdefault("orders",            {})
    if not isinstance(payload["orders"], dict):
        raise RuntimeError(f"broker_log.json orders must be an object: {path}")
    return payload


def save_broker_log(path: Path, broker_log: dict, utc_now_iso) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    broker_log["updated_at_utc"] = utc_now_iso()
    path.write_text(
        json.dumps(broker_log, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Idempotency guard
# ---------------------------------------------------------------------------

def existing_duplicate_status(
    broker_log: dict,
    idempotency_key: str,
) -> Optional[str]:
    """Return the stored status if the key already exists and is terminal."""
    entry = broker_log.get("orders", {}).get(idempotency_key)
    if not isinstance(entry, dict):
        return None
    status = str(entry.get("status", "")).strip().lower()
    if status in FINAL_DUPLICATE_STATUSES:
        return status
    return None


# ---------------------------------------------------------------------------
# Build log entry from ExecutionIntent + FillResult
# ---------------------------------------------------------------------------

def build_broker_log_entry(
    config_name: str,
    order_date: str,
    order_notional: float,
    price_hint: float,
    source_order_path: str,
    idempotency_key: str,
    intent: ExecutionIntent,
    result: FillResult,
    utc_now_iso,
) -> dict:
    """
    Produce a dict that is structurally identical to what ibkr_storage
    writes — so the same downstream consumers work unchanged.

    Key fields that differ from ibkr:
      mode = "cpapi_gateway"
      broker_order_id = whole_order_id or frac_order_id (whichever is set)
      response.whole_* and response.frac_* capture the split execution detail
    """
    filled_qty    = float(result.total_filled)
    remaining_qty = max(0.0, float(intent.target_qty) - filled_qty)
    avg_price     = float(result.avg_price)
    fill_notional = round(filled_qty * avg_price, 6) if filled_qty > 0.0 else 0.0
    now           = utc_now_iso()

    # Map ExecState → status string compatible with ibkr_storage consumers
    state = result.final_state
    if state is ExecState.DONE:
        if filled_qty > 0.0 and remaining_qty <= 1e-9:
            status_str = "filled"
            outcome    = "filled_now"
        elif filled_qty > 0.0:
            status_str = "partially_filled"
            outcome    = "partial"
        else:
            # DONE with zero fill (e.g. pure-frac that didn't fill)
            status_str = "submitted"
            outcome    = "working"
    elif state is ExecState.FAILED:
        status_str = "cancelled"
        outcome    = "failed"
    else:
        # Should not normally reach storage in a non-terminal state,
        # but handle it explicitly rather than silently coercing
        status_str = state.value.lower()
        outcome    = "unknown"

    broker_order_id = str(
        intent.whole_order_id or intent.frac_order_id or ""
    )
    client_order_id = str(intent.client_tag)

    # Response block — mirrors ibkr response but adds split-execution detail
    response: Dict = {
        "outcome":          outcome,
        "status":           status_str,
        "avg_fill_price":   avg_price,
        "fills":            [],               # CPAPI doesn't return tick-fills
        "whole_order_id":   str(intent.whole_order_id or ""),
        "whole_filled_qty": float(intent.whole_filled_qty),
        "whole_avg_price":  float(intent.whole_avg_price),
        "frac_order_id":    str(intent.frac_order_id or ""),
        "frac_filled_qty":  float(intent.frac_filled_qty),
        "frac_avg_price":   float(intent.frac_avg_price),
        "transitions":      list(intent.transitions),
        "debug": {
            "precheck_ok":      intent.debug_precheck_ok,
            "precheck_fail":    intent.debug_precheck_fail,
            "split_ok":         intent.debug_split_ok,
            "whole_submitted":  intent.debug_whole_submitted,
            "whole_filled":     intent.debug_whole_filled,
            "whole_partial":    intent.debug_whole_partial,
            "whole_timeout":    intent.debug_whole_timeout,
            "frac_submitted":   intent.debug_frac_submitted,
            "frac_filled":      intent.debug_frac_filled,
            "guard_rejected":   intent.debug_guard_rejected,
            "failed":           intent.debug_failed,
        },
    }

    # Surface errors in the response (mirrors ibkr_storage issue handling)
    if intent.errors:
        response["errors"] = list(intent.errors)
        response["error"]  = str(intent.errors[-1].get("error", ""))

    entry = {
        "idempotency_key":  idempotency_key,
        "client_order_id":  client_order_id,
        "broker_order_id":  broker_order_id,
        "perm_id":          0,        # CPAPI does not expose permId
        "config":           config_name,
        "date":             order_date,
        "symbol":           intent.symbol,
        "broker_symbol":    intent.symbol,   # no symbol remap in CPAPI path
        "side":             intent.side.value,
        "qty":              float(intent.target_qty),
        "filled_qty":       filled_qty,
        "remaining_qty":    remaining_qty,
        "price_hint":       float(price_hint),
        "filled_avg_price": avg_price,
        "order_notional":   float(order_notional),
        "fill_notional":    fill_notional,
        "status":           status_str,
        "submitted_at":     now,
        "filled_at":        now if filled_qty > 0.0 else "",
        "mode":             _MODE,
        "source_order_path": source_order_path,
        "request": {
            "conid":       intent.conid,
            "whole_qty":   float(intent.whole_qty),
            "frac_qty":    float(intent.frac_qty),
            "parent_cap":  intent.parent_cap,
            "parent_floor": intent.parent_floor,
            "client_tag":  intent.client_tag,
            "account_id":  intent.account_id,
        },
        "response": response,
    }
    return entry


# ---------------------------------------------------------------------------
# Upsert into broker_log dict (same signature as ibkr_storage)
# ---------------------------------------------------------------------------

def upsert_broker_log_entry(
    broker_log: dict,
    entry: dict,
    utc_now_iso,
) -> None:
    broker_log.setdefault("orders", {})[entry["idempotency_key"]] = {
        "status":           entry["status"],
        "client_order_id":  entry["client_order_id"],
        "broker_order_id":  entry["broker_order_id"],
        "perm_id":          entry.get("perm_id", 0),
        "config":           entry["config"],
        "date":             entry["date"],
        "symbol":           entry["symbol"],
        "broker_symbol":    entry["broker_symbol"],
        "side":             entry["side"],
        "qty":              entry["qty"],
        "filled_qty":       entry["filled_qty"],
        "remaining_qty":    entry.get("remaining_qty", 0.0),
        "filled_avg_price": entry["filled_avg_price"],
        "order_notional":   entry["order_notional"],
        "fill_notional":    entry["fill_notional"],
        "submitted_at":     entry["submitted_at"],
        "filled_at":        entry["filled_at"],
        "mode":             entry["mode"],
        "source_order_path": entry["source_order_path"],
        "request":          entry.get("request", {}),
        "response":         entry.get("response", {}),
        "updated_at_utc":   utc_now_iso(),
    }


# ---------------------------------------------------------------------------
# fills.csv (identical schema to ibkr_storage)
# ---------------------------------------------------------------------------

def append_or_replace_fills(fills_csv: Path, entries: List[dict]) -> None:
    new_df = pd.DataFrame(entries)
    if new_df.empty:
        if not fills_csv.exists():
            pd.DataFrame(columns=_FILLS_COLS).to_csv(fills_csv, index=False)
        return

    # Add any missing columns with empty string
    for col in _FILLS_COLS:
        if col not in new_df.columns:
            new_df[col] = ""
    new_df = new_df[_FILLS_COLS].copy()

    if fills_csv.exists():
        prev = pd.read_csv(fills_csv)
        for col in _FILLS_COLS:
            if col not in prev.columns:
                prev[col] = ""
        prev = prev[_FILLS_COLS].copy()
        out = (
            pd.concat([prev, new_df], ignore_index=True)
            .drop_duplicates(subset=["idempotency_key"], keep="last")
        )
    else:
        out = new_df

    fills_csv.parent.mkdir(parents=True, exist_ok=True)
    (
        out
        .sort_values(["date", "symbol", "side", "idempotency_key"])
        .reset_index(drop=True)
        .to_csv(fills_csv, index=False)
    )


# ---------------------------------------------------------------------------
# Duplicate fill entry (for idempotency-skipped orders)
# ---------------------------------------------------------------------------

def duplicate_fill_entry(
    idempotency_key: str,
    client_tag: str,
    symbol: str,
    side: str,
    qty: float,
    price_hint: float,
    order_notional: float,
    order_date: str,
    config_name: str,
    source_order_path: str,
    duplicate_status: str,
    broker_log: dict,
) -> dict:
    existing = broker_log.get("orders", {}).get(idempotency_key, {})
    return {
        "idempotency_key":  idempotency_key,
        "client_order_id":  str(existing.get("client_order_id", client_tag)),
        "broker_order_id":  str(existing.get("broker_order_id", "")),
        "perm_id":          int(existing.get("perm_id", 0) or 0),
        "config":           config_name,
        "source_order_path": source_order_path,
        "date":             order_date,
        "symbol":           symbol,
        "broker_symbol":    symbol,
        "side":             side,
        "qty":              float(qty),
        "filled_qty":       float(existing.get("filled_qty", 0.0) or 0.0),
        "remaining_qty":    float(existing.get("remaining_qty", 0.0) or 0.0),
        "price_hint":       float(price_hint),
        "filled_avg_price": float(existing.get("filled_avg_price", price_hint) or price_hint),
        "order_notional":   float(order_notional),
        "fill_notional":    float(existing.get("fill_notional", 0.0) or 0.0),
        "status":           f"duplicate_skipped:{duplicate_status}",
        "submitted_at":     str(existing.get("submitted_at", "")),
        "filled_at":        str(existing.get("filled_at", "")),
        "mode":             _MODE,
    }
