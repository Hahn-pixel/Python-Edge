from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True)
class BrokerErrorInfo:
    req_id: int
    error_code: int
    error_string: str
    advanced_order_reject_json: str
    ts_utc: str


@dataclass(frozen=True)
class OrderIssue:
    kind: str
    status: str
    broker_error: BrokerErrorInfo | None
    message: str


@dataclass(frozen=True)
class ConfigPaths:
    name: str
    execution_dir: Path
    orders_csv: Path
    fills_csv: Path
    broker_log_json: Path


@dataclass(frozen=True)
class PreparedOrder:
    config: str
    order_date: str
    symbol: str
    broker_symbol: str
    order_side: str
    qty: float
    price: float
    order_notional: float
    target_weight: float
    current_shares: float
    target_shares: float
    delta_shares: float
    source_row: Dict[str, Any]
    idempotency_key: str
    client_tag: str
    is_fractional_probe: bool = False
    min_tick: float = 0.0