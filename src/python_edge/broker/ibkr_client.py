from __future__ import annotations

import queue
import threading
import time
from typing import Dict, List, Optional, Sequence

from ibapi.client import EClient
from ibapi.common import OrderId
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.wrapper import EWrapper


class IBKRApp(EWrapper, EClient):
    def __init__(self, utc_now_iso, to_float) -> None:
        EClient.__init__(self, self)
        self._utc_now_iso = utc_now_iso
        self._to_float = to_float
        self._next_valid_id_queue: queue.Queue[int] = queue.Queue()
        self._managed_accounts_queue: queue.Queue[str] = queue.Queue()
        self._submit_lock = threading.Lock()
        self._network_thread: Optional[threading.Thread] = None
        self._errors: List[dict] = []
        self.orders_by_ib_id: Dict[int, dict] = {}
        self.ibid_by_client_tag: Dict[str, int] = {}
        self.done_contract_details: Dict[int, bool] = {}
        self.contract_details: Dict[int, dict] = {}
        self.done_open_orders = False
        self.done_positions = False
        self.position_rows: List[dict] = []
        self.order_id_cursor: Optional[int] = None

    def error(self, reqId: int, errorCode: int, errorString: str, advancedOrderRejectJson: str = "") -> None:
        self._errors.append({
            "ts_utc": self._utc_now_iso(),
            "reqId": int(reqId),
            "errorCode": int(errorCode),
            "errorString": str(errorString),
            "advancedOrderRejectJson": str(advancedOrderRejectJson or ""),
        })
        print(f"[IB][ERROR] reqId={reqId} code={errorCode} msg={errorString}")

    def nextValidId(self, orderId: int) -> None:
        self.order_id_cursor = int(orderId)
        self._next_valid_id_queue.put_nowait(int(orderId))
        print(f"[IB] nextValidId={orderId}")

    def managedAccounts(self, accountsList: str) -> None:
        self._managed_accounts_queue.put_nowait(str(accountsList))
        print(f"[IB] managedAccounts={accountsList}")

    def openOrder(self, orderId: OrderId, contract: Contract, order: Order, orderState) -> None:
        client_tag = str(getattr(order, "orderRef", "") or "")
        entry = self.orders_by_ib_id.setdefault(int(orderId), {
            "ib_order_id": int(orderId),
            "symbol": str(getattr(contract, "symbol", "") or ""),
            "broker_symbol": str(getattr(contract, "localSymbol", "") or getattr(contract, "symbol", "") or ""),
            "status": str(getattr(orderState, "status", "") or ""),
            "client_tag": client_tag,
            "perm_id": int(getattr(order, "permId", 0) or 0),
            "action": str(getattr(order, "action", "") or ""),
            "total_qty": self._to_float(getattr(order, "totalQuantity", 0.0)),
            "filled_qty": 0.0,
            "remaining_qty": self._to_float(getattr(order, "totalQuantity", 0.0)),
            "avg_fill_price": 0.0,
            "last_fill_price": 0.0,
            "submitted_at_utc": self._utc_now_iso(),
            "fills": [],
        })
        entry["client_tag"] = client_tag or str(entry.get("client_tag", ""))
        if client_tag:
            self.ibid_by_client_tag[client_tag] = int(orderId)

    def openOrderEnd(self) -> None:
        self.done_open_orders = True
        print("[IB] openOrderEnd")

    def orderStatus(self, orderId, status, filled, remaining, avgFillPrice, permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice) -> None:
        entry = self.orders_by_ib_id.setdefault(int(orderId), {
            "ib_order_id": int(orderId),
            "status": str(status or ""),
            "filled_qty": float(filled or 0.0),
            "remaining_qty": float(remaining or 0.0),
            "avg_fill_price": float(avgFillPrice or 0.0),
            "last_fill_price": float(lastFillPrice or 0.0),
            "submitted_at_utc": self._utc_now_iso(),
            "fills": [],
        })
        entry["status"] = str(status or entry.get("status", ""))
        entry["filled_qty"] = float(filled or 0.0)
        entry["remaining_qty"] = float(remaining or 0.0)
        entry["avg_fill_price"] = float(avgFillPrice or 0.0)
        entry["last_fill_price"] = float(lastFillPrice or 0.0)
        entry["perm_id"] = int(permId or entry.get("perm_id", 0) or 0)

    def contractDetails(self, reqId: int, contractDetails) -> None:
        contract = getattr(contractDetails, "contract", None)
        self.contract_details[int(reqId)] = {
            "symbol": str(getattr(contract, "symbol", "") or ""),
            "localSymbol": str(getattr(contract, "localSymbol", "") or ""),
            "primaryExchange": str(getattr(contract, "primaryExchange", "") or ""),
            "exchange": str(getattr(contract, "exchange", "") or ""),
            "currency": str(getattr(contract, "currency", "") or ""),
            "secType": str(getattr(contract, "secType", "") or ""),
            "minTick": self._to_float(getattr(contractDetails, "minTick", 0.0)),
            "validExchanges": str(getattr(contractDetails, "validExchanges", "") or ""),
            "longName": str(getattr(contractDetails, "longName", "") or ""),
        }

    def contractDetailsEnd(self, reqId: int) -> None:
        self.done_contract_details[int(reqId)] = True

    def position(self, account: str, contract: Contract, position: float, avgCost: float) -> None:
        self.position_rows.append({
            "account": str(account),
            "symbol": str(getattr(contract, "symbol", "") or ""),
            "localSymbol": str(getattr(contract, "localSymbol", "") or ""),
            "position": self._to_float(position),
            "avgCost": self._to_float(avgCost),
        })

    def positionEnd(self) -> None:
        self.done_positions = True
        print("[IB] positionEnd")

    def start_network_loop(self) -> None:
        if self._network_thread is not None and self._network_thread.is_alive():
            return
        self._network_thread = threading.Thread(target=self.run, name="ibapi-network", daemon=True)
        self._network_thread.start()

    def wait_for_next_valid_id(self, timeout_sec: float) -> int:
        return int(self._next_valid_id_queue.get(timeout=timeout_sec))

    def wait_for_managed_accounts(self, timeout_sec: float) -> str:
        return str(self._managed_accounts_queue.get(timeout=timeout_sec))

    def allocate_order_id(self) -> int:
        with self._submit_lock:
            if self.order_id_cursor is None:
                raise RuntimeError("IB nextValidId was not received")
            oid = int(self.order_id_cursor)
            self.order_id_cursor += 1
            return oid

    def wait_until_open_orders_end(self, timeout_sec: float) -> None:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            if self.done_open_orders:
                return
            time.sleep(0.1)
        raise TimeoutError("Timed out waiting for openOrderEnd")

    def wait_until_positions_end(self, timeout_sec: float) -> None:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            if self.done_positions:
                return
            time.sleep(0.1)
        raise TimeoutError("Timed out waiting for positionEnd")

    def wait_for_contract_details(self, req_id: int, timeout_sec: float) -> dict:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            if self.done_contract_details.get(int(req_id), False):
                return self.contract_details.get(int(req_id), {})
            time.sleep(0.1)
        raise TimeoutError(f"Timed out waiting for contract details req_id={req_id}")

    def wait_for_order_terminalish(self, order_id: int, timeout_sec: float) -> dict:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            entry = self.orders_by_ib_id.get(int(order_id), {})
            status = str(entry.get("status", "")).strip().lower()
            if status in {"presubmitted", "submitted", "partiallyfilled", "filled", "cancelled", "inactive", "apicancelled", "pendingcancel", "pendingsubmit", "api_pending"}:
                return entry
            time.sleep(0.1)
        return self.orders_by_ib_id.get(int(order_id), {})