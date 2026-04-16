from __future__ import annotations

import json
import threading
import time
import traceback
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import requests
import urllib3

from python_edge.broker.cpapi_models import (
    AuthStatus,
    CpapiOrderRequest,
    CpapiOrderResponse,
    CpapiOrderStatus,
    CpapiPosition,
)

# CPAPI runs on localhost with a self-signed cert — suppress noise
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TICKLE_INTERVAL_SEC = 50        # must be < 60 to keep session alive
_DEFAULT_TIMEOUT_SEC = 10        # per-request HTTP timeout
_CONFIRM_RETRY_MAX   = 3         # how many times to retry /iserver/reply


class CpapiClient:
    """
    Thin HTTP client for the Interactive Brokers Client Portal Web API.

    Responsibilities:
      - Session management (auth check, /tickle keep-alive daemon thread)
      - Order submission with auto-confirm (/iserver/reply/{id})
      - Order status polling
      - Position snapshot
      - Trade history

    Does NOT contain any business logic — pure transport layer.
    All failures raise explicitly; no silent fail-open.
    """

    def __init__(
        self,
        base_url: str = "https://localhost:5000",
        timeout_sec: float = _DEFAULT_TIMEOUT_SEC,
        verify_ssl: bool = False,
    ) -> None:
        self._base = base_url.rstrip("/")
        # Use (connect_timeout, read_timeout) tuple — first request to CPAPI
        # can be slow due to SSL handshake; allow extra time.
        self._timeout = (15.0, max(float(timeout_sec), 30.0))
        self._verify = verify_ssl
        self._session = requests.Session()
        self._session.verify = verify_ssl
        # Disable SSL adapter retries — CPAPI returns errors explicitly
        adapter = requests.adapters.HTTPAdapter(max_retries=0)
        self._session.mount("https://", adapter)
        self._session.mount("http://",  adapter)

        # Tickle keep-alive state
        self._tickle_thread: Optional[threading.Thread] = None
        self._tickle_stop   = threading.Event()
        self._tickle_errors: List[Dict[str, Any]] = []

        # Debug counters
        self.debug_tickle_ok:    int = 0
        self.debug_tickle_fail:  int = 0
        self.debug_auth_ok:      int = 0
        self.debug_auth_fail:    int = 0
        self.debug_order_submit: int = 0
        self.debug_order_reply:  int = 0
        self.debug_poll_calls:   int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _url(self, path: str) -> str:
        return urljoin(self._base + "/", path.lstrip("/"))

    def _get(self, path: str, **kwargs: Any) -> Any:
        url = self._url(path)
        resp = self._session.get(url, timeout=self._timeout, **kwargs)
        resp.raise_for_status()
        if not resp.content:
            return {}
        # CPAPI returns lists for some endpoints (positions, trades, orders)
        return resp.json()

    def _post(self, path: str, payload: Any = None, **kwargs: Any) -> Any:
        url = self._url(path)
        resp = self._session.post(
            url,
            json=payload,
            timeout=self._timeout,
            **kwargs,
        )
        resp.raise_for_status()
        return resp.json() if resp.content else {}

    def _delete(self, path: str, **kwargs: Any) -> Dict[str, Any]:
        url = self._url(path)
        resp = self._session.delete(url, timeout=self._timeout, **kwargs)
        resp.raise_for_status()
        return dict(resp.json()) if resp.content else {}

    # ------------------------------------------------------------------
    # Session / Auth
    # ------------------------------------------------------------------

    def tickle(self) -> Dict[str, Any]:
        """POST /tickle — keep session alive. Returns raw response dict."""
        raw = self._post("/v1/api/tickle")
        self.debug_tickle_ok += 1
        return dict(raw) if isinstance(raw, dict) else {}

    def auth_status(self) -> AuthStatus:
        """GET /iserver/auth/status — returns parsed AuthStatus."""
        raw = self._get("/v1/api/iserver/auth/status")
        d = raw if isinstance(raw, dict) else {}
        status = AuthStatus(
            authenticated=bool(d.get("authenticated", False)),
            competing=bool(d.get("competing", False)),
            connected=bool(d.get("connected", False)),
            message=str(d.get("message", "") or ""),
            raw=d,
        )
        if status.authenticated:
            self.debug_auth_ok += 1
        else:
            self.debug_auth_fail += 1
        return status

    def reauthenticate(self) -> Dict[str, Any]:
        """POST /iserver/reauthenticate — attempt to refresh session."""
        raw = self._post("/v1/api/iserver/reauthenticate")
        return dict(raw) if isinstance(raw, dict) else {}

    def assert_authenticated(self) -> None:
        """
        Raises RuntimeError if session is not authenticated.
        Sends /tickle first to wake the session before checking status —
        first request to CPAPI Gateway can time out if the session is idle.
        """
        try:
            self.tickle()
        except Exception as exc:
            print(f"[CPAPI][WARN] pre-auth tickle failed (non-fatal): {exc}")
        import time as _time
        _time.sleep(0.5)
        status = self.auth_status()
        if not status.authenticated:
            raise RuntimeError(
                f"CPAPI session is not authenticated: {status.message!r}"
            )

    # ------------------------------------------------------------------
    # Tickle keep-alive daemon
    # ------------------------------------------------------------------

    def start_tickle_loop(self) -> None:
        """
        Start a daemon thread that POSTs /tickle every _TICKLE_INTERVAL_SEC.
        Errors are recorded in self._tickle_errors and counted — never swallowed.
        """
        if self._tickle_thread is not None and self._tickle_thread.is_alive():
            return
        self._tickle_stop.clear()
        self._tickle_thread = threading.Thread(
            target=self._tickle_loop_body,
            name="cpapi-tickle",
            daemon=True,
        )
        self._tickle_thread.start()
        print(f"[CPAPI] tickle loop started (interval={_TICKLE_INTERVAL_SEC}s)")

    def stop_tickle_loop(self) -> None:
        self._tickle_stop.set()
        if self._tickle_thread is not None:
            self._tickle_thread.join(timeout=5.0)
        print("[CPAPI] tickle loop stopped")

    def _tickle_loop_body(self) -> None:
        while not self._tickle_stop.is_set():
            try:
                self.tickle()
                print(f"[CPAPI][TICKLE] ok (total_ok={self.debug_tickle_ok})")
            except Exception as exc:
                self.debug_tickle_fail += 1
                err = {
                    "ts": time.time(),
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                self._tickle_errors.append(err)
                print(
                    f"[CPAPI][TICKLE][FAIL] #{self.debug_tickle_fail}: {exc}"
                )
            self._tickle_stop.wait(timeout=_TICKLE_INTERVAL_SEC)

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def submit_order(
        self,
        account_id: str,
        req: CpapiOrderRequest,
    ) -> CpapiOrderResponse:
        """
        POST /iserver/account/{account}/orders

        Handles the two-step confirmation automatically (up to
        _CONFIRM_RETRY_MAX times) if the API returns a question/warning.

        Returns CpapiOrderResponse. Raises on HTTP errors.
        """
        payload = self._build_order_payload(req)
        path = f"/v1/api/iserver/account/{account_id}/orders"
        raw = self._post(path, payload={"orders": [payload]})
        self.debug_order_submit += 1

        # CPAPI may return a list
        if isinstance(raw, list):
            raw = raw[0] if raw else {}

        # CPAPI returns "order_id"/"orderId" after confirmation, but the
        # initial challenge response carries only "id". Use "id" as fallback
        # so _auto_confirm is called and can obtain the real order_id.
        order_id     = str(
            raw.get("order_id", "")
            or raw.get("orderId", "")
            or raw.get("id", "")
            or ""
        )
        local_id     = str(raw.get("local_order_id", "") or "")
        message_text = self._extract_message(raw)
        needs_reply  = bool(raw.get("encrypt_message") or message_text)

        # Auto-confirm warnings
        if needs_reply and order_id:
            order_id = self._auto_confirm(order_id, message_text)

        return CpapiOrderResponse(
            order_id=order_id,
            local_order_id=local_id or None,
            message=message_text or None,
            needs_reply=needs_reply,
            raw=dict(raw) if isinstance(raw, dict) else {},
        )

    def _build_order_payload(self, req: CpapiOrderRequest) -> Dict[str, Any]:
        """
        Build the JSON payload for /iserver/account/{account}/orders.

        CPAPI type requirements (discovered empirically):
          - conid  : int   (str causes 400 "incorrect type")
          - quantity: float or int (both accepted)
          - secType: omitted — Gateway infers from conid
        """
        payload: Dict[str, Any] = {
            "conid":     int(req.conid),   # MUST be int — str → 400
            "cOID":      req.client_tag,
            "orderType": req.order_type,
            "side":      req.side,
            "quantity":  req.quantity,
            "tif":       req.tif,
            "account":   req.account_id,
        }
        if req.price is not None:
            payload["price"] = req.price
        return payload

    def _extract_message(self, raw: Any) -> str:
        if isinstance(raw, dict):
            msg = raw.get("message") or raw.get("text") or ""
            if isinstance(msg, list):
                return " | ".join(str(m) for m in msg)
            return str(msg)
        return ""

    def _auto_confirm(self, order_id: str, original_message: str) -> str:
        """
        POST /iserver/reply/{id} with confirmed=true.
        Retries up to _CONFIRM_RETRY_MAX times.
        Returns the (possibly updated) order_id.
        """
        confirmed_id = order_id
        for attempt in range(1, _CONFIRM_RETRY_MAX + 1):
            try:
                path = f"/v1/api/iserver/reply/{confirmed_id}"
                reply_raw = self._post(path, payload={"confirmed": True})
                self.debug_order_reply += 1
                print(
                    f"[CPAPI][REPLY] attempt={attempt} id={confirmed_id} "
                    f"raw={json.dumps(reply_raw)[:120]}"
                )
                if isinstance(reply_raw, list):
                    reply_raw = reply_raw[0] if reply_raw else {}
                new_id = str(
                    reply_raw.get("order_id", "")
                    or reply_raw.get("orderId", "")
                    or reply_raw.get("id", "")   # Gateway may chain a new challenge id
                    or confirmed_id
                )
                if new_id:
                    confirmed_id = new_id
                # If no more message → confirmation done
                if not self._extract_message(reply_raw):
                    break
            except requests.HTTPError as exc:
                print(f"[CPAPI][REPLY][FAIL] attempt={attempt}: {exc}")
                if attempt == _CONFIRM_RETRY_MAX:
                    raise RuntimeError(
                        f"Order reply failed after {_CONFIRM_RETRY_MAX} attempts "
                        f"for order_id={order_id}: {exc}"
                    ) from exc
        return confirmed_id

    def cancel_order(self, account_id: str, order_id: str) -> Dict[str, Any]:
        """DELETE /iserver/account/{account}/order/{orderId}"""
        path = f"/v1/api/iserver/account/{account_id}/order/{order_id}"
        return self._delete(path)

    # ------------------------------------------------------------------
    # Order status polling
    # ------------------------------------------------------------------

    def get_live_orders(self) -> List[CpapiOrderStatus]:
        """
        GET /iserver/account/orders — returns all live/recent orders.
        """
        raw = self._get("/v1/api/iserver/account/orders")
        orders_raw = raw.get("orders", raw) if isinstance(raw, dict) else raw
        if not isinstance(orders_raw, list):
            orders_raw = []
        self.debug_poll_calls += 1
        return [self._parse_order_status(o) for o in orders_raw]

    def get_trades(self) -> List[CpapiOrderStatus]:
        """
        GET /iserver/account/trades — returns recent fills.
        """
        raw = self._get("/v1/api/iserver/account/trades")
        trades_raw = raw if isinstance(raw, list) else raw.get("trades", [])
        self.debug_poll_calls += 1
        return [self._parse_order_status(t) for t in trades_raw]

    def find_order_status(self, order_id: str) -> Optional[CpapiOrderStatus]:
        """
        Poll live orders and trades for a specific order_id.
        Returns None if not found (explicit — never raises KeyError silently).
        """
        for o in self.get_live_orders():
            if o.order_id == order_id:
                return o
        for t in self.get_trades():
            if t.order_id == order_id:
                return t
        return None

    def _parse_order_status(self, raw: Any) -> CpapiOrderStatus:
        if not isinstance(raw, dict):
            raw = {}
        order_id = str(
            raw.get("orderId", "")
            or raw.get("order_id", "")
            or raw.get("permId", "")
            or ""
        )
        status = str(
            raw.get("status", "")
            or raw.get("orderStatus", "")
            or ""
        )
        filled    = float(raw.get("filledQuantity", 0.0) or raw.get("filled", 0.0) or 0.0)
        total     = float(raw.get("totalSize", 0.0) or raw.get("quantity", 0.0) or 0.0)
        remaining = max(0.0, total - filled)
        avg_px    = float(raw.get("avgPrice", 0.0) or raw.get("price", 0.0) or 0.0)
        last_px   = float(raw.get("lastExecutionPrice", avg_px) or avg_px)
        return CpapiOrderStatus(
            order_id=order_id,
            status=status,
            filled_qty=filled,
            remaining_qty=remaining,
            avg_price=avg_px,
            last_price=last_px,
            raw=dict(raw),
        )

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def get_positions(self, account_id: str) -> List[CpapiPosition]:
        """
        GET /portfolio/{account}/positions/0
        """
        path = f"/v1/api/portfolio/{account_id}/positions/0"
        raw = self._get(path)
        rows = raw if isinstance(raw, list) else []
        result: List[CpapiPosition] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            result.append(CpapiPosition(
                conid=str(row.get("conid", "") or ""),
                symbol=str(row.get("ticker", "") or row.get("symbol", "") or ""),
                position=float(row.get("position", 0.0) or 0.0),
                avg_cost=float(row.get("avgCost", 0.0) or 0.0),
                market_value=float(row.get("mktValue", 0.0) or 0.0),
                account_id=account_id,
                raw=dict(row),
            ))
        return result

    # ------------------------------------------------------------------
    # Debug summary
    # ------------------------------------------------------------------

    def debug_summary(self) -> Dict[str, Any]:
        return {
            "tickle_ok":    self.debug_tickle_ok,
            "tickle_fail":  self.debug_tickle_fail,
            "tickle_errors_count": len(self._tickle_errors),
            "auth_ok":      self.debug_auth_ok,
            "auth_fail":    self.debug_auth_fail,
            "order_submit": self.debug_order_submit,
            "order_reply":  self.debug_order_reply,
            "poll_calls":   self.debug_poll_calls,
        }
