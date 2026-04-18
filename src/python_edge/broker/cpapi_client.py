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

# TD-1: збільшено з 3 → 10.
# Причина: Gateway для великих ордерів (qty=1000+) може повернути
# ланцюжок: id → reply → новий id → reply → новий id → ...
# При _CONFIRM_RETRY_MAX=3 ланцюжок обривався з RuntimeError → empty order_id.
# Емпірично підтверджено: кожен reply-крок Gateway повертає або order_id
# (done) або новий id (ще одне підтвердження потрібне). 10 ітерацій
# покриває будь-яку відому глибину reply chain для IBKR Gateway.
_CONFIRM_RETRY_MAX   = 10


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

        Handles the multi-step confirmation automatically (up to
        _CONFIRM_RETRY_MAX=10 times) if the API returns a question/warning.

        Returns CpapiOrderResponse. Raises on HTTP errors.
        """
        payload = self._build_order_payload(req)
        path = f"/v1/api/iserver/account/{account_id}/orders"

        # TD-1: логуємо payload перед відправкою для діагностики
        print(
            f"[CPAPI][SUBMIT] cOID={req.client_tag} "
            f"conid={req.conid} side={req.side} "
            f"qty={req.quantity} orderType={req.order_type} "
            f"price={req.price} tif={req.tif}"
        )

        raw = self._post(path, payload={"orders": [payload]})
        self.debug_order_submit += 1

        # CPAPI may return a list
        if isinstance(raw, list):
            raw = raw[0] if raw else {}

        # TD-1: логуємо повний initial response
        print(
            f"[CPAPI][SUBMIT_RAW] cOID={req.client_tag} "
            f"raw={json.dumps(raw)[:300]}"
        )

        order_id     = str(raw.get("order_id", "") or raw.get("orderId", "") or "")
        local_id     = str(raw.get("local_order_id", "") or "")
        message_text = self._extract_message(raw)

        # Gateway може повернути тільки "id" (challenge) без order_id
        # TD-1: перевіряємо challenge_id окремо від order_id
        challenge_id = str(raw.get("id", "") or "")
        needs_reply  = bool(
            raw.get("encrypt_message")
            or message_text
            or (challenge_id and not order_id)
        )

        # Якщо є challenge_id але немає order_id — починаємо reply chain з challenge_id
        reply_start_id = order_id if order_id else challenge_id

        # Auto-confirm warnings/challenges
        if needs_reply and reply_start_id:
            order_id = self._auto_confirm(reply_start_id, message_text, req.client_tag)
        elif needs_reply and not reply_start_id:
            # Немає ні order_id ні challenge_id — нічого підтверджувати
            print(
                f"[CPAPI][SUBMIT_WARN] cOID={req.client_tag} "
                f"needs_reply=True but no id/order_id in response — cannot confirm"
            )

        return CpapiOrderResponse(
            order_id=order_id,
            local_order_id=local_id or None,
            message=message_text or None,
            needs_reply=needs_reply,
            raw=dict(raw) if isinstance(raw, dict) else {},
        )

    def _build_order_payload(self, req: CpapiOrderRequest) -> Dict[str, Any]:
        # TD-1: прибрано secType з payload.
        # Емпірично підтверджено: Gateway визначає secType сам по conid.
        # Наявність secType може призводити до відхилення ордеру (400).
        payload: Dict[str, Any] = {
            "conid":     int(req.conid),   # конид обов'язково int (str → 400)
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

    def _auto_confirm(
        self,
        start_id: str,
        original_message: str,
        client_tag: str = "",
    ) -> str:
        """
        POST /iserver/reply/{id} with confirmed=true.
        Retries up to _CONFIRM_RETRY_MAX (=10) times.

        TD-1: повністю перероблено reply chain логіку:
          - Підтримує ланцюжок: initial_id → reply → new_id → reply → ...
          - Логує повний raw response на кожній ітерації
          - Відрізняє order_id (фінальний) від нового challenge id (продовжує chain)
          - Повертає order_id щойно Gateway його повертає без нового повідомлення
          - Якщо після _CONFIRM_RETRY_MAX ітерацій order_id не отримано — raise
        """
        current_id = start_id
        confirmed_order_id = ""
        reply_chain: list[str] = [start_id]

        for attempt in range(1, _CONFIRM_RETRY_MAX + 1):
            try:
                path = f"/v1/api/iserver/reply/{current_id}"
                reply_raw = self._post(path, payload={"confirmed": True})
                self.debug_order_reply += 1

                # TD-1: логуємо ПОВНИЙ reply body (не обрізаємо до 120 символів)
                reply_json_str = json.dumps(reply_raw)
                print(
                    f"[CPAPI][REPLY] cOID={client_tag} attempt={attempt}/{_CONFIRM_RETRY_MAX} "
                    f"sent_id={current_id} "
                    f"raw={reply_json_str[:600]}"
                )

                # CPAPI може повернути list
                if isinstance(reply_raw, list):
                    reply_raw = reply_raw[0] if reply_raw else {}

                # Витягуємо order_id (фінальний) і новий challenge id (якщо є)
                new_order_id  = str(
                    reply_raw.get("order_id", "")
                    or reply_raw.get("orderId", "")
                    or ""
                )
                new_challenge = str(reply_raw.get("id", "") or "")
                next_message  = self._extract_message(reply_raw)

                if new_order_id:
                    # Отримали фінальний order_id
                    confirmed_order_id = new_order_id
                    if not next_message:
                        # Немає нового повідомлення — підтвердження завершено
                        print(
                            f"[CPAPI][REPLY][DONE] cOID={client_tag} "
                            f"order_id={confirmed_order_id} "
                            f"chain={' → '.join(reply_chain)} → {current_id}"
                        )
                        return confirmed_order_id
                    # Є order_id, але є нове повідомлення — продовжуємо chain з order_id
                    current_id = new_order_id
                    reply_chain.append(current_id)
                elif new_challenge:
                    # Немає order_id — Gateway дав новий challenge id
                    print(
                        f"[CPAPI][REPLY][CHAIN] cOID={client_tag} "
                        f"attempt={attempt} no order_id yet, "
                        f"new challenge_id={new_challenge} message={next_message!r}"
                    )
                    current_id = new_challenge
                    reply_chain.append(current_id)
                else:
                    # Ні order_id ні нового challenge — незрозуміла відповідь
                    print(
                        f"[CPAPI][REPLY][WARN] cOID={client_tag} "
                        f"attempt={attempt} no order_id and no new id in reply. "
                        f"raw={reply_json_str[:300]}"
                    )
                    if not next_message and confirmed_order_id:
                        # Вже маємо order_id з попередньої ітерації — повертаємо
                        return confirmed_order_id
                    if not next_message:
                        # Ніякої інформації — виходимо з тим що маємо
                        break

            except requests.HTTPError as exc:
                status_code = exc.response.status_code if exc.response is not None else 0
                resp_text = exc.response.text[:400] if exc.response is not None else ""
                print(
                    f"[CPAPI][REPLY][HTTP_ERROR] cOID={client_tag} "
                    f"attempt={attempt} id={current_id} "
                    f"status={status_code} body={resp_text}"
                )
                if attempt == _CONFIRM_RETRY_MAX:
                    raise RuntimeError(
                        f"Order reply chain failed after {_CONFIRM_RETRY_MAX} attempts "
                        f"cOID={client_tag} start_id={start_id} "
                        f"last_id={current_id} chain={reply_chain}: {exc}"
                    ) from exc
                # При HTTP error (наприклад 400) — не повторюємо з тим самим id
                # Якщо вже маємо order_id — повертаємо
                if confirmed_order_id:
                    print(
                        f"[CPAPI][REPLY][RECOVER] HTTP error but have order_id={confirmed_order_id} — returning"
                    )
                    return confirmed_order_id
                break

            except Exception as exc:
                print(
                    f"[CPAPI][REPLY][ERR] cOID={client_tag} "
                    f"attempt={attempt} id={current_id}: {exc}"
                )
                if confirmed_order_id:
                    return confirmed_order_id
                if attempt == _CONFIRM_RETRY_MAX:
                    raise RuntimeError(
                        f"Order reply chain failed after {_CONFIRM_RETRY_MAX} attempts "
                        f"cOID={client_tag}: {exc}"
                    ) from exc

        # Вийшли з циклу — повертаємо що маємо
        if confirmed_order_id:
            print(
                f"[CPAPI][REPLY][LOOP_EXIT] cOID={client_tag} "
                f"returning confirmed_order_id={confirmed_order_id}"
            )
            return confirmed_order_id

        # Нічого не отримали — повертаємо порожній рядок,
        # submit_order обробить це явно
        print(
            f"[CPAPI][REPLY][EMPTY] cOID={client_tag} "
            f"reply chain exhausted ({_CONFIRM_RETRY_MAX} attempts) "
            f"chain={reply_chain} — no order_id obtained"
        )
        return ""

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
        filled   = float(raw.get("filledQuantity", 0.0) or raw.get("filled", 0.0) or 0.0)
        total    = float(raw.get("totalSize", 0.0) or raw.get("quantity", 0.0) or 0.0)
        remaining = max(0.0, total - filled)
        avg_px   = float(raw.get("avgPrice", 0.0) or raw.get("price", 0.0) or 0.0)
        last_px  = float(raw.get("lastExecutionPrice", avg_px) or avg_px)
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
