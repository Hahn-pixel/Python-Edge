from __future__ import annotations

import time
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from python_edge.broker.cpapi_client import CpapiClient
from python_edge.broker.cpapi_models import (
    CpapiOrderRequest,
    CpapiOrderStatus,
    ExecState,
    ExecutionIntent,
    FillResult,
    OrderSide,
)
from python_edge.broker.cpapi_pricing import (
    PriceGuardViolation,
    check_parent_guard,
    frac_limit_price,
    midprice_price,
    split_qty,
)

# ---------------------------------------------------------------------------
# Tunable defaults (all overridable by caller via engine_kwargs)
# ---------------------------------------------------------------------------

_WHOLE_FILL_POLL_INTERVAL_SEC: float = 1.0
# TD-1: збільшено default з 30 → 60 щоб великі ордери (1000+ shares)
# мали достатньо часу на fill через MIDPRICE.
# Для cleanup pipeline env6 передає 90.0 явно (TD-3).
_WHOLE_FILL_TIMEOUT_SEC: float       = 60.0
_FRAC_FILL_POLL_INTERVAL_SEC: float  = 1.0
_FRAC_FILL_TIMEOUT_SEC: float        = 20.0
_DEFAULT_TIF: str                    = "DAY"
_FRAC_SLIPPAGE_BPS: float            = 5.0
_DEFAULT_TICK: float                 = 0.01
_FILLED_STATUSES = frozenset({"Filled", "FILLED", "filled"})
_WORKING_STATUSES = frozenset({
    "Submitted", "SUBMITTED", "submitted",
    "PreSubmitted", "PRESUBMITTED", "presubmitted",
    "PendingSubmit", "PENDINGSUBMIT",
})
_CANCELLED_STATUSES = frozenset({"Cancelled", "CANCELLED", "cancelled"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _transition(intent: ExecutionIntent, new_state: ExecState, detail: str = "") -> None:
    """Record a state transition in the audit trail. Never silent."""
    prev = intent.state
    intent.state = new_state
    intent.transitions.append({
        "from": prev.value,
        "to":   new_state.value,
        "ts":   _utc_now_iso(),
        "detail": detail,
    })
    print(
        f"[SM][{intent.symbol}] {prev.value} → {new_state.value}"
        + (f" | {detail}" if detail else "")
    )


def _record_error(intent: ExecutionIntent, exc_or_msg: object, stage: str = "") -> None:
    """Append an error to the audit trail. Always explicit."""
    msg = str(exc_or_msg)
    intent.errors.append({
        "ts":    _utc_now_iso(),
        "stage": stage,
        "error": msg,
        "traceback": traceback.format_exc() if isinstance(exc_or_msg, Exception) else "",
    })
    print(f"[SM][{intent.symbol}][ERR] stage={stage} {msg}")


def _is_filled(status: CpapiOrderStatus) -> bool:
    return (
        status.status in _FILLED_STATUSES
        or (status.filled_qty > 0.0 and status.remaining_qty <= 1e-9)
    )


def _is_partial(status: CpapiOrderStatus) -> bool:
    return status.filled_qty > 0.0 and status.remaining_qty > 1e-9


def _is_working(status: CpapiOrderStatus) -> bool:
    return status.status in _WORKING_STATUSES


def _is_cancelled(status: CpapiOrderStatus) -> bool:
    return status.status in _CANCELLED_STATUSES


# ---------------------------------------------------------------------------
# State machine executor
# ---------------------------------------------------------------------------

class CpapiExecutionEngine:
    """
    Drives a single ExecutionIntent through the full state machine:

        NEW → PRECHECK → SPLIT
            → WHOLE_SUBMIT → WHOLE_WORKING → WHOLE_FILLED | WHOLE_PARTIAL | WHOLE_TIMEOUT
            → FRAC_SUBMIT  → FRAC_WORKING
            → DONE | FAILED

    Each state transition is explicit, logged, and counted.
    No silent fail-open: any bypass is recorded in debug counters.

    TD-1 note: reply chain failures (id → reply → new id → 400) are surfaced
    in submit_order inside CpapiClient. If WHOLE_SUBMIT stays in WHOLE_SUBMIT
    state (empty order_id), the engine logs the full raw response for diagnosis
    and advances to WHOLE_TIMEOUT with an explicit debug counter increment.
    """

    def __init__(
        self,
        client: CpapiClient,
        whole_timeout_sec: float = _WHOLE_FILL_TIMEOUT_SEC,
        frac_timeout_sec: float  = _FRAC_FILL_TIMEOUT_SEC,
        tif: str                 = _DEFAULT_TIF,
        frac_slippage_bps: float = _FRAC_SLIPPAGE_BPS,
        min_tick: float          = _DEFAULT_TICK,
    ) -> None:
        self._client            = client
        self._whole_timeout     = whole_timeout_sec
        self._frac_timeout      = frac_timeout_sec
        self._tif               = tif
        self._frac_slippage_bps = frac_slippage_bps
        self._min_tick          = min_tick

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def execute(self, intent: ExecutionIntent, reference_price: float) -> FillResult:
        """
        Run the intent through the full state machine.

        *reference_price* is the last known market price used for:
          - parent guard validation
          - frac leg limit price computation

        Always returns a FillResult — even on FAILED (with zeros).
        """
        _transition(intent, ExecState.PRECHECK, "starting execution")
        self._run_precheck(intent, reference_price)

        if intent.state is ExecState.FAILED:
            return self._build_fill_result(intent)

        _transition(intent, ExecState.SPLIT, "splitting qty")
        self._run_split(intent)

        if intent.state is ExecState.FAILED:
            return self._build_fill_result(intent)

        # ── Whole leg ──────────────────────────────────────────────────
        if intent.whole_qty >= 1.0:
            _transition(intent, ExecState.WHOLE_SUBMIT, "submitting whole leg")
            self._run_whole_submit(intent, reference_price)

            if intent.state is ExecState.WHOLE_SUBMIT:
                # submit failed — reply chain exhausted or empty order_id
                # TD-1: явне логування причини провалу submit
                _transition(intent, ExecState.WHOLE_TIMEOUT,
                            "whole submit failed (reply chain exhausted or empty order_id); skipping to frac")
                intent.debug_whole_timeout += 1

            elif intent.state is ExecState.WHOLE_WORKING:
                self._run_whole_wait(intent)

        # ── Frac leg (always attempted unless already FAILED) ──────────
        if intent.state not in {ExecState.FAILED}:
            if intent.frac_qty > 1e-9:
                _transition(intent, ExecState.FRAC_SUBMIT, "submitting frac leg")
                self._run_frac_submit(intent, reference_price)

                if intent.state is ExecState.FRAC_SUBMIT:
                    _transition(intent, ExecState.FAILED,
                                "frac submit failed")
                    intent.debug_failed += 1
                elif intent.state is ExecState.FRAC_WORKING:
                    self._run_frac_wait(intent)

            # If no frac leg needed, finalize here
            if intent.state not in {ExecState.FRAC_WORKING, ExecState.FAILED}:
                _transition(intent, ExecState.DONE, "no frac leg required")

        if intent.state not in {ExecState.DONE, ExecState.FAILED}:
            # Safety net — should not happen; mark explicitly
            _transition(intent, ExecState.DONE, "finalized after frac")

        return self._build_fill_result(intent)

    # ------------------------------------------------------------------
    # PRECHECK
    # ------------------------------------------------------------------

    def _run_precheck(self, intent: ExecutionIntent, reference_price: float) -> None:
        """
        Validate intent fields and parent price guard before any order is sent.
        Transitions to FAILED (explicit) if anything is wrong.
        """
        issues: List[str] = []

        if not intent.symbol:
            issues.append("symbol is empty")
        if not intent.conid:
            issues.append("conid is empty")
        if not intent.account_id:
            issues.append("account_id is empty")
        if intent.target_qty <= 0.0:
            issues.append(f"target_qty={intent.target_qty} <= 0")
        if reference_price <= 0.0:
            issues.append(f"reference_price={reference_price} <= 0")

        if issues:
            detail = "; ".join(issues)
            _record_error(intent, detail, stage="PRECHECK")
            _transition(intent, ExecState.FAILED, detail)
            intent.debug_precheck_fail += 1
            return

        try:
            check_parent_guard(intent, reference_price, label="PRECHECK")
        except PriceGuardViolation as exc:
            _record_error(intent, exc, stage="PRECHECK_GUARD")
            _transition(intent, ExecState.FAILED, str(exc))
            intent.debug_precheck_fail += 1
            return

        intent.debug_precheck_ok += 1
        # State stays PRECHECK; caller advances to SPLIT

    # ------------------------------------------------------------------
    # SPLIT
    # ------------------------------------------------------------------

    def _run_split(self, intent: ExecutionIntent) -> None:
        whole, frac = split_qty(intent.target_qty)
        intent.whole_qty = whole
        intent.frac_qty  = frac
        intent.debug_split_ok += 1
        # State stays SPLIT; caller advances

    # ------------------------------------------------------------------
    # WHOLE leg
    # ------------------------------------------------------------------

    def _run_whole_submit(self, intent: ExecutionIntent, reference_price: float) -> None:
        """
        Submit MIDPRICE order for whole_qty. Sets WHOLE_WORKING on success.

        TD-1: при провалі submit_order (empty order_id або reply chain failure)
        логуємо повний raw response для діагностики reply chain.
        """
        try:
            # midprice_price validates guard; returns None (no explicit px)
            midprice_price(intent, self._min_tick, reference_price)

            req = CpapiOrderRequest(
                conid      = intent.conid,
                side       = intent.side.value,
                quantity   = intent.whole_qty,
                order_type = "MIDPRICE",
                price      = None,
                tif        = self._tif,
                account_id = intent.account_id,
                client_tag = f"{intent.client_tag}-W",
            )
            resp = self._client.submit_order(intent.account_id, req)

            if not resp.order_id:
                # TD-1: логуємо raw response щоб діагностувати reply chain
                raw_repr = repr(resp.raw)[:500]
                raise RuntimeError(
                    f"submit_order returned empty order_id. "
                    f"Possible reply chain failure (id→reply→new_id→400). "
                    f"raw={raw_repr}"
                )

            intent.whole_order_id = resp.order_id
            intent.debug_whole_submitted += 1
            _transition(intent, ExecState.WHOLE_WORKING,
                        f"whole order_id={resp.order_id}")

        except PriceGuardViolation as exc:
            _record_error(intent, exc, stage="WHOLE_SUBMIT_GUARD")
            intent.debug_guard_rejected += 1
            # Stay in WHOLE_SUBMIT → caller will detect and advance to WHOLE_TIMEOUT

        except Exception as exc:
            # TD-1: явний лог з повним traceback для reply chain діагностики
            _record_error(intent, exc, stage="WHOLE_SUBMIT")
            print(
                f"[SM][{intent.symbol}][WHOLE_SUBMIT][TD-1_DIAG] "
                f"whole_qty={intent.whole_qty:.0f} "
                f"conid={intent.conid} "
                f"client_tag={intent.client_tag}-W "
                f"timeout_configured={self._whole_timeout:.0f}s"
            )
            # Stay in WHOLE_SUBMIT → caller advances to WHOLE_TIMEOUT

    def _run_whole_wait(self, intent: ExecutionIntent) -> None:
        """
        Poll until whole leg fills, partially fills, times out, or is cancelled.
        TD-1: logs elapsed time at each poll tick to help diagnose slow fills vs timeouts.
        """
        deadline = time.monotonic() + self._whole_timeout
        order_id = intent.whole_order_id
        start_ts = time.monotonic()

        while time.monotonic() < deadline:
            time.sleep(_WHOLE_FILL_POLL_INTERVAL_SEC)
            elapsed = time.monotonic() - start_ts
            try:
                status = self._client.find_order_status(order_id)
            except Exception as exc:
                _record_error(intent, exc, stage="WHOLE_POLL")
                continue

            if status is None:
                # Not visible yet — keep polling
                print(
                    f"[SM][{intent.symbol}][WHOLE_POLL] "
                    f"order_id={order_id} not found yet "
                    f"elapsed={elapsed:.1f}s"
                )
                continue

            print(
                f"[SM][{intent.symbol}][WHOLE_POLL] "
                f"status={status.status} filled={status.filled_qty:.6f} "
                f"remaining={status.remaining_qty:.6f} avg_px={status.avg_price:.4f} "
                f"elapsed={elapsed:.1f}s"
            )

            if _is_filled(status):
                intent.whole_filled_qty  = status.filled_qty
                intent.whole_avg_price   = status.avg_price
                intent.debug_whole_filled += 1
                _transition(intent, ExecState.WHOLE_FILLED,
                            f"filled qty={status.filled_qty:.6f} avg_px={status.avg_price:.4f} elapsed={elapsed:.1f}s")
                return

            if _is_partial(status):
                intent.whole_filled_qty  = status.filled_qty
                intent.whole_avg_price   = status.avg_price
                intent.debug_whole_partial += 1
                _transition(intent, ExecState.WHOLE_PARTIAL,
                            f"partial qty={status.filled_qty:.6f} remaining={status.remaining_qty:.6f} elapsed={elapsed:.1f}s")
                return

            if _is_cancelled(status):
                intent.whole_filled_qty = status.filled_qty
                intent.whole_avg_price  = status.avg_price
                _transition(intent, ExecState.WHOLE_TIMEOUT,
                            f"whole order cancelled by broker qty_filled={status.filled_qty:.6f} elapsed={elapsed:.1f}s")
                intent.debug_whole_timeout += 1
                return

        # Timeout reached — explicit log with configured timeout for diagnosis
        elapsed_total = time.monotonic() - start_ts
        intent.debug_whole_timeout += 1
        partial_status = self._client.find_order_status(order_id)
        filled = partial_status.filled_qty if partial_status else 0.0
        avg_px = partial_status.avg_price  if partial_status else 0.0
        intent.whole_filled_qty = filled
        intent.whole_avg_price  = avg_px
        _transition(intent, ExecState.WHOLE_TIMEOUT,
                    f"timeout after {self._whole_timeout:.0f}s "
                    f"elapsed={elapsed_total:.1f}s "
                    f"filled_so_far={filled:.6f} "
                    f"order_id={order_id}")

    # ------------------------------------------------------------------
    # FRAC leg
    # ------------------------------------------------------------------

    def _run_frac_submit(self, intent: ExecutionIntent, reference_price: float) -> None:
        """Submit limit order for frac_qty. Sets FRAC_WORKING on success."""
        try:
            limit_px = frac_limit_price(
                intent,
                self._min_tick,
                reference_price,
                self._frac_slippage_bps,
            )
            req = CpapiOrderRequest(
                conid      = intent.conid,
                side       = intent.side.value,
                quantity   = intent.frac_qty,
                order_type = "LMT",
                price      = limit_px,
                tif        = self._tif,
                account_id = intent.account_id,
                client_tag = f"{intent.client_tag}-F",
            )
            resp = self._client.submit_order(intent.account_id, req)

            if not resp.order_id:
                raw_repr = repr(resp.raw)[:500]
                raise RuntimeError(
                    f"frac submit_order returned empty order_id. raw={raw_repr}"
                )

            intent.frac_order_id = resp.order_id
            intent.debug_frac_submitted += 1
            _transition(intent, ExecState.FRAC_WORKING,
                        f"frac order_id={resp.order_id} limit_px={limit_px:.4f}")

        except PriceGuardViolation as exc:
            _record_error(intent, exc, stage="FRAC_SUBMIT_GUARD")
            intent.debug_guard_rejected += 1
            # Stay in FRAC_SUBMIT → caller detects and transitions to FAILED

        except Exception as exc:
            _record_error(intent, exc, stage="FRAC_SUBMIT")
            # Stay in FRAC_SUBMIT → caller → FAILED

    def _run_frac_wait(self, intent: ExecutionIntent) -> None:
        """Poll frac leg. Transitions to DONE regardless of fill outcome."""
        deadline = time.monotonic() + self._frac_timeout
        order_id = intent.frac_order_id

        while time.monotonic() < deadline:
            time.sleep(_FRAC_FILL_POLL_INTERVAL_SEC)
            try:
                status = self._client.find_order_status(order_id)
            except Exception as exc:
                _record_error(intent, exc, stage="FRAC_POLL")
                continue

            if status is None:
                print(f"[SM][{intent.symbol}][FRAC_POLL] order_id={order_id} not found yet")
                continue

            print(
                f"[SM][{intent.symbol}][FRAC_POLL] "
                f"status={status.status} filled={status.filled_qty:.8f} "
                f"remaining={status.remaining_qty:.8f} avg_px={status.avg_price:.4f}"
            )

            if _is_filled(status) or _is_cancelled(status):
                intent.frac_filled_qty = status.filled_qty
                intent.frac_avg_price  = status.avg_price
                intent.debug_frac_filled += 1
                _transition(intent, ExecState.DONE,
                            f"frac done status={status.status} "
                            f"filled={status.filled_qty:.8f}")
                return

            if _is_partial(status):
                intent.frac_filled_qty = status.filled_qty
                intent.frac_avg_price  = status.avg_price
                # partial frac → still consider execution done for this cycle
                intent.debug_frac_filled += 1
                _transition(intent, ExecState.DONE,
                            f"frac partial filled={status.filled_qty:.8f} "
                            f"remaining={status.remaining_qty:.8f}")
                return

        # Timeout — record partial fill and finalize
        final_status = self._client.find_order_status(order_id)
        if final_status:
            intent.frac_filled_qty = final_status.filled_qty
            intent.frac_avg_price  = final_status.avg_price
        _transition(intent, ExecState.DONE,
                    f"frac timeout after {self._frac_timeout:.0f}s "
                    f"filled={intent.frac_filled_qty:.8f}")

    # ------------------------------------------------------------------
    # Result builder
    # ------------------------------------------------------------------

    def _build_fill_result(self, intent: ExecutionIntent) -> FillResult:
        total_filled = round(
            intent.whole_filled_qty + intent.frac_filled_qty, 8
        )

        # Weighted average price across both legs
        wsum = (
            intent.whole_filled_qty * intent.whole_avg_price
            + intent.frac_filled_qty * intent.frac_avg_price
        )
        avg_px = (wsum / total_filled) if total_filled > 1e-12 else 0.0

        return FillResult(
            symbol          = intent.symbol,
            side            = intent.side.value,
            total_filled    = total_filled,
            avg_price       = round(avg_px, 6),
            whole_filled    = intent.whole_filled_qty,
            whole_avg_price = intent.whole_avg_price,
            frac_filled     = intent.frac_filled_qty,
            frac_avg_price  = intent.frac_avg_price,
            final_state     = intent.state,
            client_tag      = intent.client_tag,
        )


# ---------------------------------------------------------------------------
# Batch runner (processes a list of intents sequentially)
# ---------------------------------------------------------------------------

def run_batch(
    client: CpapiClient,
    intents: List[ExecutionIntent],
    reference_prices: Dict[str, float],
    engine_kwargs: Optional[Dict[str, Any]] = None,
) -> List[FillResult]:
    """
    Execute a list of intents sequentially.

    *reference_prices* maps symbol → last known price.
    Missing symbols get price=0.0 which will fail PRECHECK with an explicit error.

    Returns a list of FillResult in the same order as *intents*.
    """
    engine = CpapiExecutionEngine(client, **(engine_kwargs or {}))
    results: List[FillResult] = []

    for intent in intents:
        ref_price = float(reference_prices.get(intent.symbol, 0.0))
        print(
            f"[BATCH] symbol={intent.symbol} side={intent.side.value} "
            f"qty={intent.target_qty:.8f} ref_price={ref_price:.4f} "
            f"conid={intent.conid} "
            f"whole_timeout={engine._whole_timeout:.0f}s "
            f"frac_timeout={engine._frac_timeout:.0f}s"
        )
        result = engine.execute(intent, ref_price)
        results.append(result)
        print(
            f"[BATCH][RESULT] symbol={result.symbol} "
            f"state={result.final_state.value} "
            f"total_filled={result.total_filled:.8f} "
            f"avg_price={result.avg_price:.4f}"
        )

    # Summary counters
    done_count   = sum(1 for r in results if r.final_state is ExecState.DONE)
    failed_count = sum(1 for r in results if r.final_state is ExecState.FAILED)
    print(
        f"[BATCH][SUMMARY] total={len(results)} done={done_count} failed={failed_count}"
    )
    return results
