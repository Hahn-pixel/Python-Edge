"""
flatten_paper_account.py
========================
Повний flatten paper account через CPAPI:
  1. Скасовує всі working ордери
  2. Відправляє MKT ордер протилежної сторони для кожної позиції
  3. Чекає підтвердження fills

Запуск:
    python flatten_paper_account.py

Env (опціональні, є дефолти):
    CPAPI_BASE_URL      — default: https://localhost:5000
    BROKER_ACCOUNT_ID   — default: DUP561175
    FLATTEN_DRY_RUN     — 1 = тільки показати що буде зроблено, не відправляти ордери
    FLATTEN_WAIT_SEC    — секунд чекати fills після відправки (default: 45)
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

ROOT    = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
for _p in [ROOT, SRC_DIR]:
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

from python_edge.broker.cpapi_client import CpapiClient
from python_edge.broker.cpapi_models import CpapiOrderRequest

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CPAPI_BASE_URL  = str(os.getenv("CPAPI_BASE_URL",    "https://localhost:5000")).strip()
ACCOUNT_ID      = str(os.getenv("BROKER_ACCOUNT_ID", "DUP561175")).strip()
DRY_RUN         = str(os.getenv("FLATTEN_DRY_RUN",   "0")).lower() not in {"0","false","no","off"}
WAIT_SEC        = float(os.getenv("FLATTEN_WAIT_SEC", "45"))

_WORKING = frozenset({"Submitted","PreSubmitted","PendingSubmit","SUBMITTED","PRESUBMITTED"})

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _confirm_reply(client: CpapiClient, challenge_id: str) -> str:
    """POST /iserver/reply/{id} до отримання order_id або вичерпання спроб."""
    current_id = challenge_id
    for attempt in range(1, 5):
        try:
            raw = client._post(f"/v1/api/iserver/reply/{current_id}", payload={"confirmed": True})
            if isinstance(raw, list):
                raw = raw[0] if raw else {}
            print(f"  [REPLY] attempt={attempt} raw={str(raw)[:120]}")
            order_id = str(raw.get("order_id","") or raw.get("orderId","") or "")
            next_id  = str(raw.get("id","") or "")
            if order_id:
                return order_id
            if next_id and next_id != current_id:
                current_id = next_id
                continue
            # немає ні order_id ні нового id — виходимо
            break
        except Exception as exc:
            print(f"  [REPLY][FAIL] attempt={attempt}: {exc}")
            break
    return ""


def _submit_market(
    client: CpapiClient,
    conid: str,
    symbol: str,
    side: str,
    qty: float,
    tag: str,
) -> str:
    """Відправляє MKT ордер. Повертає order_id або '' при помилці."""
    req = CpapiOrderRequest(
        conid      = conid,
        side       = side,
        quantity   = qty,
        order_type = "MKT",
        price      = None,
        tif        = "DAY",
        account_id = ACCOUNT_ID,
        client_tag = tag,
    )
    try:
        resp = client.submit_order(ACCOUNT_ID, req)
        if resp.order_id:
            print(f"  [ORDER] {symbol} {side} {qty} → order_id={resp.order_id}")
            return resp.order_id
        # submit_order повернув порожній order_id — може бути challenge
        raw_id = resp.raw.get("id","") or resp.raw.get("order_id","") or ""
        if raw_id:
            print(f"  [ORDER][CHALLENGE] {symbol} challenge_id={raw_id}")
            oid = _confirm_reply(client, str(raw_id))
            if oid:
                print(f"  [ORDER] {symbol} confirmed order_id={oid}")
                return oid
        print(f"  [ORDER][FAIL] {symbol} no order_id: {resp.raw}")
        return ""
    except Exception as exc:
        print(f"  [ORDER][ERR] {symbol}: {exc}")
        return ""

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print(f"[FLATTEN] account={ACCOUNT_ID} dry_run={DRY_RUN} wait_sec={WAIT_SEC}")
    print(f"[FLATTEN] cpapi_url={CPAPI_BASE_URL}")

    client = CpapiClient(CPAPI_BASE_URL, verify_ssl=False)
    client.assert_authenticated()
    client.start_tickle_loop()

    try:
        # ── Крок 1: скасувати всі working ордери ──────────────────────
        print("\n[STEP 1] Cancelling working orders...")
        try:
            live_orders = client.get_live_orders()
        except Exception as exc:
            print(f"  [WARN] get_live_orders failed: {exc}")
            live_orders = []

        working = [o for o in live_orders if o.status in _WORKING]
        print(f"  working orders found: {len(working)}")

        for o in working:
            print(f"  [CANCEL] order_id={o.order_id} status={o.status} remaining={o.remaining_qty}")
            if not DRY_RUN:
                try:
                    result = client.cancel_order(ACCOUNT_ID, o.order_id)
                    print(f"    → {result}")
                except Exception as exc:
                    print(f"    → CANCEL FAILED: {exc}")

        if working and not DRY_RUN:
            print("  waiting 3s for cancels to settle...")
            time.sleep(3.0)

        # ── Крок 2: отримати позиції ───────────────────────────────────
        print("\n[STEP 2] Fetching positions...")
        try:
            positions = client.get_positions(ACCOUNT_ID)
        except Exception as exc:
            print(f"  [ERR] get_positions failed: {exc}")
            return 1

        nonzero = [p for p in positions if abs(p.position) > 1e-6]
        print(f"  positions found: {len(positions)}  non-zero: {len(nonzero)}")

        if not nonzero:
            print("\n[FLATTEN] No positions to close. Account is already flat.")
            return 0

        print("\n  Positions to close:")
        for p in nonzero:
            side = "SELL" if p.position > 0 else "BUY"
            qty  = abs(p.position)
            print(f"    {p.symbol:8s}  qty={p.position:>10.4f}  avg_cost={p.avg_cost:.4f}"
                  f"  → {side} {qty:.4f}")

        if DRY_RUN:
            print("\n[FLATTEN] DRY RUN — no orders sent.")
            return 0

        # ── Крок 3: відправити MKT ордери ─────────────────────────────
        print("\n[STEP 3] Sending flatten orders (MKT)...")
        submitted: list[str] = []

        from datetime import date as _date
        today = _date.today().strftime("%Y%m%d")

        for p in nonzero:
            side = "SELL" if p.position > 0 else "BUY"
            qty  = abs(p.position)
            tag  = f"flat-{today}-{p.symbol[:8].lower()}"
            oid  = _submit_market(client, p.conid, p.symbol, side, qty, tag)
            if oid:
                submitted.append(oid)
            time.sleep(0.3)

        print(f"\n  submitted: {len(submitted)}/{len(nonzero)}")

        # ── Крок 4: чекати fills ───────────────────────────────────────
        print(f"\n[STEP 4] Waiting {WAIT_SEC:.0f}s for fills...")
        deadline = time.monotonic() + WAIT_SEC
        while time.monotonic() < deadline:
            time.sleep(3.0)
            try:
                current = client.get_positions(ACCOUNT_ID)
            except Exception:
                continue
            remaining = [p for p in current if abs(p.position) > 1e-6]
            print(f"  remaining positions: {len(remaining)}", end="")
            if remaining:
                syms = [p.symbol for p in remaining]
                print(f" — {syms}", end="")
            print()
            if not remaining:
                print("\n[FLATTEN] ✅ All positions closed.")
                break
        else:
            remaining = [p for p in client.get_positions(ACCOUNT_ID) if abs(p.position) > 1e-6]
            if remaining:
                print(f"\n[FLATTEN] ⚠️  {len(remaining)} positions still open after {WAIT_SEC:.0f}s:")
                for p in remaining:
                    print(f"  {p.symbol:8s}  qty={p.position:.4f}")
                print("  Check TWS or run again.")
            else:
                print("\n[FLATTEN] ✅ All positions closed.")

        # ── Фінальний стан ─────────────────────────────────────────────
        print("\n[STEP 5] Final position check:")
        try:
            final = client.get_positions(ACCOUNT_ID)
            if not final:
                print("  No positions. Account is flat. ✅")
            else:
                for p in final:
                    if abs(p.position) > 1e-6:
                        print(f"  {p.symbol:8s}  qty={p.position:.4f}  mkt_val={p.market_value:.2f}")
        except Exception as exc:
            print(f"  [WARN] final check failed: {exc}")

    finally:
        client.stop_tickle_loop()

    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 1
    try:
        print(f"\n[EXIT] code={rc}")
        input("Press Enter to exit...")
    except Exception:
        pass
    raise SystemExit(rc)
