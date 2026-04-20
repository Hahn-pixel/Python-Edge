from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

ROOT    = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
for _p in [ROOT, SRC_DIR]:
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

from python_edge.broker.cpapi_client import CpapiClient

# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------

def _env_str(k: str, d: str) -> str:
    return str(os.getenv(k, d)).strip()

def _env_float(k: str, d: float) -> float:
    try:
        return float(os.getenv(k, str(d)))
    except Exception:
        return d

def _env_bool(k: str, d: bool) -> bool:
    return str(os.getenv(k, "1" if d else "0")).strip().lower() not in {"0", "false", "no", "off"}

def _env_set(k: str) -> set:
    return {x.strip().upper() for x in str(os.getenv(k, "")).split("|") if x.strip()}


PAUSE_ON_EXIT        = _env_str("PAUSE_ON_EXIT", "auto")
EXECUTION_ROOT       = Path(_env_str("EXECUTION_ROOT", "artifacts/execution_loop"))
CONFIG_NAMES         = [x for x in _env_str("CONFIG_NAMES", "optimal|aggressive").split("|") if x]
BROKER_ACCOUNT_ID    = _env_str("BROKER_ACCOUNT_ID", "")
CPAPI_BASE_URL       = _env_str("CPAPI_BASE_URL",    "https://localhost:5000")
CPAPI_VERIFY_SSL     = _env_bool("CPAPI_VERIFY_SSL",  False)
ALLOW_MISSING_STATE_JSON = _env_bool("ALLOW_MISSING_STATE_JSON", False)
CPAPI_TIMEOUT_SEC    = _env_float("CPAPI_TIMEOUT_SEC", 10.0)

DRIFT_TOLERANCE_SHARES        = _env_float("DRIFT_TOLERANCE_SHARES", 0.000001)
PREFER_STATE_OVER_ORDERS      = _env_bool("PREFER_STATE_OVER_ORDERS", True)
REQUIRE_BROKER_REFRESH        = _env_bool("REQUIRE_BROKER_REFRESH", True)
ALLOW_EXISTING_BROKER_POSITIONS_CSV = _env_bool("ALLOW_EXISTING_BROKER_POSITIONS_CSV", True)
CLEANUP_PREVIEW_MODE          = _env_bool("CLEANUP_PREVIEW_MODE", True)
CLEANUP_INCLUDE_UNEXPECTED    = _env_bool("CLEANUP_INCLUDE_UNEXPECTED", True)
CLEANUP_INCLUDE_DRIFT         = _env_bool("CLEANUP_INCLUDE_DRIFT", True)
CLEANUP_ONLY_SYMBOLS          = _env_set("CLEANUP_ONLY_SYMBOLS")
CLEANUP_EXCLUDE_SYMBOLS       = _env_set("CLEANUP_EXCLUDE_SYMBOLS")
CLEANUP_MIN_ABS_SHARES        = _env_float("CLEANUP_MIN_ABS_SHARES", 1.0)
TOPK_PRINT                    = int(os.getenv("TOPK_PRINT", "50"))

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

def _to_float(v: Any) -> float:
    try:
        return 0.0 if v is None else float(v)
    except Exception:
        return 0.0

def _norm(s: Any) -> str:
    return str(s or "").strip().upper()

def _norm_side(s: Any) -> str:
    v = _norm(s)
    return v if v in {"BUY", "SELL", "HOLD"} else "HOLD"

def _should_pause() -> bool:
    if PAUSE_ON_EXIT in {"0","false","no","off"}: return False
    if PAUSE_ON_EXIT in {"1","true","yes","on"}:  return True
    si, so = getattr(sys,"stdin",None), getattr(sys,"stdout",None)
    return bool(si and so and hasattr(si,"isatty") and si.isatty() and so.isatty())

def _safe_exit(code: int) -> None:
    if _should_pause():
        try:
            print(f"\n[EXIT] code={code}")
            input("Press Enter to exit...")
        except Exception:
            pass
    raise SystemExit(code)

def _enable_lb() -> None:
    for nm in ["stdout","stderr"]:
        s = getattr(sys, nm, None)
        if s and hasattr(s, "reconfigure"):
            try: s.reconfigure(line_buffering=True)
            except Exception: pass

def _save_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

# ---------------------------------------------------------------------------
# Broker data via CPAPI
# ---------------------------------------------------------------------------

def _fetch_positions(client: CpapiClient, account_id: str) -> pd.DataFrame:
    rows = client.get_positions(account_id)
    if not rows:
        return pd.DataFrame(columns=["account","symbol","localSymbol","position","avgCost"])
    data = []
    for r in rows:
        sym = _norm(r.symbol or r.conid)
        data.append({
            "account":      account_id,
            "symbol":       sym,
            "localSymbol":  sym,
            "exchange":     "",
            "primaryExchange": "",
            "currency":     "USD",
            "secType":      "STK",
            "position":     float(r.position),
            "avgCost":      float(r.avg_cost),
        })
    df = pd.DataFrame(data)
    df = df.groupby("symbol", as_index=False).agg(
        account=("account","first"),
        localSymbol=("localSymbol","first"),
        exchange=("exchange","first"),
        primaryExchange=("primaryExchange","first"),
        currency=("currency","first"),
        secType=("secType","first"),
        position=("position","sum"),
        avgCost=("avgCost","first"),
    )
    return df.sort_values("symbol").reset_index(drop=True)


def _fetch_open_orders(client: CpapiClient) -> pd.DataFrame:
    """
    Pull live orders from CPAPI and normalise to the same schema
    that run_broker_reconcile_ibkr.py produces.
    """
    try:
        statuses = client.get_live_orders()
    except Exception as exc:
        print(f"[RECON][WARN] get_live_orders failed: {exc} — returning empty")
        return pd.DataFrame(columns=["ib_order_id","symbol","localSymbol","action",
                                     "orderType","totalQuantity","lmtPrice","tif",
                                     "status","orderRef"])
    rows = []
    for s in statuses:
        raw = s.raw
        sym = _norm(raw.get("ticker","") or raw.get("symbol","") or "")
        action = _norm(raw.get("side","") or raw.get("action",""))
        rows.append({
            "ib_order_id":      str(s.order_id),
            "symbol":           sym,
            "localSymbol":      sym,
            "exchange":         str(raw.get("exchange","") or ""),
            "primaryExchange":  "",
            "currency":         "USD",
            "secType":          "STK",
            "action":           action,
            "orderType":        str(raw.get("orderType","") or ""),
            "totalQuantity":    float(s.filled_qty + s.remaining_qty),
            "lmtPrice":         float(raw.get("price", s.avg_price) or 0.0),
            "auxPrice":         0.0,
            "tif":              str(raw.get("timeInForce","DAY") or "DAY"),
            "outsideRth":       0,
            "account":          str(raw.get("account","") or ""),
            "orderRef":         str(raw.get("cOID","") or raw.get("orderRef","") or ""),
            "status":           str(s.status),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["ib_order_id","symbol","localSymbol","action",
                                     "orderType","totalQuantity","lmtPrice","tif",
                                     "status","orderRef"])
    df["symbol"] = df["symbol"].map(_norm)
    return df.sort_values(["symbol","ib_order_id"]).reset_index(drop=True)

# ---------------------------------------------------------------------------
# State / orders loading  (identical logic to ibkr reconcile)
# ---------------------------------------------------------------------------

def _load_state_positions(path: Path) -> pd.DataFrame:
    if not path.exists():
        if ALLOW_MISSING_STATE_JSON:
            print(f"[RECON][WARN] portfolio_state.json not found: {path} — treating as empty (ALLOW_MISSING_STATE_JSON=1)")
            return pd.DataFrame(columns=["symbol","expected_shares_state","state_last_price",
                                         "state_market_value","state_price_source","state_is_priced"])
        raise FileNotFoundError(f"portfolio_state.json not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    positions = payload.get("positions", {})
    rows = []
    for symbol, pos in (positions.items() if isinstance(positions, dict) else []):
        rows.append({
            "symbol":                _norm(symbol),
            "expected_shares_state": _to_float(pos.get("shares", 0.0)),
            "state_last_price":      _to_float(pos.get("last_price", 0.0)),
            "state_market_value":    _to_float(pos.get("market_value", 0.0)),
            "state_price_source":    str(pos.get("price_source","") or ""),
            "state_is_priced":       int(bool(pos.get("is_priced", False))),
        })
    if not rows:
        return pd.DataFrame(columns=["symbol","expected_shares_state","state_last_price",
                                     "state_market_value","state_price_source","state_is_priced"])
    return pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True)


def _load_orders_targets(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"orders.csv not found: {path}")
    df = pd.read_csv(path)
    if df.empty or "symbol" not in df.columns or "target_shares" not in df.columns:
        return pd.DataFrame(columns=["symbol","expected_shares_orders","price_hint","price_source_orders"])
    df["symbol"]        = df["symbol"].astype(str).map(_norm)
    df["target_shares"] = pd.to_numeric(df["target_shares"], errors="coerce").fillna(0.0)
    agg = df.groupby("symbol", as_index=False).agg(expected_shares_orders=("target_shares","sum"))
    if "price" in df.columns:
        agg = agg.merge(df.groupby("symbol",as_index=False).agg(price_hint=("price","first")), on="symbol", how="left")
    else:
        agg["price_hint"] = 0.0
    if "price_source" in df.columns:
        agg = agg.merge(df.groupby("symbol",as_index=False).agg(price_source_orders=("price_source","first")), on="symbol", how="left")
    else:
        agg["price_source_orders"] = ""
    return agg.sort_values("symbol").reset_index(drop=True)

# ---------------------------------------------------------------------------
# Reconcile logic  (mirror of ibkr reconcile)
# ---------------------------------------------------------------------------

def _build_pending(open_orders_df: pd.DataFrame) -> pd.DataFrame:
    if open_orders_df.empty:
        return pd.DataFrame(columns=["symbol","pending_side","pending_qty","pending_status","pending_order_refs"])
    work = open_orders_df.copy()
    work["status_norm"] = work["status"].astype(str).str.strip().str.lower()
    live = {"presubmitted","submitted","pendingsubmit","pendingcancel","api_pending"}
    work = work.loc[work["status_norm"].isin(live)].copy()
    if work.empty:
        return pd.DataFrame(columns=["symbol","pending_side","pending_qty","pending_status","pending_order_refs"])
    work["action"] = work["action"].map(_norm_side)
    work["signed_qty"] = work.apply(
        lambda r: abs(_to_float(r.get("totalQuantity",0.0))) *
                  (1.0 if r["action"]=="BUY" else -1.0 if r["action"]=="SELL" else 0.0), axis=1)
    g = work.groupby("symbol", as_index=False).agg(
        pending_signed_qty=("signed_qty","sum"),
        pending_status=("status", lambda s: "|".join(sorted({str(x) for x in s if str(x)}))),
        pending_order_refs=("orderRef", lambda s: "|".join(sorted({str(x) for x in s if str(x)}))),
    )
    g["pending_side"] = g["pending_signed_qty"].map(lambda x: "BUY" if x>0 else "SELL" if x<0 else "HOLD")
    g["pending_qty"]  = g["pending_signed_qty"].abs()
    return g[["symbol","pending_side","pending_qty","pending_status","pending_order_refs"]].sort_values("symbol").reset_index(drop=True)


def _merge_all(state_df, orders_df, broker_df, pending_df) -> pd.DataFrame:
    m = pd.merge(state_df, orders_df, on="symbol", how="outer")
    m = pd.merge(m, broker_df.rename(columns={"position":"broker_position","avgCost":"broker_avg_cost"}), on="symbol", how="outer")
    m = pd.merge(m, pending_df, on="symbol", how="left")

    for col, default in [("expected_shares_state",0.0),("expected_shares_orders",0.0),
                         ("broker_position",0.0),("pending_qty",0.0)]:
        if col not in m.columns: m[col] = default
        m[col] = pd.to_numeric(m[col], errors="coerce").fillna(default)

    m["expected_shares"] = m["expected_shares_state"] if PREFER_STATE_OVER_ORDERS else m["expected_shares_orders"]
    missing_mask = m["expected_shares_state"].abs() <= DRIFT_TOLERANCE_SHARES
    if PREFER_STATE_OVER_ORDERS:
        m.loc[missing_mask, "expected_shares"] = m.loc[missing_mask, "expected_shares_orders"]

    m["drift_shares"]     = m["broker_position"] - m["expected_shares"]
    m["abs_drift_shares"] = m["drift_shares"].abs()
    m["broker_has_position"]   = m["broker_position"].abs() > DRIFT_TOLERANCE_SHARES
    m["expected_has_position"] = m["expected_shares"].abs() > DRIFT_TOLERANCE_SHARES

    m["pending_signed_qty"] = m.apply(
        lambda r: float(r["pending_qty"]) * (1.0 if str(r.get("pending_side",""))=="BUY"
                                              else -1.0 if str(r.get("pending_side",""))=="SELL" else 0.0), axis=1)
    m["drift_after_pending"]      = m["broker_position"] + m["pending_signed_qty"] - m["expected_shares"]
    m["abs_drift_after_pending"]  = m["drift_after_pending"].abs()
    m["drift"]                    = m["abs_drift_shares"] > DRIFT_TOLERANCE_SHARES
    m["drift_after_pending_flag"] = m["abs_drift_after_pending"] > DRIFT_TOLERANCE_SHARES
    m["missing_at_broker"]        = m["expected_has_position"] & ~m["broker_has_position"]
    m["unexpected_at_broker"]     = ~m["expected_has_position"] & m["broker_has_position"]
    m["pending_covers_drift"]     = m["drift"] & ~m["drift_after_pending_flag"]

    def _classify(row):
        if row.get("pending_covers_drift"):  return "pending_covers_drift"
        if row.get("missing_at_broker"):     return "missing_at_broker"
        if row.get("unexpected_at_broker"):  return "unexpected_at_broker"
        if row.get("drift"):                 return "drift"
        return "ok"

    m["issue_kind"]   = m.apply(_classify, axis=1)
    m["cleanup_side"] = m["broker_position"].map(lambda x: "SELL" if x>DRIFT_TOLERANCE_SHARES else "BUY" if x<-DRIFT_TOLERANCE_SHARES else "HOLD")
    m["cleanup_qty"]  = m["broker_position"].abs()
    return m.sort_values(["abs_drift_shares","symbol"], ascending=[False,True]).reset_index(drop=True)


def _build_cleanup_preview(merged: pd.DataFrame) -> pd.DataFrame:
    cols = ["symbol","cleanup_side","cleanup_qty","cleanup_reason","broker_avg_cost",
            "broker_position","expected_shares","pending_side","pending_qty",
            "pending_status","pending_order_refs"]
    if merged.empty:
        return pd.DataFrame(columns=cols)
    work = merged.copy()
    mask = pd.Series(False, index=work.index)
    if CLEANUP_INCLUDE_UNEXPECTED:
        mask = mask | work["unexpected_at_broker"].fillna(False)
    if CLEANUP_INCLUDE_DRIFT:
        mask = mask | (work["drift"].fillna(False) & ~work["pending_covers_drift"].fillna(False))
    work = work.loc[mask].copy()
    work = work.loc[work["cleanup_qty"].abs() >= CLEANUP_MIN_ABS_SHARES].copy()
    if CLEANUP_ONLY_SYMBOLS:
        work = work.loc[work["symbol"].isin(CLEANUP_ONLY_SYMBOLS)].copy()
    if CLEANUP_EXCLUDE_SYMBOLS:
        work = work.loc[~work["symbol"].isin(CLEANUP_EXCLUDE_SYMBOLS)].copy()
    if work.empty:
        return pd.DataFrame(columns=cols)
    work["cleanup_reason"] = work["issue_kind"].astype(str)
    available = [c for c in cols if c in work.columns]
    return work[available].sort_values(["cleanup_qty","symbol"], ascending=[False,True]).reset_index(drop=True)


def _summary(config_name: str, merged: pd.DataFrame, cleanup_df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "config":               config_name,
        "generated_at_utc":     _utc_now_iso(),
        "symbols_total":        int(len(merged)),
        "drift":                int(merged["drift"].fillna(False).sum()),
        "missing_at_broker":    int(merged["missing_at_broker"].fillna(False).sum()),
        "unexpected_at_broker": int(merged["unexpected_at_broker"].fillna(False).sum()),
        "pending_covers_drift": int(merged["pending_covers_drift"].fillna(False).sum()),
        "max_abs_drift_shares": float(merged["abs_drift_shares"].max()) if len(merged) else 0.0,
        "cleanup_preview_rows": int(len(cleanup_df)),
        "cleanup_preview_enabled": int(CLEANUP_PREVIEW_MODE),
        "source": "cpapi",
    }

# ---------------------------------------------------------------------------
# Per-config runner
# ---------------------------------------------------------------------------

def _run_one(config_name: str, client: Optional[CpapiClient]) -> None:
    base = EXECUTION_ROOT / config_name
    base.mkdir(parents=True, exist_ok=True)

    paths = {
        "state":           base / "portfolio_state.json",
        "orders":          base / "orders.csv",
        "broker_pos":      base / "broker_positions.csv",
        "broker_orders":   base / "broker_open_orders.csv",
        "broker_pending":  base / "broker_pending.csv",
        "reconcile":       base / "broker_reconcile.csv",
        "recon_summary":   base / "broker_reconcile_summary.json",
        "cleanup_preview": base / "broker_cleanup_preview.csv",
    }

    # Broker positions
    if client is not None:
        if not BROKER_ACCOUNT_ID:
            raise RuntimeError("BROKER_ACCOUNT_ID is required for CPAPI reconcile")
        broker_df = _fetch_positions(client, BROKER_ACCOUNT_ID)
        _save_df(broker_df, paths["broker_pos"])
        open_orders_df = _fetch_open_orders(client)
        _save_df(open_orders_df, paths["broker_orders"])
    elif ALLOW_EXISTING_BROKER_POSITIONS_CSV and paths["broker_pos"].exists():
        broker_df = pd.read_csv(paths["broker_pos"])
        if "symbol" in broker_df.columns:
            broker_df["symbol"] = broker_df["symbol"].map(_norm)
        open_orders_df = (pd.read_csv(paths["broker_orders"])
                          if paths["broker_orders"].exists() else pd.DataFrame())
    else:
        raise RuntimeError(f"Broker positions unavailable for config={config_name}")

    state_df   = _load_state_positions(paths["state"])
    orders_df  = _load_orders_targets(paths["orders"])
    pending_df = _build_pending(open_orders_df)
    _save_df(pending_df, paths["broker_pending"])

    merged     = _merge_all(state_df, orders_df, broker_df, pending_df)
    cleanup_df = _build_cleanup_preview(merged) if CLEANUP_PREVIEW_MODE else pd.DataFrame()
    summary    = _summary(config_name, merged, cleanup_df)

    _save_df(merged,     paths["reconcile"])
    _save_df(cleanup_df, paths["cleanup_preview"])
    paths["recon_summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        f"[RECON][{config_name}][SUMMARY] "
        f"symbols_total={summary['symbols_total']} drift={summary['drift']} "
        f"missing_at_broker={summary['missing_at_broker']} "
        f"unexpected_at_broker={summary['unexpected_at_broker']} "
        f"pending_covers_drift={summary['pending_covers_drift']} "
        f"max_abs_drift_shares={summary['max_abs_drift_shares']:.8f} "
        f"cleanup_preview_rows={summary['cleanup_preview_rows']}"
    )
    for key, path in paths.items():
        if path.exists():
            print(f"[ARTIFACT] {path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    _enable_lb()
    print(f"[CFG] execution_root={EXECUTION_ROOT} configs={CONFIG_NAMES}")
    print(f"[CFG] cpapi_url={CPAPI_BASE_URL} account={BROKER_ACCOUNT_ID}")
    print(f"[CFG] require_broker_refresh={int(REQUIRE_BROKER_REFRESH)} "
          f"drift_tolerance={DRIFT_TOLERANCE_SHARES} "
          f"prefer_state_over_orders={int(PREFER_STATE_OVER_ORDERS)}")

    if not BROKER_ACCOUNT_ID:
        print("[FATAL] BROKER_ACCOUNT_ID is required")
        return 1

    client: Optional[CpapiClient] = None
    if REQUIRE_BROKER_REFRESH:
        client = CpapiClient(CPAPI_BASE_URL, CPAPI_TIMEOUT_SEC, CPAPI_VERIFY_SSL)
        client.assert_authenticated()
        client.start_tickle_loop()

    try:
        for cfg in CONFIG_NAMES:
            _run_one(cfg, client)
    finally:
        if client is not None:
            client.stop_tickle_loop()

    print("[FINAL] CPAPI broker reconcile complete")
    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 1
    _safe_exit(rc)
