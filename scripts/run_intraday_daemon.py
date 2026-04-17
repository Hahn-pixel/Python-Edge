"""
run_intraday_daemon.py — Intraday Exit Daemon

Безперервний процес що моніторить відкриті позиції під час RTH
і автоматично закриває їх при спрацюванні exit-умов.

Логіка per механізм:
  stop_loss / take_profit  → MKT order напряму через cpapi_client
                             при failed: алерт + повторює кожен цикл
  trailing_stop / max_hold → записує в orders.csv + викликає run_cpapi_execution.py

Trailing stop: volatility-scaled через ivol_20d
  Джерело ivol (варіант C):
    1. live_feature_snapshot.parquet  (оновлюється щодня pipeline)
    2. fallback: exit_meta.ivol_20d_cached (збережений exit scanner'ом)

Координація з pipeline:
  Daemon бачить artifacts/pipeline.lock → пауза → чекає зникнення → продовжує

RTH: 9:30–16:00 ET (America/New_York). Поза RTH — daemon спить.

Важливі захисти:
  - entry_price ініціалізований у цьому циклі → evaluate НЕ запускається
    (уникаємо хибних тригерів через застарілу ціну в state)
  - CPAPI snapshot: retry з паузою (як в run_cpapi_handoff.py)
  - conid_cache: шерується між configs (optimal → aggressive fallback)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parents[1]
for _p in [str(_ROOT), str(_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from python_edge.broker.cpapi_client import CpapiClient
from python_edge.broker.cpapi_models import CpapiOrderRequest
from python_edge.portfolio.exit_policy import ExitPolicy


# ──────────────────────────────────────────────────────────────
# Env helpers
# ──────────────────────────────────────────────────────────────
def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()

def _env_bool(key: str, default: bool = False) -> bool:
    v = os.getenv(key, "").strip().lower()
    if v in ("1", "true", "yes", "on"):  return True
    if v in ("0", "false", "no", "off"): return False
    return default

def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


# ──────────────────────────────────────────────────────────────
# Time / RTH helpers
# ──────────────────────────────────────────────────────────────
def _now_et() -> datetime:
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        import datetime as _dt
        utc_now = _dt.datetime.now(_dt.timezone.utc)
        return utc_now.replace(tzinfo=None) - _dt.timedelta(hours=4)

def _is_rth() -> bool:
    now = _now_et()
    if now.weekday() >= 5:
        return False
    t = now.hour * 60 + now.minute
    return 9 * 60 + 30 <= t < 16 * 60

def _today_str() -> str:
    return _now_et().date().isoformat()

def _ts() -> str:
    return _now_et().strftime("%H:%M:%S")

def _days_between(date_str: str, today_str: str) -> int:
    try:
        d1 = date.fromisoformat(date_str[:10])
        d2 = date.fromisoformat(today_str[:10])
        return max(0, (d2 - d1).days)
    except (ValueError, TypeError):
        return 0


# ──────────────────────────────────────────────────────────────
# ivol_20d зі snapshot (варіант C)
# ──────────────────────────────────────────────────────────────
def _load_ivol_from_snapshot(snapshot_path: Path) -> dict[str, float]:
    ivol: dict[str, float] = {}
    if not snapshot_path.exists():
        return ivol
    try:
        df = pd.read_parquet(snapshot_path, columns=["symbol", "ivol_20d"])
        df = df.dropna(subset=["ivol_20d"])
        df["ivol_20d"] = pd.to_numeric(df["ivol_20d"], errors="coerce")
        df = df.dropna(subset=["ivol_20d"])
        latest = df.groupby("symbol")["ivol_20d"].last()
        ivol = {str(s).upper(): float(v) for s, v in latest.items() if v > 0}
        print(f"[DAEMON][{_ts()}] ivol reloaded: {len(ivol)} symbols from snapshot")
    except Exception as exc:
        print(f"[DAEMON][{_ts()}] snapshot ivol error: {exc}")
    return ivol


def _get_ivol(symbol: str, snapshot_ivol: dict[str, float], meta: dict) -> Optional[float]:
    """Варіант C: snapshot → fallback exit_meta.ivol_20d_cached."""
    v = snapshot_ivol.get(symbol.upper())
    if v and v > 0:
        return v
    cached = meta.get("ivol_20d_cached")
    if cached:
        try:
            f = float(cached)
            if f > 0:
                return f
        except (ValueError, TypeError):
            pass
    return None


# ──────────────────────────────────────────────────────────────
# conid cache — шерується між configs
# ──────────────────────────────────────────────────────────────
def _load_merged_conid_cache(exec_root: Path, configs: list[str]) -> dict[str, int]:
    """
    Об'єднує conid_cache.json з усіх configs.
    optimal і aggressive торгують тими самими символами —
    один може мати conid якого немає в іншого.
    """
    merged: dict[str, int] = {}
    for cfg in configs:
        path = exec_root / cfg / "conid_cache.json"
        if not path.exists():
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            for k, v in raw.items():
                if isinstance(v, (int, str)) and str(v).isdigit():
                    merged[str(k).upper()] = int(v)
        except Exception:
            pass
    return merged


# ──────────────────────────────────────────────────────────────
# Price fetching — CPAPI snapshot з retry
# ──────────────────────────────────────────────────────────────
def _fetch_prices_cpapi(
    symbols: list[str],
    conid_cache: dict[str, int],
    base_url: str,
    verify_ssl: bool,
    timeout: float,
    retries: int = 3,
    wait_sec: float = 2.0,
) -> dict[str, float]:
    """
    Snapshot з retry — CPAPI потребує кількох спроб поки дані не з'являться
    (streaming підписка). Ідентична логіка до run_cpapi_handoff.py.
    """
    prices: dict[str, float] = {}
    sym_conid = {s.upper(): conid_cache[s.upper()] for s in symbols if s.upper() in conid_cache}
    if not sym_conid:
        return prices

    session = requests.Session()
    session.verify = verify_ssl
    conids_str = ",".join(str(c) for c in sym_conid.values())
    url = f"{base_url.rstrip('/')}/v1/api/iserver/marketdata/snapshot"
    conid_to_sym = {v: k for k, v in sym_conid.items()}

    for attempt in range(1, retries + 1):
        if attempt > 1:
            time.sleep(wait_sec)
        try:
            resp = session.get(
                url,
                params={"conids": conids_str, "fields": "31,84,86"},
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, list):
                continue
            for item in data:
                sym = conid_to_sym.get(item.get("conid"))
                if not sym:
                    continue
                try:
                    bid  = item.get("84")
                    ask  = item.get("86")
                    last = item.get("31")
                    if bid and ask:
                        px = (
                            float(str(bid).replace(",", "")) +
                            float(str(ask).replace(",", ""))
                        ) / 2.0
                    elif last:
                        px = float(str(last).replace(",", ""))
                    else:
                        continue
                    if px > 0:
                        prices[sym] = px
                except (ValueError, TypeError):
                    pass
            # Якщо всі символи отримали ціну — зупиняємось
            if all(prices.get(s, 0) > 0 for s in sym_conid):
                break
            missing = [s for s in sym_conid if prices.get(s, 0) <= 0]
            print(
                f"[DAEMON][{_ts()}] CPAPI attempt={attempt}/{retries} "
                f"missing={len(missing)}: {missing[:5]}"
            )
        except Exception as exc:
            print(f"[DAEMON][{_ts()}] CPAPI snapshot attempt={attempt} error: {exc}")

    return prices


# ──────────────────────────────────────────────────────────────
# Price fetching — massive fallback (per-symbol endpoint)
# ──────────────────────────────────────────────────────────────
def _fetch_price_massive_one(
    symbol: str,
    api_key: str,
    base_url: str,
    timeout: float,
) -> float:
    """
    GET /v2/snapshot/locale/us/markets/stocks/tickers/{SYMBOL}
    Той самий endpoint що в run_cpapi_handoff.py.
    """
    url = (
        f"{base_url.rstrip('/')}/v2/snapshot/locale/us"
        f"/markets/stocks/tickers/{symbol.upper()}"
    )
    try:
        resp = requests.get(url, params={"apiKey": api_key}, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
        ticker = payload.get("ticker", {}) or {}
        lq  = ticker.get("lastQuote", {}) or {}
        lt  = ticker.get("lastTrade",  {}) or {}
        day = ticker.get("day",        {}) or {}
        prv = ticker.get("prevDay",    {}) or {}

        bid   = float(lq.get("p") or 0)
        ask   = float(lq.get("P") or 0)
        last  = float(lt.get("p") or 0)
        close = float(day.get("c") or prv.get("c") or 0)

        if bid > 0 and ask > 0:
            return (bid + ask) / 2.0
        if last > 0:
            return last
        if close > 0:
            return close
    except Exception as exc:
        print(f"[DAEMON][{_ts()}] massive {symbol} error: {exc}")
    return 0.0


def _fetch_prices_massive(
    symbols: list[str],
    api_key: str,
    base_url: str,
    timeout: float,
) -> dict[str, float]:
    prices: dict[str, float] = {}
    if not symbols or not api_key:
        return prices
    for sym in symbols:
        px = _fetch_price_massive_one(sym, api_key, base_url, timeout)
        if px > 0:
            prices[sym] = px
            print(f"[DAEMON][{_ts()}] {sym} via massive: {px:.4f}")
        time.sleep(0.1)
    return prices


def _get_prices(
    symbols: list[str],
    conid_cache: dict[str, int],
    base_url: str,
    verify_ssl: bool,
    timeout: float,
    enable_massive: bool,
    massive_api_key: str,
    massive_base_url: str,
    massive_timeout: float,
) -> dict[str, float]:
    prices  = _fetch_prices_cpapi(symbols, conid_cache, base_url, verify_ssl, timeout)
    missing = [s for s in symbols if prices.get(s, 0) <= 0]
    if missing and enable_massive:
        fallback = _fetch_prices_massive(
            missing, massive_api_key, massive_base_url, massive_timeout,
        )
        for sym, px in fallback.items():
            if px > 0:
                prices[sym] = px
    return prices


# ──────────────────────────────────────────────────────────────
# State / orders helpers
# ──────────────────────────────────────────────────────────────
def _load_state(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[DAEMON] Cannot read {path}: {exc}")
        return {}

def _save_state(path: Path, state: dict) -> None:
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")

_ORDERS_COLUMNS = [
    "symbol", "order_side", "delta_shares", "target_shares",
    "price", "price_source", "exit_reason",
]

def _load_orders(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=_ORDERS_COLUMNS)
    try:
        df = pd.read_csv(path)
        for col in _ORDERS_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return df
    except Exception:
        return pd.DataFrame(columns=_ORDERS_COLUMNS)

def _append_exit_order(
    orders_path: Path,
    symbol: str,
    exit_side: str,
    shares: float,
    price: float,
    reason: str,
) -> None:
    df = _load_orders(orders_path)
    existing = df.loc[
        (df["symbol"].astype(str).str.upper() == symbol.upper()) &
        (df.get("exit_reason", pd.Series(dtype=str)).astype(str) == reason)
    ]
    if not existing.empty:
        return
    new_row = pd.DataFrame([{
        "symbol":        symbol,
        "order_side":    exit_side,
        "delta_shares":  abs(shares),
        "target_shares": 0.0,
        "price":         round(price, 4),
        "price_source":  "intraday_daemon",
        "exit_reason":   reason,
    }])
    pd.concat([df, new_row], ignore_index=True).to_csv(orders_path, index=False)


# ──────────────────────────────────────────────────────────────
# MKT order (stop_loss / take_profit)
# ──────────────────────────────────────────────────────────────
def _send_mkt_order(
    client: CpapiClient,
    account_id: str,
    conid: int,
    symbol: str,
    side: str,
    qty: float,
    dry_run: bool,
) -> tuple[bool, str]:
    if dry_run:
        print(f"[DAEMON][{_ts()}] DRY_RUN MKT {side} {qty:.0f} {symbol}")
        return True, "dry_run"
    tag = f"daemon_mkt_{symbol}_{side[:1]}_{uuid.uuid4().hex[:6]}"
    req = CpapiOrderRequest(
        conid=conid,
        account_id=account_id,
        side=side,
        quantity=qty,
        order_type="MKT",
        tif="DAY",
        client_tag=tag,
        price=None,
    )
    try:
        resp = client.submit_order(account_id, req)
        ok   = bool(resp.order_id)
        return ok, resp.order_id or resp.message or "no_order_id"
    except Exception as exc:
        return False, str(exc)


# ──────────────────────────────────────────────────────────────
# run_cpapi_execution.py як subprocess (trailing / max_hold)
# ──────────────────────────────────────────────────────────────
def _run_execution_subprocess(
    cfg: str,
    exec_root: Path,
    account_id: str,
    cpapi_base_url: str,
    dry_run: bool,
) -> bool:
    if dry_run:
        print(f"[DAEMON][{_ts()}] DRY_RUN subprocess execution for {cfg}")
        return True
    script = _ROOT / "scripts" / "run_cpapi_execution.py"
    env = os.environ.copy()
    env.update({
        "EXECUTION_ROOT":          str(exec_root),
        "CONFIG_NAMES":            cfg,
        "BROKER_ACCOUNT_ID":       account_id,
        "CPAPI_BASE_URL":          cpapi_base_url,
        "CPAPI_VERIFY_SSL":        "0",
        "CPAPI_TIMEOUT_SEC":       "10.0",
        "CPAPI_WHOLE_TIMEOUT_SEC": "30.0",
        "CPAPI_FRAC_TIMEOUT_SEC":  "20.0",
        "CPAPI_TIF":               "DAY",
        "CPAPI_FRAC_SLIPPAGE_BPS": "5.0",
        "CPAPI_RESOLVE_CONIDS":    "0",
        "RESET_BROKER_LOG":        "0",
        "PAUSE_ON_EXIT":           "0",
    })
    try:
        result = subprocess.run(
            [sys.executable, str(script)],
            env=env,
            cwd=str(_ROOT),
            timeout=120,
        )
        return result.returncode == 0
    except Exception as exc:
        print(f"[DAEMON][{_ts()}] subprocess execution error: {exc}")
        return False


# ──────────────────────────────────────────────────────────────
# Notifier (заглушка — повністю в Пріоритет 3)
# ──────────────────────────────────────────────────────────────
def _send_alert(subject: str, body: str) -> None:
    print(f"[DAEMON][ALERT][{_ts()}] {subject}")
    print(f"  {body}")
    # TODO: реалізація в run_notifier.py (Пріоритет 3)


# ──────────────────────────────────────────────────────────────
# Pipeline lock
# ──────────────────────────────────────────────────────────────
def _wait_for_pipeline(lock_file: Path, wait_sec: float) -> None:
    print(f"[DAEMON][{_ts()}] pipeline.lock detected — pausing...")
    while lock_file.exists():
        time.sleep(wait_sec)
    print(f"[DAEMON][{_ts()}] pipeline.lock gone — resuming")


# ──────────────────────────────────────────────────────────────
# Один цикл моніторингу для одного config
# ──────────────────────────────────────────────────────────────
def _process_config_cycle(
    cfg: str,
    exec_root: Path,
    client: CpapiClient,
    account_id: str,
    policy: ExitPolicy,
    snapshot_ivol: dict[str, float],
    merged_conid_cache: dict[str, int],
    base_url: str,
    verify_ssl: bool,
    timeout: float,
    enable_massive: bool,
    massive_api_key: str,
    massive_base_url: str,
    massive_timeout: float,
    dry_run: bool,
    today: str,
    cpapi_base_url: str,
) -> dict:
    counters = {
        "checked": 0, "exit_mkt": 0, "exit_lmt": 0,
        "mkt_ok": 0, "mkt_fail": 0, "lmt_ok": 0, "lmt_fail": 0,
        "no_price": 0, "pending_retry": 0, "ivol_missing": 0,
        "init_skipped": 0,
    }

    state_path  = exec_root / cfg / "portfolio_state.json"
    orders_path = exec_root / cfg / "orders.csv"

    state = _load_state(state_path)
    if not state:
        return counters

    positions: dict = state.get("positions", {})
    if not positions:
        return counters

    exit_meta: dict = state.setdefault("exit_meta", {})
    symbols = list(positions.keys())

    prices = _get_prices(
        symbols, merged_conid_cache,
        base_url, verify_ssl, timeout,
        enable_massive, massive_api_key, massive_base_url, massive_timeout,
    )

    state_dirty = False

    for symbol, pos in positions.items():
        counters["checked"] += 1
        shares = float(pos.get("shares", 0))
        if shares == 0:
            continue

        side          = 1.0 if shares > 0 else -1.0
        current_price = prices.get(symbol.upper(), 0.0)
        if current_price <= 0:
            current_price = float(pos.get("last_price", 0))
        if current_price <= 0:
            counters["no_price"] += 1
            continue

        meta         = exit_meta.setdefault(symbol, {})
        entry_price  = float(meta.get("entry_price") or 0)
        entry_date   = str(meta.get("entry_date") or today)
        peak_price   = float(meta.get("peak_price") or 0)
        pending_exit = str(meta.get("pending_exit") or "")

        # ivol: варіант C
        ivol_20d = _get_ivol(symbol, snapshot_ivol, meta)
        if ivol_20d is None:
            counters["ivol_missing"] += 1

        # ── Ініціалізація entry_price ─────────────────────────
        # ВАЖЛИВО: якщо entry_price відсутній — ініціалізуємо
        # поточною живою ціною і ПРОПУСКАЄМО evaluate в цьому циклі.
        # Це запобігає хибним тригерам через застарілу ціну в state.
        just_initialized = False
        if entry_price <= 0:
            meta["entry_price"] = current_price
            meta["entry_date"]  = today
            entry_price = current_price
            state_dirty = True
            just_initialized = True
            counters["init_skipped"] += 1
            print(
                f"[DAEMON][{_ts()}][{cfg}] {symbol} — "
                f"entry initialized={current_price:.4f} "
                f"(no eval this cycle)"
            )

        # Оновити peak_price
        if side > 0:
            if current_price > peak_price:
                meta["peak_price"] = current_price
                peak_price  = current_price
                state_dirty = True
        else:
            if peak_price <= 0 or current_price < peak_price:
                meta["peak_price"] = current_price
                peak_price  = current_price
                state_dirty = True

        # Не тригеримо якщо щойно ініціалізовано
        if just_initialized:
            continue

        # ── Retry pending MKT ─────────────────────────────────
        if pending_exit in ("stop_loss", "take_profit"):
            counters["pending_retry"] += 1
            pnl_pct   = (current_price - entry_price) / entry_price * side
            exit_side = "SELL" if side > 0 else "BUY"
            conid     = merged_conid_cache.get(symbol.upper())
            print(
                f"[DAEMON][{_ts()}][{cfg}] RETRY MKT {symbol} "
                f"reason={pending_exit} pnl={pnl_pct:+.2%}"
            )
            if conid:
                ok, oid = _send_mkt_order(
                    client, account_id, conid, symbol,
                    exit_side, abs(shares), dry_run,
                )
                if ok:
                    counters["mkt_ok"] += 1
                    meta["pending_exit"] = ""
                    state_dirty = True
                    print(f"[DAEMON][{_ts()}][{cfg}] MKT RETRY OK {symbol} oid={oid}")
                else:
                    counters["mkt_fail"] += 1
                    _send_alert(
                        f"EXIT RETRY FAILED: {cfg}/{symbol}",
                        f"reason={pending_exit} qty={abs(shares):.0f} error={oid}",
                    )
            else:
                print(
                    f"[DAEMON][{_ts()}][{cfg}] {symbol} no conid for MKT retry "
                    f"(checked {len(merged_conid_cache)} cached conids)"
                )
                _send_alert(
                    f"EXIT RETRY BLOCKED (no conid): {cfg}/{symbol}",
                    f"reason={pending_exit} — conid missing in merged cache",
                )
            continue

        # ── Evaluate exit ─────────────────────────────────────
        reason = policy.evaluate(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            current_price=current_price,
            peak_price=peak_price,
            entry_date=entry_date,
            today=today,
            ivol_20d=ivol_20d,
        )

        if not reason:
            continue

        pnl_pct   = (current_price - entry_price) / entry_price * side
        hold_days = _days_between(entry_date, today)
        exit_side = "SELL" if side > 0 else "BUY"
        trail_desc = policy.describe_trail(ivol_20d)

        print(
            f"[DAEMON][{_ts()}][{cfg}] EXIT {symbol} {exit_side} {abs(shares):.0f}sh "
            f"reason={reason} pnl={pnl_pct:+.2%} hold={hold_days}d "
            f"entry={entry_price:.4f} cur={current_price:.4f} peak={peak_price:.4f} "
            f"{trail_desc}"
        )

        # ── MKT: stop_loss / take_profit ──────────────────────
        if reason in ("stop_loss", "take_profit"):
            counters["exit_mkt"] += 1
            conid = merged_conid_cache.get(symbol.upper())
            if not conid:
                print(
                    f"[DAEMON][{_ts()}][{cfg}] {symbol} no conid in merged cache "
                    f"({len(merged_conid_cache)} entries) — cannot send MKT"
                )
                _send_alert(
                    f"EXIT BLOCKED (no conid): {cfg}/{symbol}",
                    f"reason={reason} — check conid_cache in all configs",
                )
                continue
            ok, oid = _send_mkt_order(
                client, account_id, conid, symbol,
                exit_side, abs(shares), dry_run,
            )
            if ok:
                counters["mkt_ok"] += 1
                meta["pending_exit"] = ""
                state_dirty = True
                _send_alert(
                    f"EXIT SENT: {cfg}/{symbol} {reason}",
                    f"MKT {exit_side} {abs(shares):.0f}sh @ ~{current_price:.4f} "
                    f"pnl={pnl_pct:+.2%} oid={oid}",
                )
            else:
                counters["mkt_fail"] += 1
                meta["pending_exit"] = reason
                state_dirty = True
                _send_alert(
                    f"EXIT FAILED (MKT): {cfg}/{symbol} {reason}",
                    f"qty={abs(shares):.0f} error={oid} — retrying next cycle",
                )

        # ── LMT через executor: trailing_stop / max_hold ──────
        else:
            counters["exit_lmt"] += 1
            _append_exit_order(
                orders_path, symbol, exit_side,
                abs(shares), current_price, reason,
            )
            ok = _run_execution_subprocess(
                cfg, exec_root, account_id, cpapi_base_url, dry_run,
            )
            if ok:
                counters["lmt_ok"] += 1
                _send_alert(
                    f"EXIT SENT: {cfg}/{symbol} {reason}",
                    f"LMT {exit_side} {abs(shares):.0f}sh @ {current_price:.4f} "
                    f"pnl={pnl_pct:+.2%} hold={hold_days}d {trail_desc}",
                )
            else:
                counters["lmt_fail"] += 1
                _send_alert(
                    f"EXIT FAILED (LMT): {cfg}/{symbol} {reason}",
                    f"qty={abs(shares):.0f} — retrying next cycle",
                )

    if state_dirty:
        _save_state(state_path, state)

    return counters


# ──────────────────────────────────────────────────────────────
# Main daemon loop
# ──────────────────────────────────────────────────────────────
def main() -> None:
    exec_root       = Path(_env("EXECUTION_ROOT", "artifacts/execution_loop"))
    snapshot_path   = Path(_env(
        "LIVE_FEATURE_SNAPSHOT_FILE",
        "artifacts/live_alpha/live_feature_snapshot.parquet",
    ))
    configs         = [c for c in _env("CONFIG_NAMES", "optimal|aggressive").split("|") if c]
    cpapi_base_url  = _env("CPAPI_BASE_URL", "https://localhost:5000")
    verify_ssl      = _env_bool("CPAPI_VERIFY_SSL", False)
    timeout         = _env_float("CPAPI_TIMEOUT_SEC", 10.0)
    account_id      = _env("BROKER_ACCOUNT_ID")
    poll_interval   = _env_float("POLL_INTERVAL_SEC", 30.0)
    lock_file       = Path(_env("PIPELINE_LOCK_FILE", "artifacts/pipeline.lock"))
    lock_wait       = _env_float("PIPELINE_LOCK_WAIT_SEC", 10.0)
    enable_massive  = _env_bool("ENABLE_MASSIVE_FALLBACK", True)
    massive_api_key = _env("MASSIVE_API_KEY")
    massive_base_url= _env("MASSIVE_BASE_URL", "https://api.massive.com")
    massive_timeout = _env_float("MASSIVE_TIMEOUT_SEC", 20.0)
    dry_run         = _env_bool("DAEMON_DRY_RUN", False)

    if not account_id and not dry_run:
        print("[DAEMON] BROKER_ACCOUNT_ID not set — required for MKT orders")
        print("[DAEMON] Set BROKER_ACCOUNT_ID or use DAEMON_DRY_RUN=1")
        input("\nPress Enter to exit...")
        sys.exit(1)

    policies = {cfg: ExitPolicy.from_env(cfg) for cfg in configs}
    for cfg, p in policies.items():
        print(f"[DAEMON] policy [{cfg}]: {p}")

    client = CpapiClient(cpapi_base_url, timeout, verify_ssl)
    client.start_tickle_loop()

    snapshot_ivol    = _load_ivol_from_snapshot(snapshot_path)
    ivol_loaded_date = _today_str()

    print(f"[DAEMON] started poll_interval={poll_interval}s dry_run={dry_run}")
    print(f"[DAEMON] verify_ssl={verify_ssl} cpapi={cpapi_base_url}")
    print(f"[DAEMON] configs={configs} RTH=09:30-16:00 ET")
    print(f"[DAEMON] Ctrl+C to stop")

    cycle = 0
    try:
        while True:
            # Перезавантажити ivol якщо новий день
            current_date = _today_str()
            if current_date != ivol_loaded_date:
                snapshot_ivol    = _load_ivol_from_snapshot(snapshot_path)
                ivol_loaded_date = current_date

            # Перезавантажити merged conid cache кожен цикл
            # (pipeline може оновити cache після handoff)
            merged_conid_cache = _load_merged_conid_cache(exec_root, configs)

            # Pipeline lock
            if lock_file.exists():
                _wait_for_pipeline(lock_file, lock_wait)

            # RTH check
            if not _is_rth():
                now = _now_et()
                print(
                    f"[DAEMON][{_ts()}] outside RTH "
                    f"(wd={now.weekday()} {now.strftime('%H:%M')}) — "
                    f"sleeping {poll_interval}s"
                )
                time.sleep(poll_interval)
                continue

            cycle += 1
            today = _today_str()
            print(
                f"\n[DAEMON][{_ts()}] cycle={cycle} date={today} "
                f"conids={len(merged_conid_cache)}"
            )

            for cfg in configs:
                try:
                    c = _process_config_cycle(
                        cfg=cfg,
                        exec_root=exec_root,
                        client=client,
                        account_id=account_id,
                        policy=policies[cfg],
                        snapshot_ivol=snapshot_ivol,
                        merged_conid_cache=merged_conid_cache,
                        base_url=cpapi_base_url,
                        verify_ssl=verify_ssl,
                        timeout=timeout,
                        enable_massive=enable_massive,
                        massive_api_key=massive_api_key,
                        massive_base_url=massive_base_url,
                        massive_timeout=massive_timeout,
                        dry_run=dry_run,
                        today=today,
                        cpapi_base_url=cpapi_base_url,
                    )
                    print(
                        f"[DAEMON][{_ts()}][{cfg}] "
                        f"checked={c['checked']} "
                        f"init_skipped={c['init_skipped']} "
                        f"mkt={c['exit_mkt']}(ok={c['mkt_ok']}/fail={c['mkt_fail']}) "
                        f"lmt={c['exit_lmt']}(ok={c['lmt_ok']}/fail={c['lmt_fail']}) "
                        f"no_price={c['no_price']} "
                        f"ivol_missing={c['ivol_missing']} "
                        f"pending_retry={c['pending_retry']}"
                    )
                except Exception as exc:
                    print(f"[DAEMON][{_ts()}][{cfg}] cycle error: {exc}")
                    traceback.print_exc()
                    _send_alert(f"DAEMON CYCLE ERROR: {cfg}", f"cycle={cycle} error={exc}")

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        print(f"\n[DAEMON] KeyboardInterrupt — stopping")
    finally:
        client.stop_tickle_loop()
        print("[DAEMON] stopped")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        print("\n[DAEMON] CRASHED")
        input("\nPress Enter to exit...")
        sys.exit(1)

    if _env_bool("PAUSE_ON_EXIT", False):
        input("\nPress Enter to exit...")
