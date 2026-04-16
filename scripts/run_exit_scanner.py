"""
run_exit_scanner.py — Exit Strategy Scanner

Запускається між Step 2 (execution_loop) і Step 3 (cpapi_handoff) pipeline.

Що робить:
  1. Читає portfolio_state.json per config
  2. Отримує поточні ціни з cpapi snapshot (або fallback з state)
  3. Оновлює peak_price per position (зберігає в portfolio_state.json)
  4. Оцінює exit-умови через ExitPolicy
  5. Якщо exit спрацював — додає SELL (або BUY для шорту) в orders.csv
  6. Виводить debug-лічильники: checked / exit_triggered / already_in_orders / skipped_no_price

ENV-змінні:
  EXECUTION_ROOT        — шлях до artifacts/execution_loop (default: artifacts/execution_loop)
  CONFIG_NAMES          — pipe-separated список (default: optimal|aggressive)
  CPAPI_BASE_URL        — для отримання live цін (default: https://localhost:5000)
  CPAPI_VERIFY_SSL      — 0/1
  CPAPI_TIMEOUT_SEC     — таймаут
  EXIT_SCANNER_DRY_RUN  — 1 = тільки лог, не модифікувати orders.csv
  EXIT_SCANNER_USE_STATE_PRICE — 1 = не ходити в CPAPI, брати last_price зі state
  PAUSE_ON_EXIT         — 0/1
  TODAY_OVERRIDE        — YYYY-MM-DD (для тестування)

  + Exit policy params per config (читає ExitPolicy.from_env):
  EXIT_OPTIMAL_STOP_LOSS_PCT, EXIT_OPTIMAL_TAKE_PROFIT_PCT, etc.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import date, datetime
from pathlib import Path

# ── sys.path setup ────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
for _p in [str(_ROOT), str(_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd

from python_edge.portfolio.exit_policy import ExitPolicy

# ──────────────────────────────────────────────────────────────
# Env helpers
# ──────────────────────────────────────────────────────────────
def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()

def _env_bool(key: str, default: bool = False) -> bool:
    v = os.getenv(key, "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return default

def _today_str() -> str:
    override = _env("TODAY_OVERRIDE")
    if override:
        return override[:10]
    return date.today().isoformat()


# ──────────────────────────────────────────────────────────────
# CPAPI price fetch (best effort)
# ──────────────────────────────────────────────────────────────
def _fetch_cpapi_prices(
    symbols: list[str],
    conid_cache: dict[str, int],
    base_url: str,
    verify_ssl: bool,
    timeout: float,
) -> dict[str, float]:
    """
    Повертає {symbol: mid_price} для символів що є в conid_cache.
    При будь-якій помилці — повертає порожній dict (caller використає fallback).
    """
    prices: dict[str, float] = {}
    if not symbols:
        return prices
    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        import requests

        # Тільки символи що є в кеші
        sym_conid = {s: conid_cache[s] for s in symbols if s in conid_cache}
        if not sym_conid:
            return prices

        conids_str = ",".join(str(c) for c in sym_conid.values())
        fields = "31,84,86"  # last, bid, ask
        url = f"{base_url}/v1/api/iserver/marketdata/snapshot"
        resp = requests.get(
            url,
            params={"conids": conids_str, "fields": fields},
            verify=verify_ssl,
            timeout=timeout,
        )
        if resp.status_code != 200:
            print(f"[EXIT_SCANNER] CPAPI snapshot HTTP {resp.status_code} — using state prices")
            return prices

        data = resp.json()
        if not isinstance(data, list):
            return prices

        conid_to_sym = {v: k for k, v in sym_conid.items()}
        for item in data:
            conid = item.get("conid")
            symbol = conid_to_sym.get(conid)
            if not symbol:
                continue
            bid = item.get("84")
            ask = item.get("86")
            last = item.get("31")
            try:
                if bid and ask:
                    prices[symbol] = (float(str(bid).replace(",", "")) + float(str(ask).replace(",", ""))) / 2.0
                elif last:
                    prices[symbol] = float(str(last).replace(",", ""))
            except (ValueError, TypeError):
                pass

    except Exception as exc:
        print(f"[EXIT_SCANNER] CPAPI price fetch error: {exc} — using state prices")

    return prices


# ──────────────────────────────────────────────────────────────
# portfolio_state helpers
# ──────────────────────────────────────────────────────────────
def _load_state(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[EXIT_SCANNER] Cannot read {path}: {exc}")
        return {}


def _save_state(path: Path, state: dict) -> None:
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_conid_cache(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {k: int(v) for k, v in raw.items() if str(v).isdigit() or isinstance(v, int)}
    except Exception:
        return {}


# ──────────────────────────────────────────────────────────────
# orders.csv helpers
# ──────────────────────────────────────────────────────────────
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
    except Exception as exc:
        print(f"[EXIT_SCANNER] Cannot read orders {path}: {exc}")
        return pd.DataFrame(columns=_ORDERS_COLUMNS)


def _save_orders(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False)


# ──────────────────────────────────────────────────────────────
# Головна логіка per-config
# ──────────────────────────────────────────────────────────────
def _process_config(
    cfg: str,
    exec_root: Path,
    base_url: str,
    verify_ssl: bool,
    timeout: float,
    use_state_price: bool,
    dry_run: bool,
    today: str,
) -> dict:
    """
    Повертає debug-лічильники для одного config.
    """
    counters = {
        "config": cfg,
        "positions_checked": 0,
        "exit_triggered": 0,
        "already_in_orders": 0,
        "skipped_no_price": 0,
        "skipped_no_entry_price": 0,
        "peak_price_updated": 0,
        "orders_appended": 0,
    }

    cfg_dir = exec_root / cfg
    state_path  = cfg_dir / "portfolio_state.json"
    orders_path = cfg_dir / "orders.csv"
    conid_path  = cfg_dir / "conid_cache.json"

    state = _load_state(state_path)
    if not state:
        print(f"[EXIT_SCANNER][{cfg}] portfolio_state.json not found or empty — skip")
        return counters

    positions: dict = state.get("positions", {})
    if not positions:
        print(f"[EXIT_SCANNER][{cfg}] no positions — skip")
        return counters

    # Зберігаємо / ініціалізуємо exit_meta per position
    exit_meta: dict = state.setdefault("exit_meta", {})

    policy = ExitPolicy.from_env(cfg)
    print(f"[EXIT_SCANNER][{cfg}] policy={policy}")

    # ── Отримати поточні ціни ──────────────────────────────────
    symbols = list(positions.keys())
    live_prices: dict[str, float] = {}

    if not use_state_price:
        conid_cache = _load_conid_cache(conid_path)
        live_prices = _fetch_cpapi_prices(symbols, conid_cache, base_url, verify_ssl, timeout)
        print(f"[EXIT_SCANNER][{cfg}] live prices fetched: {len(live_prices)}/{len(symbols)}")
    else:
        print(f"[EXIT_SCANNER][{cfg}] USE_STATE_PRICE=1 — skipping CPAPI")

    # ── Завантажити orders.csv ─────────────────────────────────
    orders_df = _load_orders(orders_path)
    existing_exit_symbols: set[str] = set(
        orders_df.loc[
            orders_df.get("exit_reason", pd.Series(dtype=str)).astype(str).str.len() > 0,
            "symbol"
        ].astype(str).str.upper().tolist()
    )

    new_exit_rows: list[dict] = []

    # ── Оцінити кожну позицію ──────────────────────────────────
    for symbol, pos in positions.items():
        counters["positions_checked"] += 1
        shares = float(pos.get("shares", 0))
        if shares == 0:
            continue

        side = 1.0 if shares > 0 else -1.0
        state_price = float(pos.get("last_price", 0))

        # Поточна ціна: live → fallback state
        current_price = live_prices.get(symbol, 0.0)
        if current_price <= 0:
            current_price = state_price
        if current_price <= 0:
            counters["skipped_no_price"] += 1
            print(f"[EXIT_SCANNER][{cfg}] {symbol} — no price, skip")
            continue

        # Ініціалізація exit_meta якщо нема
        meta = exit_meta.setdefault(symbol, {})
        entry_price = float(meta.get("entry_price", 0.0))
        entry_date  = str(meta.get("entry_date", today))
        peak_price  = float(meta.get("peak_price", 0.0))

        # Якщо entry_price не заповнено — ініціалізуємо поточною ціною
        # (позиція існувала до впровадження exit scanner)
        if entry_price <= 0:
            meta["entry_price"] = current_price
            meta["entry_date"]  = today
            entry_price = current_price
            entry_date  = today
            counters["skipped_no_entry_price"] += 1
            print(
                f"[EXIT_SCANNER][{cfg}] {symbol} — entry_price initialized={current_price:.4f} "
                f"(existing position, no exit this cycle)"
            )
            # Не тригеримо exit у цьому циклі — щойно ініціалізували
            # Але оновлюємо peak_price
            if peak_price <= 0 or (side > 0 and current_price > peak_price) or (side < 0 and (peak_price == 0 or current_price < peak_price)):
                meta["peak_price"] = current_price
                counters["peak_price_updated"] += 1
            continue

        # Оновити peak_price
        if side > 0:
            if current_price > peak_price:
                meta["peak_price"] = current_price
                peak_price = current_price
                counters["peak_price_updated"] += 1
        else:
            # short: peak = найнижча ціна
            if peak_price <= 0 or current_price < peak_price:
                meta["peak_price"] = current_price
                peak_price = current_price
                counters["peak_price_updated"] += 1

        # Перевірити exit
        reason = policy.evaluate(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            current_price=current_price,
            peak_price=peak_price,
            entry_date=entry_date,
            today=today,
        )

        pnl_pct = (current_price - entry_price) / entry_price * side
        hold_days = _days_between(entry_date, today)

        if reason:
            counters["exit_triggered"] += 1
            if symbol.upper() in existing_exit_symbols:
                counters["already_in_orders"] += 1
                print(
                    f"[EXIT_SCANNER][{cfg}] {symbol} exit={reason} "
                    f"pnl={pnl_pct:+.2%} hold={hold_days}d — already in orders, skip"
                )
                continue

            # Визначити order_side: закрити позицію
            exit_side = "SELL" if side > 0 else "BUY"
            abs_shares = abs(shares)
            exit_price = current_price

            new_exit_rows.append({
                "symbol":       symbol,
                "order_side":   exit_side,
                "delta_shares": abs_shares,
                "target_shares": 0.0,
                "price":        round(exit_price, 4),
                "price_source": "exit_scanner",
                "exit_reason":  reason,
            })
            counters["orders_appended"] += 1
            print(
                f"[EXIT_SCANNER][{cfg}] EXIT {symbol} {exit_side} {abs_shares:.0f}sh "
                f"reason={reason} pnl={pnl_pct:+.2%} hold={hold_days}d "
                f"entry={entry_price:.4f} cur={current_price:.4f} peak={peak_price:.4f}"
            )
        else:
            print(
                f"[EXIT_SCANNER][{cfg}] {symbol} OK "
                f"pnl={pnl_pct:+.2%} hold={hold_days}d peak={peak_price:.4f}"
            )

    # ── Зберегти оновлений state (peak_price + exit_meta) ──────
    if not dry_run:
        _save_state(state_path, state)
        print(f"[EXIT_SCANNER][{cfg}] portfolio_state.json updated (peak_price/exit_meta)")

    # ── Додати exit-ордери в orders.csv ────────────────────────
    if new_exit_rows and not dry_run:
        exit_df = pd.DataFrame(new_exit_rows)
        # Додати відсутні колонки в існуючий orders_df
        for col in _ORDERS_COLUMNS:
            if col not in orders_df.columns:
                orders_df[col] = ""
        combined = pd.concat([orders_df, exit_df], ignore_index=True)
        _save_orders(orders_path, combined)
        print(f"[EXIT_SCANNER][{cfg}] orders.csv updated: +{len(new_exit_rows)} exit rows")
    elif new_exit_rows and dry_run:
        print(f"[EXIT_SCANNER][{cfg}] DRY_RUN: would append {len(new_exit_rows)} exit rows")

    return counters


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────
def _days_between(date_str: str, today_str: str) -> int:
    try:
        d1 = date.fromisoformat(date_str[:10])
        d2 = date.fromisoformat(today_str[:10])
        return max(0, (d2 - d1).days)
    except (ValueError, TypeError):
        return 0


def main() -> None:
    exec_root   = Path(_env("EXECUTION_ROOT", "artifacts/execution_loop"))
    configs     = [c for c in _env("CONFIG_NAMES", "optimal|aggressive").split("|") if c]
    base_url    = _env("CPAPI_BASE_URL", "https://localhost:5000")
    verify_ssl  = not _env_bool("CPAPI_VERIFY_SSL", False)   # 0 → не верифікувати
    timeout     = float(_env("CPAPI_TIMEOUT_SEC", "10.0"))
    use_state_price = _env_bool("EXIT_SCANNER_USE_STATE_PRICE", False)
    dry_run     = _env_bool("EXIT_SCANNER_DRY_RUN", False)
    today       = _today_str()

    print(f"[EXIT_SCANNER] start date={today} configs={configs} dry_run={dry_run}")

    all_counters: list[dict] = []
    total_exit = 0

    for cfg in configs:
        counters = _process_config(
            cfg=cfg,
            exec_root=exec_root,
            base_url=base_url,
            verify_ssl=verify_ssl,
            timeout=timeout,
            use_state_price=use_state_price,
            dry_run=dry_run,
            today=today,
        )
        all_counters.append(counters)
        total_exit += counters["exit_triggered"]

    # ── Підсумок ───────────────────────────────────────────────
    print()
    print("[EXIT_SCANNER] ── SUMMARY ──────────────────────────────────")
    for c in all_counters:
        print(
            f"  [{c['config']}] "
            f"checked={c['positions_checked']} "
            f"exit_triggered={c['exit_triggered']} "
            f"already_in_orders={c['already_in_orders']} "
            f"skipped_no_price={c['skipped_no_price']} "
            f"skipped_no_entry={c['skipped_no_entry_price']} "
            f"peak_updated={c['peak_price_updated']} "
            f"orders_appended={c['orders_appended']}"
        )
    print(f"[EXIT_SCANNER] total exits this cycle: {total_exit}")

    if dry_run:
        print("[EXIT_SCANNER] DRY_RUN mode — no files modified")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        print("\n[EXIT_SCANNER] FAILED")
        input("\nPress Enter to exit...")
        sys.exit(1)

    pause = _env_bool("PAUSE_ON_EXIT", False)
    if pause:
        input("\nPress Enter to exit...")
