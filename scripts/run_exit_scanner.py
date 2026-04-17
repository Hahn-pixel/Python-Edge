"""
run_exit_scanner.py — Exit Strategy Scanner

Запускається між Step 2 (execution_loop) і Step 3 (cpapi_handoff) pipeline.

Що робить:
  1. Читає portfolio_state.json per config
  2. Завантажує ivol_20d зі snapshot (live_feature_snapshot.parquet)
  3. Отримує поточні ціни з CPAPI snapshot (або fallback з state)
  4. Оновлює peak_price і кешує ivol_20d в exit_meta (portfolio_state.json)
  5. Оцінює exit-умови через ExitPolicy (volatility-scaled trailing)
  6. Якщо exit спрацював — додає SELL/BUY в orders.csv

ENV-змінні:
  EXECUTION_ROOT              — default: artifacts/execution_loop
  LIVE_FEATURE_SNAPSHOT_FILE  — default: artifacts/live_alpha/live_feature_snapshot.parquet
  CONFIG_NAMES                — default: optimal|aggressive
  CPAPI_BASE_URL              — default: https://localhost:5000
  CPAPI_VERIFY_SSL            — 0/1
  CPAPI_TIMEOUT_SEC           — default: 10.0
  EXIT_SCANNER_DRY_RUN        — 1 = тільки лог, не модифікувати файли
  EXIT_SCANNER_USE_STATE_PRICE— 1 = не ходити в CPAPI, брати last_price зі state
  PAUSE_ON_EXIT               — 0/1
  TODAY_OVERRIDE              — YYYY-MM-DD (для тестування)

  + Exit policy params (ExitPolicy.from_env):
  EXIT_OPTIMAL_STOP_LOSS_PCT, EXIT_OPTIMAL_TRAIL_K, etc.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import date
from pathlib import Path

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
    if v in ("1", "true", "yes", "on"):  return True
    if v in ("0", "false", "no", "off"): return False
    return default

def _today_str() -> str:
    override = _env("TODAY_OVERRIDE")
    return override[:10] if override else date.today().isoformat()


# ──────────────────────────────────────────────────────────────
# ivol_20d зі snapshot
# ──────────────────────────────────────────────────────────────
def _load_ivol_from_snapshot(snapshot_path: Path) -> dict[str, float]:
    """
    Повертає {symbol: ivol_20d} з live_feature_snapshot.parquet.
    Бере останній рядок per symbol (найсвіжіше значення).
    """
    ivol: dict[str, float] = {}
    if not snapshot_path.exists():
        print(f"[EXIT_SCANNER] snapshot not found: {snapshot_path}")
        return ivol
    try:
        df = pd.read_parquet(snapshot_path, columns=["symbol", "ivol_20d"])
        if "symbol" not in df.columns or "ivol_20d" not in df.columns:
            print("[EXIT_SCANNER] snapshot missing symbol/ivol_20d columns")
            return ivol
        df = df.dropna(subset=["ivol_20d"])
        df["ivol_20d"] = pd.to_numeric(df["ivol_20d"], errors="coerce")
        df = df.dropna(subset=["ivol_20d"])
        # Останній рядок per symbol
        latest = df.groupby("symbol")["ivol_20d"].last()
        ivol = {str(s).upper(): float(v) for s, v in latest.items() if v > 0}
        print(f"[EXIT_SCANNER] ivol loaded: {len(ivol)} symbols from snapshot")
    except Exception as exc:
        print(f"[EXIT_SCANNER] snapshot read error: {exc}")
    return ivol


# ──────────────────────────────────────────────────────────────
# CPAPI price fetch
# ──────────────────────────────────────────────────────────────
def _fetch_cpapi_prices(
    symbols: list[str],
    conid_cache: dict[str, int],
    base_url: str,
    verify_ssl: bool,
    timeout: float,
) -> dict[str, float]:
    prices: dict[str, float] = {}
    if not symbols:
        return prices
    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        import requests

        sym_conid = {s: conid_cache[s] for s in symbols if s in conid_cache}
        if not sym_conid:
            return prices

        conids_str = ",".join(str(c) for c in sym_conid.values())
        url = f"{base_url}/v1/api/iserver/marketdata/snapshot"
        resp = requests.get(
            url,
            params={"conids": conids_str, "fields": "31,84,86"},
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
            sym   = conid_to_sym.get(conid)
            if not sym:
                continue
            try:
                bid  = item.get("84")
                ask  = item.get("86")
                last = item.get("31")
                if bid and ask:
                    prices[sym] = (
                        float(str(bid).replace(",", "")) +
                        float(str(ask).replace(",", ""))
                    ) / 2.0
                elif last:
                    prices[sym] = float(str(last).replace(",", ""))
            except (ValueError, TypeError):
                pass
    except Exception as exc:
        print(f"[EXIT_SCANNER] CPAPI price fetch error: {exc} — using state prices")
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
        print(f"[EXIT_SCANNER] Cannot read {path}: {exc}")
        return {}

def _save_state(path: Path, state: dict) -> None:
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")

def _load_conid_cache(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {k: int(v) for k, v in raw.items() if isinstance(v, (int, str)) and str(v).isdigit()}
    except Exception:
        return {}

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

def _save_orders(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False)

def _days_between(date_str: str, today_str: str) -> int:
    try:
        d1 = date.fromisoformat(date_str[:10])
        d2 = date.fromisoformat(today_str[:10])
        return max(0, (d2 - d1).days)
    except (ValueError, TypeError):
        return 0


# ──────────────────────────────────────────────────────────────
# Головна логіка per-config
# ──────────────────────────────────────────────────────────────
def _process_config(
    cfg: str,
    exec_root: Path,
    snapshot_ivol: dict[str, float],
    base_url: str,
    verify_ssl: bool,
    timeout: float,
    use_state_price: bool,
    dry_run: bool,
    today: str,
) -> dict:
    counters = {
        "config":                cfg,
        "positions_checked":     0,
        "exit_triggered":        0,
        "already_in_orders":     0,
        "skipped_no_price":      0,
        "skipped_no_entry_price":0,
        "peak_price_updated":    0,
        "ivol_from_snapshot":    0,
        "ivol_from_state":       0,
        "ivol_missing":          0,
        "orders_appended":       0,
    }

    cfg_dir     = exec_root / cfg
    state_path  = cfg_dir / "portfolio_state.json"
    orders_path = cfg_dir / "orders.csv"
    conid_path  = cfg_dir / "conid_cache.json"

    state = _load_state(state_path)
    if not state:
        print(f"[EXIT_SCANNER][{cfg}] portfolio_state.json not found — skip")
        return counters

    positions: dict = state.get("positions", {})
    if not positions:
        print(f"[EXIT_SCANNER][{cfg}] no positions — skip")
        return counters

    exit_meta: dict = state.setdefault("exit_meta", {})
    policy = ExitPolicy.from_env(cfg)
    print(f"[EXIT_SCANNER][{cfg}] policy={policy}")

    # ── Поточні ціни ──────────────────────────────────────────
    symbols = list(positions.keys())
    live_prices: dict[str, float] = {}
    if not use_state_price:
        conid_cache = _load_conid_cache(conid_path)
        live_prices = _fetch_cpapi_prices(symbols, conid_cache, base_url, verify_ssl, timeout)
        print(f"[EXIT_SCANNER][{cfg}] live prices: {len(live_prices)}/{len(symbols)}")
    else:
        print(f"[EXIT_SCANNER][{cfg}] USE_STATE_PRICE=1 — skipping CPAPI")

    # ── Існуючі exit-ордери ───────────────────────────────────
    orders_df = _load_orders(orders_path)
    existing_exit_symbols: set[str] = set(
        orders_df.loc[
            orders_df.get("exit_reason", pd.Series(dtype=str))
                     .astype(str).str.len() > 0,
            "symbol",
        ].astype(str).str.upper().tolist()
    )

    new_exit_rows: list[dict] = []

    # ── Оцінити кожну позицію ──────────────────────────────────
    for symbol, pos in positions.items():
        counters["positions_checked"] += 1
        shares = float(pos.get("shares", 0))
        if shares == 0:
            continue

        side         = 1.0 if shares > 0 else -1.0
        state_price  = float(pos.get("last_price", 0))
        current_price = live_prices.get(symbol, 0.0)
        if current_price <= 0:
            current_price = state_price
        if current_price <= 0:
            counters["skipped_no_price"] += 1
            print(f"[EXIT_SCANNER][{cfg}] {symbol} — no price, skip")
            continue

        meta        = exit_meta.setdefault(symbol, {})
        entry_price = float(meta.get("entry_price") or 0)
        entry_date  = str(meta.get("entry_date") or today)
        peak_price  = float(meta.get("peak_price") or 0)

        # ── ivol_20d: snapshot → state cache → None ───────────
        ivol_20d: float | None = None
        if symbol.upper() in snapshot_ivol:
            ivol_20d = snapshot_ivol[symbol.upper()]
            meta["ivol_20d_cached"] = ivol_20d   # кешуємо для daemon
            counters["ivol_from_snapshot"] += 1
        elif meta.get("ivol_20d_cached"):
            ivol_20d = float(meta["ivol_20d_cached"])
            counters["ivol_from_state"] += 1
            print(f"[EXIT_SCANNER][{cfg}] {symbol} ivol from state cache={ivol_20d:.4f}")
        else:
            counters["ivol_missing"] += 1
            print(f"[EXIT_SCANNER][{cfg}] {symbol} ivol missing — trailing disabled")

        # ── Ініціалізація entry_price ─────────────────────────
        if entry_price <= 0:
            meta["entry_price"] = current_price
            meta["entry_date"]  = today
            entry_price = current_price
            entry_date  = today
            counters["skipped_no_entry_price"] += 1
            print(
                f"[EXIT_SCANNER][{cfg}] {symbol} — entry initialized={current_price:.4f} "
                f"{policy.describe_trail(ivol_20d)}"
            )
            # Оновити peak і продовжити (не тригеримо в перший цикл)
            if side > 0:
                if current_price > peak_price:
                    meta["peak_price"] = current_price
                    counters["peak_price_updated"] += 1
            else:
                if peak_price <= 0 or current_price < peak_price:
                    meta["peak_price"] = current_price
                    counters["peak_price_updated"] += 1
            continue

        # ── Оновити peak_price ────────────────────────────────
        if side > 0:
            if current_price > peak_price:
                meta["peak_price"] = current_price
                peak_price = current_price
                counters["peak_price_updated"] += 1
        else:
            if peak_price <= 0 or current_price < peak_price:
                meta["peak_price"] = current_price
                peak_price = current_price
                counters["peak_price_updated"] += 1

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

        pnl_pct   = (current_price - entry_price) / entry_price * side
        hold_days = _days_between(entry_date, today)
        trail_desc = policy.describe_trail(ivol_20d)

        if reason:
            counters["exit_triggered"] += 1
            if symbol.upper() in existing_exit_symbols:
                counters["already_in_orders"] += 1
                print(
                    f"[EXIT_SCANNER][{cfg}] {symbol} exit={reason} "
                    f"pnl={pnl_pct:+.2%} hold={hold_days}d — already in orders"
                )
                continue

            exit_side  = "SELL" if side > 0 else "BUY"
            abs_shares = abs(shares)

            new_exit_rows.append({
                "symbol":        symbol,
                "order_side":    exit_side,
                "delta_shares":  abs_shares,
                "target_shares": 0.0,
                "price":         round(current_price, 4),
                "price_source":  "exit_scanner",
                "exit_reason":   reason,
            })
            counters["orders_appended"] += 1
            print(
                f"[EXIT_SCANNER][{cfg}] EXIT {symbol} {exit_side} {abs_shares:.0f}sh "
                f"reason={reason} pnl={pnl_pct:+.2%} hold={hold_days}d "
                f"entry={entry_price:.4f} cur={current_price:.4f} peak={peak_price:.4f} "
                f"{trail_desc}"
            )
        else:
            print(
                f"[EXIT_SCANNER][{cfg}] {symbol} OK "
                f"pnl={pnl_pct:+.2%} hold={hold_days}d {trail_desc}"
            )

    # ── Зберегти state ────────────────────────────────────────
    if not dry_run:
        _save_state(state_path, state)
        print(f"[EXIT_SCANNER][{cfg}] portfolio_state.json updated")

    # ── Додати exit-ордери ────────────────────────────────────
    if new_exit_rows and not dry_run:
        exit_df  = pd.DataFrame(new_exit_rows)
        combined = pd.concat([orders_df, exit_df], ignore_index=True)
        _save_orders(orders_path, combined)
        print(f"[EXIT_SCANNER][{cfg}] orders.csv +{len(new_exit_rows)} exit rows")
    elif new_exit_rows and dry_run:
        print(f"[EXIT_SCANNER][{cfg}] DRY_RUN: would append {len(new_exit_rows)} exit rows")

    return counters


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────
def main() -> None:
    exec_root    = Path(_env("EXECUTION_ROOT", "artifacts/execution_loop"))
    snapshot_path = Path(_env(
        "LIVE_FEATURE_SNAPSHOT_FILE",
        "artifacts/live_alpha/live_feature_snapshot.parquet",
    ))
    configs      = [c for c in _env("CONFIG_NAMES", "optimal|aggressive").split("|") if c]
    base_url     = _env("CPAPI_BASE_URL", "https://localhost:5000")
    verify_ssl   = not _env_bool("CPAPI_VERIFY_SSL", False)
    timeout      = float(_env("CPAPI_TIMEOUT_SEC", "10.0"))
    use_state_price = _env_bool("EXIT_SCANNER_USE_STATE_PRICE", False)
    dry_run      = _env_bool("EXIT_SCANNER_DRY_RUN", False)
    today        = _today_str()

    print(f"[EXIT_SCANNER] start date={today} configs={configs} dry_run={dry_run}")

    # Завантажити ivol один раз для всіх configs
    snapshot_ivol = _load_ivol_from_snapshot(snapshot_path)

    all_counters: list[dict] = []
    total_exit = 0

    for cfg in configs:
        counters = _process_config(
            cfg=cfg,
            exec_root=exec_root,
            snapshot_ivol=snapshot_ivol,
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
            f"ivol_snap={c['ivol_from_snapshot']} "
            f"ivol_state={c['ivol_from_state']} "
            f"ivol_missing={c['ivol_missing']} "
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

    if _env_bool("PAUSE_ON_EXIT", False):
        input("\nPress Enter to exit...")
