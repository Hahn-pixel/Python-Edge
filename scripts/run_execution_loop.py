from __future__ import annotations

import json
import math
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
for p in [ROOT, SRC_DIR]:
    ps = str(p)
    if ps not in sys.path:
        sys.path.insert(0, ps)

# Double-click runnable. Never auto-close.
PAUSE_ON_EXIT = str(os.getenv("PAUSE_ON_EXIT", "auto")).strip().lower()

FEATURE_V2_FILE = Path(os.getenv("FEATURE_V2_FILE", "data/features/feature_matrix_v2.parquet"))
FREEZE_ROOT = Path(os.getenv("FREEZE_ROOT", "artifacts/freeze_runner"))
EXECUTION_ROOT = Path(os.getenv("EXECUTION_ROOT", "artifacts/execution_loop"))
ACCOUNT_NAV = float(os.getenv("ACCOUNT_NAV", "100000.0"))
ALLOW_FRACTIONAL_SHARES = str(os.getenv("ALLOW_FRACTIONAL_SHARES", "0")).strip().lower() not in {"0", "false", "no", "off"}
MIN_ORDER_NOTIONAL = float(os.getenv("MIN_ORDER_NOTIONAL", "25.0"))
DEFAULT_PRICE_FALLBACK = float(os.getenv("DEFAULT_PRICE_FALLBACK", "100.0"))
TOPK_PRINT = int(os.getenv("TOPK_PRINT", "50"))
RESET_STATE = str(os.getenv("RESET_STATE", "0")).strip().lower() not in {"0", "false", "no", "off"}
CONFIG_NAMES = [x.strip() for x in str(os.getenv("CONFIG_NAMES", "optimal|aggressive")).split("|") if x.strip()]

PRICE_COL_CANDIDATES = [
    "close",
    "adj_close",
    "Close",
    "close_px",
    "px_close",
    "c",
    "price_close",
    "close_price",
    "vwap_close",
]


@dataclass(frozen=True)
class ConfigPaths:
    name: str
    freeze_dir: Path
    execution_dir: Path
    current_book_csv: Path
    current_summary_json: Path


def _enable_line_buffering() -> None:
    for stream_name in ["stdout", "stderr"]:
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(line_buffering=True)
            except Exception:
                pass


def _should_pause() -> bool:
    if PAUSE_ON_EXIT in {"0", "false", "no", "off"}:
        return False
    if PAUSE_ON_EXIT in {"1", "true", "yes", "on"}:
        return True
    stdin_obj = getattr(sys, "stdin", None)
    stdout_obj = getattr(sys, "stdout", None)
    return bool(stdin_obj and stdout_obj and hasattr(stdin_obj, "isatty") and hasattr(stdout_obj, "isatty") and stdin_obj.isatty() and stdout_obj.isatty())


def _safe_exit(code: int) -> None:
    if _should_pause():
        try:
            print("")
            print(f"[EXIT] code={code}")
            input("Press Enter to exit...")
        except Exception:
            pass
    raise SystemExit(code)


def _must_exist(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("float64")


def _load_feature_frame() -> pd.DataFrame:
    _must_exist(FEATURE_V2_FILE, "Feature v2 file")
    df = pd.read_parquet(FEATURE_V2_FILE)
    if df.empty:
        raise RuntimeError("Feature v2 file is empty")
    if "date" not in df.columns or "symbol" not in df.columns:
        raise RuntimeError("Feature v2 file must contain date and symbol")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df.sort_values(["date", "symbol"]).reset_index(drop=True)


def _pick_price_col(df: pd.DataFrame) -> Tuple[str, bool]:
    for col in PRICE_COL_CANDIDATES:
        if col in df.columns:
            return col, True
    return "__fallback_price__", False


def _build_price_snapshot(df: pd.DataFrame) -> Tuple[pd.Timestamp, pd.DataFrame, str]:
    last_date = pd.Timestamp(df["date"].max()).normalize()
    snap = df.loc[df["date"] == last_date].copy().reset_index(drop=True)
    price_col, found = _pick_price_col(snap)
    if not found:
        snap[price_col] = float(DEFAULT_PRICE_FALLBACK)
    snap[price_col] = _num(snap[price_col]).fillna(float(DEFAULT_PRICE_FALLBACK))
    return last_date, snap[["symbol", price_col]].copy(), price_col


def _config_paths(name: str) -> ConfigPaths:
    freeze_dir = FREEZE_ROOT / name
    execution_dir = EXECUTION_ROOT / name
    return ConfigPaths(
        name=name,
        freeze_dir=freeze_dir,
        execution_dir=execution_dir,
        current_book_csv=freeze_dir / "freeze_current_book.csv",
        current_summary_json=freeze_dir / "freeze_current_summary.json",
    )


def _load_freeze_book(paths: ConfigPaths) -> Tuple[pd.DataFrame, dict]:
    _must_exist(paths.current_book_csv, f"Freeze current book for {paths.name}")
    _must_exist(paths.current_summary_json, f"Freeze current summary for {paths.name}")
    book = pd.read_csv(paths.current_book_csv)
    with paths.current_summary_json.open("r", encoding="utf-8") as fh:
        summary = json.load(fh)
    if book.empty:
        book = pd.DataFrame(columns=["symbol", "weight"])
    if "symbol" not in book.columns or "weight" not in book.columns:
        raise RuntimeError(f"Freeze current book for {paths.name} must contain symbol and weight")
    book["symbol"] = book["symbol"].astype(str)
    book["weight"] = _num(book["weight"]).fillna(0.0)
    return book, summary


def _state_paths(execution_dir: Path) -> Tuple[Path, Path, Path]:
    return (
        execution_dir / "portfolio_state.json",
        execution_dir / "orders.csv",
        execution_dir / "execution_log.csv",
    )


def _empty_state(config_name: str) -> dict:
    return {
        "config": config_name,
        "nav": float(ACCOUNT_NAV),
        "cash": float(ACCOUNT_NAV),
        "last_rebalanced_date": None,
        "positions": {},
    }


def _load_state(config_name: str, state_json: Path) -> dict:
    if RESET_STATE or not state_json.exists():
        return _empty_state(config_name)
    with state_json.open("r", encoding="utf-8") as fh:
        state = json.load(fh)
    if not isinstance(state, dict):
        raise RuntimeError(f"Invalid portfolio state json for {config_name}")
    state.setdefault("config", config_name)
    state.setdefault("nav", float(ACCOUNT_NAV))
    state.setdefault("cash", float(ACCOUNT_NAV))
    state.setdefault("last_rebalanced_date", None)
    state.setdefault("positions", {})
    if not isinstance(state["positions"], dict):
        raise RuntimeError(f"Positions must be a dict in state for {config_name}")
    return state


def _save_state(state: dict, state_json: Path) -> None:
    state_json.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _current_position_shares(state: dict, symbol: str) -> float:
    pos = state.get("positions", {}).get(symbol, {})
    try:
        return float(pos.get("shares", 0.0))
    except Exception:
        return 0.0


def _round_shares(target_shares: float) -> float:
    if ALLOW_FRACTIONAL_SHARES:
        return float(target_shares)
    if target_shares >= 0.0:
        return float(math.floor(target_shares + 1e-12))
    return float(math.ceil(target_shares - 1e-12))


def _build_target_table(book: pd.DataFrame, price_df: pd.DataFrame, price_col: str, nav: float) -> pd.DataFrame:
    target = book[["symbol", "weight"]].copy()
    target = target.merge(price_df, on="symbol", how="left")
    target[price_col] = _num(target[price_col]).fillna(float(DEFAULT_PRICE_FALLBACK))
    target["target_notional"] = _num(target["weight"]) * float(nav)
    target["target_shares_raw"] = target["target_notional"] / (_num(target[price_col]) + 1e-12)
    target["target_shares"] = target["target_shares_raw"].map(_round_shares)
    return target


def _build_orders(target: pd.DataFrame, state: dict, price_col: str, current_date: pd.Timestamp) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    symbols = sorted(set(target["symbol"].astype(str).tolist()) | set(state.get("positions", {}).keys()))
    target_map = {str(row["symbol"]): row for _, row in target.iterrows()}
    for symbol in symbols:
        row = target_map.get(symbol)
        price = float(row[price_col]) if row is not None else float(DEFAULT_PRICE_FALLBACK)
        target_weight = float(row["weight"]) if row is not None else 0.0
        target_notional = float(row["target_notional"]) if row is not None else 0.0
        target_shares = float(row["target_shares"]) if row is not None else 0.0
        current_shares = _current_position_shares(state, symbol)
        delta_shares = float(target_shares - current_shares)
        order_notional = float(delta_shares * price)
        order_side = "BUY" if delta_shares > 0 else "SELL" if delta_shares < 0 else "HOLD"
        skip_reason = ""
        if order_side != "HOLD" and abs(order_notional) < float(MIN_ORDER_NOTIONAL):
            skip_reason = "below_min_notional"
            delta_shares = 0.0
            order_notional = 0.0
            order_side = "HOLD"
        rows.append(
            {
                "date": str(current_date.date()),
                "symbol": symbol,
                "price": price,
                "target_weight": target_weight,
                "target_notional": target_notional,
                "current_shares": current_shares,
                "target_shares": target_shares,
                "delta_shares": delta_shares,
                "order_side": order_side,
                "order_notional": order_notional,
                "skip_reason": skip_reason,
            }
        )
    orders = pd.DataFrame(rows)
    if len(orders):
        orders = orders.sort_values(["order_side", "order_notional", "symbol"], ascending=[True, False, True]).reset_index(drop=True)
    return orders


def _apply_orders_to_state(state: dict, orders: pd.DataFrame, config_name: str, current_date: pd.Timestamp) -> dict:
    new_state = json.loads(json.dumps(state))
    positions = new_state.setdefault("positions", {})
    cash = float(new_state.get("cash", ACCOUNT_NAV))
    for _, row in orders.iterrows():
        symbol = str(row["symbol"])
        price = float(row["price"])
        delta_shares = float(row["delta_shares"])
        if abs(delta_shares) <= 0.0:
            continue
        current_shares = float(positions.get(symbol, {}).get("shares", 0.0))
        next_shares = float(current_shares + delta_shares)
        cash -= float(delta_shares * price)
        if abs(next_shares) < 1e-12:
            positions.pop(symbol, None)
        else:
            positions[symbol] = {
                "shares": next_shares,
                "last_price": price,
                "market_value": float(next_shares * price),
            }
    market_value = float(sum(float(v.get("market_value", 0.0)) for v in positions.values()))
    new_state["config"] = config_name
    new_state["cash"] = cash
    new_state["nav"] = float(cash + market_value)
    new_state["last_rebalanced_date"] = str(current_date.date())
    return new_state


def _append_execution_log(execution_log_csv: Path, log_row: Dict[str, object]) -> None:
    row_df = pd.DataFrame([log_row])
    if execution_log_csv.exists():
        prev = pd.read_csv(execution_log_csv)
        out = pd.concat([prev, row_df], ignore_index=True)
    else:
        out = row_df
    out.to_csv(execution_log_csv, index=False)


def _run_one_config(paths: ConfigPaths, price_df: pd.DataFrame, price_col: str, current_date: pd.Timestamp) -> None:
    print(f"[EXEC][{paths.name}] loading freeze artifacts")
    book, freeze_summary = _load_freeze_book(paths)
    paths.execution_dir.mkdir(parents=True, exist_ok=True)
    state_json, orders_csv, execution_log_csv = _state_paths(paths.execution_dir)
    state = _load_state(paths.name, state_json)

    target = _build_target_table(book, price_df, price_col, float(state.get("nav", ACCOUNT_NAV)))
    orders = _build_orders(target, state, price_col, current_date)
    next_state = _apply_orders_to_state(state, orders, paths.name, current_date)

    target.to_csv(paths.execution_dir / "target_book.csv", index=False)
    orders.to_csv(orders_csv, index=False)
    _save_state(next_state, state_json)

    live_active_names = int(len(target.loc[target["target_shares"].abs() > 0.0]))
    order_count = int((orders["order_side"] != "HOLD").sum()) if len(orders) else 0
    gross_target = float(target["target_notional"].abs().sum() / (float(state.get("nav", ACCOUNT_NAV)) + 1e-12)) if len(target) else 0.0

    log_row = {
        "date": str(current_date.date()),
        "config": paths.name,
        "starting_nav": float(state.get("nav", ACCOUNT_NAV)),
        "ending_nav": float(next_state.get("nav", ACCOUNT_NAV)),
        "cash_after": float(next_state.get("cash", ACCOUNT_NAV)),
        "live_active_names": live_active_names,
        "order_count": order_count,
        "gross_target": gross_target,
        "freeze_live_current_date": freeze_summary.get("live_current_date", ""),
        "freeze_replay_current_date": freeze_summary.get("replay_current_date", ""),
    }
    _append_execution_log(execution_log_csv, log_row)

    print(f"[EXEC][{paths.name}] starting_nav={float(state.get('nav', ACCOUNT_NAV)):.2f} ending_nav={float(next_state.get('nav', ACCOUNT_NAV)):.2f} order_count={order_count} active_names={live_active_names} gross_target={gross_target:.4f}")
    if len(orders):
        print(f"[EXEC][{paths.name}][ORDERS_TOP]")
        print(orders[["symbol", "order_side", "delta_shares", "price", "order_notional", "target_weight", "skip_reason"]].head(min(TOPK_PRINT, len(orders))).to_string(index=False))
    print(f"[ARTIFACT] {paths.execution_dir / 'target_book.csv'}")
    print(f"[ARTIFACT] {orders_csv}")
    print(f"[ARTIFACT] {state_json}")
    print(f"[ARTIFACT] {execution_log_csv}")


def main() -> int:
    _enable_line_buffering()
    print(f"[CFG] feature_v2_file={FEATURE_V2_FILE}")
    print(f"[CFG] freeze_root={FREEZE_ROOT}")
    print(f"[CFG] execution_root={EXECUTION_ROOT}")
    print(f"[CFG] account_nav={ACCOUNT_NAV}")
    print(f"[CFG] configs={CONFIG_NAMES}")
    print(f"[CFG] allow_fractional_shares={int(ALLOW_FRACTIONAL_SHARES)} min_order_notional={MIN_ORDER_NOTIONAL}")

    feature_df = _load_feature_frame()
    current_date, price_df, price_col = _build_price_snapshot(feature_df)
    print(f"[DATA] current_date={current_date.date()} price_col={price_col} symbols={len(price_df)}")

    for name in CONFIG_NAMES:
        paths = _config_paths(name)
        _run_one_config(paths, price_df, price_col, current_date)

    print("[FINAL] execution loop complete")
    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 1
    _safe_exit(rc)
