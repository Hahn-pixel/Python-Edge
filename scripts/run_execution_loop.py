from __future__ import annotations

import json
import math
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
for p in [ROOT, SRC_DIR]:
    ps = str(p)
    if ps not in sys.path:
        sys.path.insert(0, ps)

PAUSE_ON_EXIT = str(os.getenv("PAUSE_ON_EXIT", "auto")).strip().lower()
LIVE_FEATURE_SNAPSHOT_FILE = Path(os.getenv("LIVE_FEATURE_SNAPSHOT_FILE", "artifacts/live_alpha/live_feature_snapshot.parquet"))
FREEZE_ROOT = Path(os.getenv("FREEZE_ROOT", "artifacts/freeze_runner"))
EXECUTION_ROOT = Path(os.getenv("EXECUTION_ROOT", "artifacts/execution_loop"))
ACCOUNT_NAV = float(os.getenv("ACCOUNT_NAV", "100000.0"))
ALLOW_FRACTIONAL_SHARES = str(os.getenv("ALLOW_FRACTIONAL_SHARES", "0")).strip().lower() not in {"0", "false", "no", "off"}
FRACTIONAL_MODE = str(os.getenv("FRACTIONAL_MODE", "auto")).strip().lower()
MIN_ORDER_NOTIONAL = float(os.getenv("MIN_ORDER_NOTIONAL", "25.0"))
DEFAULT_PRICE_FALLBACK = float(os.getenv("DEFAULT_PRICE_FALLBACK", "100.0"))
TOPK_PRINT = int(os.getenv("TOPK_PRINT", "50"))
RESET_STATE = str(os.getenv("RESET_STATE", "0")).strip().lower() not in {"0", "false", "no", "off"}
CONFIG_NAMES = [x.strip() for x in str(os.getenv("CONFIG_NAMES", "optimal|aggressive")).split("|") if x.strip()]
SKIP_EMPTY_FREEZE_CONFIGS = str(os.getenv("SKIP_EMPTY_FREEZE_CONFIGS", "1")).strip().lower() not in {"0", "false", "no", "off"}
REQUIRE_FREEZE_DATE_MATCH_LIVE_PRICES = str(os.getenv("REQUIRE_FREEZE_DATE_MATCH_LIVE_PRICES", "1")).strip().lower() not in {"0", "false", "no", "off"}
MAX_SINGLE_NAME_WEIGHT = float(os.getenv("MAX_SINGLE_NAME_WEIGHT", "0.05"))
MAX_SINGLE_NAME_NOTIONAL = float(os.getenv("MAX_SINGLE_NAME_NOTIONAL", "10000.0"))
MIN_PRICE_TO_TRADE = float(os.getenv("MIN_PRICE_TO_TRADE", "1.0"))
MAX_PRICE_TO_TRADE = float(os.getenv("MAX_PRICE_TO_TRADE", "1000000.0"))
COMMISSION_BPS = float(os.getenv("COMMISSION_BPS", "0.50"))
COMMISSION_MIN_PER_ORDER = float(os.getenv("COMMISSION_MIN_PER_ORDER", "0.35"))
SLIPPAGE_BPS = float(os.getenv("SLIPPAGE_BPS", "1.50"))
ALIGN_STATE_FROM_BROKER_ONCE = str(os.getenv("ALIGN_STATE_FROM_BROKER_ONCE", "0")).strip().lower() not in {"0", "false", "no", "off"}
ALIGN_STATE_REQUIRE_BROKER_POSITIONS = str(os.getenv("ALIGN_STATE_REQUIRE_BROKER_POSITIONS", "1")).strip().lower() not in {"0", "false", "no", "off"}
ALIGN_STATE_CASH_MODE = str(os.getenv("ALIGN_STATE_CASH_MODE", "preserve_nav")).strip().lower() or "preserve_nav"
SKIP_LEGACY_SYMBOLS_NOT_IN_TARGET = str(os.getenv("SKIP_LEGACY_SYMBOLS_NOT_IN_TARGET", "1")).strip().lower() not in {"0", "false", "no", "off"}

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


def _fractional_enabled_effective() -> bool:
    if FRACTIONAL_MODE == "fractional":
        return True
    if FRACTIONAL_MODE == "integer":
        return False
    return bool(ALLOW_FRACTIONAL_SHARES)


def _estimate_commission(order_notional_abs: float) -> float:
    if order_notional_abs <= 0.0:
        return 0.0
    pct_fee = float(COMMISSION_BPS) / 10000.0 * float(order_notional_abs)
    return float(max(float(COMMISSION_MIN_PER_ORDER), pct_fee))


def _estimate_slippage(order_notional_abs: float) -> float:
    if order_notional_abs <= 0.0:
        return 0.0
    return float(SLIPPAGE_BPS) / 10000.0 * float(order_notional_abs)


def _load_live_feature_frame() -> pd.DataFrame:
    _must_exist(LIVE_FEATURE_SNAPSHOT_FILE, "Live feature snapshot")
    df = pd.read_parquet(LIVE_FEATURE_SNAPSHOT_FILE)
    if df.empty:
        raise RuntimeError("Live feature snapshot is empty")
    if "date" not in df.columns or "symbol" not in df.columns:
        raise RuntimeError("Live feature snapshot must contain date and symbol")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["symbol"] = df["symbol"].astype(str).str.upper()
    return df.sort_values(["date", "symbol"]).reset_index(drop=True)


def _pick_price_col(df: pd.DataFrame) -> Tuple[str, bool]:
    for col in PRICE_COL_CANDIDATES:
        if col in df.columns:
            return col, True
    return "__fallback_price__", False


def _build_price_snapshot(df: pd.DataFrame) -> Tuple[pd.Timestamp, pd.DataFrame, str, int]:
    last_date = pd.Timestamp(df["date"].max()).normalize()
    snap = df.loc[df["date"] == last_date].copy().reset_index(drop=True)
    price_col, found = _pick_price_col(snap)
    fallback_count = 0
    if not found:
        snap[price_col] = float(DEFAULT_PRICE_FALLBACK)
        fallback_count = int(len(snap))
    else:
        price_num = _num(snap[price_col])
        fallback_mask = price_num.isna() | (price_num <= 0.0)
        fallback_count = int(fallback_mask.sum())
        price_num = price_num.mask(fallback_mask, float(DEFAULT_PRICE_FALLBACK))
        snap[price_col] = price_num
    return last_date, snap[["symbol", price_col]].copy(), price_col, fallback_count


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
    book["symbol"] = book["symbol"].astype(str).str.upper()
    book["weight"] = _num(book["weight"]).fillna(0.0)
    return book, summary


def _state_paths(execution_dir: Path) -> Tuple[Path, Path, Path, Path, Path, Path]:
    return (
        execution_dir / "portfolio_state.json",
        execution_dir / "orders.csv",
        execution_dir / "execution_log.csv",
        execution_dir / "positions_mark_to_market.csv",
        execution_dir / "broker_positions.csv",
        execution_dir / "state_alignment_once.json",
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


def _round_shares(target_shares: float, fractional_enabled: bool) -> float:
    if fractional_enabled:
        return float(target_shares)
    if target_shares >= 0.0:
        return float(math.floor(target_shares + 1e-12))
    return float(math.ceil(target_shares - 1e-12))


def _apply_execution_risk_guards(target: pd.DataFrame, price_col: str, nav: float) -> Tuple[pd.DataFrame, Dict[str, int]]:
    out = target.copy()
    counters = {
        "dropped_price_below_min": 0,
        "dropped_price_above_max": 0,
        "clipped_weight_cap": 0,
        "clipped_notional_cap": 0,
    }
    price = _num(out[price_col]).fillna(float(DEFAULT_PRICE_FALLBACK))
    below_min = price < float(MIN_PRICE_TO_TRADE)
    above_max = price > float(MAX_PRICE_TO_TRADE)
    counters["dropped_price_below_min"] = int(below_min.sum())
    counters["dropped_price_above_max"] = int(above_max.sum())
    out.loc[below_min | above_max, "weight"] = 0.0

    weight_before = _num(out["weight"]).copy()
    out["weight"] = _num(out["weight"]).clip(lower=-float(MAX_SINGLE_NAME_WEIGHT), upper=float(MAX_SINGLE_NAME_WEIGHT))
    counters["clipped_weight_cap"] = int((weight_before != _num(out["weight"]).sum()).sum()) if False else int((weight_before != _num(out["weight"])).sum())

    if float(MAX_SINGLE_NAME_NOTIONAL) > 0.0:
        weight_notional_cap = float(MAX_SINGLE_NAME_NOTIONAL) / (float(nav) + 1e-12)
        weight_after_cap = _num(out["weight"]).clip(lower=-weight_notional_cap, upper=weight_notional_cap)
        counters["clipped_notional_cap"] = int((_num(out["weight"]) != weight_after_cap).sum())
        out["weight"] = weight_after_cap

    return out, counters


def _build_target_table(book: pd.DataFrame, price_df: pd.DataFrame, price_col: str, nav: float, fractional_enabled: bool) -> Tuple[pd.DataFrame, int, Dict[str, int]]:
    target = book[["symbol", "weight"]].copy()
    target = target.merge(price_df, on="symbol", how="left")
    missing_price_mask = target[price_col].isna()
    merged_missing = int(missing_price_mask.sum())
    target[price_col] = _num(target[price_col]).fillna(float(DEFAULT_PRICE_FALLBACK))
    target, risk_counters = _apply_execution_risk_guards(target, price_col=price_col, nav=nav)
    target["target_notional"] = _num(target["weight"]) * float(nav)
    target["target_shares_raw"] = target["target_notional"] / (_num(target[price_col]) + 1e-12)
    target["target_shares"] = target["target_shares_raw"].map(lambda x: _round_shares(float(x), fractional_enabled=fractional_enabled))
    target["fractional_enabled_effective"] = int(fractional_enabled)
    return target, merged_missing, risk_counters


def _empty_order_diag() -> Dict[str, int]:
    return {
        "already_at_target": 0,
        "below_min_notional": 0,
        "rounded_to_zero_from_nonzero_target": 0,
        "entered_new_position": 0,
        "exited_position": 0,
        "increased_existing_position": 0,
        "decreased_existing_position": 0,
        "symbol_only_in_state": 0,
        "symbol_only_in_target": 0,
    }


def _build_orders(target: pd.DataFrame, state: dict, price_col: str, current_date: pd.Timestamp) -> Tuple[pd.DataFrame, Dict[str, int]]:
    rows: List[Dict[str, object]] = []
    diag = _empty_order_diag()
    target_symbol_set = set(target["symbol"].astype(str).tolist())
    state_symbol_set = set(state.get("positions", {}).keys())
    symbols = sorted(target_symbol_set | state_symbol_set)
    target_map = {str(row["symbol"]): row for _, row in target.iterrows()}

    for symbol in symbols:
        row = target_map.get(symbol)
        if row is None:
            diag["symbol_only_in_state"] += 1
        elif symbol not in state_symbol_set:
            diag["symbol_only_in_target"] += 1

        price = float(row[price_col]) if row is not None else float(DEFAULT_PRICE_FALLBACK)
        target_weight = float(row["weight"]) if row is not None else 0.0
        target_notional = float(row["target_notional"]) if row is not None else 0.0
        target_shares_raw = float(row["target_shares_raw"]) if row is not None else 0.0
        target_shares = float(row["target_shares"]) if row is not None else 0.0
        legacy_skipped = row is None and SKIP_LEGACY_SYMBOLS_NOT_IN_TARGET
        current_shares = _current_position_shares(state, symbol)
        raw_delta_shares = float(target_shares - current_shares)
        delta_shares = raw_delta_shares
        order_notional = float(delta_shares * price)
        order_notional_abs = abs(order_notional)
        order_side = "BUY" if delta_shares > 0 else "SELL" if delta_shares < 0 else "HOLD"
        skip_reason = ""

        if abs(target_shares_raw) > 1e-12 and abs(target_shares) <= 1e-12 and abs(current_shares) <= 1e-12:
            diag["rounded_to_zero_from_nonzero_target"] += 1
            if not skip_reason:
                skip_reason = "rounded_to_zero"

        if legacy_skipped:
            target_shares = current_shares
            raw_delta_shares = 0.0
            delta_shares = 0.0
            order_notional = 0.0
            order_notional_abs = 0.0
            order_side = "HOLD"
            skip_reason = "legacy_symbol_not_in_target"

        if order_side == "HOLD":
            diag["already_at_target"] += 1
            if not skip_reason:
                skip_reason = "already_at_target"
        elif order_notional_abs < float(MIN_ORDER_NOTIONAL):
            diag["below_min_notional"] += 1
            skip_reason = "below_min_notional"
            delta_shares = 0.0
            order_notional = 0.0
            order_notional_abs = 0.0
            order_side = "HOLD"
        else:
            if abs(current_shares) <= 1e-12 and abs(target_shares) > 1e-12:
                diag["entered_new_position"] += 1
            elif abs(current_shares) > 1e-12 and abs(target_shares) <= 1e-12:
                diag["exited_position"] += 1
            elif abs(target_shares) > abs(current_shares):
                diag["increased_existing_position"] += 1
            else:
                diag["decreased_existing_position"] += 1

        estimated_commission = _estimate_commission(order_notional_abs)
        estimated_slippage = _estimate_slippage(order_notional_abs)
        estimated_total_cost = float(estimated_commission + estimated_slippage)
        rows.append(
            {
                "date": str(current_date.date()),
                "symbol": symbol,
                "price": price,
                "target_weight": target_weight,
                "target_notional": target_notional,
                "target_shares_raw": target_shares_raw,
                "current_shares": current_shares,
                "target_shares": target_shares,
                "raw_delta_shares": raw_delta_shares,
                "delta_shares": delta_shares,
                "order_side": order_side,
                "order_notional": order_notional,
                "order_notional_abs": order_notional_abs,
                "estimated_commission": estimated_commission,
                "estimated_slippage": estimated_slippage,
                "estimated_total_cost": estimated_total_cost,
                "skip_reason": skip_reason,
            }
        )
    orders = pd.DataFrame(rows)
    if len(orders):
        orders = orders.sort_values(["order_side", "order_notional_abs", "symbol"], ascending=[True, False, True]).reset_index(drop=True)
    return orders, diag


def _mark_positions_to_market(positions: Dict[str, dict], price_df: pd.DataFrame, price_col: str) -> Tuple[Dict[str, dict], pd.DataFrame, int]:
    price_map = dict(zip(price_df["symbol"].astype(str), _num(price_df[price_col]).fillna(float(DEFAULT_PRICE_FALLBACK))))
    updated: Dict[str, dict] = {}
    rows: List[Dict[str, object]] = []
    fallback_count = 0
    for symbol, pos in positions.items():
        shares = float(pos.get("shares", 0.0))
        if abs(shares) < 1e-12:
            continue
        price = float(price_map.get(symbol, float(DEFAULT_PRICE_FALLBACK)))
        if symbol not in price_map:
            fallback_count += 1
        market_value = float(shares * price)
        updated[symbol] = {
            "shares": shares,
            "last_price": price,
            "market_value": market_value,
        }
        rows.append({"symbol": symbol, "shares": shares, "last_price": price, "market_value": market_value})
    mtm_df = pd.DataFrame(rows)
    if len(mtm_df):
        mtm_df = mtm_df.sort_values(["market_value", "symbol"], ascending=[False, True]).reset_index(drop=True)
    return updated, mtm_df, fallback_count


def _broker_positions_to_state_positions(broker_positions_csv: Path, price_df: pd.DataFrame, price_col: str) -> Tuple[Dict[str, dict], float, int]:
    _must_exist(broker_positions_csv, "broker_positions.csv for one-time alignment")
    df = pd.read_csv(broker_positions_csv)
    if df.empty:
        if ALIGN_STATE_REQUIRE_BROKER_POSITIONS:
            raise RuntimeError(f"broker_positions.csv is empty: {broker_positions_csv}")
        return {}, 0.0, 0
    if "symbol" not in df.columns or "position" not in df.columns:
        raise RuntimeError(f"broker_positions.csv missing symbol/position columns: {broker_positions_csv}")
    df = df.copy()
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["position"] = pd.to_numeric(df["position"], errors="coerce").fillna(0.0)
    df = df.loc[df["symbol"] != ""].copy()
    df = df.groupby("symbol", as_index=False).agg(position=("position", "sum"))
    price_map = dict(zip(price_df["symbol"].astype(str), _num(price_df[price_col]).fillna(float(DEFAULT_PRICE_FALLBACK))))
    positions: Dict[str, dict] = {}
    fallback_count = 0
    market_value = 0.0
    for _, row in df.iterrows():
        symbol = str(row["symbol"])
        shares = float(row["position"])
        if abs(shares) <= 1e-12:
            continue
        price = float(price_map.get(symbol, float(DEFAULT_PRICE_FALLBACK)))
        if symbol not in price_map:
            fallback_count += 1
        mv = float(shares * price)
        positions[symbol] = {"shares": shares, "last_price": price, "market_value": mv}
        market_value += mv
    return positions, market_value, fallback_count


def _maybe_align_state_once(state: dict, paths: ConfigPaths, price_df: pd.DataFrame, price_col: str, current_date: pd.Timestamp) -> Tuple[dict, Dict[str, object]]:
    _, _, _, _, broker_positions_csv, alignment_marker_json = _state_paths(paths.execution_dir)
    diag: Dict[str, object] = {
        "alignment_requested": int(ALIGN_STATE_FROM_BROKER_ONCE),
        "alignment_applied": 0,
        "alignment_reason": "not_requested",
        "alignment_marker_json": str(alignment_marker_json),
        "alignment_broker_positions_csv": str(broker_positions_csv),
        "alignment_fallback_price_count": 0,
    }
    if not ALIGN_STATE_FROM_BROKER_ONCE:
        return state, diag
    if alignment_marker_json.exists():
        diag["alignment_reason"] = "already_applied"
        return state, diag
    positions, market_value, fallback_count = _broker_positions_to_state_positions(broker_positions_csv, price_df, price_col)
    aligned_state = json.loads(json.dumps(state))
    aligned_state["config"] = paths.name
    aligned_state["positions"] = positions
    if ALIGN_STATE_CASH_MODE == "zero":
        cash = 0.0
        nav = float(market_value)
    else:
        nav = float(state.get("nav", ACCOUNT_NAV))
        cash = float(nav - market_value)
    aligned_state["cash"] = float(cash)
    aligned_state["nav"] = float(nav)
    aligned_state["last_rebalanced_date"] = str(current_date.date())
    aligned_state["state_alignment"] = {
        "applied_at_date": str(current_date.date()),
        "source": str(broker_positions_csv),
        "cash_mode": ALIGN_STATE_CASH_MODE,
        "positions_count": int(len(positions)),
        "market_value": float(market_value),
        "cash_after_alignment": float(cash),
        "nav_after_alignment": float(nav),
        "fallback_price_count": int(fallback_count),
    }
    alignment_marker_payload = {
        "applied_at_date": str(current_date.date()),
        "config": paths.name,
        "source": str(broker_positions_csv),
        "cash_mode": ALIGN_STATE_CASH_MODE,
        "positions_count": int(len(positions)),
        "market_value": float(market_value),
        "cash_after_alignment": float(cash),
        "nav_after_alignment": float(nav),
        "fallback_price_count": int(fallback_count),
    }
    alignment_marker_json.write_text(json.dumps(alignment_marker_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _save_state(aligned_state, _state_paths(paths.execution_dir)[0])
    diag.update({
        "alignment_applied": 1,
        "alignment_reason": "applied_from_broker_positions",
        "alignment_positions_count": int(len(positions)),
        "alignment_market_value": float(market_value),
        "alignment_cash_after": float(cash),
        "alignment_nav_after": float(nav),
        "alignment_fallback_price_count": int(fallback_count),
    })
    return aligned_state, diag


def _apply_orders_to_state(state: dict, orders: pd.DataFrame, config_name: str, current_date: pd.Timestamp, price_df: pd.DataFrame, price_col: str) -> Tuple[dict, pd.DataFrame, int]:
    new_state = json.loads(json.dumps(state))
    positions = new_state.setdefault("positions", {})
    cash = float(new_state.get("cash", ACCOUNT_NAV))
    total_estimated_cost = float(_num(orders.get("estimated_total_cost", pd.Series(dtype="float64"))).sum()) if len(orders) else 0.0
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
    cash -= total_estimated_cost

    mtm_positions, mtm_df, mtm_fallback_count = _mark_positions_to_market(positions, price_df, price_col)
    market_value = float(mtm_df["market_value"].sum()) if len(mtm_df) else 0.0
    new_state["positions"] = mtm_positions
    new_state["config"] = config_name
    new_state["cash"] = cash
    new_state["nav"] = float(cash + market_value)
    new_state["last_rebalanced_date"] = str(current_date.date())
    new_state["last_estimated_trading_cost"] = total_estimated_cost
    return new_state, mtm_df, mtm_fallback_count


def _append_execution_log(execution_log_csv: Path, log_row: Dict[str, object]) -> None:
    row_df = pd.DataFrame([log_row])
    if execution_log_csv.exists():
        prev = pd.read_csv(execution_log_csv)
        out = pd.concat([prev, row_df], ignore_index=True)
    else:
        out = row_df
    out.to_csv(execution_log_csv, index=False)


def _run_one_config(paths: ConfigPaths, price_df: pd.DataFrame, price_col: str, current_date: pd.Timestamp, fractional_enabled: bool) -> None:
    print(f"[EXEC][{paths.name}] loading freeze artifacts")
    book, freeze_summary = _load_freeze_book(paths)
    freeze_live_active_names = int(freeze_summary.get("live_active_names", 0) or 0)
    freeze_live_current_date = str(freeze_summary.get("live_current_date", "")).strip()

    if REQUIRE_FREEZE_DATE_MATCH_LIVE_PRICES and freeze_live_current_date and freeze_live_current_date != str(current_date.date()):
        raise RuntimeError(
            f"Freeze/live price date mismatch for {paths.name}: freeze_live_current_date={freeze_live_current_date} live_price_date={current_date.date()}"
        )

    paths.execution_dir.mkdir(parents=True, exist_ok=True)
    state_json, orders_csv, execution_log_csv, mtm_csv, broker_positions_csv, alignment_marker_json = _state_paths(paths.execution_dir)

    if SKIP_EMPTY_FREEZE_CONFIGS and freeze_live_active_names <= 0:
        print(f"[EXEC][{paths.name}][SKIP] freeze summary has live_active_names=0")
        _append_execution_log(
            execution_log_csv,
            {
                "date": str(current_date.date()),
                "config": paths.name,
                "starting_nav": float(ACCOUNT_NAV),
                "ending_nav": float(ACCOUNT_NAV),
                "cash_after": float(ACCOUNT_NAV),
                "market_value_after": 0.0,
                "live_active_names": 0,
                "order_count": 0,
                "gross_target": 0.0,
                "freeze_live_current_date": freeze_summary.get("live_current_date", ""),
                "freeze_replay_current_date": freeze_summary.get("replay_current_date", ""),
                "skip_reason": "freeze_live_active_names_zero",
                "price_col": price_col,
                "merged_missing_prices": 0,
                "fallback_price_rows": 0,
                "mtm_fallback_count": 0,
                "fractional_enabled_effective": int(fractional_enabled),
                "risk_dropped_price_below_min": 0,
                "risk_dropped_price_above_max": 0,
                "risk_clipped_weight_cap": 0,
                "risk_clipped_notional_cap": 0,
                "diag_already_at_target": 0,
                "diag_below_min_notional": 0,
                "diag_rounded_to_zero_from_nonzero_target": 0,
                "diag_entered_new_position": 0,
                "diag_exited_position": 0,
                "diag_increased_existing_position": 0,
                "diag_decreased_existing_position": 0,
                "diag_symbol_only_in_state": 0,
                "diag_symbol_only_in_target": 0,
                "estimated_commission_total": 0.0,
                "estimated_slippage_total": 0.0,
                "estimated_total_cost": 0.0,
                "alignment_applied": 0,
                "alignment_reason": "skip_empty_freeze_config",
            },
        )
        print(f"[ARTIFACT] {execution_log_csv}")
        return

    state = _load_state(paths.name, state_json)
    state, alignment_diag = _maybe_align_state_once(state, paths, price_df, price_col, current_date)
    starting_nav = float(state.get("nav", ACCOUNT_NAV))
    starting_cash = float(state.get("cash", ACCOUNT_NAV))
    print(
        f"[EXEC][{paths.name}][ALIGN] requested={alignment_diag.get('alignment_requested', 0)} applied={alignment_diag.get('alignment_applied', 0)} "
        f"reason={alignment_diag.get('alignment_reason', '')} marker={alignment_diag.get('alignment_marker_json', '')}"
    )

    target, merged_missing, risk_counters = _build_target_table(book, price_df, price_col, starting_nav, fractional_enabled=fractional_enabled)
    live_active_names = int(len(target.loc[target["target_shares"].abs() > 0.0]))
    gross_target = float(target["target_notional"].abs().sum() / (starting_nav + 1e-12)) if len(target) else 0.0
    fallback_price_rows = int((_num(target[price_col]) == float(DEFAULT_PRICE_FALLBACK)).sum()) if len(target) else 0

    bootstrap_only = int(alignment_diag.get("alignment_applied", 0)) == 1
    if bootstrap_only:
        orders = pd.DataFrame(
            columns=[
                "date", "symbol", "price", "target_weight", "target_notional", "target_shares_raw",
                "current_shares", "target_shares", "raw_delta_shares", "delta_shares", "order_side",
                "order_notional", "order_notional_abs", "estimated_commission", "estimated_slippage",
                "estimated_total_cost", "skip_reason",
            ]
        )
        order_diag = _empty_order_diag()
        mtm_positions, mtm_df, mtm_fallback_count = _mark_positions_to_market(state.get("positions", {}), price_df, price_col)
        next_state = json.loads(json.dumps(state))
        next_state["positions"] = mtm_positions
        next_state["config"] = paths.name
        next_state["last_rebalanced_date"] = str(current_date.date())
        next_state["last_estimated_trading_cost"] = 0.0
        next_state["nav"] = float(next_state.get("cash", ACCOUNT_NAV)) + float(mtm_df["market_value"].sum() if len(mtm_df) else 0.0)
        target = target.copy()
        target["bootstrap_only_skip"] = 1
        skip_reason_value = "aligned_state_bootstrap_only"
    else:
        orders, order_diag = _build_orders(target, state, price_col, current_date)
        next_state, mtm_df, mtm_fallback_count = _apply_orders_to_state(state, orders, paths.name, current_date, price_df, price_col)
        skip_reason_value = ""

    target.to_csv(paths.execution_dir / "target_book.csv", index=False)
    orders.to_csv(orders_csv, index=False)
    mtm_df.to_csv(mtm_csv, index=False)
    _save_state(next_state, state_json)

    order_count = int((orders["order_side"] != "HOLD").sum()) if len(orders) and "order_side" in orders.columns else 0
    ending_nav = float(next_state.get("nav", ACCOUNT_NAV))
    cash_after = float(next_state.get("cash", ACCOUNT_NAV))
    market_value_after = float(mtm_df["market_value"].sum()) if len(mtm_df) else 0.0
    nav_recon_error = float(ending_nav - (cash_after + market_value_after))
    realized_trade_cash_flow = float(starting_cash - cash_after)
    estimated_commission_total = float(_num(orders.get("estimated_commission", pd.Series(dtype="float64"))).sum()) if len(orders) else 0.0
    estimated_slippage_total = float(_num(orders.get("estimated_slippage", pd.Series(dtype="float64"))).sum()) if len(orders) else 0.0
    estimated_total_cost = float(_num(orders.get("estimated_total_cost", pd.Series(dtype="float64"))).sum()) if len(orders) else 0.0

    log_row = {
        "date": str(current_date.date()),
        "config": paths.name,
        "starting_nav": starting_nav,
        "ending_nav": ending_nav,
        "cash_after": cash_after,
        "market_value_after": market_value_after,
        "live_active_names": live_active_names,
        "order_count": order_count,
        "gross_target": gross_target,
        "freeze_live_current_date": freeze_summary.get("live_current_date", ""),
        "freeze_replay_current_date": freeze_summary.get("replay_current_date", ""),
        "skip_reason": skip_reason_value,
        "price_col": price_col,
        "merged_missing_prices": merged_missing,
        "fallback_price_rows": fallback_price_rows,
        "mtm_fallback_count": mtm_fallback_count,
        "nav_recon_error": nav_recon_error,
        "realized_trade_cash_flow": realized_trade_cash_flow,
        "fractional_enabled_effective": int(fractional_enabled),
        "risk_dropped_price_below_min": int(risk_counters.get("dropped_price_below_min", 0)),
        "risk_dropped_price_above_max": int(risk_counters.get("dropped_price_above_max", 0)),
        "risk_clipped_weight_cap": int(risk_counters.get("clipped_weight_cap", 0)),
        "risk_clipped_notional_cap": int(risk_counters.get("clipped_notional_cap", 0)),
        "diag_already_at_target": int(order_diag.get("already_at_target", 0)),
        "diag_below_min_notional": int(order_diag.get("below_min_notional", 0)),
        "diag_rounded_to_zero_from_nonzero_target": int(order_diag.get("rounded_to_zero_from_nonzero_target", 0)),
        "diag_entered_new_position": int(order_diag.get("entered_new_position", 0)),
        "diag_exited_position": int(order_diag.get("exited_position", 0)),
        "diag_increased_existing_position": int(order_diag.get("increased_existing_position", 0)),
        "diag_decreased_existing_position": int(order_diag.get("decreased_existing_position", 0)),
        "diag_symbol_only_in_state": int(order_diag.get("symbol_only_in_state", 0)),
        "diag_symbol_only_in_target": int(order_diag.get("symbol_only_in_target", 0)),
        "estimated_commission_total": estimated_commission_total,
        "estimated_slippage_total": estimated_slippage_total,
        "estimated_total_cost": estimated_total_cost,
        "alignment_applied": int(alignment_diag.get("alignment_applied", 0)),
        "alignment_reason": str(alignment_diag.get("alignment_reason", "")),
        "alignment_positions_count": int(alignment_diag.get("alignment_positions_count", 0) or 0),
        "alignment_market_value": float(alignment_diag.get("alignment_market_value", 0.0) or 0.0),
        "alignment_cash_after": float(alignment_diag.get("alignment_cash_after", 0.0) or 0.0),
        "alignment_nav_after": float(alignment_diag.get("alignment_nav_after", 0.0) or 0.0),
        "alignment_fallback_price_count": int(alignment_diag.get("alignment_fallback_price_count", 0) or 0),
    }
    _append_execution_log(execution_log_csv, log_row)

    print(
        f"[EXEC][{paths.name}] starting_nav={starting_nav:.2f} ending_nav={ending_nav:.2f} "
        f"cash_after={cash_after:.2f} market_value_after={market_value_after:.2f} nav_recon_error={nav_recon_error:.6f} "
        f"order_count={order_count} active_names={live_active_names} gross_target={gross_target:.4f} "
        f"price_col={price_col} merged_missing_prices={merged_missing} fallback_price_rows={fallback_price_rows} mtm_fallback_count={mtm_fallback_count} "
        f"fractional_enabled_effective={int(fractional_enabled)} bootstrap_only={int(bootstrap_only)} skip_reason={skip_reason_value}"
    )
    print(
        f"[EXEC][{paths.name}][RISK] dropped_price_below_min={risk_counters.get('dropped_price_below_min', 0)} "
        f"dropped_price_above_max={risk_counters.get('dropped_price_above_max', 0)} "
        f"clipped_weight_cap={risk_counters.get('clipped_weight_cap', 0)} "
        f"clipped_notional_cap={risk_counters.get('clipped_notional_cap', 0)}"
    )
    print(
        f"[EXEC][{paths.name}][DIAG] already_at_target={order_diag.get('already_at_target', 0)} "
        f"below_min_notional={order_diag.get('below_min_notional', 0)} "
        f"rounded_to_zero_from_nonzero_target={order_diag.get('rounded_to_zero_from_nonzero_target', 0)} "
        f"entered_new_position={order_diag.get('entered_new_position', 0)} "
        f"exited_position={order_diag.get('exited_position', 0)} "
        f"increased_existing_position={order_diag.get('increased_existing_position', 0)} "
        f"decreased_existing_position={order_diag.get('decreased_existing_position', 0)} "
        f"symbol_only_in_state={order_diag.get('symbol_only_in_state', 0)} "
        f"symbol_only_in_target={order_diag.get('symbol_only_in_target', 0)}"
    )
    print(
        f"[EXEC][{paths.name}][COST] estimated_commission_total={estimated_commission_total:.4f} "
        f"estimated_slippage_total={estimated_slippage_total:.4f} estimated_total_cost={estimated_total_cost:.4f}"
    )
    if bootstrap_only:
        print(f"[EXEC][{paths.name}][BOOTSTRAP_ONLY] state aligned from broker_positions.csv; skipping live order generation on this run")
    elif len(orders):
        print(f"[EXEC][{paths.name}][ORDERS_TOP]")
        print(
            orders[[
                "symbol", "order_side", "current_shares", "target_shares_raw", "target_shares",
                "raw_delta_shares", "delta_shares", "price", "order_notional", "target_weight",
                "estimated_commission", "estimated_slippage", "estimated_total_cost", "skip_reason"
            ]].head(min(TOPK_PRINT, len(orders))).to_string(index=False)
        )
    if len(mtm_df):
        print(f"[EXEC][{paths.name}][POSITIONS_MTM_TOP]")
        print(mtm_df.head(min(TOPK_PRINT, len(mtm_df))).to_string(index=False))
    print(f"[ARTIFACT] {paths.execution_dir / 'target_book.csv'}")
    print(f"[ARTIFACT] {orders_csv}")
    print(f"[ARTIFACT] {mtm_csv}")
    print(f"[ARTIFACT] {state_json}")
    print(f"[ARTIFACT] {execution_log_csv}")
    print(f"[ARTIFACT] {alignment_marker_json}")


def main() -> int:
    _enable_line_buffering()
    fractional_enabled = _fractional_enabled_effective()
    print(f"[CFG] live_feature_snapshot_file={LIVE_FEATURE_SNAPSHOT_FILE}")
    print(f"[CFG] freeze_root={FREEZE_ROOT}")
    print(f"[CFG] execution_root={EXECUTION_ROOT}")
    print(f"[CFG] account_nav={ACCOUNT_NAV}")
    print(f"[CFG] configs={CONFIG_NAMES}")
    print(
        f"[CFG] allow_fractional_shares={int(ALLOW_FRACTIONAL_SHARES)} fractional_mode={FRACTIONAL_MODE} fractional_enabled_effective={int(fractional_enabled)} "
        f"min_order_notional={MIN_ORDER_NOTIONAL} skip_empty_freeze_configs={int(SKIP_EMPTY_FREEZE_CONFIGS)} "
        f"require_freeze_date_match_live_prices={int(REQUIRE_FREEZE_DATE_MATCH_LIVE_PRICES)}"
    )
    print(
        f"[CFG] max_single_name_weight={MAX_SINGLE_NAME_WEIGHT} max_single_name_notional={MAX_SINGLE_NAME_NOTIONAL} "
        f"min_price_to_trade={MIN_PRICE_TO_TRADE} max_price_to_trade={MAX_PRICE_TO_TRADE}"
    )
    print(
        f"[CFG] commission_bps={COMMISSION_BPS} commission_min_per_order={COMMISSION_MIN_PER_ORDER} slippage_bps={SLIPPAGE_BPS}"
    )
    print(
        f"[CFG] align_state_from_broker_once={int(ALIGN_STATE_FROM_BROKER_ONCE)} align_state_require_broker_positions={int(ALIGN_STATE_REQUIRE_BROKER_POSITIONS)} "
        f"align_state_cash_mode={ALIGN_STATE_CASH_MODE} skip_legacy_symbols_not_in_target={int(SKIP_LEGACY_SYMBOLS_NOT_IN_TARGET)}"
    )

    feature_df = _load_live_feature_frame()
    current_date, price_df, price_col, snapshot_fallback_count = _build_price_snapshot(feature_df)
    print(
        f"[DATA] current_date={current_date.date()} price_col={price_col} symbols={len(price_df)} "
        f"snapshot_fallback_count={snapshot_fallback_count}"
    )

    for name in CONFIG_NAMES:
        paths = _config_paths(name)
        _run_one_config(paths, price_df, price_col, current_date, fractional_enabled=fractional_enabled)

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
