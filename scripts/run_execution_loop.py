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
USE_STATE_PRICE_FALLBACK_FOR_LEGACY = str(os.getenv("USE_STATE_PRICE_FALLBACK_FOR_LEGACY", "1")).strip().lower() not in {"0", "false", "no", "off"}
USE_STATE_PRICE_FALLBACK_FOR_MTM = str(os.getenv("USE_STATE_PRICE_FALLBACK_FOR_MTM", "1")).strip().lower() not in {"0", "false", "no", "off"}
SANITIZE_STATE_PRICES = str(os.getenv("SANITIZE_STATE_PRICES", "1")).strip().lower() not in {"0", "false", "no", "off"}
STATE_FALLBACK_PRICE_TOLERANCE = float(os.getenv("STATE_FALLBACK_PRICE_TOLERANCE", "0.000001"))
PERSIST_DEFAULT_FALLBACK_TO_STATE = str(os.getenv("PERSIST_DEFAULT_FALLBACK_TO_STATE", "0")).strip().lower() not in {"0", "false", "no", "off"}

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
BROKER_PRICE_COL_CANDIDATES = [
    "marketPrice",
    "market_price",
    "markPrice",
    "mark_price",
    "lastPrice",
    "last_price",
    "price",
    "avgCost",
    "averageCost",
    "average_cost",
    "costBasisPrice",
    "cost_basis_price",
]


@dataclass(frozen=True)
class ConfigPaths:
    name: str
    freeze_dir: Path
    execution_dir: Path
    current_book_csv: Path
    current_summary_json: Path


@dataclass(frozen=True)
class PriceLookupResult:
    price: float
    source: str
    used_default_fallback: int
    used_state_fallback: int
    is_priced: int


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


def _json_safe_state(state: dict) -> dict:
    new_state = json.loads(json.dumps(state))
    positions = new_state.get("positions", {})
    if isinstance(positions, dict):
        for _symbol, pos in positions.items():
            if not isinstance(pos, dict):
                continue
            price = pos.get("last_price", None)
            try:
                price_float = float(price)
            except Exception:
                price_float = float("nan")
            if not math.isfinite(price_float) or price_float <= 0.0:
                pos["last_price"] = None
            mv = pos.get("market_value", None)
            try:
                mv_float = float(mv)
            except Exception:
                mv_float = float("nan")
            if not math.isfinite(mv_float):
                pos["market_value"] = None
    return new_state


def _save_state(state: dict, state_json: Path) -> None:
    state_json.write_text(json.dumps(_json_safe_state(state), ensure_ascii=False, indent=2), encoding="utf-8")


def _current_position_shares(state: dict, symbol: str) -> float:
    pos = state.get("positions", {}).get(symbol, {})
    try:
        return float(pos.get("shares", 0.0))
    except Exception:
        return 0.0


def _extract_state_last_price(state: dict, symbol: str) -> float:
    pos = state.get("positions", {}).get(symbol, {})
    try:
        value = float(pos.get("last_price", float("nan")))
    except Exception:
        return float("nan")
    if math.isfinite(value) and value > 0.0:
        return value
    return float("nan")


def _build_live_price_map(price_df: pd.DataFrame, price_col: str) -> Dict[str, float]:
    if price_df.empty:
        return {}
    symbols = price_df["symbol"].astype(str).tolist()
    prices = _num(price_df[price_col]).fillna(float(DEFAULT_PRICE_FALLBACK)).tolist()
    return dict(zip(symbols, prices))


def _lookup_price(symbol: str, live_price_map: Dict[str, float], state: dict, allow_state_fallback: bool, allow_default_fallback: bool) -> PriceLookupResult:
    live_value = live_price_map.get(symbol)
    if live_value is not None:
        price = float(live_value)
        if math.isfinite(price) and price > 0.0:
            return PriceLookupResult(price=price, source="live_snapshot", used_default_fallback=0, used_state_fallback=0, is_priced=1)
    if allow_state_fallback:
        state_price = _extract_state_last_price(state, symbol)
        if math.isfinite(state_price) and state_price > 0.0:
            return PriceLookupResult(price=state_price, source="state_last_price", used_default_fallback=0, used_state_fallback=1, is_priced=1)
    if allow_default_fallback:
        return PriceLookupResult(price=float(DEFAULT_PRICE_FALLBACK), source="default_fallback", used_default_fallback=1, used_state_fallback=0, is_priced=1)
    return PriceLookupResult(price=float("nan"), source="unpriced", used_default_fallback=0, used_state_fallback=0, is_priced=0)


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
    counters["clipped_weight_cap"] = int((weight_before != _num(out["weight"])).sum())

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
        "legacy_state_price_fallback": 0,
        "legacy_default_price_fallback": 0,
        "legacy_unpriced": 0,
    }


def _build_orders(target: pd.DataFrame, state: dict, price_df: pd.DataFrame, price_col: str, current_date: pd.Timestamp) -> Tuple[pd.DataFrame, Dict[str, int]]:
    rows: List[Dict[str, object]] = []
    diag = _empty_order_diag()
    live_price_map = _build_live_price_map(price_df, price_col)
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

        if row is not None:
            price = float(row[price_col])
            price_source = "target_live_snapshot"
            target_weight = float(row["weight"])
            target_notional = float(row["target_notional"])
            target_shares_raw = float(row["target_shares_raw"])
            target_shares = float(row["target_shares"])
            is_priced = 1
        else:
            price_lookup = _lookup_price(symbol, live_price_map=live_price_map, state=state, allow_state_fallback=USE_STATE_PRICE_FALLBACK_FOR_LEGACY, allow_default_fallback=False)
            price = float(price_lookup.price) if price_lookup.is_priced else float("nan")
            price_source = price_lookup.source
            is_priced = int(price_lookup.is_priced)
            diag["legacy_state_price_fallback"] += int(price_lookup.used_state_fallback)
            diag["legacy_default_price_fallback"] += int(price_lookup.used_default_fallback)
            diag["legacy_unpriced"] += int(1 - price_lookup.is_priced)
            target_weight = 0.0
            target_notional = 0.0
            target_shares_raw = 0.0
            target_shares = 0.0

        legacy_skipped = row is None and SKIP_LEGACY_SYMBOLS_NOT_IN_TARGET
        current_shares = _current_position_shares(state, symbol)
        raw_delta_shares = float(target_shares - current_shares)
        delta_shares = raw_delta_shares
        order_notional = float(delta_shares * price) if is_priced else 0.0
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
            skip_reason = "legacy_symbol_not_in_target" if is_priced else "legacy_symbol_not_in_target_unpriced"

        if not is_priced:
            order_notional = 0.0
            order_notional_abs = 0.0
            order_side = "HOLD"
            if not skip_reason:
                skip_reason = "unpriced_symbol"

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
                "price": price if is_priced else pd.NA,
                "price_source": price_source,
                "is_priced": is_priced,
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


def _mark_positions_to_market(positions: Dict[str, dict], price_df: pd.DataFrame, price_col: str) -> Tuple[Dict[str, dict], pd.DataFrame, int, int, int]:
    live_price_map = _build_live_price_map(price_df, price_col)
    updated: Dict[str, dict] = {}
    rows: List[Dict[str, object]] = []
    default_fallback_count = 0
    state_fallback_count = 0
    unpriced_count = 0
    state_wrapper = {"positions": positions}
    for symbol, pos in positions.items():
        shares = float(pos.get("shares", 0.0))
        if abs(shares) < 1e-12:
            continue
        price_lookup = _lookup_price(symbol, live_price_map=live_price_map, state=state_wrapper, allow_state_fallback=USE_STATE_PRICE_FALLBACK_FOR_MTM, allow_default_fallback=False)
        default_fallback_count += int(price_lookup.used_default_fallback)
        state_fallback_count += int(price_lookup.used_state_fallback)
        is_priced = int(price_lookup.is_priced)
        unpriced_count += int(1 - is_priced)
        if is_priced:
            price = float(price_lookup.price)
            market_value = float(shares * price)
            last_price_value = price
            market_value_value = market_value
        else:
            price = float("nan")
            market_value = 0.0
            last_price_value = None
            market_value_value = None
        updated[symbol] = {
            "shares": shares,
            "last_price": last_price_value,
            "market_value": market_value_value,
            "price_source": price_lookup.source,
            "is_priced": is_priced,
        }
        rows.append(
            {
                "symbol": symbol,
                "shares": shares,
                "last_price": last_price_value,
                "price_source": price_lookup.source,
                "is_priced": is_priced,
                "market_value": market_value_value,
            }
        )
    mtm_df = pd.DataFrame(rows)
    if len(mtm_df):
        if "market_value" in mtm_df.columns:
            mtm_df["_mv_sort"] = _num(mtm_df["market_value"]).fillna(float("-inf"))
            mtm_df = mtm_df.sort_values(["_mv_sort", "symbol"], ascending=[False, True]).drop(columns=["_mv_sort"]).reset_index(drop=True)
        else:
            mtm_df = mtm_df.sort_values(["symbol"], ascending=[True]).reset_index(drop=True)
    return updated, mtm_df, default_fallback_count, state_fallback_count, unpriced_count


def _pick_broker_price_from_row(row: pd.Series) -> Tuple[float, str]:
    for col in BROKER_PRICE_COL_CANDIDATES:
        if col in row.index:
            try:
                value = float(row[col])
            except Exception:
                continue
            if math.isfinite(value) and value > 0.0:
                return value, f"broker_positions:{col}"
    return float("nan"), ""


def _broker_positions_to_state_positions(broker_positions_csv: Path, price_df: pd.DataFrame, price_col: str) -> Tuple[Dict[str, dict], float, int, int, int, int, int]:
    _must_exist(broker_positions_csv, "broker_positions.csv for one-time alignment")
    df = pd.read_csv(broker_positions_csv)
    if df.empty:
        if ALIGN_STATE_REQUIRE_BROKER_POSITIONS:
            raise RuntimeError(f"broker_positions.csv is empty: {broker_positions_csv}")
        return {}, 0.0, 0, 0, 0, 0, 0
    if "symbol" not in df.columns or "position" not in df.columns:
        raise RuntimeError(f"broker_positions.csv missing symbol/position columns: {broker_positions_csv}")
    df = df.copy()
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["position"] = pd.to_numeric(df["position"], errors="coerce").fillna(0.0)
    df = df.loc[df["symbol"] != ""].copy()
    df = df.groupby("symbol", as_index=False).first()
    live_price_map = _build_live_price_map(price_df, price_col)
    positions: Dict[str, dict] = {}
    market_value = 0.0
    default_fallback_count = 0
    state_fallback_count = 0
    broker_price_count = 0
    live_snapshot_count = 0
    unpriced_count = 0
    for _, row in df.iterrows():
        symbol = str(row["symbol"])
        shares = float(pd.to_numeric(pd.Series([row["position"]]), errors="coerce").iloc[0])
        if abs(shares) <= 1e-12:
            continue
        broker_price, broker_source = _pick_broker_price_from_row(row)
        if math.isfinite(broker_price) and broker_price > 0.0:
            price = float(broker_price)
            source = broker_source
            broker_price_count += 1
            is_priced = 1
        else:
            live_value = live_price_map.get(symbol)
            if live_value is not None and math.isfinite(float(live_value)) and float(live_value) > 0.0:
                price = float(live_value)
                source = "live_snapshot"
                live_snapshot_count += 1
                is_priced = 1
            else:
                source = "unpriced_alignment"
                is_priced = 0
                unpriced_count += 1
                if PERSIST_DEFAULT_FALLBACK_TO_STATE:
                    price = float(DEFAULT_PRICE_FALLBACK)
                    default_fallback_count += 1
                    source = "default_fallback_alignment"
                    is_priced = 1
                else:
                    price = float("nan")
        mv = float(shares * price) if is_priced else 0.0
        positions[symbol] = {
            "shares": shares,
            "last_price": price if is_priced else None,
            "market_value": mv if is_priced else None,
            "price_source": source,
            "is_priced": is_priced,
        }
        market_value += mv
    return positions, market_value, default_fallback_count, state_fallback_count, broker_price_count, live_snapshot_count, unpriced_count


def _sanitize_state_prices(state: dict, price_df: pd.DataFrame, price_col: str) -> Tuple[dict, Dict[str, int]]:
    diag = {
        "state_prices_sanitized": 0,
        "state_prices_updated_from_live": 0,
        "state_prices_cleared_default_like": 0,
        "state_prices_still_default_like": 0,
        "state_positions_missing_live_price": 0,
    }
    if not SANITIZE_STATE_PRICES:
        return state, diag
    positions = state.get("positions", {})
    if not isinstance(positions, dict) or not positions:
        return state, diag
    live_price_map = _build_live_price_map(price_df, price_col)
    new_state = json.loads(json.dumps(state))
    new_positions = new_state.setdefault("positions", {})
    tol = float(STATE_FALLBACK_PRICE_TOLERANCE)
    for symbol, pos in list(new_positions.items()):
        try:
            shares = float(pos.get("shares", 0.0))
        except Exception:
            shares = 0.0
        if abs(shares) <= 1e-12:
            continue
        try:
            state_price = float(pos.get("last_price", float("nan")))
        except Exception:
            state_price = float("nan")
        live_price = live_price_map.get(symbol)
        has_live = live_price is not None and math.isfinite(float(live_price)) and float(live_price) > 0.0
        suspicious_default_like = math.isfinite(state_price) and abs(state_price - float(DEFAULT_PRICE_FALLBACK)) <= tol
        if has_live and (not math.isfinite(state_price) or state_price <= 0.0 or suspicious_default_like):
            live_value = float(live_price)
            pos["last_price"] = live_value
            pos["market_value"] = float(shares * live_value)
            pos["price_source"] = "sanitized_live_snapshot"
            pos["is_priced"] = 1
            diag["state_prices_sanitized"] += 1
            diag["state_prices_updated_from_live"] += 1
        elif suspicious_default_like:
            pos["last_price"] = None
            pos["market_value"] = None
            pos["price_source"] = "cleared_default_like"
            pos["is_priced"] = 0
            diag["state_prices_sanitized"] += 1
            diag["state_prices_cleared_default_like"] += 1
            diag["state_positions_missing_live_price"] += int(not has_live)
        elif not has_live and (not math.isfinite(state_price) or state_price <= 0.0):
            pos["last_price"] = None
            pos["market_value"] = None
            pos["price_source"] = "unpriced_state"
            pos["is_priced"] = 0
            diag["state_positions_missing_live_price"] += 1
    return new_state, diag


def _maybe_align_state_once(state: dict, paths: ConfigPaths, price_df: pd.DataFrame, price_col: str, current_date: pd.Timestamp) -> Tuple[dict, Dict[str, object]]:
    _, _, _, _, broker_positions_csv, alignment_marker_json = _state_paths(paths.execution_dir)
    diag: Dict[str, object] = {
        "alignment_requested": int(ALIGN_STATE_FROM_BROKER_ONCE),
        "alignment_applied": 0,
        "alignment_reason": "not_requested",
        "alignment_marker_json": str(alignment_marker_json),
        "alignment_broker_positions_csv": str(broker_positions_csv),
        "alignment_default_fallback_price_count": 0,
        "alignment_state_fallback_price_count": 0,
        "alignment_broker_price_count": 0,
        "alignment_live_snapshot_price_count": 0,
        "alignment_unpriced_count": 0,
    }
    if not ALIGN_STATE_FROM_BROKER_ONCE:
        return state, diag
    if alignment_marker_json.exists():
        diag["alignment_reason"] = "already_applied"
        return state, diag
    positions, market_value, default_fallback_count, state_fallback_count, broker_price_count, live_snapshot_count, unpriced_count = _broker_positions_to_state_positions(broker_positions_csv, price_df, price_col)
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
        "default_fallback_price_count": int(default_fallback_count),
        "state_fallback_price_count": int(state_fallback_count),
        "broker_price_count": int(broker_price_count),
        "live_snapshot_price_count": int(live_snapshot_count),
        "unpriced_count": int(unpriced_count),
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
        "default_fallback_price_count": int(default_fallback_count),
        "state_fallback_price_count": int(state_fallback_count),
        "broker_price_count": int(broker_price_count),
        "live_snapshot_price_count": int(live_snapshot_count),
        "unpriced_count": int(unpriced_count),
    }
    alignment_marker_json.write_text(json.dumps(alignment_marker_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _save_state(aligned_state, _state_paths(paths.execution_dir)[0])
    diag.update(
        {
            "alignment_applied": 1,
            "alignment_reason": "applied_from_broker_positions",
            "alignment_positions_count": int(len(positions)),
            "alignment_market_value": float(market_value),
            "alignment_cash_after": float(cash),
            "alignment_nav_after": float(nav),
            "alignment_default_fallback_price_count": int(default_fallback_count),
            "alignment_state_fallback_price_count": int(state_fallback_count),
            "alignment_broker_price_count": int(broker_price_count),
            "alignment_live_snapshot_price_count": int(live_snapshot_count),
            "alignment_unpriced_count": int(unpriced_count),
        }
    )
    return aligned_state, diag


def _apply_orders_to_state(state: dict, orders: pd.DataFrame, config_name: str, current_date: pd.Timestamp, price_df: pd.DataFrame, price_col: str) -> Tuple[dict, pd.DataFrame, int, int, int]:
    new_state = json.loads(json.dumps(state))
    positions = new_state.setdefault("positions", {})
    cash = float(new_state.get("cash", ACCOUNT_NAV))
    total_estimated_cost = float(_num(orders.get("estimated_total_cost", pd.Series(dtype="float64"))).sum()) if len(orders) else 0.0
    for _, row in orders.iterrows():
        symbol = str(row["symbol"])
        delta_shares = float(row["delta_shares"])
        if abs(delta_shares) <= 0.0:
            continue
        price_value = row.get("price", pd.NA)
        try:
            price = float(price_value)
        except Exception:
            price = float("nan")
        if not math.isfinite(price) or price <= 0.0:
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
                "price_source": "post_trade",
                "is_priced": 1,
            }
    cash -= total_estimated_cost

    mtm_positions, mtm_df, mtm_default_fallback_count, mtm_state_fallback_count, mtm_unpriced_count = _mark_positions_to_market(positions, price_df, price_col)
    market_value = float(_num(mtm_df["market_value"]).fillna(0.0).sum()) if len(mtm_df) else 0.0
    new_state["positions"] = mtm_positions
    new_state["config"] = config_name
    new_state["cash"] = cash
    new_state["nav"] = float(cash + market_value)
    new_state["last_rebalanced_date"] = str(current_date.date())
    new_state["last_estimated_trading_cost"] = total_estimated_cost
    return new_state, mtm_df, mtm_default_fallback_count, mtm_state_fallback_count, mtm_unpriced_count


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
                "mtm_default_fallback_count": 0,
                "mtm_state_fallback_count": 0,
                "mtm_unpriced_count": 0,
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
                "diag_legacy_state_price_fallback": 0,
                "diag_legacy_default_price_fallback": 0,
                "diag_legacy_unpriced": 0,
                "estimated_commission_total": 0.0,
                "estimated_slippage_total": 0.0,
                "estimated_total_cost": 0.0,
                "alignment_applied": 0,
                "alignment_reason": "skip_empty_freeze_config",
                "alignment_default_fallback_price_count": 0,
                "alignment_state_fallback_price_count": 0,
                "alignment_broker_price_count": 0,
                "alignment_live_snapshot_price_count": 0,
                "alignment_unpriced_count": 0,
                "state_prices_sanitized": 0,
                "state_prices_updated_from_live": 0,
                "state_prices_cleared_default_like": 0,
                "state_positions_missing_live_price": 0,
            },
        )
        print(f"[ARTIFACT] {execution_log_csv}")
        return

    state = _load_state(paths.name, state_json)
    state, alignment_diag = _maybe_align_state_once(state, paths, price_df, price_col, current_date)
    state, sanitation_diag = _sanitize_state_prices(state, price_df, price_col)
    if sanitation_diag.get("state_prices_sanitized", 0) > 0:
        _save_state(state, state_json)
    starting_nav = float(state.get("nav", ACCOUNT_NAV))
    starting_cash = float(state.get("cash", ACCOUNT_NAV))
    print(
        f"[EXEC][{paths.name}][ALIGN] requested={alignment_diag.get('alignment_requested', 0)} applied={alignment_diag.get('alignment_applied', 0)} "
        f"reason={alignment_diag.get('alignment_reason', '')} marker={alignment_diag.get('alignment_marker_json', '')}"
    )
    print(
        f"[EXEC][{paths.name}][STATE_SANITY] sanitized={sanitation_diag.get('state_prices_sanitized', 0)} "
        f"updated_from_live={sanitation_diag.get('state_prices_updated_from_live', 0)} "
        f"cleared_default_like={sanitation_diag.get('state_prices_cleared_default_like', 0)} "
        f"missing_live_price={sanitation_diag.get('state_positions_missing_live_price', 0)}"
    )

    target, merged_missing, risk_counters = _build_target_table(book, price_df, price_col, starting_nav, fractional_enabled=fractional_enabled)
    live_active_names = int(len(target.loc[target["target_shares"].abs() > 0.0]))
    gross_target = float(target["target_notional"].abs().sum() / (starting_nav + 1e-12)) if len(target) else 0.0
    fallback_price_rows = int((_num(target[price_col]) == float(DEFAULT_PRICE_FALLBACK)).sum()) if len(target) else 0

    bootstrap_only = int(alignment_diag.get("alignment_applied", 0)) == 1
    if bootstrap_only:
        orders = pd.DataFrame(
            columns=[
                "date", "symbol", "price", "price_source", "is_priced", "target_weight", "target_notional", "target_shares_raw",
                "current_shares", "target_shares", "raw_delta_shares", "delta_shares", "order_side",
                "order_notional", "order_notional_abs", "estimated_commission", "estimated_slippage",
                "estimated_total_cost", "skip_reason",
            ]
        )
        order_diag = _empty_order_diag()
        mtm_positions, mtm_df, mtm_default_fallback_count, mtm_state_fallback_count, mtm_unpriced_count = _mark_positions_to_market(state.get("positions", {}), price_df, price_col)
        next_state = json.loads(json.dumps(state))
        next_state["positions"] = mtm_positions
        next_state["config"] = paths.name
        next_state["last_rebalanced_date"] = str(current_date.date())
        next_state["last_estimated_trading_cost"] = 0.0
        next_state["nav"] = float(next_state.get("cash", ACCOUNT_NAV)) + float(_num(mtm_df["market_value"]).fillna(0.0).sum() if len(mtm_df) else 0.0)
        target = target.copy()
        target["bootstrap_only_skip"] = 1
        skip_reason_value = "aligned_state_bootstrap_only"
    else:
        orders, order_diag = _build_orders(target, state, price_df, price_col, current_date)
        next_state, mtm_df, mtm_default_fallback_count, mtm_state_fallback_count, mtm_unpriced_count = _apply_orders_to_state(state, orders, paths.name, current_date, price_df, price_col)
        skip_reason_value = ""

    target.to_csv(paths.execution_dir / "target_book.csv", index=False)
    orders.to_csv(orders_csv, index=False)
    mtm_df.to_csv(mtm_csv, index=False)
    _save_state(next_state, state_json)

    order_count = int((orders["order_side"] != "HOLD").sum()) if len(orders) and "order_side" in orders.columns else 0
    ending_nav = float(next_state.get("nav", ACCOUNT_NAV))
    cash_after = float(next_state.get("cash", ACCOUNT_NAV))
    market_value_after = float(_num(mtm_df["market_value"]).fillna(0.0).sum()) if len(mtm_df) else 0.0
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
        "mtm_default_fallback_count": mtm_default_fallback_count,
        "mtm_state_fallback_count": mtm_state_fallback_count,
        "mtm_unpriced_count": mtm_unpriced_count,
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
        "diag_legacy_state_price_fallback": int(order_diag.get("legacy_state_price_fallback", 0)),
        "diag_legacy_default_price_fallback": int(order_diag.get("legacy_default_price_fallback", 0)),
        "diag_legacy_unpriced": int(order_diag.get("legacy_unpriced", 0)),
        "estimated_commission_total": estimated_commission_total,
        "estimated_slippage_total": estimated_slippage_total,
        "estimated_total_cost": estimated_total_cost,
        "alignment_applied": int(alignment_diag.get("alignment_applied", 0)),
        "alignment_reason": str(alignment_diag.get("alignment_reason", "")),
        "alignment_positions_count": int(alignment_diag.get("alignment_positions_count", 0) or 0),
        "alignment_market_value": float(alignment_diag.get("alignment_market_value", 0.0) or 0.0),
        "alignment_cash_after": float(alignment_diag.get("alignment_cash_after", 0.0) or 0.0),
        "alignment_nav_after": float(alignment_diag.get("alignment_nav_after", 0.0) or 0.0),
        "alignment_default_fallback_price_count": int(alignment_diag.get("alignment_default_fallback_price_count", 0) or 0),
        "alignment_state_fallback_price_count": int(alignment_diag.get("alignment_state_fallback_price_count", 0) or 0),
        "alignment_broker_price_count": int(alignment_diag.get("alignment_broker_price_count", 0) or 0),
        "alignment_live_snapshot_price_count": int(alignment_diag.get("alignment_live_snapshot_price_count", 0) or 0),
        "alignment_unpriced_count": int(alignment_diag.get("alignment_unpriced_count", 0) or 0),
        "state_prices_sanitized": int(sanitation_diag.get("state_prices_sanitized", 0)),
        "state_prices_updated_from_live": int(sanitation_diag.get("state_prices_updated_from_live", 0)),
        "state_prices_cleared_default_like": int(sanitation_diag.get("state_prices_cleared_default_like", 0)),
        "state_positions_missing_live_price": int(sanitation_diag.get("state_positions_missing_live_price", 0)),
    }
    _append_execution_log(execution_log_csv, log_row)

    print(
        f"[EXEC][{paths.name}] starting_nav={starting_nav:.2f} ending_nav={ending_nav:.2f} "
        f"cash_after={cash_after:.2f} market_value_after={market_value_after:.2f} nav_recon_error={nav_recon_error:.6f} "
        f"order_count={order_count} active_names={live_active_names} gross_target={gross_target:.4f} "
        f"price_col={price_col} merged_missing_prices={merged_missing} fallback_price_rows={fallback_price_rows} "
        f"mtm_default_fallback_count={mtm_default_fallback_count} mtm_state_fallback_count={mtm_state_fallback_count} mtm_unpriced_count={mtm_unpriced_count} "
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
        f"symbol_only_in_target={order_diag.get('symbol_only_in_target', 0)} "
        f"legacy_state_price_fallback={order_diag.get('legacy_state_price_fallback', 0)} "
        f"legacy_default_price_fallback={order_diag.get('legacy_default_price_fallback', 0)} "
        f"legacy_unpriced={order_diag.get('legacy_unpriced', 0)}"
    )
    print(
        f"[EXEC][{paths.name}][COST] estimated_commission_total={estimated_commission_total:.4f} "
        f"estimated_slippage_total={estimated_slippage_total:.4f} estimated_total_cost={estimated_total_cost:.4f}"
    )
    if bootstrap_only:
        print(f"[EXEC][{paths.name}][BOOTSTRAP_ONLY] state aligned from broker_positions.csv; skipping live order generation on this run")
    elif len(orders):
        print(f"[EXEC][{paths.name}][ORDERS_TOP]")
        print(orders.head(min(TOPK_PRINT, len(orders))).to_string(index=False))
    else:
        print(f"[EXEC][{paths.name}][ORDERS_TOP] none")
    print(f"[ARTIFACT] {orders_csv}")
    print(f"[ARTIFACT] {mtm_csv}")
    print(f"[ARTIFACT] {state_json}")
    print(f"[ARTIFACT] {execution_log_csv}")


def main() -> int:
    _enable_line_buffering()
    print(
        f"[CFG] live_feature_snapshot_file={LIVE_FEATURE_SNAPSHOT_FILE} freeze_root={FREEZE_ROOT} execution_root={EXECUTION_ROOT} "
        f"account_nav={ACCOUNT_NAV:.2f} fractional_mode={FRACTIONAL_MODE} allow_fractional_shares={int(ALLOW_FRACTIONAL_SHARES)}"
    )
    print(
        f"[CFG] min_order_notional={MIN_ORDER_NOTIONAL:.2f} default_price_fallback={DEFAULT_PRICE_FALLBACK:.2f} "
        f"topk_print={TOPK_PRINT} reset_state={int(RESET_STATE)} config_names={'|'.join(CONFIG_NAMES)}"
    )
    print(
        f"[CFG] require_freeze_date_match_live_prices={int(REQUIRE_FREEZE_DATE_MATCH_LIVE_PRICES)} "
        f"skip_empty_freeze_configs={int(SKIP_EMPTY_FREEZE_CONFIGS)} min_price_to_trade={MIN_PRICE_TO_TRADE} max_price_to_trade={MAX_PRICE_TO_TRADE}"
    )
    print(
        f"[CFG] max_single_name_weight={MAX_SINGLE_NAME_WEIGHT:.6f} max_single_name_notional={MAX_SINGLE_NAME_NOTIONAL:.2f} "
        f"commission_bps={COMMISSION_BPS:.4f} commission_min_per_order={COMMISSION_MIN_PER_ORDER:.4f} slippage_bps={SLIPPAGE_BPS:.4f}"
    )
    print(
        f"[CFG] align_state_from_broker_once={int(ALIGN_STATE_FROM_BROKER_ONCE)} align_state_require_broker_positions={int(ALIGN_STATE_REQUIRE_BROKER_POSITIONS)} "
        f"align_state_cash_mode={ALIGN_STATE_CASH_MODE} skip_legacy_symbols_not_in_target={int(SKIP_LEGACY_SYMBOLS_NOT_IN_TARGET)}"
    )
    print(
        f"[CFG] use_state_price_fallback_for_legacy={int(USE_STATE_PRICE_FALLBACK_FOR_LEGACY)} "
        f"use_state_price_fallback_for_mtm={int(USE_STATE_PRICE_FALLBACK_FOR_MTM)} sanitize_state_prices={int(SANITIZE_STATE_PRICES)} "
        f"persist_default_fallback_to_state={int(PERSIST_DEFAULT_FALLBACK_TO_STATE)}"
    )

    feature_df = _load_live_feature_frame()
    current_date, price_df, price_col, snapshot_fallback_count = _build_price_snapshot(feature_df)
    print(
        f"[LIVE_PRICES] current_date={current_date.date()} symbols={len(price_df)} price_col={price_col} snapshot_fallback_count={snapshot_fallback_count}"
    )

    fractional_enabled = _fractional_enabled_effective()
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
