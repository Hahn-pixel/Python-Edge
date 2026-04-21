from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from python_edge.broker.cpapi_client import CpapiClient

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TRSRV_PATH        = "/v1/api/trsrv/stocks"
_SEARCH_PATH       = "/v1/api/iserver/secdef/search"
_INTER_REQUEST_SEC = 0.25
_DEFAULT_SEC_TYPE  = "STK"
_CACHE_FILENAME    = "conid_cache.json"

# ---------------------------------------------------------------------------
# Ticker alias table
#
# Деякі тікери мають нестандартні символи (крапка, слеш) які IBKR API
# не розуміє напряму. Таблиця: {наш_тікер: [варіанти_для_lookup]}
#
# Правила lookup: перебираємо варіанти по порядку, беремо перший результат.
# Якщо жоден не спрацював — повертаємо None.
#
# BRK.B → пробуємо "BRK B" (IBKR internal), "BRKB" (деякі провайдери)
# BRK.A → аналогічно
# Додавайте нові alias за потреби.
# ---------------------------------------------------------------------------

_TICKER_ALIASES: Dict[str, List[str]] = {
    "BRK.B":  ["BRK B", "BRKB",  "BRK/B"],
    "BRK.A":  ["BRK A", "BRKA",  "BRK/A"],
    "BF.B":   ["BF B",  "BFB"],
    "BF.A":   ["BF A",  "BFA"],
    # Додайте інші нестандартні тікери тут
}


def _get_lookup_variants(symbol: str) -> List[str]:
    """
    Повертає список варіантів для lookup.
    Якщо є в таблиці alias — повертає alias + оригінал як fallback.
    Інакше повертає просто оригінал.
    """
    upper = symbol.upper().strip()
    if upper in _TICKER_ALIASES:
        variants = list(_TICKER_ALIASES[upper])
        # Додаємо оригінал якщо його ще немає — на випадок
        # якщо IBKR колись почне підтримувати крапковий формат
        if upper not in variants:
            variants.append(upper)
        return variants
    return [upper]


# ---------------------------------------------------------------------------
# Single-symbol lookup — /trsrv/stocks (primary)
# ---------------------------------------------------------------------------

def _lookup_trsrv(client: CpapiClient, symbol: str) -> Optional[str]:
    """
    GET /trsrv/stocks?symbols=SYMBOL
    Returns the conid of the first US STK contract.
    symbol тут вже нормалізований (після alias lookup).
    """
    try:
        raw = client._get(f"{_TRSRV_PATH}?symbols={symbol}")
    except Exception as exc:
        print(f"[CONID][{symbol}] trsrv request failed: {exc}")
        return None

    data    = raw if isinstance(raw, dict) else {}
    # Gateway повертає ключ рівно таким як відправили
    entries = data.get(symbol, data.get(symbol.upper(), []))
    if not isinstance(entries, list):
        return None

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("assetClass", "")).upper() != _DEFAULT_SEC_TYPE:
            continue
        for contract in entry.get("contracts", []):
            if not isinstance(contract, dict):
                continue
            if contract.get("isUS", False):
                conid = str(contract.get("conid", "") or "")
                if conid and conid != "0":
                    return conid

    return None


# ---------------------------------------------------------------------------
# Single-symbol lookup — /iserver/secdef/search (fallback)
# ---------------------------------------------------------------------------

def _lookup_secdef(client: CpapiClient, symbol: str) -> Optional[str]:
    """
    POST /iserver/secdef/search
    Fallback when trsrv returns nothing.
    conid is on the top-level row, not inside sections.
    symbol тут вже нормалізований (після alias lookup).
    """
    try:
        raw = client._post(
            _SEARCH_PATH,
            payload={"symbol": symbol, "secType": _DEFAULT_SEC_TYPE, "name": False},
        )
    except Exception as exc:
        print(f"[CONID][{symbol}] secdef/search failed: {exc}")
        return None

    rows = raw if isinstance(raw, list) else []
    for row in rows:
        if not isinstance(row, dict):
            continue
        row_symbol = str(row.get("symbol", "")).upper().strip()
        # Приймаємо якщо символ відповідає будь-якому з варіантів
        if row_symbol != symbol.upper().strip():
            continue
        conid = str(row.get("conid", "") or "")
        if conid and conid != "0":
            sections = row.get("sections", [])
            has_stk = any(
                str(s.get("secType", "")).upper() == "STK"
                for s in sections
                if isinstance(s, dict)
            )
            if has_stk:
                return conid

    return None


# ---------------------------------------------------------------------------
# Resolve one symbol (з підтримкою alias)
# ---------------------------------------------------------------------------

def _resolve_one(client: CpapiClient, symbol: str) -> Optional[str]:
    """
    Перебирає всі варіанти тікера (оригінал + alias) і повертає
    перший знайдений conid. Логує який саме варіант спрацював.
    """
    variants = _get_lookup_variants(symbol)
    if len(variants) > 1:
        print(f"[CONID][{symbol}] trying {len(variants)} variants: {variants}")

    for variant in variants:
        conid = _lookup_trsrv(client, variant)
        if conid:
            if variant != symbol.upper():
                print(f"[CONID][{symbol}] resolved via alias '{variant}' → {conid}")
            return conid
        time.sleep(_INTER_REQUEST_SEC)

        conid = _lookup_secdef(client, variant)
        if conid:
            if variant != symbol.upper():
                print(f"[CONID][{symbol}] resolved via secdef alias '{variant}' → {conid}")
            return conid
        time.sleep(_INTER_REQUEST_SEC)

    return None


# ---------------------------------------------------------------------------
# Batch resolver
# ---------------------------------------------------------------------------

def resolve_conids(
    client: CpapiClient,
    symbols: List[str],
    inter_request_sec: float = _INTER_REQUEST_SEC,
) -> Tuple[Dict[str, str], List[str]]:
    resolved:   Dict[str, str] = {}
    unresolved: List[str]      = []

    for symbol in symbols:
        conid = _resolve_one(client, symbol)
        if conid:
            # Зберігаємо під оригінальним символом (BRK.B, не "BRK B")
            resolved[symbol] = conid
            print(f"[CONID][{symbol}] → {conid}")
        else:
            unresolved.append(symbol)
            print(f"[CONID][{symbol}] NOT FOUND")
        time.sleep(inter_request_sec)

    print(
        f"[CONID][SUMMARY] "
        f"total={len(symbols)} resolved={len(resolved)} unresolved={len(unresolved)}"
    )
    if unresolved:
        print(f"[CONID][UNRESOLVED] {unresolved}")

    return resolved, unresolved


# ---------------------------------------------------------------------------
# Cache: load / save / merge
# ---------------------------------------------------------------------------

def load_conid_cache(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {}
        return {str(k): str(v) for k, v in data.items() if k and v}
    except Exception as exc:
        print(f"[CONID_CACHE] failed to load {path}: {exc}")
        return {}


def save_conid_cache(path: Path, cache: Dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(sorted(cache.items())), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[CONID_CACHE] saved {len(cache)} entries → {path}")


def update_conid_cache(
    cache_path: Path,
    client: CpapiClient,
    symbols: List[str],
    force_refresh: bool = False,
) -> Tuple[Dict[str, str], List[str]]:
    cache = {} if force_refresh else load_conid_cache(cache_path)
    missing = [s for s in symbols if s not in cache]

    if not missing:
        print(f"[CONID_CACHE] all {len(symbols)} symbols already cached")
        return cache, [s for s in symbols if not cache.get(s)]

    print(f"[CONID_CACHE] resolving {len(missing)} missing symbols: {missing}")
    new_resolved, _ = resolve_conids(client, missing)
    cache.update(new_resolved)
    save_conid_cache(cache_path, cache)

    final_unresolved = [s for s in symbols if not cache.get(s)]
    return cache, final_unresolved


# ---------------------------------------------------------------------------
# Convenience: resolve from orders.csv path
# ---------------------------------------------------------------------------

def resolve_for_orders_csv(
    client: CpapiClient,
    orders_csv: Path,
    force_refresh: bool = False,
) -> Tuple[Dict[str, str], List[str]]:
    if not orders_csv.exists():
        raise FileNotFoundError(f"orders.csv not found: {orders_csv}")

    import pandas as pd
    df = pd.read_csv(orders_csv)
    if "symbol" not in df.columns:
        raise RuntimeError(f"orders.csv has no 'symbol' column: {orders_csv}")

    symbols = (
        df["symbol"].dropna().astype(str).str.strip().str.upper().unique().tolist()
    )
    if "order_side" in df.columns:
        live = set(
            df.loc[df["order_side"].str.upper() != "HOLD", "symbol"]
            .dropna().astype(str).str.strip().str.upper().tolist()
        )
        symbols = [s for s in symbols if s in live]

    cache_path = orders_csv.parent / _CACHE_FILENAME
    return update_conid_cache(cache_path, client, symbols, force_refresh)
