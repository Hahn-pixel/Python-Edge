from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

DEFAULT_BASE_URL = "https://api.massive.com"


@dataclass(frozen=True)
class UniverseConfig:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    locale: str = "us"
    market: str = "stocks"
    ticker_type: str = "CS"
    min_price: float = 5.0
    min_dollar_volume: float = 10_000_000.0
    top_n: int = 500
    max_reference_pages: int = 50
    request_timeout_sec: int = 30
    output_dir: Path = Path("artifacts/daily_cycle/universe")
    as_of_date: Optional[str] = None
    history_check_mode: str = "skipped_skeleton"


class MassiveClient:
    def __init__(self, api_key: str, base_url: str = DEFAULT_BASE_URL, timeout_sec: int = 30):
        if not api_key:
            raise ValueError("MASSIVE_API_KEY is required")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout_sec = int(timeout_sec)
        self._session = requests.Session()

    def _request_json(self, url: str, params: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        final_params = dict(params or {})
        final_params.setdefault("apiKey", self._api_key)
        response = self._session.get(url, params=final_params, timeout=self._timeout_sec)
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict) and str(payload.get("status", "")).upper() in {"ERROR", "NOT_AUTHORIZED"}:
            raise RuntimeError(f"Massive API error for {url}: {payload}")
        return payload

    def iter_reference_tickers(self, locale: str, market: str, ticker_type: str, max_pages: int) -> List[Dict[str, object]]:
        url = f"{self._base_url}/v3/reference/tickers"
        params: Optional[Dict[str, object]] = {
            "active": "true",
            "locale": locale,
            "market": market,
            "type": ticker_type,
            "limit": 1000,
            "sort": "ticker",
            "order": "asc",
        }
        all_rows: List[Dict[str, object]] = []
        page_no = 0
        while url and page_no < max_pages:
            page_no += 1
            payload = self._request_json(url, params=params)
            rows = payload.get("results", [])
            if not isinstance(rows, list):
                raise RuntimeError(f"Unexpected tickers payload shape: {type(rows)!r}")
            all_rows.extend([row for row in rows if isinstance(row, dict)])
            next_url = payload.get("next_url")
            if next_url:
                url = str(next_url)
                params = None
            else:
                url = ""
        return all_rows

    def get_grouped_daily(self, trading_date: str, locale: str, market: str) -> List[Dict[str, object]]:
        url = f"{self._base_url}/v2/aggs/grouped/locale/{locale}/market/{market}/{trading_date}"
        payload = self._request_json(url, params={"adjusted": "true", "include_otc": "false"})
        rows = payload.get("results", [])
        if not isinstance(rows, list):
            raise RuntimeError(f"Unexpected grouped daily payload shape: {type(rows)!r}")
        return [row for row in rows if isinstance(row, dict)]



def load_config_from_env(root_dir: Path) -> UniverseConfig:
    api_key = str(os.getenv("MASSIVE_API_KEY", "")).strip()
    if not api_key:
        raise RuntimeError("MASSIVE_API_KEY is missing from env")
    output_dir = Path(os.getenv("UNIVERSE_OUTPUT_DIR", root_dir / "artifacts" / "daily_cycle" / "universe"))
    return UniverseConfig(
        api_key=api_key,
        base_url=str(os.getenv("MASSIVE_BASE_URL", DEFAULT_BASE_URL)).strip() or DEFAULT_BASE_URL,
        locale=str(os.getenv("UNIVERSE_LOCALE", "us")).strip().lower() or "us",
        market=str(os.getenv("UNIVERSE_MARKET", "stocks")).strip().lower() or "stocks",
        ticker_type=str(os.getenv("UNIVERSE_TICKER_TYPE", "CS")).strip().upper() or "CS",
        min_price=float(os.getenv("UNIVERSE_MIN_PRICE", "5.0")),
        min_dollar_volume=float(os.getenv("UNIVERSE_MIN_DOLLAR_VOL", "10000000")),
        top_n=int(os.getenv("UNIVERSE_TOP_N", "500")),
        max_reference_pages=int(os.getenv("UNIVERSE_MAX_REFERENCE_PAGES", "50")),
        request_timeout_sec=int(os.getenv("UNIVERSE_REQUEST_TIMEOUT_SEC", "30")),
        output_dir=output_dir,
        as_of_date=str(os.getenv("UNIVERSE_AS_OF_DATE", "")).strip() or None,
        history_check_mode=str(os.getenv("UNIVERSE_HISTORY_CHECK_MODE", "skipped_skeleton")).strip() or "skipped_skeleton",
    )



def resolve_latest_trading_date(client: MassiveClient, config: UniverseConfig) -> str:
    if config.as_of_date:
        return config.as_of_date
    today_utc = datetime.now(timezone.utc).date()
    for days_back in range(0, 10):
        candidate = (today_utc - timedelta(days=days_back)).isoformat()
        rows = client.get_grouped_daily(candidate, locale=config.locale, market=config.market)
        if rows:
            return candidate
    raise RuntimeError("Could not resolve a recent trading date from Massive grouped daily endpoint")



def _normalize_reference_frame(rows: List[Dict[str, object]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["ticker", "name", "primary_exchange", "type", "locale", "market", "active"])
    df = pd.DataFrame(rows).copy()
    if "ticker" not in df.columns:
        raise RuntimeError("Reference ticker payload does not contain 'ticker'")
    keep_cols = [col for col in ["ticker", "name", "primary_exchange", "type", "locale", "market", "active"] if col in df.columns]
    df = df.loc[:, keep_cols].copy()
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df = df.drop_duplicates(subset=["ticker"], keep="last").reset_index(drop=True)
    return df



def _normalize_grouped_daily_frame(rows: List[Dict[str, object]], trading_date: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["ticker", "close", "volume", "vwap", "dollar_volume", "trade_date"])
    df = pd.DataFrame(rows).copy()
    rename_map = {
        "T": "ticker",
        "c": "close",
        "v": "volume",
        "vw": "vwap",
        "o": "open",
        "h": "high",
        "l": "low",
        "n": "transactions",
        "t": "ts",
    }
    df = df.rename(columns=rename_map)
    needed = ["ticker", "close", "volume"]
    for col in needed:
        if col not in df.columns:
            raise RuntimeError(f"Grouped daily payload missing required column: {col}")
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    if "vwap" in df.columns:
        df["vwap"] = pd.to_numeric(df["vwap"], errors="coerce")
    else:
        df["vwap"] = pd.NA
    df["dollar_volume"] = df["close"] * df["volume"]
    df["trade_date"] = trading_date
    keep_cols = [col for col in ["ticker", "trade_date", "close", "volume", "vwap", "dollar_volume", "open", "high", "low", "transactions"] if col in df.columns]
    df = df.loc[:, keep_cols].copy()
    df = df.drop_duplicates(subset=["ticker"], keep="last").reset_index(drop=True)
    return df



def build_universe_snapshot(config: UniverseConfig) -> Tuple[pd.DataFrame, Dict[str, object]]:
    client = MassiveClient(api_key=config.api_key, base_url=config.base_url, timeout_sec=config.request_timeout_sec)
    trade_date = resolve_latest_trading_date(client, config)
    reference_rows = client.iter_reference_tickers(
        locale=config.locale,
        market=config.market,
        ticker_type=config.ticker_type,
        max_pages=config.max_reference_pages,
    )
    market_rows = client.get_grouped_daily(trade_date, locale=config.locale, market=config.market)

    ref_df = _normalize_reference_frame(reference_rows)
    px_df = _normalize_grouped_daily_frame(market_rows, trading_date=trade_date)
    merged = ref_df.merge(px_df, on="ticker", how="left", validate="one_to_one")

    merged["has_market_bar"] = merged["close"].notna() & merged["volume"].notna()
    merged["passes_price"] = merged["has_market_bar"] & (merged["close"] >= config.min_price)
    merged["passes_liquidity"] = merged["has_market_bar"] & (merged["dollar_volume"] >= config.min_dollar_volume)
    merged["passes_history"] = True
    merged["history_check_mode"] = config.history_check_mode
    merged["eligible"] = merged["passes_price"] & merged["passes_liquidity"] & merged["passes_history"]

    merged["drop_reason"] = ""
    merged.loc[~merged["has_market_bar"], "drop_reason"] = "missing_grouped_daily_bar"
    merged.loc[merged["has_market_bar"] & ~merged["passes_price"], "drop_reason"] = "price_below_threshold"
    merged.loc[merged["has_market_bar"] & merged["passes_price"] & ~merged["passes_liquidity"], "drop_reason"] = "dollar_volume_below_threshold"
    merged.loc[merged["eligible"], "drop_reason"] = "eligible"

    eligible_df = merged.loc[merged["eligible"]].copy()
    eligible_df = eligible_df.sort_values(["dollar_volume", "ticker"], ascending=[False, True]).reset_index(drop=True)
    selected_tickers = set(eligible_df.head(config.top_n)["ticker"].astype(str))
    merged["selected"] = merged["ticker"].isin(selected_tickers)

    selected_df = merged.loc[merged["selected"]].copy()
    selected_df = selected_df.sort_values(["dollar_volume", "ticker"], ascending=[False, True]).reset_index(drop=True)

    summary = {
        "as_of_date": trade_date,
        "base_url": config.base_url,
        "locale": config.locale,
        "market": config.market,
        "ticker_type": config.ticker_type,
        "history_check_mode": config.history_check_mode,
        "min_price": config.min_price,
        "min_dollar_volume": config.min_dollar_volume,
        "top_n": config.top_n,
        "counters": {
            "candidates_total": int(len(merged)),
            "missing_grouped_daily_bar": int((~merged["has_market_bar"]).sum()),
            "dropped_price": int((merged["has_market_bar"] & ~merged["passes_price"]).sum()),
            "dropped_liquidity": int((merged["has_market_bar"] & merged["passes_price"] & ~merged["passes_liquidity"]).sum()),
            "dropped_history": int((~merged["passes_history"]).sum()),
            "eligible_total": int(merged["eligible"].sum()),
            "selected_total": int(merged["selected"].sum()),
        },
        "selected_preview": selected_df.head(20)[["ticker", "close", "dollar_volume", "primary_exchange", "name"]].fillna("").to_dict(orient="records"),
    }
    return merged, summary



def save_universe_outputs(snapshot_df: pd.DataFrame, summary: Dict[str, object], output_dir: Path) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / "universe_snapshot.parquet"
    summary_path = output_dir / "universe_summary.json"
    snapshot_df.to_parquet(parquet_path, index=False)
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    return parquet_path, summary_path