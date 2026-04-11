from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests


DEFAULT_BASE_URL = "https://api.massive.com"
DEFAULT_LOCALE = "us"
DEFAULT_MARKET = "stocks"
DEFAULT_TICKER_TYPE = "CS"
DEFAULT_OUTPUT_ROOT = Path("artifacts/daily_cycle/universe")
DEFAULT_PROFILE = "trading_us_core_v1"


@dataclass(frozen=True)
class UniverseEligibilityPolicy:
    min_price: float = 7.50
    min_median_dollar_vol_20d: float = 20_000_000.0
    min_history_days: int = 25
    max_nan_ratio: float = 0.02
    max_missing_days_20d: int = 1
    allowed_ticker_types: Tuple[str, ...] = ("CS",)
    allowed_primary_exchanges: Tuple[str, ...] = tuple()
    allowed_exchanges: Tuple[str, ...] = tuple()
    require_active: bool = True


@dataclass(frozen=True)
class UniverseConfig:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    locale: str = DEFAULT_LOCALE
    market: str = DEFAULT_MARKET
    ticker_type: str = DEFAULT_TICKER_TYPE
    universe_profile: str = DEFAULT_PROFILE
    target_size: int = 175
    sector_cap: int = 18
    top_n: int = 175
    shortlist_multiplier: int = 3
    history_lookback_days: int = 45
    history_batch_limit: int = 600
    output_dir: Path = DEFAULT_OUTPUT_ROOT
    request_timeout_sec: float = 30.0
    request_sleep_sec: float = 0.2
    request_max_retries: int = 3
    request_backoff_sec: float = 1.0
    overview_enrichment_enabled: bool = True
    overview_enrichment_max: int = 1200
    rebalance_freq: str = "weekly"
    reuse_last: bool = True
    history_check_mode: str = "daily_aggs_lookback"
    eligibility: UniverseEligibilityPolicy = field(default_factory=UniverseEligibilityPolicy)


class UniverseBuildError(RuntimeError):
    pass


class MassiveClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        timeout_sec: float,
        sleep_sec: float,
        max_retries: int,
        backoff_sec: float,
    ) -> None:
        self.api_key = str(api_key).strip()
        self.base_url = str(base_url).rstrip("/")
        self.timeout_sec = float(timeout_sec)
        self.sleep_sec = float(sleep_sec)
        self.max_retries = max(1, int(max_retries))
        self.backoff_sec = float(backoff_sec)
        self.session = requests.Session()

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.api_key:
            raise UniverseBuildError("MASSIVE_API_KEY is required")
        query = dict(params or {})
        query["apiKey"] = self.api_key
        url = f"{self.base_url}{path}"
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.get(url, params=query, timeout=self.timeout_sec)
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise UniverseBuildError(f"Unexpected payload type from {url}: {type(payload)!r}")
                time.sleep(max(0.0, self.sleep_sec))
                return payload
            except Exception as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(max(0.0, self.backoff_sec) * attempt)
        raise UniverseBuildError(f"Massive GET failed path={path} params={query}: {last_exc}") from last_exc

    def list_tickers(self, locale: str, market: str, ticker_type: str) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        cursor: Optional[str] = None
        while True:
            params: Dict[str, Any] = {
                "locale": locale,
                "market": market,
                "type": ticker_type,
                "active": "true",
                "limit": 1000,
            }
            if cursor:
                params["cursor"] = cursor
            payload = self._get("/v3/reference/tickers", params)
            batch = payload.get("results", [])
            if isinstance(batch, list):
                rows.extend(x for x in batch if isinstance(x, dict))
            next_url = str(payload.get("next_url", "") or "").strip()
            if not next_url:
                break
            if "cursor=" in next_url:
                cursor = next_url.split("cursor=", 1)[1].split("&", 1)[0]
            else:
                break
        df = pd.DataFrame(rows)
        if df.empty:
            raise UniverseBuildError("Massive returned no tickers")
        return df

    def grouped_daily_snapshot(self, locale: str, market: str) -> pd.DataFrame:
        trade_date = pd.Timestamp.now(tz="UTC").date()
        for _ in range(10):
            date_str = trade_date.isoformat()
            payload = self._get(
                "/v2/aggs/grouped/locale/{}/market/{}/{}".format(locale, market, date_str),
                {"adjusted": "true"},
            )
            rows = payload.get("results", [])
            df = pd.DataFrame(rows if isinstance(rows, list) else [])
            if not df.empty:
                return df
            trade_date = trade_date - timedelta(days=1)
        raise UniverseBuildError("Massive returned no grouped daily rows for recent trading dates")

    def ticker_overview(self, symbol: str) -> Dict[str, Any]:
        return self._get(f"/v3/reference/tickers/{symbol}", {"active": "true"})

    def daily_history(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        payload = self._get(
            f"/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}",
            {"adjusted": "true", "sort": "asc", "limit": 5000},
        )
        rows = payload.get("results", [])
        df = pd.DataFrame(rows if isinstance(rows, list) else [])
        if df.empty:
            return pd.DataFrame(columns=["symbol", "trade_date", "close", "volume", "dollar_volume"])
        out = pd.DataFrame()
        out["symbol"] = str(symbol).upper()
        out["timestamp_ms"] = pd.to_numeric(df.get("t", pd.Series(dtype="float64")), errors="coerce")
        ts = pd.to_datetime(out["timestamp_ms"], unit="ms", utc=True, errors="coerce")
        out["trade_date"] = ts.dt.strftime("%Y-%m-%d")
        out["close"] = pd.to_numeric(df.get("c", pd.Series(dtype="float64")), errors="coerce")
        out["volume"] = pd.to_numeric(df.get("v", pd.Series(dtype="float64")), errors="coerce")
        out["open"] = pd.to_numeric(df.get("o", pd.Series(dtype="float64")), errors="coerce")
        out["high"] = pd.to_numeric(df.get("h", pd.Series(dtype="float64")), errors="coerce")
        out["low"] = pd.to_numeric(df.get("l", pd.Series(dtype="float64")), errors="coerce")
        out["dollar_volume"] = out["close"].fillna(0.0) * out["volume"].fillna(0.0)
        return out



def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")



def _env_flag(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, "1" if default else "0")).strip().lower()
    return raw not in {"0", "false", "no", "off", ""}



def _env_float(name: str, default: float) -> float:
    raw = str(os.getenv(name, str(default))).strip()
    try:
        return float(raw)
    except Exception as exc:
        raise UniverseBuildError(f"Invalid float env {name}={raw!r}") from exc



def _env_int(name: str, default: int) -> int:
    raw = str(os.getenv(name, str(default))).strip()
    try:
        return int(raw)
    except Exception as exc:
        raise UniverseBuildError(f"Invalid int env {name}={raw!r}") from exc



def _env_tuple(name: str, default: Tuple[str, ...]) -> Tuple[str, ...]:
    raw = str(os.getenv(name, "")).strip()
    if not raw:
        return default
    items = [x.strip().upper() for x in raw.split("|") if x.strip()]
    return tuple(items)



def load_config_from_env(_root: Path | None = None) -> UniverseConfig:
    api_key = str(os.getenv("MASSIVE_API_KEY", "")).strip()
    default_ticker_type = str(os.getenv("UNIVERSE_TICKER_TYPE", DEFAULT_TICKER_TYPE)).strip().upper() or DEFAULT_TICKER_TYPE
    eligibility = UniverseEligibilityPolicy(
        min_price=_env_float("UNIVERSE_MIN_PRICE", 7.50),
        min_median_dollar_vol_20d=_env_float("UNIVERSE_MIN_MEDIAN_DOLLAR_VOL_20D", 20_000_000.0),
        min_history_days=_env_int("UNIVERSE_MIN_HISTORY_DAYS", 25),
        max_nan_ratio=_env_float("UNIVERSE_MAX_NAN_RATIO", 0.02),
        max_missing_days_20d=_env_int("UNIVERSE_MAX_MISSING_DAYS_20D", 1),
        allowed_ticker_types=_env_tuple("UNIVERSE_ALLOWED_TICKER_TYPES", (default_ticker_type,)),
        allowed_primary_exchanges=_env_tuple("UNIVERSE_ALLOWED_PRIMARY_EXCHANGES", tuple()),
        allowed_exchanges=_env_tuple("UNIVERSE_ALLOWED_EXCHANGES", tuple()),
        require_active=_env_flag("UNIVERSE_REQUIRE_ACTIVE", True),
    )
    output_dir = Path(str(os.getenv("UNIVERSE_OUTPUT_DIR", str(DEFAULT_OUTPUT_ROOT))).strip() or str(DEFAULT_OUTPUT_ROOT))
    target_size = _env_int("UNIVERSE_TARGET_SIZE", 175)
    return UniverseConfig(
        api_key=api_key,
        base_url=str(os.getenv("MASSIVE_BASE_URL", DEFAULT_BASE_URL)).strip() or DEFAULT_BASE_URL,
        locale=str(os.getenv("UNIVERSE_LOCALE", DEFAULT_LOCALE)).strip() or DEFAULT_LOCALE,
        market=str(os.getenv("UNIVERSE_MARKET", DEFAULT_MARKET)).strip() or DEFAULT_MARKET,
        ticker_type=default_ticker_type,
        universe_profile=str(os.getenv("UNIVERSE_PROFILE", DEFAULT_PROFILE)).strip() or DEFAULT_PROFILE,
        target_size=target_size,
        sector_cap=_env_int("UNIVERSE_SECTOR_CAP", 18),
        top_n=_env_int("UNIVERSE_TOP_N", target_size),
        shortlist_multiplier=_env_int("UNIVERSE_SHORTLIST_MULTIPLIER", 3),
        history_lookback_days=_env_int("UNIVERSE_HISTORY_LOOKBACK_DAYS", 45),
        history_batch_limit=_env_int("UNIVERSE_HISTORY_BATCH_LIMIT", 600),
        output_dir=output_dir,
        request_timeout_sec=_env_float("UNIVERSE_REQUEST_TIMEOUT_SEC", 30.0),
        request_sleep_sec=_env_float("UNIVERSE_REQUEST_SLEEP_SEC", 0.2),
        request_max_retries=_env_int("UNIVERSE_REQUEST_MAX_RETRIES", 3),
        request_backoff_sec=_env_float("UNIVERSE_REQUEST_BACKOFF_SEC", 1.0),
        overview_enrichment_enabled=_env_flag("UNIVERSE_OVERVIEW_ENRICHMENT_ENABLED", True),
        overview_enrichment_max=_env_int("UNIVERSE_OVERVIEW_ENRICHMENT_MAX", 1200),
        rebalance_freq=str(os.getenv("UNIVERSE_REBALANCE_FREQ", "weekly")).strip() or "weekly",
        reuse_last=_env_flag("UNIVERSE_REUSE_LAST", True),
        history_check_mode=str(os.getenv("UNIVERSE_HISTORY_CHECK_MODE", "daily_aggs_lookback")).strip() or "daily_aggs_lookback",
        eligibility=eligibility,
    )



def _normalize_ticker_reference(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename_map = {
        "ticker": "symbol",
        "primary_exchange": "primary_exchange",
        "market": "market",
        "locale": "locale",
        "type": "ticker_type",
        "active": "active",
        "name": "name",
        "currency_name": "currency_name",
    }
    for src, dst in rename_map.items():
        if src in out.columns and src != dst:
            out[dst] = out[src]
    if "symbol" not in out.columns and "ticker" in out.columns:
        out["symbol"] = out["ticker"]
    out["symbol"] = out.get("symbol", pd.Series(dtype="object")).astype(str).str.upper()
    if "active" in out.columns:
        out["active"] = out["active"].fillna(False).astype(bool)
    else:
        out["active"] = True
    if "ticker_type" not in out.columns:
        out["ticker_type"] = ""
    out["ticker_type"] = out["ticker_type"].astype(str).str.upper()
    if "primary_exchange" not in out.columns:
        out["primary_exchange"] = ""
    out["primary_exchange"] = out["primary_exchange"].astype(str).str.upper()
    if "market" not in out.columns:
        out["market"] = ""
    if "locale" not in out.columns:
        out["locale"] = ""
    return out



def _normalize_grouped_daily(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename_map = {
        "T": "symbol",
        "c": "close",
        "h": "high",
        "l": "low",
        "o": "open",
        "v": "volume",
        "vw": "vwap",
        "t": "timestamp_ms",
        "n": "transactions",
    }
    for src, dst in rename_map.items():
        if src in out.columns and dst not in out.columns:
            out[dst] = out[src]
    if "symbol" not in out.columns:
        raise UniverseBuildError("Grouped daily payload missing symbol/T column")
    out["symbol"] = out["symbol"].astype(str).str.upper()
    for col in ["close", "open", "high", "low", "volume", "vwap"]:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if "timestamp_ms" not in out.columns:
        out["timestamp_ms"] = pd.NA
    ts = pd.to_datetime(out["timestamp_ms"], unit="ms", utc=True, errors="coerce")
    fallback_date = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    out["trade_date"] = ts.dt.strftime("%Y-%m-%d").fillna(fallback_date)
    out["as_of_date"] = out["trade_date"]
    out["date"] = out["trade_date"]
    out["session_date"] = out["trade_date"]
    out["dollar_volume_1d"] = out["close"].fillna(0.0) * out["volume"].fillna(0.0)
    out["median_dollar_volume_20d"] = out["dollar_volume_1d"]
    out["history_days"] = 1
    out["nan_ratio"] = 1.0
    out["missing_days_20d"] = 999
    return out



def _extract_overview_result(payload: Dict[str, Any]) -> Dict[str, Any]:
    result = payload.get("results", payload)
    if not isinstance(result, dict):
        return {}
    return result



def _enrich_with_overview(client: MassiveClient, df: pd.DataFrame, enabled: bool, max_rows: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if not enabled:
        return df.copy(), {"overview_enrichment_enabled": 0, "overview_enriched": 0, "overview_errors": 0}
    out = df.copy()
    overview_rows: List[Dict[str, Any]] = []
    enriched = 0
    errors = 0
    limit = min(int(max_rows), len(out))
    for symbol in out.head(limit)["symbol"].astype(str):
        try:
            payload = client.ticker_overview(symbol)
            item = _extract_overview_result(payload)
            overview_rows.append({
                "symbol": symbol,
                "overview_primary_exchange": str(item.get("primary_exchange", "") or "").upper(),
                "overview_name": str(item.get("name", "") or ""),
                "overview_active": bool(item.get("active", False)),
                "overview_market": str(item.get("market", "") or ""),
                "overview_locale": str(item.get("locale", "") or ""),
                "overview_type": str(item.get("type", "") or "").upper(),
                "overview_currency_name": str(item.get("currency_name", "") or ""),
                "overview_sic_description": str(item.get("sic_description", "") or ""),
                "overview_branding_icon_url": str(((item.get("branding") or {}) if isinstance(item.get("branding"), dict) else {}).get("icon_url", "") or ""),
            })
            enriched += 1
        except Exception:
            errors += 1
    if overview_rows:
        out = out.merge(pd.DataFrame(overview_rows), on="symbol", how="left")
        out["primary_exchange"] = out["overview_primary_exchange"].fillna(out["primary_exchange"])
        out["active"] = out["overview_active"].fillna(out["active"])
        out["ticker_type"] = out["overview_type"].fillna(out["ticker_type"])
        if "name" in out.columns:
            out["name"] = out["overview_name"].fillna(out["name"])
        else:
            out["name"] = out["overview_name"]
    return out, {
        "overview_enrichment_enabled": 1,
        "overview_enriched": enriched,
        "overview_errors": errors,
    }



def _passes_allowed(value: str, allowed: Iterable[str]) -> bool:
    allowed_norm = [str(x).strip().upper() for x in allowed if str(x).strip()]
    if not allowed_norm:
        return True
    return str(value or "").strip().upper() in set(allowed_norm)



def _business_days_in_window(end_date_str: str, periods: int) -> List[str]:
    end_ts = pd.Timestamp(end_date_str)
    dates = pd.bdate_range(end=end_ts, periods=max(1, periods))
    return [x.strftime("%Y-%m-%d") for x in dates]



def _compute_history_metrics(client: MassiveClient, symbols: List[str], end_date: str, lookback_days: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    end_ts = pd.Timestamp(end_date)
    start_ts = end_ts - pd.Timedelta(days=max(lookback_days, 20) * 2)
    start_date = start_ts.strftime("%Y-%m-%d")
    expected_recent_20 = set(_business_days_in_window(end_date, 20))

    for symbol in symbols:
        hist = client.daily_history(symbol, start_date, end_date)
        if hist.empty:
            rows.append({
                "symbol": symbol,
                "history_days": 0,
                "median_dollar_volume_20d": 0.0,
                "nan_ratio": 1.0,
                "missing_days_20d": len(expected_recent_20),
            })
            continue

        hist = hist.copy()
        hist = hist.drop_duplicates(subset=["trade_date"]).sort_values("trade_date")
        history_days = int(len(hist))
        recent20 = hist.tail(20).copy()
        present_dates = set(recent20["trade_date"].dropna().astype(str).tolist())
        missing_days_20d = int(len(expected_recent_20 - present_dates))

        critical_cols = ["trade_date", "close", "volume", "dollar_volume"]
        nan_ratio = float(recent20[critical_cols].isna().mean().mean()) if not recent20.empty else 1.0
        median_dollar_volume_20d = float(pd.to_numeric(recent20["dollar_volume"], errors="coerce").dropna().median()) if not recent20.empty else 0.0
        if pd.isna(median_dollar_volume_20d):
            median_dollar_volume_20d = 0.0

        rows.append({
            "symbol": symbol,
            "history_days": history_days,
            "median_dollar_volume_20d": median_dollar_volume_20d,
            "nan_ratio": nan_ratio,
            "missing_days_20d": missing_days_20d,
        })

    return pd.DataFrame(rows)



def apply_eligibility_policy(df: pd.DataFrame, policy: UniverseEligibilityPolicy) -> Tuple[pd.DataFrame, Dict[str, int]]:
    out = df.copy()
    out["passes_active"] = True if not policy.require_active else out["active"].fillna(False).astype(bool)
    out["passes_ticker_type"] = out["ticker_type"].astype(str).str.upper().map(lambda x: _passes_allowed(x, policy.allowed_ticker_types))
    out["passes_primary_exchange"] = out["primary_exchange"].astype(str).str.upper().map(lambda x: _passes_allowed(x, policy.allowed_primary_exchanges))
    exchange_col = out["primary_exchange"] if "primary_exchange" in out.columns else pd.Series([""] * len(out), index=out.index)
    out["passes_exchange"] = exchange_col.astype(str).str.upper().map(lambda x: _passes_allowed(x, policy.allowed_exchanges))
    out["passes_price"] = pd.to_numeric(out["close"], errors="coerce").fillna(0.0) >= float(policy.min_price)
    out["passes_liquidity"] = pd.to_numeric(out["median_dollar_volume_20d"], errors="coerce").fillna(0.0) >= float(policy.min_median_dollar_vol_20d)
    out["passes_history"] = pd.to_numeric(out["history_days"], errors="coerce").fillna(0).astype(int) >= int(policy.min_history_days)
    out["passes_nan_ratio"] = pd.to_numeric(out["nan_ratio"], errors="coerce").fillna(1.0) <= float(policy.max_nan_ratio)
    out["passes_missing_days"] = pd.to_numeric(out["missing_days_20d"], errors="coerce").fillna(999).astype(int) <= int(policy.max_missing_days_20d)

    rule_cols = [
        "passes_active",
        "passes_ticker_type",
        "passes_primary_exchange",
        "passes_exchange",
        "passes_price",
        "passes_liquidity",
        "passes_history",
        "passes_nan_ratio",
        "passes_missing_days",
    ]
    out["eligible"] = out[rule_cols].all(axis=1)

    def _drop_reason(row: pd.Series) -> str:
        if bool(row.get("eligible", False)):
            return ""
        for key in rule_cols:
            if not bool(row.get(key, False)):
                return key.replace("passes_", "")
        return "unknown"

    out["drop_reason"] = out.apply(_drop_reason, axis=1)

    counters = {
        "candidates_total": int(len(out)),
        "eligible_total": int(out["eligible"].sum()),
        "dropped_active": int((~out["passes_active"]).sum()),
        "dropped_ticker_type": int((~out["passes_ticker_type"]).sum()),
        "dropped_primary_exchange": int((~out["passes_primary_exchange"]).sum()),
        "dropped_exchange": int((~out["passes_exchange"]).sum()),
        "dropped_price": int((~out["passes_price"]).sum()),
        "dropped_liquidity": int((~out["passes_liquidity"]).sum()),
        "dropped_history": int((~out["passes_history"]).sum()),
        "dropped_nan_ratio": int((~out["passes_nan_ratio"]).sum()),
        "dropped_missing_days": int((~out["passes_missing_days"]).sum()),
    }
    return out, counters



def _apply_sector_cap(df: pd.DataFrame, sector_cap: int, target_size: int) -> pd.DataFrame:
    out = df.copy()
    if "overview_sic_description" not in out.columns:
        out["overview_sic_description"] = "UNKNOWN"
    out["sector_key"] = out["overview_sic_description"].fillna("UNKNOWN").astype(str).replace({"": "UNKNOWN"})
    out = out.sort_values(["median_dollar_volume_20d", "dollar_volume_1d", "close", "symbol"], ascending=[False, False, False, True]).copy()
    out["sector_rank"] = out.groupby("sector_key").cumcount() + 1
    out = out.loc[out["sector_rank"] <= max(1, int(sector_cap))].copy()
    out = out.head(max(1, int(target_size))).copy()
    out["selected"] = True
    return out



def _build_summary(config: UniverseConfig, counters: Dict[str, int], extra: Dict[str, Any], selected: pd.DataFrame) -> Dict[str, Any]:
    latest_trade_date = ""
    if "trade_date" in selected.columns and not selected.empty:
        latest_trade_date = str(selected["trade_date"].iloc[0])
    summary: Dict[str, Any] = {
        "ts_utc": _utc_now_iso(),
        "universe_profile": config.universe_profile,
        "base_url": config.base_url,
        "locale": config.locale,
        "market": config.market,
        "ticker_type": config.ticker_type,
        "target_size": int(config.target_size),
        "top_n": int(config.top_n),
        "shortlist_multiplier": int(config.shortlist_multiplier),
        "history_lookback_days": int(config.history_lookback_days),
        "history_batch_limit": int(config.history_batch_limit),
        "sector_cap": int(config.sector_cap),
        "history_check_mode": config.history_check_mode,
        "rebalance_freq": config.rebalance_freq,
        "reuse_last": int(bool(config.reuse_last)),
        "latest_trade_date": latest_trade_date,
        **counters,
        **extra,
        "selected_total": int(len(selected)),
        "eligibility_policy": asdict(config.eligibility),
    }
    return summary



def build_universe_snapshot(config: UniverseConfig) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    client = MassiveClient(
        api_key=config.api_key,
        base_url=config.base_url,
        timeout_sec=config.request_timeout_sec,
        sleep_sec=config.request_sleep_sec,
        max_retries=config.request_max_retries,
        backoff_sec=config.request_backoff_sec,
    )

    tickers_df = _normalize_ticker_reference(client.list_tickers(config.locale, config.market, config.ticker_type))
    grouped_df = _normalize_grouped_daily(client.grouped_daily_snapshot(config.locale, config.market))

    merged = tickers_df.merge(grouped_df, on="symbol", how="inner", suffixes=("", "_grouped"))
    if merged.empty:
        raise UniverseBuildError("Universe merge produced zero rows")

    merged, overview_summary = _enrich_with_overview(
        client,
        merged,
        enabled=config.overview_enrichment_enabled,
        max_rows=config.overview_enrichment_max,
    )

    pre_ranked = merged.sort_values(
        ["dollar_volume_1d", "close", "symbol"],
        ascending=[False, False, True],
    ).copy()
    shortlist_size = min(len(pre_ranked), max(config.target_size, config.top_n) * max(1, int(config.shortlist_multiplier)))
    shortlist_size = min(shortlist_size, max(1, int(config.history_batch_limit)))
    shortlist = pre_ranked.head(shortlist_size).copy()

    latest_trade_date = str(shortlist["trade_date"].iloc[0]) if not shortlist.empty else pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    history_metrics = _compute_history_metrics(
        client=client,
        symbols=shortlist["symbol"].astype(str).tolist(),
        end_date=latest_trade_date,
        lookback_days=config.history_lookback_days,
    )

    merged = merged.drop(columns=["median_dollar_volume_20d", "history_days", "nan_ratio", "missing_days_20d"], errors="ignore")
    merged = merged.merge(history_metrics, on="symbol", how="left")
    merged["median_dollar_volume_20d"] = pd.to_numeric(merged["median_dollar_volume_20d"], errors="coerce").fillna(0.0)
    merged["history_days"] = pd.to_numeric(merged["history_days"], errors="coerce").fillna(0).astype(int)
    merged["nan_ratio"] = pd.to_numeric(merged["nan_ratio"], errors="coerce").fillna(1.0)
    merged["missing_days_20d"] = pd.to_numeric(merged["missing_days_20d"], errors="coerce").fillna(999).astype(int)

    eligible_df, counters = apply_eligibility_policy(merged, config.eligibility)
    ranked_eligible = eligible_df.loc[eligible_df["eligible"]].copy()
    ranked_eligible = ranked_eligible.sort_values(
        ["median_dollar_volume_20d", "dollar_volume_1d", "close", "symbol"],
        ascending=[False, False, False, True],
    ).copy()

    selected = _apply_sector_cap(ranked_eligible, config.sector_cap, config.target_size)
    selected["selected_rank"] = range(1, len(selected) + 1)
    if "trade_date" not in selected.columns:
        fallback_date = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
        selected["trade_date"] = fallback_date
        selected["as_of_date"] = fallback_date
        selected["date"] = fallback_date
        selected["session_date"] = fallback_date
    summary = _build_summary(
        config,
        counters,
        {
            **overview_summary,
            "shortlist_size": int(shortlist_size),
            "history_metrics_symbols": int(len(history_metrics)),
        },
        selected,
    )
    return selected.reset_index(drop=True), summary, eligible_df.reset_index(drop=True)



def build_and_save_universe_snapshot(config: UniverseConfig) -> Tuple[Path, Path]:
    selected, summary, eligible_df = build_universe_snapshot(config)

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = output_dir / "universe_snapshot.parquet"
    eligible_path = output_dir / "universe_eligibility_debug.parquet"
    summary_path = output_dir / "universe_summary.json"

    selected.to_parquet(snapshot_path, index=False)
    eligible_df.to_parquet(eligible_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    return snapshot_path, summary_path


UniversePolicy = UniverseEligibilityPolicy


__all__ = [
    "UniverseBuildError",
    "UniverseConfig",
    "UniverseEligibilityPolicy",
    "UniversePolicy",
    "apply_eligibility_policy",
    "build_and_save_universe_snapshot",
    "build_universe_snapshot",
    "load_config_from_env",
]


if __name__ == "__main__":
    try:
        cfg = load_config_from_env()
        snapshot_path, summary_path = build_and_save_universe_snapshot(cfg)
        print(json.dumps({
            "snapshot_path": str(snapshot_path),
            "summary_path": str(summary_path),
        }, indent=2, ensure_ascii=False))
    except Exception:
        traceback.print_exc()
        try:
            input("Press Enter to exit...")
        except Exception:
            pass
        raise
    try:
        input("Press Enter to exit...")
    except Exception:
        pass
