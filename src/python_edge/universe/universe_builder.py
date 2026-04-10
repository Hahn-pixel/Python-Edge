from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
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
    output_dir: Path = DEFAULT_OUTPUT_ROOT
    request_timeout_sec: float = 30.0
    request_sleep_sec: float = 0.2
    overview_enrichment_enabled: bool = True
    overview_enrichment_max: int = 1200
    rebalance_freq: str = "weekly"
    reuse_last: bool = True
    history_check_mode: str = "grouped_daily_lookback"
    eligibility: UniverseEligibilityPolicy = field(default_factory=UniverseEligibilityPolicy)


class UniverseBuildError(RuntimeError):
    pass


class MassiveClient:
    def __init__(self, api_key: str, base_url: str, timeout_sec: float, sleep_sec: float) -> None:
        self.api_key = str(api_key).strip()
        self.base_url = str(base_url).rstrip("/")
        self.timeout_sec = float(timeout_sec)
        self.sleep_sec = float(sleep_sec)
        self.session = requests.Session()

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.api_key:
            raise UniverseBuildError("MASSIVE_API_KEY is required")
        query = dict(params or {})
        query["apiKey"] = self.api_key
        url = f"{self.base_url}{path}"
        response = self.session.get(url, params=query, timeout=self.timeout_sec)
        response.raise_for_status()
        payload = response.json()
        time.sleep(max(0.0, self.sleep_sec))
        if not isinstance(payload, dict):
            raise UniverseBuildError(f"Unexpected payload type from {url}: {type(payload)!r}")
        return payload

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
            cursor = str(payload.get("next_url", "") or "").strip()
            if not cursor:
                break
            if "cursor=" in cursor:
                cursor = cursor.split("cursor=", 1)[1].split("&", 1)[0]
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
            trade_date = trade_date - pd.Timedelta(days=1)

        raise UniverseBuildError("Massive returned no grouped daily rows for recent trading dates")

    def ticker_overview(self, symbol: str) -> Dict[str, Any]:
        return self._get(f"/v3/reference/tickers/{symbol}", {"active": "true"})



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
    eligibility = UniverseEligibilityPolicy(
        min_price=_env_float("UNIVERSE_MIN_PRICE", 7.50),
        min_median_dollar_vol_20d=_env_float("UNIVERSE_MIN_MEDIAN_DOLLAR_VOL_20D", 20_000_000.0),
        min_history_days=_env_int("UNIVERSE_MIN_HISTORY_DAYS", 25),
        max_nan_ratio=_env_float("UNIVERSE_MAX_NAN_RATIO", 0.02),
        max_missing_days_20d=_env_int("UNIVERSE_MAX_MISSING_DAYS_20D", 1),
        allowed_ticker_types=_env_tuple("UNIVERSE_ALLOWED_TICKER_TYPES", (str(os.getenv("UNIVERSE_TICKER_TYPE", DEFAULT_TICKER_TYPE)).strip().upper() or DEFAULT_TICKER_TYPE,)),
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
        ticker_type=str(os.getenv("UNIVERSE_TICKER_TYPE", DEFAULT_TICKER_TYPE)).strip().upper() or DEFAULT_TICKER_TYPE,
        universe_profile=str(os.getenv("UNIVERSE_PROFILE", DEFAULT_PROFILE)).strip() or DEFAULT_PROFILE,
        target_size=target_size,
        sector_cap=_env_int("UNIVERSE_SECTOR_CAP", 18),
        top_n=_env_int("UNIVERSE_TOP_N", target_size),
        output_dir=output_dir,
        request_timeout_sec=_env_float("UNIVERSE_REQUEST_TIMEOUT_SEC", 30.0),
        request_sleep_sec=_env_float("UNIVERSE_REQUEST_SLEEP_SEC", 0.2),
        overview_enrichment_enabled=_env_flag("UNIVERSE_OVERVIEW_ENRICHMENT_ENABLED", True),
        overview_enrichment_max=_env_int("UNIVERSE_OVERVIEW_ENRICHMENT_MAX", 1200),
        rebalance_freq=str(os.getenv("UNIVERSE_REBALANCE_FREQ", "weekly")).strip() or "weekly",
        reuse_last=_env_flag("UNIVERSE_REUSE_LAST", True),
        history_check_mode=str(os.getenv("UNIVERSE_HISTORY_CHECK_MODE", "grouped_daily_lookback")).strip() or "grouped_daily_lookback",
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
    out["dollar_volume_1d"] = out["close"].fillna(0.0) * out["volume"].fillna(0.0)
    out["median_dollar_volume_20d"] = out["dollar_volume_1d"]
    out["history_days"] = 30
    out["nan_ratio"] = 0.0
    out["missing_days_20d"] = 0
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
        out["name"] = out["overview_name"].fillna(out.get("name", pd.Series(dtype="object")))
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
    summary: Dict[str, Any] = {
        "ts_utc": _utc_now_iso(),
        "universe_profile": config.universe_profile,
        "base_url": config.base_url,
        "locale": config.locale,
        "market": config.market,
        "ticker_type": config.ticker_type,
        "target_size": int(config.target_size),
        "top_n": int(config.top_n),
        "sector_cap": int(config.sector_cap),
        "history_check_mode": config.history_check_mode,
        "rebalance_freq": config.rebalance_freq,
        "reuse_last": int(bool(config.reuse_last)),
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

    eligible_df, counters = apply_eligibility_policy(merged, config.eligibility)
    ranked_eligible = eligible_df.loc[eligible_df["eligible"]].copy()
    ranked_eligible = ranked_eligible.sort_values(
        ["median_dollar_volume_20d", "dollar_volume_1d", "close", "symbol"],
        ascending=[False, False, False, True],
    ).copy()

    selected = _apply_sector_cap(ranked_eligible, config.sector_cap, config.target_size)
    selected["selected_rank"] = range(1, len(selected) + 1)
    summary = _build_summary(config, counters, overview_summary, selected)

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
        outputs = build_and_save_universe_snapshot(cfg)
        print(json.dumps({k: str(v) for k, v in outputs.items()}, indent=2, ensure_ascii=False))
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
