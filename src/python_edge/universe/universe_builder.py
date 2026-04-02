from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

DEFAULT_BASE_URL = "https://api.massive.com"
SIC_SECTOR_RULES: List[Tuple[str, List[str]]] = [
    ("Technology", ["software", "semiconductor", "comput", "electronics", "internet", "digital", "chip", "network", "telecom equipment"]),
    ("Financials", ["bank", "financial", "insurance", "asset", "capital", "credit", "investment", "brokerage", "mortgage", "finance"]),
    ("Healthcare", ["pharmaceutical", "biotech", "medical", "health", "diagnostic", "therapeutic", "drug", "hospital"]),
    ("Energy", ["oil", "gas", "petroleum", "drilling", "exploration", "pipeline", "energy", "refining"]),
    ("Utilities", ["electric", "utility", "utilities", "power generation", "natural gas distribution", "water supply"]),
    ("Industrials", ["industrial", "machinery", "aerospace", "defense", "railroad", "transportation", "trucking", "shipping", "construction", "engineering", "manufacturing", "equipment"]),
    ("Consumer Discretionary", ["retail", "apparel", "restaurant", "entertainment", "hotel", "leisure", "automotive", "travel", "gaming"]),
    ("Consumer Staples", ["food", "beverage", "tobacco", "household", "personal products", "grocery", "packaged goods"]),
    ("Materials", ["chemical", "metal", "mining", "paper", "packaging", "plastic", "materials", "steel", "aluminum", "copper", "gold", "silver"]),
    ("Real Estate", ["real estate", "property", "rental", "warehouse", "office buildings"]),
    ("Communication Services", ["telecommunication", "media", "broadcast", "publishing", "advertising", "streaming", "content"]),
]


@dataclass(frozen=True)
class UniversePolicy:
    profile: str
    target_size: int
    min_price: float
    min_median_dollar_vol_20d: float
    min_history_days: int
    max_nan_ratio: float
    max_missing_days_20d: int
    sector_cap: int
    locale: str
    market: str
    ticker_type: str
    allow_otc: bool
    allow_etf_like_names: bool
    allow_adr_like_names: bool
    require_usd: bool
    require_active: bool
    max_reference_pages: int
    grouped_lookback_days: int
    grouped_sleep_sec: float
    overview_enrichment_enabled: bool
    overview_enrichment_max: int
    overview_sleep_sec: float
    max_candidates_stage2: int
    rebalance_freq: str
    reuse_last: bool
    reuse_max_age_days: int


@dataclass(frozen=True)
class UniverseConfig:
    api_key: str
    base_url: str
    output_dir: Path
    overview_cache_file: Path
    request_timeout_sec: int
    as_of_date: Optional[str]
    policy: UniversePolicy


@dataclass(frozen=True)
class MassiveClient:
    api_key: str
    base_url: str
    timeout_sec: int

    def __post_init__(self) -> None:
        if not self.api_key:
            raise RuntimeError("MASSIVE_API_KEY is missing from env")
        object.__setattr__(self, "_session", requests.Session())

    def _request_json(self, url: str, params: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        final_params = dict(params or {})
        final_params.setdefault("apiKey", self.api_key)
        response = self._session.get(url, params=final_params, timeout=self.timeout_sec)
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict) and str(payload.get("status", "")).upper() in {"ERROR", "NOT_AUTHORIZED"}:
            raise RuntimeError(f"Massive API error for {url}: {payload}")
        if not isinstance(payload, dict):
            raise RuntimeError(f"Unexpected Massive payload type for {url}: {type(payload).__name__}")
        return payload

    def list_reference_tickers(self, policy: UniversePolicy) -> pd.DataFrame:
        url = f"{self.base_url}/v3/reference/tickers"
        params: Optional[Dict[str, object]] = {
            "active": "true" if policy.require_active else "false",
            "locale": policy.locale,
            "market": policy.market,
            "type": policy.ticker_type,
            "limit": 1000,
            "sort": "ticker",
            "order": "asc",
        }
        rows: List[Dict[str, object]] = []
        page_no = 0
        while url and page_no < int(policy.max_reference_pages):
            page_no += 1
            payload = self._request_json(url, params=params)
            batch = payload.get("results", [])
            if isinstance(batch, list):
                rows.extend([x for x in batch if isinstance(x, dict)])
            next_url = payload.get("next_url")
            if next_url:
                url = str(next_url)
                params = None
            else:
                url = ""
            if page_no % 5 == 0 or not url:
                print(f"[REFERENCE] page={page_no} rows={len(rows)}")
        if not rows:
            raise RuntimeError("No reference tickers returned by Massive")
        df = pd.DataFrame(rows)
        if "ticker" not in df.columns:
            raise RuntimeError("Reference tickers payload missing 'ticker'")
        df["ticker"] = df["ticker"].astype(str).str.upper()
        return df.sort_values("ticker").drop_duplicates(subset=["ticker"], keep="last").reset_index(drop=True)

    def get_grouped_daily(self, trading_date: str, policy: UniversePolicy) -> pd.DataFrame:
        url = f"{self.base_url}/v2/aggs/grouped/locale/{policy.locale}/market/{policy.market}/{trading_date}"
        payload = self._request_json(url, params={"adjusted": "true", "include_otc": "true" if policy.allow_otc else "false"})
        rows = payload.get("results", [])
        if not isinstance(rows, list) or not rows:
            return pd.DataFrame(columns=["ticker", "date", "open", "high", "low", "close", "volume"])
        df = pd.DataFrame(rows).rename(columns={"T": "ticker", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "vw": "vwap", "n": "transactions"})
        for col in ["ticker", "open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                raise RuntimeError(f"Grouped daily payload for {trading_date} missing column: {col}")
        df["ticker"] = df["ticker"].astype(str).str.upper()
        df["date"] = pd.Timestamp(trading_date)
        keep = [c for c in ["ticker", "date", "open", "high", "low", "close", "volume", "vwap", "transactions"] if c in df.columns]
        return df[keep].copy().sort_values(["ticker"]).reset_index(drop=True)

    def get_ticker_overview(self, ticker: str) -> Dict[str, object]:
        url = f"{self.base_url}/v3/reference/tickers/{ticker}"
        payload = self._request_json(url)
        result = payload.get("results", {})
        return result if isinstance(result, dict) else {}


def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, "1" if default else "0")).strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _policy_from_profile(profile: str) -> UniversePolicy:
    p = profile.strip().lower()
    base = UniversePolicy(
        profile=profile,
        target_size=int(os.getenv("UNIVERSE_TARGET_SIZE", "175")),
        min_price=float(os.getenv("UNIVERSE_MIN_PRICE", "7.5")),
        min_median_dollar_vol_20d=float(os.getenv("UNIVERSE_MIN_MEDIAN_DOLLAR_VOL_20D", "20000000")),
        min_history_days=int(os.getenv("UNIVERSE_MIN_HISTORY_DAYS", "25")),
        max_nan_ratio=float(os.getenv("UNIVERSE_MAX_NAN_RATIO", "0.02")),
        max_missing_days_20d=int(os.getenv("UNIVERSE_MAX_MISSING_DAYS_20D", "1")),
        sector_cap=int(os.getenv("UNIVERSE_SECTOR_CAP", "18")),
        locale=str(os.getenv("UNIVERSE_LOCALE", "us")).strip().lower() or "us",
        market=str(os.getenv("UNIVERSE_MARKET", "stocks")).strip().lower() or "stocks",
        ticker_type=str(os.getenv("UNIVERSE_TICKER_TYPE", "CS")).strip().upper() or "CS",
        allow_otc=_env_bool("UNIVERSE_ALLOW_OTC", False),
        allow_etf_like_names=_env_bool("UNIVERSE_ALLOW_ETF_LIKE_NAMES", False),
        allow_adr_like_names=_env_bool("UNIVERSE_ALLOW_ADR_LIKE_NAMES", False),
        require_usd=_env_bool("UNIVERSE_REQUIRE_USD", True),
        require_active=_env_bool("UNIVERSE_REQUIRE_ACTIVE", True),
        max_reference_pages=int(os.getenv("UNIVERSE_MAX_REFERENCE_PAGES", "50")),
        grouped_lookback_days=int(os.getenv("UNIVERSE_LOOKBACK_DAYS", "40")),
        grouped_sleep_sec=float(os.getenv("UNIVERSE_GROUPED_SLEEP_SEC", "0.05")),
        overview_enrichment_enabled=_env_bool("OVERVIEW_ENRICHMENT_ENABLED", True),
        overview_enrichment_max=int(os.getenv("OVERVIEW_ENRICHMENT_MAX", "1200")),
        overview_sleep_sec=float(os.getenv("OVERVIEW_SLEEP_MS", "25")) / 1000.0,
        max_candidates_stage2=int(os.getenv("UNIVERSE_MAX_CANDIDATES_STAGE2", "1200")),
        rebalance_freq=str(os.getenv("UNIVERSE_REBALANCE_FREQ", "weekly")).strip().lower() or "weekly",
        reuse_last=_env_bool("UNIVERSE_REUSE_LAST", True),
        reuse_max_age_days=int(os.getenv("UNIVERSE_REUSE_MAX_AGE_DAYS", "7")),
    )
    if p == "research_us_core_v1":
        return UniversePolicy(**{**asdict(base), "target_size": max(base.target_size, 250), "min_price": min(base.min_price, 5.0), "min_median_dollar_vol_20d": min(base.min_median_dollar_vol_20d, 10000000.0), "max_nan_ratio": max(base.max_nan_ratio, 0.05), "max_missing_days_20d": max(base.max_missing_days_20d, 2), "sector_cap": max(base.sector_cap, 25)})
    if p == "trading_us_execution_safe_v1":
        return UniversePolicy(**{**asdict(base), "target_size": min(base.target_size, 150), "min_price": max(base.min_price, 10.0), "min_median_dollar_vol_20d": max(base.min_median_dollar_vol_20d, 30000000.0), "max_nan_ratio": min(base.max_nan_ratio, 0.01), "max_missing_days_20d": min(base.max_missing_days_20d, 1), "sector_cap": min(base.sector_cap, 15)})
    return base


def load_config_from_env(root_dir: Path) -> UniverseConfig:
    api_key = str(os.getenv("MASSIVE_API_KEY", "")).strip()
    if not api_key:
        raise RuntimeError("MASSIVE_API_KEY is missing from env")
    output_dir = Path(os.getenv("UNIVERSE_OUT_DIR", root_dir / "artifacts" / "daily_cycle" / "universe"))
    overview_cache_file = Path(os.getenv("OVERVIEW_CACHE_FILE", output_dir / "universe_overview_cache.parquet"))
    return UniverseConfig(
        api_key=api_key,
        base_url=str(os.getenv("MASSIVE_BASE_URL", DEFAULT_BASE_URL)).strip().rstrip("/") or DEFAULT_BASE_URL,
        output_dir=output_dir,
        overview_cache_file=overview_cache_file,
        request_timeout_sec=int(os.getenv("UNIVERSE_REQUEST_TIMEOUT_SEC", "30")),
        as_of_date=str(os.getenv("UNIVERSE_AS_OF_DATE", "")).strip() or None,
        policy=_policy_from_profile(str(os.getenv("UNIVERSE_PROFILE", "trading_us_core_v1")).strip() or "trading_us_core_v1"),
    )


def _normalize_text(x: object) -> str:
    return " ".join(str(x or "").strip().lower().split())


def _map_sector_from_sic(sic_description: object, sic_code: object) -> str:
    desc = _normalize_text(sic_description)
    if desc:
        for sector, tokens in SIC_SECTOR_RULES:
            if any(token in desc for token in tokens):
                return sector
    code_str = str(sic_code or "").strip()
    code_int: Optional[int] = int(code_str) if code_str.isdigit() else None
    if code_int is not None:
        if 100 <= code_int <= 999:
            return "Materials"
        if 1000 <= code_int <= 1499:
            return "Energy"
        if 1500 <= code_int <= 4999:
            return "Industrials"
        if 4800 <= code_int <= 4899:
            return "Communication Services"
        if 4900 <= code_int <= 4999:
            return "Utilities"
        if 5000 <= code_int <= 5999:
            return "Consumer Discretionary"
        if 6000 <= code_int <= 6799:
            return "Financials"
        if 7000 <= code_int <= 7999:
            return "Consumer Discretionary"
        if 8000 <= code_int <= 8999:
            return "Healthcare"
    return "UNKNOWN"


def _clean_industry(sic_description: object) -> str:
    s = str(sic_description or "").strip()
    return s if s and s.lower() not in {"nan", "none"} else "UNKNOWN"


def _bucketize_ranked(series: pd.Series, labels: List[str]) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    if x.notna().sum() < len(labels):
        return pd.Series(["UNKNOWN"] * len(series), index=series.index, dtype="object")
    ranks = x.rank(method="first", pct=True)
    bins = np.linspace(0.0, 1.0, num=len(labels) + 1)
    out = pd.cut(ranks, bins=bins, labels=labels, include_lowest=True)
    return out.astype("object").fillna("UNKNOWN")


def _resolve_as_of_date(client: MassiveClient, config: UniverseConfig) -> pd.Timestamp:
    if config.as_of_date:
        return pd.Timestamp(config.as_of_date).normalize()
    today_utc = datetime.now(timezone.utc).date()
    for days_back in range(1, 11):
        candidate = pd.Timestamp(today_utc - timedelta(days=days_back)).normalize()
        if int(candidate.weekday()) >= 5:
            continue
        day_df = client.get_grouped_daily(str(candidate.date()), config.policy)
        if len(day_df):
            return candidate
    raise RuntimeError("Could not resolve recent trading date from Massive grouped daily endpoint")


def _is_rebalance_day(as_of_date: pd.Timestamp, rebalance_freq: str) -> bool:
    if rebalance_freq == "daily":
        return True
    if rebalance_freq == "weekly":
        return int(as_of_date.weekday()) == 0
    return True


def _reuse_previous_snapshot_if_allowed(config: UniverseConfig, as_of_date: pd.Timestamp) -> Optional[Tuple[pd.DataFrame, Dict[str, object]]]:
    snapshot_path = config.output_dir / "universe_snapshot.parquet"
    summary_path = config.output_dir / "universe_summary.json"
    if (not config.policy.reuse_last) or _is_rebalance_day(as_of_date, config.policy.rebalance_freq):
        return None
    if not snapshot_path.exists() or not summary_path.exists():
        return None
    snapshot = pd.read_parquet(snapshot_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    prev_trade_date = str(summary.get("trade_date", "")).strip()
    if not prev_trade_date:
        return None
    age_days = int((as_of_date - pd.Timestamp(prev_trade_date).normalize()).days)
    if age_days < 0 or age_days > int(config.policy.reuse_max_age_days):
        return None
    reused = dict(summary)
    reused["reused_for_date"] = str(as_of_date.date())
    reused["reused_from_date"] = prev_trade_date
    print(f"[REBALANCE] reuse_last=1 -> using prior snapshot dated {prev_trade_date} for as_of_date={as_of_date.date()}")
    return snapshot, reused


def _date_range(end_date: pd.Timestamp, n_days: int) -> List[pd.Timestamp]:
    out: List[pd.Timestamp] = []
    cur = end_date.normalize()
    while len(out) < int(n_days):
        if int(cur.weekday()) < 5:
            out.append(cur)
        cur = cur - pd.Timedelta(days=1)
    out.reverse()
    return out


def _fetch_grouped_history(client: MassiveClient, as_of_date: pd.Timestamp, policy: UniversePolicy) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    days = _date_range(as_of_date, policy.grouped_lookback_days)
    for idx, dt in enumerate(days, start=1):
        day_df = client.get_grouped_daily(str(dt.date()), policy)
        if len(day_df):
            frames.append(day_df)
        if idx % 5 == 0 or idx == len(days):
            rows = int(sum(len(x) for x in frames))
            print(f"[GROUPED] {idx}/{len(days)} days fetched rows={rows}")
        if policy.grouped_sleep_sec > 0.0:
            time.sleep(policy.grouped_sleep_sec)
    if not frames:
        raise RuntimeError("Grouped daily history is empty")
    panel = pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)
    panel["dollar_vol"] = pd.to_numeric(panel["close"], errors="coerce") * pd.to_numeric(panel["volume"], errors="coerce")
    return panel


def _core_reference_filter(ref: pd.DataFrame, policy: UniversePolicy) -> Tuple[pd.DataFrame, Dict[str, int]]:
    out = ref.copy()
    for col in ["locale", "market", "type", "name", "primary_exchange", "currency_name", "active", "sic_code", "sic_description"]:
        if col not in out.columns:
            out[col] = None
    out["locale"] = out["locale"].astype(str).str.lower()
    out["market"] = out["market"].astype(str).str.lower()
    out["type"] = out["type"].astype(str).str.upper()
    out["name"] = out["name"].astype(str)
    out["primary_exchange"] = out["primary_exchange"].astype(str)
    out["currency_name"] = out["currency_name"].astype(str).str.lower()
    out["active"] = out["active"].astype(str).str.lower()

    out["flag_us_listed"] = out["locale"].eq(policy.locale)
    out["flag_usd"] = out["currency_name"].eq("usd")
    out["flag_common_stock"] = out["type"].eq(policy.ticker_type)
    out["flag_exchange_otc"] = out["primary_exchange"].str.contains("OTC", case=False, na=False)
    out["flag_name_etf_like"] = out["name"].str.contains("ETF|ETN|Fund|Trust", case=False, na=False)
    out["flag_name_adr_like"] = out["name"].str.contains("ADR|Depositary", case=False, na=False)
    out["flag_active_ok"] = out["active"].isin(["true", "1"]) if policy.require_active else True

    out["passes_reference_locale"] = out["flag_us_listed"]
    out["passes_reference_currency"] = out["flag_usd"] if policy.require_usd else True
    out["passes_reference_type"] = out["flag_common_stock"]
    out["passes_reference_otc"] = ~out["flag_exchange_otc"] if not policy.allow_otc else True
    out["passes_reference_etf_like"] = ~out["flag_name_etf_like"] if not policy.allow_etf_like_names else True
    out["passes_reference_adr_like"] = ~out["flag_name_adr_like"] if not policy.allow_adr_like_names else True
    out["passes_reference_active"] = out["flag_active_ok"]
    out["passes_reference"] = (
        out["passes_reference_locale"]
        & out["passes_reference_currency"]
        & out["passes_reference_type"]
        & out["passes_reference_otc"]
        & out["passes_reference_etf_like"]
        & out["passes_reference_adr_like"]
        & out["passes_reference_active"]
    )

    out["drop_reason_reference"] = ""
    out.loc[~out["passes_reference_locale"], "drop_reason_reference"] = "reference_locale"
    out.loc[out["passes_reference_locale"] & ~out["passes_reference_currency"], "drop_reason_reference"] = "reference_currency"
    out.loc[out["passes_reference_locale"] & out["passes_reference_currency"] & ~out["passes_reference_type"], "drop_reason_reference"] = "reference_type"
    out.loc[out["passes_reference_locale"] & out["passes_reference_currency"] & out["passes_reference_type"] & ~out["passes_reference_otc"], "drop_reason_reference"] = "reference_otc"
    out.loc[out["passes_reference_locale"] & out["passes_reference_currency"] & out["passes_reference_type"] & out["passes_reference_otc"] & ~out["passes_reference_etf_like"], "drop_reason_reference"] = "reference_etf_like"
    out.loc[out["passes_reference_locale"] & out["passes_reference_currency"] & out["passes_reference_type"] & out["passes_reference_otc"] & out["passes_reference_etf_like"] & ~out["passes_reference_adr_like"], "drop_reason_reference"] = "reference_adr_like"
    out.loc[out["passes_reference_locale"] & out["passes_reference_currency"] & out["passes_reference_type"] & out["passes_reference_otc"] & out["passes_reference_etf_like"] & out["passes_reference_adr_like"] & ~out["passes_reference_active"], "drop_reason_reference"] = "reference_active"
    out.loc[out["passes_reference"], "drop_reason_reference"] = "passes_reference"

    counters = {
        "reference_total": int(len(out)),
        "reference_dropped_locale": int((~out["passes_reference_locale"]).sum()),
        "reference_dropped_currency": int((out["passes_reference_locale"] & ~out["passes_reference_currency"]).sum()),
        "reference_dropped_type": int((out["passes_reference_locale"] & out["passes_reference_currency"] & ~out["passes_reference_type"]).sum()),
        "reference_dropped_otc": int((out["passes_reference_locale"] & out["passes_reference_currency"] & out["passes_reference_type"] & ~out["passes_reference_otc"]).sum()),
        "reference_dropped_etf_like": int((out["passes_reference_locale"] & out["passes_reference_currency"] & out["passes_reference_type"] & out["passes_reference_otc"] & ~out["passes_reference_etf_like"]).sum()),
        "reference_dropped_adr_like": int((out["passes_reference_locale"] & out["passes_reference_currency"] & out["passes_reference_type"] & out["passes_reference_otc"] & out["passes_reference_etf_like"] & ~out["passes_reference_adr_like"]).sum()),
        "reference_dropped_active": int((out["passes_reference_locale"] & out["passes_reference_currency"] & out["passes_reference_type"] & out["passes_reference_otc"] & out["passes_reference_etf_like"] & out["passes_reference_adr_like"] & ~out["passes_reference_active"]).sum()),
        "reference_kept": int(out["passes_reference"].sum()),
    }
    return out.loc[out["passes_reference"]].copy().reset_index(drop=True), counters


def _compute_symbol_metrics(core_ref: pd.DataFrame, panel: pd.DataFrame, policy: UniversePolicy) -> pd.DataFrame:
    keep_cols = [c for c in ["ticker", "name", "sic_code", "sic_description", "market", "locale", "type", "primary_exchange"] if c in core_ref.columns]
    metrics = core_ref[keep_cols].copy().drop_duplicates(subset=["ticker"]).reset_index(drop=True)
    p = panel.copy()
    p["ret_1d"] = p.groupby("ticker", sort=False)["close"].pct_change(1)
    g = p.groupby("ticker", sort=False)

    agg = pd.DataFrame({"ticker": sorted(p["ticker"].dropna().astype(str).unique().tolist())})
    agg = agg.merge(g["date"].count().rename("history_days").reset_index(), on="ticker", how="left")
    agg = agg.merge(g["close"].last().rename("last_close").reset_index(), on="ticker", how="left")
    agg = agg.merge(g["dollar_vol"].apply(lambda s: float(pd.to_numeric(s, errors="coerce").tail(20).median())).rename("median_dollar_vol_20d").reset_index(), on="ticker", how="left")
    agg = agg.merge(g["dollar_vol"].apply(lambda s: int(pd.to_numeric(s, errors="coerce").tail(20).isna().sum())).rename("missing_days_20d").reset_index(), on="ticker", how="left")
    agg = agg.merge(g["ret_1d"].apply(lambda s: float(pd.to_numeric(s, errors="coerce").tail(20).std(ddof=0))).rename("volatility_20d").reset_index(), on="ticker", how="left")
    agg = agg.merge(g[["open", "high", "low", "close", "volume"]].apply(lambda x: float(x.isna().mean().mean())).rename("nan_ratio_ohlcv").reset_index(), on="ticker", how="left")

    out = metrics.merge(agg, on="ticker", how="left")
    out["history_days"] = pd.to_numeric(out["history_days"], errors="coerce").fillna(0).astype(int)
    out["last_close"] = pd.to_numeric(out["last_close"], errors="coerce")
    out["median_dollar_vol_20d"] = pd.to_numeric(out["median_dollar_vol_20d"], errors="coerce")
    out["missing_days_20d"] = pd.to_numeric(out["missing_days_20d"], errors="coerce").fillna(999).astype(int)
    out["volatility_20d"] = pd.to_numeric(out["volatility_20d"], errors="coerce")
    out["nan_ratio_ohlcv"] = pd.to_numeric(out["nan_ratio_ohlcv"], errors="coerce").fillna(1.0)

    out["eligible_price"] = out["last_close"] >= float(policy.min_price)
    out["eligible_liquidity"] = out["median_dollar_vol_20d"] >= float(policy.min_median_dollar_vol_20d)
    out["eligible_history"] = out["history_days"] >= int(policy.min_history_days)
    out["eligible_data_quality"] = (out["nan_ratio_ohlcv"] <= float(policy.max_nan_ratio)) & (out["missing_days_20d"] <= int(policy.max_missing_days_20d))
    out["eligible"] = out["eligible_price"] & out["eligible_liquidity"] & out["eligible_history"] & out["eligible_data_quality"]

    out["drop_reason_stage2"] = ""
    out.loc[~out["eligible_price"], "drop_reason_stage2"] = "price"
    out.loc[out["eligible_price"] & ~out["eligible_liquidity"], "drop_reason_stage2"] = "liquidity"
    out.loc[out["eligible_price"] & out["eligible_liquidity"] & ~out["eligible_history"], "drop_reason_stage2"] = "history"
    out.loc[out["eligible_price"] & out["eligible_liquidity"] & out["eligible_history"] & ~out["eligible_data_quality"], "drop_reason_stage2"] = "data_quality"
    out.loc[out["eligible"], "drop_reason_stage2"] = "passes_stage2"
    return out.sort_values(["median_dollar_vol_20d", "ticker"], ascending=[False, True]).reset_index(drop=True)


def _load_overview_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["ticker", "sic_code", "sic_description", "overview_name"])
    df = pd.read_parquet(path)
    if df.empty:
        return pd.DataFrame(columns=["ticker", "sic_code", "sic_description", "overview_name"])
    df["ticker"] = df["ticker"].astype(str).str.upper()
    return df.drop_duplicates(subset=["ticker"], keep="last").reset_index(drop=True)


def _write_overview_cache(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _extract_overview_fields(ticker: str, payload: Dict[str, object]) -> Dict[str, object]:
    return {"ticker": str(ticker).upper(), "sic_code": payload.get("sic_code"), "sic_description": payload.get("sic_description"), "overview_name": payload.get("name")}


def _enrich_with_overview(client: MassiveClient, classified: pd.DataFrame, config: UniverseConfig) -> Tuple[pd.DataFrame, Dict[str, object]]:
    if not config.policy.overview_enrichment_enabled:
        return classified, {"overview_enriched_total": 0, "overview_cache_hits": 0, "overview_api_calls": 0, "overview_missing_sic_total": int(len(classified)), "overview_enrichment_enabled": 0}

    capped = classified.sort_values(["median_dollar_vol_20d", "ticker"], ascending=[False, True]).head(int(config.policy.overview_enrichment_max)).copy().reset_index(drop=True)
    cache = _load_overview_cache(config.overview_cache_file)
    cache_map = cache.set_index("ticker", drop=False).to_dict(orient="index") if len(cache) else {}

    rows: List[Dict[str, object]] = []
    cache_hits = 0
    api_calls = 0
    for idx, row in capped.iterrows():
        ticker = str(row["ticker"]).upper()
        cached = cache_map.get(ticker)
        if cached is not None and ((pd.notna(cached.get("sic_code")) and str(cached.get("sic_code")).strip() != "") or (pd.notna(cached.get("sic_description")) and str(cached.get("sic_description")).strip() != "")):
            rows.append(dict(cached))
            cache_hits += 1
        else:
            rows.append(_extract_overview_fields(ticker, client.get_ticker_overview(ticker)))
            api_calls += 1
            if config.policy.overview_sleep_sec > 0.0:
                time.sleep(config.policy.overview_sleep_sec)
        if (idx + 1) % 100 == 0 or (idx + 1) == len(capped):
            print(f"[OVERVIEW] {idx + 1}/{len(capped)} processed cache_hits={cache_hits} api_calls={api_calls}")

    overview_df = pd.DataFrame(rows)
    if len(overview_df):
        merged_cache = pd.concat([cache, overview_df], ignore_index=True) if len(cache) else overview_df.copy()
        merged_cache = merged_cache.drop_duplicates(subset=["ticker"], keep="last").reset_index(drop=True)
        _write_overview_cache(merged_cache, config.overview_cache_file)

    out = classified.merge(overview_df[["ticker", "sic_code", "sic_description", "overview_name"]], on="ticker", how="left", suffixes=("", "_ov"))
    out["sic_code"] = out["sic_code_ov"].where(out["sic_code_ov"].notna(), out.get("sic_code"))
    out["sic_description"] = out["sic_description_ov"].where(out["sic_description_ov"].notna(), out.get("sic_description"))
    if "name" in out.columns and "overview_name" in out.columns:
        out["name"] = out["overview_name"].where(out["overview_name"].notna(), out["name"])
    drop_cols = [c for c in ["sic_code_ov", "sic_description_ov", "overview_name"] if c in out.columns]
    if drop_cols:
        out = out.drop(columns=drop_cols)

    missing_sic_total = int((((out["sic_description"].isna()) | out["sic_description"].astype(str).str.strip().eq("")) & ((out["sic_code"].isna()) | out["sic_code"].astype(str).str.strip().eq(""))).sum())
    return out, {"overview_enriched_total": int(len(overview_df)), "overview_cache_hits": int(cache_hits), "overview_api_calls": int(api_calls), "overview_missing_sic_total": missing_sic_total, "overview_enrichment_enabled": 1}


def _classify(metrics: pd.DataFrame) -> pd.DataFrame:
    out = metrics.copy()
    sic_desc = out["sic_description"] if "sic_description" in out.columns else pd.Series([None] * len(out), index=out.index)
    sic_code = out["sic_code"] if "sic_code" in out.columns else pd.Series([None] * len(out), index=out.index)
    out["industry"] = pd.Series([_clean_industry(sd) for sd in sic_desc.tolist()], index=out.index, dtype="object")
    out["sector"] = pd.Series([_map_sector_from_sic(sd, sc) for sd, sc in zip(sic_desc.tolist(), sic_code.tolist())], index=out.index, dtype="object")
    out["liquidity_bucket"] = _bucketize_ranked(out["median_dollar_vol_20d"], ["LOW", "MID", "HIGH"])
    out["volatility_bucket"] = _bucketize_ranked(out["volatility_20d"], ["LOW", "MID", "HIGH"])
    out["price_bucket"] = _bucketize_ranked(out["last_close"], ["LOW", "MID", "HIGH"])
    out["flag_adr"] = False
    out["flag_etf"] = False
    return out


def _policy_select(classified: pd.DataFrame, policy: UniversePolicy) -> Tuple[pd.DataFrame, Dict[str, int]]:
    eligible = classified.loc[classified["eligible"]].copy()
    if eligible.empty:
        return eligible, {"selected_total": 0, "selected_sector_cap_skips": 0, "selected_refill_from_over_cap": 0}
    eligible = eligible.sort_values(["median_dollar_vol_20d", "ticker"], ascending=[False, True]).reset_index(drop=True)
    if int(policy.max_candidates_stage2) > 0 and len(eligible) > int(policy.max_candidates_stage2):
        eligible = eligible.head(int(policy.max_candidates_stage2)).copy().reset_index(drop=True)

    selected_parts: List[pd.DataFrame] = []
    sector_counts: Dict[str, int] = {}
    sector_cap_skips = 0
    for _, row in eligible.iterrows():
        sector = str(row.get("sector", "UNKNOWN") or "UNKNOWN")
        current = int(sector_counts.get(sector, 0))
        if sector != "UNKNOWN" and int(policy.sector_cap) > 0 and current >= int(policy.sector_cap):
            sector_cap_skips += 1
            continue
        selected_parts.append(pd.DataFrame([row]))
        sector_counts[sector] = current + 1
        if len(selected_parts) >= int(policy.target_size):
            break

    chosen = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame(columns=eligible.columns)
    refill_count = 0
    if len(chosen) < int(policy.target_size):
        already = set(chosen["ticker"].astype(str).tolist()) if len(chosen) else set()
        refill = eligible.loc[~eligible["ticker"].astype(str).isin(already)].head(int(policy.target_size) - len(chosen)).copy()
        refill_count = int(len(refill))
        if len(refill):
            chosen = pd.concat([chosen, refill], ignore_index=True)
    chosen["selected"] = True
    return chosen.reset_index(drop=True), {"selected_total": int(len(chosen)), "selected_sector_cap_skips": int(sector_cap_skips), "selected_refill_from_over_cap": int(refill_count)}


def _build_snapshot(reference_all: pd.DataFrame, classified: pd.DataFrame, selected: pd.DataFrame, as_of_date: pd.Timestamp, config: UniverseConfig) -> pd.DataFrame:
    out = reference_all.copy()
    stage2_cols = [c for c in classified.columns if c != "ticker"]
    out = out.merge(classified[["ticker", *stage2_cols]], on="ticker", how="left")

    classified_set = set(classified["ticker"].astype(str).tolist()) if len(classified) else set()
    selected_set = set(selected["ticker"].astype(str).tolist()) if len(selected) else set()

    if "passes_reference" not in out.columns:
        out["passes_reference"] = out["ticker"].astype(str).isin(classified_set)
    if "drop_reason_reference" not in out.columns:
        out["drop_reason_reference"] = np.where(out["passes_reference"], "passes_reference", "reference_filtered_out")
    if "drop_reason_stage2" not in out.columns:
        out["drop_reason_stage2"] = ""

    out["trade_date"] = str(as_of_date.date())
    out["universe_profile"] = config.policy.profile
    out["selected"] = out["ticker"].astype(str).isin(selected_set)
    out["final_drop_reason"] = out["drop_reason_reference"].where(out["drop_reason_reference"] != "passes_reference", out["drop_reason_stage2"])
    out.loc[out["selected"], "final_drop_reason"] = "selected"
    return out.sort_values(["selected", "median_dollar_vol_20d", "ticker"], ascending=[False, False, True]).reset_index(drop=True)


def _summary_dict(reference_counters: Dict[str, int], classified: pd.DataFrame, selected: pd.DataFrame, as_of_date: pd.Timestamp, config: UniverseConfig, overview_summary: Dict[str, object], selection_summary: Dict[str, int]) -> Dict[str, object]:
    stage2_kept = int(len(classified))
    selected_total = int(len(selected))
    return {
        "trade_date": str(as_of_date.date()),
        "universe_profile": config.policy.profile,
        "rebalance_freq": config.policy.rebalance_freq,
        "policy": asdict(config.policy),
        **reference_counters,
        "stage2_candidates_total": stage2_kept,
        "dropped_price": int((~classified["eligible_price"]).sum()) if len(classified) else 0,
        "dropped_liquidity": int((classified["eligible_price"] & ~classified["eligible_liquidity"]).sum()) if len(classified) else 0,
        "dropped_history": int((classified["eligible_price"] & classified["eligible_liquidity"] & ~classified["eligible_history"]).sum()) if len(classified) else 0,
        "dropped_data_quality": int((classified["eligible_price"] & classified["eligible_liquidity"] & classified["eligible_history"] & ~classified["eligible_data_quality"]).sum()) if len(classified) else 0,
        "eligible_total": int(classified["eligible"].sum()) if len(classified) else 0,
        **selection_summary,
        "sector_counts_selected": selected.groupby("sector", sort=False).size().to_dict() if len(selected) else {},
        "sector_unknown_selected": int(selected["sector"].astype(str).eq("UNKNOWN").sum()) if len(selected) else 0,
        "industry_unknown_selected": int(selected["industry"].astype(str).eq("UNKNOWN").sum()) if len(selected) else 0,
        "liquidity_bucket_counts_selected": selected.groupby("liquidity_bucket", sort=False).size().to_dict() if len(selected) else {},
        "volatility_bucket_counts_selected": selected.groupby("volatility_bucket", sort=False).size().to_dict() if len(selected) else {},
        "price_bucket_counts_selected": selected.groupby("price_bucket", sort=False).size().to_dict() if len(selected) else {},
        "selected_preview": selected[[c for c in ["ticker", "sector", "industry", "sic_code", "sic_description", "last_close", "median_dollar_vol_20d", "liquidity_bucket", "volatility_bucket", "price_bucket"] if c in selected.columns]].head(min(25, selected_total)).fillna("").to_dict(orient="records") if len(selected) else [],
        **overview_summary,
    }


def _write_outputs(snapshot: pd.DataFrame, summary: Dict[str, object], config: UniverseConfig) -> Tuple[Path, Path]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = config.output_dir / "universe_snapshot.parquet"
    summary_path = config.output_dir / "universe_summary.json"
    snapshot.to_parquet(snapshot_path, index=False)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] snapshot={snapshot_path}")
    print(f"[OK] summary={summary_path}")
    return snapshot_path, summary_path


def build_universe_snapshot(config: UniverseConfig) -> Tuple[pd.DataFrame, Dict[str, object]]:
    client = MassiveClient(api_key=config.api_key, base_url=config.base_url, timeout_sec=config.request_timeout_sec)
    as_of_date = _resolve_as_of_date(client, config)

    print(f"[CFG] base_url={config.base_url}")
    print(f"[CFG] universe_profile={config.policy.profile}")
    print(f"[CFG] target_size={config.policy.target_size} sector_cap={config.policy.sector_cap}")
    print(f"[CFG] min_price={config.policy.min_price:.2f} min_median_dollar_vol_20d={config.policy.min_median_dollar_vol_20d:.2f}")
    print(f"[CFG] min_history_days={config.policy.min_history_days} max_nan_ratio={config.policy.max_nan_ratio:.4f} max_missing_days_20d={config.policy.max_missing_days_20d}")
    print(f"[CFG] locale={config.policy.locale} market={config.policy.market} ticker_type={config.policy.ticker_type}")
    print(f"[CFG] rebalance_freq={config.policy.rebalance_freq} reuse_last={int(config.policy.reuse_last)} reuse_max_age_days={config.policy.reuse_max_age_days}")
    print(f"[CFG] overview_enrichment_enabled={int(config.policy.overview_enrichment_enabled)} overview_enrichment_max={config.policy.overview_enrichment_max}")
    print(f"[DATA] as_of_date={as_of_date.date()}")

    reused = _reuse_previous_snapshot_if_allowed(config, as_of_date)
    if reused is not None:
        return reused

    ref = client.list_reference_tickers(config.policy)
    core_ref, reference_counters = _core_reference_filter(ref, config.policy)
    print(f"[UNIVERSE] reference_total={reference_counters['reference_total']}")
    print(f"[UNIVERSE] reference_kept={reference_counters['reference_kept']}")
    print(f"[UNIVERSE] reference_columns={sorted(ref.columns.astype(str).tolist())[:30]}")
    print(f"[UNIVERSE] reference_drop_counters={{'locale': {reference_counters['reference_dropped_locale']}, 'currency': {reference_counters['reference_dropped_currency']}, 'type': {reference_counters['reference_dropped_type']}, 'otc': {reference_counters['reference_dropped_otc']}, 'etf_like': {reference_counters['reference_dropped_etf_like']}, 'adr_like': {reference_counters['reference_dropped_adr_like']}, 'active': {reference_counters['reference_dropped_active']}}}")

    panel = _fetch_grouped_history(client, as_of_date, config.policy)
    panel = panel.loc[panel["ticker"].astype(str).isin(set(core_ref["ticker"].astype(str).tolist()))].copy()
    print(f"[UNIVERSE] grouped_rows_after_core_filter={len(panel)} symbols={panel['ticker'].nunique()}")

    metrics = _compute_symbol_metrics(core_ref, panel, config.policy)
    classified = _classify(metrics)
    if len(classified) > int(config.policy.max_candidates_stage2):
        classified = classified.sort_values(["median_dollar_vol_20d", "ticker"], ascending=[False, True]).head(int(config.policy.max_candidates_stage2)).reset_index(drop=True)
        print(f"[UNIVERSE] stage2 candidate cap applied -> {len(classified)}")

    classified, overview_summary = _enrich_with_overview(client, classified, config)
    classified = _classify(classified)
    selected, selection_summary = _policy_select(classified, config.policy)
    snapshot = _build_snapshot(ref, classified, selected, as_of_date, config)
    summary = _summary_dict(reference_counters, classified, selected, as_of_date, config, overview_summary, selection_summary)

    print(f"[UNIVERSE] dropped_price={summary['dropped_price']}")
    print(f"[UNIVERSE] dropped_liquidity={summary['dropped_liquidity']}")
    print(f"[UNIVERSE] dropped_history={summary['dropped_history']}")
    print(f"[UNIVERSE] dropped_data_quality={summary['dropped_data_quality']}")
    print(f"[UNIVERSE] eligible_total={summary['eligible_total']}")
    print(f"[UNIVERSE] selected_total={summary['selected_total']}")
    print(f"[UNIVERSE] selected_sector_cap_skips={summary['selected_sector_cap_skips']} selected_refill_from_over_cap={summary['selected_refill_from_over_cap']}")
    print(f"[UNIVERSE] overview_enriched_total={summary['overview_enriched_total']} overview_cache_hits={summary['overview_cache_hits']} overview_api_calls={summary['overview_api_calls']} overview_missing_sic_total={summary['overview_missing_sic_total']}")
    print(f"[UNIVERSE] sector_unknown_selected={summary['sector_unknown_selected']}")
    print(f"[UNIVERSE] industry_unknown_selected={summary['industry_unknown_selected']}")
    print(f"[UNIVERSE] sector_counts_selected={summary['sector_counts_selected']}")

    if len(selected):
        print("[UNIVERSE][TOP_SELECTED]")
        print(selected[[c for c in ["ticker", "sector", "industry", "sic_code", "sic_description", "last_close", "median_dollar_vol_20d", "liquidity_bucket", "volatility_bucket", "price_bucket"] if c in selected.columns]].head(min(25, len(selected))).to_string(index=False))

    return snapshot, summary


def build_and_save_universe_snapshot(config: UniverseConfig) -> Tuple[Path, Path]:
    snapshot, summary = build_universe_snapshot(config)
    return _write_outputs(snapshot, summary, config)