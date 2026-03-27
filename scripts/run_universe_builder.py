from __future__ import annotations

import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
for p in [ROOT, SRC_DIR]:
    ps = str(p)
    if ps not in sys.path:
        sys.path.insert(0, ps)

PAUSE_ON_EXIT = str(os.getenv("PAUSE_ON_EXIT", "auto")).strip().lower()
DEFAULT_BASE_URL = str(os.getenv("MASSIVE_BASE_URL", "https://api.massive.com")).strip().rstrip("/")
OUT_DIR = Path(os.getenv("UNIVERSE_OUT_DIR", "artifacts/daily_cycle/universe"))
UNIVERSE_PROFILE = str(os.getenv("UNIVERSE_PROFILE", "trading_us_core_v1")).strip()
UNIVERSE_TARGET_SIZE = int(os.getenv("UNIVERSE_TARGET_SIZE", "175"))
UNIVERSE_REBALANCE_FREQ = str(os.getenv("UNIVERSE_REBALANCE_FREQ", "weekly")).strip().lower()
UNIVERSE_REUSE_LAST = str(os.getenv("UNIVERSE_REUSE_LAST", "1")).strip().lower() not in {"0", "false", "no", "off"}
UNIVERSE_LOOKBACK_DAYS = int(os.getenv("UNIVERSE_LOOKBACK_DAYS", "40"))
UNIVERSE_MIN_PRICE = float(os.getenv("UNIVERSE_MIN_PRICE", "7.5"))
UNIVERSE_MIN_MEDIAN_DOLLAR_VOL_20D = float(os.getenv("UNIVERSE_MIN_MEDIAN_DOLLAR_VOL_20D", "20000000"))
UNIVERSE_MIN_HISTORY_DAYS = int(os.getenv("UNIVERSE_MIN_HISTORY_DAYS", "25"))
UNIVERSE_MAX_NAN_RATIO = float(os.getenv("UNIVERSE_MAX_NAN_RATIO", "0.02"))
UNIVERSE_MAX_MISSING_DAYS_20D = int(os.getenv("UNIVERSE_MAX_MISSING_DAYS_20D", "1"))
UNIVERSE_REQUEST_TIMEOUT_SEC = int(os.getenv("UNIVERSE_REQUEST_TIMEOUT_SEC", "30"))
UNIVERSE_SECTOR_CAP = int(os.getenv("UNIVERSE_SECTOR_CAP", "18"))
UNIVERSE_MAX_CANDIDATES_STAGE2 = int(os.getenv("UNIVERSE_MAX_CANDIDATES_STAGE2", "1200"))
OVERVIEW_ENRICHMENT_ENABLED = str(os.getenv("OVERVIEW_ENRICHMENT_ENABLED", "1")).strip().lower() not in {"0", "false", "no", "off"}
OVERVIEW_ENRICHMENT_MAX = int(os.getenv("OVERVIEW_ENRICHMENT_MAX", "1200"))
OVERVIEW_SLEEP_MS = int(os.getenv("OVERVIEW_SLEEP_MS", "25"))
OVERVIEW_CACHE_FILE = Path(os.getenv("OVERVIEW_CACHE_FILE", "artifacts/daily_cycle/universe/universe_overview_cache.parquet"))

SIC_SECTOR_RULES: List[Tuple[str, List[str]]] = [
    ("Technology", [
        "software", "semiconductor", "comput", "electronics", "communication equipment", "communications equipment",
        "data processing", "programming", "internet", "digital", "micro", "chip", "network", "telecom equipment",
    ]),
    ("Financials", [
        "bank", "banks", "financial", "insurance", "asset", "capital", "credit", "investment", "brokerage",
        "consumer lending", "mortgage", "trust", "finance",
    ]),
    ("Healthcare", [
        "pharmaceutical", "biotech", "biological", "medical", "health", "surgical", "diagnostic", "therapeutic",
        "laborator", "drug", "hospital", "care services",
    ]),
    ("Energy", [
        "oil", "gas", "petroleum", "drilling", "exploration", "pipeline", "energy", "coal", "refining",
    ]),
    ("Utilities", [
        "electric", "water supply", "water", "utility", "utilities", "power generation", "natural gas distribution",
    ]),
    ("Industrials", [
        "industrial", "machinery", "aerospace", "defense", "railroad", "transportation", "trucking", "air freight",
        "shipping", "construction", "engineering", "manufacturing", "equipment", "metal fabricat", "tools",
    ]),
    ("Consumer Discretionary", [
        "retail", "apparel", "restaurants", "restaurant", "entertainment", "hotel", "hotels", "leisure", "automotive",
        "auto", "consumer goods", "specialty stores", "internet retail", "travel", "gaming",
    ]),
    ("Consumer Staples", [
        "food", "beverage", "beverages", "tobacco", "household", "personal products", "grocery", "packaged goods",
        "consumer staples",
    ]),
    ("Materials", [
        "chemical", "chemicals", "metal", "metals", "mining", "paper", "forest products", "packaging", "plastic",
        "materials", "steel", "aluminum", "copper", "gold", "silver",
    ]),
    ("Real Estate", [
        "real estate", "property", "lessor", "rental", "warehouse", "office buildings",
    ]),
    ("Communication Services", [
        "telecommunication", "telecommunications", "media", "broadcast", "publishing", "advertising", "streaming",
        "social network", "content",
    ]),
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
    stricter_trading_thresholds: bool


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
        return payload

    def list_reference_tickers(self) -> pd.DataFrame:
        url = f"{self.base_url}/v3/reference/tickers"
        params: Optional[Dict[str, object]] = {
            "market": "stocks",
            "active": "true",
            "limit": 1000,
            "sort": "ticker",
            "order": "asc",
        }
        rows: List[Dict[str, object]] = []
        while url:
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
        if not rows:
            raise RuntimeError("No reference tickers returned by Massive")
        df = pd.DataFrame(rows)
        if "ticker" not in df.columns:
            raise RuntimeError("Reference tickers payload missing 'ticker'")
        df["ticker"] = df["ticker"].astype(str).str.upper()
        return df.sort_values("ticker").reset_index(drop=True)

    def get_grouped_daily(self, day: str) -> pd.DataFrame:
        url = f"{self.base_url}/v2/aggs/grouped/locale/us/market/stocks/{day}"
        payload = self._request_json(url, params={"adjusted": "true"})
        rows = payload.get("results", [])
        if not isinstance(rows, list) or not rows:
            return pd.DataFrame(columns=["ticker", "date", "open", "high", "low", "close", "volume"])
        df = pd.DataFrame(rows).rename(columns={"T": "ticker", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        for col in ["ticker", "open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                raise RuntimeError(f"Grouped daily payload for {day} missing column: {col}")
        df["ticker"] = df["ticker"].astype(str).str.upper()
        df["date"] = pd.Timestamp(day)
        return df[["ticker", "date", "open", "high", "low", "close", "volume"]].copy().sort_values(["ticker"]).reset_index(drop=True)

    def get_ticker_overview(self, ticker: str) -> Dict[str, object]:
        url = f"{self.base_url}/v3/reference/tickers/{ticker}"
        payload = self._request_json(url)
        result = payload.get("results", {})
        if not isinstance(result, dict):
            return {}
        return result


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


def _policy_from_profile(profile: str) -> UniversePolicy:
    p = profile.strip().lower()
    base = UniversePolicy(
        profile=profile,
        target_size=int(UNIVERSE_TARGET_SIZE),
        min_price=float(UNIVERSE_MIN_PRICE),
        min_median_dollar_vol_20d=float(UNIVERSE_MIN_MEDIAN_DOLLAR_VOL_20D),
        min_history_days=int(UNIVERSE_MIN_HISTORY_DAYS),
        max_nan_ratio=float(UNIVERSE_MAX_NAN_RATIO),
        max_missing_days_20d=int(UNIVERSE_MAX_MISSING_DAYS_20D),
        sector_cap=int(UNIVERSE_SECTOR_CAP),
        stricter_trading_thresholds=True,
    )
    if p == "research_us_core_v1":
        return UniversePolicy(
            profile=profile,
            target_size=max(int(base.target_size), 200),
            min_price=max(5.0, base.min_price - 2.5),
            min_median_dollar_vol_20d=max(10000000.0, base.min_median_dollar_vol_20d * 0.5),
            min_history_days=max(20, base.min_history_days),
            max_nan_ratio=min(0.05, base.max_nan_ratio * 2.0),
            max_missing_days_20d=max(2, base.max_missing_days_20D),
            sector_cap=max(base.sector_cap, 25),
            stricter_trading_thresholds=False,
        )
    if p == "trading_us_execution_safe_v1":
        return UniversePolicy(
            profile=profile,
            target_size=min(int(base.target_size), 150),
            min_price=max(10.0, base.min_price),
            min_median_dollar_vol_20d=max(30000000.0, base.min_median_dollar_vol_20d),
            min_history_days=max(25, base.min_history_days),
            max_nan_ratio=min(0.01, base.max_nan_ratio),
            max_missing_days_20d=min(1, base.max_missing_days_20d),
            sector_cap=min(base.sector_cap, 15),
            stricter_trading_thresholds=True,
        )
    return base


def _is_rebalance_day(as_of_date: pd.Timestamp) -> bool:
    if UNIVERSE_REBALANCE_FREQ == "daily":
        return True
    if UNIVERSE_REBALANCE_FREQ == "weekly":
        return int(as_of_date.weekday()) == 0
    return True


def _reuse_previous_snapshot_if_allowed(as_of_date: pd.Timestamp) -> Optional[Tuple[pd.DataFrame, Dict[str, object]]]:
    snapshot_path = OUT_DIR / "universe_snapshot.parquet"
    summary_path = OUT_DIR / "universe_summary.json"
    if not UNIVERSE_REUSE_LAST or _is_rebalance_day(as_of_date):
        return None
    if not snapshot_path.exists() or not summary_path.exists():
        return None
    snap = pd.read_parquet(snapshot_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    prev_trade_date = str(summary.get("trade_date", "")).strip()
    if not prev_trade_date:
        return None
    prev_dt = pd.Timestamp(prev_trade_date)
    age_days = int((as_of_date.normalize() - prev_dt.normalize()).days)
    if age_days < 0 or age_days > 7:
        return None
    print(f"[REBALANCE] weekly reuse active -> using prior snapshot dated {prev_dt.date()} for as_of_date={as_of_date.date()}")
    summary = dict(summary)
    summary["reused_for_date"] = str(as_of_date.date())
    summary["reused_from_date"] = prev_trade_date
    return snap, summary


def _date_range(end_date: pd.Timestamp, n_days: int) -> List[pd.Timestamp]:
    out: List[pd.Timestamp] = []
    cur = end_date.normalize()
    while len(out) < int(n_days):
        if int(cur.weekday()) < 5:
            out.append(cur)
        cur = cur - pd.Timedelta(days=1)
    out.reverse()
    return out


def _fetch_grouped_history(client: MassiveClient, end_date: pd.Timestamp, n_days: int) -> pd.DataFrame:
    days = _date_range(end_date, n_days)
    frames: List[pd.DataFrame] = []
    for idx, dt in enumerate(days, start=1):
        day_str = str(dt.date())
        day_df = client.get_grouped_daily(day_str)
        if len(day_df):
            frames.append(day_df)
        if idx % 5 == 0 or idx == len(days):
            rows = int(sum(len(x) for x in frames))
            print(f"[GROUPED] {idx}/{len(days)} days fetched rows={rows}")
        time.sleep(0.05)
    if not frames:
        raise RuntimeError("Grouped daily history is empty")
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    panel["dollar_vol"] = pd.to_numeric(panel["close"], errors="coerce") * pd.to_numeric(panel["volume"], errors="coerce")
    return panel


def _core_reference_filter(ref: pd.DataFrame) -> pd.DataFrame:
    out = ref.copy()
    for col in ["locale", "market", "type", "name", "primary_exchange", "currency_name", "sic_code", "sic_description"]:
        if col not in out.columns:
            out[col] = None
    out["locale"] = out["locale"].astype(str).str.lower()
    out["market"] = out["market"].astype(str).str.lower()
    out["type"] = out["type"].astype(str).str.upper()
    out["name"] = out["name"].astype(str)
    out["primary_exchange"] = out["primary_exchange"].astype(str)
    out["currency_name"] = out["currency_name"].astype(str).str.lower()

    out["flag_us_listed"] = (out["locale"] == "us") | (out["currency_name"] == "usd")
    out["flag_common_stock"] = out["type"].eq("CS")
    out["flag_exchange_otc"] = out["primary_exchange"].str.contains("OTC", case=False, na=False)
    out["flag_name_etf_like"] = out["name"].str.contains("ETF|ETN|Fund|Trust", case=False, na=False)
    out["flag_name_adr_like"] = out["name"].str.contains("ADR|Depositary", case=False, na=False)

    core = out.loc[
        out["flag_us_listed"]
        & out["flag_common_stock"]
        & (~out["flag_exchange_otc"])
        & (~out["flag_name_etf_like"])
        & (~out["flag_name_adr_like"])
    ].copy()
    return core.reset_index(drop=True)


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

    out["drop_reason"] = ""
    out.loc[~out["eligible_price"], "drop_reason"] = "price"
    out.loc[out["eligible_price"] & (~out["eligible_liquidity"]), "drop_reason"] = "liquidity"
    out.loc[out["eligible_price"] & out["eligible_liquidity"] & (~out["eligible_history"]), "drop_reason"] = "history"
    out.loc[out["eligible_price"] & out["eligible_liquidity"] & out["eligible_history"] & (~out["eligible_data_quality"]), "drop_reason"] = "data_quality"
    return out.sort_values(["median_dollar_vol_20d", "ticker"], ascending=[False, True]).reset_index(drop=True)


def _load_overview_cache() -> pd.DataFrame:
    if not OVERVIEW_CACHE_FILE.exists():
        return pd.DataFrame(columns=["ticker", "sic_code", "sic_description", "overview_name"])
    df = pd.read_parquet(OVERVIEW_CACHE_FILE)
    if df.empty:
        return pd.DataFrame(columns=["ticker", "sic_code", "sic_description", "overview_name"])
    df["ticker"] = df["ticker"].astype(str).str.upper()
    return df.drop_duplicates(subset=["ticker"], keep="last").reset_index(drop=True)


def _write_overview_cache(df: pd.DataFrame) -> None:
    OVERVIEW_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OVERVIEW_CACHE_FILE, index=False)


def _extract_overview_fields(ticker: str, payload: Dict[str, object]) -> Dict[str, object]:
    return {
        "ticker": str(ticker).upper(),
        "sic_code": payload.get("sic_code"),
        "sic_description": payload.get("sic_description"),
        "overview_name": payload.get("name"),
    }


def _enrich_with_overview(client: MassiveClient, classified: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    if not OVERVIEW_ENRICHMENT_ENABLED:
        return classified, {
            "overview_enriched_total": 0,
            "overview_cache_hits": 0,
            "overview_api_calls": 0,
            "overview_missing_sic_total": int(len(classified)),
            "overview_enrichment_enabled": 0,
        }

    capped = classified.sort_values(["median_dollar_vol_20d", "ticker"], ascending=[False, True]).head(int(OVERVIEW_ENRICHMENT_MAX)).copy().reset_index(drop=True)
    cache = _load_overview_cache()
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
            payload = client.get_ticker_overview(ticker)
            rows.append(_extract_overview_fields(ticker, payload))
            api_calls += 1
            time.sleep(max(0.0, float(OVERVIEW_SLEEP_MS) / 1000.0))
        if (idx + 1) % 100 == 0 or (idx + 1) == len(capped):
            print(f"[OVERVIEW] {idx + 1}/{len(capped)} processed cache_hits={cache_hits} api_calls={api_calls}")

    overview_df = pd.DataFrame(rows)
    if len(overview_df):
        merged_cache = pd.concat([cache, overview_df], ignore_index=True) if len(cache) else overview_df.copy()
        merged_cache = merged_cache.drop_duplicates(subset=["ticker"], keep="last").reset_index(drop=True)
        _write_overview_cache(merged_cache)

    out = classified.merge(overview_df[["ticker", "sic_code", "sic_description", "overview_name"]], on="ticker", how="left", suffixes=("", "_ov"))
    out["sic_code"] = out["sic_code_ov"].where(out["sic_code_ov"].notna(), out.get("sic_code"))
    out["sic_description"] = out["sic_description_ov"].where(out["sic_description_ov"].notna(), out.get("sic_description"))
    if "name" in out.columns and "overview_name" in out.columns:
        out["name"] = out["overview_name"].where(out["overview_name"].notna(), out["name"])
    drop_cols = [c for c in ["sic_code_ov", "sic_description_ov", "overview_name"] if c in out.columns]
    if drop_cols:
        out = out.drop(columns=drop_cols)

    missing_sic_total = int(
        (
            (
                out["sic_description"].isna()
                | out["sic_description"].astype(str).str.strip().eq("")
            )
            & (
                out["sic_code"].isna()
                | out["sic_code"].astype(str).str.strip().eq("")
            )
        ).sum()
    )

    return out, {
        "overview_enriched_total": int(len(overview_df)),
        "overview_cache_hits": int(cache_hits),
        "overview_api_calls": int(api_calls),
        "overview_missing_sic_total": missing_sic_total,
        "overview_enrichment_enabled": 1,
    }


def _bucketize_ranked(series: pd.Series, labels: List[str]) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    if x.notna().sum() < len(labels):
        return pd.Series(["UNKNOWN"] * len(series), index=series.index, dtype="object")
    ranks = x.rank(method="first", pct=True)
    bins = np.linspace(0.0, 1.0, num=len(labels) + 1)
    out = pd.cut(ranks, bins=bins, labels=labels, include_lowest=True)
    return out.astype("object").fillna("UNKNOWN")


def _normalize_text(x: object) -> str:
    return " ".join(str(x or "").strip().lower().split())


def _map_sector_from_sic(sic_description: object, sic_code: object) -> str:
    desc = _normalize_text(sic_description)
    if desc:
        for sector, tokens in SIC_SECTOR_RULES:
            if any(token in desc for token in tokens):
                return sector
    code_str = str(sic_code or "").strip()
    code_int: Optional[int] = None
    if code_str.isdigit():
        try:
            code_int = int(code_str)
        except Exception:
            code_int = None
    if code_int is not None:
        if 100 <= code_int <= 999:
            return "Materials"
        if 1000 <= code_int <= 1499:
            return "Energy"
        if 1500 <= code_int <= 1799:
            return "Industrials"
        if 2000 <= code_int <= 3999:
            return "Industrials"
        if 4000 <= code_int <= 4799:
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
    if s and s.lower() not in {"nan", "none"}:
        return s
    return "UNKNOWN"


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


def _policy_select(classified: pd.DataFrame, policy: UniversePolicy) -> pd.DataFrame:
    eligible = classified.loc[classified["eligible"]].copy()
    if eligible.empty:
        return eligible
    eligible = eligible.sort_values(["median_dollar_vol_20d", "ticker"], ascending=[False, True]).reset_index(drop=True)
    if not policy.stricter_trading_thresholds or int(policy.sector_cap) <= 0:
        chosen = eligible.head(int(policy.target_size)).copy()
        chosen["selected"] = True
        return chosen.reset_index(drop=True)

    selected_parts: List[pd.DataFrame] = []
    sector_counts: Dict[str, int] = {}
    for _, row in eligible.iterrows():
        sector = str(row.get("sector", "UNKNOWN") or "UNKNOWN")
        current = int(sector_counts.get(sector, 0))
        if sector != "UNKNOWN" and current >= int(policy.sector_cap):
            continue
        selected_parts.append(pd.DataFrame([row]))
        sector_counts[sector] = current + 1
        if len(selected_parts) >= int(policy.target_size):
            break
    chosen = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame(columns=eligible.columns)
    if len(chosen) < int(policy.target_size):
        already = set(chosen["ticker"].astype(str).tolist()) if len(chosen) else set()
        refill = eligible.loc[~eligible["ticker"].astype(str).isin(already)].head(int(policy.target_size) - len(chosen)).copy()
        if len(refill):
            chosen = pd.concat([chosen, refill], ignore_index=True)
    chosen["selected"] = True
    return chosen.reset_index(drop=True)


def _build_snapshot(classified: pd.DataFrame, selected: pd.DataFrame, as_of_date: pd.Timestamp, policy: UniversePolicy) -> pd.DataFrame:
    out = classified.copy()
    selected_set = set(selected["ticker"].astype(str).tolist()) if len(selected) else set()
    out["trade_date"] = str(as_of_date.date())
    out["universe_profile"] = policy.profile
    out["selected"] = out["ticker"].astype(str).isin(selected_set)
    return out.sort_values(["selected", "median_dollar_vol_20d", "ticker"], ascending=[False, False, True]).reset_index(drop=True)


def _write_outputs(snapshot: pd.DataFrame, summary: Dict[str, object]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_path = OUT_DIR / "universe_snapshot.parquet"
    summary_path = OUT_DIR / "universe_summary.json"
    snapshot.to_parquet(snapshot_path, index=False)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] snapshot={snapshot_path}")
    print(f"[OK] summary={summary_path}")


def main() -> int:
    api_key = str(os.getenv("MASSIVE_API_KEY", "")).strip()
    client = MassiveClient(api_key=api_key, base_url=DEFAULT_BASE_URL, timeout_sec=UNIVERSE_REQUEST_TIMEOUT_SEC)
    policy = _policy_from_profile(UNIVERSE_PROFILE)
    as_of_date = pd.Timestamp.now("UTC").tz_localize(None).normalize() - pd.Timedelta(days=1)
    while int(as_of_date.weekday()) >= 5:
        as_of_date = as_of_date - pd.Timedelta(days=1)

    print(f"[CFG] base_url={DEFAULT_BASE_URL}")
    print(f"[CFG] universe_profile={policy.profile}")
    print(f"[CFG] target_size={policy.target_size} sector_cap={policy.sector_cap}")
    print(f"[CFG] min_price={policy.min_price:.2f} min_median_dollar_vol_20d={policy.min_median_dollar_vol_20d:.2f}")
    print(f"[CFG] min_history_days={policy.min_history_days} max_nan_ratio={policy.max_nan_ratio:.4f} max_missing_days_20d={policy.max_missing_days_20d}")
    print(f"[CFG] rebalance_freq={UNIVERSE_REBALANCE_FREQ} reuse_last={int(UNIVERSE_REUSE_LAST)}")
    print(f"[CFG] overview_enrichment_enabled={int(OVERVIEW_ENRICHMENT_ENABLED)} overview_enrichment_max={OVERVIEW_ENRICHMENT_MAX}")
    print(f"[DATA] as_of_date={as_of_date.date()}")

    reused = _reuse_previous_snapshot_if_allowed(as_of_date)
    if reused is not None:
        snap, summary = reused
        _write_outputs(snap, summary)
        return 0

    ref = client.list_reference_tickers()
    candidates_total = int(len(ref))
    core_ref = _core_reference_filter(ref)
    print(f"[UNIVERSE] reference_total={candidates_total}")
    print(f"[UNIVERSE] core_candidates_after_reference_filter={len(core_ref)}")
    print(f"[UNIVERSE] reference_columns={sorted(ref.columns.astype(str).tolist())[:30]}")

    panel = _fetch_grouped_history(client, as_of_date, UNIVERSE_LOOKBACK_DAYS)
    panel = panel.loc[panel["ticker"].astype(str).isin(set(core_ref["ticker"].astype(str).tolist()))].copy()
    print(f"[UNIVERSE] grouped_rows_after_core_filter={len(panel)} symbols={panel['ticker'].nunique()}")

    metrics = _compute_symbol_metrics(core_ref, panel, policy)
    classified = _classify(metrics)

    if len(classified) > int(UNIVERSE_MAX_CANDIDATES_STAGE2):
        classified = classified.sort_values(["median_dollar_vol_20d", "ticker"], ascending=[False, True]).head(int(UNIVERSE_MAX_CANDIDATES_STAGE2)).reset_index(drop=True)
        print(f"[UNIVERSE] stage2 candidate cap applied -> {len(classified)}")

    classified, overview_summary = _enrich_with_overview(client, classified)
    classified = _classify(classified)
    selected = _policy_select(classified, policy)
    snapshot = _build_snapshot(classified, selected, as_of_date, policy)

    sector_counts_selected = selected.groupby("sector", sort=False).size().to_dict() if len(selected) else {}
    sector_unknown_selected = int(selected["sector"].astype(str).eq("UNKNOWN").sum()) if len(selected) else 0
    industry_unknown_selected = int(selected["industry"].astype(str).eq("UNKNOWN").sum()) if len(selected) else 0
    summary = {
        "trade_date": str(as_of_date.date()),
        "universe_profile": policy.profile,
        "rebalance_freq": UNIVERSE_REBALANCE_FREQ,
        "reference_total": candidates_total,
        "candidates_total": int(len(classified)),
        "dropped_price": int((~classified["eligible_price"]).sum()),
        "dropped_liquidity": int((classified["eligible_price"] & (~classified["eligible_liquidity"])).sum()),
        "dropped_history": int((classified["eligible_price"] & classified["eligible_liquidity"] & (~classified["eligible_history"])).sum()),
        "dropped_data_quality": int((classified["eligible_price"] & classified["eligible_liquidity"] & classified["eligible_history"] & (~classified["eligible_data_quality"])).sum()),
        "eligible_total": int(classified["eligible"].sum()),
        "selected_total": int(len(selected)),
        "sector_counts_selected": sector_counts_selected,
        "sector_unknown_selected": sector_unknown_selected,
        "industry_unknown_selected": industry_unknown_selected,
        "liquidity_bucket_counts_selected": selected.groupby("liquidity_bucket", sort=False).size().to_dict() if len(selected) else {},
        "volatility_bucket_counts_selected": selected.groupby("volatility_bucket", sort=False).size().to_dict() if len(selected) else {},
        "price_bucket_counts_selected": selected.groupby("price_bucket", sort=False).size().to_dict() if len(selected) else {},
        **overview_summary,
    }

    print(f"[UNIVERSE] dropped_price={summary['dropped_price']}")
    print(f"[UNIVERSE] dropped_liquidity={summary['dropped_liquidity']}")
    print(f"[UNIVERSE] dropped_history={summary['dropped_history']}")
    print(f"[UNIVERSE] dropped_data_quality={summary['dropped_data_quality']}")
    print(f"[UNIVERSE] eligible_total={summary['eligible_total']}")
    print(f"[UNIVERSE] selected_total={summary['selected_total']}")
    print(f"[UNIVERSE] overview_enriched_total={summary['overview_enriched_total']} overview_cache_hits={summary['overview_cache_hits']} overview_api_calls={summary['overview_api_calls']} overview_missing_sic_total={summary['overview_missing_sic_total']}")
    print(f"[UNIVERSE] sector_unknown_selected={summary['sector_unknown_selected']}")
    print(f"[UNIVERSE] industry_unknown_selected={summary['industry_unknown_selected']}")
    print(f"[UNIVERSE] sector_counts_selected={summary['sector_counts_selected']}")

    if len(selected):
        print("[UNIVERSE][TOP_SELECTED]")
        print(
            selected[["ticker", "sector", "industry", "sic_code", "sic_description", "last_close", "median_dollar_vol_20d", "liquidity_bucket", "volatility_bucket", "price_bucket"]]
            .head(min(25, len(selected)))
            .to_string(index=False)
        )

    _write_outputs(snapshot, summary)
    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 1
    _safe_exit(rc)