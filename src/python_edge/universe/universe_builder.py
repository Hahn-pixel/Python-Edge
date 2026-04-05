# ===== FILE: src/python_edge/universe/universe_builder.py =====
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests


DEFAULT_BASE_URL = "https://api.massive.com"
SNAPSHOT_FILE_NAME = "universe_snapshot.parquet"
SUMMARY_FILE_NAME = "universe_summary.json"
REFERENCE_FILE_NAME = "universe_reference_raw.parquet"
GROUPED_FILE_NAME = "universe_grouped_daily_raw.parquet"
ELIGIBILITY_FILE_NAME = "universe_eligibility_debug.parquet"


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(str(value).strip())


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    return float(str(value).strip())


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_symbol_series(values: pd.Series) -> pd.Series:
    out = values.astype(str).str.strip().str.upper()
    return out.where(~out.isin({"", "NAN", "NONE"}), "")


@dataclass(frozen=True)
class UniversePolicy:
    profile: str
    locale: str
    market: str
    ticker_type: str
    active_only: bool
    allowed_currency: str
    allowed_base_types: Tuple[str, ...]
    exclude_otc: bool
    exclude_etfs: bool
    exclude_adr: bool
    min_price: float
    min_median_dollar_vol_20d: float
    min_history_days: int
    target_size: int
    grouped_lookback_days: int
    max_missing_days_lookback: int
    grouped_sleep_sec: float
    request_pages_limit: int


@dataclass(frozen=True)
class UniverseConfig:
    api_key: str
    base_url: str
    output_dir: Path
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
            raise RuntimeError("MASSIVE_API_KEY is missing from config")
        if not self.base_url:
            raise RuntimeError("Massive base_url is empty")

    def _get_json(self, url: str, params: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        call_params = dict(params or {})
        call_params["apiKey"] = self.api_key
        response = requests.get(url, params=call_params, timeout=self.timeout_sec)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError(f"Unexpected Massive response type: {type(payload)!r}")
        return payload

    def get_reference_tickers(self, policy: UniversePolicy) -> pd.DataFrame:
        rows: List[Dict[str, object]] = []
        params: Optional[Dict[str, object]] = {
            "market": policy.market,
            "locale": policy.locale,
            "active": "true" if policy.active_only else "false",
            "limit": 1000,
        }
        if policy.ticker_type:
            params["type"] = policy.ticker_type
        url = f"{self.base_url}/v3/reference/tickers"
        page_no = 0
        while url:
            page_no += 1
            payload = self._get_json(url, params)
            batch = payload.get("results", [])
            if isinstance(batch, list):
                for item in batch:
                    if isinstance(item, dict):
                        rows.append(item)
            next_url = payload.get("next_url")
            if next_url:
                url = str(next_url)
                params = None
            else:
                url = ""
            print(f"[REFERENCE] page={page_no} rows_total={len(rows)}")
            if page_no >= policy.request_pages_limit:
                print(f"[REFERENCE][WARN] page limit reached limit={policy.request_pages_limit}")
                break
        if not rows:
            raise RuntimeError("No reference tickers returned by Massive")
        df = pd.DataFrame(rows)
        if "ticker" not in df.columns:
            raise RuntimeError("Reference tickers payload missing 'ticker'")
        df["ticker"] = _normalize_symbol_series(df["ticker"])
        df = df.loc[df["ticker"].ne("")].copy().reset_index(drop=True)
        return df

    def get_grouped_daily(self, session_date: str, policy: UniversePolicy) -> pd.DataFrame:
        url = f"{self.base_url}/v2/aggs/grouped/locale/{policy.locale}/market/{policy.market}/{session_date}"
        payload = self._get_json(url, {"adjusted": "true"})
        rows = payload.get("results", [])
        if not isinstance(rows, list):
            raise RuntimeError(f"Grouped daily results invalid for {session_date}")
        if not rows:
            return pd.DataFrame(columns=["ticker", "date", "close", "volume", "dollar_volume"])
        df = pd.DataFrame(rows)
        rename_map = {"T": "ticker", "c": "close", "v": "volume"}
        df = df.rename(columns=rename_map)
        if "ticker" not in df.columns:
            raise RuntimeError(f"Grouped daily payload missing ticker for {session_date}")
        for column in ["close", "volume"]:
            if column not in df.columns:
                df[column] = pd.NA
        df["ticker"] = _normalize_symbol_series(df["ticker"])
        df["date"] = pd.to_datetime(session_date).normalize()
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        df["dollar_volume"] = df["close"] * df["volume"]
        df = df.loc[df["ticker"].ne("")].copy().reset_index(drop=True)
        return df


def load_config_from_env(root_dir: Path) -> UniverseConfig:
    api_key = _env_str("MASSIVE_API_KEY", "")
    if not api_key:
        raise RuntimeError("MASSIVE_API_KEY is missing from env")
    policy = UniversePolicy(
        profile=_env_str("UNIVERSE_PROFILE", "trading_us_core_v1"),
        locale=_env_str("UNIVERSE_LOCALE", "us"),
        market=_env_str("UNIVERSE_MARKET", "stocks"),
        ticker_type=_env_str("UNIVERSE_TICKER_TYPE", "CS"),
        active_only=_env_bool("UNIVERSE_ACTIVE_ONLY", True),
        allowed_currency=_env_str("UNIVERSE_ALLOWED_CURRENCY", "USD"),
        allowed_base_types=tuple(
            x.strip().upper()
            for x in _env_str("UNIVERSE_ALLOWED_BASE_TYPES", "CS").split("|")
            if x.strip()
        ),
        exclude_otc=_env_bool("UNIVERSE_EXCLUDE_OTC", True),
        exclude_etfs=_env_bool("UNIVERSE_EXCLUDE_ETFS", True),
        exclude_adr=_env_bool("UNIVERSE_EXCLUDE_ADR", True),
        min_price=_env_float("UNIVERSE_MIN_PRICE", 7.5),
        min_median_dollar_vol_20d=_env_float("UNIVERSE_MIN_MEDIAN_DOLLAR_VOL_20D", 20_000_000.0),
        min_history_days=_env_int("UNIVERSE_MIN_HISTORY_DAYS", 25),
        target_size=_env_int("UNIVERSE_TARGET_SIZE", 175),
        grouped_lookback_days=_env_int("UNIVERSE_GROUPED_LOOKBACK_DAYS", 35),
        max_missing_days_lookback=_env_int("UNIVERSE_MAX_MISSING_DAYS_LOOKBACK", 1),
        grouped_sleep_sec=_env_float("UNIVERSE_GROUPED_SLEEP_SEC", 0.0),
        request_pages_limit=_env_int("UNIVERSE_REQUEST_PAGES_LIMIT", 50),
    )
    output_dir = Path(_env_str("UNIVERSE_OUT_DIR", str(root_dir / "artifacts" / "daily_cycle" / "universe")))
    return UniverseConfig(
        api_key=api_key,
        base_url=_env_str("MASSIVE_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
        output_dir=output_dir,
        request_timeout_sec=_env_int("UNIVERSE_REQUEST_TIMEOUT_SEC", 30),
        as_of_date=_env_str("UNIVERSE_AS_OF_DATE", "") or None,
        policy=policy,
    )


def _choose_as_of_date(config: UniverseConfig) -> datetime:
    if config.as_of_date:
        parsed = pd.Timestamp(config.as_of_date)
        if parsed.tzinfo is None:
            parsed = parsed.tz_localize("UTC")
        else:
            parsed = parsed.tz_convert("UTC")
        return parsed.to_pydatetime()
    now_utc = _utc_now()
    return datetime(now_utc.year, now_utc.month, now_utc.day, tzinfo=timezone.utc)


def _build_reference_flags(reference_df: pd.DataFrame, policy: UniversePolicy) -> pd.DataFrame:
    df = reference_df.copy()
    for column in ["ticker", "locale", "market", "type", "currency_name", "primary_exchange", "name"]:
        if column not in df.columns:
            df[column] = ""
    if "active" not in df.columns:
        df["active"] = False
    df["type_norm"] = df["type"].astype(str).str.upper().str.strip()
    df["locale_norm"] = df["locale"].astype(str).str.lower().str.strip()
    df["currency_norm"] = df["currency_name"].astype(str).str.upper().str.strip()
    df["primary_exchange_norm"] = df["primary_exchange"].astype(str).str.upper().str.strip()
    name_blob = (
        df["name"].astype(str).str.lower().fillna("")
        + " "
        + df["ticker"].astype(str).str.lower().fillna("")
        + " "
        + df["primary_exchange"].astype(str).str.lower().fillna("")
    )
    df["passes_locale"] = df["locale_norm"].eq(policy.locale.lower())
    df["passes_type"] = df["type_norm"].isin(set(policy.allowed_base_types))
    df["passes_currency"] = df["currency_norm"].eq(policy.allowed_currency.upper())
    if policy.active_only:
        df["passes_active"] = df["active"].fillna(False).astype(bool)
    else:
        df["passes_active"] = True
    df["is_otc_like"] = df["primary_exchange_norm"].str.contains("OTC", na=False)
    df["is_etf_like"] = name_blob.str.contains(r"\betf\b|\bfund\b|\btrust\b|\bspdr\b|\bishares\b|\bvanguard\b|\betn\b|\bnotes\b", regex=True, na=False)
    df["is_adr_like"] = name_blob.str.contains(r"\badr\b|\bads\b|\bsponsored\b|\bdepositary\b", regex=True, na=False)
    df["passes_otc"] = ~df["is_otc_like"] if policy.exclude_otc else True
    df["passes_etf"] = ~df["is_etf_like"] if policy.exclude_etfs else True
    df["passes_adr"] = ~df["is_adr_like"] if policy.exclude_adr else True
    df["passes_reference"] = (
        df["passes_locale"]
        & df["passes_type"]
        & df["passes_currency"]
        & df["passes_active"]
        & df["passes_otc"]
        & df["passes_etf"]
        & df["passes_adr"]
    )

    reason = pd.Series("passes_reference", index=df.index, dtype="object")
    reason = reason.mask(~df["passes_locale"], "reference_locale")
    reason = reason.mask(df["passes_locale"] & ~df["passes_type"], "reference_type")
    reason = reason.mask(df["passes_locale"] & df["passes_type"] & ~df["passes_currency"], "reference_currency")
    reason = reason.mask(df["passes_locale"] & df["passes_type"] & df["passes_currency"] & ~df["passes_active"], "reference_active")
    reason = reason.mask(df["passes_locale"] & df["passes_type"] & df["passes_currency"] & df["passes_active"] & ~df["passes_otc"], "reference_otc")
    reason = reason.mask(df["passes_locale"] & df["passes_type"] & df["passes_currency"] & df["passes_active"] & df["passes_otc"] & ~df["passes_etf"], "reference_etf_like")
    reason = reason.mask(df["passes_locale"] & df["passes_type"] & df["passes_currency"] & df["passes_active"] & df["passes_otc"] & df["passes_etf"] & ~df["passes_adr"], "reference_adr_like")
    df["drop_reason_reference"] = reason
    return df


def _get_recent_business_days(end_date: datetime, count: int) -> List[datetime]:
    out: List[datetime] = []
    cursor = pd.Timestamp(end_date.date())
    while len(out) < count:
        if cursor.weekday() < 5:
            out.append(cursor.to_pydatetime().replace(tzinfo=timezone.utc))
        cursor = cursor - pd.Timedelta(days=1)
    out.reverse()
    return out


def _fetch_grouped_history(client: MassiveClient, policy: UniversePolicy, as_of_date: datetime) -> pd.DataFrame:
    days = _get_recent_business_days(as_of_date, policy.grouped_lookback_days)
    frames: List[pd.DataFrame] = []
    for idx, dt in enumerate(days, start=1):
        df = client.get_grouped_daily(dt.date().isoformat(), policy)
        if len(df):
            frames.append(df)
        if idx % 5 == 0 or idx == len(days):
            rows_total = int(sum(len(x) for x in frames))
            print(f"[GROUPED] {idx}/{len(days)} days fetched rows={rows_total}")
        if policy.grouped_sleep_sec > 0.0:
            time.sleep(policy.grouped_sleep_sec)
    if not frames:
        raise RuntimeError("Grouped daily history is empty")
    out = pd.concat(frames, ignore_index=True)
    out["ticker"] = _normalize_symbol_series(out["ticker"])
    out = out.loc[out["ticker"].ne("")].copy().reset_index(drop=True)
    return out


def _classify_eligibility(reference_df: pd.DataFrame, grouped_df: pd.DataFrame, as_of_date: datetime, policy: UniversePolicy) -> pd.DataFrame:
    grouped = grouped_df.copy()
    grouped["date"] = pd.to_datetime(grouped["date"], errors="coerce").dt.normalize()
    grouped["close"] = pd.to_numeric(grouped["close"], errors="coerce")
    grouped["dollar_volume"] = pd.to_numeric(grouped["dollar_volume"], errors="coerce")

    agg = (
        grouped.sort_values(["ticker", "date"])
        .groupby("ticker", as_index=False)
        .agg(
            history_days=("date", lambda s: int(pd.Series(s).dropna().nunique())),
            last_trade_date=("date", "max"),
            last_close=("close", "last"),
            median_dollar_vol_20d=("dollar_volume", lambda s: float(pd.Series(s).dropna().tail(20).median()) if len(pd.Series(s).dropna()) else float("nan")),
        )
    )
    agg["missing_days_lookback"] = (policy.grouped_lookback_days - agg["history_days"]).clip(lower=0)

    out = reference_df.merge(agg, on="ticker", how="left")
    out["trade_date"] = pd.Timestamp(as_of_date.date())
    out["symbol"] = out["ticker"]
    out["as_of_date"] = out["trade_date"]

    out["history_days"] = pd.to_numeric(out["history_days"], errors="coerce").fillna(0).astype(int)
    out["missing_days_lookback"] = pd.to_numeric(out["missing_days_lookback"], errors="coerce").fillna(policy.grouped_lookback_days).astype(int)
    out["last_close"] = pd.to_numeric(out["last_close"], errors="coerce")
    out["median_dollar_vol_20d"] = pd.to_numeric(out["median_dollar_vol_20d"], errors="coerce")

    out["eligible_price"] = out["last_close"].ge(policy.min_price).fillna(False)
    out["eligible_liquidity"] = out["median_dollar_vol_20d"].ge(policy.min_median_dollar_vol_20d).fillna(False)
    out["eligible_history_depth"] = out["history_days"].ge(policy.min_history_days).fillna(False)
    out["eligible_missing_days"] = out["missing_days_lookback"].le(policy.max_missing_days_lookback).fillna(False)

    out["eligible"] = (
        out["passes_reference"]
        & out["eligible_price"]
        & out["eligible_liquidity"]
        & out["eligible_history_depth"]
        & out["eligible_missing_days"]
    )
    out["selected"] = out["eligible"]
    out["is_selected"] = out["selected"]

    reason = pd.Series("eligible", index=out.index, dtype="object")
    reason = reason.mask(~out["passes_reference"], out["drop_reason_reference"])
    reason = reason.mask(out["passes_reference"] & ~out["eligible_price"], "price")
    reason = reason.mask(out["passes_reference"] & out["eligible_price"] & ~out["eligible_liquidity"], "liquidity")
    reason = reason.mask(out["passes_reference"] & out["eligible_price"] & out["eligible_liquidity"] & ~out["eligible_history_depth"], "history_depth")
    reason = reason.mask(out["passes_reference"] & out["eligible_price"] & out["eligible_liquidity"] & out["eligible_history_depth"] & ~out["eligible_missing_days"], "missing_days")
    out["drop_reason"] = reason
    out["eligibility_stage"] = out["drop_reason"].where(~out["eligible"], "selected")

    keep_cols = [
        "trade_date",
        "as_of_date",
        "ticker",
        "symbol",
        "name",
        "market",
        "locale",
        "type",
        "primary_exchange",
        "currency_name",
        "active",
        "last_trade_date",
        "last_close",
        "median_dollar_vol_20d",
        "history_days",
        "missing_days_lookback",
        "passes_reference",
        "eligible_price",
        "eligible_liquidity",
        "eligible_history_depth",
        "eligible_missing_days",
        "eligible",
        "selected",
        "is_selected",
        "drop_reason_reference",
        "drop_reason",
        "eligibility_stage",
    ]
    for col in keep_cols:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[keep_cols].copy()
    return out.sort_values(["selected", "median_dollar_vol_20d", "ticker"], ascending=[False, False, True]).reset_index(drop=True)


def _build_summary(classified: pd.DataFrame, config: UniverseConfig, as_of_date: datetime) -> Dict[str, object]:
    selected = classified.loc[classified["selected"]].copy()
    return {
        "trade_date": str(as_of_date.date()),
        "universe_profile": config.policy.profile,
        "policy": asdict(config.policy),
        "reference_total": int(len(classified)),
        "eligible_total": int(classified["eligible"].sum()),
        "selected_total": int(selected["selected"].sum()) if len(selected) else 0,
        "dropped_reference": int((~classified["passes_reference"]).sum()),
        "dropped_price": int((classified["passes_reference"] & ~classified["eligible_price"]).sum()),
        "dropped_liquidity": int((classified["passes_reference"] & classified["eligible_price"] & ~classified["eligible_liquidity"]).sum()),
        "dropped_history_depth": int((classified["passes_reference"] & classified["eligible_price"] & classified["eligible_liquidity"] & ~classified["eligible_history_depth"]).sum()),
        "dropped_missing_days": int((classified["passes_reference"] & classified["eligible_price"] & classified["eligible_liquidity"] & classified["eligible_history_depth"] & ~classified["eligible_missing_days"]).sum()),
        "overview_missing_sic_total": 0,
        "sector_unknown_selected": 0,
        "industry_unknown_selected": 0,
        "generated_at_utc": _utc_now().isoformat(),
    }


def _write_outputs(
    classified: pd.DataFrame,
    summary: Dict[str, object],
    reference_raw: pd.DataFrame,
    grouped_raw: pd.DataFrame,
    config: UniverseConfig,
) -> Tuple[Path, Path]:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = config.output_dir / SNAPSHOT_FILE_NAME
    summary_path = config.output_dir / SUMMARY_FILE_NAME
    reference_path = config.output_dir / REFERENCE_FILE_NAME
    grouped_path = config.output_dir / GROUPED_FILE_NAME
    debug_path = config.output_dir / ELIGIBILITY_FILE_NAME

    classified.to_parquet(snapshot_path, index=False)
    classified.to_parquet(debug_path, index=False)
    reference_raw.to_parquet(reference_path, index=False)
    grouped_raw.to_parquet(grouped_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"[UNIVERSE][WRITE] snapshot={snapshot_path}")
    print(f"[UNIVERSE][WRITE] summary={summary_path}")
    print(f"[UNIVERSE][WRITE] reference_raw={reference_path}")
    print(f"[UNIVERSE][WRITE] grouped_raw={grouped_path}")
    print(f"[UNIVERSE][WRITE] eligibility_debug={debug_path}")

    selected = classified.loc[classified["selected"]].copy()
    if len(selected):
        cols = [c for c in ["ticker", "symbol", "last_close", "median_dollar_vol_20d", "history_days", "missing_days_lookback"] if c in selected.columns]
        print("[UNIVERSE][TOP_SELECTED]")
        print(selected[cols].head(min(25, len(selected))).to_string(index=False))
    else:
        print("[UNIVERSE][TOP_SELECTED] none")

    return snapshot_path, summary_path


def _build_universe_snapshot_full(
    config: UniverseConfig,
) -> Tuple[pd.DataFrame, Dict[str, object], pd.DataFrame, pd.DataFrame]:
    as_of_date = _choose_as_of_date(config)
    print(
        "[CFG][UNIVERSE] "
        f"profile={config.policy.profile} as_of_date={as_of_date.date().isoformat()} "
        f"min_price={config.policy.min_price:.2f} "
        f"min_median_dollar_vol_20d={config.policy.min_median_dollar_vol_20d:.2f} "
        f"min_history_days={config.policy.min_history_days} "
        f"max_missing_days_lookback={config.policy.max_missing_days_lookback} "
        f"allowed_base_types={'|'.join(config.policy.allowed_base_types)} "
        f"exclude_otc={int(config.policy.exclude_otc)} "
        f"exclude_etfs={int(config.policy.exclude_etfs)} "
        f"exclude_adr={int(config.policy.exclude_adr)}"
    )

    client = MassiveClient(
        api_key=config.api_key,
        base_url=config.base_url,
        timeout_sec=config.request_timeout_sec,
    )
    reference_raw = client.get_reference_tickers(config.policy)
    reference_classified = _build_reference_flags(reference_raw, config.policy)
    grouped_raw = _fetch_grouped_history(client, config.policy, as_of_date)
    classified = _classify_eligibility(reference_classified, grouped_raw, as_of_date, config.policy)
    summary = _build_summary(classified, config, as_of_date)

    print(
        "[UNIVERSE][SUMMARY] "
        f"reference_total={summary['reference_total']} eligible_total={summary['eligible_total']} "
        f"selected_total={summary['selected_total']} dropped_reference={summary['dropped_reference']} "
        f"dropped_price={summary['dropped_price']} dropped_liquidity={summary['dropped_liquidity']} "
        f"dropped_history_depth={summary['dropped_history_depth']} dropped_missing_days={summary['dropped_missing_days']}"
    )

    return classified, summary, reference_raw, grouped_raw


def build_universe_snapshot(config: UniverseConfig) -> Tuple[pd.DataFrame, Dict[str, object]]:
    classified, summary, _reference_raw, _grouped_raw = _build_universe_snapshot_full(config)
    return classified, summary


def build_and_save_universe_snapshot(config: UniverseConfig) -> Tuple[Path, Path]:
    classified, summary, reference_raw, grouped_raw = _build_universe_snapshot_full(config)
    return _write_outputs(classified, summary, reference_raw, grouped_raw, config)
