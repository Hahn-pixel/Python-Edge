from __future__ import annotations

import json
import os
import sys
import traceback
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import requests

warnings.filterwarnings(
    "ignore",
    message="An input array is constant; the correlation coefficient is not defined.",
)

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
for p in [ROOT, SRC_DIR]:
    ps = str(p)
    if ps not in sys.path:
        sys.path.insert(0, ps)

from python_edge.model.alpha_factory_core import (
    SurvivorConfig,
    ValidationConfig,
    build_recipe_registry,
    derive_base_factory_inputs,
    generate_factory_alphas,
)
from python_edge.model.alpha_factory_specs import SEED_RECIPES

DEFAULT_BASE_URL = "https://api.massive.com"
PAUSE_ON_EXIT = str(os.getenv("PAUSE_ON_EXIT", "auto")).strip().lower()

_UNIVERSE_DIR = Path(os.getenv("UNIVERSE_OUTPUT_DIR", "artifacts/daily_cycle/universe"))
UNIVERSE_SNAPSHOT_FILE = Path(os.getenv(
    "UNIVERSE_SNAPSHOT_FILE",
    str(_UNIVERSE_DIR / "universe_snapshot.parquet"),
))
# DYNAMIC_UNIVERSE_FILE зберігаємо для backward compat (broad_sector fallback),
# але більше не використовуємо як основне джерело символів.
DYNAMIC_UNIVERSE_FILE = Path(os.getenv(
    "DYNAMIC_UNIVERSE_FILE",
    str(_UNIVERSE_DIR / "dynamic_universe.parquet"),
))

LIVE_ALPHA_OUT_DIR = Path(os.getenv("LIVE_ALPHA_OUT_DIR", "artifacts/live_alpha"))
LIVE_ALPHA_LOOKBACK_DAYS = int(os.getenv("LIVE_ALPHA_LOOKBACK_DAYS", "260"))
LIVE_ALPHA_MAX_SYMBOLS = int(os.getenv("LIVE_ALPHA_MAX_SYMBOLS", "500"))
LIVE_ALPHA_REQUEST_TIMEOUT_SEC = int(os.getenv("LIVE_ALPHA_REQUEST_TIMEOUT_SEC", "30"))
LIVE_ALPHA_SCOPE = str(os.getenv("LIVE_ALPHA_SCOPE", "registry")).strip().lower()
LIVE_ALPHA_SURVIVOR_TOP_N = int(os.getenv("LIVE_ALPHA_SURVIVOR_TOP_N", "6"))
LIVE_ALPHA_MIN_NON_NA = int(os.getenv("LIVE_ALPHA_MIN_NON_NA", "100"))
LIVE_ALPHA_MAX_NAN_RATIO = float(os.getenv("LIVE_ALPHA_MAX_NAN_RATIO", "0.995"))
LIVE_ALPHA_MIN_UNIQUE = int(os.getenv("LIVE_ALPHA_MIN_UNIQUE", "5"))
LIVE_ALPHA_PROXY_ENABLE = str(os.getenv("LIVE_ALPHA_PROXY_ENABLE", "1")).strip().lower() not in {"0", "false", "no", "off"}
LIVE_ALPHA_PROXY_TIMEOUT_SEC = int(os.getenv("LIVE_ALPHA_PROXY_TIMEOUT_SEC", str(LIVE_ALPHA_REQUEST_TIMEOUT_SEC)))
LIVE_ALPHA_INTERACTION_ENABLE = str(os.getenv("LIVE_ALPHA_INTERACTION_ENABLE", "1")).strip().lower() not in {"0", "false", "no", "off"}
LIVE_ALPHA_INTERACTION_TOP_K = int(os.getenv("LIVE_ALPHA_INTERACTION_TOP_K", "24"))
LIVE_ALPHA_INTERACTION_GATES = tuple(
    x.strip() for x in str(os.getenv("LIVE_ALPHA_INTERACTION_GATES", "oil_up|dollar_up|macro_risk_off")).split("|") if x.strip()
)

COMMODITY_PROXY_SYMBOLS: Tuple[str, ...] = ("GLD", "SLV", "USO", "DBC")
INTERNATIONAL_PROXY_SYMBOLS: Tuple[str, ...] = ("EWJ", "EWH", "EWT", "VGK", "SPY")
GLOBAL_REGIME_PROXY_SYMBOLS: Tuple[str, ...] = ("VIXY", "UUP", "TLT")
ALL_PROXY_SYMBOLS: Tuple[str, ...] = COMMODITY_PROXY_SYMBOLS + INTERNATIONAL_PROXY_SYMBOLS + GLOBAL_REGIME_PROXY_SYMBOLS


@dataclass(frozen=True)
class MassiveClient:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    timeout_sec: int = 30

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

    def get_daily_bars(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        url = f"{self.base_url.rstrip('/')}/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
        params: Optional[Dict[str, object]] = {"adjusted": "true", "sort": "asc", "limit": 50000}
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
            return pd.DataFrame(columns=["date", "symbol", "open", "high", "low", "close", "volume"])
        df = pd.DataFrame(rows).rename(columns={"t": "ts", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                raise RuntimeError(f"Daily agg payload for {symbol} missing column: {col}")
        df["date"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_convert(None).dt.normalize()
        df["symbol"] = str(symbol).upper()
        keep = df[["date", "symbol", "open", "high", "low", "close", "volume"]].copy()
        keep = keep.drop_duplicates(subset=["date", "symbol"], keep="last").sort_values(["date", "symbol"]).reset_index(drop=True)
        return keep


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


def _find_snapshot_date_column(snap: pd.DataFrame) -> str:
    for candidate in ["trade_date", "as_of_date", "date", "session_date"]:
        if candidate in snap.columns:
            return candidate
    raise RuntimeError("Universe snapshot does not contain trade_date/as_of_date/date/session_date")


def _symbol_column(snap: pd.DataFrame) -> str:
    for candidate in ["ticker", "symbol"]:
        if candidate in snap.columns:
            return candidate
    raise RuntimeError("Universe snapshot must contain ticker or symbol")


# ---------------------------------------------------------------------------
# Broad sector map loader
# ---------------------------------------------------------------------------

def _load_broad_sector_map() -> Dict[str, str]:
    """
    Load symbol → broad_sector from universe_snapshot.parquet.
    Falls back to dynamic_universe.parquet only if snapshot lacks broad_sector.
    Returns empty dict if broad_sector column not found in either file.
    """
    # PATCHED: universe_snapshot.parquet is now the primary source.
    # dynamic_universe.parquet is only used as a fallback for broad_sector,
    # never as the primary symbol source.
    for path in [UNIVERSE_SNAPSHOT_FILE, DYNAMIC_UNIVERSE_FILE]:
        if not path.exists():
            continue
        try:
            df = pd.read_parquet(path)
            if df.empty:
                continue
            sym_col = _symbol_column(df)
            if "broad_sector" not in df.columns:
                continue
            df = df.copy()
            df[sym_col] = df[sym_col].astype(str).str.upper().str.strip()
            df = df.drop_duplicates(subset=[sym_col])
            result: Dict[str, str] = dict(zip(df[sym_col], df["broad_sector"].astype(str)))
            print(
                f"[BROAD_SECTOR] loaded from {path.name}"
                f" symbols={len(result)} unique_sectors={len(set(result.values()))}"
            )
            return result
        except Exception as exc:
            print(f"[BROAD_SECTOR][WARN] failed to load from {path}: {exc!r}")
    print("[BROAD_SECTOR][WARN] broad_sector not found in any universe file — sector diagnostics disabled")
    return {}


def _attach_broad_sector(panel: pd.DataFrame, broad_sector_map: Dict[str, str]) -> pd.DataFrame:
    """Attach broad_sector column to panel. Unmapped symbols → 'other'."""
    if not broad_sector_map:
        return panel
    out = panel.copy()
    out["broad_sector"] = out["symbol"].astype(str).str.upper().map(broad_sector_map).fillna("other")
    print(
        f"[BROAD_SECTOR] attached to panel"
        f" symbols={out['symbol'].nunique()} unique_sectors={out['broad_sector'].nunique()}"
    )
    return out


# ---------------------------------------------------------------------------
# Universe loaders
# ---------------------------------------------------------------------------

def _load_selected_universe() -> tuple[list[str], str]:
    """
    Load selected symbols and as_of_date from universe_snapshot.parquet.

    PATCHED: removed _load_from_dynamic_universe() priority.
    Previously dynamic_universe.parquet was always preferred, causing the
    snapshot date to be stuck at its creation date (2026-04-10) even after
    universe_snapshot.parquet was refreshed daily. Now we always read from
    universe_snapshot.parquet which is updated by run_universe_builder.py.
    """
    if not UNIVERSE_SNAPSHOT_FILE.exists():
        raise FileNotFoundError(f"Universe snapshot not found: {UNIVERSE_SNAPSHOT_FILE}")
    snap = pd.read_parquet(UNIVERSE_SNAPSHOT_FILE)
    if snap.empty:
        raise RuntimeError("Universe snapshot is empty")
    if "selected" not in snap.columns:
        raise RuntimeError("Universe snapshot must contain selected")
    symbol_col = _symbol_column(snap)
    date_col = _find_snapshot_date_column(snap)
    snap[symbol_col] = snap[symbol_col].astype(str).str.strip().str.upper()
    snap["selected"] = snap["selected"].fillna(False).astype(bool)
    snap[date_col] = pd.to_datetime(snap[date_col], errors="coerce").dt.normalize()
    valid_dates = snap[date_col].dropna()
    if valid_dates.empty:
        raise RuntimeError(f"Universe snapshot has no valid dates in column={date_col}")
    latest_date = pd.Timestamp(valid_dates.max()).normalize()
    snap = snap.loc[snap[date_col] == latest_date].copy()
    snap = snap.loc[snap[symbol_col].ne("") & snap[symbol_col].ne("NAN")].copy()
    selected = snap.loc[snap["selected"]].copy()
    if selected.empty:
        raise RuntimeError("Universe snapshot has zero selected symbols on current date")
    sort_candidates = [c for c in ["selected_rank", "liquidity_rank", symbol_col] if c in selected.columns]
    if sort_candidates:
        selected = selected.sort_values(sort_candidates, ascending=[True] * len(sort_candidates), na_position="last")
    else:
        selected = selected.sort_values([symbol_col], ascending=[True])
    selected = selected.drop_duplicates(subset=[symbol_col], keep="first").reset_index(drop=True)
    symbols = selected[symbol_col].astype(str).tolist()[:LIVE_ALPHA_MAX_SYMBOLS]
    if not symbols:
        raise RuntimeError("Universe snapshot produced zero selected symbols after filtering current date")
    print(
        "[UNIVERSE][LOAD] source=universe_snapshot.parquet"
        f" snapshot={UNIVERSE_SNAPSHOT_FILE} symbol_col={symbol_col} date_col={date_col}"
        f" latest_date={latest_date.date().isoformat()}"
        f" rows_current={len(snap)} selected_current={len(selected)} max_symbols={LIVE_ALPHA_MAX_SYMBOLS}"
    )
    return symbols, latest_date.date().isoformat()


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def _build_recipes() -> tuple[Sequence[object], pd.DataFrame, str]:
    if LIVE_ALPHA_SCOPE == "seed":
        return tuple(SEED_RECIPES), pd.DataFrame(), "seed"
    survivor_cfg = SurvivorConfig(top_n_families=LIVE_ALPHA_SURVIVOR_TOP_N)
    recipes, detail, source = build_recipe_registry(survivor_cfg=survivor_cfg)
    return recipes, detail, source


def _fetch_history(symbols: Sequence[str], end_date: str, timeout_sec: int) -> pd.DataFrame:
    api_key = str(os.getenv("MASSIVE_API_KEY", "")).strip()
    client = MassiveClient(api_key=api_key, timeout_sec=timeout_sec)
    end_dt = datetime.fromisoformat(end_date).date()
    start_dt = end_dt - timedelta(days=LIVE_ALPHA_LOOKBACK_DAYS)
    start_date = start_dt.isoformat()
    frames: List[pd.DataFrame] = []
    failed: List[str] = []
    for idx, symbol in enumerate(symbols, start=1):
        try:
            df = client.get_daily_bars(symbol, start_date=start_date, end_date=end_date)
            if len(df):
                frames.append(df)
            else:
                failed.append(symbol)
            if idx % 25 == 0 or idx == len(symbols):
                print(f"[FETCH] {idx}/{len(symbols)} complete rows={sum(len(x) for x in frames)} failed={len(failed)}")
        except Exception:
            failed.append(symbol)
            print(f"[FETCH][WARN] symbol={symbol} failed")
    if not frames:
        raise RuntimeError("No daily bars fetched from Massive for selected universe")
    panel = pd.concat(frames, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    print(f"[FETCH] final_symbols={panel['symbol'].nunique()} rows={len(panel)} failed_symbols={len(failed)}")
    if failed:
        print(f"[FETCH][FAILED_SAMPLE] {failed[:20]}")
    return panel


def _add_target(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy().sort_values(["symbol", "date"]).reset_index(drop=True)
    next_close = out.groupby("symbol", sort=False)["close"].shift(-1)
    out["target_fwd_ret_1d"] = (pd.to_numeric(next_close, errors="coerce") / (pd.to_numeric(out["close"], errors="coerce") + 1e-12)) - 1.0
    return out.sort_values(["date", "symbol"]).reset_index(drop=True)


def _rolling_z(series: pd.Series, window: int) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").astype("float64")
    mean = x.rolling(window, min_periods=max(5, window // 2)).mean()
    std = x.rolling(window, min_periods=max(5, window // 2)).std()
    out = (x - mean) / (std + 1e-12)
    return out.replace([float("inf"), float("-inf")], pd.NA)


def _prepare_proxy_feature_frame(proxy_panel: pd.DataFrame) -> pd.DataFrame:
    if proxy_panel.empty:
        return pd.DataFrame(columns=["date"])
    work = proxy_panel.copy().sort_values(["symbol", "date"]).reset_index(drop=True)
    work["ret_1d"] = work.groupby("symbol", sort=False)["close"].pct_change(1)
    work["ret_5d"] = work.groupby("symbol", sort=False)["close"].pct_change(5)
    work["ret_20d"] = work.groupby("symbol", sort=False)["close"].pct_change(20)
    work["momentum_20d"] = work["ret_20d"]
    work["volatility_20d"] = (
        work.groupby("symbol", sort=False)["ret_1d"]
        .rolling(20, min_periods=10).std()
        .reset_index(level=0, drop=True)
    )
    work["z_ret_1d"] = work.groupby("symbol", sort=False)["ret_1d"].transform(lambda s: _rolling_z(s, 20))
    work["z_ret_5d"] = work.groupby("symbol", sort=False)["ret_5d"].transform(lambda s: _rolling_z(s, 20))
    work["z_ret_20d"] = work.groupby("symbol", sort=False)["ret_20d"].transform(lambda s: _rolling_z(s, 20))
    work["z_momentum_20d"] = work.groupby("symbol", sort=False)["momentum_20d"].transform(lambda s: _rolling_z(s, 20))
    work["z_volatility_20d"] = work.groupby("symbol", sort=False)["volatility_20d"].transform(lambda s: _rolling_z(s, 20))
    metrics = ["ret_1d", "ret_5d", "ret_20d", "momentum_20d", "volatility_20d",
               "z_ret_1d", "z_ret_5d", "z_ret_20d", "z_momentum_20d", "z_volatility_20d"]
    pivots: Dict[str, pd.DataFrame] = {
        metric: work.pivot(index="date", columns="symbol", values=metric) for metric in metrics
    }
    base_index = sorted(work["date"].dropna().unique())
    feature_parts: List[pd.DataFrame] = []
    per_proxy_cols: Dict[str, pd.Series] = {}
    for symbol in COMMODITY_PROXY_SYMBOLS + GLOBAL_REGIME_PROXY_SYMBOLS:
        prefix = symbol.lower()
        for metric in metrics:
            frame = pivots[metric]
            col_name = f"{prefix}_{metric}"
            per_proxy_cols[col_name] = frame[symbol] if symbol in frame.columns else pd.Series(index=base_index, dtype="float64")
    feature_parts.append(pd.DataFrame(per_proxy_cols, index=base_index))

    def _avg_cols(frame: pd.DataFrame, cols: List[str]) -> pd.Series:
        if not cols:
            return pd.Series(index=frame.index, dtype="float64")
        return frame[cols].mean(axis=1)

    asia_components = [sym for sym in ["EWJ", "EWH", "EWT"] if sym in pivots["ret_1d"].columns]
    europe_components = [sym for sym in ["VGK"] if sym in pivots["ret_1d"].columns]
    us_components = [sym for sym in ["SPY"] if sym in pivots["ret_1d"].columns]
    region_cols = {
        "asia_ret_1d": _avg_cols(pivots["ret_1d"], asia_components),
        "asia_ret_5d": _avg_cols(pivots["ret_5d"], asia_components),
        "asia_ret_20d": _avg_cols(pivots["ret_20d"], asia_components),
        "asia_momentum_20d": _avg_cols(pivots["momentum_20d"], asia_components),
        "asia_volatility_20d": _avg_cols(pivots["volatility_20d"], asia_components),
        "europe_ret_1d": _avg_cols(pivots["ret_1d"], europe_components),
        "europe_ret_5d": _avg_cols(pivots["ret_5d"], europe_components),
        "europe_ret_20d": _avg_cols(pivots["ret_20d"], europe_components),
        "europe_momentum_20d": _avg_cols(pivots["momentum_20d"], europe_components),
        "europe_volatility_20d": _avg_cols(pivots["volatility_20d"], europe_components),
        "us_ret_1d": _avg_cols(pivots["ret_1d"], us_components),
        "us_ret_5d": _avg_cols(pivots["ret_5d"], us_components),
        "us_ret_20d": _avg_cols(pivots["ret_20d"], us_components),
        "us_momentum_20d": _avg_cols(pivots["momentum_20d"], us_components),
        "us_volatility_20d": _avg_cols(pivots["volatility_20d"], us_components),
    }
    feature_parts.append(pd.DataFrame(region_cols, index=base_index))
    commodity_prefixes = {"gld", "slv", "uso", "dbc"}
    direct_proxy_df = feature_parts[0]
    basket_df = pd.DataFrame({
        "commodities_basket_ret_1d": direct_proxy_df[[c for c in direct_proxy_df.columns if c.endswith("_ret_1d") and c.split("_")[0] in commodity_prefixes]].mean(axis=1),
        "commodities_basket_ret_5d": direct_proxy_df[[c for c in direct_proxy_df.columns if c.endswith("_ret_5d") and c.split("_")[0] in commodity_prefixes]].mean(axis=1),
        "commodities_basket_ret_20d": direct_proxy_df[[c for c in direct_proxy_df.columns if c.endswith("_ret_20d") and c.split("_")[0] in commodity_prefixes]].mean(axis=1),
        "commodities_basket_momentum_20d": direct_proxy_df[[c for c in direct_proxy_df.columns if c.endswith("_momentum_20d") and c.split("_")[0] in commodity_prefixes]].mean(axis=1),
        "commodities_basket_volatility_20d": direct_proxy_df[[c for c in direct_proxy_df.columns if c.endswith("_volatility_20d") and c.split("_")[0] in commodity_prefixes]].mean(axis=1),
    }, index=base_index)
    feature_parts.append(basket_df)
    base_df = pd.concat(feature_parts, axis=1)
    derived_df = pd.DataFrame({
        "oil_up": (pd.to_numeric(base_df.get("uso_ret_1d"), errors="coerce") > 0.0).astype("float64"),
        "global_risk_on": (pd.concat([
            pd.to_numeric(base_df.get("us_ret_1d"), errors="coerce"),
            pd.to_numeric(base_df.get("asia_ret_1d"), errors="coerce"),
            pd.to_numeric(base_df.get("europe_ret_1d"), errors="coerce"),
            pd.to_numeric(base_df.get("commodities_basket_ret_1d"), errors="coerce"),
        ], axis=1).mean(axis=1) > 0.0).astype("float64"),
        "asia_us_lead": pd.to_numeric(base_df["asia_ret_1d"], errors="coerce").shift(1) - pd.to_numeric(base_df["us_ret_1d"], errors="coerce"),
        "europe_us_lead": pd.to_numeric(base_df["europe_ret_1d"], errors="coerce").shift(1) - pd.to_numeric(base_df["us_ret_1d"], errors="coerce"),
        "global_divergence": (pd.to_numeric(base_df["asia_ret_1d"], errors="coerce") + pd.to_numeric(base_df["europe_ret_1d"], errors="coerce")) / 2.0 - pd.to_numeric(base_df["us_ret_1d"], errors="coerce"),
        "vol_spike": (pd.to_numeric(base_df.get("vixy_ret_1d"), errors="coerce") > 0.0).astype("float64"),
        "dollar_up": (pd.to_numeric(base_df.get("uup_ret_5d"), errors="coerce") > 0.0).astype("float64"),
        "yield_up": (pd.to_numeric(base_df.get("tlt_ret_5d"), errors="coerce") < 0.0).astype("float64"),
        "duration_bid": (pd.to_numeric(base_df.get("tlt_ret_5d"), errors="coerce") > 0.0).astype("float64"),
        "vol_dollar_divergence": pd.to_numeric(base_df.get("vixy_ret_5d"), errors="coerce") - pd.to_numeric(base_df.get("uup_ret_5d"), errors="coerce"),
        "rates_equity_tension": (-pd.to_numeric(base_df.get("tlt_ret_5d"), errors="coerce")) - pd.to_numeric(base_df["us_ret_5d"], errors="coerce"),
    }, index=base_index)
    macro_gate_score = (
        pd.to_numeric(derived_df["vol_spike"], errors="coerce").fillna(0.0)
        + pd.to_numeric(derived_df["dollar_up"], errors="coerce").fillna(0.0)
        + pd.to_numeric(derived_df["duration_bid"], errors="coerce").fillna(0.0)
    )
    derived_df["macro_risk_off"] = (macro_gate_score >= 2.0).astype("float64")
    derived_df["macro_risk_on"] = (macro_gate_score <= 1.0).astype("float64")
    derived_df["macro_stress_score"] = (
        _rolling_z(pd.to_numeric(base_df.get("vixy_ret_5d"), errors="coerce"), 20).fillna(0.0)
        + _rolling_z(pd.to_numeric(base_df.get("uup_ret_5d"), errors="coerce"), 20).fillna(0.0)
        - _rolling_z(pd.to_numeric(base_df.get("tlt_ret_5d"), errors="coerce"), 20).fillna(0.0)
    ) / 3.0
    z_source_cols = [
        "commodities_basket_ret_1d", "commodities_basket_ret_5d", "commodities_basket_ret_20d",
        "commodities_basket_momentum_20d", "commodities_basket_volatility_20d",
        "asia_us_lead", "europe_us_lead", "global_divergence",
        "vol_dollar_divergence", "rates_equity_tension", "macro_stress_score",
    ]
    combined = pd.concat([base_df, derived_df], axis=1)
    z_df = pd.DataFrame(
        {f"{col}_z": _rolling_z(pd.to_numeric(combined[col], errors="coerce"), 20) for col in z_source_cols},
        index=base_index,
    )
    out = pd.concat([base_df, derived_df, z_df], axis=1).copy()
    out = out.reset_index().rename(columns={"index": "date"}).sort_values("date").reset_index(drop=True)
    return out


def _merge_proxy_features(panel: pd.DataFrame, proxy_feature_frame: pd.DataFrame) -> pd.DataFrame:
    if proxy_feature_frame.empty:
        return panel.copy()
    out = panel.merge(proxy_feature_frame, on="date", how="left")
    return out.sort_values(["date", "symbol"]).reset_index(drop=True)


def _add_interaction_layer(
    feature_panel: pd.DataFrame,
    alpha_frame: pd.DataFrame,
    manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not LIVE_ALPHA_INTERACTION_ENABLE:
        return alpha_frame, manifest
    if alpha_frame.empty:
        return alpha_frame, manifest
    required_cols = list(LIVE_ALPHA_INTERACTION_GATES)
    missing = [c for c in required_cols if c not in feature_panel.columns]
    if missing:
        print(f"[INTERACTION][WARN] missing feature columns for interaction layer: {missing}")
        return alpha_frame, manifest
    if "alpha" in manifest.columns and "selector_score" in manifest.columns:
        ranked = manifest.copy()
        ranked["selector_score"] = pd.to_numeric(ranked["selector_score"], errors="coerce").fillna(0.0)
        ranked = ranked.sort_values(["selector_score", "alpha"], ascending=[False, True])
        candidate_alphas = ranked["alpha"].astype(str).tolist()
    elif "alpha" in manifest.columns:
        candidate_alphas = manifest["alpha"].astype(str).tolist()
    else:
        candidate_alphas = [c for c in alpha_frame.columns if str(c).startswith("alpha_")]
    candidate_alphas = [c for c in candidate_alphas if c in alpha_frame.columns]
    candidate_alphas = candidate_alphas[: max(1, LIVE_ALPHA_INTERACTION_TOP_K)]
    if not candidate_alphas:
        return alpha_frame, manifest
    gates_all = {
        "oil_up": pd.to_numeric(feature_panel["oil_up"], errors="coerce").fillna(0.0).astype("float64"),
        "global_risk_on": pd.to_numeric(feature_panel["global_risk_on"], errors="coerce").fillna(0.0).astype("float64"),
        "vol_spike": pd.to_numeric(feature_panel["vol_spike"], errors="coerce").fillna(0.0).astype("float64"),
        "dollar_up": pd.to_numeric(feature_panel["dollar_up"], errors="coerce").fillna(0.0).astype("float64"),
        "yield_up": pd.to_numeric(feature_panel["yield_up"], errors="coerce").fillna(0.0).astype("float64"),
        "macro_risk_off": pd.to_numeric(feature_panel["macro_risk_off"], errors="coerce").fillna(0.0).astype("float64"),
        "macro_risk_on": pd.to_numeric(feature_panel["macro_risk_on"], errors="coerce").fillna(0.0).astype("float64"),
    }
    family_map = {
        "oil_up": "interaction_oil_up", "global_risk_on": "interaction_global_risk_on",
        "vol_spike": "interaction_vol_spike", "dollar_up": "interaction_dollar_up",
        "yield_up": "interaction_yield_up", "macro_risk_off": "interaction_macro_risk_off",
        "macro_risk_on": "interaction_macro_risk_on",
    }
    selected_gate_names = [g for g in LIVE_ALPHA_INTERACTION_GATES if g in gates_all]
    if not selected_gate_names:
        print(f"[INTERACTION][WARN] no enabled gates: {LIVE_ALPHA_INTERACTION_GATES}")
        return alpha_frame, manifest
    gates = {k: gates_all[k] for k in selected_gate_names}
    out = alpha_frame.copy()
    interaction_cols: Dict[str, pd.Series] = {}
    manifest_rows: List[Dict[str, object]] = []
    for alpha_col in candidate_alphas:
        alpha_series = pd.to_numeric(out[alpha_col], errors="coerce")
        for gate_name, gate_series in gates.items():
            new_col = f"{alpha_col}__x_{gate_name}"
            new_series = alpha_series * gate_series
            interaction_cols[new_col] = new_series
            manifest_rows.append({
                "alpha": new_col, "family": family_map[gate_name], "wave": "wave_context",
                "left": alpha_col, "modulator": gate_name, "transform": "raw",
                "regime": "none", "interaction": "global_gate", "lag": 0,
                "source": "post_build", "parents": alpha_col,
                "non_na": int(new_series.notna().sum()),
                "nan_ratio": float(new_series.isna().mean()),
                "unique": int(new_series.dropna().nunique()),
                "selector_score": 0.0, "shortlist_rank": 1_000_000,
            })
    if interaction_cols:
        out = pd.concat([out, pd.DataFrame(interaction_cols, index=out.index)], axis=1).copy()
    manifest_out = manifest.copy()
    if manifest_rows:
        manifest_out = pd.concat([manifest_out, pd.DataFrame(manifest_rows)], ignore_index=True, sort=False)
    print(f"[INTERACTION] base_alpha_count={len(candidate_alphas)} enabled_gates={selected_gate_names} added_columns={len(manifest_rows)}")
    return out, manifest_out


def _build_proxy_context(end_date: str) -> tuple[pd.DataFrame, Dict[str, object]]:
    if not LIVE_ALPHA_PROXY_ENABLE:
        return pd.DataFrame(columns=["date"]), {"proxy_enabled": 0, "proxy_rows": 0, "proxy_symbols": 0}
    proxy_panel = _fetch_history(ALL_PROXY_SYMBOLS, end_date=end_date, timeout_sec=LIVE_ALPHA_PROXY_TIMEOUT_SEC)
    feature_frame = _prepare_proxy_feature_frame(proxy_panel)
    return feature_frame, {
        "proxy_enabled": 1,
        "proxy_rows": int(len(proxy_panel)),
        "proxy_symbols": int(proxy_panel["symbol"].nunique()) if len(proxy_panel) else 0,
        "proxy_feature_rows": int(len(feature_frame)),
        "proxy_feature_columns": int(max(0, len(feature_frame.columns) - 1)),
        "commodity_proxy_symbols": int(len(COMMODITY_PROXY_SYMBOLS)),
        "international_proxy_symbols": int(len(INTERNATIONAL_PROXY_SYMBOLS)),
        "global_regime_proxy_symbols": int(len(GLOBAL_REGIME_PROXY_SYMBOLS)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print(f"[CFG] universe_snapshot_file={UNIVERSE_SNAPSHOT_FILE}")
    print(f"[CFG] live_alpha_out_dir={LIVE_ALPHA_OUT_DIR}")
    print(f"[CFG] live_alpha_scope={LIVE_ALPHA_SCOPE} survivor_top_n={LIVE_ALPHA_SURVIVOR_TOP_N}")
    print(f"[CFG] lookback_days={LIVE_ALPHA_LOOKBACK_DAYS} max_symbols={LIVE_ALPHA_MAX_SYMBOLS}")
    print(f"[CFG] proxy_enable={int(LIVE_ALPHA_PROXY_ENABLE)} interaction_enable={int(LIVE_ALPHA_INTERACTION_ENABLE)} interaction_top_k={LIVE_ALPHA_INTERACTION_TOP_K} interaction_gates={'|'.join(LIVE_ALPHA_INTERACTION_GATES)}")

    broad_sector_map = _load_broad_sector_map()

    symbols, end_date = _load_selected_universe()
    print(f"[UNIVERSE] selected_symbols={len(symbols)} as_of_date={end_date}")

    raw_panel = _fetch_history(symbols, end_date=end_date, timeout_sec=LIVE_ALPHA_REQUEST_TIMEOUT_SEC)
    proxy_feature_frame, proxy_meta = _build_proxy_context(end_date=end_date)
    enriched_panel = _merge_proxy_features(raw_panel, proxy_feature_frame)

    feature_panel = derive_base_factory_inputs(enriched_panel)
    feature_panel = _add_target(feature_panel)

    feature_panel = _attach_broad_sector(feature_panel, broad_sector_map)

    recipes, recipe_detail, recipe_source = _build_recipes()
    print(f"[FACTORY] recipe_source={recipe_source} recipe_count={len(recipes)}")

    build_result = generate_factory_alphas(
        feature_panel,
        recipes=recipes,
        cfg=ValidationConfig(
            max_nan_ratio=LIVE_ALPHA_MAX_NAN_RATIO,
            min_non_na=LIVE_ALPHA_MIN_NON_NA,
            min_unique=LIVE_ALPHA_MIN_UNIQUE,
        ),
    )

    interaction_frame, interaction_manifest = _add_interaction_layer(
        feature_panel=feature_panel,
        alpha_frame=build_result.frame,
        manifest=build_result.manifest,
    )

    if "broad_sector" in feature_panel.columns and "broad_sector" not in interaction_frame.columns:
        if "symbol" in interaction_frame.columns:
            sector_lookup = (
                feature_panel[["symbol", "broad_sector"]]
                .drop_duplicates(subset=["symbol"])
                .set_index("symbol")["broad_sector"]
            )
            interaction_frame = interaction_frame.copy()
            interaction_frame["broad_sector"] = (
                interaction_frame["symbol"].astype(str).str.upper().map(sector_lookup).fillna("other")
            )

    out_dir = LIVE_ALPHA_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = out_dir / "live_alpha_snapshot.parquet"
    manifest_path = out_dir / "live_alpha_manifest.csv"
    dropped_path = out_dir / "live_alpha_dropped.csv"
    meta_path = out_dir / "live_alpha_meta.json"
    base_path = out_dir / "live_feature_snapshot.parquet"
    recipe_detail_path = out_dir / "live_factory_recipe_detail.csv"
    proxy_feature_path = out_dir / "live_proxy_feature_snapshot.parquet"

    feature_panel.to_parquet(base_path, index=False)
    proxy_feature_frame.to_parquet(proxy_feature_path, index=False)
    interaction_frame.to_parquet(snapshot_path, index=False)
    interaction_manifest.to_csv(manifest_path, index=False)
    build_result.dropped.to_csv(dropped_path, index=False)
    if len(recipe_detail):
        recipe_detail.to_csv(recipe_detail_path, index=False)

    alpha_cols = [c for c in interaction_frame.columns if str(c).startswith("alpha_")]
    meta = {
        "as_of_date": end_date,
        "selected_symbols": len(symbols),
        "fetched_symbols": int(raw_panel["symbol"].nunique()),
        "rows_raw": int(len(raw_panel)),
        "rows_features": int(len(feature_panel)),
        "rows_snapshot": int(len(interaction_frame)),
        "alpha_columns_kept": int(len(alpha_cols)),
        "alpha_columns_dropped": int(len(build_result.dropped)),
        "recipe_source": recipe_source,
        "recipe_count_requested": int(len(recipes)),
        "broad_sector_attached": int(bool(broad_sector_map)),
        "broad_sector_unique": int(interaction_frame["broad_sector"].nunique()) if "broad_sector" in interaction_frame.columns else 0,
        **proxy_meta,
        "interaction_added": int(len([c for c in interaction_frame.columns if "__x_" in str(c)])),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] base_snapshot={base_path}")
    print(f"[OK] proxy_feature_snapshot={proxy_feature_path}")
    print(f"[OK] live_alpha_snapshot={snapshot_path}")
    print(f"[OK] manifest={manifest_path}")
    print(f"[OK] dropped={dropped_path}")
    print(f"[OK] meta={meta_path}")
    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 1
    _safe_exit(rc)
