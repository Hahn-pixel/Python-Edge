from __future__ import annotations

import json
import os
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd
import requests

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

UNIVERSE_SNAPSHOT_FILE = Path(os.getenv("UNIVERSE_SNAPSHOT_FILE", "artifacts/daily_cycle/universe/universe_snapshot.parquet"))
LIVE_ALPHA_OUT_DIR = Path(os.getenv("LIVE_ALPHA_OUT_DIR", "artifacts/live_alpha"))
LIVE_ALPHA_LOOKBACK_DAYS = int(os.getenv("LIVE_ALPHA_LOOKBACK_DAYS", "260"))
LIVE_ALPHA_MAX_SYMBOLS = int(os.getenv("LIVE_ALPHA_MAX_SYMBOLS", "500"))
LIVE_ALPHA_REQUEST_TIMEOUT_SEC = int(os.getenv("LIVE_ALPHA_REQUEST_TIMEOUT_SEC", "30"))
LIVE_ALPHA_SCOPE = str(os.getenv("LIVE_ALPHA_SCOPE", "registry")).strip().lower()
LIVE_ALPHA_SURVIVOR_TOP_N = int(os.getenv("LIVE_ALPHA_SURVIVOR_TOP_N", "6"))
LIVE_ALPHA_MIN_NON_NA = int(os.getenv("LIVE_ALPHA_MIN_NON_NA", "100"))
LIVE_ALPHA_MAX_NAN_RATIO = float(os.getenv("LIVE_ALPHA_MAX_NAN_RATIO", "0.995"))
LIVE_ALPHA_MIN_UNIQUE = int(os.getenv("LIVE_ALPHA_MIN_UNIQUE", "5"))


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
        params: Optional[Dict[str, object]] = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
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


def _load_selected_universe() -> tuple[list[str], str]:
    if not UNIVERSE_SNAPSHOT_FILE.exists():
        raise FileNotFoundError(f"Universe snapshot not found: {UNIVERSE_SNAPSHOT_FILE}")

    snap = pd.read_parquet(UNIVERSE_SNAPSHOT_FILE)
    if snap.empty:
        raise RuntimeError("Universe snapshot is empty")
    if "ticker" not in snap.columns or "selected" not in snap.columns:
        raise RuntimeError("Universe snapshot must contain ticker and selected")

    date_col = _find_snapshot_date_column(snap)
    snap["ticker"] = snap["ticker"].astype(str).str.strip().str.upper()
    snap["selected"] = snap["selected"].fillna(False).astype(bool)
    snap[date_col] = pd.to_datetime(snap[date_col], errors="coerce").dt.normalize()

    valid_dates = snap[date_col].dropna()
    if valid_dates.empty:
        raise RuntimeError(f"Universe snapshot has no valid dates in column={date_col}")
    latest_date = pd.Timestamp(valid_dates.max()).normalize()
    snap = snap.loc[snap[date_col] == latest_date].copy()

    snap = snap.loc[snap["ticker"].ne("") & snap["ticker"].ne("NAN")].copy()
    selected = snap.loc[snap["selected"]].copy()
    if selected.empty:
        raise RuntimeError("Universe snapshot has zero selected symbols on current date")

    if "liquidity_rank" in selected.columns:
        selected["liquidity_rank"] = pd.to_numeric(selected["liquidity_rank"], errors="coerce")
        selected = selected.sort_values(["liquidity_rank", "ticker"], ascending=[True, True], na_position="last")
    else:
        selected = selected.sort_values(["ticker"], ascending=[True])

    selected = selected.drop_duplicates(subset=["ticker"], keep="first").reset_index(drop=True)
    symbols = selected["ticker"].astype(str).tolist()[:LIVE_ALPHA_MAX_SYMBOLS]
    if not symbols:
        raise RuntimeError("Universe snapshot produced zero selected symbols after filtering current date")

    print(
        "[UNIVERSE][LOAD] "
        f"snapshot={UNIVERSE_SNAPSHOT_FILE} date_col={date_col} latest_date={latest_date.date().isoformat()} "
        f"rows_current={len(snap)} selected_current={len(selected)} max_symbols={LIVE_ALPHA_MAX_SYMBOLS}"
    )
    return symbols, latest_date.date().isoformat()


def _build_recipes() -> tuple[Sequence[object], pd.DataFrame, str]:
    if LIVE_ALPHA_SCOPE == "seed":
        return tuple(SEED_RECIPES), pd.DataFrame(), "seed"
    survivor_cfg = SurvivorConfig(top_n_families=LIVE_ALPHA_SURVIVOR_TOP_N)
    recipes, detail, source = build_recipe_registry(survivor_cfg=survivor_cfg)
    return recipes, detail, source


def _fetch_history(symbols: Sequence[str], end_date: str) -> pd.DataFrame:
    api_key = str(os.getenv("MASSIVE_API_KEY", "")).strip()
    client = MassiveClient(api_key=api_key, timeout_sec=LIVE_ALPHA_REQUEST_TIMEOUT_SEC)
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


def main() -> int:
    print(f"[CFG] universe_snapshot_file={UNIVERSE_SNAPSHOT_FILE}")
    print(f"[CFG] live_alpha_out_dir={LIVE_ALPHA_OUT_DIR}")
    print(f"[CFG] live_alpha_scope={LIVE_ALPHA_SCOPE} survivor_top_n={LIVE_ALPHA_SURVIVOR_TOP_N}")
    print(f"[CFG] lookback_days={LIVE_ALPHA_LOOKBACK_DAYS} max_symbols={LIVE_ALPHA_MAX_SYMBOLS}")

    symbols, end_date = _load_selected_universe()
    print(f"[UNIVERSE] selected_symbols={len(symbols)} as_of_date={end_date}")

    raw_panel = _fetch_history(symbols, end_date=end_date)
    feature_panel = derive_base_factory_inputs(raw_panel)
    feature_panel = _add_target(feature_panel)

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
    out_dir = LIVE_ALPHA_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = out_dir / "live_alpha_snapshot.parquet"
    manifest_path = out_dir / "live_alpha_manifest.csv"
    dropped_path = out_dir / "live_alpha_dropped.csv"
    meta_path = out_dir / "live_alpha_meta.json"
    base_path = out_dir / "live_feature_snapshot.parquet"
    recipe_detail_path = out_dir / "live_factory_recipe_detail.csv"

    feature_panel.to_parquet(base_path, index=False)
    build_result.frame.to_parquet(snapshot_path, index=False)
    build_result.manifest.to_csv(manifest_path, index=False)
    build_result.dropped.to_csv(dropped_path, index=False)
    if len(recipe_detail):
        recipe_detail.to_csv(recipe_detail_path, index=False)

    alpha_cols = [c for c in build_result.frame.columns if str(c).startswith("alpha_")]
    meta = {
        "as_of_date": end_date,
        "selected_symbols": len(symbols),
        "fetched_symbols": int(raw_panel["symbol"].nunique()),
        "rows_raw": int(len(raw_panel)),
        "rows_features": int(len(feature_panel)),
        "rows_snapshot": int(len(build_result.frame)),
        "alpha_columns_kept": int(len(alpha_cols)),
        "alpha_columns_dropped": int(len(build_result.dropped)),
        "recipe_source": recipe_source,
        "recipe_count_requested": int(len(recipes)),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] base_snapshot={base_path}")
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
