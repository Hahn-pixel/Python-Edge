from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class AggsLoadResult:
    symbol: str
    tf: str
    path: Path
    df: pd.DataFrame


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _pick_full_file(ticker_dir: Path, tf: str, start: str, end: str) -> Optional[Path]:
    p = ticker_dir / f"aggs_{tf}_{start}_{end}__FULL.json"
    return p if p.exists() else None


def _list_shards(ticker_dir: Path, tf: str) -> List[Path]:
    return sorted(ticker_dir.glob(f"aggs_{tf}_????-??-??_????-??-??.json"))


def _json_to_df_aggs(obj: dict) -> pd.DataFrame:
    rows = obj.get("results") or []
    if not rows:
        return pd.DataFrame(columns=["t", "o", "h", "l", "c", "v", "vw", "n"])
    df = pd.DataFrame.from_records(rows)
    # expected columns: t,o,h,l,c,v,vw,n (Polygon style). Keep what exists.
    # Ensure t is int, ms epoch.
    df["t"] = pd.to_numeric(df["t"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["t"]).copy()
    df["t"] = df["t"].astype("int64")
    df = df.sort_values("t").drop_duplicates("t", keep="first")
    return df


def load_aggs(
    dataset_root: Path,
    symbol: str,
    tf: str,
    start: str,
    end: str,
    prefer_full: bool = True,
) -> AggsLoadResult:
    """
    Loads aggs for a symbol and timeframe.
    Expects folder layout:
      data/raw/massive_dataset/<SYMBOL>/aggs_<tf>_...json
    Prefers FULL file if present.
    """
    tdir = dataset_root / symbol
    if not tdir.exists():
        raise FileNotFoundError(f"Missing ticker dir: {tdir}")

    full = _pick_full_file(tdir, tf, start, end) if prefer_full else None
    if full is not None:
        obj = _read_json(full)
        df = _json_to_df_aggs(obj)
        return AggsLoadResult(symbol=symbol, tf=tf, path=full, df=df)

    shards = _list_shards(tdir, tf)
    if not shards:
        raise FileNotFoundError(f"No shard files for {symbol} tf={tf} in {tdir}")

    dfs = []
    for p in shards:
        obj = _read_json(p)
        dfs.append(_json_to_df_aggs(obj))

    df = pd.concat(dfs, axis=0, ignore_index=True) if dfs else pd.DataFrame()
    if not df.empty:
        df = df.sort_values("t").drop_duplicates("t", keep="first").reset_index(drop=True)

    return AggsLoadResult(symbol=symbol, tf=tf, path=shards[0].parent, df=df)


def to_daily_index(df_aggs: pd.DataFrame, tz: str = "UTC") -> pd.DataFrame:
    """
    Adds:
      - dt_utc: UTC timestamp
      - date: YYYY-MM-DD (UTC date)
    """
    if df_aggs.empty:
        return df_aggs.copy()
    out = df_aggs.copy()
    out["dt_utc"] = pd.to_datetime(out["t"], unit="ms", utc=True)
    out["date"] = out["dt_utc"].dt.date.astype(str)
    return out