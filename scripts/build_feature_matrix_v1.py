from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from python_edge.model.alpha_factory_core import derive_base_factory_inputs

DEFAULT_SOURCE_CANDIDATES = (
    ROOT / "data" / "features" / "feature_matrix_source.parquet",
    ROOT / "data" / "features" / "base_ohlcv.parquet",
    ROOT / "data" / "raw" / "base_ohlcv.parquet",
    ROOT / "data" / "alpha_library_v2" / "alpha_library_v2.parquet",
)
OUT_DIR = ROOT / "data" / "features"
OUT_PARQUET = OUT_DIR / "feature_matrix_v1.parquet"
OUT_META = OUT_DIR / "feature_matrix_v1.meta.json"

REQUIRED_MIN_COLS = ["date", "symbol", "open", "high", "low", "close", "volume"]


def _pick_source_file() -> Path:
    env_path = os.environ.get("FEATURE_SOURCE_FILE", "").strip()
    if env_path:
        p = Path(env_path)
        if not p.is_absolute():
            p = (ROOT / env_path).resolve()
        return p
    for p in DEFAULT_SOURCE_CANDIDATES:
        if p.exists():
            return p
    return DEFAULT_SOURCE_CANDIDATES[-1]


def _normalize_input(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.loc[:, ~out.columns.duplicated()].copy()
    alpha_cols = [c for c in out.columns if str(c).startswith("alpha_")]
    if alpha_cols:
        out = out.drop(columns=alpha_cols).copy()
    rename_map = {}
    cols_lower = {str(c).lower(): str(c) for c in out.columns}
    aliases = {
        "date": ["date", "datetime", "timestamp", "time"],
        "symbol": ["symbol", "ticker", "sym"],
        "open": ["open", "o"],
        "high": ["high", "h"],
        "low": ["low", "l"],
        "close": ["close", "c", "adj_close", "adjclose"],
        "volume": ["volume", "v", "vol"],
    }
    for target, cands in aliases.items():
        if target in out.columns:
            continue
        for cand in cands:
            if cand in cols_lower:
                rename_map[cols_lower[cand]] = target
                break
    if rename_map:
        out = out.rename(columns=rename_map)

    missing = [c for c in REQUIRED_MIN_COLS if c not in out.columns]
    if missing:
        raise RuntimeError(f"build_feature_matrix_v1: missing required columns after normalization: {missing}")

    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["symbol"] = out["symbol"].astype(str).str.strip()
    out = out.dropna(subset=["date", "symbol"]).copy()
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["open", "high", "low", "close", "volume"]).copy()
    out = out.sort_values(["symbol", "date"]).reset_index(drop=True)
    out = out.loc[:, ~out.columns.duplicated()].copy()
    return out


def main() -> None:
    source_file = _pick_source_file()
    if not source_file.exists():
        raise FileNotFoundError(
            "Feature source file not found. Set FEATURE_SOURCE_FILE to a parquet/csv containing at least: "
            f"{REQUIRED_MIN_COLS}. Tried default path: {source_file}"
        )

    if source_file.suffix.lower() == ".parquet":
        raw_df = pd.read_parquet(source_file)
    elif source_file.suffix.lower() == ".csv":
        raw_df = pd.read_csv(source_file)
    else:
        raise RuntimeError(f"Unsupported FEATURE_SOURCE_FILE extension: {source_file.suffix}")

    if raw_df.empty:
        raise RuntimeError(f"Feature source file is empty: {source_file}")

    norm_df = _normalize_input(raw_df)
    feat_df = derive_base_factory_inputs(norm_df)
    if feat_df.empty:
        raise RuntimeError("derive_base_factory_inputs returned empty frame")
    feat_df = feat_df.loc[:, ~feat_df.columns.duplicated()].copy()
    alpha_cols = [c for c in feat_df.columns if str(c).startswith("alpha_")]
    if alpha_cols:
        feat_df = feat_df.drop(columns=alpha_cols).copy()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    feat_df.to_parquet(OUT_PARQUET, index=False)

    meta = {
        "dropped_alpha_columns_from_source": alpha_cols,
        "source_file": str(source_file),
        "rows": int(len(feat_df)),
        "cols": int(len(feat_df.columns)),
        "symbols": int(feat_df["symbol"].nunique()) if "symbol" in feat_df.columns else 0,
        "dates": int(feat_df["date"].nunique()) if "date" in feat_df.columns else 0,
        "columns": list(feat_df.columns),
    }
    OUT_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[FEATURE_MATRIX] source={source_file}")
    print(f"[FEATURE_MATRIX] rows={len(feat_df)} cols={len(feat_df.columns)}")
    print(f"[FEATURE_MATRIX] symbols={feat_df['symbol'].nunique()} dates={feat_df['date'].nunique()}")
    print(f"[ARTIFACT] {OUT_PARQUET}")
    print(f"[ARTIFACT] {OUT_META}")


if __name__ == "__main__":
    rc = 1
    try:
        main()
        rc = 0
    except Exception:
        traceback.print_exc()
        rc = 1
    try:
        if os.environ.get("PAUSE_ON_EXIT", "1").strip().lower() not in {"0", "false", "no", "off"}:
            print(f"\n[EXIT] code={rc}")
            input("Press Enter to exit...")
    except Exception:
        pass
    raise SystemExit(rc)
