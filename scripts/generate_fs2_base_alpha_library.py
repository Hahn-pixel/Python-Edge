from __future__ import annotations

import json
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
for p in [ROOT, SRC_DIR]:
    ps = str(p)
    if ps not in sys.path:
        sys.path.insert(0, ps)

EPS = 1e-12

FEATURE_V2_FILE = Path(os.getenv("FEATURE_V2_FILE", "data/features/feature_matrix_v2.parquet"))
FEATURE_V2_MANIFEST = Path(os.getenv("FEATURE_V2_MANIFEST", "data/features/feature_matrix_v2_manifest.csv"))
FEATURE_V2_DIAG_CSV = Path(os.getenv("FEATURE_V2_DIAG_CSV", "data/features/feature_matrix_v2_diag.csv"))
OUT_DIR = Path(os.getenv("FS2_ALPHA_OUT_DIR", "data/alpha_library_fs2_base"))
OUT_PARQUET = OUT_DIR / "alpha_library_fs2_base.parquet"
OUT_MANIFEST = OUT_DIR / "alpha_library_fs2_base.manifest.csv"
OUT_DROPPED = OUT_DIR / "alpha_library_fs2_base.dropped.csv"
OUT_META = OUT_DIR / "alpha_library_fs2_base.meta.json"
PAUSE_ON_EXIT = str(os.getenv("PAUSE_ON_EXIT", "auto")).strip().lower()

TARGET_COL = str(os.getenv("TARGET_COL", "target_fwd_ret_1d")).strip()
MIN_NON_NA = int(os.getenv("MIN_NON_NA", "200"))
MIN_UNIQUE = int(os.getenv("MIN_UNIQUE", "5"))
MAX_NAN_RATIO = float(os.getenv("MAX_NAN_RATIO", "0.995"))
MAX_BASE_FEATURES = int(os.getenv("MAX_BASE_FEATURES", "0"))
INCLUDE_Z_FEATURES = str(os.getenv("INCLUDE_Z_FEATURES", "1")).strip().lower() not in {"0", "false", "no", "off"}
EXCLUDE_TARGET_DERIVED = str(os.getenv("EXCLUDE_TARGET_DERIVED", "1")).strip().lower() not in {"0", "false", "no", "off"}
EXCLUDE_MARKET_WIDE_CONSTANTS = str(os.getenv("EXCLUDE_MARKET_WIDE_CONSTANTS", "1")).strip().lower() not in {"0", "false", "no", "off"}
TOPK_PRINT = int(os.getenv("TOPK_PRINT", "40"))

ENABLE_RAW = str(os.getenv("ENABLE_RAW", "1")).strip().lower() not in {"0", "false", "no", "off"}
ENABLE_SIGNED_LOG = str(os.getenv("ENABLE_SIGNED_LOG", "1")).strip().lower() not in {"0", "false", "no", "off"}
ENABLE_EMA3 = str(os.getenv("ENABLE_EMA3", "1")).strip().lower() not in {"0", "false", "no", "off"}
ENABLE_LAG1 = str(os.getenv("ENABLE_LAG1", "1")).strip().lower() not in {"0", "false", "no", "off"}
ENABLE_TANH_Z = str(os.getenv("ENABLE_TANH_Z", "1")).strip().lower() not in {"0", "false", "no", "off"}


@dataclass(frozen=True)
class AlphaRecipe:
    alpha: str
    base_feature: str
    transform: str
    cluster: str
    source: str
    formula: str


def _enable_line_buffering() -> None:
    for stream_name in ["stdout", "stderr"]:
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(line_buffering=True)
            except Exception:
                pass


def _should_pause() -> bool:
    if PAUSE_ON_EXIT in {"0", "false", "no", "off"}:
        return False
    if PAUSE_ON_EXIT in {"1", "true", "yes", "on"}:
        return True
    stdin_obj = getattr(sys, "stdin", None)
    stdout_obj = getattr(sys, "stdout", None)
    stdin_ok = bool(stdin_obj is not None and hasattr(stdin_obj, "isatty") and stdin_obj.isatty())
    stdout_ok = bool(stdout_obj is not None and hasattr(stdout_obj, "isatty") and stdout_obj.isatty())
    return stdin_ok and stdout_ok


def _safe_exit(code: int) -> None:
    if _should_pause():
        try:
            print("")
            print(f"[EXIT] code={code}")
            input("Press Enter to exit...")
        except Exception:
            pass
    raise SystemExit(code)


def _must_exist(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("float64")


def _signed_log(s: pd.Series) -> pd.Series:
    x = _num(s)
    return np.sign(x) * np.log1p(np.abs(x))


def _robust_cs_zscore(frame: pd.DataFrame, col: str) -> pd.Series:
    def _z(g: pd.Series) -> pd.Series:
        x = _num(g)
        med = float(x.median()) if x.notna().any() else 0.0
        mad = float((x - med).abs().median()) if x.notna().any() else 0.0
        if mad > EPS:
            z = (x - med) / (1.4826 * mad)
        else:
            mean = float(x.mean()) if x.notna().any() else 0.0
            std = float(x.std(ddof=0)) if x.notna().any() else 0.0
            z = (x - mean) / (std + EPS)
        return z.replace([np.inf, -np.inf], np.nan)
    return frame.groupby("date", sort=False)[col].transform(_z)


def _tanh_z(frame: pd.DataFrame, col: str) -> pd.Series:
    return np.tanh(_robust_cs_zscore(frame, col))


def _ema3_by_symbol(frame: pd.DataFrame, col: str) -> pd.Series:
    return frame.groupby("symbol", sort=False)[col].transform(lambda s: _num(s).ewm(span=3, adjust=False, min_periods=1).mean())


def _lag1_by_symbol(frame: pd.DataFrame, col: str) -> pd.Series:
    return frame.groupby("symbol", sort=False)[col].shift(1)


def _base_columns(df: pd.DataFrame) -> List[str]:
    cols = ["date", "symbol", "open", "high", "low", "close", "volume", TARGET_COL]
    keep = [c for c in cols if c in df.columns]
    if "date" not in keep or "symbol" not in keep or TARGET_COL not in keep:
        raise RuntimeError(f"Required columns missing. Need date, symbol, {TARGET_COL}")
    return keep


def _is_target_derived(source: str, formula: str, feature: str) -> bool:
    hay = " | ".join([str(source), str(formula), str(feature)]).lower()
    bad_terms = [
        TARGET_COL.lower(),
        "target",
        "rank_pct(target)",
        "cs_dispersion",
        "winner_loser",
        "top_tail_strength",
        "bottom_tail_strength",
    ]
    return any(term in hay for term in bad_terms)


def _is_market_wide_constant(cluster: str, feature: str) -> bool:
    feat = str(feature)
    if not EXCLUDE_MARKET_WIDE_CONSTANTS:
        return False
    if cluster == "cs_pressure" and any(x in feat for x in ["breadth", "dispersion", "target_rank", "winner_loser"]):
        return True
    return False


def _load_feature_manifest() -> pd.DataFrame:
    _must_exist(FEATURE_V2_MANIFEST, "feature v2 manifest")
    manifest = pd.read_csv(FEATURE_V2_MANIFEST)
    if "name" not in manifest.columns:
        raise RuntimeError(f"Manifest missing name column: {FEATURE_V2_MANIFEST}")
    return manifest


def _load_diag() -> pd.DataFrame:
    if FEATURE_V2_DIAG_CSV.exists():
        diag = pd.read_csv(FEATURE_V2_DIAG_CSV)
        if "feature" in diag.columns:
            return diag
    return pd.DataFrame(columns=["feature", "non_na", "nan_ratio", "unique", "keep"])


def choose_safe_base_features(feature_df: pd.DataFrame, manifest_df: pd.DataFrame, diag_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    work = manifest_df.copy()
    work = work.rename(columns={"name": "feature"})
    work = work.merge(diag_df, on="feature", how="left", suffixes=("", "_diag"))
    for col, default in [("cluster", "unknown"), ("source", "unknown"), ("formula", ""), ("non_na", 0), ("nan_ratio", 1.0), ("unique", 0), ("keep", 0)]:
        if col not in work.columns:
            work[col] = default
    work["feature_present"] = work["feature"].astype(str).isin(set(feature_df.columns)).astype(int)
    work["is_z_feature"] = work["feature"].astype(str).str.startswith("z_").astype(int)
    work["target_derived"] = work.apply(lambda r: int(_is_target_derived(str(r["source"]), str(r["formula"]), str(r["feature"]))), axis=1)
    work["market_wide_constant"] = work.apply(lambda r: int(_is_market_wide_constant(str(r["cluster"]), str(r["feature"]))), axis=1)
    work["eligible"] = 1
    work.loc[work["feature_present"] != 1, "eligible"] = 0
    work.loc[pd.to_numeric(work["non_na"], errors="coerce").fillna(0) < MIN_NON_NA, "eligible"] = 0
    work.loc[pd.to_numeric(work["nan_ratio"], errors="coerce").fillna(1.0) > MAX_NAN_RATIO, "eligible"] = 0
    work.loc[pd.to_numeric(work["unique"], errors="coerce").fillna(0) < MIN_UNIQUE, "eligible"] = 0
    if not INCLUDE_Z_FEATURES:
        work.loc[work["is_z_feature"] == 1, "eligible"] = 0
    if EXCLUDE_TARGET_DERIVED:
        work.loc[work["target_derived"] == 1, "eligible"] = 0
    if EXCLUDE_MARKET_WIDE_CONSTANTS:
        work.loc[work["market_wide_constant"] == 1, "eligible"] = 0
    safe_df = work.loc[work["eligible"] == 1].copy()
    dropped_df = work.loc[work["eligible"] != 1].copy()
    safe_df = safe_df.sort_values(["cluster", "is_z_feature", "feature"], ascending=[True, True, True]).reset_index(drop=True)
    if MAX_BASE_FEATURES > 0:
        safe_df = safe_df.head(MAX_BASE_FEATURES).copy()
    return safe_df, dropped_df


def build_recipes(base_feature_df: pd.DataFrame) -> List[AlphaRecipe]:
    recipes: List[AlphaRecipe] = []
    for _, row in base_feature_df.iterrows():
        feature = str(row["feature"])
        cluster = str(row["cluster"])
        source = str(row["source"])
        formula = str(row["formula"])
        family = feature.replace("z_", "") if feature.startswith("z_") else feature
        if ENABLE_RAW:
            recipes.append(AlphaRecipe(alpha=f"alpha_{family}__raw", base_feature=feature, transform="raw", cluster=cluster, source=source, formula=formula))
        if ENABLE_SIGNED_LOG:
            recipes.append(AlphaRecipe(alpha=f"alpha_{family}__signed_log", base_feature=feature, transform="signed_log", cluster=cluster, source=source, formula=formula))
        if ENABLE_EMA3:
            recipes.append(AlphaRecipe(alpha=f"alpha_{family}__ema3", base_feature=feature, transform="ema3", cluster=cluster, source=source, formula=formula))
        if ENABLE_LAG1:
            recipes.append(AlphaRecipe(alpha=f"alpha_{family}__lag1", base_feature=feature, transform="lag1", cluster=cluster, source=source, formula=formula))
        if ENABLE_TANH_Z and not feature.startswith("z_"):
            recipes.append(AlphaRecipe(alpha=f"alpha_{family}__tanh_z", base_feature=feature, transform="tanh_z", cluster=cluster, source=source, formula=formula))
    dedup: Dict[str, AlphaRecipe] = {}
    for r in recipes:
        dedup[r.alpha] = r
    return list(dedup.values())


def materialize_alpha(frame: pd.DataFrame, recipe: AlphaRecipe) -> pd.Series:
    x = _num(frame[recipe.base_feature])
    if recipe.transform == "raw":
        return x
    if recipe.transform == "signed_log":
        return _signed_log(x)
    if recipe.transform == "ema3":
        return _ema3_by_symbol(frame.assign(_x=x), "_x")
    if recipe.transform == "lag1":
        return _lag1_by_symbol(frame.assign(_x=x), "_x")
    if recipe.transform == "tanh_z":
        return _tanh_z(frame.assign(_x=x), "_x")
    raise RuntimeError(f"Unsupported transform: {recipe.transform}")


def validate_alpha_series(s: pd.Series) -> Dict[str, object]:
    x = _num(s)
    non_na = int(x.notna().sum())
    nan_ratio = float(x.isna().mean()) if len(x) else 1.0
    unique = int(x.nunique(dropna=True)) if len(x) else 0
    std = float(x.std(ddof=0)) if non_na else float("nan")
    keep = int(non_na >= MIN_NON_NA and nan_ratio <= MAX_NAN_RATIO and unique >= MIN_UNIQUE and pd.notna(std) and std > 0.0)
    return {
        "non_na": non_na,
        "nan_ratio": nan_ratio,
        "unique": unique,
        "std": std,
        "keep": keep,
    }


def main() -> int:
    _enable_line_buffering()
    print(f"[CFG] feature_v2_file={FEATURE_V2_FILE}")
    print(f"[CFG] feature_v2_manifest={FEATURE_V2_MANIFEST}")
    print(f"[CFG] feature_v2_diag_csv={FEATURE_V2_DIAG_CSV}")
    print(f"[CFG] out_dir={OUT_DIR}")
    print(f"[CFG] target_col={TARGET_COL}")
    print(f"[CFG] min_non_na={MIN_NON_NA} max_nan_ratio={MAX_NAN_RATIO} min_unique={MIN_UNIQUE} max_base_features={MAX_BASE_FEATURES}")
    print(f"[CFG] include_z_features={int(INCLUDE_Z_FEATURES)} exclude_target_derived={int(EXCLUDE_TARGET_DERIVED)} exclude_market_wide_constants={int(EXCLUDE_MARKET_WIDE_CONSTANTS)}")
    print(f"[CFG] transforms raw={int(ENABLE_RAW)} signed_log={int(ENABLE_SIGNED_LOG)} ema3={int(ENABLE_EMA3)} lag1={int(ENABLE_LAG1)} tanh_z={int(ENABLE_TANH_Z)}")

    _must_exist(FEATURE_V2_FILE, "feature v2 file")
    feature_df = pd.read_parquet(FEATURE_V2_FILE)
    if feature_df.empty:
        raise RuntimeError("feature_matrix_v2.parquet is empty")
    feature_df["date"] = pd.to_datetime(feature_df["date"]).dt.normalize()
    feature_df = feature_df.sort_values(["symbol", "date"]).reset_index(drop=True)

    manifest_df = _load_feature_manifest()
    diag_df = _load_diag()
    safe_df, dropped_features_df = choose_safe_base_features(feature_df, manifest_df, diag_df)
    if safe_df.empty:
        raise RuntimeError("No safe fs2 base features remain after leakage and quality filtering")

    print(f"[FS2] safe_base_features={len(safe_df)} dropped_features={len(dropped_features_df)}")
    print("[FS2][SAFE_BASES]")
    print(safe_df[["feature", "cluster", "source", "formula"]].head(TOPK_PRINT).to_string(index=False))
    if len(dropped_features_df):
        print("[FS2][DROPPED_BASES]")
        show_cols = [c for c in ["feature", "cluster", "target_derived", "market_wide_constant", "non_na", "nan_ratio", "unique", "eligible"] if c in dropped_features_df.columns]
        print(dropped_features_df[show_cols].head(TOPK_PRINT).to_string(index=False))

    recipes = build_recipes(safe_df)
    if not recipes:
        raise RuntimeError("No alpha recipes built from safe fs2 features")
    print(f"[FS2] recipes={len(recipes)}")

    base_cols = _base_columns(feature_df)
    out_df = feature_df[base_cols].copy()
    kept_rows: List[Dict[str, object]] = []
    dropped_rows: List[Dict[str, object]] = []

    for idx, recipe in enumerate(recipes, start=1):
        alpha_series = materialize_alpha(feature_df, recipe)
        stats = validate_alpha_series(alpha_series)
        row = {
            "alpha": recipe.alpha,
            "base_feature": recipe.base_feature,
            "transform": recipe.transform,
            "cluster": recipe.cluster,
            "source": recipe.source,
            "formula": recipe.formula,
            **stats,
        }
        if int(stats["keep"]) == 1:
            out_df[recipe.alpha] = alpha_series
            kept_rows.append(row)
        else:
            dropped_rows.append(row)
        if idx % 25 == 0 or idx == len(recipes):
            print(f"[FS2][BUILD] {idx}/{len(recipes)} kept={len(kept_rows)} dropped={len(dropped_rows)}")

    if not kept_rows:
        raise RuntimeError("All fs2 alpha recipes were dropped by validation")

    manifest_out = pd.DataFrame(kept_rows).sort_values(["cluster", "base_feature", "transform", "alpha"]).reset_index(drop=True)
    dropped_out = pd.DataFrame(dropped_rows).sort_values(["cluster", "base_feature", "transform", "alpha"]).reset_index(drop=True) if dropped_rows else pd.DataFrame(columns=["alpha"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(OUT_PARQUET, index=False)
    manifest_out.to_csv(OUT_MANIFEST, index=False)
    dropped_out.to_csv(OUT_DROPPED, index=False)

    meta = {
        "rows": int(len(out_df)),
        "columns": int(len(out_df.columns)),
        "safe_base_features": int(len(safe_df)),
        "dropped_base_features": int(len(dropped_features_df)),
        "recipes_requested": int(len(recipes)),
        "alphas_kept": int(len(manifest_out)),
        "alphas_dropped": int(len(dropped_out)),
        "clusters": manifest_out.groupby("cluster", sort=False)["alpha"].count().astype(int).to_dict() if len(manifest_out) else {},
        "filters": {
            "include_z_features": int(INCLUDE_Z_FEATURES),
            "exclude_target_derived": int(EXCLUDE_TARGET_DERIVED),
            "exclude_market_wide_constants": int(EXCLUDE_MARKET_WIDE_CONSTANTS),
            "min_non_na": MIN_NON_NA,
            "max_nan_ratio": MAX_NAN_RATIO,
            "min_unique": MIN_UNIQUE,
        },
        "transforms": {
            "raw": int(ENABLE_RAW),
            "signed_log": int(ENABLE_SIGNED_LOG),
            "ema3": int(ENABLE_EMA3),
            "lag1": int(ENABLE_LAG1),
            "tanh_z": int(ENABLE_TANH_Z),
        },
        "inputs": {
            "feature_v2_file": str(FEATURE_V2_FILE),
            "feature_v2_manifest": str(FEATURE_V2_MANIFEST),
            "feature_v2_diag_csv": str(FEATURE_V2_DIAG_CSV),
        },
    }
    OUT_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[FS2][TOP_KEPT_ALPHAS]")
    print(manifest_out.head(TOPK_PRINT).to_string(index=False))
    print(f"[ARTIFACT] {OUT_PARQUET}")
    print(f"[ARTIFACT] {OUT_MANIFEST}")
    print(f"[ARTIFACT] {OUT_DROPPED}")
    print(f"[ARTIFACT] {OUT_META}")
    print("[FINAL] fs2 base alpha library generation complete")
    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 1
    _safe_exit(rc)
