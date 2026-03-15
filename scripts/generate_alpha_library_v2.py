from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from python_edge.model.alpha_factory_core import ValidationConfig, derive_base_factory_inputs, generate_factory_alphas
from python_edge.model.alpha_factory_specs import ALL_RECIPES

FEATURE_FILE = ROOT / "data" / "features" / "feature_matrix_v1.parquet"
OUT_DIR = ROOT / "data" / "alpha_library_v2"
OUT_PARQUET = OUT_DIR / "alpha_library_v2.parquet"
OUT_META = OUT_DIR / "alpha_library_v2.meta.json"
OUT_MANIFEST = OUT_DIR / "alpha_library_v2.manifest.csv"
OUT_DROPPED = OUT_DIR / "alpha_library_v2.dropped.csv"


def main() -> None:
    if not FEATURE_FILE.exists():
        raise FileNotFoundError(f"Feature file not found: {FEATURE_FILE}")

    df = pd.read_parquet(FEATURE_FILE)
    if df.empty:
        raise RuntimeError("feature_matrix_v1.parquet is empty")

    base_df = derive_base_factory_inputs(df)
    result = generate_factory_alphas(base_df, recipes=ALL_RECIPES, cfg=ValidationConfig())
    final_df = result.frame
    alpha_cols = [c for c in final_df.columns if c.startswith("alpha_")]
    if not alpha_cols:
        raise RuntimeError("No alpha columns generated")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(OUT_PARQUET, index=False)
    result.manifest.to_csv(OUT_MANIFEST, index=False)
    result.dropped.to_csv(OUT_DROPPED, index=False)

    meta = {
        "rows": int(len(final_df)),
        "alpha_count": int(len(alpha_cols)),
        "recipe_count": int(len(ALL_RECIPES)),
        "kept_count": int(len(result.manifest)),
        "dropped_count": int(len(result.dropped)),
        "alpha_cols": alpha_cols,
        "waves": result.manifest["wave"].value_counts().to_dict() if len(result.manifest) else {},
        "families": result.manifest["family"].value_counts().to_dict() if len(result.manifest) else {},
    }
    OUT_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ALPHA_FACTORY] rows={len(final_df)}")
    print(f"[ALPHA_FACTORY] recipe_count={len(ALL_RECIPES)} kept={len(result.manifest)} dropped={len(result.dropped)}")
    print(f"[ALPHA_FACTORY] alpha_count={len(alpha_cols)}")
    print(f"[ARTIFACT] {OUT_PARQUET}")
    print(f"[ARTIFACT] {OUT_META}")
    print(f"[ARTIFACT] {OUT_MANIFEST}")
    print(f"[ARTIFACT] {OUT_DROPPED}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)