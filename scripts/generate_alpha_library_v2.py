from __future__ import annotations

import json
import os
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

from python_edge.model.alpha_factory_core import (
    SurvivorConfig,
    ValidationConfig,
    build_recipe_registry,
    derive_base_factory_inputs,
    generate_factory_alphas,
)
from python_edge.model.alpha_factory_specs import SEED_RECIPES

FEATURE_FILE = ROOT / "data" / "features" / "feature_matrix_v1.parquet"
OUT_DIR = ROOT / "data" / "alpha_library_v2"
OUT_PARQUET = OUT_DIR / "alpha_library_v2.parquet"
OUT_META = OUT_DIR / "alpha_library_v2.meta.json"
OUT_MANIFEST = OUT_DIR / "alpha_library_v2.manifest.csv"
OUT_DROPPED = OUT_DIR / "alpha_library_v2.dropped.csv"
DEFAULT_SURVIVOR_MANIFEST = OUT_MANIFEST


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if raw == "":
        return int(default)
    return int(raw)


def _env_csv(name: str) -> tuple[str, ...]:
    raw = os.environ.get(name, "").strip()
    if raw == "":
        return ()
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _build_survivor_cfg() -> SurvivorConfig:
    explicit_families = _env_csv("ALPHA_SURVIVOR_FAMILIES")
    manifest_env = os.environ.get("ALPHA_SURVIVOR_MANIFEST", "").strip()
    manifest_path = manifest_env if manifest_env else str(DEFAULT_SURVIVOR_MANIFEST)
    return SurvivorConfig(
        manifest_path=manifest_path,
        min_recipes_per_family=_env_int("ALPHA_SURVIVOR_MIN_RECIPES", 2),
        top_n_families=_env_int("ALPHA_SURVIVOR_TOP_N", 6),
        explicit_families=explicit_families,
    )


def main() -> None:
    if not FEATURE_FILE.exists():
        raise FileNotFoundError(f"Feature file not found: {FEATURE_FILE}")

    df = pd.read_parquet(FEATURE_FILE)
    if df.empty:
        raise RuntimeError("feature_matrix_v1.parquet is empty")

    survivor_cfg = _build_survivor_cfg()
    recipes, survivor_detail, survivor_source = build_recipe_registry(seed_recipes=SEED_RECIPES, survivor_cfg=survivor_cfg)

    base_df = derive_base_factory_inputs(df)
    result = generate_factory_alphas(base_df, recipes=recipes, cfg=ValidationConfig())
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
        "recipe_count": int(len(recipes)),
        "seed_recipe_count": int(len(SEED_RECIPES)),
        "survivor_recipe_count": int(max(0, len(recipes) - len(SEED_RECIPES))),
        "kept_count": int(len(result.manifest)),
        "dropped_count": int(len(result.dropped)),
        "alpha_cols": alpha_cols,
        "waves": result.manifest["wave"].value_counts().to_dict() if len(result.manifest) else {},
        "families": result.manifest["family"].value_counts().to_dict() if len(result.manifest) else {},
        "survivor_source": survivor_source,
        "survivor_families": survivor_detail.to_dict(orient="records"),
    }
    OUT_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ALPHA_FACTORY] rows={len(final_df)}")
    print(f"[ALPHA_FACTORY] recipe_count={len(recipes)} seed={len(SEED_RECIPES)} survivor={len(recipes) - len(SEED_RECIPES)}")
    print(f"[ALPHA_FACTORY] kept={len(result.manifest)} dropped={len(result.dropped)} alpha_count={len(alpha_cols)}")
    print(f"[ALPHA_FACTORY] survivor_source={survivor_source}")
    if len(survivor_detail):
        print("[ALPHA_FACTORY][SURVIVORS]")
        print(survivor_detail.to_string(index=False))
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