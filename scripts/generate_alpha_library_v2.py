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
from python_edge.model.alpha_factory_core import derive_base_factory_inputs, generate_factory_alphas

FEATURE_FILE = ROOT / "data" / "features" / "feature_matrix_v1.parquet"
OUT_DIR = ROOT / "data" / "alpha_library_v2"
OUT_PARQUET = OUT_DIR / "alpha_library_v2.parquet"
OUT_META = OUT_DIR / "alpha_library_v2.meta.json"


def main() -> None:
    if not FEATURE_FILE.exists():
        raise FileNotFoundError(f"Feature file not found: {FEATURE_FILE}")

    df = pd.read_parquet(FEATURE_FILE)
    if df.empty:
        raise RuntimeError("feature_matrix_v1.parquet is empty")

    df = derive_base_factory_inputs(df)
    df = generate_factory_alphas(df)

    alpha_cols = [c for c in df.columns if c.startswith("alpha_")]
    if not alpha_cols:
        raise RuntimeError("No alpha columns generated")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)

    meta = {
        "rows": int(len(df)),
        "alpha_count": int(len(alpha_cols)),
        "alpha_cols": alpha_cols,
    }
    OUT_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ALPHA_FACTORY] rows={len(df)}")
    print(f"[ALPHA_FACTORY] alpha_count={len(alpha_cols)}")
    print(f"[ARTIFACT] {OUT_PARQUET}")
    print(f"[ARTIFACT] {OUT_META}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)