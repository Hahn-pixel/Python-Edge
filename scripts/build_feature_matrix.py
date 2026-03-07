from __future__ import annotations

import json
import traceback
from pathlib import Path

from python_edge.features.build_feature_matrix import build_feature_matrix


DATASET_ROOT = Path("D:/massive_dataset")
START = "2023-01-01"
END = "2026-02-28"
OUT_DIR = Path("data") / "features"
OUT_PARQUET = OUT_DIR / "feature_matrix_v1.parquet"
OUT_SUMMARY_JSON = OUT_DIR / "feature_matrix_v1_summary.json"


def main() -> int:
    print(f"[CFG] dataset_root={DATASET_ROOT}")
    print(f"[CFG] start={START} end={END}")
    print(f"[CFG] out_parquet={OUT_PARQUET}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = build_feature_matrix(
        dataset_root=DATASET_ROOT,
        start=START,
        end=END,
        symbols=None,
    )
    if df.empty:
        raise RuntimeError("Feature matrix is empty")

    df.to_parquet(OUT_PARQUET, index=False)
    print(f"[SAVE] parquet={OUT_PARQUET} rows={len(df)}")

    summary = {
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "symbols": int(df["symbol"].nunique()) if "symbol" in df.columns else 0,
        "dates": int(df["date"].nunique()) if "date" in df.columns else 0,
        "min_date": str(df["date"].min()) if "date" in df.columns and len(df) > 0 else None,
        "max_date": str(df["date"].max()) if "date" in df.columns and len(df) > 0 else None,
        "columns": list(df.columns),
    }
    OUT_SUMMARY_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[SAVE] summary={OUT_SUMMARY_JSON}")
    print("[DONE] feature matrix build completed")
    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except Exception:
        print("[ERROR] Unhandled exception:")
        print()
        traceback.print_exc()
        rc = 1
    finally:
        try:
            input("Press Enter to exit...")
        except EOFError:
            pass
    raise SystemExit(rc)