from __future__ import annotations

import traceback
from pathlib import Path

import pandas as pd


DIAG_DIR = Path("data") / "diagnostics"
MODEL_NAME = "full_regime_stack_neutralized_sized_barbell_adaptive_exits"


def load_baseline_portfolios() -> pd.DataFrame:
    pattern = f"portfolio__{MODEL_NAME}__fold*.parquet"
    files = sorted(DIAG_DIR.glob(pattern))

    if not files:
        raise RuntimeError(f"no portfolio files found for pattern: {pattern}")

    frames = [pd.read_parquet(p) for p in files]
    out = pd.concat(frames, axis=0, ignore_index=True)
    return out


def build_alpha_age_surface(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "date",
        "symbol",
        "side",
        "weight",
        "target_fwd_ret_1d",
        "position_age",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"missing columns: {missing}")

    out = df.copy()
    out["pnl"] = pd.to_numeric(out["weight"], errors="coerce").fillna(0.0) * pd.to_numeric(
        out["target_fwd_ret_1d"], errors="coerce"
    ).fillna(0.0)

    grouped = (
        out.groupby("position_age", as_index=False)
        .agg(
            trades=("pnl", "count"),
            avg_pnl=("pnl", "mean"),
            sum_pnl=("pnl", "sum"),
            avg_weight=("weight", "mean"),
        )
        .sort_values("position_age")
        .reset_index(drop=True)
    )

    grouped["cum_pnl"] = grouped["sum_pnl"].cumsum()
    return grouped


def main() -> int:
    print("[ALPHA AGE] loading baseline portfolios")
    df = load_baseline_portfolios()

    surface = build_alpha_age_surface(df)

    print("\n[ALPHA AGE SURFACE]\n")
    print(surface.to_string(index=False))
    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except Exception:
        print("[ERROR]")
        print()
        traceback.print_exc()
        rc = 1
    finally:
        try:
            input("Press Enter to exit...")
        except EOFError:
            pass
    raise SystemExit(rc)