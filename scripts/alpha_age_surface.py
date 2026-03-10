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


def build_surface(df: pd.DataFrame, label: str) -> pd.DataFrame:

    out = df.copy()

    out["pnl"] = (
        pd.to_numeric(out["weight"], errors="coerce").fillna(0.0)
        * pd.to_numeric(out["target_fwd_ret_1d"], errors="coerce").fillna(0.0)
    )

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
    grouped["surface"] = label

    return grouped


def main() -> int:
    print("[ALPHA AGE] loading baseline portfolios")
    df = load_baseline_portfolios()

    long_df = df[df["side"] > 0]
    short_df = df[df["side"] < 0]

    surf_long = build_surface(long_df, "LONG")
    surf_short = build_surface(short_df, "SHORT")
    surf_all = build_surface(df, "COMBINED")

    surface = pd.concat(
        [surf_long, surf_short, surf_all],
        axis=0,
        ignore_index=True,
    )

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