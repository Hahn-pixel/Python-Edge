from __future__ import annotations

import traceback
import pandas as pd

from python_edge.model.cs_normalize import cs_zscore
from python_edge.model.ranker_linear import linear_rank_score
from python_edge.portfolio.construct import build_long_short_portfolio
from python_edge.wf.evaluate_ranker import evaluate_long_short


FEATURE_FILE = "data/features/feature_matrix_v1.parquet"


FEATURES = [
    "momentum_20d",
    "str_3d",
    "overnight_drift_20d",
    "volume_shock",
    "ivol_20d",
]


WEIGHTS = {
    "z_momentum_20d": 1.0,
    "z_str_3d": -0.5,
    "z_overnight_drift_20d": 0.5,
    "z_volume_shock": -0.2,
    "z_ivol_20d": -0.3,
}


def main():

    print("[LOAD] feature matrix")

    df = pd.read_parquet(FEATURE_FILE)

    df = cs_zscore(df, FEATURES)

    zcols = [f"z_{f}" for f in FEATURES]

    df = linear_rank_score(df, zcols, WEIGHTS)

    df = build_long_short_portfolio(df)

    res = evaluate_long_short(df)

    print("[RESULT]")

    print(res.tail())


if __name__ == "__main__":

    try:
        main()

    except Exception:

        print("[ERROR]")

        traceback.print_exc()

    finally:

        try:
            input("Press Enter to exit...")
        except EOFError:
            pass