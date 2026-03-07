from __future__ import annotations

from pathlib import Path

import pandas as pd

from python_edge.data.load_local_aggs import load_symbol_1d
from python_edge.data.schema_checks import run_all_schema_checks
from python_edge.data.sessionize import prepare_1d_panel
from python_edge.data.universe import discover_symbols, filter_symbols, print_universe_diagnostics
from python_edge.features.add_ivol_20d import add_ivol_20d
from python_edge.features.add_momentum_20d import add_momentum_20d
from python_edge.features.add_overnight_drift_20d import add_overnight_drift_20d
from python_edge.features.add_str_3d import add_str_3d
from python_edge.features.add_volume_shock import add_volume_shock



def _add_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["target_fwd_ret_1d"] = out["close"].shift(-1) / out["close"] - 1.0
    out["target_fwd_ret_3d"] = out["close"].shift(-3) / out["close"] - 1.0
    out["target_fwd_ret_5d"] = out["close"].shift(-5) / out["close"] - 1.0
    return out



def _build_symbol_frame(dataset_root: Path, symbol: str, start: str | None, end: str | None) -> pd.DataFrame:
    df = load_symbol_1d(dataset_root=dataset_root, symbol=symbol, start=start, end=end)
    run_all_schema_checks(df, tf_name=f"{symbol}:1D")
    df = prepare_1d_panel(df)
    df = add_momentum_20d(df)
    df = add_str_3d(df)
    df = add_overnight_drift_20d(df)
    df = add_volume_shock(df)
    df = add_ivol_20d(df, market_proxy=None)
    df = _add_targets(df)
    df = df.reset_index().rename(columns={"index": "timestamp"})
    keep_cols = [
        "timestamp",
        "date",
        "session_date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "prev_close",
        "gap_ret",
        "meta_price",
        "meta_dollar_volume",
        "momentum_20d",
        "str_3d",
        "overnight_ret_1d",
        "overnight_drift_20d",
        "volume_ma20",
        "volume_shock",
        "ivol_20d",
        "target_fwd_ret_1d",
        "target_fwd_ret_3d",
        "target_fwd_ret_5d",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols].copy()



def print_feature_matrix_summary(df: pd.DataFrame) -> None:
    print(f"[FEATURE] rows={len(df)}")
    print(f"[FEATURE] symbols={df['symbol'].nunique() if 'symbol' in df.columns else 0}")
    if "date" in df.columns:
        print(f"[FEATURE] dates={df['date'].nunique()}")
        print(f"[FEATURE] min_date={df['date'].min()}")
        print(f"[FEATURE] max_date={df['date'].max()}")



def print_feature_coverage(df: pd.DataFrame, feature_cols: list[str]) -> None:
    for col in feature_cols:
        if col not in df.columns:
            print(f"[FEATURE][WARN] missing_feature_col={col}")
            continue
        non_na = int(df[col].notna().sum())
        cov = non_na / max(len(df), 1)
        print(f"[FEATURE][COVERAGE] {col} non_na={non_na} coverage={cov:.4f}")



def build_feature_matrix(
    dataset_root: Path,
    start: str | None = None,
    end: str | None = None,
    symbols: list[str] | None = None,
) -> pd.DataFrame:
    dataset_root = Path(dataset_root)
    if symbols is None:
        symbols = filter_symbols(discover_symbols(dataset_root))
    else:
        symbols = filter_symbols(symbols)

    print_universe_diagnostics(dataset_root, symbols)

    frames: list[pd.DataFrame] = []
    errors: list[tuple[str, str]] = []
    for i, sym in enumerate(symbols, start=1):
        print(f"[BUILD] {i}/{len(symbols)} symbol={sym}")
        try:
            sdf = _build_symbol_frame(dataset_root, sym, start, end)
            if sdf.empty:
                raise RuntimeError("symbol frame is empty after feature build")
            frames.append(sdf)
        except Exception as exc:
            errors.append((sym, repr(exc)))
            print(f"[BUILD][ERROR] symbol={sym} err={exc!r}")

    if not frames:
        raise RuntimeError(f"No symbol frames built successfully. errors={errors[:10]}")

    out = pd.concat(frames, axis=0, ignore_index=True)
    out = out.sort_values(["date", "symbol"], ascending=[True, True]).reset_index(drop=True)

    feature_cols = [
        "momentum_20d",
        "str_3d",
        "overnight_drift_20d",
        "volume_shock",
        "ivol_20d",
    ]
    print_feature_matrix_summary(out)
    print_feature_coverage(out, feature_cols)

    if errors:
        print(f"[BUILD][WARN] failed_symbols={len(errors)}")
        for sym, err in errors[:20]:
            print(f"[BUILD][WARN] symbol={sym} err={err}")

    return out