from __future__ import annotations

from pathlib import Path

import pandas as pd

from python_edge.data.load_local_aggs import load_symbol_1d, load_symbol_15m
from python_edge.data.schema_checks import run_all_schema_checks
from python_edge.data.sessionize import prepare_1d_panel, prepare_15m_panel
from python_edge.data.universe import discover_symbols, filter_symbols, print_universe_diagnostics
from python_edge.features.add_interactions import INTERACTION_COLS, add_interactions
from python_edge.features.add_intraday_pressure import add_intraday_pressure
from python_edge.features.add_intraday_rs import add_intraday_rs
from python_edge.features.add_ivol_20d import add_ivol_20d
from python_edge.features.add_liq_rank import add_liq_rank
from python_edge.features.add_market_breadth import add_market_breadth
from python_edge.features.add_momentum_20d import add_momentum_20d
from python_edge.features.add_overnight_drift_20d import add_overnight_drift_20d
from python_edge.features.add_str_3d import add_str_3d
from python_edge.features.add_vol_compression import add_vol_compression
from python_edge.features.add_volume_shock import add_volume_shock
from python_edge.features.diagnostics import print_feature_coverage, print_feature_matrix_summary, print_feature_warnings
from python_edge.model.targets import add_all_forward_return_targets


BASE_FEATURE_COLS = [
    "momentum_20d",
    "str_3d",
    "overnight_drift_20d",
    "volume_shock",
    "ivol_20d",
    "vol_compression",
    "intraday_rs",
    "intraday_pressure",
    "liq_rank",
    "market_breadth",
]

FEATURE_COLS = BASE_FEATURE_COLS + INTERACTION_COLS



def _build_symbol_frame(dataset_root: Path, symbol: str, start: str | None, end: str | None) -> pd.DataFrame:
    df1d = load_symbol_1d(dataset_root=dataset_root, symbol=symbol, start=start, end=end)
    run_all_schema_checks(df1d, tf_name=f"{symbol}:1D")
    df1d = prepare_1d_panel(df1d)
    df1d = add_momentum_20d(df1d)
    df1d = add_str_3d(df1d)
    df1d = add_overnight_drift_20d(df1d)
    df1d = add_volume_shock(df1d)
    df1d = add_ivol_20d(df1d, market_proxy=None)
    df1d = add_vol_compression(df1d)

    try:
        df15 = load_symbol_15m(dataset_root=dataset_root, symbol=symbol, start=start, end=end)
        run_all_schema_checks(df15, tf_name=f"{symbol}:15m")
        df15 = prepare_15m_panel(df15)
        df1d = add_intraday_rs(df15, df1d)
        df1d = add_intraday_pressure(df15, df1d)
    except Exception as exc:
        print(f"[BUILD][WARN] symbol={symbol} intraday_features_skipped err={exc!r}")
        df1d["intraday_rs"] = pd.NA
        df1d["intraday_pressure"] = pd.NA

    df1d = add_all_forward_return_targets(df1d)
    df1d = df1d.reset_index().rename(columns={"index": "timestamp"})

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
        "vol_5d",
        "vol_20d",
        "vol_compression",
        "intraday_rs",
        "intraday_pressure",
        "target_fwd_ret_1d",
        "target_fwd_ret_3d",
        "target_fwd_ret_5d",
    ]
    keep_cols = [c for c in keep_cols if c in df1d.columns]
    return df1d[keep_cols].copy()



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

    out = add_liq_rank(out)
    out = add_market_breadth(out)
    out = add_interactions(out)

    print_feature_matrix_summary(out)
    print_feature_coverage(out, FEATURE_COLS)
    print_feature_warnings(out, FEATURE_COLS)

    if errors:
        print(f"[BUILD][WARN] failed_symbols={len(errors)}")
        for sym, err in errors[:20]:
            print(f"[BUILD][WARN] symbol={sym} err={err}")

    return out