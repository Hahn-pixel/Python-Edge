from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

FEATURE_COLS = [
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
    "mom_x_volume_shock",
    "intraday_rs_x_volume_shock",
    "mom_x_vol_compression",
    "mom_x_market_breadth",
    "intraday_pressure_x_volume_shock",
    "liq_rank_x_intraday_rs",
]

EXCLUDED_SYMBOLS = {"VIXCBOE"}


def _discover_symbols(dataset_root: Path) -> list[str]:
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {dataset_root}")

    out: list[str] = []
    for child in sorted(dataset_root.iterdir(), key=lambda p: p.name.upper()):
        if not child.is_dir():
            continue
        name = child.name.strip()
        if not name:
            continue
        if name.startswith("."):
            continue
        if name in EXCLUDED_SYMBOLS:
            continue
        out.append(name)
    return out



def _read_json_payload(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8-sig")
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected payload type in {path}: {type(payload).__name__}")
    return payload



def _load_aggs_for_symbol(dataset_root: Path, symbol: str, tf: str) -> pd.DataFrame:
    sym_dir = dataset_root / symbol
    if not sym_dir.exists() or not sym_dir.is_dir():
        raise FileNotFoundError(f"Symbol directory not found: {sym_dir}")

    tf_map = {
        "1D": "aggs_1d_*__FULL.json",
        "15m": "aggs_15m_*__FULL.json",
    }
    if tf not in tf_map:
        raise ValueError(f"Unsupported tf={tf!r}")

    files = sorted(sym_dir.glob(tf_map[tf]), key=lambda p: p.name.lower())
    if not files:
        raise FileNotFoundError(f"No files found for symbol={symbol} tf={tf}")

    frames: list[pd.DataFrame] = []
    rename_map = {
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
        "t": "timestamp",
        "n": "transactions",
        "vw": "vwap",
    }

    for path in files:
        payload = _read_json_payload(path)
        results = payload.get("results")
        if not isinstance(results, list) or len(results) == 0:
            continue
        df = pd.DataFrame(results).rename(columns=rename_map)
        required = ["timestamp", "open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise RuntimeError(f"Missing required fields in {path}: {missing}")
        keep_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        if "transactions" in df.columns:
            keep_cols.append("transactions")
        if "vwap" in df.columns:
            keep_cols.append("vwap")
        df = df[keep_cols].copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True, errors="coerce")
        if df["timestamp"].isna().any():
            raise RuntimeError(f"Bad timestamps in {path}")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        if "transactions" in df.columns:
            df["transactions"] = pd.to_numeric(df["transactions"], errors="coerce")
        if "vwap" in df.columns:
            df["vwap"] = pd.to_numeric(df["vwap"], errors="coerce")
        df["symbol"] = symbol
        df["tf"] = tf
        frames.append(df)

    if not frames:
        raise RuntimeError(f"No frames loaded for symbol={symbol} tf={tf}")

    out = pd.concat(frames, axis=0, ignore_index=True)
    out = out.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    out = out.set_index("timestamp")
    out.index.name = "timestamp"
    return out



def _prepare_daily_panel(df1d: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    out = df1d.copy()
    if start is not None:
        out = out.loc[out.index >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        out = out.loc[out.index <= pd.Timestamp(end, tz="UTC")]
    if out.empty:
        raise RuntimeError("Daily panel empty after date filter")

    out["date"] = out.index.tz_convert("UTC").normalize().tz_localize(None)
    out["session_date"] = out["date"]
    out["prev_close"] = out["close"].shift(1)
    out["gap_ret"] = out["open"] / out["prev_close"] - 1.0
    out["meta_price"] = pd.to_numeric(out["close"], errors="coerce")
    out["meta_dollar_volume"] = pd.to_numeric(out["close"], errors="coerce") * pd.to_numeric(out["volume"], errors="coerce")

    out["momentum_20d"] = out["close"] / out["close"].shift(20) - 1.0
    out["str_3d"] = -(out["close"] / out["close"].shift(3) - 1.0)

    overnight_ret = out["open"] / out["close"].shift(1) - 1.0
    out["overnight_ret_1d"] = overnight_ret
    out["overnight_drift_20d"] = overnight_ret.rolling(20, min_periods=20).mean()

    vol = pd.to_numeric(out["volume"], errors="coerce")
    out["volume_ma20"] = vol.rolling(20, min_periods=20).mean()
    out["volume_shock"] = vol / out["volume_ma20"]

    ret = out["close"].pct_change()
    out["ivol_20d"] = ret.rolling(20, min_periods=20).std()
    out["vol_5d"] = ret.rolling(5, min_periods=5).std()
    out["vol_20d"] = ret.rolling(20, min_periods=20).std()
    out["vol_compression"] = out["vol_5d"] / out["vol_20d"]

    out["target_fwd_ret_1d"] = out["close"].shift(-1) / out["close"] - 1.0
    out["target_fwd_ret_3d"] = out["close"].shift(-3) / out["close"] - 1.0
    out["target_fwd_ret_5d"] = out["close"].shift(-5) / out["close"] - 1.0
    return out



def _prepare_intraday_panel(df15: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    out = df15.copy()
    if start is not None:
        out = out.loc[out.index >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        out = out.loc[out.index <= pd.Timestamp(end, tz="UTC")]
    if out.empty:
        raise RuntimeError("Intraday panel empty after date filter")
    out["date"] = out.index.tz_convert("UTC").normalize().tz_localize(None)
    out["session_date"] = out["date"]
    return out



def _add_intraday_features(df1d: pd.DataFrame, df15: pd.DataFrame) -> pd.DataFrame:
    out = df1d.copy()
    x = df15.copy()
    x["bar_ret"] = x["close"] / x["open"] - 1.0
    rng = (x["high"] - x["low"]).replace(0.0, pd.NA)
    x["close_pos_in_bar"] = (x["close"] - x["low"]) / rng

    daily = x.groupby("session_date", as_index=False).agg(
        intraday_rs=("bar_ret", "sum"),
        intraday_pressure=("close_pos_in_bar", "last"),
    )
    out = out.reset_index().merge(daily, on="session_date", how="left").set_index("timestamp")
    return out



def _finalize_cross_section(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(["date", "symbol"], ascending=[True, True]).reset_index(drop=True)

    out["liq_rank"] = out.groupby("date")["meta_dollar_volume"].rank(method="average", pct=True)

    out = out.sort_values(["symbol", "date"], ascending=[True, True]).reset_index(drop=True)
    out["ret_1d_tmp"] = out.groupby("symbol")["close"].pct_change()
    breadth = out.groupby("date", as_index=False).agg(
        market_breadth=("ret_1d_tmp", lambda s: float((s > 0).mean()) if len(s.dropna()) > 0 else float("nan")),
        market_ret_mean=("ret_1d_tmp", "mean"),
    )
    out = out.merge(breadth, on="date", how="left")
    out = out.drop(columns=["ret_1d_tmp"])

    out["mom_x_volume_shock"] = out["momentum_20d"] * out["volume_shock"]
    out["intraday_rs_x_volume_shock"] = out["intraday_rs"] * out["volume_shock"]
    out["mom_x_vol_compression"] = out["momentum_20d"] * out["vol_compression"]
    out["mom_x_market_breadth"] = out["momentum_20d"] * out["market_breadth"]
    out["intraday_pressure_x_volume_shock"] = out["intraday_pressure"] * out["volume_shock"]
    out["liq_rank_x_intraday_rs"] = out["liq_rank"] * out["intraday_rs"]
    return out



def _print_diagnostics(df: pd.DataFrame) -> None:
    print(f"[FEATURE] rows={len(df)}")
    print(f"[FEATURE] cols={len(df.columns)}")
    if "symbol" in df.columns:
        print(f"[FEATURE] symbols={df['symbol'].nunique()}")
    if "date" in df.columns and len(df) > 0:
        print(f"[FEATURE] dates={df['date'].nunique()}")
        print(f"[FEATURE] min_date={df['date'].min()}")
        print(f"[FEATURE] max_date={df['date'].max()}")
    total = max(len(df), 1)
    for col in FEATURE_COLS:
        if col not in df.columns:
            print(f"[FEATURE][WARN] missing_feature_col={col}")
            continue
        non_na = int(df[col].notna().sum())
        cov = non_na / total
        print(f"[FEATURE][COVERAGE] {col} non_na={non_na} coverage={cov:.4f}")



def build_feature_matrix(
    dataset_root: Path,
    start: str | None = None,
    end: str | None = None,
    symbols: list[str] | None = None,
) -> pd.DataFrame:
    dataset_root = Path(dataset_root)
    if symbols is None:
        symbols = _discover_symbols(dataset_root)
    else:
        symbols = [str(s).strip() for s in symbols if str(s).strip() and str(s).strip() not in EXCLUDED_SYMBOLS]

    print(f"[UNIVERSE] root={dataset_root}")
    print(f"[UNIVERSE] symbols_total={len(symbols)}")

    frames: list[pd.DataFrame] = []
    errors: list[tuple[str, str]] = []

    for i, sym in enumerate(symbols, start=1):
        print(f"[BUILD] {i}/{len(symbols)} symbol={sym}")
        try:
            df1d_raw = _load_aggs_for_symbol(dataset_root, sym, tf="1D")
            df1d = _prepare_daily_panel(df1d_raw, start=start, end=end)

            try:
                df15_raw = _load_aggs_for_symbol(dataset_root, sym, tf="15m")
                df15 = _prepare_intraday_panel(df15_raw, start=start, end=end)
                df1d = _add_intraday_features(df1d, df15)
            except Exception as exc:
                print(f"[BUILD][WARN] symbol={sym} intraday_features_skipped err={exc!r}")
                df1d = df1d.reset_index()
                df1d["intraday_rs"] = pd.NA
                df1d["intraday_pressure"] = pd.NA
                df1d = df1d.set_index("timestamp")

            sdf = df1d.reset_index()
            keep_cols = [
                "timestamp", "date", "session_date", "symbol", "open", "high", "low", "close", "volume",
                "prev_close", "gap_ret", "meta_price", "meta_dollar_volume", "momentum_20d", "str_3d",
                "overnight_ret_1d", "overnight_drift_20d", "volume_ma20", "volume_shock", "ivol_20d",
                "vol_5d", "vol_20d", "vol_compression", "intraday_rs", "intraday_pressure",
                "target_fwd_ret_1d", "target_fwd_ret_3d", "target_fwd_ret_5d",
            ]
            keep_cols = [c for c in keep_cols if c in sdf.columns]
            sdf = sdf[keep_cols].copy()
            if sdf.empty:
                raise RuntimeError("symbol frame is empty after feature build")
            frames.append(sdf)
        except Exception as exc:
            errors.append((sym, repr(exc)))
            print(f"[BUILD][ERROR] symbol={sym} err={exc!r}")

    if not frames:
        raise RuntimeError(f"No symbol frames built successfully. errors={errors[:10]}")

    out = pd.concat(frames, axis=0, ignore_index=True)
    out = _finalize_cross_section(out)
    _print_diagnostics(out)

    if errors:
        print(f"[BUILD][WARN] failed_symbols={len(errors)}")
        for sym, err in errors[:20]:
            print(f"[BUILD][WARN] symbol={sym} err={err}")

    return out