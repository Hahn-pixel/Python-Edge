from __future__ import annotations

import numpy as np
import pandas as pd

from python_edge.model.alpha_factory_specs import ALL_RECIPES, BASE_SIGNALS, MODULATORS

EPS = 1e-12


def _safe(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("float64")


def _cs_z_from_series(df: pd.DataFrame, s: pd.Series) -> pd.Series:
    x = _safe(s)
    grp = x.groupby(df["date"], sort=False)
    mean = grp.transform("mean")
    std = grp.transform("std").replace(0.0, np.nan)
    out = (x - mean) / (std + EPS)
    return out.replace([np.inf, -np.inf], np.nan)


def _rank_from_series(df: pd.DataFrame, s: pd.Series) -> pd.Series:
    return _safe(s).groupby(df["date"], sort=False).rank(method="average", pct=True) - 0.5


def _lag(df: pd.DataFrame, s: pd.Series, k: int) -> pd.Series:
    return _safe(s).groupby(df["symbol"], sort=False).shift(k)


def _ema(df: pd.DataFrame, s: pd.Series) -> pd.Series:
    return _safe(s).groupby(df["symbol"], sort=False).transform(lambda x: x.ewm(span=3, adjust=False).mean())


def derive_base_factory_inputs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    required = ["date", "symbol", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise RuntimeError(f"derive_base_factory_inputs: missing columns: {missing}")

    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out = out.sort_values(["symbol", "date"]).reset_index(drop=True)

    prev_close = _safe(out.groupby("symbol", sort=False)["close"].shift(1))
    out["ret1"] = _safe(out.groupby("symbol", sort=False)["close"].pct_change())
    out["mom3_raw"] = _safe(out.groupby("symbol", sort=False)["close"].pct_change(3))
    out["momentum_20d"] = _safe(out.groupby("symbol", sort=False)["close"].pct_change(20))
    out["str_3d"] = -_safe(out.groupby("symbol", sort=False)["close"].pct_change(3))
    out["gap_ret"] = (_safe(out["open"]) / (prev_close + EPS)) - 1.0
    out["oc_body_pct"] = (_safe(out["close"]) - _safe(out["open"])) / (_safe(out["open"]) + EPS)
    out["dollar_vol"] = _safe(out["close"]) * _safe(out["volume"])
    out["liq"] = np.log1p(_safe(out["dollar_vol"]).clip(lower=0.0))

    vol_mean_20 = out.groupby("symbol", sort=False)["volume"].rolling(20, min_periods=10).mean().reset_index(level=0, drop=True)
    out["rel_volume_20"] = _safe(out["volume"]) / (_safe(vol_mean_20) + EPS)
    out["volume_shock"] = _safe(out["rel_volume_20"])

    hl_range_pct = (_safe(out["high"]) - _safe(out["low"])) / (_safe(out["close"]) + EPS)
    range5 = hl_range_pct.groupby(out["symbol"], sort=False).rolling(5, min_periods=3).mean().reset_index(level=0, drop=True)
    range20 = hl_range_pct.groupby(out["symbol"], sort=False).rolling(20, min_periods=10).mean().reset_index(level=0, drop=True)
    out["range_comp_5_20"] = _safe(range5) / (_safe(range20) + EPS)

    out["market_breadth"] = out.groupby("date", sort=False)["ret1"].transform(lambda s: pd.to_numeric(s, errors="coerce").gt(0).mean())
    out["overnight_drift_20d"] = _safe(out["gap_ret"]).groupby(out["symbol"], sort=False).rolling(20, min_periods=10).mean().reset_index(level=0, drop=True)
    out["ivol_20d"] = _safe(out["ret1"]).groupby(out["symbol"], sort=False).rolling(20, min_periods=10).std().reset_index(level=0, drop=True)
    rv10 = _safe(out["ret1"]).groupby(out["symbol"], sort=False).rolling(10, min_periods=5).std().reset_index(level=0, drop=True)
    atr14 = _safe(out["ret1"]).abs().groupby(out["symbol"], sort=False).rolling(14, min_periods=7).mean().reset_index(level=0, drop=True)
    out["vol_compression"] = rv10 / (atr14 + EPS)

    out["intraday_rs"] = -_cs_z_from_series(out, out["ret1"])
    out["intraday_pressure"] = _cs_z_from_series(out, out["oc_body_pct"])

    out["rev1_base"] = -_cs_z_from_series(out, out["ret1"])
    out["gap_base"] = _cs_z_from_series(out, out["gap_ret"])
    out["pressure_base"] = _cs_z_from_series(out, out["oc_body_pct"])
    out["mom3_base"] = _cs_z_from_series(out, out["mom3_raw"])

    out["meta_price"] = _safe(out["close"])
    out["meta_dollar_volume"] = _safe(out["dollar_vol"])
    return out.sort_values(["date", "symbol"]).reset_index(drop=True)


def generate_factory_alphas(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    base_lookup = {b.name: b.source_col for b in BASE_SIGNALS}
    mod_lookup = {m.name: m.source_col for m in MODULATORS}

    for r in ALL_RECIPES:
        if r.left not in base_lookup:
            raise RuntimeError(f"Unknown base signal: {r.left}")
        base_col = base_lookup[r.left]
        if base_col not in out.columns:
            raise RuntimeError(f"Missing base column: {base_col}")
        base = _safe(out[base_col])

        if r.transform == "z":
            base = _cs_z_from_series(out, base)
        elif r.transform == "rank":
            base = _rank_from_series(out, base)
        elif r.transform == "tanh":
            base = np.tanh(base)
        elif r.transform == "clip3":
            base = base.clip(-3.0, 3.0)
        elif r.transform == "sign":
            base = np.sign(base)
        elif r.transform == "signed_square":
            base = np.sign(base) * (base.abs() ** 2.0)
        elif r.transform == "lag1":
            base = _lag(out, base, 1)
        elif r.transform == "lag2":
            base = _lag(out, base, 2)
        elif r.transform == "ema3":
            base = _ema(out, base)
        elif r.transform != "raw":
            raise RuntimeError(f"Unsupported transform: {r.transform}")

        if r.modulator is not None:
            if r.modulator not in mod_lookup:
                raise RuntimeError(f"Unknown modulator: {r.modulator}")
            mod_col = mod_lookup[r.modulator]
            if mod_col not in out.columns:
                raise RuntimeError(f"Missing modulator column: {mod_col}")
            mod = _safe(out[mod_col])
            mod_rank = mod.groupby(out["date"], sort=False).rank(method="average", pct=True)
            if r.regime == "hi":
                base = base * (mod_rank >= 0.70).astype("float64")
            elif r.regime == "lo":
                base = base * (mod_rank <= 0.30).astype("float64")
            elif r.regime == "z":
                base = base * _cs_z_from_series(out, mod)
            elif r.regime == "rank":
                base = base * (_safe(mod_rank) - 0.5)
            elif r.regime != "none":
                raise RuntimeError(f"Unsupported regime mode: {r.regime}")

        out[f"alpha_{r.name}"] = _safe(base)

    return out