from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from python_edge.model.alpha_factory_specs import ALL_RECIPES, BASE_SIGNALS, MODULATORS, AlphaRecipe

EPS = 1e-12


@dataclass(frozen=True)
class ValidationConfig:
    max_nan_ratio: float = 0.995
    min_non_na: int = 200
    min_unique: int = 5


@dataclass(frozen=True)
class FactoryBuildResult:
    frame: pd.DataFrame
    manifest: pd.DataFrame
    dropped: pd.DataFrame


# ----------------------------
# generic helpers
# ----------------------------

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


# ----------------------------
# base features
# ----------------------------

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


# ----------------------------
# alpha generation
# ----------------------------

def _transform_base(df: pd.DataFrame, base: pd.Series, transform: str) -> pd.Series:
    if transform == "raw":
        return _safe(base)
    if transform == "z":
        return _cs_z_from_series(df, base)
    if transform == "rank":
        return _rank_from_series(df, base)
    if transform == "tanh":
        return np.tanh(_safe(base))
    if transform == "clip3":
        return _safe(base).clip(-3.0, 3.0)
    if transform == "sign":
        return np.sign(_safe(base))
    if transform == "signed_square":
        x = _safe(base)
        return np.sign(x) * (x.abs() ** 2.0)
    if transform == "lag1":
        return _lag(df, base, 1)
    if transform == "lag2":
        return _lag(df, base, 2)
    if transform == "ema3":
        return _ema(df, base)
    raise RuntimeError(f"Unsupported transform: {transform}")


def _apply_regime(df: pd.DataFrame, base: pd.Series, mod: pd.Series, regime: str) -> pd.Series:
    mod_rank = _safe(mod).groupby(df["date"], sort=False).rank(method="average", pct=True)
    if regime == "none":
        return _safe(base)
    if regime == "hi":
        return _safe(base) * (mod_rank >= 0.70).astype("float64")
    if regime == "lo":
        return _safe(base) * (mod_rank <= 0.30).astype("float64")
    if regime == "z":
        return _safe(base) * _cs_z_from_series(df, mod)
    if regime == "rank":
        return _safe(base) * (_safe(mod_rank) - 0.5)
    raise RuntimeError(f"Unsupported regime mode: {regime}")


def build_alpha_matrix(df: pd.DataFrame, recipes: Sequence[AlphaRecipe] = ALL_RECIPES) -> pd.DataFrame:
    base_lookup = {b.name: b.source_col for b in BASE_SIGNALS}
    mod_lookup = {m.name: m.source_col for m in MODULATORS}
    series_map: Dict[str, pd.Series] = {}

    for recipe in recipes:
        if recipe.left not in base_lookup:
            raise RuntimeError(f"Unknown base signal: {recipe.left}")
        base_col = base_lookup[recipe.left]
        if base_col not in df.columns:
            raise RuntimeError(f"Missing base column: {base_col}")
        base = _transform_base(df, df[base_col], recipe.transform)

        if recipe.modulator is not None:
            if recipe.modulator not in mod_lookup:
                raise RuntimeError(f"Unknown modulator: {recipe.modulator}")
            mod_col = mod_lookup[recipe.modulator]
            if mod_col not in df.columns:
                raise RuntimeError(f"Missing modulator column: {mod_col}")
            base = _apply_regime(df, base, df[mod_col], recipe.regime)

        series_map[f"alpha_{recipe.name}"] = _safe(base)

    return pd.DataFrame(series_map, index=df.index)


# ----------------------------
# validation / manifest
# ----------------------------

def validate_alpha_matrix(alpha_df: pd.DataFrame, recipes: Sequence[AlphaRecipe], cfg: ValidationConfig | None = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cfg = cfg or ValidationConfig()
    recipe_map = {f"alpha_{r.name}": r for r in recipes}
    keep_cols: List[str] = []
    keep_rows: List[dict] = []
    drop_rows: List[dict] = []

    for col in alpha_df.columns:
        s = _safe(alpha_df[col])
        non_na = int(s.notna().sum())
        nan_ratio = float(s.isna().mean())
        unique = int(s.dropna().nunique())
        recipe = recipe_map[col]
        row = {
            "alpha": col,
            "family": recipe.family,
            "wave": recipe.wave,
            "left": recipe.left,
            "modulator": recipe.modulator,
            "transform": recipe.transform,
            "regime": recipe.regime,
            "lag": recipe.lag,
            "non_na": non_na,
            "nan_ratio": nan_ratio,
            "unique": unique,
        }
        reason = None
        if non_na < cfg.min_non_na:
            reason = "too_few_non_na"
        elif nan_ratio > cfg.max_nan_ratio:
            reason = "too_sparse"
        elif unique < cfg.min_unique:
            reason = "too_few_unique"

        if reason is None:
            keep_cols.append(col)
            keep_rows.append(row)
        else:
            row["drop_reason"] = reason
            drop_rows.append(row)

    manifest_df = pd.DataFrame(keep_rows).sort_values(["wave", "family", "alpha"]).reset_index(drop=True)
    dropped_df = pd.DataFrame(drop_rows).sort_values(["drop_reason", "alpha"]).reset_index(drop=True) if drop_rows else pd.DataFrame(columns=["alpha", "drop_reason"])
    return alpha_df[keep_cols].copy(), manifest_df, dropped_df


def generate_factory_alphas(df: pd.DataFrame, recipes: Sequence[AlphaRecipe] = ALL_RECIPES, cfg: ValidationConfig | None = None) -> FactoryBuildResult:
    alpha_df = build_alpha_matrix(df, recipes=recipes)
    alpha_df, manifest_df, dropped_df = validate_alpha_matrix(alpha_df, recipes=recipes, cfg=cfg)
    out = pd.concat([df.reset_index(drop=True), alpha_df.reset_index(drop=True)], axis=1)
    return FactoryBuildResult(frame=out, manifest=manifest_df, dropped=dropped_df)