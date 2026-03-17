from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from python_edge.model.alpha_factory_specs import (
    ALL_RECIPES,
    BASE_SIGNAL_REGISTRY,
    DEFAULT_SURVIVOR_MIN_RECIPES,
    DEFAULT_SURVIVOR_TOP_N,
    MODULATOR_REGISTRY,
    RECENT_SURVIVOR_DEFAULT_PRIORITY,
    SEED_RECIPES,
    SURVIVOR_DEFAULT_PRIORITY,
    AlphaRecipe,
    SurvivorExpansionPlan,
    survivor_registry_bundle,
)

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


@dataclass(frozen=True)
class SurvivorConfig:
    manifest_path: str | None = None
    min_recipes_per_family: int = DEFAULT_SURVIVOR_MIN_RECIPES
    top_n_families: int = DEFAULT_SURVIVOR_TOP_N
    explicit_families: tuple[str, ...] = ()


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


def _signed_log(s: pd.Series) -> pd.Series:
    x = _safe(s)
    return np.sign(x) * np.log1p(x.abs())


def _sqrt_signed(s: pd.Series) -> pd.Series:
    x = _safe(s)
    return np.sign(x) * np.sqrt(x.abs())


def _cube(s: pd.Series) -> pd.Series:
    x = _safe(s)
    return x ** 3.0


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
# registry helpers
# ----------------------------

def build_registry_lookup() -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    base_lookup = {b.name: b.source_col for b in BASE_SIGNAL_REGISTRY}
    mod_lookup = {m.name: m.source_col for m in MODULATOR_REGISTRY}
    family_lookup = {b.name: b.family for b in BASE_SIGNAL_REGISTRY}
    return base_lookup, mod_lookup, family_lookup


def load_survivor_families(cfg: SurvivorConfig | None = None) -> tuple[list[str], pd.DataFrame, str]:
    cfg = cfg or SurvivorConfig()
    if cfg.explicit_families:
        ordered = [f for f in SURVIVOR_DEFAULT_PRIORITY if f in set(cfg.explicit_families)]
        extras = [f for f in cfg.explicit_families if f not in ordered]
        return ordered + extras, pd.DataFrame(columns=["family", "recipes"]), "explicit"

    manifest_path = Path(cfg.manifest_path) if cfg.manifest_path else None
    if manifest_path and manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        if "family" not in manifest.columns:
            raise RuntimeError(f"Survivor manifest missing 'family' column: {manifest_path}")
        family_counts = manifest["family"].value_counts().rename_axis("family").reset_index(name="recipes")
        family_counts = family_counts[family_counts["recipes"] >= int(cfg.min_recipes_per_family)].copy()
        if family_counts.empty:
            return list(SURVIVOR_DEFAULT_PRIORITY[: cfg.top_n_families]), family_counts, "default_priority"
        priority_rank = {fam: idx for idx, fam in enumerate(SURVIVOR_DEFAULT_PRIORITY)}
        family_counts["priority_rank"] = family_counts["family"].map(lambda x: priority_rank.get(str(x), 999))
        family_counts = family_counts.sort_values(["recipes", "priority_rank", "family"], ascending=[False, True, True]).reset_index(drop=True)
        selected = family_counts.head(int(cfg.top_n_families)).copy()
        return selected["family"].astype(str).tolist(), selected[["family", "recipes"]].copy(), str(manifest_path)

    fallback = list(SURVIVOR_DEFAULT_PRIORITY[: cfg.top_n_families])
    return fallback, pd.DataFrame(columns=["family", "recipes"]), "default_priority"


def _family_seed_map(seed_recipes: Sequence[AlphaRecipe]) -> dict[str, list[AlphaRecipe]]:
    out: dict[str, list[AlphaRecipe]] = {}
    for recipe in seed_recipes:
        out.setdefault(recipe.family, []).append(recipe)
    return out


def _unique_ordered(values: Iterable[str | None]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value is None:
            continue
        v = str(value).strip()
        if not v or v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _survivor_recipe_name(left: str, transform: str, interaction: str, modulator: str | None, regime: str, lag: int) -> str:
    parts = [left, transform, interaction]
    if modulator:
        parts.append(modulator)
    parts.append(regime)
    if lag > 0:
        parts.append(f"lag{lag}")
    return "_".join(parts)


def _recent_family_rows(manifest_path: str | None) -> pd.DataFrame:
    if not manifest_path:
        return pd.DataFrame(columns=["alpha", "family", "selector_score", "shortlist_rank"])
    path = Path(manifest_path)
    if not path.exists():
        return pd.DataFrame(columns=["alpha", "family", "selector_score", "shortlist_rank"])
    manifest = pd.read_csv(path)
    if manifest.empty:
        return pd.DataFrame(columns=["alpha", "family", "selector_score", "shortlist_rank"])
    alpha_col = "alpha" if "alpha" in manifest.columns else None
    family_col = "family" if "family" in manifest.columns else None
    score_col = "selector_score" if "selector_score" in manifest.columns else None
    rank_col = "shortlist_rank" if "shortlist_rank" in manifest.columns else None
    if alpha_col is None or family_col is None:
        return pd.DataFrame(columns=["alpha", "family", "selector_score", "shortlist_rank"])
    out = manifest[[alpha_col, family_col]].copy()
    out.columns = ["alpha", "family"]
    out["selector_score"] = pd.to_numeric(manifest[score_col], errors="coerce").fillna(0.0) if score_col else 0.0
    out["shortlist_rank"] = pd.to_numeric(manifest[rank_col], errors="coerce").fillna(1_000_000.0) if rank_col else 1_000_000.0
    priority = {fam: idx for idx, fam in enumerate(RECENT_SURVIVOR_DEFAULT_PRIORITY)}
    out["priority_rank"] = out["family"].astype(str).map(lambda x: priority.get(x, 999))
    out = out.sort_values(["priority_rank", "selector_score", "shortlist_rank", "alpha"], ascending=[True, False, True, True]).reset_index(drop=True)
    return out[["alpha", "family", "selector_score", "shortlist_rank"]].copy()


def _recent_survivor_families(cfg: SurvivorConfig | None = None) -> tuple[list[str], pd.DataFrame, str]:
    cfg = cfg or SurvivorConfig()
    manifest_path = cfg.manifest_path if cfg.manifest_path else None
    recent_rows = _recent_family_rows(manifest_path)
    if recent_rows.empty:
        fallback = list(RECENT_SURVIVOR_DEFAULT_PRIORITY[: max(1, cfg.top_n_families)])
        return fallback, pd.DataFrame(columns=["family", "recipes"]), "recent_default_priority"
    grouped = recent_rows.groupby("family", sort=False).size().rename("recipes").reset_index()
    priority = {fam: idx for idx, fam in enumerate(RECENT_SURVIVOR_DEFAULT_PRIORITY)}
    grouped["priority_rank"] = grouped["family"].astype(str).map(lambda x: priority.get(x, 999))
    grouped = grouped.sort_values(["priority_rank", "recipes", "family"], ascending=[True, False, True]).reset_index(drop=True)
    selected = grouped.head(max(1, int(cfg.top_n_families))).copy()
    return selected["family"].astype(str).tolist(), selected[["family", "recipes"]].copy(), "recent_manifest"


def expand_survivor_recipes(
    seed_recipes: Sequence[AlphaRecipe] = SEED_RECIPES,
    survivor_cfg: SurvivorConfig | None = None,
    expansion_plans: Sequence[SurvivorExpansionPlan] | None = None,
) -> tuple[tuple[AlphaRecipe, ...], pd.DataFrame, str]:
    survivor_families, survivor_detail, survivor_source = load_survivor_families(survivor_cfg)
    family_seed_map = _family_seed_map(seed_recipes)
    out: list[AlphaRecipe] = []
    seen: set[str] = set()
    expansion_plans = tuple(expansion_plans or survivor_registry_bundle(include_wave7=False))

    for family in survivor_families:
        family_seeds = family_seed_map.get(family, [])
        if not family_seeds:
            continue
        left = family_seeds[0].left
        inherited_modulators = _unique_ordered(r.modulator for r in family_seeds if r.modulator)
        inherited_regimes = _unique_ordered(r.regime for r in family_seeds if r.regime != "none") or ["z", "rank"]
        family_parent_names = tuple(sorted({r.name for r in family_seeds}))

        for plan in expansion_plans:
            candidate_modulators = list(plan.modulators) if plan.modulators else []
            if plan.inherit_modulators:
                candidate_modulators.extend(inherited_modulators)
            candidate_modulators = _unique_ordered(candidate_modulators)[: max(1, int(plan.max_modulators_per_family))]

            candidate_regimes = list(plan.regimes)
            if plan.inherit_regimes:
                candidate_regimes = _unique_ordered(inherited_regimes + candidate_regimes)

            for transform in plan.transforms:
                inferred_lag = 0
                if transform == "lag1":
                    inferred_lag = 1
                elif transform == "lag2":
                    inferred_lag = 2
                for interaction in plan.interactions:
                    for modulator in candidate_modulators:
                        for regime in candidate_regimes:
                            name = _survivor_recipe_name(left, transform, interaction, modulator, regime, inferred_lag)
                            if name in seen:
                                continue
                            seen.add(name)
                            out.append(
                                AlphaRecipe(
                                    name=name,
                                    left=left,
                                    modulator=modulator,
                                    transform=transform,
                                    regime=regime,
                                    interaction=interaction,
                                    lag=inferred_lag,
                                    family=family,
                                    wave=plan.wave,
                                    parents=family_parent_names,
                                    source=f"survivor:{survivor_source}",
                                )
                            )
    return tuple(out), survivor_detail, survivor_source


def expand_recent_survivor_recipes(
    seed_recipes: Sequence[AlphaRecipe] = SEED_RECIPES,
    survivor_cfg: SurvivorConfig | None = None,
) -> tuple[tuple[AlphaRecipe, ...], pd.DataFrame, str]:
    recent_families, recent_detail, recent_source = _recent_survivor_families(survivor_cfg)
    family_seed_map = _family_seed_map(seed_recipes)
    out: list[AlphaRecipe] = []
    seen: set[str] = set()
    for family in recent_families:
        family_seeds = family_seed_map.get(family, [])
        if not family_seeds:
            continue
        left = family_seeds[0].left
        inherited_modulators = _unique_ordered(r.modulator for r in family_seeds if r.modulator)
        inherited_regimes = _unique_ordered(r.regime for r in family_seeds if r.regime != "none") or ["z", "rank"]
        family_parent_names = tuple(sorted({r.name for r in family_seeds}))
        for plan in survivor_registry_bundle(include_wave7=True):
            if plan.wave != "wave7":
                continue
            candidate_modulators = list(plan.modulators) if plan.modulators else []
            if plan.inherit_modulators:
                candidate_modulators.extend(inherited_modulators)
            candidate_modulators = _unique_ordered(candidate_modulators)[: max(1, int(plan.max_modulators_per_family))]
            candidate_regimes = list(plan.regimes)
            if plan.inherit_regimes:
                candidate_regimes = _unique_ordered(inherited_regimes + candidate_regimes)
            for transform in plan.transforms:
                inferred_lag = 0
                if transform == "lag1":
                    inferred_lag = 1
                elif transform == "lag2":
                    inferred_lag = 2
                for interaction in plan.interactions:
                    for modulator in candidate_modulators:
                        for regime in candidate_regimes:
                            name = _survivor_recipe_name(left, transform, interaction, modulator, regime, inferred_lag)
                            if name in seen:
                                continue
                            seen.add(name)
                            out.append(
                                AlphaRecipe(
                                    name=name,
                                    left=left,
                                    modulator=modulator,
                                    transform=transform,
                                    regime=regime,
                                    interaction=interaction,
                                    lag=inferred_lag,
                                    family=family,
                                    wave="wave7",
                                    parents=family_parent_names,
                                    source=f"recent_survivor:{recent_source}",
                                )
                            )
    return tuple(out), recent_detail, recent_source


def build_recipe_registry(
    seed_recipes: Sequence[AlphaRecipe] = SEED_RECIPES,
    survivor_cfg: SurvivorConfig | None = None,
) -> tuple[tuple[AlphaRecipe, ...], pd.DataFrame, str]:
    survivor_recipes, survivor_detail, survivor_source = expand_survivor_recipes(seed_recipes=seed_recipes, survivor_cfg=survivor_cfg)
    recent_recipes, recent_detail, recent_source = expand_recent_survivor_recipes(seed_recipes=seed_recipes, survivor_cfg=survivor_cfg)
    recipes = tuple(seed_recipes) + tuple(survivor_recipes) + tuple(recent_recipes)

    detail_frames: list[pd.DataFrame] = []
    if len(survivor_detail):
        s = survivor_detail.copy()
        s["source"] = survivor_source
        detail_frames.append(s)
    if len(recent_detail):
        r = recent_detail.copy()
        r["source"] = recent_source
        detail_frames.append(r)
    detail = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame(columns=["family", "recipes", "source"])
    source = "+".join(part for part in [survivor_source, recent_source] if part)
    return recipes, detail, source


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
    if transform == "signed_log":
        return _signed_log(base)
    if transform == "sqrt_signed":
        return _sqrt_signed(base)
    if transform == "cube":
        return _cube(base)
    if transform == "tanh_z":
        return np.tanh(_cs_z_from_series(df, base))
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


def _apply_interaction(df: pd.DataFrame, base: pd.Series, mod: pd.Series, interaction: str, regime: str) -> pd.Series:
    if interaction == "regime":
        return _apply_regime(df, base, mod, regime)
    if interaction == "raw_mul":
        return _safe(base) * _safe(mod)
    if interaction == "z_mul":
        return _safe(base) * _cs_z_from_series(df, mod)
    if interaction == "rank_mul":
        return _safe(base) * _rank_from_series(df, mod)
    raise RuntimeError(f"Unsupported interaction mode: {interaction}")


def build_alpha_matrix(df: pd.DataFrame, recipes: Sequence[AlphaRecipe] = ALL_RECIPES) -> pd.DataFrame:
    base_lookup, mod_lookup, _ = build_registry_lookup()
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
            base = _apply_interaction(df, base, df[mod_col], recipe.interaction, recipe.regime)

        series_map[f"alpha_{recipe.name}"] = _safe(base)

    return pd.DataFrame(series_map, index=df.index)


# ----------------------------
# validation / manifest
# ----------------------------

def validate_alpha_matrix(alpha_df: pd.DataFrame, recipes: Sequence[AlphaRecipe], cfg: ValidationConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
            "interaction": recipe.interaction,
            "lag": recipe.lag,
            "source": recipe.source,
            "parents": "|".join(recipe.parents),
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

    manifest_df = pd.DataFrame(keep_rows)
    if len(manifest_df):
        manifest_df = manifest_df.sort_values(["wave", "family", "alpha"]).reset_index(drop=True)
    dropped_df = pd.DataFrame(drop_rows)
    if len(dropped_df):
        dropped_df = dropped_df.sort_values(["drop_reason", "alpha"]).reset_index(drop=True)
    else:
        dropped_df = pd.DataFrame(columns=["alpha", "drop_reason"])
    return alpha_df[keep_cols].copy(), manifest_df, dropped_df


def generate_factory_alphas(df: pd.DataFrame, recipes: Sequence[AlphaRecipe] = ALL_RECIPES, cfg: ValidationConfig | None = None) -> FactoryBuildResult:
    alpha_df = build_alpha_matrix(df, recipes=recipes)
    alpha_df, manifest_df, dropped_df = validate_alpha_matrix(alpha_df, recipes=recipes, cfg=cfg)
    out = pd.concat([df.reset_index(drop=True), alpha_df.reset_index(drop=True)], axis=1)
    return FactoryBuildResult(frame=out, manifest=manifest_df, dropped=dropped_df)


__all__ = [
    "FactoryBuildResult",
    "SurvivorConfig",
    "ValidationConfig",
    "build_alpha_matrix",
    "build_recipe_registry",
    "build_registry_lookup",
    "derive_base_factory_inputs",
    "expand_recent_survivor_recipes",
    "expand_survivor_recipes",
    "generate_factory_alphas",
    "load_survivor_families",
    "validate_alpha_matrix",
]