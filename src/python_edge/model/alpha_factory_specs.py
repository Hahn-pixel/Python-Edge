from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TransformName = Literal[
    "raw",
    "z",
    "rank",
    "tanh",
    "clip3",
    "sign",
    "signed_square",
    "signed_log",
    "sqrt_signed",
    "cube",
    "tanh_z",
    "lag1",
    "lag2",
    "ema3",
]
RegimeMode = Literal["none", "hi", "lo", "z", "rank"]
InteractionMode = Literal["regime", "raw_mul", "z_mul", "rank_mul"]


@dataclass(frozen=True)
class BaseSignalSpec:
    name: str
    source_col: str
    family: str


@dataclass(frozen=True)
class ModulatorSpec:
    name: str
    source_col: str
    family: str


@dataclass(frozen=True)
class AlphaRecipe:
    name: str
    left: str
    modulator: str | None
    transform: TransformName
    regime: RegimeMode = "none"
    interaction: InteractionMode = "regime"
    lag: int = 0
    family: str = "generic"
    wave: str = "wave0"
    parents: tuple[str, ...] = ()
    source: str = "registry"


@dataclass(frozen=True)
class FamilySeed:
    family: str
    base_signal: str
    modulators: tuple[str, ...]


@dataclass(frozen=True)
class WaveTransformSeed:
    transform: TransformName
    regimes: tuple[RegimeMode, ...]
    interaction: InteractionMode = "regime"


@dataclass(frozen=True)
class WaveTemporalSeed:
    transform: TransformName
    regimes: tuple[RegimeMode, ...]
    interaction: InteractionMode = "regime"
    lag: int = 0


@dataclass(frozen=True)
class SurvivorExpansionPlan:
    wave: str
    transforms: tuple[TransformName, ...]
    regimes: tuple[RegimeMode, ...]
    interactions: tuple[InteractionMode, ...]
    modulators: tuple[str, ...] = ()
    inherit_modulators: bool = True
    inherit_regimes: bool = True
    max_modulators_per_family: int = 3


BASE_SIGNAL_REGISTRY: tuple[BaseSignalSpec, ...] = (
    BaseSignalSpec("rev1", "rev1_base", "rev1"),
    BaseSignalSpec("gap", "gap_base", "gap"),
    BaseSignalSpec("pressure", "pressure_base", "pressure"),
    BaseSignalSpec("mom3", "mom3_base", "mom3"),
    BaseSignalSpec("momentum_20d", "momentum_20d", "momentum_20d"),
    BaseSignalSpec("str_3d", "str_3d", "str_3d"),
    BaseSignalSpec("overnight_drift_20d", "overnight_drift_20d", "overnight_drift_20d"),
    BaseSignalSpec("ivol_20d", "ivol_20d", "ivol_20d"),
    BaseSignalSpec("vol_compression", "vol_compression", "vol_compression"),
    BaseSignalSpec("intraday_rs", "intraday_rs", "intraday_rs"),
    BaseSignalSpec("intraday_pressure", "intraday_pressure", "intraday_pressure"),
)


MODULATOR_REGISTRY: tuple[ModulatorSpec, ...] = (
    ModulatorSpec("rvol", "rel_volume_20", "liquidity"),
    ModulatorSpec("liq", "liq", "liquidity"),
    ModulatorSpec("breadth", "market_breadth", "breadth"),
    ModulatorSpec("range_comp", "range_comp_5_20", "volatility"),
    ModulatorSpec("volshock", "volume_shock", "liquidity"),
)


WAVE1_REGISTRY: tuple[FamilySeed, ...] = (
    FamilySeed("rev1", "rev1", ("rvol", "liq", "breadth")),
    FamilySeed("gap", "gap", ("rvol", "liq", "breadth")),
    FamilySeed("pressure", "pressure", ("liq", "rvol", "breadth")),
    FamilySeed("mom3", "mom3", ("rvol", "breadth")),
    FamilySeed("momentum_20d", "momentum_20d", ("breadth", "liq")),
    FamilySeed("str_3d", "str_3d", ("breadth", "rvol")),
    FamilySeed("overnight_drift_20d", "overnight_drift_20d", ("breadth", "volshock")),
    FamilySeed("ivol_20d", "ivol_20d", ("liq", "volshock")),
    FamilySeed("vol_compression", "vol_compression", ("liq", "range_comp")),
    FamilySeed("intraday_rs", "intraday_rs", ("breadth", "rvol")),
    FamilySeed("intraday_pressure", "intraday_pressure", ("liq", "breadth")),
)


WAVE1_REGIMES: tuple[RegimeMode, ...] = ("hi", "lo", "z", "rank")


WAVE2_REGISTRY: tuple[WaveTransformSeed, ...] = (
    WaveTransformSeed("tanh", ("z",)),
    WaveTransformSeed("clip3", ("hi", "lo")),
    WaveTransformSeed("sign", ("rank", "hi")),
    WaveTransformSeed("signed_square", ("z",)),
)


WAVE3_REGISTRY: tuple[WaveTemporalSeed, ...] = (
    WaveTemporalSeed("lag1", ("hi", "z", "lo"), lag=1),
    WaveTemporalSeed("lag2", ("hi", "z"), lag=2),
    WaveTemporalSeed("ema3", ("z", "hi"), lag=0),
)


WAVE4_SURVIVOR_REGISTRY: tuple[SurvivorExpansionPlan, ...] = (
    SurvivorExpansionPlan(
        wave="wave4",
        transforms=("signed_log", "sqrt_signed", "cube", "tanh_z"),
        regimes=("z", "rank", "hi", "lo"),
        interactions=("regime",),
        inherit_modulators=True,
        inherit_regimes=True,
        max_modulators_per_family=3,
    ),
)


WAVE5_SURVIVOR_REGISTRY: tuple[SurvivorExpansionPlan, ...] = (
    SurvivorExpansionPlan(
        wave="wave5",
        transforms=("raw", "z"),
        regimes=("z", "rank", "hi"),
        interactions=("raw_mul", "z_mul", "rank_mul"),
        inherit_modulators=True,
        inherit_regimes=True,
        max_modulators_per_family=3,
    ),
)


WAVE6_SURVIVOR_REGISTRY: tuple[SurvivorExpansionPlan, ...] = (
    SurvivorExpansionPlan(
        wave="wave6",
        transforms=("ema3", "lag1", "signed_log"),
        regimes=("z", "rank", "hi", "lo"),
        interactions=("regime", "z_mul"),
        inherit_modulators=True,
        inherit_regimes=True,
        max_modulators_per_family=3,
    ),
)


WAVE7_RECENT_SURVIVOR_REGISTRY: tuple[SurvivorExpansionPlan, ...] = (
    SurvivorExpansionPlan(
        wave="wave7",
        transforms=("signed_log", "z", "ema3", "lag1"),
        regimes=("z", "hi", "lo", "rank"),
        interactions=("regime", "z_mul", "rank_mul"),
        inherit_modulators=True,
        inherit_regimes=True,
        max_modulators_per_family=2,
    ),
)


SURVIVOR_DEFAULT_PRIORITY: tuple[str, ...] = (
    "rev1",
    "pressure",
    "gap",
    "mom3",
    "momentum_20d",
    "intraday_rs",
    "intraday_pressure",
    "vol_compression",
    "overnight_drift_20d",
    "ivol_20d",
    "str_3d",
)


RECENT_SURVIVOR_DEFAULT_PRIORITY: tuple[str, ...] = (
    "intraday_pressure",
    "intraday_rs",
    "ivol_20d",
    "rev1",
    "gap",
    "mom3",
)

DEFAULT_SURVIVOR_MIN_RECIPES = 2
DEFAULT_SURVIVOR_TOP_N = 6


def _recipe_name(
    left: str,
    modulator: str | None,
    transform: TransformName,
    regime: RegimeMode,
    interaction: InteractionMode,
    lag: int,
) -> str:
    parts: list[str] = [left, transform, interaction]
    if modulator:
        parts.append(modulator)
    parts.append(regime)
    if lag > 0:
        parts.append(f"lag{lag}")
    return "_".join(parts)


def _append_unique(out: list[AlphaRecipe], seen: set[str], recipe: AlphaRecipe) -> None:
    key = recipe.name
    if key in seen:
        return
    seen.add(key)
    out.append(recipe)


def build_wave1_recipes() -> tuple[AlphaRecipe, ...]:
    out: list[AlphaRecipe] = []
    seen: set[str] = set()
    for seed in WAVE1_REGISTRY:
        for modulator in seed.modulators:
            for regime in WAVE1_REGIMES:
                name = _recipe_name(seed.base_signal, modulator, "raw", regime, "regime", 0)
                _append_unique(
                    out,
                    seen,
                    AlphaRecipe(
                        name=name,
                        left=seed.base_signal,
                        modulator=modulator,
                        transform="raw",
                        regime=regime,
                        interaction="regime",
                        lag=0,
                        family=seed.family,
                        wave="wave1",
                        parents=(seed.family,),
                        source="registry:wave1",
                    ),
                )
    return tuple(out)


def build_wave2_recipes() -> tuple[AlphaRecipe, ...]:
    out: list[AlphaRecipe] = []
    seen: set[str] = set()
    for seed in WAVE1_REGISTRY:
        for modulator in seed.modulators:
            for transform_seed in WAVE2_REGISTRY:
                for regime in transform_seed.regimes:
                    name = _recipe_name(seed.base_signal, modulator, transform_seed.transform, regime, transform_seed.interaction, 0)
                    _append_unique(
                        out,
                        seen,
                        AlphaRecipe(
                            name=name,
                            left=seed.base_signal,
                            modulator=modulator,
                            transform=transform_seed.transform,
                            regime=regime,
                            interaction=transform_seed.interaction,
                            lag=0,
                            family=seed.family,
                            wave="wave2",
                            parents=(seed.family,),
                            source="registry:wave2",
                        ),
                    )
    return tuple(out)


def build_wave3_recipes() -> tuple[AlphaRecipe, ...]:
    out: list[AlphaRecipe] = []
    seen: set[str] = set()
    for seed in WAVE1_REGISTRY:
        for modulator in seed.modulators:
            for temporal_seed in WAVE3_REGISTRY:
                for regime in temporal_seed.regimes:
                    name = _recipe_name(seed.base_signal, modulator, temporal_seed.transform, regime, temporal_seed.interaction, temporal_seed.lag)
                    _append_unique(
                        out,
                        seen,
                        AlphaRecipe(
                            name=name,
                            left=seed.base_signal,
                            modulator=modulator,
                            transform=temporal_seed.transform,
                            regime=regime,
                            interaction=temporal_seed.interaction,
                            lag=temporal_seed.lag,
                            family=seed.family,
                            wave="wave3",
                            parents=(seed.family,),
                            source="registry:wave3",
                        ),
                    )
    return tuple(out)


WAVE1_RECIPES = build_wave1_recipes()
WAVE2_RECIPES = build_wave2_recipes()
WAVE3_RECIPES = build_wave3_recipes()
SEED_RECIPES = WAVE1_RECIPES + WAVE2_RECIPES + WAVE3_RECIPES
ALL_RECIPES = SEED_RECIPES


def survivor_registry_bundle(include_wave7: bool = False) -> tuple[SurvivorExpansionPlan, ...]:
    base = WAVE4_SURVIVOR_REGISTRY + WAVE5_SURVIVOR_REGISTRY + WAVE6_SURVIVOR_REGISTRY
    if include_wave7:
        return base + WAVE7_RECENT_SURVIVOR_REGISTRY
    return base


__all__ = [
    "ALL_RECIPES",
    "BASE_SIGNAL_REGISTRY",
    "DEFAULT_SURVIVOR_MIN_RECIPES",
    "DEFAULT_SURVIVOR_TOP_N",
    "FamilySeed",
    "InteractionMode",
    "MODULATOR_REGISTRY",
    "ModulatorSpec",
    "RECENT_SURVIVOR_DEFAULT_PRIORITY",
    "RegimeMode",
    "SEED_RECIPES",
    "SURVIVOR_DEFAULT_PRIORITY",
    "SurvivorExpansionPlan",
    "TransformName",
    "WAVE1_RECIPES",
    "WAVE2_RECIPES",
    "WAVE3_RECIPES",
    "WAVE4_SURVIVOR_REGISTRY",
    "WAVE5_SURVIVOR_REGISTRY",
    "WAVE6_SURVIVOR_REGISTRY",
    "WAVE7_RECENT_SURVIVOR_REGISTRY",
    "WaveTemporalSeed",
    "WaveTransformSeed",
    "AlphaRecipe",
    "BaseSignalSpec",
    "build_wave1_recipes",
    "build_wave2_recipes",
    "build_wave3_recipes",
    "survivor_registry_bundle",
]