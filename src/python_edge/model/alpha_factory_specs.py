from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TransformName = Literal["raw", "z", "rank", "tanh", "clip3", "sign", "signed_square", "lag1", "lag2", "ema3"]
RegimeMode = Literal["none", "hi", "lo", "z", "rank"]


@dataclass(frozen=True)
class BaseSignalSpec:
    name: str
    source_col: str


@dataclass(frozen=True)
class ModulatorSpec:
    name: str
    source_col: str


@dataclass(frozen=True)
class AlphaRecipe:
    name: str
    left: str
    modulator: str | None
    transform: TransformName
    regime: RegimeMode = "none"
    lag: int = 0
    family: str = "generic"


BASE_SIGNALS = (
    BaseSignalSpec("rev1", "rev1_base"),
    BaseSignalSpec("gap", "gap_base"),
    BaseSignalSpec("pressure", "pressure_base"),
    BaseSignalSpec("mom3", "mom3_base"),
    BaseSignalSpec("momentum_20d", "momentum_20d"),
    BaseSignalSpec("str_3d", "str_3d"),
    BaseSignalSpec("overnight_drift_20d", "overnight_drift_20d"),
    BaseSignalSpec("ivol_20d", "ivol_20d"),
    BaseSignalSpec("vol_compression", "vol_compression"),
    BaseSignalSpec("intraday_rs", "intraday_rs"),
    BaseSignalSpec("intraday_pressure", "intraday_pressure"),
)


MODULATORS = (
    ModulatorSpec("rvol", "rel_volume_20"),
    ModulatorSpec("liq", "liq"),
    ModulatorSpec("breadth", "market_breadth"),
    ModulatorSpec("range_comp", "range_comp_5_20"),
    ModulatorSpec("volshock", "volume_shock"),
)


WAVE1_RECIPES = (
    AlphaRecipe("rev1_hi_rvol", "rev1", "rvol", "raw", "hi", 0, "wave1"),
    AlphaRecipe("rev1_lo_rvol", "rev1", "rvol", "raw", "lo", 0, "wave1"),
    AlphaRecipe("rev1_z_rvol", "rev1", "rvol", "raw", "z", 0, "wave1"),
    AlphaRecipe("rev1_rank_rvol", "rev1", "rvol", "raw", "rank", 0, "wave1"),
    AlphaRecipe("gap_hi_rvol", "gap", "rvol", "raw", "hi", 0, "wave1"),
    AlphaRecipe("gap_z_rvol", "gap", "rvol", "raw", "z", 0, "wave1"),
    AlphaRecipe("pressure_hi_liq", "pressure", "liq", "raw", "hi", 0, "wave1"),
    AlphaRecipe("pressure_z_liq", "pressure", "liq", "raw", "z", 0, "wave1"),
    AlphaRecipe("mom3_hi_rvol", "mom3", "rvol", "raw", "hi", 0, "wave1"),
)


WAVE2_RECIPES = (
    AlphaRecipe("rev1_tanh_rvol", "rev1", "rvol", "tanh", "z", 0, "wave2"),
    AlphaRecipe("rev1_clip_breadth", "rev1", "breadth", "clip3", "hi", 0, "wave2"),
    AlphaRecipe("gap_tanh_breadth", "gap", "breadth", "tanh", "z", 0, "wave2"),
    AlphaRecipe("mom3_square_rvol", "mom3", "rvol", "signed_square", "z", 0, "wave2"),
    AlphaRecipe("volcomp_liq", "vol_compression", "liq", "raw", "hi", 0, "wave2"),
    AlphaRecipe("intraday_rs_breadth", "intraday_rs", "breadth", "raw", "z", 0, "wave2"),
)


WAVE3_RECIPES = (
    AlphaRecipe("rev1_hi_rvol_lag1", "rev1", "rvol", "lag1", "hi", 1, "wave3"),
    AlphaRecipe("rev1_hi_rvol_lag2", "rev1", "rvol", "lag2", "hi", 2, "wave3"),
    AlphaRecipe("rev1_rvol_ema3", "rev1", "rvol", "ema3", "z", 0, "wave3"),
    AlphaRecipe("gap_rvol_lag1", "gap", "rvol", "lag1", "z", 1, "wave3"),
    AlphaRecipe("pressure_liq_lag1", "pressure", "liq", "lag1", "hi", 1, "wave3"),
)


ALL_RECIPES = WAVE1_RECIPES + WAVE2_RECIPES + WAVE3_RECIPES