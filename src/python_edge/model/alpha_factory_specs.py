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
    wave: str = "wave0"


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


WAVE1_PAIRS = {
    "rev1": ("rvol", "liq", "breadth"),
    "gap": ("rvol", "liq", "breadth"),
    "pressure": ("liq", "rvol", "breadth"),
    "mom3": ("rvol", "breadth"),
    "momentum_20d": ("breadth", "liq"),
    "str_3d": ("breadth", "rvol"),
    "overnight_drift_20d": ("breadth", "volshock"),
    "ivol_20d": ("liq", "volshock"),
    "vol_compression": ("liq", "range_comp"),
    "intraday_rs": ("breadth", "rvol"),
    "intraday_pressure": ("liq", "breadth"),
}


WAVE2_SPECS = (
    ("rev1", "rvol", "tanh", "z"),
    ("rev1", "breadth", "clip3", "hi"),
    ("rev1", "liq", "sign", "rank"),
    ("gap", "breadth", "tanh", "z"),
    ("gap", "liq", "clip3", "hi"),
    ("pressure", "rvol", "tanh", "z"),
    ("pressure", "liq", "sign", "hi"),
    ("mom3", "rvol", "signed_square", "z"),
    ("momentum_20d", "breadth", "tanh", "hi"),
    ("str_3d", "breadth", "clip3", "lo"),
    ("overnight_drift_20d", "breadth", "tanh", "hi"),
    ("ivol_20d", "liq", "clip3", "lo"),
    ("vol_compression", "liq", "tanh", "hi"),
    ("intraday_rs", "breadth", "tanh", "z"),
    ("intraday_pressure", "liq", "signed_square", "z"),
)


WAVE3_SPECS = (
    ("rev1", "rvol", "lag1", "hi", 1),
    ("rev1", "rvol", "lag2", "hi", 2),
    ("rev1", "rvol", "ema3", "z", 0),
    ("gap", "rvol", "lag1", "z", 1),
    ("gap", "breadth", "ema3", "z", 0),
    ("pressure", "liq", "lag1", "hi", 1),
    ("pressure", "liq", "ema3", "hi", 0),
    ("mom3", "rvol", "lag1", "hi", 1),
    ("mom3", "breadth", "ema3", "z", 0),
    ("momentum_20d", "breadth", "lag1", "hi", 1),
    ("str_3d", "breadth", "lag1", "lo", 1),
    ("overnight_drift_20d", "breadth", "lag1", "hi", 1),
    ("ivol_20d", "liq", "lag1", "lo", 1),
    ("vol_compression", "liq", "lag1", "hi", 1),
    ("intraday_rs", "breadth", "ema3", "hi", 0),
    ("intraday_pressure", "liq", "lag1", "z", 1),
)


def _build_wave1() -> tuple[AlphaRecipe, ...]:
    out: list[AlphaRecipe] = []
    for left, modulators in WAVE1_PAIRS.items():
        for mod in modulators:
            for regime in ("hi", "lo", "z", "rank"):
                name = f"{left}_{regime}_{mod}"
                out.append(AlphaRecipe(name, left, mod, "raw", regime, 0, left, "wave1"))
    return tuple(out)


def _build_wave2() -> tuple[AlphaRecipe, ...]:
    return tuple(
        AlphaRecipe(f"{left}_{transform}_{regime}_{mod}", left, mod, transform, regime, 0, left, "wave2")
        for left, mod, transform, regime in WAVE2_SPECS
    )


def _build_wave3() -> tuple[AlphaRecipe, ...]:
    return tuple(
        AlphaRecipe(f"{left}_{transform}_{regime}_{mod}", left, mod, transform, regime, lag, left, "wave3")
        for left, mod, transform, regime, lag in WAVE3_SPECS
    )


WAVE1_RECIPES = _build_wave1()
WAVE2_RECIPES = _build_wave2()
WAVE3_RECIPES = _build_wave3()
ALL_RECIPES = WAVE1_RECIPES + WAVE2_RECIPES + WAVE3_RECIPES