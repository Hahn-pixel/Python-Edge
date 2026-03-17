from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class SurvivorConfig:
    enabled: bool = True
    min_keep_count: int = 1
    family_cap: int = 4
    include_families: Tuple[str, ...] = ()
    exclude_families: Tuple[str, ...] = ()
    include_waves: Tuple[str, ...] = ()
    recent_only: bool = False


@dataclass(frozen=True)
class RecipeSpec:
    family: str
    wave: str
    signal: str
    transform: str = "raw"
    interaction: str = "none"
    regime: str = "none"
    modulator: str = "none"
    lag: str = "none"
    horizon: str = "none"
    smoothing: str = "none"
    source_alpha: str = ""
    tags: Tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, object]:
        return {
            "family": self.family,
            "wave": self.wave,
            "signal": self.signal,
            "transform": self.transform,
            "interaction": self.interaction,
            "regime": self.regime,
            "modulator": self.modulator,
            "lag": self.lag,
            "horizon": self.horizon,
            "smoothing": self.smoothing,
            "source_alpha": self.source_alpha,
            "tags": list(self.tags),
        }


RECENT_SURVIVOR_FAMILY_PRIORITY: Tuple[str, ...] = (
    "intraday_pressure",
    "intraday_rs",
    "ivol_20d",
    "rev1",
    "gap",
    "mom3",
)


RECENT_SURVIVOR_LIBRARY: Dict[str, Dict[str, Sequence[str]]] = {
    "intraday_pressure": {
        "signals": ["intraday_pressure"],
        "transforms": ["lag1", "ema3", "z"],
        "interactions": ["regime", "z_mul"],
        "regimes": ["liq_lo", "liq_z"],
        "modulators": ["liq", "rvol"],
        "lags": ["lag1", "lag2"],
        "horizons": ["none"],
        "smoothing": ["lag1", "ema3"],
    },
    "intraday_rs": {
        "signals": ["intraday_rs"],
        "transforms": ["ema2", "ema3", "ema5", "z"],
        "interactions": ["regime", "z_mul", "rank_mul"],
        "regimes": ["rvol_z", "rvol_hi"],
        "modulators": ["rvol", "liq"],
        "lags": ["none", "lag1"],
        "horizons": ["none"],
        "smoothing": ["ema2", "ema3", "ema5"],
    },
    "ivol_20d": {
        "signals": ["ivol_20d"],
        "transforms": ["signed_log", "z", "tanh"],
        "interactions": ["z_mul", "rank_mul"],
        "regimes": ["none"],
        "modulators": ["liq", "rvol"],
        "lags": ["none", "lag1"],
        "horizons": ["none"],
        "smoothing": ["none"],
    },
    "rev1": {
        "signals": ["rev1"],
        "transforms": ["z", "signed_log"],
        "interactions": ["z_mul", "rank_mul"],
        "regimes": ["none"],
        "modulators": ["rvol", "liq"],
        "lags": ["none", "lag1"],
        "horizons": ["h1", "h2"],
        "smoothing": ["none"],
    },
    "gap": {
        "signals": ["gap"],
        "transforms": ["z", "signed_log"],
        "interactions": ["z_mul", "rank_mul"],
        "regimes": ["rvol_z", "liq_z"],
        "modulators": ["rvol", "liq"],
        "lags": ["none", "lag1"],
        "horizons": ["h1", "h2"],
        "smoothing": ["none"],
    },
    "mom3": {
        "signals": ["mom3"],
        "transforms": ["z", "signed_log", "ema3"],
        "interactions": ["z_mul", "rank_mul"],
        "regimes": ["rvol_z", "none"],
        "modulators": ["rvol", "liq"],
        "lags": ["none", "lag1"],
        "horizons": ["h1", "h2"],
        "smoothing": ["ema3", "none"],
    },
}


def _norm_str_list(values: Iterable[object]) -> List[str]:
    out: List[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text.lower() == "nan":
            continue
        out.append(text)
    return out


def recipe_name_from_spec(spec: RecipeSpec) -> str:
    parts: List[str] = [spec.family]
    if spec.transform != "none":
        parts.append(spec.transform)
    if spec.interaction != "none":
        parts.append(spec.interaction)
    if spec.regime != "none":
        parts.append(spec.regime)
    if spec.modulator != "none":
        parts.append(spec.modulator)
    if spec.lag != "none":
        parts.append(spec.lag)
    if spec.horizon != "none":
        parts.append(spec.horizon)
    if spec.smoothing != "none" and spec.smoothing not in parts:
        parts.append(spec.smoothing)
    return "_".join(parts)


def base_registry_specs() -> List[RecipeSpec]:
    specs: List[RecipeSpec] = []
    for family, cfg in RECENT_SURVIVOR_LIBRARY.items():
        specs.append(
            RecipeSpec(
                family=family,
                wave="wave6",
                signal=cfg["signals"][0],
                transform=cfg["transforms"][0],
                interaction=cfg["interactions"][0],
                regime=cfg["regimes"][0],
                modulator=cfg["modulators"][0],
                lag=cfg["lags"][0],
                horizon=cfg["horizons"][0],
                smoothing=cfg["smoothing"][0],
                tags=("baseline",),
            )
        )
    return specs


def targeted_wave7_specs_from_recent_survivors(
    recent_survivors: Sequence[Dict[str, object]],
    family_cap: int = 2,
) -> List[RecipeSpec]:
    if not recent_survivors:
        return []
    specs: List[RecipeSpec] = []
    family_counts: Dict[str, int] = {}
    for row in recent_survivors:
        family = str(row.get("family", "")).strip()
        alpha_name = str(row.get("alpha", "")).strip()
        if not family or family not in RECENT_SURVIVOR_LIBRARY:
            continue
        if family_counts.get(family, 0) >= family_cap:
            continue
        cfg = RECENT_SURVIVOR_LIBRARY[family]
        signal = cfg["signals"][0]
        for transform in _norm_str_list(cfg["transforms"]):
            for interaction in _norm_str_list(cfg["interactions"]):
                for regime in _norm_str_list(cfg["regimes"]):
                    for modulator in _norm_str_list(cfg["modulators"]):
                        for lag in _norm_str_list(cfg["lags"]):
                            for horizon in _norm_str_list(cfg["horizons"]):
                                for smoothing in _norm_str_list(cfg["smoothing"]):
                                    specs.append(
                                        RecipeSpec(
                                            family=family,
                                            wave="wave7",
                                            signal=signal,
                                            transform=transform,
                                            interaction=interaction,
                                            regime=regime,
                                            modulator=modulator,
                                            lag=lag,
                                            horizon=horizon,
                                            smoothing=smoothing,
                                            source_alpha=alpha_name,
                                            tags=("recent_survivor",),
                                        )
                                    )
        family_counts[family] = family_counts.get(family, 0) + 1
    return specs


def prioritized_recent_survivors(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    priority = {name: idx for idx, name in enumerate(RECENT_SURVIVOR_FAMILY_PRIORITY)}

    def sort_key(row: Dict[str, object]) -> Tuple[float, float, int, str]:
        family = str(row.get("family", ""))
        selector_score = float(row.get("selector_score", 0.0) or 0.0)
        shortlist_rank = float(row.get("shortlist_rank", 1_000_000) or 1_000_000)
        fam_rank = priority.get(family, 999)
        return (-selector_score, shortlist_rank, fam_rank, str(row.get("alpha", "")))

    unique: Dict[str, Dict[str, object]] = {}
    for row in rows:
        alpha = str(row.get("alpha", "")).strip()
        if not alpha:
            continue
        if alpha not in unique:
            unique[alpha] = dict(row)
    return sorted(unique.values(), key=sort_key)