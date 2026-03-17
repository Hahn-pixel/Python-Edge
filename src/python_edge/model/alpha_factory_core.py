from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

from python_edge.model.alpha_factory_specs import (
    RecipeSpec,
    SurvivorConfig,
    base_registry_specs,
    prioritized_recent_survivors,
    recipe_name_from_spec,
    targeted_wave7_specs_from_recent_survivors,
)


REQUIRED_REGISTRY_COLUMNS = [
    "family",
    "wave",
    "signal",
    "transform",
    "interaction",
    "regime",
    "modulator",
    "lag",
    "horizon",
    "smoothing",
    "source_alpha",
    "tags",
    "recipe_name",
]


def _empty_registry() -> pd.DataFrame:
    return pd.DataFrame(columns=REQUIRED_REGISTRY_COLUMNS)


def _ensure_registry_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in REQUIRED_REGISTRY_COLUMNS:
        if col not in out.columns:
            out[col] = "" if col != "tags" else [[] for _ in range(len(out))]
    return out[REQUIRED_REGISTRY_COLUMNS].copy()


def _specs_to_frame(specs: Sequence[RecipeSpec]) -> pd.DataFrame:
    if not specs:
        return _empty_registry()
    rows: List[Dict[str, object]] = []
    for spec in specs:
        row = asdict(spec)
        row["recipe_name"] = recipe_name_from_spec(spec)
        rows.append(row)
    out = pd.DataFrame(rows)
    out["tags"] = out["tags"].apply(lambda x: list(x) if isinstance(x, (list, tuple)) else [str(x)])
    return _ensure_registry_columns(out)


def _normalize_manifest_rows(manifest_df: Optional[pd.DataFrame]) -> List[Dict[str, object]]:
    if manifest_df is None or manifest_df.empty:
        return []
    cols = {str(c): str(c) for c in manifest_df.columns}
    alpha_col = None
    family_col = None
    selector_col = None
    shortlist_col = None
    for candidate in ["alpha", "alpha_name", "name"]:
        if candidate in cols:
            alpha_col = cols[candidate]
            break
    for candidate in ["family", "source_family"]:
        if candidate in cols:
            family_col = cols[candidate]
            break
    for candidate in ["selector_score", "score"]:
        if candidate in cols:
            selector_col = cols[candidate]
            break
    for candidate in ["shortlist_rank", "rank"]:
        if candidate in cols:
            shortlist_col = cols[candidate]
            break
    if alpha_col is None:
        return []
    rows: List[Dict[str, object]] = []
    for _, row in manifest_df.iterrows():
        rows.append(
            {
                "alpha": row.get(alpha_col, ""),
                "family": row.get(family_col, "") if family_col is not None else "",
                "selector_score": row.get(selector_col, 0.0) if selector_col is not None else 0.0,
                "shortlist_rank": row.get(shortlist_col, 1_000_000) if shortlist_col is not None else 1_000_000,
            }
        )
    return rows


def _apply_survivor_filter(registry_df: pd.DataFrame, survivor_cfg: Optional[SurvivorConfig]) -> pd.DataFrame:
    if survivor_cfg is None or not survivor_cfg.enabled or registry_df.empty:
        return registry_df.copy()
    out = registry_df.copy()
    if survivor_cfg.include_families:
        out = out[out["family"].isin(set(survivor_cfg.include_families))].copy()
    if survivor_cfg.exclude_families:
        out = out[~out["family"].isin(set(survivor_cfg.exclude_families))].copy()
    if survivor_cfg.include_waves:
        out = out[out["wave"].isin(set(survivor_cfg.include_waves))].copy()
    if out.empty:
        return out
    if survivor_cfg.family_cap > 0:
        out = out.sort_values(["family", "wave", "recipe_name"], ascending=[True, True, True]).copy()
        out["_family_rank"] = out.groupby("family", sort=False).cumcount() + 1
        out = out[out["_family_rank"] <= survivor_cfg.family_cap].copy()
        out = out.drop(columns=["_family_rank"])
    return out.reset_index(drop=True)


def build_recipe_registry(
    manifest_df: Optional[pd.DataFrame] = None,
    survivor_cfg: Optional[SurvivorConfig] = None,
    include_base: bool = True,
    include_wave7: bool = True,
) -> pd.DataFrame:
    specs: List[RecipeSpec] = []
    if include_base:
        specs.extend(base_registry_specs())
    recent_rows = prioritized_recent_survivors(_normalize_manifest_rows(manifest_df))
    if include_wave7 and recent_rows:
        family_cap = survivor_cfg.family_cap if survivor_cfg is not None and survivor_cfg.family_cap > 0 else 2
        specs.extend(targeted_wave7_specs_from_recent_survivors(recent_rows, family_cap=family_cap))
    registry_df = _specs_to_frame(specs)
    registry_df = _apply_survivor_filter(registry_df, survivor_cfg)
    if registry_df.empty:
        return _empty_registry()
    registry_df = registry_df.drop_duplicates(subset=["recipe_name"], keep="first").reset_index(drop=True)
    return _ensure_registry_columns(registry_df)


def registry_to_recipe_dicts(registry_df: pd.DataFrame) -> List[Dict[str, object]]:
    if registry_df is None or registry_df.empty:
        return []
    frame = _ensure_registry_columns(registry_df)
    rows: List[Dict[str, object]] = []
    for _, row in frame.iterrows():
        payload = row.to_dict()
        payload["tags"] = list(payload.get("tags", []))
        rows.append(payload)
    return rows


def registry_summary(registry_df: pd.DataFrame) -> Dict[str, object]:
    if registry_df is None or registry_df.empty:
        return {
            "rows": 0,
            "waves": {},
            "families": {},
        }
    frame = _ensure_registry_columns(registry_df)
    return {
        "rows": int(len(frame)),
        "waves": frame["wave"].value_counts(dropna=False).to_dict(),
        "families": frame["family"].value_counts(dropna=False).to_dict(),
    }