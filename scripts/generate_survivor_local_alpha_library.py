from __future__ import annotations

import json
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in [ROOT, SRC]:
    ps = str(p)
    if ps not in sys.path:
        sys.path.insert(0, ps)

from python_edge.model.alpha_factory_core import ValidationConfig, derive_base_factory_inputs, generate_factory_alphas
from python_edge.model.alpha_factory_specs import AlphaRecipe, BASE_SIGNAL_REGISTRY, SEED_RECIPES

FEATURE_FILE = Path(os.getenv("FEATURE_FILE", ROOT / "data" / "features" / "feature_matrix_v1.parquet"))
ALIVE_SUMMARY_CSV = Path(os.getenv("ALIVE_SUMMARY_CSV", ROOT / "artifacts" / "single_alpha_audit_wf" / "single_alpha_audit__alive.csv"))
RESIDUAL_COMPONENT_SUMMARY_CSV = Path(os.getenv("RESIDUAL_COMPONENT_SUMMARY_CSV", ROOT / "artifacts" / "residual_layer_wf" / "residual_component_summary.csv"))
OUT_DIR = Path(os.getenv("SURVIVOR_LOCAL_OUT_DIR", ROOT / "data" / "alpha_library_survivor_local"))
OUT_PARQUET = OUT_DIR / "alpha_library_survivor_local.parquet"
OUT_META = OUT_DIR / "alpha_library_survivor_local.meta.json"
OUT_MANIFEST = OUT_DIR / "alpha_library_survivor_local.manifest.csv"
OUT_DROPPED = OUT_DIR / "alpha_library_survivor_local.dropped.csv"
OUT_RECIPE_CSV = OUT_DIR / "alpha_library_survivor_local.recipes.csv"
OUT_FAMILY_CSV = OUT_DIR / "alpha_library_survivor_local.focus_families.csv"

PAUSE_ON_EXIT = str(os.getenv("PAUSE_ON_EXIT", "auto")).strip().lower()
TOP_FAMILIES = int(os.getenv("TOP_FAMILIES", "3"))
MAX_SEED_PER_FAMILY = int(os.getenv("MAX_SEED_PER_FAMILY", "999"))
EXPLICIT_FAMILIES = tuple(x.strip() for x in str(os.getenv("EXPLICIT_FAMILIES", "")).split(",") if x.strip())
ONLY_EXPLICIT_FAMILIES = str(os.getenv("ONLY_EXPLICIT_FAMILIES", "0")).strip().lower() not in {"0", "false", "no", "off"}

VALIDATION_MAX_NAN_RATIO = float(os.getenv("VALIDATION_MAX_NAN_RATIO", "0.995"))
VALIDATION_MIN_NON_NA = int(os.getenv("VALIDATION_MIN_NON_NA", "200"))
VALIDATION_MIN_UNIQUE = int(os.getenv("VALIDATION_MIN_UNIQUE", "5"))

ENABLE_IVOL_FAMILY = str(os.getenv("ENABLE_IVOL_FAMILY", "1")).strip().lower() not in {"0", "false", "no", "off"}
ENABLE_STR_FAMILY = str(os.getenv("ENABLE_STR_FAMILY", "1")).strip().lower() not in {"0", "false", "no", "off"}
ENABLE_DEFAULT_FAMILY = str(os.getenv("ENABLE_DEFAULT_FAMILY", "1")).strip().lower() not in {"0", "false", "no", "off"}


@dataclass(frozen=True)
class FamilyPlan:
    family: str
    transforms: Tuple[str, ...]
    regimes: Tuple[str, ...]
    interactions: Tuple[str, ...]
    modulators: Tuple[str, ...]
    include_none_modulator: bool = False


def _enable_line_buffering() -> None:
    for stream_name in ["stdout", "stderr"]:
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(line_buffering=True)
            except Exception:
                pass


def _should_pause() -> bool:
    if PAUSE_ON_EXIT in {"0", "false", "no", "off"}:
        return False
    if PAUSE_ON_EXIT in {"1", "true", "yes", "on"}:
        return True
    stdin_obj = getattr(sys, "stdin", None)
    stdout_obj = getattr(sys, "stdout", None)
    stdin_ok = bool(stdin_obj is not None and hasattr(stdin_obj, "isatty") and stdin_obj.isatty())
    stdout_ok = bool(stdout_obj is not None and hasattr(stdout_obj, "isatty") and stdout_obj.isatty())
    return stdin_ok and stdout_ok


def _safe_exit(code: int) -> None:
    if _should_pause():
        try:
            print("")
            print(f"[EXIT] code={code}")
            input("Press Enter to exit...")
        except Exception:
            pass
    raise SystemExit(code)


def _must_exist(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def _family_base_map() -> Dict[str, str]:
    return {spec.family: spec.name for spec in BASE_SIGNAL_REGISTRY}


def _known_families() -> List[str]:
    return sorted(_family_base_map().keys(), key=len, reverse=True)


def _infer_family_from_alpha(alpha_name: str) -> str:
    alpha = str(alpha_name)
    for fam in _known_families():
        token = f"alpha_{fam}_"
        if token in alpha:
            return fam
        if alpha.startswith(f"alpha_{fam}"):
            return fam
    return "unknown"


def _load_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df if len(df) else pd.DataFrame(columns=df.columns)


def _family_scores_from_alive(alive_df: pd.DataFrame) -> pd.DataFrame:
    if alive_df.empty:
        return pd.DataFrame(columns=["family", "alive_count", "alive_score"])
    work = alive_df.copy()
    if "family" not in work.columns:
        work["family"] = work["alpha"].astype(str).map(_infer_family_from_alpha)
    for col in ["last_2_fold_mean_sharpe", "last_fold_sharpe", "daily_ic_mean", "standalone_alive"]:
        if col not in work.columns:
            work[col] = 0.0
    work["alive_score"] = pd.to_numeric(work["last_2_fold_mean_sharpe"], errors="coerce").fillna(0.0)
    work["alive_score"] += 0.5 * pd.to_numeric(work["last_fold_sharpe"], errors="coerce").fillna(0.0)
    work["alive_score"] += 25.0 * pd.to_numeric(work["daily_ic_mean"], errors="coerce").fillna(0.0)
    grouped = work.groupby("family", sort=False).agg(
        alive_count=("alpha", "count"),
        alive_score=("alive_score", "sum"),
    ).reset_index()
    return grouped


def _family_scores_from_residual(residual_df: pd.DataFrame) -> pd.DataFrame:
    if residual_df.empty:
        return pd.DataFrame(columns=["family", "residual_count", "residual_score"])
    work = residual_df.copy()
    if "alpha" not in work.columns:
        return pd.DataFrame(columns=["family", "residual_count", "residual_score"])
    if "family" not in work.columns:
        work["family"] = work["alpha"].astype(str).map(_infer_family_from_alpha)
    for col in ["oos_sharpe_mean", "last_2_fold_mean_sharpe", "test_mean_ic", "sign_stability", "tail_positive_rate"]:
        if col not in work.columns:
            work[col] = 0.0
    work["residual_score"] = pd.to_numeric(work["oos_sharpe_mean"], errors="coerce").fillna(0.0)
    work["residual_score"] += pd.to_numeric(work["last_2_fold_mean_sharpe"], errors="coerce").fillna(0.0)
    work["residual_score"] += 15.0 * pd.to_numeric(work["test_mean_ic"], errors="coerce").fillna(0.0)
    work["residual_score"] += 0.25 * pd.to_numeric(work["sign_stability"], errors="coerce").fillna(0.0)
    work["residual_score"] += 0.25 * pd.to_numeric(work["tail_positive_rate"], errors="coerce").fillna(0.0)
    grouped = work.groupby("family", sort=False).agg(
        residual_count=("alpha", "count"),
        residual_score=("residual_score", "sum"),
    ).reset_index()
    return grouped


def choose_focus_families(alive_df: pd.DataFrame, residual_df: pd.DataFrame) -> pd.DataFrame:
    alive_scores = _family_scores_from_alive(alive_df)
    residual_scores = _family_scores_from_residual(residual_df)
    merged = pd.merge(alive_scores, residual_scores, on="family", how="outer")
    if merged.empty:
        merged = pd.DataFrame({"family": list(EXPLICIT_FAMILIES)}) if EXPLICIT_FAMILIES else pd.DataFrame(columns=["family"])
    for col in ["alive_count", "alive_score", "residual_count", "residual_score"]:
        if col not in merged.columns:
            merged[col] = 0.0
    merged["alive_count"] = pd.to_numeric(merged["alive_count"], errors="coerce").fillna(0).astype(int)
    merged["residual_count"] = pd.to_numeric(merged["residual_count"], errors="coerce").fillna(0).astype(int)
    merged["alive_score"] = pd.to_numeric(merged["alive_score"], errors="coerce").fillna(0.0)
    merged["residual_score"] = pd.to_numeric(merged["residual_score"], errors="coerce").fillna(0.0)
    merged["explicit"] = merged["family"].astype(str).isin(set(EXPLICIT_FAMILIES)).astype(int)
    merged["focus_score"] = 2.0 * merged["alive_score"] + merged["residual_score"] + 5.0 * merged["explicit"]
    merged = merged.loc[merged["family"].astype(str) != "unknown"].copy()
    merged = merged.sort_values(["explicit", "focus_score", "alive_count", "residual_count", "family"], ascending=[False, False, False, False, True]).reset_index(drop=True)
    if ONLY_EXPLICIT_FAMILIES:
        merged = merged.loc[merged["explicit"] == 1].copy()
    else:
        if TOP_FAMILIES > 0:
            top = merged.head(TOP_FAMILIES).copy()
            if EXPLICIT_FAMILIES:
                explicit_rows = merged.loc[merged["explicit"] == 1].copy()
                merged = pd.concat([explicit_rows, top], ignore_index=True)
                merged = merged.drop_duplicates(subset=["family"]).reset_index(drop=True)
            else:
                merged = top
    if merged.empty and EXPLICIT_FAMILIES:
        merged = pd.DataFrame({"family": list(EXPLICIT_FAMILIES)})
        merged["alive_count"] = 0
        merged["alive_score"] = 0.0
        merged["residual_count"] = 0
        merged["residual_score"] = 0.0
        merged["explicit"] = 1
        merged["focus_score"] = 5.0
    return merged


def plan_for_family(family: str) -> FamilyPlan:
    fam = str(family)
    if fam == "ivol_20d" and ENABLE_IVOL_FAMILY:
        return FamilyPlan(
            family=fam,
            transforms=("raw", "z", "signed_log", "tanh_z", "ema3", "lag1"),
            regimes=("z", "hi", "lo", "rank"),
            interactions=("regime", "z_mul", "rank_mul"),
            modulators=("volshock", "liq", "rvol"),
            include_none_modulator=False,
        )
    if fam == "str_3d" and ENABLE_STR_FAMILY:
        return FamilyPlan(
            family=fam,
            transforms=("raw", "z", "signed_log", "ema3", "lag1"),
            regimes=("z", "hi", "lo", "rank"),
            interactions=("regime", "z_mul", "rank_mul"),
            modulators=("rvol", "breadth", "volshock"),
            include_none_modulator=False,
        )
    return FamilyPlan(
        family=fam,
        transforms=("raw", "z", "signed_log", "ema3"),
        regimes=("z", "hi", "lo"),
        interactions=("regime", "z_mul"),
        modulators=("rvol", "liq", "breadth", "volshock"),
        include_none_modulator=False,
    )


def _seed_recipes_for_families(families: Sequence[str]) -> List[AlphaRecipe]:
    wanted = set(str(x) for x in families)
    out: List[AlphaRecipe] = []
    counts: Dict[str, int] = {k: 0 for k in wanted}
    for recipe in SEED_RECIPES:
        fam = str(recipe.family)
        if fam not in wanted:
            continue
        if MAX_SEED_PER_FAMILY > 0 and counts.get(fam, 0) >= MAX_SEED_PER_FAMILY:
            continue
        out.append(recipe)
        counts[fam] = counts.get(fam, 0) + 1
    return out


def _make_recipe_name(left: str, transform: str, interaction: str, modulator: str | None, regime: str, lag: int) -> str:
    parts: List[str] = [left, transform, interaction]
    if modulator:
        parts.append(modulator)
    if regime != "none":
        parts.extend(["regime", regime])
    if lag > 0:
        parts.append(f"lag{lag}")
    return "alpha_" + "_".join(parts)


def _make_targeted_recipes(families: Sequence[str]) -> List[AlphaRecipe]:
    base_map = _family_base_map()
    out: List[AlphaRecipe] = []
    seen: set[str] = set()
    for family in families:
        plan = plan_for_family(family)
        if family == "ivol_20d" and not ENABLE_IVOL_FAMILY:
            continue
        if family == "str_3d" and not ENABLE_STR_FAMILY:
            continue
        if family not in {"ivol_20d", "str_3d"} and not ENABLE_DEFAULT_FAMILY:
            continue
        left = base_map.get(family)
        if not left:
            continue
        parent_names = tuple(sorted({r.name for r in SEED_RECIPES if str(r.family) == str(family)}))
        modulator_list: List[str | None] = list(plan.modulators)
        if plan.include_none_modulator:
            modulator_list.append(None)
        for transform in plan.transforms:
            lag = 0
            if transform == "lag1":
                lag = 1
            elif transform == "lag2":
                lag = 2
            for interaction in plan.interactions:
                for modulator in modulator_list:
                    if modulator is None and interaction != "regime":
                        continue
                    for regime in plan.regimes:
                        name = _make_recipe_name(left, transform, interaction, modulator, regime, lag)
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
                                lag=lag,
                                family=family,
                                wave="wave_focus",
                                parents=parent_names,
                                source="survivor_local:focused",
                            )
                        )
    return out


def _recipes_to_frame(recipes: Sequence[AlphaRecipe]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for r in recipes:
        rows.append(
            {
                "alpha": r.name,
                "family": r.family,
                "wave": r.wave,
                "left": r.left,
                "modulator": r.modulator,
                "transform": r.transform,
                "regime": r.regime,
                "interaction": r.interaction,
                "lag": r.lag,
                "source": r.source,
                "parents": "|".join(r.parents),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    _enable_line_buffering()
    print(f"[CFG] feature_file={FEATURE_FILE}")
    print(f"[CFG] alive_summary_csv={ALIVE_SUMMARY_CSV}")
    print(f"[CFG] residual_component_summary_csv={RESIDUAL_COMPONENT_SUMMARY_CSV}")
    print(f"[CFG] out_dir={OUT_DIR}")
    print(f"[CFG] explicit_families={list(EXPLICIT_FAMILIES)} only_explicit={int(ONLY_EXPLICIT_FAMILIES)} top_families={TOP_FAMILIES}")
    print(f"[CFG] enable_ivol_family={int(ENABLE_IVOL_FAMILY)} enable_str_family={int(ENABLE_STR_FAMILY)} enable_default_family={int(ENABLE_DEFAULT_FAMILY)}")
    print(f"[CFG] validation max_nan_ratio={VALIDATION_MAX_NAN_RATIO} min_non_na={VALIDATION_MIN_NON_NA} min_unique={VALIDATION_MIN_UNIQUE}")

    _must_exist(FEATURE_FILE, "Feature file")
    df = pd.read_parquet(FEATURE_FILE)
    if df.empty:
        raise RuntimeError("feature_matrix_v1.parquet is empty")

    alive_df = _load_csv_if_exists(ALIVE_SUMMARY_CSV)
    residual_df = _load_csv_if_exists(RESIDUAL_COMPONENT_SUMMARY_CSV)

    focus_family_df = choose_focus_families(alive_df, residual_df)
    if focus_family_df.empty:
        raise RuntimeError("No focus families selected. Provide EXPLICIT_FAMILIES or the upstream audit artifacts.")
    focus_families = focus_family_df["family"].astype(str).tolist()
    print("[FOCUS][FAMILIES]")
    print(focus_family_df.to_string(index=False))

    seed_recipes = _seed_recipes_for_families(focus_families)
    targeted_recipes = _make_targeted_recipes(focus_families)
    all_recipes = list(seed_recipes) + list(targeted_recipes)
    if not all_recipes:
        raise RuntimeError("No recipes generated for selected focus families")

    recipe_df = _recipes_to_frame(all_recipes)
    recipe_df = recipe_df.drop_duplicates(subset=["alpha"]).reset_index(drop=True)
    dedup_recipe_names = recipe_df["alpha"].astype(str).tolist()
    recipe_lookup = {r.name: r for r in all_recipes}
    final_recipes = [recipe_lookup[name] for name in dedup_recipe_names if name in recipe_lookup]

    print(f"[RECIPES] total={len(final_recipes)} seed={len(seed_recipes)} targeted={len(targeted_recipes)}")
    print("[RECIPES][BY_FAMILY]")
    print(recipe_df.groupby(["family", "wave"], sort=False).size().rename("recipes").reset_index().to_string(index=False))

    base_df = derive_base_factory_inputs(df)
    result = generate_factory_alphas(
        base_df,
        recipes=final_recipes,
        cfg=ValidationConfig(
            max_nan_ratio=VALIDATION_MAX_NAN_RATIO,
            min_non_na=VALIDATION_MIN_NON_NA,
            min_unique=VALIDATION_MIN_UNIQUE,
        ),
    )

    final_df = result.frame
    alpha_cols = [c for c in final_df.columns if c.startswith("alpha_")]
    if not alpha_cols:
        raise RuntimeError("No alpha columns survived validation")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(OUT_PARQUET, index=False)
    result.manifest.to_csv(OUT_MANIFEST, index=False)
    result.dropped.to_csv(OUT_DROPPED, index=False)
    recipe_df.to_csv(OUT_RECIPE_CSV, index=False)
    focus_family_df.to_csv(OUT_FAMILY_CSV, index=False)

    meta = {
        "rows": int(len(final_df)),
        "alpha_count": int(len(alpha_cols)),
        "recipe_count_requested": int(len(final_recipes)),
        "seed_recipe_count": int(len(seed_recipes)),
        "targeted_recipe_count": int(len(targeted_recipes)),
        "kept_count": int(len(result.manifest)),
        "dropped_count": int(len(result.dropped)),
        "focus_families": focus_families,
        "waves": result.manifest["wave"].value_counts().to_dict() if len(result.manifest) else {},
        "families": result.manifest["family"].value_counts().to_dict() if len(result.manifest) else {},
        "inputs": {
            "feature_file": str(FEATURE_FILE),
            "alive_summary_csv": str(ALIVE_SUMMARY_CSV),
            "residual_component_summary_csv": str(RESIDUAL_COMPONENT_SUMMARY_CSV),
            "explicit_families": list(EXPLICIT_FAMILIES),
            "only_explicit_families": int(ONLY_EXPLICIT_FAMILIES),
            "top_families": TOP_FAMILIES,
        },
        "validation": {
            "max_nan_ratio": VALIDATION_MAX_NAN_RATIO,
            "min_non_na": VALIDATION_MIN_NON_NA,
            "min_unique": VALIDATION_MIN_UNIQUE,
        },
    }
    OUT_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ALPHA_FACTORY][FOCUSED] rows={len(final_df)} alpha_count={len(alpha_cols)}")
    print(f"[ALPHA_FACTORY][FOCUSED] kept={len(result.manifest)} dropped={len(result.dropped)}")
    print(f"[ARTIFACT] {OUT_PARQUET}")
    print(f"[ARTIFACT] {OUT_META}")
    print(f"[ARTIFACT] {OUT_MANIFEST}")
    print(f"[ARTIFACT] {OUT_DROPPED}")
    print(f"[ARTIFACT] {OUT_RECIPE_CSV}")
    print(f"[ARTIFACT] {OUT_FAMILY_CSV}")
    print("[FINAL] survivor-local alpha library generation complete")
    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 1
    _safe_exit(rc)
