from __future__ import annotations

import json
import math
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "alpha_library_v2"
META_PATH = DATA_DIR / "alpha_library_v2.meta.json"
MANIFEST_PATH = DATA_DIR / "alpha_library_v2.manifest.csv"
DROPPED_PATH = DATA_DIR / "alpha_library_v2.dropped.csv"
PARQUET_PATH = DATA_DIR / "alpha_library_v2.parquet"
OUT_DIR = DATA_DIR / "diagnostics"


@dataclass(frozen=True)
class SelectorConfig:
    shortlist_target: int = 120
    max_per_family: int = 24
    max_per_wave: int = 40
    max_per_signature: int = 2
    corr_screen_top_n: int = 160
    corr_threshold: float = 0.985
    min_non_na_ratio: float = 0.60


def _safe_input(prompt: str) -> None:
    try:
        input(prompt)
    except EOFError:
        pass


def _must_exist(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return path


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _structural_score(manifest: pd.DataFrame) -> pd.DataFrame:
    out = manifest.copy()
    for col in ["non_na", "nan_ratio", "unique", "lag"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    wave_bonus_map = {
        "wave1": 0.80,
        "wave2": 0.95,
        "wave3": 1.00,
        "wave4": 1.10,
        "wave5": 1.25,
        "wave6": 1.15,
    }
    transform_bonus_map = {
        "raw": 0.90,
        "z": 1.00,
        "rank": 1.00,
        "tanh": 1.05,
        "clip3": 0.95,
        "sign": 0.80,
        "signed_square": 1.05,
        "signed_log": 1.10,
        "sqrt_signed": 1.05,
        "cube": 1.10,
        "tanh_z": 1.15,
        "lag1": 0.95,
        "lag2": 0.85,
        "ema3": 1.05,
    }
    interaction_bonus_map = {
        "regime": 1.00,
        "raw_mul": 1.12,
        "z_mul": 1.18,
        "rank_mul": 1.14,
    }
    regime_bonus_map = {
        "none": 0.85,
        "hi": 1.00,
        "lo": 1.00,
        "z": 1.10,
        "rank": 1.08,
    }

    family_counts = out["family"].value_counts(dropna=False)
    wave_counts = out["wave"].value_counts(dropna=False)
    sig_counts = out.groupby(["family", "left", "modulator", "transform", "regime", "interaction", "lag"], dropna=False).size().rename("signature_size").reset_index()
    out = out.merge(sig_counts, on=["family", "left", "modulator", "transform", "regime", "interaction", "lag"], how="left")

    out["non_na_ratio"] = out["non_na"] / out["non_na"].max()
    out["unique_ratio"] = np.log1p(out["unique"].clip(lower=0.0)) / np.log1p(max(1.0, float(out["unique"].max())))
    out["density_score"] = (1.0 - out["nan_ratio"].clip(lower=0.0, upper=1.0))
    out["wave_bonus"] = out["wave"].map(wave_bonus_map).fillna(1.0)
    out["transform_bonus"] = out["transform"].map(transform_bonus_map).fillna(1.0)
    out["interaction_bonus"] = out["interaction"].map(interaction_bonus_map).fillna(1.0)
    out["regime_bonus"] = out["regime"].map(regime_bonus_map).fillna(1.0)
    out["family_penalty"] = out["family"].map(lambda x: 1.0 / math.sqrt(float(family_counts.get(x, 1))))
    out["wave_penalty"] = out["wave"].map(lambda x: 1.0 / math.sqrt(float(wave_counts.get(x, 1))))
    out["signature_penalty"] = 1.0 / np.sqrt(out["signature_size"].clip(lower=1.0))

    out["selector_score"] = (
        3.00 * out["density_score"]
        + 2.00 * out["non_na_ratio"]
        + 1.50 * out["unique_ratio"]
        + 1.25 * out["wave_bonus"]
        + 1.00 * out["transform_bonus"]
        + 0.90 * out["interaction_bonus"]
        + 0.80 * out["regime_bonus"]
        + 2.10 * out["family_penalty"]
        + 1.20 * out["wave_penalty"]
        + 1.60 * out["signature_penalty"]
    )
    return out.sort_values(["selector_score", "family", "wave", "alpha"], ascending=[False, True, True, True]).reset_index(drop=True)


def _signature_col(df: pd.DataFrame) -> pd.Series:
    cols = ["family", "left", "modulator", "transform", "regime", "interaction", "lag"]
    return df[cols].astype(str).agg("|".join, axis=1)


def _apply_structural_caps(scored: pd.DataFrame, cfg: SelectorConfig) -> tuple[pd.DataFrame, dict]:
    counters = {
        "input_manifest_rows": int(len(scored)),
        "removed_low_non_na_ratio": 0,
        "removed_family_cap": 0,
        "removed_wave_cap": 0,
        "removed_signature_cap": 0,
        "removed_shortlist_tail": 0,
        "kept_after_structural_caps": 0,
    }

    work = scored.copy()
    work["non_na_ratio"] = pd.to_numeric(work["non_na_ratio"], errors="coerce")
    low_non_na_mask = work["non_na_ratio"].fillna(0.0) < float(cfg.min_non_na_ratio)
    counters["removed_low_non_na_ratio"] = int(low_non_na_mask.sum())
    work = work.loc[~low_non_na_mask].copy()
    if work.empty:
        counters["kept_after_structural_caps"] = 0
        return work, counters

    work["family_rank"] = work.groupby("family", sort=False).cumcount() + 1
    fam_mask = work["family_rank"] <= int(cfg.max_per_family)
    counters["removed_family_cap"] = int((~fam_mask).sum())
    work = work.loc[fam_mask].copy()

    work["wave_rank"] = work.groupby("wave", sort=False).cumcount() + 1
    wave_mask = work["wave_rank"] <= int(cfg.max_per_wave)
    counters["removed_wave_cap"] = int((~wave_mask).sum())
    work = work.loc[wave_mask].copy()

    work["signature"] = _signature_col(work)
    work["signature_rank"] = work.groupby("signature", sort=False).cumcount() + 1
    sig_mask = work["signature_rank"] <= int(cfg.max_per_signature)
    counters["removed_signature_cap"] = int((~sig_mask).sum())
    work = work.loc[sig_mask].copy()

    if len(work) > int(cfg.shortlist_target):
        counters["removed_shortlist_tail"] = int(len(work) - int(cfg.shortlist_target))
        work = work.head(int(cfg.shortlist_target)).copy()

    counters["kept_after_structural_caps"] = int(len(work))
    return work.reset_index(drop=True), counters


def _find_alpha_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("alpha_")]


def _corr_prune(shortlist: pd.DataFrame, alpha_df: pd.DataFrame, cfg: SelectorConfig) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    counters = {
        "corr_available": 0,
        "corr_input_candidates": int(len(shortlist)),
        "corr_removed_high_corr": 0,
        "corr_kept": int(len(shortlist)),
    }
    dropped_rows: List[dict] = []

    if alpha_df.empty or shortlist.empty:
        return shortlist.copy(), counters, pd.DataFrame(columns=["alpha", "dropped_vs", "abs_corr", "reason"])

    cols_present = [a for a in shortlist["alpha"].tolist() if a in alpha_df.columns]
    if len(cols_present) < 2:
        return shortlist.copy(), counters, pd.DataFrame(columns=["alpha", "dropped_vs", "abs_corr", "reason"])

    counters["corr_available"] = 1
    top_cols = cols_present[: int(cfg.corr_screen_top_n)]
    matrix = alpha_df[top_cols].copy()
    for col in top_cols:
        matrix[col] = pd.to_numeric(matrix[col], errors="coerce")

    non_na_ratio = matrix.notna().mean(axis=0)
    valid_cols = [c for c in top_cols if float(non_na_ratio.get(c, 0.0)) >= float(cfg.min_non_na_ratio)]
    if len(valid_cols) < 2:
        return shortlist[shortlist["alpha"].isin(valid_cols)].copy(), counters, pd.DataFrame(columns=["alpha", "dropped_vs", "abs_corr", "reason"])

    corr = matrix[valid_cols].corr(method="spearman", min_periods=250).abs()
    keep: List[str] = []
    removed: set[str] = set()

    for col in valid_cols:
        if col in removed:
            continue
        keep.append(col)
        for other in valid_cols:
            if other == col or other in removed:
                continue
            val = corr.loc[col, other]
            if pd.notna(val) and float(val) >= float(cfg.corr_threshold):
                removed.add(other)
                dropped_rows.append({
                    "alpha": other,
                    "dropped_vs": col,
                    "abs_corr": float(val),
                    "reason": "high_corr_prune",
                })

    counters["corr_removed_high_corr"] = int(len(removed))
    keep_set = set(keep)
    final_shortlist = shortlist[shortlist["alpha"].isin(keep_set)].copy().reset_index(drop=True)
    counters["corr_kept"] = int(len(final_shortlist))
    dropped_df = pd.DataFrame(dropped_rows)
    if len(dropped_df):
        dropped_df = dropped_df.sort_values(["abs_corr", "alpha"], ascending=[False, True]).reset_index(drop=True)
    else:
        dropped_df = pd.DataFrame(columns=["alpha", "dropped_vs", "abs_corr", "reason"])
    return final_shortlist, counters, dropped_df


def _family_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    g = manifest.groupby("family", dropna=False)
    out = g.agg(
        count=("alpha", "size"),
        avg_non_na=("non_na", "mean"),
        avg_nan_ratio=("nan_ratio", "mean"),
        avg_unique=("unique", "mean"),
        waves=("wave", lambda s: "|".join(sorted(pd.Series(s).astype(str).unique().tolist()))),
    ).reset_index()
    return out.sort_values(["count", "family"], ascending=[False, True]).reset_index(drop=True)


def _wave_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    g = manifest.groupby("wave", dropna=False)
    out = g.agg(
        count=("alpha", "size"),
        families=("family", lambda s: int(pd.Series(s).nunique())),
        avg_non_na=("non_na", "mean"),
        avg_nan_ratio=("nan_ratio", "mean"),
        avg_unique=("unique", "mean"),
    ).reset_index()
    return out.sort_values(["count", "wave"], ascending=[False, True]).reset_index(drop=True)


def _family_wave_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    out = manifest.groupby(["family", "wave"], dropna=False).agg(
        count=("alpha", "size"),
        avg_non_na=("non_na", "mean"),
        avg_nan_ratio=("nan_ratio", "mean"),
    ).reset_index()
    return out.sort_values(["count", "family", "wave"], ascending=[False, True, True]).reset_index(drop=True)


def _drop_reason_summary(dropped: pd.DataFrame) -> pd.DataFrame:
    if dropped.empty or "drop_reason" not in dropped.columns:
        return pd.DataFrame(columns=["drop_reason", "count"])
    out = dropped.groupby("drop_reason", dropna=False).size().rename("count").reset_index()
    return out.sort_values(["count", "drop_reason"], ascending=[False, True]).reset_index(drop=True)


def _format_json_for_log(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def main() -> int:
    _must_exist(META_PATH)
    _must_exist(MANIFEST_PATH)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    meta = _load_json(META_PATH)
    manifest = _load_csv(MANIFEST_PATH)
    dropped = _load_csv(DROPPED_PATH) if DROPPED_PATH.exists() else pd.DataFrame()
    manifest = _coerce_numeric(manifest, ["non_na", "nan_ratio", "unique", "lag"])

    required_cols = ["alpha", "family", "wave", "left", "modulator", "transform", "regime", "interaction", "lag", "non_na", "nan_ratio", "unique"]
    missing = [c for c in required_cols if c not in manifest.columns]
    if missing:
        raise RuntimeError(f"Manifest missing required columns: {missing}")

    cfg = SelectorConfig()
    scored = _structural_score(manifest)
    structurally_capped, structural_counters = _apply_structural_caps(scored, cfg)

    alpha_df = pd.DataFrame()
    parquet_loaded = 0
    alpha_cols_in_parquet = 0
    if PARQUET_PATH.exists():
        try:
            parquet_df = pd.read_parquet(PARQUET_PATH)
            alpha_cols = _find_alpha_columns(parquet_df)
            if alpha_cols:
                alpha_df = parquet_df[alpha_cols].copy()
                parquet_loaded = 1
                alpha_cols_in_parquet = int(len(alpha_cols))
        except Exception as exc:
            print(f"[WARN] Failed to read parquet for correlation diagnostics: {exc}")

    final_shortlist, corr_counters, corr_dropped = _corr_prune(structurally_capped, alpha_df, cfg)
    final_shortlist = final_shortlist.copy()
    final_shortlist["shortlist_rank"] = np.arange(1, len(final_shortlist) + 1)

    family_summary = _family_summary(scored)
    wave_summary = _wave_summary(scored)
    family_wave_summary = _family_wave_summary(scored)
    drop_reason_summary = _drop_reason_summary(dropped)

    selector_drop_rows: List[dict] = []
    kept_set = set(final_shortlist["alpha"].tolist())
    capped_set = set(structurally_capped["alpha"].tolist())
    for _, row in scored.iterrows():
        alpha = str(row["alpha"])
        if alpha in kept_set:
            continue
        reason = "not_in_shortlist"
        if alpha not in capped_set:
            reason = "structural_cap_or_low_non_na"
        selector_drop_rows.append({
            "alpha": alpha,
            "family": row["family"],
            "wave": row["wave"],
            "selector_score": float(row["selector_score"]),
            "reason": reason,
        })
    selector_dropped = pd.DataFrame(selector_drop_rows)
    if len(corr_dropped):
        merged = selector_dropped.merge(corr_dropped[["alpha", "reason", "dropped_vs", "abs_corr"]], on="alpha", how="left", suffixes=("", "_corr"))
        merged["reason"] = merged["reason_corr"].fillna(merged["reason"])
        merged = merged.drop(columns=[c for c in ["reason_corr"] if c in merged.columns])
        selector_dropped = merged

    family_summary.to_csv(OUT_DIR / "alpha_family_summary.csv", index=False)
    wave_summary.to_csv(OUT_DIR / "alpha_wave_summary.csv", index=False)
    family_wave_summary.to_csv(OUT_DIR / "alpha_family_wave_summary.csv", index=False)
    drop_reason_summary.to_csv(OUT_DIR / "alpha_drop_reason_summary.csv", index=False)
    scored.to_csv(OUT_DIR / "alpha_manifest_scored.csv", index=False)
    structurally_capped.to_csv(OUT_DIR / "alpha_structural_shortlist.csv", index=False)
    final_shortlist.to_csv(OUT_DIR / "alpha_candidate_shortlist.csv", index=False)
    selector_dropped.to_csv(OUT_DIR / "alpha_selector_dropped.csv", index=False)
    corr_dropped.to_csv(OUT_DIR / "alpha_corr_pruned.csv", index=False)

    debug = {
        "meta_rows": int(meta.get("rows", 0)),
        "meta_alpha_count": int(meta.get("alpha_count", 0)),
        "meta_recipe_count": int(meta.get("recipe_count", 0)),
        "manifest_rows": int(len(manifest)),
        "parquet_loaded": int(parquet_loaded),
        "parquet_alpha_cols": int(alpha_cols_in_parquet),
        "selector_config": {
            "shortlist_target": int(cfg.shortlist_target),
            "max_per_family": int(cfg.max_per_family),
            "max_per_wave": int(cfg.max_per_wave),
            "max_per_signature": int(cfg.max_per_signature),
            "corr_screen_top_n": int(cfg.corr_screen_top_n),
            "corr_threshold": float(cfg.corr_threshold),
            "min_non_na_ratio": float(cfg.min_non_na_ratio),
        },
        "structural_counters": structural_counters,
        "corr_counters": corr_counters,
        "final_shortlist_count": int(len(final_shortlist)),
        "final_shortlist_by_family": final_shortlist["family"].value_counts().to_dict() if len(final_shortlist) else {},
        "final_shortlist_by_wave": final_shortlist["wave"].value_counts().to_dict() if len(final_shortlist) else {},
    }
    (OUT_DIR / "alpha_debug_counters.json").write_text(_format_json_for_log(debug), encoding="utf-8")

    print(f"[INPUT] meta={META_PATH}")
    print(f"[INPUT] manifest={MANIFEST_PATH}")
    print(f"[INPUT] dropped={DROPPED_PATH}")
    print(f"[INPUT] parquet_exists={PARQUET_PATH.exists()}")
    print(f"[META] rows={meta.get('rows', 0)} alpha_count={meta.get('alpha_count', 0)} recipe_count={meta.get('recipe_count', 0)}")
    print(f"[MANIFEST] rows={len(manifest)}")
    print(f"[SELECTOR] structural_kept={structural_counters['kept_after_structural_caps']} final_shortlist={len(final_shortlist)}")
    print(f"[SELECTOR] corr_available={corr_counters['corr_available']} corr_removed_high_corr={corr_counters['corr_removed_high_corr']}")
    print("[SELECTOR][STRUCTURAL_COUNTERS]")
    print(_format_json_for_log(structural_counters))
    print("[SELECTOR][CORR_COUNTERS]")
    print(_format_json_for_log(corr_counters))

    print("[TOP_FAMILIES]")
    print(family_summary.head(12).to_string(index=False))
    print("[TOP_WAVES]")
    print(wave_summary.head(12).to_string(index=False))
    print("[SHORTLIST_HEAD]")
    preview_cols = [c for c in ["shortlist_rank", "alpha", "family", "wave", "transform", "interaction", "regime", "selector_score"] if c in final_shortlist.columns]
    if len(final_shortlist):
        print(final_shortlist[preview_cols].head(30).to_string(index=False))
    else:
        print("(empty shortlist)")

    print(f"[ARTIFACT] {OUT_DIR / 'alpha_family_summary.csv'}")
    print(f"[ARTIFACT] {OUT_DIR / 'alpha_wave_summary.csv'}")
    print(f"[ARTIFACT] {OUT_DIR / 'alpha_family_wave_summary.csv'}")
    print(f"[ARTIFACT] {OUT_DIR / 'alpha_drop_reason_summary.csv'}")
    print(f"[ARTIFACT] {OUT_DIR / 'alpha_manifest_scored.csv'}")
    print(f"[ARTIFACT] {OUT_DIR / 'alpha_structural_shortlist.csv'}")
    print(f"[ARTIFACT] {OUT_DIR / 'alpha_candidate_shortlist.csv'}")
    print(f"[ARTIFACT] {OUT_DIR / 'alpha_selector_dropped.csv'}")
    print(f"[ARTIFACT] {OUT_DIR / 'alpha_corr_pruned.csv'}")
    print(f"[ARTIFACT] {OUT_DIR / 'alpha_debug_counters.json'}")
    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 1
    finally:
        _safe_input("Press Enter to exit")
    raise SystemExit(rc)