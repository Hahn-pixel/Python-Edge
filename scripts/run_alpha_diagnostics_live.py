from __future__ import annotations

import json
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
for p in [ROOT, SRC_DIR]:
    ps = str(p)
    if ps not in sys.path:
        sys.path.insert(0, ps)

PAUSE_ON_EXIT = str(os.getenv("PAUSE_ON_EXIT", "auto")).strip().lower()

LIVE_ALPHA_DIR = Path(os.getenv("LIVE_ALPHA_DIR", "artifacts/live_alpha"))
LIVE_ALPHA_SNAPSHOT_FILE = Path(os.getenv("LIVE_ALPHA_SNAPSHOT_FILE", str(LIVE_ALPHA_DIR / "live_alpha_snapshot.parquet")))
LIVE_ALPHA_MANIFEST_FILE = Path(os.getenv("LIVE_ALPHA_MANIFEST_FILE", str(LIVE_ALPHA_DIR / "live_alpha_manifest.csv")))
LIVE_ALPHA_META_FILE = Path(os.getenv("LIVE_ALPHA_META_FILE", str(LIVE_ALPHA_DIR / "live_alpha_meta.json")))
LIVE_ALPHA_DIAG_DIR = Path(os.getenv("LIVE_ALPHA_DIAG_DIR", str(LIVE_ALPHA_DIR / "diagnostics")))

TARGET_COL = str(os.getenv("LIVE_ALPHA_TARGET_COL", "target_fwd_ret_1d")).strip() or "target_fwd_ret_1d"
DATE_COL_CANDIDATES = tuple(
    x.strip() for x in str(os.getenv("LIVE_ALPHA_DATE_COL_CANDIDATES", "date|trade_date|as_of_date|session_date")).split("|") if x.strip()
)
SYMBOL_COL_CANDIDATES = tuple(
    x.strip() for x in str(os.getenv("LIVE_ALPHA_SYMBOL_COL_CANDIDATES", "symbol|ticker")).split("|") if x.strip()
)
MIN_CROSS_SECTION = int(os.getenv("LIVE_ALPHA_DIAG_MIN_CROSS_SECTION", "20"))
MIN_IC_DAYS = int(os.getenv("LIVE_ALPHA_DIAG_MIN_IC_DAYS", "20"))
TOP_N_PRINT = int(os.getenv("LIVE_ALPHA_DIAG_TOP_N_PRINT", "20"))



def _should_pause() -> bool:
    if PAUSE_ON_EXIT in {"0", "false", "no", "off"}:
        return False
    if PAUSE_ON_EXIT in {"1", "true", "yes", "on"}:
        return True
    stdin_obj = getattr(sys, "stdin", None)
    stdout_obj = getattr(sys, "stdout", None)
    return bool(stdin_obj and stdout_obj and hasattr(stdin_obj, "isatty") and hasattr(stdout_obj, "isatty") and stdin_obj.isatty() and stdout_obj.isatty())



def _safe_exit(code: int) -> None:
    if _should_pause():
        try:
            print("")
            print(f"[EXIT] code={code}")
            input("Press Enter to exit...")
        except Exception:
            pass
    raise SystemExit(code)



def _pick_existing(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path



def _find_column(df: pd.DataFrame, candidates: Tuple[str, ...], label: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise RuntimeError(f"Could not find {label} column. Tried: {list(candidates)}")



def _safe_spearman(x: pd.Series, y: pd.Series) -> float:
    pair = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(pair) < max(3, MIN_CROSS_SECTION):
        return float("nan")
    if pair["x"].nunique() <= 1 or pair["y"].nunique() <= 1:
        return float("nan")
    x_rank = pair["x"].rank(method="average")
    y_rank = pair["y"].rank(method="average")
    corr = x_rank.corr(y_rank)
    return float(corr) if pd.notna(corr) else float("nan")



def _classify_family(alpha_name: str, manifest_family: str) -> Tuple[str, str]:
    fam = str(manifest_family or "").strip()
    if fam:
        group = "interaction" if fam.startswith("interaction_") else "base"
        return fam, group
    if "__x_" in alpha_name:
        suffix = alpha_name.split("__x_", 1)[1]
        return f"interaction_{suffix}", "interaction"
    return "unclassified", "base"



def _load_manifest_family_map(manifest_path: Path) -> Dict[str, str]:
    if not manifest_path.exists():
        return {}
    manifest = pd.read_csv(manifest_path)
    if "alpha" not in manifest.columns:
        return {}
    family_col = "family" if "family" in manifest.columns else None
    if family_col is None:
        return {str(a): "" for a in manifest["alpha"].astype(str)}
    out: Dict[str, str] = {}
    for _, row in manifest.iterrows():
        alpha = str(row.get("alpha", "") or "").strip()
        if not alpha:
            continue
        out[alpha] = str(row.get(family_col, "") or "").strip()
    return out



def _compute_daily_ic(frame: pd.DataFrame, date_col: str, alpha_col: str, target_col: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    grouped = frame[[date_col, alpha_col, target_col]].copy()
    for date_value, sub in grouped.groupby(date_col, sort=True):
        alpha_series = pd.to_numeric(sub[alpha_col], errors="coerce")
        target_series = pd.to_numeric(sub[target_col], errors="coerce")
        valid = pd.DataFrame({"alpha": alpha_series, "target": target_series}).dropna()
        non_na = int(len(valid))
        ic = _safe_spearman(valid["alpha"], valid["target"])
        rows.append(
            {
                "date": pd.Timestamp(date_value).strftime("%Y-%m-%d") if pd.notna(date_value) else "",
                "ic": ic,
                "non_na": non_na,
                "alpha_unique": int(valid["alpha"].nunique()) if non_na else 0,
                "target_unique": int(valid["target"].nunique()) if non_na else 0,
            }
        )
    return pd.DataFrame(rows)



def _summarize_alpha(frame: pd.DataFrame, date_col: str, symbol_col: str, alpha_col: str, target_col: str, family: str, family_group: str) -> Tuple[Dict[str, object], pd.DataFrame]:
    daily_ic = _compute_daily_ic(frame, date_col, alpha_col, target_col)
    valid_ic = pd.to_numeric(daily_ic["ic"], errors="coerce").dropna()
    ic_days = int(len(valid_ic))
    mean_ic = float(valid_ic.mean()) if ic_days else float("nan")
    std_ic = float(valid_ic.std(ddof=1)) if ic_days >= 2 else float("nan")
    ic_ir = float(mean_ic / std_ic) if ic_days >= 2 and pd.notna(std_ic) and abs(std_ic) > 1e-12 else float("nan")
    sign_stability = float((valid_ic > 0.0).mean()) if ic_days else float("nan")

    alpha_series = pd.to_numeric(frame[alpha_col], errors="coerce")
    target_series = pd.to_numeric(frame[target_col], errors="coerce")
    valid_pair = pd.DataFrame({"alpha": alpha_series, "target": target_series}).dropna()
    coverage = float(len(valid_pair) / max(1, len(frame)))
    alpha_std = float(valid_pair["alpha"].std(ddof=1)) if len(valid_pair) >= 2 else float("nan")

    summary = {
        "alpha": alpha_col,
        "family": family,
        "family_group": family_group,
        "rows_total": int(len(frame)),
        "rows_non_na": int(len(valid_pair)),
        "coverage": coverage,
        "unique_values": int(valid_pair["alpha"].nunique()) if len(valid_pair) else 0,
        "alpha_std": alpha_std,
        "ic_days": ic_days,
        "mean_ic": mean_ic,
        "mean_abs_ic": float(valid_ic.abs().mean()) if ic_days else float("nan"),
        "ic_std": std_ic,
        "ic_ir": ic_ir,
        "sign_stability": sign_stability,
        "positive_ic_days": int((valid_ic > 0.0).sum()) if ic_days else 0,
        "negative_ic_days": int((valid_ic < 0.0).sum()) if ic_days else 0,
        "eligible_for_ranking": int(ic_days >= MIN_IC_DAYS),
    }
    return summary, daily_ic



def main() -> int:
    snapshot_path = _pick_existing(LIVE_ALPHA_SNAPSHOT_FILE, "LIVE_ALPHA_SNAPSHOT_FILE")
    manifest_path = _pick_existing(LIVE_ALPHA_MANIFEST_FILE, "LIVE_ALPHA_MANIFEST_FILE")
    meta_path = _pick_existing(LIVE_ALPHA_META_FILE, "LIVE_ALPHA_META_FILE")

    LIVE_ALPHA_DIAG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[CFG] snapshot={snapshot_path}")
    print(f"[CFG] manifest={manifest_path}")
    print(f"[CFG] meta={meta_path}")
    print(f"[CFG] output_dir={LIVE_ALPHA_DIAG_DIR}")
    print(f"[CFG] target_col={TARGET_COL} min_cross_section={MIN_CROSS_SECTION} min_ic_days={MIN_IC_DAYS}")

    frame = pd.read_parquet(snapshot_path)
    if frame.empty:
        raise RuntimeError("live alpha snapshot is empty")

    date_col = _find_column(frame, DATE_COL_CANDIDATES, "date")
    symbol_col = _find_column(frame, SYMBOL_COL_CANDIDATES, "symbol")
    if TARGET_COL not in frame.columns:
        raise RuntimeError(f"Target column missing from snapshot: {TARGET_COL}")

    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    frame[symbol_col] = frame[symbol_col].astype(str).str.upper()
    frame = frame.sort_values([date_col, symbol_col]).reset_index(drop=True)

    family_map = _load_manifest_family_map(manifest_path)
    alpha_cols = [c for c in frame.columns if str(c).startswith("alpha_")]
    if not alpha_cols:
        raise RuntimeError("No alpha columns found in live alpha snapshot")

    print(f"[DATA] rows={len(frame)} symbols={frame[symbol_col].nunique()} dates={frame[date_col].nunique()} alpha_cols={len(alpha_cols)}")

    alpha_rows: List[Dict[str, object]] = []
    daily_ic_frames: List[pd.DataFrame] = []

    for idx, alpha_col in enumerate(alpha_cols, start=1):
        family, family_group = _classify_family(alpha_col, family_map.get(alpha_col, ""))
        summary, daily_ic = _summarize_alpha(frame, date_col, symbol_col, alpha_col, TARGET_COL, family, family_group)
        alpha_rows.append(summary)
        if not daily_ic.empty:
            daily_ic = daily_ic.copy()
            daily_ic["alpha"] = alpha_col
            daily_ic["family"] = family
            daily_ic["family_group"] = family_group
            daily_ic_frames.append(daily_ic)
        if idx % 100 == 0 or idx == len(alpha_cols):
            print(f"[ALPHA] processed={idx}/{len(alpha_cols)}")

    alpha_diag = pd.DataFrame(alpha_rows)
    alpha_diag = alpha_diag.sort_values(["eligible_for_ranking", "mean_ic", "ic_ir", "coverage", "alpha"], ascending=[False, False, False, False, True]).reset_index(drop=True)

    daily_ic_all = pd.concat(daily_ic_frames, ignore_index=True) if daily_ic_frames else pd.DataFrame(columns=["date", "ic", "non_na", "alpha_unique", "target_unique", "alpha", "family", "family_group"])

    family_diag = (
        alpha_diag.groupby(["family", "family_group"], dropna=False)
        .agg(
            alpha_count=("alpha", "count"),
            eligible_count=("eligible_for_ranking", "sum"),
            mean_ic=("mean_ic", "mean"),
            mean_abs_ic=("mean_abs_ic", "mean"),
            mean_ic_ir=("ic_ir", "mean"),
            mean_sign_stability=("sign_stability", "mean"),
            mean_coverage=("coverage", "mean"),
        )
        .reset_index()
        .sort_values(["mean_ic", "mean_ic_ir", "alpha_count", "family"], ascending=[False, False, False, True])
        .reset_index(drop=True)
    )

    interaction_family_diag = family_diag.loc[family_diag["family_group"] == "interaction"].copy().reset_index(drop=True)
    base_family_diag = family_diag.loc[family_diag["family_group"] == "base"].copy().reset_index(drop=True)

    top_alpha_path = LIVE_ALPHA_DIAG_DIR / "alpha_diagnostics_live.csv"
    family_path = LIVE_ALPHA_DIAG_DIR / "alpha_family_diagnostics_live.csv"
    interaction_family_path = LIVE_ALPHA_DIAG_DIR / "alpha_interaction_family_diagnostics_live.csv"
    daily_ic_path = LIVE_ALPHA_DIAG_DIR / "alpha_daily_ic_live.csv"
    summary_path = LIVE_ALPHA_DIAG_DIR / "alpha_diagnostics_live_summary.json"

    alpha_diag.to_csv(top_alpha_path, index=False)
    family_diag.to_csv(family_path, index=False)
    interaction_family_diag.to_csv(interaction_family_path, index=False)
    daily_ic_all.to_csv(daily_ic_path, index=False)

    with open(meta_path, "r", encoding="utf-8") as f:
        live_meta = json.load(f)

    summary = {
        "snapshot_path": str(snapshot_path),
        "manifest_path": str(manifest_path),
        "target_col": TARGET_COL,
        "rows": int(len(frame)),
        "symbols": int(frame[symbol_col].nunique()),
        "dates": int(frame[date_col].nunique()),
        "alpha_cols": int(len(alpha_cols)),
        "eligible_alpha_count": int(alpha_diag["eligible_for_ranking"].sum()),
        "top_alpha_mean_ic": alpha_diag[["alpha", "family", "mean_ic", "ic_ir", "sign_stability"]].head(TOP_N_PRINT).to_dict(orient="records"),
        "top_family_mean_ic": family_diag.head(TOP_N_PRINT).to_dict(orient="records"),
        "top_interaction_family_mean_ic": interaction_family_diag.head(TOP_N_PRINT).to_dict(orient="records"),
        "top_base_family_mean_ic": base_family_diag.head(TOP_N_PRINT).to_dict(orient="records"),
        "live_alpha_meta": live_meta,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] alpha_diag={top_alpha_path}")
    print(f"[OK] family_diag={family_path}")
    print(f"[OK] interaction_family_diag={interaction_family_path}")
    print(f"[OK] daily_ic={daily_ic_path}")
    print(f"[OK] summary={summary_path}")

    print("[TOP_ALPHA]")
    print(alpha_diag[["alpha", "family", "mean_ic", "ic_ir", "sign_stability", "coverage"]].head(TOP_N_PRINT).to_string(index=False))
    print("[TOP_FAMILY]")
    print(family_diag[["family", "family_group", "alpha_count", "eligible_count", "mean_ic", "mean_ic_ir"]].head(TOP_N_PRINT).to_string(index=False))

    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 1
    _safe_exit(rc)
