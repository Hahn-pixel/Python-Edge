from __future__ import annotations

import json
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
for p in [ROOT, SRC_DIR]:
    ps = str(p)
    if ps not in sys.path:
        sys.path.insert(0, ps)

try:
    from python_edge.portfolio.holding_inertia import apply_holding_inertia
    HAS_HOLDING_INERTIA = True
except Exception:
    HAS_HOLDING_INERTIA = False

    def apply_holding_inertia(df: pd.DataFrame, enter_pct: float = 0.10, exit_pct: float = 0.22) -> pd.DataFrame:
        out = df.copy().sort_values(["date", "symbol"]).reset_index(drop=True)
        out["rank_pct"] = out.groupby("date", sort=False)["score"].rank(method="average", pct=True)
        out["side"] = 0.0
        prev_side_by_symbol: Dict[str, float] = {}
        for dt in sorted(out["date"].dropna().unique()):
            idx = out.index[out["date"] == dt]
            day = out.loc[idx, ["symbol", "rank_pct"]].copy()
            next_side = pd.Series(0.0, index=day.index, dtype="float64")
            for row_idx, row in day.iterrows():
                sym = str(row["symbol"])
                rp = float(row["rank_pct"])
                prev_side = float(prev_side_by_symbol.get(sym, 0.0))
                if rp >= (1.0 - enter_pct):
                    curr_side = 1.0
                elif rp <= enter_pct:
                    curr_side = -1.0
                elif prev_side > 0.0 and rp >= (1.0 - exit_pct):
                    curr_side = 1.0
                elif prev_side < 0.0 and rp <= exit_pct:
                    curr_side = -1.0
                else:
                    curr_side = 0.0
                next_side.loc[row_idx] = curr_side
                prev_side_by_symbol[sym] = curr_side
            out.loc[idx, "side"] = next_side.values
        return out

try:
    from python_edge.portfolio.turnover_control import cap_daily_turnover
    HAS_TURNOVER_CONTROL = True
except Exception:
    HAS_TURNOVER_CONTROL = False

    def cap_daily_turnover(df: pd.DataFrame, weight_col: str = "weight", max_daily_turnover: float = 0.20) -> pd.DataFrame:
        out = df.copy().sort_values(["symbol", "date"]).reset_index(drop=True)
        out["prev_weight"] = out.groupby("symbol", sort=False)[weight_col].shift(1).fillna(0.0)
        out["turnover"] = (pd.to_numeric(out[weight_col], errors="coerce") - pd.to_numeric(out["prev_weight"], errors="coerce")).abs()
        day_turn = out.groupby("date", sort=False)["turnover"].transform("sum")
        scale = np.where(day_turn > max_daily_turnover, max_daily_turnover / (day_turn + 1e-12), 1.0)
        out[weight_col] = pd.to_numeric(out[weight_col], errors="coerce") * scale
        out["turnover"] = pd.to_numeric(out["turnover"], errors="coerce") * scale
        out["dbg_turnover_control_fallback"] = 1
        return out

EPS = 1e-12

ALPHA_LIB_FILE = Path(os.getenv("ALPHA_LIB_FILE", "data/alpha_library_fs2_base/alpha_library_fs2_base.parquet"))
FEATURE_V2_FILE = Path(os.getenv("FEATURE_V2_FILE", "data/features/feature_matrix_v2.parquet"))
FEATURE_V2_MANIFEST = Path(os.getenv("FEATURE_V2_MANIFEST", "data/features/feature_matrix_v2_manifest.csv"))
FEATURE_V2_DIAG_CSV = Path(os.getenv("FEATURE_V2_DIAG_CSV", "data/features/feature_matrix_v2_diag.csv"))
TARGET_COL = str(os.getenv("TARGET_COL", "target_fwd_ret_1d")).strip()
OUT_DIR = Path(os.getenv("RESIDUAL_WF_OUT_DIR", "artifacts/residual_layer_fs2_interactions"))
PAUSE_ON_EXIT = str(os.getenv("PAUSE_ON_EXIT", "auto")).strip().lower()

TRAIN_DAYS = int(os.getenv("WF_TRAIN_DAYS", "252"))
TEST_DAYS = int(os.getenv("WF_TEST_DAYS", "63"))
STEP_DAYS = int(os.getenv("WF_STEP_DAYS", "63"))
PURGE_DAYS = int(os.getenv("WF_PURGE_DAYS", "5"))
EMBARGO_DAYS = int(os.getenv("WF_EMBARGO_DAYS", "5"))
MIN_TRAIN_ROWS = int(os.getenv("MIN_TRAIN_ROWS", "3000"))
MIN_TEST_ROWS = int(os.getenv("MIN_TEST_ROWS", "300"))
MIN_DAILY_IC_CS = int(os.getenv("MIN_DAILY_IC_CS", "20"))
MIN_REGIME_DAYS = int(os.getenv("MIN_REGIME_DAYS", "20"))
MIN_BASE_NON_NA = int(os.getenv("MIN_BASE_NON_NA", "500"))

ENTER_PCT = float(os.getenv("ENTER_PCT", "0.10"))
EXIT_PCT = float(os.getenv("EXIT_PCT", "0.22"))
WEIGHT_CAP = float(os.getenv("WEIGHT_CAP", "0.05"))
GROSS_TARGET = float(os.getenv("GROSS_TARGET", "0.85"))
MAX_DAILY_TURNOVER = float(os.getenv("MAX_DAILY_TURNOVER", "0.20"))
COST_BPS = float(os.getenv("COST_BPS", "8.0"))

SIGN_LOCK_IC_ABS = float(os.getenv("SIGN_LOCK_IC_ABS", "0.0100"))
SIGN_LOCK_POS_RATE = float(os.getenv("SIGN_LOCK_POS_RATE", "0.55"))
TAIL_TOP_PCT = float(os.getenv("TAIL_TOP_PCT", "0.10"))
TOPK_PRINT = int(os.getenv("TOPK_PRINT", "40"))

ALPHA_NAME_FILTER = str(os.getenv("ALPHA_NAME_FILTER", "")).strip()
ALPHA_LIMIT = int(os.getenv("ALPHA_LIMIT", "0"))
INTERACTION_BASE_LIMIT = int(os.getenv("INTERACTION_BASE_LIMIT", "20"))

ENABLE_VOL_REGIME = str(os.getenv("ENABLE_VOL_REGIME", "1")).strip().lower() not in {"0", "false", "no", "off"}
ENABLE_LIQ_REGIME = str(os.getenv("ENABLE_LIQ_REGIME", "1")).strip().lower() not in {"0", "false", "no", "off"}
ENABLE_TREND_REGIME = str(os.getenv("ENABLE_TREND_REGIME", "1")).strip().lower() not in {"0", "false", "no", "off"}
ENABLE_BREADTH_REGIME = str(os.getenv("ENABLE_BREADTH_REGIME", "1")).strip().lower() not in {"0", "false", "no", "off"}
ENABLE_INTERACTIONS = str(os.getenv("ENABLE_INTERACTIONS", "1")).strip().lower() not in {"0", "false", "no", "off"}
ENABLE_PLAIN_BASE = str(os.getenv("ENABLE_PLAIN_BASE", "1")).strip().lower() not in {"0", "false", "no", "off"}
AUTO_BUILD_FS2_IF_MISSING = str(os.getenv("AUTO_BUILD_FS2_IF_MISSING", "1")).strip().lower() not in {"0", "false", "no", "off"}
INCLUDE_Z_FEATURES = str(os.getenv("INCLUDE_Z_FEATURES", "1")).strip().lower() not in {"0", "false", "no", "off"}
EXCLUDE_TARGET_DERIVED = str(os.getenv("EXCLUDE_TARGET_DERIVED", "1")).strip().lower() not in {"0", "false", "no", "off"}
EXCLUDE_MARKET_WIDE_CONSTANTS = str(os.getenv("EXCLUDE_MARKET_WIDE_CONSTANTS", "1")).strip().lower() not in {"0", "false", "no", "off"}
FS2_ENABLE_RAW = str(os.getenv("FS2_ENABLE_RAW", "1")).strip().lower() not in {"0", "false", "no", "off"}
FS2_ENABLE_SIGNED_LOG = str(os.getenv("FS2_ENABLE_SIGNED_LOG", "1")).strip().lower() not in {"0", "false", "no", "off"}
FS2_ENABLE_EMA3 = str(os.getenv("FS2_ENABLE_EMA3", "1")).strip().lower() not in {"0", "false", "no", "off"}
FS2_ENABLE_LAG1 = str(os.getenv("FS2_ENABLE_LAG1", "1")).strip().lower() not in {"0", "false", "no", "off"}
FS2_ENABLE_TANH_Z = str(os.getenv("FS2_ENABLE_TANH_Z", "1")).strip().lower() not in {"0", "false", "no", "off"}


@dataclass(frozen=True)
class WFSplit:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass(frozen=True)
class ShellConfig:
    enter_pct: float
    exit_pct: float
    weight_cap: float
    gross_target: float
    max_daily_turnover: float


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


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("float64")


def _robust_zscore_series(s: pd.Series) -> pd.Series:
    x = _num(s)
    valid = x.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=x.index, dtype="float64")
    med = float(valid.median())
    mad = float((valid - med).abs().median())
    if mad > EPS:
        out = (x - med) / (1.4826 * mad)
    else:
        mean = float(valid.mean())
        std = float(valid.std(ddof=0))
        out = (x - mean) / (std + EPS)
    return out.replace([np.inf, -np.inf], np.nan)


def _safe_mean(values: Sequence[float]) -> float:
    vals = [float(x) for x in values if pd.notna(x)]
    return float(np.mean(vals)) if vals else float("nan")


def _safe_last(values: Sequence[float]) -> float:
    vals = [float(x) for x in values if pd.notna(x)]
    return float(vals[-1]) if vals else float("nan")


def _last_n_mean(values: Sequence[float], n: int) -> float:
    vals = [float(x) for x in values if pd.notna(x)]
    return float(np.mean(vals[-n:])) if vals else float("nan")


def _positive_fraction(values: Sequence[float]) -> float:
    vals = [float(x) for x in values if pd.notna(x)]
    return float(sum(1 for x in vals if x > 0.0) / len(vals)) if vals else float("nan")


def _daily_ic_series(frame: pd.DataFrame, factor_col: str, target_col: str, min_cs: int) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for dt, g in frame.groupby("date", sort=False):
        x = g[[factor_col, target_col]].dropna()
        if len(x) < min_cs:
            continue
        if x[factor_col].nunique(dropna=True) <= 1 or x[target_col].nunique(dropna=True) <= 1:
            continue
        ic = x[factor_col].corr(x[target_col], method="spearman")
        if pd.notna(ic):
            rows.append({"date": pd.Timestamp(dt).normalize(), "daily_ic": float(ic), "cross_section_n": int(len(x))})
    return pd.DataFrame(rows)


def _tail_spread_series(frame: pd.DataFrame, factor_col: str, target_col: str, top_pct: float) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for dt, g in frame.groupby("date", sort=False):
        x = g[[factor_col, target_col]].dropna().copy()
        if len(x) < max(MIN_DAILY_IC_CS, 10):
            continue
        x["rank_pct"] = x[factor_col].rank(method="average", pct=True)
        top = x.loc[x["rank_pct"] >= (1.0 - top_pct), target_col]
        bot = x.loc[x["rank_pct"] <= top_pct, target_col]
        if len(top) == 0 or len(bot) == 0:
            continue
        rows.append({"date": pd.Timestamp(dt).normalize(), "tail_spread": float(top.mean() - bot.mean())})
    return pd.DataFrame(rows)


def _target_col(df: pd.DataFrame) -> str:
    if TARGET_COL in df.columns:
        return TARGET_COL
    for c in ["target_fwd_ret_1d", "fwd_ret_1d", "ret_fwd_1d"]:
        if c in df.columns:
            return c
    raise RuntimeError("Missing target column")


def _signed_log(s: pd.Series) -> pd.Series:
    x = _num(s)
    return np.sign(x) * np.log1p(np.abs(x))


def _ema3_by_symbol(frame: pd.DataFrame, col: str) -> pd.Series:
    return frame.groupby("symbol", sort=False)[col].transform(lambda s: _num(s).ewm(span=3, adjust=False, min_periods=1).mean())


def _lag1_by_symbol(frame: pd.DataFrame, col: str) -> pd.Series:
    return frame.groupby("symbol", sort=False)[col].shift(1)


def _is_target_derived(source: str, formula: str, feature: str) -> bool:
    hay = " | ".join([str(source), str(formula), str(feature)]).lower()
    bad_terms = [
        TARGET_COL.lower(),
        "target",
        "rank_pct(target)",
        "cs_dispersion",
        "winner_loser",
        "top_tail_strength",
        "bottom_tail_strength",
    ]
    return any(term in hay for term in bad_terms)


def _is_market_wide_constant(cluster: str, feature: str) -> bool:
    feat = str(feature)
    if not EXCLUDE_MARKET_WIDE_CONSTANTS:
        return False
    if cluster == "cs_pressure" and any(x in feat for x in ["breadth", "dispersion", "target_rank", "winner_loser"]):
        return True
    return False


def _load_safe_fs2_features(feat_df: pd.DataFrame) -> pd.DataFrame:
    if FEATURE_V2_MANIFEST.exists():
        manifest = pd.read_csv(FEATURE_V2_MANIFEST).rename(columns={"name": "feature"})
    else:
        feature_cols = [c for c in feat_df.columns if c.startswith("fs2_") or c.startswith("z_fs2_")]
        manifest = pd.DataFrame({"feature": feature_cols, "cluster": "unknown", "source": "unknown", "formula": ""})
    diag = pd.read_csv(FEATURE_V2_DIAG_CSV) if FEATURE_V2_DIAG_CSV.exists() else pd.DataFrame(columns=["feature", "non_na", "nan_ratio", "unique", "keep"])
    work = manifest.merge(diag, on="feature", how="left", suffixes=("", "_diag"))
    for col, default in [("cluster", "unknown"), ("source", "unknown"), ("formula", ""), ("non_na", 0), ("nan_ratio", 1.0), ("unique", 0), ("keep", 0)]:
        if col not in work.columns:
            work[col] = default
    work["feature_present"] = work["feature"].astype(str).isin(set(feat_df.columns)).astype(int)
    work["is_z_feature"] = work["feature"].astype(str).str.startswith("z_").astype(int)
    work["target_derived"] = work.apply(lambda r: int(_is_target_derived(str(r["source"]), str(r["formula"]), str(r["feature"]))), axis=1)
    work["market_wide_constant"] = work.apply(lambda r: int(_is_market_wide_constant(str(r["cluster"]), str(r["feature"]))), axis=1)
    work["eligible"] = 1
    work.loc[work["feature_present"] != 1, "eligible"] = 0
    work.loc[pd.to_numeric(work["non_na"], errors="coerce").fillna(0) < MIN_BASE_NON_NA, "eligible"] = 0
    work.loc[pd.to_numeric(work["nan_ratio"], errors="coerce").fillna(1.0) > 0.995, "eligible"] = 0
    work.loc[pd.to_numeric(work["unique"], errors="coerce").fillna(0) < 5, "eligible"] = 0
    if not INCLUDE_Z_FEATURES:
        work.loc[work["is_z_feature"] == 1, "eligible"] = 0
    if EXCLUDE_TARGET_DERIVED:
        work.loc[work["target_derived"] == 1, "eligible"] = 0
    if EXCLUDE_MARKET_WIDE_CONSTANTS:
        work.loc[work["market_wide_constant"] == 1, "eligible"] = 0
    return work.loc[work["eligible"] == 1].copy().reset_index(drop=True)


def _build_fs2_alpha_library_from_features(feat_df: pd.DataFrame) -> pd.DataFrame:
    safe_df = _load_safe_fs2_features(feat_df)
    if safe_df.empty:
        raise RuntimeError("Alpha library missing and no safe fs2 features available for auto-build")
    tgt_col = _target_col(feat_df)
    base_cols = [c for c in ["date", "symbol", "open", "high", "low", "close", "volume", tgt_col] if c in feat_df.columns]
    out = feat_df[base_cols + [c for c in feat_df.columns if c.startswith("fs2_") or c.startswith("z_fs2_")]].copy()
    alpha_data: Dict[str, pd.Series] = {}
    for _, row in safe_df.iterrows():
        feature = str(row["feature"])
        family = feature.replace("z_", "") if feature.startswith("z_") else feature
        x = _num(feat_df[feature])
        if FS2_ENABLE_RAW:
            alpha_data[f"alpha_{family}__raw"] = x
        if FS2_ENABLE_SIGNED_LOG:
            alpha_data[f"alpha_{family}__signed_log"] = _signed_log(x)
        if FS2_ENABLE_EMA3:
            alpha_data[f"alpha_{family}__ema3"] = _ema3_by_symbol(feat_df.assign(_x=x), "_x")
        if FS2_ENABLE_LAG1:
            alpha_data[f"alpha_{family}__lag1"] = _lag1_by_symbol(feat_df.assign(_x=x), "_x")
        if FS2_ENABLE_TANH_Z and not feature.startswith("z_"):
            alpha_data[f"alpha_{family}__tanh_z"] = np.tanh(feat_df.groupby("date", sort=False)[feature].transform(_robust_zscore_series))
    if not alpha_data:
        raise RuntimeError("Auto-build from fs2 features produced zero alpha columns")
    alpha_df = pd.DataFrame(alpha_data, index=feat_df.index)
    out = pd.concat([out, alpha_df], axis=1).copy()
    return out


def _load_data() -> Tuple[pd.DataFrame, List[str]]:
    _must_exist(FEATURE_V2_FILE, "Feature v2 file")
    feat_df = pd.read_parquet(FEATURE_V2_FILE)
    if feat_df.empty:
        raise RuntimeError("Feature v2 file is empty")
    feat_df["date"] = pd.to_datetime(feat_df["date"]).dt.normalize()
    feat_df = feat_df.sort_values(["date", "symbol"]).reset_index(drop=True)

    if ALPHA_LIB_FILE.exists():
        alpha_df = pd.read_parquet(ALPHA_LIB_FILE)
        if alpha_df.empty:
            raise RuntimeError("Alpha library is empty")
        alpha_df["date"] = pd.to_datetime(alpha_df["date"]).dt.normalize()
        keep_feat_cols = [c for c in feat_df.columns if c.startswith("fs2_") or c.startswith("z_fs2_")]
        merge_cols = ["date", "symbol"] + [c for c in keep_feat_cols if c not in {"date", "symbol"}]
        merged = alpha_df.merge(feat_df[merge_cols].copy(), on=["date", "symbol"], how="left")
        data_source = str(ALPHA_LIB_FILE)
    else:
        if not AUTO_BUILD_FS2_IF_MISSING:
            raise FileNotFoundError(f"Alpha library not found: {ALPHA_LIB_FILE}")
        merged = _build_fs2_alpha_library_from_features(feat_df)
        data_source = "AUTO_BUILD_FROM_FEATURE_V2"

    alpha_cols = sorted([c for c in merged.columns if c.startswith("alpha_")])
    if ALPHA_NAME_FILTER:
        alpha_cols = [c for c in alpha_cols if ALPHA_NAME_FILTER.lower() in c.lower()]
    if ALPHA_LIMIT > 0:
        alpha_cols = alpha_cols[:ALPHA_LIMIT]
    if not alpha_cols:
        raise RuntimeError("No alpha columns remain after filtering")
    merged["__alpha_data_source__"] = data_source
    return merged.sort_values(["date", "symbol"]).reset_index(drop=True), alpha_cols


def _pick_regime_source(df: pd.DataFrame, candidates: Sequence[str]) -> Tuple[pd.Series, str]:
    for c in candidates:
        if c in df.columns:
            return _num(df[c]), c
    return pd.Series(np.nan, index=df.index, dtype="float64"), "none"


def _add_explicit_regime_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    out = df.copy()
    tgt_col = _target_col(out)
    vol_src, vol_name = _pick_regime_source(out, [
        "fs2_realized_vol_accel", "z_fs2_realized_vol_accel", "fs2_vol_of_vol_short", "z_fs2_vol_of_vol_short", "fs2_shock_reversion_ratio", "z_fs2_shock_reversion_ratio",
    ])
    liq_src, liq_name = _pick_regime_source(out, [
        "fs2_dollar_vol_accel_proxy", "z_fs2_dollar_vol_accel_proxy", "fs2_intraday_strength_liq_interaction", "z_fs2_intraday_strength_liq_interaction",
    ])
    trend_src, trend_name = _pick_regime_source(out, [
        "fs2_intraday_pressure_rs_mix", "z_fs2_intraday_pressure_rs_mix", "fs2_intraday_rs_proxy", "z_fs2_intraday_rs_proxy",
    ])
    breadth_src, breadth_name = _pick_regime_source(out, [
        "fs2_market_breadth_thrust_proxy", "z_fs2_market_breadth_thrust_proxy", "fs2_breadth_reversal_proxy", "z_fs2_breadth_reversal_proxy",
    ])

    regime_meta: List[Dict[str, object]] = []
    regime_cols: List[str] = []

    day_df = out[["date"]].drop_duplicates().sort_values("date").reset_index(drop=True)
    day_level = out.groupby("date", sort=False).agg(
        vol_proxy=(vol_name if vol_name != "none" else tgt_col, lambda s: float(_num(s).median())),
        liq_proxy=(liq_name if liq_name != "none" else tgt_col, lambda s: float(_num(s).median())),
        trend_proxy=(trend_name if trend_name != "none" else tgt_col, lambda s: float(_num(s).median())),
        breadth_proxy=(breadth_name if breadth_name != "none" else tgt_col, lambda s: float(_num(s).median())),
        target_abs=(tgt_col, lambda s: float(_num(s).abs().median())),
    ).reset_index()

    def _add_hi_lo(name: str, source_col: str, enabled: bool) -> None:
        if not enabled:
            return
        ser = _num(day_level[source_col])
        q_hi = float(ser.quantile(0.67)) if ser.notna().any() else float("nan")
        q_lo = float(ser.quantile(0.33)) if ser.notna().any() else float("nan")
        hi_col = f"regime_{name}_hi"
        lo_col = f"regime_{name}_lo"
        day_level[hi_col] = np.where(ser >= q_hi, 1.0, 0.0) if pd.notna(q_hi) else 0.0
        day_level[lo_col] = np.where(ser <= q_lo, 1.0, 0.0) if pd.notna(q_lo) else 0.0
        regime_cols.extend([hi_col, lo_col])
        regime_meta.append({"regime": hi_col, "kind": name, "source": source_col, "logic": "date-level >= q67"})
        regime_meta.append({"regime": lo_col, "kind": name, "source": source_col, "logic": "date-level <= q33"})

    _add_hi_lo("vol", "vol_proxy", ENABLE_VOL_REGIME)
    _add_hi_lo("liq", "liq_proxy", ENABLE_LIQ_REGIME)
    _add_hi_lo("trend", "trend_proxy", ENABLE_TREND_REGIME)
    _add_hi_lo("breadth", "breadth_proxy", ENABLE_BREADTH_REGIME)

    if ENABLE_TREND_REGIME:
        sign_col = "regime_trend_pos"
        day_level[sign_col] = np.where(_num(day_level["trend_proxy"]) >= 0.0, 1.0, 0.0)
        regime_cols.append(sign_col)
        regime_meta.append({"regime": sign_col, "kind": "trend", "source": "trend_proxy", "logic": "date-level >= 0"})

    day_level["regime_target_vol_hi"] = np.where(_num(day_level["target_abs"]) >= float(_num(day_level["target_abs"]).quantile(0.67)), 1.0, 0.0)
    regime_cols.append("regime_target_vol_hi")
    regime_meta.append({"regime": "regime_target_vol_hi", "kind": "target_abs", "source": tgt_col, "logic": "date-level abs(target) >= q67"})

    out = out.merge(day_level[["date"] + regime_cols], on="date", how="left")
    meta_df = pd.DataFrame(regime_meta)
    return out, regime_cols, meta_df


def build_walkforward_splits(dates: Sequence[pd.Timestamp]) -> List[WFSplit]:
    uniq = pd.Index(sorted(pd.to_datetime(pd.Series(dates)).dt.normalize().unique()))
    if len(uniq) < (TRAIN_DAYS + TEST_DAYS + PURGE_DAYS + EMBARGO_DAYS + 5):
        raise RuntimeError("Not enough dates for walkforward configuration")
    splits: List[WFSplit] = []
    fold_id = 1
    train_end_idx = TRAIN_DAYS - 1
    while True:
        test_start_idx = train_end_idx + 1 + PURGE_DAYS + EMBARGO_DAYS
        test_end_idx = test_start_idx + TEST_DAYS - 1
        if test_end_idx >= len(uniq):
            break
        train_start_idx = train_end_idx - TRAIN_DAYS + 1
        splits.append(WFSplit(fold_id=fold_id, train_start=uniq[train_start_idx], train_end=uniq[train_end_idx], test_start=uniq[test_start_idx], test_end=uniq[test_end_idx]))
        fold_id += 1
        train_end_idx += STEP_DAYS
    if not splits:
        raise RuntimeError("No walkforward splits generated")
    return splits


def _split_frame(df: pd.DataFrame, split: WFSplit) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df.loc[(df["date"] >= split.train_start) & (df["date"] <= split.train_end)].copy()
    test_df = df.loc[(df["date"] >= split.test_start) & (df["date"] <= split.test_end)].copy()
    return train_df, test_df


def _rank_base_alphas(train_df: pd.DataFrame, alpha_cols: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for alpha in alpha_cols:
        ser = _num(train_df[alpha])
        if int(ser.notna().sum()) < MIN_BASE_NON_NA:
            continue
        scored = train_df[["date", "symbol", TARGET_COL]].copy()
        scored["score_raw"] = ser
        scored["score"] = scored.groupby("date", sort=False)["score_raw"].transform(_robust_zscore_series)
        ic_df = _daily_ic_series(scored, "score", TARGET_COL, MIN_DAILY_IC_CS)
        tail_df = _tail_spread_series(scored, "score", TARGET_COL, TAIL_TOP_PCT)
        rows.append({
            "alpha": alpha,
            "train_mean_ic": float(ic_df["daily_ic"].mean()) if len(ic_df) else float("nan"),
            "train_abs_ic": float(ic_df["daily_ic"].abs().mean()) if len(ic_df) else float("nan"),
            "train_sign_stability": float((ic_df["daily_ic"] > 0.0).mean()) if len(ic_df) else float("nan"),
            "train_tail_mean": float(tail_df["tail_spread"].mean()) if len(tail_df) else float("nan"),
            "train_tail_positive": float((tail_df["tail_spread"] > 0.0).mean()) if len(tail_df) else float("nan"),
            "train_non_na": int(ser.notna().sum()),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out["rank_score"] = pd.to_numeric(out["train_abs_ic"], errors="coerce").fillna(0.0)
        out["rank_score"] += 0.25 * pd.to_numeric(out["train_tail_positive"], errors="coerce").fillna(0.0)
        out["rank_score"] += 0.10 * pd.to_numeric(out["train_tail_mean"], errors="coerce").fillna(0.0)
        out = out.sort_values(["rank_score", "train_non_na", "alpha"], ascending=[False, False, True]).reset_index(drop=True)
    return out


def _sign_decision(train_ic_df: pd.DataFrame) -> Dict[str, object]:
    if train_ic_df.empty:
        return {
            "train_mean_ic": float("nan"),
            "train_abs_mean_ic": float("nan"),
            "train_pos_rate": float("nan"),
            "train_sign_naive": 1.0,
            "train_sign_locked": 1.0,
            "sign_lock_triggered": 1,
            "sign_lock_reason": "no_train_ic",
        }
    mean_ic = float(train_ic_df["daily_ic"].mean())
    abs_mean_ic = float(train_ic_df["daily_ic"].abs().mean())
    pos_rate = float((train_ic_df["daily_ic"] > 0.0).mean())
    naive_sign = 1.0 if mean_ic >= 0.0 else -1.0
    lock = False
    reason = "ok"
    locked_sign = naive_sign
    if abs(mean_ic) < SIGN_LOCK_IC_ABS:
        lock = True
        reason = "weak_mean_ic"
    elif max(pos_rate, 1.0 - pos_rate) < SIGN_LOCK_POS_RATE:
        lock = True
        reason = "weak_pos_rate"
    if lock:
        if pos_rate > 0.50:
            locked_sign = 1.0
        elif pos_rate < 0.50:
            locked_sign = -1.0
        else:
            locked_sign = 1.0
    return {
        "train_mean_ic": mean_ic,
        "train_abs_mean_ic": abs_mean_ic,
        "train_pos_rate": pos_rate,
        "train_sign_naive": naive_sign,
        "train_sign_locked": locked_sign,
        "sign_lock_triggered": int(lock),
        "sign_lock_reason": reason,
    }


def _build_interaction_candidates(train_df: pd.DataFrame, test_df: pd.DataFrame, base_rank_df: pd.DataFrame, regime_cols: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if base_rank_df.empty:
        raise RuntimeError("No base alphas ranked for interaction engine")
    selected = base_rank_df.head(INTERACTION_BASE_LIMIT)["alpha"].astype(str).tolist() if INTERACTION_BASE_LIMIT > 0 else base_rank_df["alpha"].astype(str).tolist()
    rows: List[Dict[str, object]] = []
    train_out = train_df[["date", "symbol", TARGET_COL]].copy()
    test_out = test_df[["date", "symbol", TARGET_COL]].copy()

    for alpha in selected:
        if ENABLE_PLAIN_BASE:
            name = f"base::{alpha}"
            train_out[name] = train_df.groupby("date", sort=False)[alpha].transform(_robust_zscore_series)
            test_out[name] = test_df.groupby("date", sort=False)[alpha].transform(_robust_zscore_series)
            rows.append({"candidate": name, "alpha": alpha, "regime": "none", "kind": "base"})
        if not ENABLE_INTERACTIONS:
            continue
        for regime in regime_cols:
            tr_mask = _num(train_df[regime]).fillna(0.0)
            te_mask = _num(test_df[regime]).fillna(0.0)
            if float(tr_mask.groupby(train_df["date"], sort=False).first().mean()) <= 0.0:
                continue
            name_gate = f"gate::{alpha}::{regime}"
            train_out[name_gate] = train_df.groupby("date", sort=False)[alpha].transform(_robust_zscore_series) * tr_mask
            test_out[name_gate] = test_df.groupby("date", sort=False)[alpha].transform(_robust_zscore_series) * te_mask
            rows.append({"candidate": name_gate, "alpha": alpha, "regime": regime, "kind": "gate"})

            name_sign = f"sign::{alpha}::{regime}"
            train_out[name_sign] = train_df.groupby("date", sort=False)[alpha].transform(_robust_zscore_series) * (2.0 * tr_mask - 1.0)
            test_out[name_sign] = test_df.groupby("date", sort=False)[alpha].transform(_robust_zscore_series) * (2.0 * te_mask - 1.0)
            rows.append({"candidate": name_sign, "alpha": alpha, "regime": regime, "kind": "sign"})

    cand_meta = pd.DataFrame(rows).drop_duplicates(subset=["candidate"]).reset_index(drop=True)
    return train_out, test_out, cand_meta


def _fit_residual_beta(y: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    xtx = x.T @ x
    ridge = 1e-8 * np.eye(xtx.shape[0], dtype="float64")
    beta = np.linalg.solve(xtx + ridge, x.T @ y)
    resid = y - (x @ beta)
    return beta, {
        "fit_r2": float(1.0 - (np.var(resid) / (np.var(y) + EPS))),
        "beta_l2": float(np.sqrt(np.sum(beta ** 2))),
    }


def _build_components(train_cand_df: pd.DataFrame, test_cand_df: pd.DataFrame, candidate_rank_df: pd.DataFrame, max_components: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = candidate_rank_df.head(max_components)["candidate"].astype(str).tolist()
    train_work = train_cand_df[["date", "symbol", TARGET_COL] + ordered].copy()
    test_work = test_cand_df[["date", "symbol", TARGET_COL] + ordered].copy()
    coef_rows: List[Dict[str, object]] = []
    prior_components_train: List[str] = []
    prior_components_test: List[str] = []

    for idx, cand in enumerate(ordered, start=1):
        comp = f"comp{idx}"
        y_train = _num(train_work[cand])
        y_test = _num(test_work[cand])
        if idx == 1:
            train_work[f"{comp}_resid_raw"] = y_train
            test_work[f"{comp}_resid_raw"] = y_test
            coef_rows.append({"component": comp, "candidate": cand, "fit_r2": 0.0, "beta_l2": 0.0, "prior_count": 0})
        else:
            valid = pd.concat([y_train.rename("y"), train_work[prior_components_train].apply(_num)], axis=1).dropna()
            if len(valid) >= max(500, MIN_DAILY_IC_CS * 10):
                beta, info = _fit_residual_beta(valid["y"].to_numpy(dtype="float64"), valid[prior_components_train].to_numpy(dtype="float64"))
                train_all_x = train_work[prior_components_train].apply(_num).fillna(0.0).to_numpy(dtype="float64")
                test_all_x = test_work[prior_components_test].apply(_num).fillna(0.0).to_numpy(dtype="float64")
                train_work[f"{comp}_resid_raw"] = y_train.fillna(0.0).to_numpy(dtype="float64") - (train_all_x @ beta)
                test_work[f"{comp}_resid_raw"] = y_test.fillna(0.0).to_numpy(dtype="float64") - (test_all_x @ beta)
                coef_rows.append({"component": comp, "candidate": cand, **info, "prior_count": len(prior_components_train)})
            else:
                train_work[f"{comp}_resid_raw"] = y_train
                test_work[f"{comp}_resid_raw"] = y_test
                coef_rows.append({"component": comp, "candidate": cand, "fit_r2": 0.0, "beta_l2": 0.0, "prior_count": len(prior_components_train), "resid_fallback": 1})
        train_work[f"{comp}_resid"] = train_work.groupby("date", sort=False)[f"{comp}_resid_raw"].transform(_robust_zscore_series)
        test_work[f"{comp}_resid"] = test_work.groupby("date", sort=False)[f"{comp}_resid_raw"].transform(_robust_zscore_series)
        prior_components_train.append(f"{comp}_resid")
        prior_components_test.append(f"{comp}_resid")
    return train_work, test_work, pd.DataFrame(coef_rows)


def build_portfolio(scored_df: pd.DataFrame, shell: ShellConfig) -> pd.DataFrame:
    out = scored_df.copy().sort_values(["date", "symbol"]).reset_index(drop=True)
    out["score"] = _num(out["score"]).fillna(0.0)
    out = apply_holding_inertia(out, enter_pct=shell.enter_pct, exit_pct=shell.exit_pct)
    if "side" not in out.columns:
        raise RuntimeError("Holding inertia did not return side column")
    out["raw_strength"] = _num(out["score"]).abs().fillna(0.0)
    out.loc[_num(out["side"]) == 0.0, "raw_strength"] = 0.0
    pieces: List[pd.DataFrame] = []
    for _, g in out.groupby("date", sort=False):
        gg = g.copy()
        gg["weight"] = 0.0
        long_mask = _num(gg["side"]) > 0.0
        short_mask = _num(gg["side"]) < 0.0
        long_strength = float(gg.loc[long_mask, "raw_strength"].sum())
        short_strength = float(gg.loc[short_mask, "raw_strength"].sum())
        if long_strength > EPS:
            gg.loc[long_mask, "weight"] = 0.5 * gg.loc[long_mask, "raw_strength"] / long_strength
        if short_strength > EPS:
            gg.loc[short_mask, "weight"] = -0.5 * gg.loc[short_mask, "raw_strength"] / short_strength
        gg["weight"] = _num(gg["weight"]).clip(lower=-shell.weight_cap, upper=shell.weight_cap)
        gross = float(gg["weight"].abs().sum())
        if gross > EPS:
            gg["weight"] = gg["weight"] * (shell.gross_target / gross)
        pieces.append(gg)
    out = pd.concat(pieces, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
    out = cap_daily_turnover(out, weight_col="weight", max_daily_turnover=shell.max_daily_turnover)
    out["prev_weight"] = out.groupby("symbol", sort=False)["weight"].shift(1).fillna(0.0)
    out["turnover"] = (_num(out["weight"]) - _num(out["prev_weight"])).abs()
    out["gross_ret"] = _num(out["weight"]) * _num(out[TARGET_COL])
    out["cost_ret"] = _num(out["turnover"]) * (COST_BPS / 10000.0)
    out["net_ret"] = _num(out["gross_ret"]) - _num(out["cost_ret"])
    return out


def evaluate_portfolio(port_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    daily = port_df.groupby("date", sort=False, as_index=False).agg(
        gross_ret=("gross_ret", "sum"),
        cost_ret=("cost_ret", "sum"),
        net_ret=("net_ret", "sum"),
        turnover=("turnover", "sum"),
        gross_exposure=("weight", lambda s: float(_num(s).abs().sum())),
        names_active=("side", lambda s: int((_num(s).abs() > 0).sum())),
    )
    daily = daily.sort_values("date").reset_index(drop=True)
    daily["equity"] = (1.0 + daily["net_ret"].fillna(0.0)).cumprod()
    daily["rolling_peak"] = daily["equity"].cummax()
    daily["drawdown"] = np.where(daily["rolling_peak"] > 0.0, daily["equity"] / daily["rolling_peak"] - 1.0, 0.0)
    mean_daily = float(daily["net_ret"].mean()) if len(daily) else float("nan")
    std_daily = float(daily["net_ret"].std(ddof=0)) if len(daily) else float("nan")
    sharpe = float((mean_daily / (std_daily + EPS)) * np.sqrt(252.0)) if len(daily) else float("nan")
    return daily, {
        "days": float(len(daily)),
        "mean_daily": mean_daily,
        "std_daily": std_daily,
        "sharpe": sharpe,
        "cum_ret": float(daily["equity"].iloc[-1] - 1.0) if len(daily) else float("nan"),
        "max_drawdown": float(daily["drawdown"].min()) if len(daily) else float("nan"),
        "avg_turnover": float(daily["turnover"].mean()) if len(daily) else float("nan"),
        "avg_names_active": float(daily["names_active"].mean()) if len(daily) else float("nan"),
        "avg_gross_exposure": float(daily["gross_exposure"].mean()) if len(daily) else float("nan"),
    }


def _evaluate_candidate(train_df: pd.DataFrame, test_df: pd.DataFrame, candidate_col: str) -> Tuple[Dict[str, object], pd.DataFrame, pd.DataFrame]:
    train_scored = pd.DataFrame({
        "date": train_df["date"].values,
        "symbol": train_df["symbol"].values,
        TARGET_COL: _num(train_df[TARGET_COL]).values,
        "score": _num(train_df[candidate_col]).values,
    })
    train_ic_df = _daily_ic_series(train_scored, "score", TARGET_COL, MIN_DAILY_IC_CS)
    sign_info = _sign_decision(train_ic_df)
    test_scored = pd.DataFrame({
        "date": test_df["date"].values,
        "symbol": test_df["symbol"].values,
        TARGET_COL: _num(test_df[TARGET_COL]).values,
        "score": (_num(test_df[candidate_col]) * float(sign_info["train_sign_locked"])).values,
    })
    test_ic_df = _daily_ic_series(test_scored, "score", TARGET_COL, MIN_DAILY_IC_CS)
    if len(test_ic_df):
        ic_only = test_ic_df[["date", "daily_ic"]].copy()
    else:
        ic_only = pd.DataFrame(columns=["date", "daily_ic"])
    return sign_info, test_scored, ic_only


def _rank_candidates(train_cand_df: pd.DataFrame, test_cand_df: pd.DataFrame, cand_meta: pd.DataFrame, split: WFSplit) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    rows: List[Dict[str, object]] = []
    diag_frames: List[pd.DataFrame] = []
    for _, rec in cand_meta.iterrows():
        cand = str(rec["candidate"])
        sign_info, test_scored, ic_df = _evaluate_candidate(train_cand_df, test_cand_df, cand)
        tail_df = _tail_spread_series(test_scored, "score", TARGET_COL, TAIL_TOP_PCT)
        rows.append({
            "fold_id": int(split.fold_id),
            "candidate": cand,
            "alpha": rec["alpha"],
            "regime": rec["regime"],
            "kind": rec["kind"],
            **sign_info,
            "test_mean_ic": float(ic_df["daily_ic"].mean()) if len(ic_df) else float("nan"),
            "test_sign_stability": float((ic_df["daily_ic"] > 0.0).mean()) if len(ic_df) else float("nan"),
            "tail_mean_spread": float(tail_df["tail_spread"].mean()) if len(tail_df) else float("nan"),
            "tail_positive_rate": float((tail_df["tail_spread"] > 0.0).mean()) if len(tail_df) else float("nan"),
        })
        if len(ic_df):
            tmp = ic_df.copy()
            tmp["fold_id"] = int(split.fold_id)
            tmp["candidate"] = cand
            diag_frames.append(tmp)
    rank_df = pd.DataFrame(rows)
    if len(rank_df):
        rank_df["rank_score"] = pd.to_numeric(rank_df["test_mean_ic"], errors="coerce").fillna(0.0).abs()
        rank_df["rank_score"] += 0.50 * pd.to_numeric(rank_df["tail_positive_rate"], errors="coerce").fillna(0.0)
        rank_df["rank_score"] += 0.25 * pd.to_numeric(rank_df["tail_mean_spread"], errors="coerce").fillna(0.0)
        rank_df = rank_df.sort_values(["rank_score", "candidate"], ascending=[False, True]).reset_index(drop=True)
    return rank_df, diag_frames


def run_fold(df: pd.DataFrame, alpha_cols: Sequence[str], regime_cols: Sequence[str], split: WFSplit, shell: ShellConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, test_df = _split_frame(df, split)
    if len(train_df) < MIN_TRAIN_ROWS:
        raise RuntimeError(f"Fold {split.fold_id}: too few train rows: {len(train_df)}")
    if len(test_df) < MIN_TEST_ROWS:
        raise RuntimeError(f"Fold {split.fold_id}: too few test rows: {len(test_df)}")

    base_rank_df = _rank_base_alphas(train_df, alpha_cols)
    if base_rank_df.empty:
        raise RuntimeError(f"Fold {split.fold_id}: base rank is empty")
    train_cand_df, test_cand_df, cand_meta = _build_interaction_candidates(train_df, test_df, base_rank_df, regime_cols)
    cand_rank_df, cand_diag_frames = _rank_candidates(train_cand_df, test_cand_df, cand_meta, split)
    if cand_rank_df.empty:
        raise RuntimeError(f"Fold {split.fold_id}: candidate rank is empty")

    train_comp_df, test_comp_df, coef_df = _build_components(train_cand_df, test_cand_df, cand_rank_df, max_components=3)
    component_rows: List[Dict[str, object]] = []
    daily_rows: List[pd.DataFrame] = []
    incremental_rows: List[Dict[str, object]] = []

    top_candidates = coef_df.merge(cand_rank_df[["candidate", "alpha", "regime", "kind", "rank_score"]], on="candidate", how="left")
    top_candidates = top_candidates.sort_values("component").reset_index(drop=True)
    for _, rec in top_candidates.iterrows():
        comp = str(rec["component"])
        train_ic_df = _daily_ic_series(train_comp_df.assign(score=_num(train_comp_df[f"{comp}_resid"])), "score", TARGET_COL, MIN_DAILY_IC_CS)
        sign_info = _sign_decision(train_ic_df)
        test_scored = test_comp_df[["date", "symbol", TARGET_COL]].copy()
        test_scored["score"] = _num(test_comp_df[f"{comp}_resid"]) * float(sign_info["train_sign_locked"])
        test_ic_df = _daily_ic_series(test_scored, "score", TARGET_COL, MIN_DAILY_IC_CS)
        tail_df = _tail_spread_series(test_scored, "score", TARGET_COL, TAIL_TOP_PCT)
        port_df = build_portfolio(test_scored, shell)
        daily_df, summary = evaluate_portfolio(port_df)
        daily_df["fold_id"] = int(split.fold_id)
        daily_df["component"] = comp
        daily_df["candidate"] = rec["candidate"]
        daily_df["layer_kind"] = "component"
        daily_rows.append(daily_df)
        component_rows.append({
            "fold_id": int(split.fold_id),
            "component": comp,
            "candidate": rec["candidate"],
            "alpha": rec["alpha"],
            "regime": rec["regime"],
            "kind": rec["kind"],
            "layer_kind": "component",
            **summary,
            **sign_info,
            "test_mean_ic": float(test_ic_df["daily_ic"].mean()) if len(test_ic_df) else float("nan"),
            "test_sign_stability": float((test_ic_df["daily_ic"] > 0.0).mean()) if len(test_ic_df) else float("nan"),
            "tail_mean_spread": float(tail_df["tail_spread"].mean()) if len(tail_df) else float("nan"),
            "tail_positive_rate": float((tail_df["tail_spread"] > 0.0).mean()) if len(tail_df) else float("nan"),
            "rank_score": rec.get("rank_score", float("nan")),
        })

        cumulative_name = f"cum{int(comp.removeprefix('comp'))}"
        cumulative_df = test_comp_df[["date", "symbol", TARGET_COL]].copy()
        cumulative_df["score"] = 0.0
        for j in range(1, int(comp.removeprefix("comp")) + 1):
            comp_j = f"comp{j}"
            sign_j = next(x for x in component_rows if x["fold_id"] == int(split.fold_id) and x["component"] == comp_j)["train_sign_locked"]
            cumulative_df["score"] = _num(cumulative_df["score"]) + (_num(test_comp_df[f"{comp_j}_resid"]) * float(sign_j))
        cumulative_df["score"] = cumulative_df.groupby("date", sort=False)["score"].transform(_robust_zscore_series)
        cum_port_df = build_portfolio(cumulative_df, shell)
        cum_daily_df, cum_summary = evaluate_portfolio(cum_port_df)
        cum_daily_df["fold_id"] = int(split.fold_id)
        cum_daily_df["component"] = cumulative_name
        cum_daily_df["candidate"] = rec["candidate"]
        cum_daily_df["layer_kind"] = "cumulative"
        daily_rows.append(cum_daily_df)
        prev_sharpe = component_rows[-2]["sharpe"] if len(component_rows) >= 2 else float("nan")
        incremental_rows.append({
            "fold_id": int(split.fold_id),
            "component": cumulative_name,
            "candidate_added": rec["candidate"],
            "alpha_added": rec["alpha"],
            "regime_added": rec["regime"],
            "kind_added": rec["kind"],
            "component_count": int(comp.removeprefix("comp")),
            "cum_sharpe": float(cum_summary["sharpe"]),
            "cum_mean_daily": float(cum_summary["mean_daily"]),
            "cum_ret": float(cum_summary["cum_ret"]),
            "cum_max_drawdown": float(cum_summary["max_drawdown"]),
            "cum_turnover": float(cum_summary["avg_turnover"]),
            "delta_vs_prev_component_sharpe": float(cum_summary["sharpe"] - prev_sharpe) if pd.notna(prev_sharpe) else float("nan"),
        })

    component_df = pd.DataFrame(component_rows)
    daily_all_df = pd.concat(daily_rows, ignore_index=True) if daily_rows else pd.DataFrame()
    incremental_df = pd.DataFrame(incremental_rows)
    candidate_diag_df = pd.concat(cand_diag_frames, ignore_index=True) if cand_diag_frames else pd.DataFrame(columns=["date", "daily_ic"])
    coef_df["fold_id"] = int(split.fold_id)
    coef_out = top_candidates.copy() if len(top_candidates) else coef_df.copy()
    if len(coef_out):
        coef_out["fold_id"] = int(split.fold_id)
    return base_rank_df.assign(fold_id=int(split.fold_id)), cand_rank_df, component_df, daily_all_df, incremental_df, coef_out


def _summarize_components(component_fold_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    if component_fold_df.empty:
        return pd.DataFrame()
    work = component_fold_df.sort_values(["component", "fold_id"]).reset_index(drop=True)
    for component, g in work.groupby("component", sort=False):
        rows.append({
            "component": component,
            "candidate": str(g["candidate"].iloc[0]),
            "alpha": str(g["alpha"].iloc[0]),
            "regime": str(g["regime"].iloc[0]),
            "kind": str(g["kind"].iloc[0]),
            "folds": int(len(g)),
            "oos_sharpe_mean": _safe_mean(g["sharpe"].tolist()),
            "last_fold_sharpe": _safe_last(g["sharpe"].tolist()),
            "last_2_fold_mean_sharpe": _last_n_mean(g["sharpe"].tolist(), 2),
            "test_mean_ic": _safe_mean(g["test_mean_ic"].tolist()),
            "sign_stability": _safe_mean(g["test_sign_stability"].tolist()),
            "tail_positive_rate": _safe_mean(g["tail_positive_rate"].tolist()),
            "max_drawdown": float(pd.to_numeric(g["max_drawdown"], errors="coerce").min()),
            "turnover_mean": _safe_mean(g["avg_turnover"].tolist()),
            "positive_sharpe_fraction": _positive_fraction(g["sharpe"].tolist()),
            "sign_lock_rate": _safe_mean(g["sign_lock_triggered"].tolist()),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["last_2_fold_mean_sharpe", "oos_sharpe_mean", "component"], ascending=[False, False, True]).reset_index(drop=True)
    return out


def _summarize_incremental(incremental_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    if incremental_df.empty:
        return pd.DataFrame()
    for component, g in incremental_df.sort_values(["component", "fold_id"]).groupby("component", sort=False):
        rows.append({
            "component": component,
            "candidate_added": str(g["candidate_added"].iloc[0]),
            "alpha_added": str(g["alpha_added"].iloc[0]),
            "regime_added": str(g["regime_added"].iloc[0]),
            "kind_added": str(g["kind_added"].iloc[0]),
            "folds": int(len(g)),
            "cum_sharpe_mean": _safe_mean(g["cum_sharpe"].tolist()),
            "last_fold_cum_sharpe": _safe_last(g["cum_sharpe"].tolist()),
            "last_2_fold_cum_sharpe": _last_n_mean(g["cum_sharpe"].tolist(), 2),
            "delta_vs_prev_component_sharpe_mean": _safe_mean(g["delta_vs_prev_component_sharpe"].tolist()),
            "cum_turnover_mean": _safe_mean(g["cum_turnover"].tolist()),
            "cum_max_drawdown": float(pd.to_numeric(g["cum_max_drawdown"], errors="coerce").min()),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["last_2_fold_cum_sharpe", "cum_sharpe_mean", "component"], ascending=[False, False, True]).reset_index(drop=True)
    return out


def main() -> int:
    _enable_line_buffering()
    shell = ShellConfig(enter_pct=ENTER_PCT, exit_pct=EXIT_PCT, weight_cap=WEIGHT_CAP, gross_target=GROSS_TARGET, max_daily_turnover=MAX_DAILY_TURNOVER)
    print(f"[CFG] alpha_lib_file={ALPHA_LIB_FILE}")
    print(f"[CFG] feature_v2_file={FEATURE_V2_FILE}")
    print(f"[CFG] out_dir={OUT_DIR}")
    print(f"[CFG] wf train_days={TRAIN_DAYS} test_days={TEST_DAYS} step_days={STEP_DAYS} purge_days={PURGE_DAYS} embargo_days={EMBARGO_DAYS}")
    print(f"[CFG] shell gross_target={GROSS_TARGET} weight_cap={WEIGHT_CAP} max_daily_turnover={MAX_DAILY_TURNOVER} enter_pct={ENTER_PCT} exit_pct={EXIT_PCT} cost_bps={COST_BPS}")
    print(f"[CFG] base_filter alpha_name_filter={ALPHA_NAME_FILTER!r} alpha_limit={ALPHA_LIMIT} interaction_base_limit={INTERACTION_BASE_LIMIT}")
    print(f"[CFG] regimes vol={int(ENABLE_VOL_REGIME)} liq={int(ENABLE_LIQ_REGIME)} trend={int(ENABLE_TREND_REGIME)} breadth={int(ENABLE_BREADTH_REGIME)}")
    print(f"[CFG] engine plain_base={int(ENABLE_PLAIN_BASE)} interactions={int(ENABLE_INTERACTIONS)} sign_lock_ic_abs={SIGN_LOCK_IC_ABS} sign_lock_pos_rate={SIGN_LOCK_POS_RATE}")
    print(f"[CFG] auto_build_fs2_if_missing={int(AUTO_BUILD_FS2_IF_MISSING)} include_z_features={int(INCLUDE_Z_FEATURES)} exclude_target_derived={int(EXCLUDE_TARGET_DERIVED)} exclude_market_wide_constants={int(EXCLUDE_MARKET_WIDE_CONSTANTS)}")

    df, alpha_cols = _load_data()
    df, regime_cols, regime_meta = _add_explicit_regime_features(df)
    if not regime_cols:
        raise RuntimeError("No explicit regime features were built")
    print(f"[DATA] rows={len(df)} dates={df['date'].nunique()} symbols={df['symbol'].nunique()} alpha_cols={len(alpha_cols)} regimes={len(regime_cols)}")
    print(f"[DATA] alpha_source={df['__alpha_data_source__'].iloc[0] if '__alpha_data_source__' in df.columns else 'unknown'}")
    print("[REGIMES]")
    print(regime_meta.to_string(index=False))

    splits = build_walkforward_splits(df["date"])
    print(f"[WF] folds={len(splits)}")
    for sp in splits:
        print(f"[WF][FOLD {sp.fold_id}] train={sp.train_start.date()}..{sp.train_end.date()} test={sp.test_start.date()}..{sp.test_end.date()}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    regime_meta.to_csv(OUT_DIR / "explicit_regime_manifest.csv", index=False)

    base_rank_all: List[pd.DataFrame] = []
    cand_rank_all: List[pd.DataFrame] = []
    component_all: List[pd.DataFrame] = []
    daily_all: List[pd.DataFrame] = []
    incremental_all: List[pd.DataFrame] = []
    coef_all: List[pd.DataFrame] = []

    for split in splits:
        base_rank_df, cand_rank_df, component_df, daily_df, incremental_df, coef_df = run_fold(df, alpha_cols, regime_cols, split, shell)
        base_rank_all.append(base_rank_df)
        cand_rank_all.append(cand_rank_df.assign(fold_id=int(split.fold_id)))
        component_all.append(component_df)
        daily_all.append(daily_df)
        incremental_all.append(incremental_df)
        coef_all.append(coef_df)

        base_rank_df.to_csv(OUT_DIR / f"base_rank__fold{split.fold_id}.csv", index=False)
        cand_rank_df.to_csv(OUT_DIR / f"interaction_rank__fold{split.fold_id}.csv", index=False)
        component_df.to_csv(OUT_DIR / f"residual_component_metrics__fold{split.fold_id}.csv", index=False)
        daily_df.to_csv(OUT_DIR / f"residual_daily__fold{split.fold_id}.csv", index=False)
        incremental_df.to_csv(OUT_DIR / f"residual_incremental__fold{split.fold_id}.csv", index=False)
        coef_df.to_csv(OUT_DIR / f"residual_coefficients__fold{split.fold_id}.csv", index=False)

        for _, row in component_df.sort_values("component").iterrows():
            print(f"[FOLD {split.fold_id}][{row['component']}] candidate={row['candidate']} sharpe={row['sharpe']:.4f} mean_ic={row['test_mean_ic']:.5f} sign_stability={row['test_sign_stability']:.4f} maxdd={row['max_drawdown']:.4f} turnover={row['avg_turnover']:.4f}")
        for _, row in incremental_df.sort_values("component_count").iterrows():
            print(f"[FOLD {split.fold_id}][incremental][{row['component']}] add={row['candidate_added']} cum_sharpe={row['cum_sharpe']:.4f} delta_vs_prev_component_sharpe={row['delta_vs_prev_component_sharpe']:.4f}")

    base_rank_all_df = pd.concat(base_rank_all, ignore_index=True) if base_rank_all else pd.DataFrame()
    cand_rank_all_df = pd.concat(cand_rank_all, ignore_index=True) if cand_rank_all else pd.DataFrame()
    component_all_df = pd.concat(component_all, ignore_index=True) if component_all else pd.DataFrame()
    daily_all_df = pd.concat(daily_all, ignore_index=True) if daily_all else pd.DataFrame()
    incremental_all_df = pd.concat(incremental_all, ignore_index=True) if incremental_all else pd.DataFrame()
    coef_all_df = pd.concat(coef_all, ignore_index=True) if coef_all else pd.DataFrame()

    component_summary_df = _summarize_components(component_all_df)
    incremental_summary_df = _summarize_incremental(incremental_all_df)

    base_rank_all_df.to_csv(OUT_DIR / "base_rank__all_folds.csv", index=False)
    cand_rank_all_df.to_csv(OUT_DIR / "interaction_rank__all_folds.csv", index=False)
    component_all_df.to_csv(OUT_DIR / "residual_component_metrics__all_folds.csv", index=False)
    daily_all_df.to_csv(OUT_DIR / "residual_daily__all_folds.csv", index=False)
    incremental_all_df.to_csv(OUT_DIR / "residual_incremental__all_folds.csv", index=False)
    coef_all_df.to_csv(OUT_DIR / "residual_coefficients__all_folds.csv", index=False)
    component_summary_df.to_csv(OUT_DIR / "residual_component_summary.csv", index=False)
    incremental_summary_df.to_csv(OUT_DIR / "residual_incremental_summary.csv", index=False)

    if len(component_summary_df):
        print("[SUMMARY][COMPONENTS]")
        print(component_summary_df.head(TOPK_PRINT).to_string(index=False))
    if len(incremental_summary_df):
        print("[SUMMARY][INCREMENTAL]")
        print(incremental_summary_df.head(TOPK_PRINT).to_string(index=False))

    meta = {
        "alpha_lib_file": str(ALPHA_LIB_FILE),
        "feature_v2_file": str(FEATURE_V2_FILE),
        "target_col": TARGET_COL,
        "wf": {
            "train_days": TRAIN_DAYS,
            "test_days": TEST_DAYS,
            "step_days": STEP_DAYS,
            "purge_days": PURGE_DAYS,
            "embargo_days": EMBARGO_DAYS,
            "folds": len(splits),
        },
        "shell": {
            "enter_pct": ENTER_PCT,
            "exit_pct": EXIT_PCT,
            "weight_cap": WEIGHT_CAP,
            "gross_target": GROSS_TARGET,
            "max_daily_turnover": MAX_DAILY_TURNOVER,
            "cost_bps": COST_BPS,
        },
        "engine": {
            "plain_base": int(ENABLE_PLAIN_BASE),
            "interactions": int(ENABLE_INTERACTIONS),
            "interaction_base_limit": INTERACTION_BASE_LIMIT,
            "sign_lock_ic_abs": SIGN_LOCK_IC_ABS,
            "sign_lock_pos_rate": SIGN_LOCK_POS_RATE,
            "auto_build_fs2_if_missing": int(AUTO_BUILD_FS2_IF_MISSING),
        },
        "regimes": regime_meta.to_dict("records"),
        "modules": {
            "holding_inertia": int(HAS_HOLDING_INERTIA),
            "turnover_control": int(HAS_TURNOVER_CONTROL),
        },
        "rows": {
            "base_rank_rows": int(len(base_rank_all_df)),
            "interaction_rank_rows": int(len(cand_rank_all_df)),
            "component_rows": int(len(component_all_df)),
            "daily_rows": int(len(daily_all_df)),
            "incremental_rows": int(len(incremental_all_df)),
            "coef_rows": int(len(coef_all_df)),
        },
    }
    meta_path = OUT_DIR / "residual_layer_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ARTIFACT] {OUT_DIR / 'explicit_regime_manifest.csv'}")
    print(f"[ARTIFACT] {OUT_DIR / 'interaction_rank__all_folds.csv'}")
    print(f"[ARTIFACT] {OUT_DIR / 'residual_component_summary.csv'}")
    print(f"[ARTIFACT] {OUT_DIR / 'residual_incremental_summary.csv'}")
    print(f"[ARTIFACT] {meta_path}")
    print("[FINAL] residual interaction engine walk-forward complete")
    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 1
    _safe_exit(rc)
