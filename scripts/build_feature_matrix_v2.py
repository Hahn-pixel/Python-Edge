from __future__ import annotations

import json
import math
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
    from python_edge.features.add_intraday_pressure import add_intraday_pressure
except Exception:
    def add_intraday_pressure(df15: pd.DataFrame, df1d: pd.DataFrame) -> pd.DataFrame:
        return df1d.copy()

try:
    from python_edge.features.add_intraday_rs import add_intraday_rs
except Exception:
    def add_intraday_rs(df15: pd.DataFrame, df1d: pd.DataFrame) -> pd.DataFrame:
        return df1d.copy()

try:
    from python_edge.model.cs_normalize import cs_zscore
except Exception:
    def cs_zscore(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        out = df.copy()
        for col in feature_cols:
            grp = out.groupby("date", sort=False)[col]
            mean = grp.transform("mean")
            std = grp.transform("std").replace(0, pd.NA)
            out[f"z_{col}"] = (pd.to_numeric(out[col], errors="coerce") - mean) / std
        return out

FEATURE_V1_FILE = Path(os.getenv("FEATURE_V1_FILE", "data/features/feature_matrix_v1.parquet"))
FEATURE_V2_OUT_FILE = Path(os.getenv("FEATURE_V2_OUT_FILE", "data/features/feature_matrix_v2.parquet"))
FEATURE_V2_MANIFEST = Path(os.getenv("FEATURE_V2_MANIFEST", "data/features/feature_matrix_v2_manifest.csv"))
FEATURE_V2_DIAG_JSON = Path(os.getenv("FEATURE_V2_DIAG_JSON", "data/features/feature_matrix_v2_diag.json"))
FEATURE_V2_DIAG_CSV = Path(os.getenv("FEATURE_V2_DIAG_CSV", "data/features/feature_matrix_v2_diag.csv"))
PAUSE_ON_EXIT = str(os.getenv("PAUSE_ON_EXIT", "auto")).strip().lower()

ENABLE_INTRADAY_CLUSTER = str(os.getenv("ENABLE_INTRADAY_CLUSTER", "1")).strip().lower() not in {"0", "false", "no", "off"}
ENABLE_VOL_CLUSTER = str(os.getenv("ENABLE_VOL_CLUSTER", "1")).strip().lower() not in {"0", "false", "no", "off"}
ENABLE_CS_CLUSTER = str(os.getenv("ENABLE_CS_CLUSTER", "1")).strip().lower() not in {"0", "false", "no", "off"}
ENABLE_ZSCORES = str(os.getenv("ENABLE_ZSCORES", "1")).strip().lower() not in {"0", "false", "no", "off"}

CS_Z_WINDOW = int(os.getenv("CS_Z_WINDOW", "0"))
ROLL_SHORT = int(os.getenv("ROLL_SHORT", "5"))
ROLL_MED = int(os.getenv("ROLL_MED", "10"))
ROLL_LONG = int(os.getenv("ROLL_LONG", "20"))
ROLL_XLONG = int(os.getenv("ROLL_XLONG", "60"))
MIN_UNIQUE = int(os.getenv("MIN_UNIQUE", "5"))
MAX_NAN_RATIO = float(os.getenv("MAX_NAN_RATIO", "0.995"))
MIN_NON_NA = int(os.getenv("MIN_NON_NA", "200"))
TOPK_PRINT = int(os.getenv("TOPK_PRINT", "40"))

EPS = 1e-12


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    cluster: str
    source: str
    formula: str


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


def _series_or_nan(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return _num(df[col])
    return pd.Series(np.nan, index=df.index, dtype="float64")


def _clip_inf(s: pd.Series) -> pd.Series:
    return _num(s).replace([np.inf, -np.inf], np.nan)


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return _clip_inf(_num(a) / (_num(b).abs() + EPS))


def _signed_log(s: pd.Series) -> pd.Series:
    x = _num(s)
    return np.sign(x) * np.log1p(np.abs(x))


def _tanh_z_by_symbol(s: pd.Series, win: int) -> pd.Series:
    x = _num(s)
    mean = x.rolling(win, min_periods=max(3, win // 3)).mean()
    std = x.rolling(win, min_periods=max(3, win // 3)).std(ddof=0)
    z = (x - mean) / (std + EPS)
    return np.tanh(z)


def _pct_rank_by_date(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby("date", sort=False)[col].rank(method="average", pct=True)


def _rolling_rank_last(s: pd.Series, win: int) -> pd.Series:
    x = _num(s)
    def _last_rank(window_vals: pd.Series) -> float:
        arr = pd.Series(window_vals)
        return float(arr.rank(method="average", pct=True).iloc[-1])
    return x.rolling(win, min_periods=max(3, win // 3)).apply(_last_rank, raw=False)


def _add_symbol_roll(df: pd.DataFrame, col: str, win: int, op: str) -> pd.Series:
    grp = df.groupby("symbol", sort=False)[col]
    if op == "mean":
        return grp.transform(lambda s: _num(s).rolling(win, min_periods=max(3, win // 3)).mean())
    if op == "std":
        return grp.transform(lambda s: _num(s).rolling(win, min_periods=max(3, win // 3)).std(ddof=0))
    if op == "sum":
        return grp.transform(lambda s: _num(s).rolling(win, min_periods=max(3, win // 3)).sum())
    if op == "min":
        return grp.transform(lambda s: _num(s).rolling(win, min_periods=max(3, win // 3)).min())
    if op == "max":
        return grp.transform(lambda s: _num(s).rolling(win, min_periods=max(3, win // 3)).max())
    raise RuntimeError(f"Unsupported rolling op: {op}")


def _target_col(df: pd.DataFrame) -> str:
    for c in ["target_fwd_ret_1d", "fwd_ret_1d", "ret_fwd_1d"]:
        if c in df.columns:
            return c
    raise RuntimeError("Missing target column: expected one of target_fwd_ret_1d / fwd_ret_1d / ret_fwd_1d")


def _ensure_base_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    required = ["date", "symbol"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise RuntimeError(f"feature_matrix_v1 missing required columns: {missing}")
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out = out.sort_values(["symbol", "date"]).reset_index(drop=True)
    if "session_date" not in out.columns:
        out["session_date"] = out["date"]
    return out


def _choose_intraday_proxy(df: pd.DataFrame) -> Tuple[pd.Series, str]:
    for col in ["intraday_rs", "intraday_pressure", "pressure", "rev1", "mom3"]:
        if col in df.columns:
            return _num(df[col]), col
    return pd.Series(np.nan, index=df.index, dtype="float64"), "none"


def _choose_vol_proxy(df: pd.DataFrame) -> Tuple[pd.Series, str]:
    for col in ["ivol_20d", "vol_spike_5_20", "range_comp_5_20", "atr_20", "ret_std_20"]:
        if col in df.columns:
            return _num(df[col]), col
    return pd.Series(np.nan, index=df.index, dtype="float64"), "none"


def _choose_liq_proxy(df: pd.DataFrame) -> Tuple[pd.Series, str]:
    for col in ["liq", "cs_liq_z", "dollar_vol", "dollar_volume", "rel_volume_20"]:
        if col in df.columns:
            return _num(df[col]), col
    return pd.Series(np.nan, index=df.index, dtype="float64"), "none"


def _choose_rvol_proxy(df: pd.DataFrame) -> Tuple[pd.Series, str]:
    for col in ["rvol", "rel_volume_20", "volume_shock", "volshock"]:
        if col in df.columns:
            return _num(df[col]), col
    return pd.Series(np.nan, index=df.index, dtype="float64"), "none"


def _register(specs: List[FeatureSpec], name: str, cluster: str, source: str, formula: str) -> None:
    specs.append(FeatureSpec(name=name, cluster=cluster, source=source, formula=formula))


def build_feature_matrix_v2(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = _ensure_base_columns(df)
    tgt = _target_col(out)
    feature_specs: List[FeatureSpec] = []

    target = _num(out[tgt])
    intraday_proxy, intraday_src = _choose_intraday_proxy(out)
    vol_proxy, vol_src = _choose_vol_proxy(out)
    liq_proxy, liq_src = _choose_liq_proxy(out)
    rvol_proxy, rvol_src = _choose_rvol_proxy(out)
    pressure_proxy = _series_or_nan(out, "intraday_pressure")
    rs_proxy = _series_or_nan(out, "intraday_rs")
    breadth_proxy = _series_or_nan(out, "market_breadth")
    adv_decl_proxy = _series_or_nan(out, "adv_decl_proxy")
    close_proxy = _series_or_nan(out, "close")
    high_proxy = _series_or_nan(out, "high")
    low_proxy = _series_or_nan(out, "low")
    open_proxy = _series_or_nan(out, "open")

    if ENABLE_INTRADAY_CLUSTER:
        out["fs2_intraday_pressure_proxy"] = pressure_proxy.fillna(intraday_proxy)
        _register(feature_specs, "fs2_intraday_pressure_proxy", "intraday_liquidity", pressure_proxy.name if hasattr(pressure_proxy, 'name') else intraday_src, "intraday_pressure or fallback intraday proxy")

        out["fs2_intraday_rs_proxy"] = rs_proxy.fillna(intraday_proxy)
        _register(feature_specs, "fs2_intraday_rs_proxy", "intraday_liquidity", rs_proxy.name if hasattr(rs_proxy, 'name') else intraday_src, "intraday_rs or fallback intraday proxy")

        out["fs2_intraday_pressure_rs_mix"] = _clip_inf(0.5 * _num(out["fs2_intraday_pressure_proxy"]) + 0.5 * _num(out["fs2_intraday_rs_proxy"]))
        _register(feature_specs, "fs2_intraday_pressure_rs_mix", "intraday_liquidity", f"{intraday_src}", "0.5*pressure + 0.5*intraday_rs")

        out["fs2_intraday_reversal_pressure"] = _clip_inf(-1.0 * _num(out["fs2_intraday_rs_proxy"]) * (1.0 + _num(out["fs2_intraday_pressure_proxy"])))
        _register(feature_specs, "fs2_intraday_reversal_pressure", "intraday_liquidity", f"{intraday_src}", "-intraday_rs*(1+intraday_pressure)")

        out["fs2_intraday_strength_liq_interaction"] = _clip_inf(_num(out["fs2_intraday_rs_proxy"]) * _signed_log(liq_proxy))
        _register(feature_specs, "fs2_intraday_strength_liq_interaction", "intraday_liquidity", f"{intraday_src}|{liq_src}", "intraday_rs * signed_log(liq)")

        out["fs2_intraday_strength_rvol_interaction"] = _clip_inf(_num(out["fs2_intraday_rs_proxy"]) * _signed_log(rvol_proxy))
        _register(feature_specs, "fs2_intraday_strength_rvol_interaction", "intraday_liquidity", f"{intraday_src}|{rvol_src}", "intraday_rs * signed_log(rvol)")

        out["fs2_gap_fill_fraction_proxy"] = _clip_inf(-_num(intraday_proxy) / (_num(vol_proxy).abs() + 1.0))
        _register(feature_specs, "fs2_gap_fill_fraction_proxy", "intraday_liquidity", f"{intraday_src}|{vol_src}", "-intraday_proxy / (abs(vol_proxy)+1)")

        out["fs2_intraday_range_expansion_proxy"] = _clip_inf(_num(intraday_proxy).abs() * (1.0 + _num(rvol_proxy).abs()))
        _register(feature_specs, "fs2_intraday_range_expansion_proxy", "intraday_liquidity", f"{intraday_src}|{rvol_src}", "abs(intraday_proxy)*(1+abs(rvol))")

        out["fs2_dollar_vol_accel_proxy"] = _add_symbol_roll(pd.DataFrame({"symbol": out["symbol"], "date": out["date"], "x": liq_proxy}), "x", ROLL_SHORT, "mean") - _add_symbol_roll(pd.DataFrame({"symbol": out["symbol"], "date": out["date"], "x": liq_proxy}), "x", ROLL_LONG, "mean")
        _register(feature_specs, "fs2_dollar_vol_accel_proxy", "intraday_liquidity", liq_src, f"mean_{ROLL_SHORT}(liq)-mean_{ROLL_LONG}(liq)")

    if ENABLE_VOL_CLUSTER:
        out["fs2_realized_vol_accel"] = _add_symbol_roll(pd.DataFrame({"symbol": out["symbol"], "date": out["date"], "x": vol_proxy.abs()}), "x", ROLL_SHORT, "mean") - _add_symbol_roll(pd.DataFrame({"symbol": out["symbol"], "date": out["date"], "x": vol_proxy.abs()}), "x", ROLL_LONG, "mean")
        _register(feature_specs, "fs2_realized_vol_accel", "vol_structure", vol_src, f"mean_{ROLL_SHORT}(abs(vol))-mean_{ROLL_LONG}(abs(vol))")

        out["fs2_vol_of_vol_short"] = _add_symbol_roll(pd.DataFrame({"symbol": out["symbol"], "date": out["date"], "x": vol_proxy}), "x", ROLL_SHORT, "std")
        _register(feature_specs, "fs2_vol_of_vol_short", "vol_structure", vol_src, f"std_{ROLL_SHORT}(vol_proxy)")

        out["fs2_shock_reversion_ratio"] = _safe_div(vol_proxy, _add_symbol_roll(pd.DataFrame({"symbol": out["symbol"], "date": out["date"], "x": vol_proxy.abs()}), "x", ROLL_LONG, "mean"))
        _register(feature_specs, "fs2_shock_reversion_ratio", "vol_structure", vol_src, f"vol_proxy / mean_{ROLL_LONG}(abs(vol_proxy))")

        out["fs2_range_compression_streak"] = _safe_div(_add_symbol_roll(pd.DataFrame({"symbol": out["symbol"], "date": out["date"], "x": vol_proxy.abs()}), "x", ROLL_SHORT, "mean"), _add_symbol_roll(pd.DataFrame({"symbol": out["symbol"], "date": out["date"], "x": vol_proxy.abs()}), "x", ROLL_XLONG, "mean"))
        _register(feature_specs, "fs2_range_compression_streak", "vol_structure", vol_src, f"mean_{ROLL_SHORT}(abs(vol))/mean_{ROLL_XLONG}(abs(vol))")

        out["fs2_overnight_vs_intraday_vol_ratio_proxy"] = _safe_div(target.abs(), intraday_proxy.abs() + 1e-6)
        _register(feature_specs, "fs2_overnight_vs_intraday_vol_ratio_proxy", "vol_structure", f"{tgt}|{intraday_src}", "abs(target)/abs(intraday_proxy)")

        out["fs2_hl_vol_ratio_proxy"] = _safe_div((high_proxy - low_proxy).abs(), target.abs() + 1e-6)
        _register(feature_specs, "fs2_hl_vol_ratio_proxy", "vol_structure", "high|low|target", "abs(high-low)/abs(target)")

        out["fs2_vol_liq_mix"] = _clip_inf(_signed_log(vol_proxy.abs() + 1.0) - _signed_log(liq_proxy.abs() + 1.0))
        _register(feature_specs, "fs2_vol_liq_mix", "vol_structure", f"{vol_src}|{liq_src}", "signed_log(abs(vol)+1)-signed_log(abs(liq)+1)")

    if ENABLE_CS_CLUSTER:
        out["fs2_target_rank_pct"] = _pct_rank_by_date(pd.DataFrame({"date": out["date"], "x": target}).rename(columns={"x": "target"}), "target")
        _register(feature_specs, "fs2_target_rank_pct", "cs_pressure", tgt, "cross-sectional percentile rank of target")

        out["fs2_cs_winner_loser_spread_proxy"] = 2.0 * _num(out["fs2_target_rank_pct"]) - 1.0
        _register(feature_specs, "fs2_cs_winner_loser_spread_proxy", "cs_pressure", tgt, "2*rank_pct(target)-1")

        out["fs2_cs_dispersion_short"] = out.groupby("date", sort=False)[tgt].transform(lambda s: _num(s).std(ddof=0))
        _register(feature_specs, "fs2_cs_dispersion_short", "cs_pressure", tgt, "per-date std(target)")

        temp_df = pd.DataFrame({"symbol": out["symbol"], "date": out["date"], "x": _num(out["fs2_cs_dispersion_short"])})
        out["fs2_cs_dispersion_accel"] = _add_symbol_roll(temp_df, "x", ROLL_SHORT, "mean") - _add_symbol_roll(temp_df, "x", ROLL_LONG, "mean")
        _register(feature_specs, "fs2_cs_dispersion_accel", "cs_pressure", tgt, f"mean_{ROLL_SHORT}(cs_dispersion)-mean_{ROLL_LONG}(cs_dispersion)")

        out["fs2_market_breadth_thrust_proxy"] = breadth_proxy.fillna(adv_decl_proxy).fillna(_num(out["fs2_cs_winner_loser_spread_proxy"]))
        _register(feature_specs, "fs2_market_breadth_thrust_proxy", "cs_pressure", "market_breadth|adv_decl_proxy|target", "breadth proxy or fallback winner-loser spread")

        out["fs2_breadth_reversal_proxy"] = _clip_inf(-_num(out["fs2_market_breadth_thrust_proxy"]) * _signed_log(_num(out["fs2_cs_dispersion_short"]) + 1.0))
        _register(feature_specs, "fs2_breadth_reversal_proxy", "cs_pressure", "breadth|cs_dispersion", "-breadth * signed_log(cs_dispersion+1)")

        out["fs2_top_tail_strength_proxy"] = _clip_inf(_num(out["fs2_target_rank_pct"]) * _signed_log(rvol_proxy.abs() + 1.0))
        _register(feature_specs, "fs2_top_tail_strength_proxy", "cs_pressure", f"{tgt}|{rvol_src}", "rank_pct(target) * signed_log(abs(rvol)+1)")

        out["fs2_bottom_tail_strength_proxy"] = _clip_inf((1.0 - _num(out["fs2_target_rank_pct"])) * _signed_log(rvol_proxy.abs() + 1.0))
        _register(feature_specs, "fs2_bottom_tail_strength_proxy", "cs_pressure", f"{tgt}|{rvol_src}", "(1-rank_pct(target)) * signed_log(abs(rvol)+1)")

        out["fs2_pressure_dispersion_mix"] = _clip_inf(_signed_log(_num(out.get("fs2_intraday_pressure_proxy", pd.Series(np.nan, index=out.index))) + 1.0) * _signed_log(_num(out["fs2_cs_dispersion_short"]) + 1.0))
        _register(feature_specs, "fs2_pressure_dispersion_mix", "cs_pressure", "intraday_pressure|cs_dispersion", "signed_log(pressure+1)*signed_log(cs_dispersion+1)")

    new_features = [x.name for x in feature_specs]
    if ENABLE_ZSCORES and new_features:
        out = cs_zscore(out, new_features)
        z_created = [f"z_{c}" for c in new_features if f"z_{c}" in out.columns]
        for zc in z_created:
            _register(feature_specs, zc, "cs_zscore", zc.removeprefix("z_"), f"cross-sectional zscore of {zc.removeprefix('z_')}")

    manifest = pd.DataFrame([x.__dict__ for x in feature_specs])
    manifest = manifest.drop_duplicates(subset=["name"]).reset_index(drop=True)
    return out, manifest


def build_diagnostics(df: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for _, rec in manifest.iterrows():
        col = str(rec["name"])
        s = _num(df[col]) if col in df.columns else pd.Series(dtype="float64")
        non_na = int(s.notna().sum())
        nan_ratio = float(s.isna().mean()) if len(s) else 1.0
        unique = int(s.nunique(dropna=True)) if len(s) else 0
        std = float(s.std(ddof=0)) if non_na else float("nan")
        mean_abs = float(s.abs().mean()) if non_na else float("nan")
        keep = int(non_na >= MIN_NON_NA and nan_ratio <= MAX_NAN_RATIO and unique >= MIN_UNIQUE)
        rows.append({
            "feature": col,
            "cluster": rec.get("cluster", "unknown"),
            "source": rec.get("source", "unknown"),
            "formula": rec.get("formula", ""),
            "non_na": non_na,
            "nan_ratio": nan_ratio,
            "unique": unique,
            "std": std,
            "mean_abs": mean_abs,
            "keep": keep,
        })
    diag = pd.DataFrame(rows)
    if len(diag):
        diag = diag.sort_values(["keep", "cluster", "non_na", "unique", "feature"], ascending=[False, True, False, False, True]).reset_index(drop=True)
    return diag


def main() -> int:
    _enable_line_buffering()
    print(f"[CFG] feature_v1_file={FEATURE_V1_FILE}")
    print(f"[CFG] feature_v2_out_file={FEATURE_V2_OUT_FILE}")
    print(f"[CFG] enable_intraday_cluster={int(ENABLE_INTRADAY_CLUSTER)} enable_vol_cluster={int(ENABLE_VOL_CLUSTER)} enable_cs_cluster={int(ENABLE_CS_CLUSTER)} enable_zscores={int(ENABLE_ZSCORES)}")
    print(f"[CFG] roll_short={ROLL_SHORT} roll_med={ROLL_MED} roll_long={ROLL_LONG} roll_xlong={ROLL_XLONG}")
    print(f"[CFG] min_non_na={MIN_NON_NA} max_nan_ratio={MAX_NAN_RATIO} min_unique={MIN_UNIQUE}")

    _must_exist(FEATURE_V1_FILE, "Feature v1 file")
    df = pd.read_parquet(FEATURE_V1_FILE)
    if df.empty:
        raise RuntimeError("feature_matrix_v1.parquet is empty")

    out_df, manifest = build_feature_matrix_v2(df)
    diag = build_diagnostics(out_df, manifest)
    keep_features = diag.loc[diag["keep"] == 1, "feature"].astype(str).tolist()
    base_cols = [c for c in out_df.columns if c not in manifest["name"].astype(str).tolist()]
    final_df = out_df[base_cols + keep_features].copy()

    FEATURE_V2_OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    FEATURE_V2_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    FEATURE_V2_DIAG_JSON.parent.mkdir(parents=True, exist_ok=True)

    final_df.to_parquet(FEATURE_V2_OUT_FILE, index=False)
    manifest.to_csv(FEATURE_V2_MANIFEST, index=False)
    diag.to_csv(FEATURE_V2_DIAG_CSV, index=False)

    meta = {
        "rows": int(len(final_df)),
        "columns": int(len(final_df.columns)),
        "new_feature_requested": int(len(manifest)),
        "new_feature_kept": int(len(keep_features)),
        "new_feature_dropped": int(len(manifest) - len(keep_features)),
        "clusters": diag.groupby("cluster", sort=False)["keep"].sum().astype(int).to_dict() if len(diag) else {},
        "inputs": {
            "feature_v1_file": str(FEATURE_V1_FILE),
            "feature_v2_out_file": str(FEATURE_V2_OUT_FILE),
        },
        "thresholds": {
            "min_non_na": MIN_NON_NA,
            "max_nan_ratio": MAX_NAN_RATIO,
            "min_unique": MIN_UNIQUE,
        },
    }
    FEATURE_V2_DIAG_JSON.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[FEATURE_V2] rows={len(final_df)} cols={len(final_df.columns)} new_kept={len(keep_features)} new_dropped={len(manifest) - len(keep_features)}")
    if len(diag):
        print("[FEATURE_V2][TOP_KEPT]")
        print(diag.loc[diag["keep"] == 1].head(TOPK_PRINT).to_string(index=False))
        dropped = diag.loc[diag["keep"] != 1].copy()
        if len(dropped):
            print("[FEATURE_V2][TOP_DROPPED]")
            print(dropped.head(min(TOPK_PRINT, len(dropped))).to_string(index=False))

    print(f"[ARTIFACT] {FEATURE_V2_OUT_FILE}")
    print(f"[ARTIFACT] {FEATURE_V2_MANIFEST}")
    print(f"[ARTIFACT] {FEATURE_V2_DIAG_CSV}")
    print(f"[ARTIFACT] {FEATURE_V2_DIAG_JSON}")
    print("[FINAL] feature_matrix_v2 build complete")
    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 1
    _safe_exit(rc)
