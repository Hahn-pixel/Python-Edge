from __future__ import annotations

import glob
import json
import math
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_COL = os.getenv("DIAG_TARGET_COL", "target_fwd_ret_1d")
WEIGHT_COL = os.getenv("DIAG_WEIGHT_COL", "weight")
SCORE_COL = os.getenv("DIAG_SCORE_COL", "score")
DATE_COL = os.getenv("DIAG_DATE_COL", "date")
SYMBOL_COL = os.getenv("DIAG_SYMBOL_COL", "symbol")
FOLD_COL = os.getenv("DIAG_FOLD_COL", "fold_id")
PORTFOLIO_GLOB = os.getenv("DIAG_PORTFOLIO_GLOB", r"data/diagnostics/portfolio__diag_barbell_no_exits__fold*.parquet")
OUT_DIR = Path(os.getenv("DIAG_OUT_DIR", r"data/diagnostics"))
DIAG_NAME = str(os.getenv("DIAG_NAME", "diag_barbell_no_exits")).strip() or "diag_barbell_no_exits"
DELAYS = [int(x.strip()) for x in os.getenv("DIAG_DELAYS", "0,1,2").split(",") if x.strip()]
COST_MULTS = [float(x.strip()) for x in os.getenv("DIAG_COST_MULTS", "0,0.5,1,1.5,2").split(",") if x.strip()]
HORIZONS = [int(x.strip()) for x in os.getenv("DIAG_HORIZONS", "1,2,3,5").split(",") if x.strip()]
TOP_PCT = float(os.getenv("DIAG_TOP_PCT", "0.10"))
BOTTOM_PCT = float(os.getenv("DIAG_BOTTOM_PCT", str(TOP_PCT)))
QUANTILES = int(os.getenv("DIAG_QUANTILES", "10"))
EXPORT_JSON = str(os.getenv("DIAG_EXPORT_JSON", "1")).strip() == "1"
WAIT_ON_EXIT = str(os.getenv("DIAG_WAIT_ON_EXIT", "1")).strip() == "1"
PAUSE_ON_EXIT_ENV = str(os.getenv("PAUSE_ON_EXIT", "auto")).strip().lower()

EPS = 1e-12


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


def _should_pause_on_exit() -> bool:
    if PAUSE_ON_EXIT_ENV in {"0", "false", "no", "off"}:
        return False
    if PAUSE_ON_EXIT_ENV in {"1", "true", "yes", "on"}:
        return True
    if not WAIT_ON_EXIT:
        return False
    stdin_obj = getattr(sys, "stdin", None)
    stdout_obj = getattr(sys, "stdout", None)
    stdin_is_tty = bool(stdin_obj is not None and hasattr(stdin_obj, "isatty") and stdin_obj.isatty())
    stdout_is_tty = bool(stdout_obj is not None and hasattr(stdout_obj, "isatty") and stdout_obj.isatty())
    return stdin_is_tty and stdout_is_tty


def _print(tag: str, **kwargs: float | int | str) -> None:
    parts = [tag]
    for key, value in kwargs.items():
        if isinstance(value, float):
            if math.isfinite(value):
                parts.append(f"{key}={value:.6f}")
            else:
                parts.append(f"{key}=nan")
        else:
            parts.append(f"{key}={value}")
    print(" ".join(parts))


def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("float64")


def _ann_sharpe(ret: pd.Series) -> float:
    s = _safe_num(ret).dropna()
    if s.empty:
        return float("nan")
    std = float(s.std())
    if std <= EPS:
        return float("nan")
    return float((float(s.mean()) / std) * math.sqrt(252.0))


def _cum_ret(ret: pd.Series) -> float:
    s = _safe_num(ret).fillna(0.0)
    if s.empty:
        return 0.0
    return float((1.0 + s).prod() - 1.0)


def _read_inputs() -> tuple[pd.DataFrame, list[str]]:
    paths = sorted(glob.glob(PORTFOLIO_GLOB))
    if not paths:
        raise FileNotFoundError(f"No parquet files matched: {PORTFOLIO_GLOB}")
    _print("[DIAG][INPUT]", files=len(paths), glob=PORTFOLIO_GLOB)
    frames: list[pd.DataFrame] = []
    for i, path in enumerate(paths, start=1):
        _print("[DIAG][INPUT_FILE]", i=i, path=path)
        frame = pd.read_parquet(path)
        frame["__source_file"] = path
        if FOLD_COL not in frame.columns:
            frame[FOLD_COL] = i
        frames.append(frame)
    df = pd.concat(frames, axis=0, ignore_index=True)
    return df, paths


def _validate_and_prepare(df: pd.DataFrame) -> pd.DataFrame:
    required = [DATE_COL, SYMBOL_COL, WEIGHT_COL, SCORE_COL, TARGET_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    out = df.copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL])
    out[WEIGHT_COL] = _safe_num(out[WEIGHT_COL]).fillna(0.0)
    out[SCORE_COL] = _safe_num(out[SCORE_COL])
    out[TARGET_COL] = _safe_num(out[TARGET_COL])

    for col in ["cost_trading", "cost_borrow", "cost_total", "raw_turnover", "capped_turnover", "cap_hit", "turnover_budget_left", "cs_dispersion", "cs_top_bottom_spread", "cs_signal_breadth", "cs_nonzero_frac", "cs_signal_count", "score_conf"]:
        if col in out.columns:
            out[col] = _safe_num(out[col])

    if "cost_trading" not in out.columns:
        out["cost_trading"] = 0.0
    if "cost_borrow" not in out.columns:
        out["cost_borrow"] = 0.0
    if "cost_total" not in out.columns:
        out["cost_total"] = out["cost_trading"].fillna(0.0) + out["cost_borrow"].fillna(0.0)

    out = out.sort_values([DATE_COL, SYMBOL_COL, FOLD_COL]).reset_index(drop=True)
    return out


def _daily_from_contrib(df: pd.DataFrame, net_col: str) -> pd.DataFrame:
    tmp = df.copy()
    tmp["gross_ret_contrib"] = tmp[WEIGHT_COL] * tmp[TARGET_COL].fillna(0.0)
    tmp[net_col] = tmp["gross_ret_contrib"] - tmp["cost_total"].fillna(0.0)
    tmp["long_contrib"] = np.where(tmp[WEIGHT_COL] > 0.0, tmp["gross_ret_contrib"], 0.0)
    tmp["short_contrib"] = np.where(tmp[WEIGHT_COL] < 0.0, tmp["gross_ret_contrib"], 0.0)
    grouped = tmp.groupby(DATE_COL, as_index=False).agg(
        portfolio_ret=(net_col, "sum"),
        gross_ret=("gross_ret_contrib", "sum"),
        long_ret=("long_contrib", "sum"),
        short_ret=("short_contrib", "sum"),
        rows=(SYMBOL_COL, "size"),
    )
    if "raw_turnover" in tmp.columns:
        grouped["raw_turnover"] = tmp.groupby(DATE_COL)["raw_turnover"].mean().values
    if "capped_turnover" in tmp.columns:
        grouped["capped_turnover"] = tmp.groupby(DATE_COL)["capped_turnover"].mean().values
    if "cap_hit" in tmp.columns:
        grouped["cap_hit_rate"] = tmp.groupby(DATE_COL)["cap_hit"].mean().values
    if "turnover_budget_left" in tmp.columns:
        grouped["turnover_budget_left"] = tmp.groupby(DATE_COL)["turnover_budget_left"].mean().values
    if "cs_dispersion" in tmp.columns:
        grouped["cs_dispersion"] = tmp.groupby(DATE_COL)["cs_dispersion"].mean().values
    return grouped


def _overall_summary(df: pd.DataFrame) -> dict[str, float]:
    daily = _daily_from_contrib(df, net_col="net_ret_contrib")
    s = _safe_num(daily["portfolio_ret"]).dropna()
    out: dict[str, float] = {
        "days": float(len(s)),
        "avg_daily_ret": float(s.mean()) if not s.empty else 0.0,
        "std_daily_ret": float(s.std()) if len(s) > 1 else 0.0,
        "win_rate_days": float((s > 0.0).mean()) if not s.empty else 0.0,
        "cum_ret": _cum_ret(s),
        "daily_sharpe": float((float(s.mean()) / float(s.std())) if len(s) > 1 and float(s.std()) > EPS else float("nan")),
        "ann_sharpe": _ann_sharpe(s),
    }
    for src, dst in [
        ("raw_turnover", "avg_raw_turnover"),
        ("capped_turnover", "avg_turnover"),
        ("cap_hit_rate", "cap_hit_rate"),
        ("turnover_budget_left", "avg_turnover_budget_left"),
        ("gross_ret", "avg_gross_ret"),
        ("cs_dispersion", "avg_cs_dispersion"),
    ]:
        if src in daily.columns:
            out[dst] = float(_safe_num(daily[src]).mean())
    return out


def _delay_test(df: pd.DataFrame, delays: list[int]) -> list[dict[str, float]]:
    base = df.copy()
    base = base.sort_values([SYMBOL_COL, DATE_COL]).reset_index(drop=True)
    results: list[dict[str, float]] = []
    for delay in delays:
        shifted = base.copy()
        shifted["delay_target"] = shifted.groupby(SYMBOL_COL)[TARGET_COL].shift(-delay)
        shifted = shifted.loc[shifted["delay_target"].notna()].copy()
        shifted["delay_gross"] = shifted[WEIGHT_COL] * shifted["delay_target"]
        shifted["delay_net"] = shifted["delay_gross"] - shifted["cost_total"].fillna(0.0)
        daily = shifted.groupby(DATE_COL, as_index=False).agg(portfolio_ret=("delay_net", "sum"))
        ret = _safe_num(daily["portfolio_ret"]).dropna()
        results.append({
            "delay": float(delay),
            "days": float(len(ret)),
            "avg_daily_ret": float(ret.mean()) if not ret.empty else 0.0,
            "cum_ret": _cum_ret(ret),
            "ann_sharpe": _ann_sharpe(ret),
        })
    return results


def _rank_persistence(df: pd.DataFrame) -> dict[str, float]:
    work = df[[DATE_COL, SYMBOL_COL, SCORE_COL]].copy()
    work = work.dropna(subset=[SCORE_COL]).sort_values([DATE_COL, SYMBOL_COL]).reset_index(drop=True)
    dates = list(pd.Index(work[DATE_COL].drop_duplicates()).sort_values())
    score_corrs: list[float] = []
    rank_corrs: list[float] = []
    top_overlaps: list[float] = []
    bottom_overlaps: list[float] = []

    by_date: dict[pd.Timestamp, pd.DataFrame] = {d: g[[SYMBOL_COL, SCORE_COL]].copy() for d, g in work.groupby(DATE_COL)}
    for i in range(len(dates) - 1):
        d0 = dates[i]
        d1 = dates[i + 1]
        a = by_date[d0].rename(columns={SCORE_COL: "score_a"})
        b = by_date[d1].rename(columns={SCORE_COL: "score_b"})
        merged = a.merge(b, on=SYMBOL_COL, how="inner")
        if merged.shape[0] < 5:
            continue
        sa = _safe_num(merged["score_a"])
        sb = _safe_num(merged["score_b"])
        score_corr = sa.corr(sb, method="pearson")
        if pd.notna(score_corr):
            score_corrs.append(float(score_corr))
        ra = sa.rank(method="average", pct=True)
        rb = sb.rank(method="average", pct=True)
        rank_corr = ra.corr(rb, method="pearson")
        if pd.notna(rank_corr):
            rank_corrs.append(float(rank_corr))

        top_n = max(1, int(round(float(merged.shape[0]) * TOP_PCT)))
        bot_n = max(1, int(round(float(merged.shape[0]) * BOTTOM_PCT)))
        top_a = set(merged.nlargest(top_n, "score_a")[SYMBOL_COL].tolist())
        top_b = set(merged.nlargest(top_n, "score_b")[SYMBOL_COL].tolist())
        bot_a = set(merged.nsmallest(bot_n, "score_a")[SYMBOL_COL].tolist())
        bot_b = set(merged.nsmallest(bot_n, "score_b")[SYMBOL_COL].tolist())
        top_overlaps.append(float(len(top_a & top_b)) / float(top_n))
        bottom_overlaps.append(float(len(bot_a & bot_b)) / float(bot_n))

    return {
        "mean_score_corr": float(np.mean(score_corrs)) if score_corrs else float("nan"),
        "mean_rank_corr": float(np.mean(rank_corrs)) if rank_corrs else float("nan"),
        "mean_top_overlap": float(np.mean(top_overlaps)) if top_overlaps else float("nan"),
        "mean_bottom_overlap": float(np.mean(bottom_overlaps)) if bottom_overlaps else float("nan"),
        "pairs": float(len(score_corrs)),
    }


def _cost_sensitivity(df: pd.DataFrame, mults: list[float]) -> list[dict[str, float]]:
    results: list[dict[str, float]] = []
    gross = (df[WEIGHT_COL] * df[TARGET_COL].fillna(0.0)).rename("gross")
    for mult in mults:
        net = gross - (df["cost_total"].fillna(0.0) * float(mult))
        daily = pd.DataFrame({DATE_COL: df[DATE_COL], "portfolio_ret": net}).groupby(DATE_COL, as_index=False)["portfolio_ret"].sum()
        ret = _safe_num(daily["portfolio_ret"]).dropna()
        results.append({
            "mult": float(mult),
            "days": float(len(ret)),
            "avg_daily_ret": float(ret.mean()) if not ret.empty else 0.0,
            "cum_ret": _cum_ret(ret),
            "ann_sharpe": _ann_sharpe(ret),
        })
    return results


def _forward_cumret_from_1d(series: pd.Series, horizon: int) -> pd.Series:
    s = _safe_num(series).fillna(0.0)
    out = pd.Series(np.nan, index=s.index, dtype="float64")
    n = len(s)
    if horizon <= 0:
        return out
    vals = s.to_numpy(dtype="float64")
    for i in range(0, max(0, n - horizon + 1)):
        out.iat[i] = float(np.prod(1.0 + vals[i:i + horizon]) - 1.0)
    return out


def _alpha_decay(df: pd.DataFrame, horizons: list[int]) -> list[dict[str, float]]:
    work = df[[DATE_COL, SYMBOL_COL, SCORE_COL, TARGET_COL]].copy().sort_values([SYMBOL_COL, DATE_COL]).reset_index(drop=True)
    for h in horizons:
        work[f"fwd_{h}d"] = work.groupby(SYMBOL_COL)[TARGET_COL].transform(lambda s: _forward_cumret_from_1d(s, h))

    results: list[dict[str, float]] = []
    for h in horizons:
        col = f"fwd_{h}d"
        top_vals: list[float] = []
        bot_vals: list[float] = []
        used_days = 0
        for _d, g in work.groupby(DATE_COL):
            gg = g[[SCORE_COL, col]].dropna().copy()
            if gg.empty:
                continue
            n = gg.shape[0]
            top_n = max(1, int(round(float(n) * TOP_PCT)))
            bot_n = max(1, int(round(float(n) * BOTTOM_PCT)))
            top_vals.append(float(gg.nlargest(top_n, SCORE_COL)[col].mean()))
            bot_vals.append(float(gg.nsmallest(bot_n, SCORE_COL)[col].mean()))
            used_days += 1
        results.append({
            "horizon_days": float(h),
            "days": float(used_days),
            "avg_top_ret": float(np.mean(top_vals)) if top_vals else float("nan"),
            "avg_bottom_ret": float(np.mean(bot_vals)) if bot_vals else float("nan"),
            "avg_top_bottom_spread": float(np.mean(np.array(top_vals) - np.array(bot_vals))) if top_vals and bot_vals else float("nan"),
        })
    return results


def _concentration(df: pd.DataFrame) -> dict[str, float]:
    top1: list[float] = []
    top5: list[float] = []
    top10: list[float] = []
    for _d, g in df.groupby(DATE_COL):
        abs_w = _safe_num(g[WEIGHT_COL]).abs().sort_values(ascending=False).reset_index(drop=True)
        gross = float(abs_w.sum())
        if gross <= EPS:
            continue
        shares = abs_w / gross
        top1.append(float(shares.iloc[:1].sum()))
        top5.append(float(shares.iloc[:5].sum()))
        top10.append(float(shares.iloc[:10].sum()))
    return {
        "top1_share": float(np.mean(top1)) if top1 else float("nan"),
        "top5_share": float(np.mean(top5)) if top5 else float("nan"),
        "top10_share": float(np.mean(top10)) if top10 else float("nan"),
        "max_abs_score": float(_safe_num(df[SCORE_COL]).abs().max()),
    }


def _quantile_monotonicity(df: pd.DataFrame, quantiles: int = 10) -> dict[str, float]:
    slopes_1d: list[float] = []
    monotonic_days = 0
    used_days = 0
    for _d, g in df.groupby(DATE_COL):
        gg = g[[SCORE_COL, TARGET_COL]].dropna().copy()
        if gg.shape[0] < max(quantiles, 10):
            continue
        try:
            gg["q"] = pd.qcut(gg[SCORE_COL], q=quantiles, labels=False, duplicates="drop")
        except Exception:
            continue
        if gg["q"].nunique() < max(4, quantiles // 2):
            continue
        curve = gg.groupby("q", as_index=False)[TARGET_COL].mean().sort_values("q")
        x = curve["q"].astype("float64")
        y = _safe_num(curve[TARGET_COL])
        if len(x) < 2:
            continue
        coef = np.polyfit(x.to_numpy(), y.to_numpy(), 1)
        slope = float(coef[0])
        slopes_1d.append(slope)
        diffs = np.diff(y.to_numpy())
        if bool(np.all(diffs >= -1e-12)):
            monotonic_days += 1
        used_days += 1
    return {
        "days": float(used_days),
        "mean_slope_1d": float(np.mean(slopes_1d)) if slopes_1d else float("nan"),
        "monotonic_up_days_frac": float(monotonic_days / used_days) if used_days > 0 else float("nan"),
    }


def _breadth_dispersion(df: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    cols = [
        "cs_dispersion",
        "cs_top_bottom_spread",
        "cs_signal_breadth",
        "cs_nonzero_frac",
        "cs_signal_count",
        "score_conf",
    ]
    for col in cols:
        if col not in df.columns:
            continue
        daily = _safe_num(df.groupby(DATE_COL)[col].mean())
        out[f"{col}_mean"] = float(daily.mean())
        out[f"{col}_p05"] = float(daily.quantile(0.05))
        out[f"{col}_p95"] = float(daily.quantile(0.95))
    if "cs_dispersion" in df.columns:
        disp_daily = _safe_num(df.groupby(DATE_COL)["cs_dispersion"].mean())
        out["dispersion_gt_1p5_days"] = float((disp_daily > 1.5).sum())
        out["dispersion_lt_0p25_days"] = float((disp_daily < 0.25).sum())
    return out


def _long_short(df: pd.DataFrame) -> dict[str, float]:
    work = df.copy()
    work["gross_ret_contrib"] = work[WEIGHT_COL] * work[TARGET_COL].fillna(0.0)
    long_daily = work.loc[work[WEIGHT_COL] > 0.0].groupby(DATE_COL)["gross_ret_contrib"].sum()
    short_daily = work.loc[work[WEIGHT_COL] < 0.0].groupby(DATE_COL)["gross_ret_contrib"].sum()
    return {
        "long_avg_daily_ret": float(_safe_num(long_daily).mean()) if len(long_daily) else float("nan"),
        "long_cum_ret": _cum_ret(long_daily),
        "short_avg_daily_ret": float(_safe_num(short_daily).mean()) if len(short_daily) else float("nan"),
        "short_cum_ret": _cum_ret(short_daily),
    }


def _turnover(df: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    if "raw_turnover" in df.columns:
        out["avg_raw_turnover"] = float(_safe_num(df["raw_turnover"]).mean())
    if "capped_turnover" in df.columns:
        out["avg_capped_turnover"] = float(_safe_num(df["capped_turnover"]).mean())
    if "cap_hit" in df.columns:
        out["avg_cap_hit_rate"] = float(_safe_num(df["cap_hit"]).mean())
        out["full_bind_days"] = float((_safe_num(df.groupby(DATE_COL)["cap_hit"].mean()) >= 0.999999).sum())
    if "turnover_budget_left" in df.columns:
        out["avg_turnover_budget_left"] = float(_safe_num(df["turnover_budget_left"]).mean())
    if "raw_turnover" in df.columns and "capped_turnover" in df.columns:
        out["avg_turnover_clipped"] = float((_safe_num(df["raw_turnover"]) - _safe_num(df["capped_turnover"])).mean())
    return out


def _universe_stability(df: pd.DataFrame) -> dict[str, float]:
    dates = sorted(pd.Index(df[DATE_COL].drop_duplicates()))
    by_date = {d: set(df.loc[df[DATE_COL] == d, SYMBOL_COL].astype(str).tolist()) for d in dates}
    counts = [float(len(v)) for v in by_date.values()]
    overlaps: list[float] = []
    entries: list[float] = []
    exits: list[float] = []
    for i in range(len(dates) - 1):
        a = by_date[dates[i]]
        b = by_date[dates[i + 1]]
        denom = float(max(1, len(a | b)))
        overlaps.append(float(len(a & b)) / denom)
        entries.append(float(len(b - a)))
        exits.append(float(len(a - b)))
    return {
        "days": float(len(counts)),
        "symbols_mean": float(np.mean(counts)) if counts else float("nan"),
        "symbols_min": float(np.min(counts)) if counts else float("nan"),
        "symbols_max": float(np.max(counts)) if counts else float("nan"),
        "mean_jaccard_overlap": float(np.mean(overlaps)) if overlaps else float("nan"),
        "mean_entries": float(np.mean(entries)) if entries else float("nan"),
        "mean_exits": float(np.mean(exits)) if exits else float("nan"),
    }


def _per_fold_summary(df: pd.DataFrame) -> list[dict[str, float]]:
    if FOLD_COL not in df.columns:
        return []
    out: list[dict[str, float]] = []
    for fold_id, g in df.groupby(FOLD_COL):
        summ = _overall_summary(g)
        out.append({
            "fold_id": float(fold_id),
            "days": float(summ.get("days", 0.0)),
            "avg_daily_ret": float(summ.get("avg_daily_ret", 0.0)),
            "cum_ret": float(summ.get("cum_ret", 0.0)),
            "ann_sharpe": float(summ.get("ann_sharpe", float("nan"))),
            "avg_raw_turnover": float(summ.get("avg_raw_turnover", 0.0)),
            "avg_turnover": float(summ.get("avg_turnover", 0.0)),
            "cap_hit_rate": float(summ.get("cap_hit_rate", 0.0)),
        })
    return out


def _debug_counters(df: pd.DataFrame) -> dict[str, float]:
    grouped_score_std = _safe_num(df.groupby(DATE_COL)[SCORE_COL].std())
    out = {
        "rows": float(len(df)),
        "days": float(df[DATE_COL].nunique()),
        "symbols": float(df[SYMBOL_COL].nunique()),
        "duplicate_date_symbol_rows": float(df.duplicated(subset=[DATE_COL, SYMBOL_COL]).sum()),
        "nan_close_rows": float(df[ TARGET_COL ].isna().sum()),
        "nan_weight_rows": float(df[WEIGHT_COL].isna().sum()),
        "nan_score_rows": float(df[SCORE_COL].isna().sum()),
        "zero_dispersion_days": float((grouped_score_std.fillna(0.0) <= EPS).sum()),
        "all_equal_score_days": float((grouped_score_std.fillna(0.0) <= EPS).sum()),
        "turnover_cap_hit_days": float((_safe_num(df.groupby(DATE_COL)["cap_hit"].mean()) >= 0.999999).sum()) if "cap_hit" in df.columns else 0.0,
    }
    return out


def _risk_flags(overall: dict[str, float], delay_rows: list[dict[str, float]], persist: dict[str, float], turnover: dict[str, float], breadth: dict[str, float]) -> dict[str, float | int]:
    delay0 = next((r for r in delay_rows if int(r["delay"]) == 0), None)
    delay1 = next((r for r in delay_rows if int(r["delay"]) == 1), None)
    delay2 = next((r for r in delay_rows if int(r["delay"]) == 2), None)
    delay1_ratio = float("nan")
    if delay0 is not None and delay1 is not None:
        base = float(delay0.get("avg_daily_ret", 0.0))
        d1 = float(delay1.get("avg_daily_ret", 0.0))
        if abs(base) > EPS:
            delay1_ratio = d1 / base
    out: dict[str, float | int] = {
        "flag_delay_flip": int(delay0 is not None and delay1 is not None and float(delay0["avg_daily_ret"]) > 0.0 and float(delay1["avg_daily_ret"]) < 0.0),
        "flag_rank_instability": int(pd.notna(persist.get("mean_rank_corr", float("nan"))) and float(persist.get("mean_rank_corr", 0.0)) < 0.10),
        "flag_turnover_full_bind": int(float(turnover.get("avg_cap_hit_rate", 0.0)) >= 0.999999),
        "flag_dense_signal": int(float(breadth.get("cs_nonzero_frac_mean", 0.0)) >= 0.95),
        "flag_high_dispersion": int(float(breadth.get("cs_dispersion_mean", 0.0)) >= 1.25),
        "flag_leakage_not_proven": 1,
    }
    if pd.notna(delay1_ratio):
        out["delay1_ret_ratio_vs_delay0"] = float(delay1_ratio)
    if delay2 is not None:
        out["delay2_avg_daily_ret"] = float(delay2["avg_daily_ret"])
    out["overall_ann_sharpe"] = float(overall.get("ann_sharpe", float("nan")))
    return out


def main() -> int:
    _enable_line_buffering()
    df_raw, paths = _read_inputs()
    df = _validate_and_prepare(df_raw)

    debug = _debug_counters(df)
    for k, v in debug.items():
        _print("[DIAG][DEBUG]", metric=k, value=v)

    overall = _overall_summary(df)
    _print("[DIAG][OVERALL]", **overall)

    delay_rows = _delay_test(df, DELAYS)
    for row in delay_rows:
        _print("[DIAG][DELAY]", **row)

    persist = _rank_persistence(df)
    _print("[DIAG][PERSIST]", **persist)

    cost_rows = _cost_sensitivity(df, COST_MULTS)
    for row in cost_rows:
        _print("[DIAG][COST]", **row)

    decay_rows = _alpha_decay(df, HORIZONS)
    for row in decay_rows:
        _print("[DIAG][DECAY]", **row)

    concentration = _concentration(df)
    _print("[DIAG][CONCENTRATION]", **concentration)

    breadth = _breadth_dispersion(df)
    _print("[DIAG][BREADTH]", **breadth)

    ls = _long_short(df)
    _print("[DIAG][LS]", **ls)

    turnover = _turnover(df)
    _print("[DIAG][TURNOVER]", **turnover)

    monotonicity = _quantile_monotonicity(df, quantiles=QUANTILES)
    _print("[DIAG][QUANTILES]", **monotonicity)

    universe = _universe_stability(df)
    _print("[DIAG][UNIVERSE]", **universe)

    fold_rows = _per_fold_summary(df)
    for row in fold_rows:
        _print("[DIAG][FOLD]", **row)

    flags = _risk_flags(overall, delay_rows, persist, turnover, breadth)
    _print("[DIAG][FLAGS]", **flags)

    report = {
        "config": {
            "diag_name": DIAG_NAME,
            "portfolio_glob": PORTFOLIO_GLOB,
            "target_col": TARGET_COL,
            "weight_col": WEIGHT_COL,
            "score_col": SCORE_COL,
            "date_col": DATE_COL,
            "symbol_col": SYMBOL_COL,
            "fold_col": FOLD_COL,
            "delays": DELAYS,
            "cost_mults": COST_MULTS,
            "horizons": HORIZONS,
            "top_pct": TOP_PCT,
            "bottom_pct": BOTTOM_PCT,
            "quantiles": QUANTILES,
            "input_files": paths,
        },
        "debug": debug,
        "overall": overall,
        "delay": delay_rows,
        "persistence": persist,
        "cost": cost_rows,
        "decay": decay_rows,
        "concentration": concentration,
        "breadth": breadth,
        "long_short": ls,
        "turnover": turnover,
        "quantiles": monotonicity,
        "universe": universe,
        "folds": fold_rows,
        "flags": flags,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if EXPORT_JSON:
        out_path = OUT_DIR / f"{DIAG_NAME}__summary.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        _print("[DIAG][EXPORT]", json=str(out_path))

    print("[DIAG] completed successfully")
    return 0


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    except Exception:
        print("[ERROR] Unhandled exception:\n")
        traceback.print_exc()
        exit_code = 1
    finally:
        if _should_pause_on_exit():
            try:
                input("\nPress Enter to exit...")
            except EOFError:
                pass
    raise SystemExit(exit_code)
