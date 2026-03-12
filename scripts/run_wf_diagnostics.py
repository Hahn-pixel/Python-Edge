from __future__ import annotations

import json
import math
import os
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_PORTFOLIO_GLOB = r"data/diagnostics/portfolio__*.parquet"
DEFAULT_DELAYS = (0, 1, 2)
DEFAULT_COST_MULTS = (0.0, 0.5, 1.0, 1.5, 2.0)
DEFAULT_HORIZONS = (1, 2, 3, 5)
DEFAULT_TOP_PCT = 0.10
DEFAULT_NAME = "wf_diag"
DEFAULT_OUT_DIR = Path("data/diagnostics")


@dataclass
class DiagContext:
    portfolio_glob: str
    out_dir: Path
    name: str
    top_pct: float
    delays: tuple[int, ...]
    cost_mults: tuple[float, ...]
    horizons: tuple[int, ...]
    export_json: bool
    wait_on_exit: bool


class ConfigError(RuntimeError):
    pass


class DataError(RuntimeError):
    pass


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value if value else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if raw == "":
        return default
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    raise ConfigError(f"Invalid boolean env {name}={raw!r}")


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"Invalid float env {name}={raw!r}") from exc


def _parse_int_tuple(raw: str, default: tuple[int, ...]) -> tuple[int, ...]:
    text = raw.strip()
    if text == "":
        return default
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if part == "":
            continue
        out.append(int(part))
    if not out:
        raise ConfigError("Empty integer tuple after parsing")
    return tuple(out)


def _parse_float_tuple(raw: str, default: tuple[float, ...]) -> tuple[float, ...]:
    text = raw.strip()
    if text == "":
        return default
    out: list[float] = []
    for part in text.split(","):
        part = part.strip()
        if part == "":
            continue
        out.append(float(part))
    if not out:
        raise ConfigError("Empty float tuple after parsing")
    return tuple(out)


def _print(tag: str, **kwargs: object) -> None:
    parts = [tag]
    for key, value in kwargs.items():
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                parts.append(f"{key}=nan")
            else:
                parts.append(f"{key}={value:.6f}")
        else:
            parts.append(f"{key}={value}")
    print(" ".join(parts))


def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    aa = pd.to_numeric(a, errors="coerce")
    bb = pd.to_numeric(b, errors="coerce")
    mask = aa.notna() & bb.notna()
    if int(mask.sum()) < 5:
        return float("nan")
    aa = aa.loc[mask]
    bb = bb.loc[mask]
    if float(aa.std()) == 0.0 or float(bb.std()) == 0.0:
        return float("nan")
    return float(aa.corr(bb))


def _summarize_daily_return_series(s: pd.Series) -> dict[str, float]:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if x.empty:
        return {}
    out: dict[str, float] = {}
    out["days"] = float(len(x))
    out["avg_daily_ret"] = float(x.mean())
    out["std_daily_ret"] = float(x.std())
    out["win_rate_days"] = float((x > 0.0).mean())
    out["cum_ret"] = float((1.0 + x).prod() - 1.0)
    if float(x.std()) > 0.0:
        out["daily_sharpe"] = float(x.mean() / x.std())
        out["ann_sharpe"] = float((x.mean() / x.std()) * math.sqrt(252.0))
    else:
        out["daily_sharpe"] = float("nan")
        out["ann_sharpe"] = float("nan")
    return out


def _read_single_parquet(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception:
        try:
            import pyarrow.parquet as pq  # type: ignore

            return pq.read_table(path).to_pandas()
        except Exception as exc:
            raise DataError(
                f"Failed to read parquet file {path}. Install pyarrow if needed. Inner error: {exc}"
            ) from exc


def load_portfolio_frames(paths: Iterable[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        df = _read_single_parquet(path)
        df["source_file"] = str(path)
        frames.append(df)
    if not frames:
        raise DataError("No parquet frames loaded")
    out = pd.concat(frames, axis=0, ignore_index=True)
    if "date" not in out.columns:
        raise DataError("Portfolio parquet is missing required column: date")
    if "symbol" not in out.columns:
        raise DataError("Portfolio parquet is missing required column: symbol")
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if str(out["date"].dtype).startswith("datetime64[ns,"):
        out["date"] = out["date"].dt.tz_localize(None)
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
        out["timestamp"] = out["timestamp"].dt.tz_localize(None)
    if "session_date" in out.columns:
        out["session_date"] = pd.to_datetime(out["session_date"], errors="coerce")
        if str(out["session_date"].dtype).startswith("datetime64[ns,"):
            out["session_date"] = out["session_date"].dt.tz_localize(None)
    out = out.sort_values(["date", "symbol", "source_file"], ascending=[True, True, True]).reset_index(drop=True)
    return out


def build_context() -> DiagContext:
    portfolio_glob = _env_str("DIAG_PORTFOLIO_GLOB", DEFAULT_PORTFOLIO_GLOB)
    out_dir = Path(_env_str("DIAG_OUT_DIR", str(DEFAULT_OUT_DIR)))
    name = _env_str("DIAG_NAME", DEFAULT_NAME)
    top_pct = _env_float("DIAG_TOP_PCT", DEFAULT_TOP_PCT)
    delays = _parse_int_tuple(os.environ.get("DIAG_DELAYS", ""), DEFAULT_DELAYS)
    cost_mults = _parse_float_tuple(os.environ.get("DIAG_COST_MULTS", ""), DEFAULT_COST_MULTS)
    horizons = _parse_int_tuple(os.environ.get("DIAG_HORIZONS", ""), DEFAULT_HORIZONS)
    export_json = _env_bool("DIAG_EXPORT_JSON", True)
    wait_on_exit = _env_bool("DIAG_WAIT_ON_EXIT", True)
    if not (0.0 < top_pct < 0.5):
        raise ConfigError(f"DIAG_TOP_PCT must be in (0, 0.5), got {top_pct}")
    if any(x < 0 for x in delays):
        raise ConfigError(f"DIAG_DELAYS must be >= 0, got {delays}")
    if any(x <= 0 for x in horizons):
        raise ConfigError(f"DIAG_HORIZONS must be > 0, got {horizons}")
    return DiagContext(
        portfolio_glob=portfolio_glob,
        out_dir=out_dir,
        name=name,
        top_pct=top_pct,
        delays=delays,
        cost_mults=cost_mults,
        horizons=horizons,
        export_json=export_json,
        wait_on_exit=wait_on_exit,
    )


def resolve_input_paths(ctx: DiagContext) -> list[Path]:
    paths = sorted(Path().glob(ctx.portfolio_glob))
    if not paths:
        raise DataError(f"No files matched DIAG_PORTFOLIO_GLOB={ctx.portfolio_glob!r}")
    return paths


def add_base_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    required = ["date", "symbol", "weight", "close"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise DataError(f"Missing required columns: {missing}")
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(0.0)
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    if "score" in out.columns:
        out["score"] = pd.to_numeric(out["score"], errors="coerce")
    if "score_rank_pct" in out.columns:
        out["score_rank_pct"] = pd.to_numeric(out["score_rank_pct"], errors="coerce")
    elif "rank_pct" in out.columns:
        out["score_rank_pct"] = pd.to_numeric(out["rank_pct"], errors="coerce")
    else:
        out["score_rank_pct"] = out.groupby("date")["score"].rank(method="average", pct=True)
    if "cost_total" not in out.columns:
        out["cost_total"] = 0.0
    out["cost_total"] = pd.to_numeric(out["cost_total"], errors="coerce").fillna(0.0)
    if "target_fwd_ret_1d" in out.columns:
        out["target_fwd_ret_1d"] = pd.to_numeric(out["target_fwd_ret_1d"], errors="coerce")
    else:
        sym = out[["symbol", "date", "close"]].drop_duplicates().sort_values(["symbol", "date"], ascending=[True, True]).copy()
        sym["target_fwd_ret_1d"] = sym.groupby("symbol")["close"].shift(-1) / sym["close"] - 1.0
        out = out.merge(sym[["symbol", "date", "target_fwd_ret_1d"]], on=["symbol", "date"], how="left")
    out["gross_ret_contrib"] = out["weight"] * out["target_fwd_ret_1d"].fillna(0.0)
    out["net_ret_contrib"] = out["gross_ret_contrib"] - out["cost_total"]
    return out


def collect_debug_counters(df: pd.DataFrame) -> dict[str, int]:
    counters: dict[str, int] = {}
    dup = df.duplicated(subset=["date", "symbol"], keep=False)
    counters["rows"] = int(len(df))
    counters["days"] = int(df["date"].nunique())
    counters["symbols"] = int(df["symbol"].nunique())
    counters["duplicate_date_symbol_rows"] = int(dup.sum())
    counters["nan_close_rows"] = int(pd.to_numeric(df["close"], errors="coerce").isna().sum())
    counters["nan_weight_rows"] = int(pd.to_numeric(df["weight"], errors="coerce").isna().sum())
    counters["nan_score_rows"] = int(pd.to_numeric(df.get("score"), errors="coerce").isna().sum()) if "score" in df.columns else -1
    if "cs_dispersion" in df.columns:
        daily_disp = df.groupby("date")["cs_dispersion"].mean()
        counters["zero_dispersion_days"] = int((pd.to_numeric(daily_disp, errors="coerce").fillna(0.0) == 0.0).sum())
    else:
        counters["zero_dispersion_days"] = -1
    if "score" in df.columns:
        all_equal_days = 0
        for _, g in df.groupby("date", sort=True):
            s = pd.to_numeric(g["score"], errors="coerce").dropna()
            if not s.empty and float(s.std()) == 0.0:
                all_equal_days += 1
        counters["all_equal_score_days"] = int(all_equal_days)
    else:
        counters["all_equal_score_days"] = -1
    if "cap_hit" in df.columns:
        counters["turnover_cap_hit_days"] = int((df.groupby("date")["cap_hit"].mean() > 0.0).sum())
    else:
        counters["turnover_cap_hit_days"] = -1
    return counters


def build_daily_panel(df: pd.DataFrame) -> pd.DataFrame:
    agg_spec: dict[str, tuple[str, str]] = {
        "portfolio_ret": ("net_ret_contrib", "sum"),
        "gross_ret": ("gross_ret_contrib", "sum"),
        "net_exposure": ("weight", "sum"),
        "gross_exposure": ("weight", lambda s: float(pd.to_numeric(s, errors="coerce").abs().sum())),
        "long_gross": ("weight", lambda s: float(pd.to_numeric(s, errors="coerce").clip(lower=0.0).sum())),
        "short_gross": ("weight", lambda s: float((-pd.to_numeric(s, errors="coerce").clip(upper=0.0)).sum())),
        "positions": ("weight", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0.0) != 0.0).sum())),
    }
    optional_mean_cols = [
        "raw_turnover",
        "capped_turnover",
        "cap_hit",
        "turnover_budget_left",
        "cost_total",
        "cost_trading",
        "cost_borrow",
        "cs_dispersion",
        "cs_top_bottom_spread",
        "cs_signal_breadth",
        "cs_nonzero_frac",
        "cs_signal_count",
        "score_conf",
        "score_abs_rank_pct",
        "risk_unit",
        "score_risk_adj",
        "score_alpha_to_risk",
        "risk_penalty_rate",
        "risk_beta_rank",
        "risk_vol_rank",
        "risk_liq_penalty",
        "risk_market_penalty",
        "cash_weight",
        "deployed_gross",
        "long_budget",
        "short_budget",
        "execution_participation",
        "execution_participation_flag",
    ]
    for col in optional_mean_cols:
        if col in df.columns:
            agg_spec[col] = (col, "mean")
    daily = df.groupby("date", as_index=False).agg(**agg_spec)
    rename_map = {
        "cap_hit": "cap_hit_rate",
        "execution_participation_flag": "participation_limit_hit_rate",
        "cost_total": "avg_cost_total_row",
        "cost_trading": "avg_cost_trading_row",
        "cost_borrow": "avg_cost_borrow_row",
    }
    daily = daily.rename(columns=rename_map)
    return daily


def compute_delay_return_columns(df: pd.DataFrame, delays: tuple[int, ...]) -> pd.DataFrame:
    out = df.copy()
    sym = out[["symbol", "date", "close"]].drop_duplicates().sort_values(["symbol", "date"], ascending=[True, True]).copy()
    for delay in delays:
        col = f"ret_delay_{delay}"
        entry_px = sym.groupby("symbol")["close"].shift(-delay)
        exit_px = sym.groupby("symbol")["close"].shift(-(delay + 1))
        sym[col] = exit_px / entry_px - 1.0
    merge_cols = ["symbol", "date"] + [f"ret_delay_{d}" for d in delays]
    out = out.merge(sym[merge_cols], on=["symbol", "date"], how="left")
    return out


def run_delay_diagnostics(df: pd.DataFrame, delays: tuple[int, ...]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for delay in delays:
        ret_col = f"ret_delay_{delay}"
        tmp = df[["date", "weight", "cost_total", ret_col]].copy()
        tmp["delay_net_ret"] = pd.to_numeric(tmp["weight"], errors="coerce").fillna(0.0) * pd.to_numeric(tmp[ret_col], errors="coerce").fillna(0.0) - pd.to_numeric(tmp["cost_total"], errors="coerce").fillna(0.0)
        daily = tmp.groupby("date")["delay_net_ret"].sum()
        summary = _summarize_daily_return_series(daily)
        out[str(delay)] = summary
        _print(
            "[DIAG][DELAY]",
            delay=delay,
            days=int(summary.get("days", 0.0)),
            avg_daily_ret=summary.get("avg_daily_ret", float("nan")),
            cum_ret=summary.get("cum_ret", float("nan")),
            ann_sharpe=summary.get("ann_sharpe", float("nan")),
        )
    return out


def run_rank_persistence(df: pd.DataFrame, top_pct: float) -> dict[str, float]:
    cols = ["date", "symbol", "score", "score_rank_pct"]
    x = df[cols].drop_duplicates(subset=["date", "symbol"]).sort_values(["date", "symbol"], ascending=[True, True])
    dates = list(x["date"].dropna().sort_values().unique())
    score_corrs: list[float] = []
    rank_corrs: list[float] = []
    top_overlaps: list[float] = []
    bottom_overlaps: list[float] = []
    for d0, d1 in zip(dates[:-1], dates[1:]):
        a = x.loc[x["date"] == d0].set_index("symbol")
        b = x.loc[x["date"] == d1].set_index("symbol")
        m = a.join(b, how="inner", lsuffix="_0", rsuffix="_1")
        if len(m) < 5:
            continue
        score_corr = _safe_corr(m["score_0"], m["score_1"])
        rank_corr = _safe_corr(m["score_rank_pct_0"], m["score_rank_pct_1"])
        if not math.isnan(score_corr):
            score_corrs.append(score_corr)
        if not math.isnan(rank_corr):
            rank_corrs.append(rank_corr)
        lo = top_pct
        hi = 1.0 - top_pct
        top0 = set(m.index[m["score_rank_pct_0"] >= hi])
        top1 = set(m.index[m["score_rank_pct_1"] >= hi])
        bot0 = set(m.index[m["score_rank_pct_0"] <= lo])
        bot1 = set(m.index[m["score_rank_pct_1"] <= lo])
        if top0 or top1:
            top_overlaps.append(float(len(top0 & top1) / max(1, len(top0 | top1))))
        if bot0 or bot1:
            bottom_overlaps.append(float(len(bot0 & bot1) / max(1, len(bot0 | bot1))))
    out = {
        "mean_score_corr": float(pd.Series(score_corrs).mean()) if score_corrs else float("nan"),
        "mean_rank_corr": float(pd.Series(rank_corrs).mean()) if rank_corrs else float("nan"),
        "mean_top_overlap": float(pd.Series(top_overlaps).mean()) if top_overlaps else float("nan"),
        "mean_bottom_overlap": float(pd.Series(bottom_overlaps).mean()) if bottom_overlaps else float("nan"),
        "pairs": float(len(score_corrs)),
    }
    _print("[DIAG][PERSIST]", **out)
    return out


def run_cost_sensitivity(df: pd.DataFrame, cost_mults: tuple[float, ...]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    weight = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)
    gross = weight * pd.to_numeric(df["target_fwd_ret_1d"], errors="coerce").fillna(0.0)
    costs = pd.to_numeric(df["cost_total"], errors="coerce").fillna(0.0)
    dates = df["date"]
    for mult in cost_mults:
        daily = (gross - costs * mult).groupby(dates).sum()
        summary = _summarize_daily_return_series(daily)
        out[f"{mult:g}"] = summary
        _print(
            "[DIAG][COST]",
            mult=mult,
            days=int(summary.get("days", 0.0)),
            avg_daily_ret=summary.get("avg_daily_ret", float("nan")),
            cum_ret=summary.get("cum_ret", float("nan")),
            ann_sharpe=summary.get("ann_sharpe", float("nan")),
        )
    return out


def run_alpha_decay(df: pd.DataFrame, top_pct: float, horizons: tuple[int, ...]) -> dict[str, dict[str, float]]:
    sym = df[["date", "symbol", "close", "score"]].drop_duplicates(subset=["date", "symbol"]).sort_values(["symbol", "date"], ascending=[True, True]).copy()
    sym["close"] = pd.to_numeric(sym["close"], errors="coerce")
    sym["score"] = pd.to_numeric(sym["score"], errors="coerce")
    for horizon in horizons:
        sym[f"fwd_{horizon}d"] = sym.groupby("symbol")["close"].shift(-horizon) / sym["close"] - 1.0
    out: dict[str, dict[str, float]] = {}
    lo = top_pct
    hi = 1.0 - top_pct
    for horizon in horizons:
        day_spreads: list[float] = []
        top_means: list[float] = []
        bottom_means: list[float] = []
        col = f"fwd_{horizon}d"
        for _, g in sym.groupby("date", sort=True):
            gg = g[["score", col]].dropna()
            if len(gg) < 20:
                continue
            q_low = float(gg["score"].quantile(lo))
            q_high = float(gg["score"].quantile(hi))
            top = pd.to_numeric(gg.loc[gg["score"] >= q_high, col], errors="coerce").dropna()
            bottom = pd.to_numeric(gg.loc[gg["score"] <= q_low, col], errors="coerce").dropna()
            if top.empty or bottom.empty:
                continue
            top_mean = float(top.mean())
            bottom_mean = float(bottom.mean())
            top_means.append(top_mean)
            bottom_means.append(bottom_mean)
            day_spreads.append(top_mean - bottom_mean)
        result = {
            "horizon_days": float(horizon),
            "days": float(len(day_spreads)),
            "avg_top_ret": float(pd.Series(top_means).mean()) if top_means else float("nan"),
            "avg_bottom_ret": float(pd.Series(bottom_means).mean()) if bottom_means else float("nan"),
            "avg_top_bottom_spread": float(pd.Series(day_spreads).mean()) if day_spreads else float("nan"),
        }
        out[str(horizon)] = result
        _print("[DIAG][DECAY]", **result)
    return out


def run_score_concentration(df: pd.DataFrame) -> dict[str, float]:
    if "score" not in df.columns:
        return {"error": "missing_score"}
    rows: list[dict[str, float]] = []
    for _, g in df.groupby("date", sort=True):
        score_abs = pd.to_numeric(g["score"], errors="coerce").abs().dropna().sort_values(ascending=False)
        if score_abs.empty:
            continue
        total = float(score_abs.sum())
        if total <= 0.0:
            continue
        rows.append(
            {
                "top1_share": float(score_abs.iloc[:1].sum() / total),
                "top5_share": float(score_abs.iloc[:5].sum() / total),
                "top10_share": float(score_abs.iloc[:10].sum() / total),
                "max_abs_score": float(score_abs.iloc[0]),
            }
        )
    panel = pd.DataFrame(rows)
    if panel.empty:
        return {}
    out = {c: float(panel[c].mean()) for c in panel.columns}
    _print("[DIAG][CONCENTRATION]", **out)
    return out


def run_breadth_dispersion(df: pd.DataFrame, daily: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for col in ["cs_dispersion", "cs_top_bottom_spread", "cs_signal_breadth", "cs_nonzero_frac", "cs_signal_count", "score_conf"]:
        if col in daily.columns:
            s = pd.to_numeric(daily[col], errors="coerce").dropna()
            if not s.empty:
                out[f"{col}_mean"] = float(s.mean())
                out[f"{col}_p05"] = float(s.quantile(0.05))
                out[f"{col}_p95"] = float(s.quantile(0.95))
    if "cs_dispersion" in daily.columns:
        disp = pd.to_numeric(daily["cs_dispersion"], errors="coerce").dropna()
        out["dispersion_gt_1p5_days"] = float((disp > 1.5).sum())
        out["dispersion_lt_0p25_days"] = float((disp < 0.25).sum())
    _print("[DIAG][BREADTH]", **out)
    return out


def run_long_short_decomp(df: pd.DataFrame) -> dict[str, float]:
    weight = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)
    fwd = pd.to_numeric(df["target_fwd_ret_1d"], errors="coerce").fillna(0.0)
    costs = pd.to_numeric(df["cost_total"], errors="coerce").fillna(0.0)
    long_mask = weight > 0.0
    short_mask = weight < 0.0
    daily = pd.DataFrame({"date": df["date"]})
    daily["long_net"] = 0.0
    daily.loc[long_mask, "long_net"] = (weight.loc[long_mask] * fwd.loc[long_mask]) - costs.loc[long_mask]
    daily["short_net"] = 0.0
    daily.loc[short_mask, "short_net"] = (weight.loc[short_mask] * fwd.loc[short_mask]) - costs.loc[short_mask]
    long_daily = daily.groupby("date")["long_net"].sum()
    short_daily = daily.groupby("date")["short_net"].sum()
    long_sum = _summarize_daily_return_series(long_daily)
    short_sum = _summarize_daily_return_series(short_daily)
    out = {
        "long_avg_daily_ret": long_sum.get("avg_daily_ret", float("nan")),
        "long_cum_ret": long_sum.get("cum_ret", float("nan")),
        "short_avg_daily_ret": short_sum.get("avg_daily_ret", float("nan")),
        "short_cum_ret": short_sum.get("cum_ret", float("nan")),
    }
    _print("[DIAG][LS]", **out)
    return out


def run_cap_binding(daily: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for col in ["raw_turnover", "capped_turnover", "cap_hit_rate", "turnover_budget_left"]:
        if col in daily.columns:
            s = pd.to_numeric(daily[col], errors="coerce").dropna()
            if not s.empty:
                out[f"avg_{col}"] = float(s.mean())
    if "raw_turnover" in daily.columns and "capped_turnover" in daily.columns:
        raw = pd.to_numeric(daily["raw_turnover"], errors="coerce").fillna(0.0)
        cap = pd.to_numeric(daily["capped_turnover"], errors="coerce").fillna(0.0)
        out["avg_turnover_clipped"] = float((raw - cap).clip(lower=0.0).mean())
    if "cap_hit_rate" in daily.columns:
        hit = pd.to_numeric(daily["cap_hit_rate"], errors="coerce").fillna(0.0)
        out["full_bind_days"] = float((hit >= 0.999999).sum())
    _print("[DIAG][TURNOVER]", **out)
    return out


def export_json(ctx: DiagContext, payload: dict[str, object]) -> Path | None:
    if not ctx.export_json:
        return None
    ctx.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = ctx.out_dir / f"{ctx.name}__summary.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def run_all(ctx: DiagContext) -> dict[str, object]:
    paths = resolve_input_paths(ctx)
    _print("[DIAG][INPUT]", files=len(paths), glob=ctx.portfolio_glob)
    for idx, path in enumerate(paths, start=1):
        print(f"[DIAG][INPUT_FILE] i={idx} path={path}")
    df = load_portfolio_frames(paths)
    df = add_base_columns(df)
    debug = collect_debug_counters(df)
    for key, value in debug.items():
        _print("[DIAG][DEBUG]", metric=key, value=value)
    daily = build_daily_panel(df)
    overall = _summarize_daily_return_series(pd.to_numeric(daily["portfolio_ret"], errors="coerce"))
    if "raw_turnover" in daily.columns:
        overall["avg_raw_turnover"] = float(pd.to_numeric(daily["raw_turnover"], errors="coerce").mean())
    if "capped_turnover" in daily.columns:
        overall["avg_turnover"] = float(pd.to_numeric(daily["capped_turnover"], errors="coerce").mean())
    if "cap_hit_rate" in daily.columns:
        overall["cap_hit_rate"] = float(pd.to_numeric(daily["cap_hit_rate"], errors="coerce").mean())
    if "cs_dispersion" in daily.columns:
        overall["avg_cs_dispersion"] = float(pd.to_numeric(daily["cs_dispersion"], errors="coerce").mean())
    _print("[DIAG][OVERALL]", **overall)
    df = compute_delay_return_columns(df, ctx.delays)
    delay = run_delay_diagnostics(df, ctx.delays)
    persistence = run_rank_persistence(df, ctx.top_pct)
    costs = run_cost_sensitivity(df, ctx.cost_mults)
    decay = run_alpha_decay(df, ctx.top_pct, ctx.horizons)
    concentration = run_score_concentration(df)
    breadth = run_breadth_dispersion(df, daily)
    long_short = run_long_short_decomp(df)
    turnover = run_cap_binding(daily)
    payload: dict[str, object] = {
        "config": {**asdict(ctx), "out_dir": str(ctx.out_dir)},
        "input_files": [str(p) for p in paths],
        "debug": debug,
        "overall": overall,
        "delay": delay,
        "persistence": persistence,
        "cost_sensitivity": costs,
        "alpha_decay": decay,
        "score_concentration": concentration,
        "breadth_dispersion": breadth,
        "long_short": long_short,
        "turnover": turnover,
    }
    out_json = export_json(ctx, payload)
    if out_json is not None:
        print(f"[DIAG][EXPORT] json={out_json}")
    return payload


def _wait_if_needed(wait_on_exit: bool) -> None:
    if not wait_on_exit:
        return
    try:
        input("\nPress Enter to exit...")
    except EOFError:
        pass


def main() -> int:
    ctx = build_context()
    run_all(ctx)
    print("[DIAG] completed successfully")
    return 0


if __name__ == "__main__":
    rc = 1
    wait_on_exit = True
    try:
        wait_on_exit = _env_bool("DIAG_WAIT_ON_EXIT", True)
        rc = main()
    except Exception as exc:
        print(f"[ERROR] {exc}")
        print()
        traceback.print_exc()
        rc = 1
    finally:
        _wait_if_needed(wait_on_exit)
    raise SystemExit(rc)
