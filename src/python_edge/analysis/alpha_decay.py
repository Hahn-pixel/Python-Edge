from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class AlphaDecayConfig:
    score_col: str = "score"
    symbol_col: str = "symbol"
    date_col: str = "date"
    close_col: str = "close"
    horizons: tuple[int, ...] = (1, 2, 3, 5, 10, 15, 20)
    quantiles: tuple[float, ...] = (0.02, 0.05, 0.10, 0.20)



def _validate_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"alpha_decay: missing columns: {missing}")



def add_forward_returns(
    df: pd.DataFrame,
    symbol_col: str,
    close_col: str,
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values([symbol_col, "date"], ascending=[True, True]).reset_index(drop=True)

    for h in horizons:
        out[f"fwd_ret_{h}d"] = out.groupby(symbol_col)[close_col].shift(-h) / out[close_col] - 1.0

    return out



def add_cross_sectional_score_ranks(
    df: pd.DataFrame,
    date_col: str,
    score_col: str,
) -> pd.DataFrame:
    out = df.copy()
    out["score_rank_pct"] = out.groupby(date_col)[score_col].rank(method="average", pct=True)
    return out



def build_alpha_decay_surface(
    df: pd.DataFrame,
    cfg: AlphaDecayConfig,
) -> pd.DataFrame:
    _validate_columns(df, [cfg.score_col, cfg.symbol_col, cfg.date_col, cfg.close_col])

    out = df.copy()
    out = out[[cfg.date_col, cfg.symbol_col, cfg.close_col, cfg.score_col]].copy()
    out[cfg.date_col] = pd.to_datetime(out[cfg.date_col], errors="coerce")
    out = out.dropna(subset=[cfg.date_col, cfg.symbol_col, cfg.close_col, cfg.score_col]).copy()

    out = add_forward_returns(out, symbol_col=cfg.symbol_col, close_col=cfg.close_col, horizons=cfg.horizons)
    out = add_cross_sectional_score_ranks(out, date_col=cfg.date_col, score_col=cfg.score_col)

    rows: list[dict[str, float | int | str]] = []

    for q in cfg.quantiles:
        long_mask = out["score_rank_pct"] >= (1.0 - q)
        short_mask = out["score_rank_pct"] <= q

        for h in cfg.horizons:
            col = f"fwd_ret_{h}d"

            long_vals = pd.to_numeric(out.loc[long_mask, col], errors="coerce").dropna()
            short_vals = pd.to_numeric(out.loc[short_mask, col], errors="coerce").dropna()

            long_mean = float(long_vals.mean()) if len(long_vals) > 0 else float("nan")
            short_mean = float(short_vals.mean()) if len(short_vals) > 0 else float("nan")
            spread_mean = long_mean - short_mean if pd.notna(long_mean) and pd.notna(short_mean) else float("nan")

            rows.append(
                {
                    "bucket": f"top_{int(q * 100)}pct",
                    "side": "long",
                    "horizon_days": h,
                    "mean_fwd_ret": long_mean,
                    "n_obs": int(len(long_vals)),
                }
            )
            rows.append(
                {
                    "bucket": f"bottom_{int(q * 100)}pct",
                    "side": "short",
                    "horizon_days": h,
                    "mean_fwd_ret": short_mean,
                    "n_obs": int(len(short_vals)),
                }
            )
            rows.append(
                {
                    "bucket": f"spread_{int(q * 100)}pct",
                    "side": "long_short",
                    "horizon_days": h,
                    "mean_fwd_ret": spread_mean,
                    "n_obs": int(min(len(long_vals), len(short_vals))),
                }
            )

    res = pd.DataFrame(rows)
    if res.empty:
        raise RuntimeError("alpha_decay: empty surface")
    return res



def summarize_decay_turning_points(surface_df: pd.DataFrame) -> pd.DataFrame:
    required = ["bucket", "side", "horizon_days", "mean_fwd_ret"]
    _validate_columns(surface_df, required)

    rows: list[dict[str, float | int | str]] = []
    for (bucket, side), g in surface_df.groupby(["bucket", "side"], sort=True):
        gg = g.sort_values("horizon_days").reset_index(drop=True)
        best_idx = gg["mean_fwd_ret"].idxmax() if side != "short" else gg["mean_fwd_ret"].idxmin()
        best_row = gg.loc[best_idx]

        rows.append(
            {
                "bucket": str(bucket),
                "side": str(side),
                "best_horizon_days": int(best_row["horizon_days"]),
                "best_mean_fwd_ret": float(best_row["mean_fwd_ret"]),
            }
        )

    return pd.DataFrame(rows)



def print_alpha_decay_summary(surface_df: pd.DataFrame) -> None:
    turning = summarize_decay_turning_points(surface_df)
    print("[ALPHA_DECAY][TURNING_POINTS]")
    print(turning.to_string(index=False))