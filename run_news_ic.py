"""
run_news_ic.py
─────────────────────────────────────────────────────────────────────
Аналіз IC (Information Coefficient) новинного сигналу.

Ціни завантажуються з massive.com /v2/aggs/ticker/{t}/range/1/day/{from}/{to}
і кешуються в artifacts/news/ic_price_cache.parquet щоб не повторювати запити.

Методологія:
    news_score(s, t) = (n_positive - n_negative) / max(n_articles, 1)
    return_fwd(s, t) = close(t+H) / close(t) - 1
    rank_IC(t)       = spearmanr(news_score, return_fwd) по cross-section

    summary:
        mean_IC, std_IC, t_stat = mean_IC / (std_IC / sqrt(N)), IR = mean_IC / std_IC

Env vars:
    MASSIVE_API_KEY            — обов'язково
    NEWS_IC_ARCHIVE_DIR        — архів новин (default: artifacts/news/archive)
    NEWS_IC_HORIZON            — forward return horizon в днях (default: 1)
    NEWS_IC_MIN_SYMBOLS        — мін символів з новинами для дня (default: 3)
    NEWS_IC_MIN_DAYS           — мін днів для розрахунку (default: 5)
    NEWS_IC_OUTPUT             — вихідний CSV (default: artifacts/news/news_ic_results.csv)
    NEWS_IC_PRICE_CACHE        — кеш цін (default: artifacts/news/ic_price_cache.parquet)
    NEWS_IC_REQUEST_DELAY_MS   — затримка між запитами до massive ms (default: 200)
    NEWS_IC_REFRESH_CACHE      — "1" = перезавантажити кеш (default: 0)
─────────────────────────────────────────────────────────────────────
"""

import os
import sys
import json
import math
import time
import traceback
import requests
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent

MASSIVE_API_KEY          = os.environ.get("MASSIVE_API_KEY", "")
NEWS_IC_ARCHIVE_DIR      = Path(os.getenv("NEWS_IC_ARCHIVE_DIR",    "artifacts/news/archive"))
NEWS_IC_HORIZON          = int(os.getenv("NEWS_IC_HORIZON",         "1"))
NEWS_IC_MIN_SYMBOLS      = int(os.getenv("NEWS_IC_MIN_SYMBOLS",     "3"))
NEWS_IC_MIN_DAYS         = int(os.getenv("NEWS_IC_MIN_DAYS",        "5"))
NEWS_IC_OUTPUT           = Path(os.getenv("NEWS_IC_OUTPUT",         "artifacts/news/news_ic_results.csv"))
NEWS_IC_PRICE_CACHE      = Path(os.getenv("NEWS_IC_PRICE_CACHE",    "artifacts/news/ic_price_cache.parquet"))
NEWS_IC_REQUEST_DELAY_MS = int(os.getenv("NEWS_IC_REQUEST_DELAY_MS","200"))
NEWS_IC_REFRESH_CACHE    = os.getenv("NEWS_IC_REFRESH_CACHE", "0") == "1"


# ── spearman (без scipy) ──────────────────────────────────────────────────────

def _spearman_r(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 3:
        return float("nan")

    def _rank(v: list[float]) -> list[float]:
        sv   = sorted(enumerate(v), key=lambda t: t[1])
        rnks = [0.0] * n
        i    = 0
        while i < n:
            j = i
            while j < n - 1 and sv[j + 1][1] == sv[j][1]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                rnks[sv[k][0]] = avg_rank
            i = j + 1
        return rnks

    rx = _rank(x)
    ry = _rank(y)
    mx = sum(rx) / n
    my = sum(ry) / n
    num   = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    den_x = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(n)))
    den_y = math.sqrt(sum((ry[i] - my) ** 2 for i in range(n)))
    if den_x < 1e-12 or den_y < 1e-12:
        return float("nan")
    return num / (den_x * den_y)


# ── load archive ──────────────────────────────────────────────────────────────

def _load_archive(archive_dir: Path) -> pd.DataFrame:
    archive_dir = ROOT / archive_dir if not archive_dir.is_absolute() else archive_dir
    files = sorted(archive_dir.glob("news_risk_flags_*.json"))
    if not files:
        raise FileNotFoundError(
            f"Архівні файли не знайдено в {archive_dir}\n"
            f"Запустіть run_news_backfill.py або fetch_massive_news.py."
        )

    rows = []
    for f in files:
        stem     = f.stem
        date_str = stem.split("news_risk_flags_")[-1]
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            continue
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        trade_date = data.get("trade_date", date_str)
        for sym, d in data.get("symbols", {}).items():
            n_art = int(d.get("n_articles", 0))
            n_pos = int(d.get("n_positive", 0))
            n_neg = int(d.get("n_negative", 0))
            score = (n_pos - n_neg) / max(n_art, 1) if n_art > 0 else float("nan")
            rows.append({
                "trade_date": trade_date,
                "symbol":     sym.upper(),
                "n_articles": n_art,
                "n_positive": n_pos,
                "n_negative": n_neg,
                "news_score": score,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Архів порожній.")
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.normalize()
    return df.sort_values(["trade_date", "symbol"]).reset_index(drop=True)


# ── massive price fetch ───────────────────────────────────────────────────────

def _fetch_prices_massive(symbols: list[str], from_date: str, to_date: str) -> pd.DataFrame:
    """
    Завантажує денні ціни закриття з massive.com для списку символів.
    from_date, to_date — YYYY-MM-DD рядки.
    Повертає DataFrame: symbol | date | close
    """
    if not MASSIVE_API_KEY:
        raise RuntimeError("MASSIVE_API_KEY не знайдено в середовищі.")

    headers = {"Authorization": f"Bearer {MASSIVE_API_KEY}"}
    rows    = []
    total   = len(symbols)

    for i, sym in enumerate(symbols):
        url = f"https://api.massive.com/v2/aggs/ticker/{sym}/range/1/day/{from_date}/{to_date}"
        try:
            resp = requests.get(url, headers=headers, params={"adjusted": "true", "limit": 1000}, timeout=20)
            if resp.status_code == 200:
                data    = resp.json()
                results = data.get("results", [])
                for bar in results:
                    # massive aggs: t=timestamp ms, c=close
                    ts = bar.get("t")
                    c  = bar.get("c")
                    if ts is not None and c is not None:
                        dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
                        rows.append({"symbol": sym, "date": dt, "close": float(c)})
            elif resp.status_code == 404:
                pass  # символ не знайдено — пропускаємо
            else:
                print(f"  [WARN] {sym}: HTTP {resp.status_code}")
        except Exception as e:
            print(f"  [WARN] {sym}: {type(e).__name__}: {e}")

        print(f"  [{i+1:4d}/{total}] {sym:<8s}  bars={sum(1 for r in rows if r['symbol']==sym)}")
        if i < total - 1:
            time.sleep(NEWS_IC_REQUEST_DELAY_MS / 1000.0)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"]  = pd.to_datetime(df["date"]).dt.normalize()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"]).drop_duplicates(subset=["symbol", "date"], keep="last")
    return df.sort_values(["symbol", "date"]).reset_index(drop=True)


def _load_prices(symbols: list[str], from_date: str, to_date: str) -> pd.DataFrame:
    """
    Завантажує ціни з кешу або з massive якщо кеш відсутній/застарілий.
    Кеш зберігається в NEWS_IC_PRICE_CACHE.
    """
    cache_path = ROOT / NEWS_IC_PRICE_CACHE

    # спробувати кеш
    if not NEWS_IC_REFRESH_CACHE and cache_path.exists():
        try:
            cached = pd.read_parquet(cache_path)
            cached["date"] = pd.to_datetime(cached["date"]).dt.normalize()
            cached_from = cached["date"].min().strftime("%Y-%m-%d")
            cached_to   = cached["date"].max().strftime("%Y-%m-%d")
            cached_syms = set(cached["symbol"].unique())
            need_syms   = set(s.upper() for s in symbols)

            # кеш покриває потрібний діапазон і символи
            if cached_from <= from_date and cached_to >= to_date and need_syms <= cached_syms:
                print(f"  [CACHE] завантажено з кешу: {cache_path}")
                print(f"  [CACHE] діапазон: {cached_from} → {cached_to}  символів: {len(cached_syms)}")
                mask = (
                    (cached["date"] >= pd.Timestamp(from_date)) &
                    (cached["date"] <= pd.Timestamp(to_date)) &
                    (cached["symbol"].isin(need_syms))
                )
                return cached.loc[mask].copy().reset_index(drop=True)
        except Exception as e:
            print(f"  [WARN] помилка читання кешу: {e} — завантажуємо з massive")

    # завантажити з massive
    print(f"  → завантаження цін з massive: {len(symbols)} символів  {from_date} → {to_date}")
    print(f"  → оцінка: ~{len(symbols) * NEWS_IC_REQUEST_DELAY_MS / 1000 / 60:.1f} хвилин")
    prices = _fetch_prices_massive(symbols, from_date, to_date)

    if prices.empty:
        raise RuntimeError("massive не повернув жодного бару цін.")

    # зберегти кеш
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # якщо кеш існує — merge з ним
    if cache_path.exists():
        try:
            old = pd.read_parquet(cache_path)
            old["date"] = pd.to_datetime(old["date"]).dt.normalize()
            prices = pd.concat([old, prices], ignore_index=True)
            prices = prices.drop_duplicates(subset=["symbol", "date"], keep="last")
            prices = prices.sort_values(["symbol", "date"]).reset_index(drop=True)
        except Exception:
            pass

    prices.to_parquet(cache_path, index=False)
    print(f"  [CACHE] збережено: {cache_path}  ({len(prices)} барів)")

    return prices


# ── forward returns ───────────────────────────────────────────────────────────

def _build_forward_returns(prices: pd.DataFrame, horizon: int) -> pd.DataFrame:
    prices = prices.sort_values(["symbol", "date"]).copy()
    prices["fwd_close"]  = prices.groupby("symbol")["close"].shift(-horizon)
    prices["fwd_return"] = prices["fwd_close"] / prices["close"] - 1.0
    result = prices[["symbol", "date", "fwd_return"]].dropna(subset=["fwd_return"])
    return result.rename(columns={"date": "trade_date"})


# ── IC calculation ────────────────────────────────────────────────────────────

def _calc_ic(news_df: pd.DataFrame, fwd_df: pd.DataFrame) -> pd.DataFrame:
    merged = news_df.merge(fwd_df, on=["trade_date", "symbol"], how="inner")
    merged = merged[merged["n_articles"] > 0].copy()
    merged = merged.dropna(subset=["news_score", "fwd_return"])

    rows = []
    for trade_date, grp in merged.groupby("trade_date"):
        n = len(grp)
        if n < NEWS_IC_MIN_SYMBOLS:
            continue
        ic = _spearman_r(grp["news_score"].tolist(), grp["fwd_return"].tolist())
        rows.append({"trade_date": trade_date, "ic": ic, "n_symbols": n})

    ic_df = pd.DataFrame(rows)
    if not ic_df.empty:
        ic_df = ic_df.dropna(subset=["ic"])
        ic_df = ic_df.sort_values("trade_date").reset_index(drop=True)
    return ic_df


# ── summary ───────────────────────────────────────────────────────────────────

def _print_summary(ic_df: pd.DataFrame, horizon: int) -> dict:
    n = len(ic_df)
    if n < NEWS_IC_MIN_DAYS:
        print(f"\n[IC] Недостатньо днів: {n} < {NEWS_IC_MIN_DAYS}")
        print(f"     Потрібно ще {NEWS_IC_MIN_DAYS - n} днів.")
        return {}

    ic_vals = ic_df["ic"].tolist()
    mean_ic = float(np.mean(ic_vals))
    std_ic  = float(np.std(ic_vals, ddof=1)) if n > 1 else float("nan")
    t_stat  = mean_ic / (std_ic / math.sqrt(n)) if std_ic > 1e-12 else float("nan")
    ir      = mean_ic / std_ic if std_ic > 1e-12 else float("nan")
    pct_pos = float(sum(1 for v in ic_vals if v > 0) / n * 100)
    ic_min  = float(min(ic_vals))
    ic_max  = float(max(ic_vals))

    print(f"\n{'='*55}")
    print(f"  News Sentiment IC  |  horizon={horizon}d  |  N={n} днів")
    print(f"{'='*55}")
    print(f"  mean_IC :  {mean_ic:+.4f}")
    print(f"  std_IC  :  {std_ic:.4f}")
    print(f"  t-stat  :  {t_stat:+.2f}  {'✓ значимий (>2)' if abs(t_stat) > 2 else '✗ незначимий'}")
    print(f"  IR      :  {ir:+.3f}")
    print(f"  % IC>0  :  {pct_pos:.1f}%")
    print(f"  IC range:  [{ic_min:+.4f}, {ic_max:+.4f}]")
    print(f"{'='*55}")

    if abs(mean_ic) < 0.02:
        verdict = "СЛАБКИЙ — сигнал майже не корелює з returns"
    elif abs(t_stat) < 2:
        verdict = "НЕСТАБІЛЬНИЙ — IC є але статистично незначимий (мало днів)"
    elif mean_ic > 0.03:
        verdict = "СИЛЬНИЙ — сигнал варто включати з вагою"
    else:
        verdict = "ПОМІРНИЙ — використовувати обережно"
    print(f"  Висновок: {verdict}")
    print(f"{'='*55}\n")

    print("  По днях:")
    print(f"  {'date':<12} {'IC':>7} {'n_sym':>6}")
    print(f"  {'-'*28}")
    for _, row in ic_df.iterrows():
        print(f"  {str(row['trade_date'])[:10]:<12} {row['ic']:>+7.4f} {int(row['n_symbols']):>6}")

    return {
        "horizon_days":     horizon,
        "n_days":           n,
        "mean_ic":          mean_ic,
        "std_ic":           std_ic,
        "t_stat":           t_stat,
        "ir":               ir,
        "pct_ic_positive":  pct_pos,
        "ic_min":           ic_min,
        "ic_max":           ic_max,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    print(f"\n[IC] run_news_ic — старт")
    print(f"  archive_dir:   {NEWS_IC_ARCHIVE_DIR}")
    print(f"  price_cache:   {NEWS_IC_PRICE_CACHE}")
    print(f"  horizon:       {NEWS_IC_HORIZON}d forward return")
    print(f"  min_symbols:   {NEWS_IC_MIN_SYMBOLS} per day")
    print(f"  min_days:      {NEWS_IC_MIN_DAYS}")
    print(f"  refresh_cache: {NEWS_IC_REFRESH_CACHE}")

    # 1. завантажити архів новин
    print(f"\n[IC] завантаження архіву новин...")
    news_df = _load_archive(NEWS_IC_ARCHIVE_DIR)
    dates   = sorted(news_df["trade_date"].unique())
    print(f"  архівних днів: {len(dates)}")
    print(f"  діапазон:      {str(dates[0])[:10]} → {str(dates[-1])[:10]}")
    print(f"  рядків:        {len(news_df)}")

    if len(dates) < NEWS_IC_MIN_DAYS:
        print(f"\n[IC] Недостатньо днів: {len(dates)} < {NEWS_IC_MIN_DAYS}")
        return 0

    # визначити символи і діапазон дат для цін
    symbols   = sorted(news_df["symbol"].unique().tolist())
    from_date = str(dates[0])[:10]
    # +H+1 днів щоб мати forward return для останнього дня архіву
    to_dt     = (dates[-1] + timedelta(days=NEWS_IC_HORIZON + 5))
    to_date   = min(to_dt, datetime.now(timezone.utc).replace(tzinfo=None)).strftime("%Y-%m-%d")
    to_date   = str(to_dt)[:10]

    # 2. завантажити ціни
    print(f"\n[IC] завантаження цін з massive...")
    prices = _load_prices(symbols, from_date, to_date)
    price_dates = sorted(prices["date"].unique())
    print(f"  цінових днів:  {len(price_dates)}")
    print(f"  діапазон:      {str(price_dates[0])[:10]} → {str(price_dates[-1])[:10]}")
    print(f"  символів:      {prices['symbol'].nunique()}")

    # 3. forward returns
    print(f"\n[IC] розрахунок {NEWS_IC_HORIZON}d forward returns...")
    fwd_df = _build_forward_returns(prices, NEWS_IC_HORIZON)
    print(f"  пар (symbol, date) з fwd_return: {len(fwd_df)}")

    if fwd_df.empty:
        print("[IC] [ERROR] forward returns порожні — перевірте ціни з massive")
        return 1

    # 4. IC per day
    print(f"\n[IC] розрахунок cross-sectional IC per day...")
    ic_df = _calc_ic(news_df, fwd_df)
    print(f"  днів з достатньою кількістю символів: {len(ic_df)}")

    # 5. summary
    summary = _print_summary(ic_df, NEWS_IC_HORIZON)

    # 6. зберегти результати
    if not ic_df.empty:
        out_path = ROOT / NEWS_IC_OUTPUT
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ic_df["trade_date"] = ic_df["trade_date"].astype(str).str[:10]
        ic_df.to_csv(out_path, index=False)
        print(f"[IC] результати збережено: {out_path}")

        if summary:
            summary_path = out_path.with_name("news_ic_summary.json")
            summary["generated_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"[IC] summary збережено:    {summary_path}")

    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        rc = 0
    except Exception:
        traceback.print_exc()
        print("\n[CRASHED]")
    finally:
        print()
        input("Press Enter to exit...")
    sys.exit(rc)
