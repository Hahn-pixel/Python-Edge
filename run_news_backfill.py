"""
run_news_backfill.py
─────────────────────────────────────────────────────────────────────
Завантажує архів новинного sentiment за історичний діапазон дат.
Зберігає по одному файлу на торговий день у форматі
artifacts/news/archive/news_risk_flags_YYYY-MM-DD.json

Після завершення запустіть run_news_ic.py для IC аналізу.

Env vars:
    MASSIVE_API_KEY           — обов'язково
    BACKFILL_START_DATE       — початок діапазону YYYY-MM-DD (default: 2024-01-02)
    BACKFILL_END_DATE         — кінець діапазону YYYY-MM-DD   (default: вчора)
    BACKFILL_SYMBOLS          — pipe-розділений список символів (default: з universe)
    BACKFILL_ARCHIVE_DIR      — куди зберігати (default: artifacts/news/archive)
    BACKFILL_LIMIT_PER_SYMBOL — max статей на символ на день (default: 10)
    BACKFILL_REQUEST_DELAY_MS — затримка між запитами ms (default: 300)
    BACKFILL_SKIP_EXISTING    — "1" = пропускати вже існуючі файли (default: 1)
    BACKFILL_SKIP_WEEKENDS    — "1" = пропускати сб/нд (default: 1)
─────────────────────────────────────────────────────────────────────
"""

import os
import sys
import json
import time
import traceback
import requests
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent

MASSIVE_API_KEY           = os.environ.get("MASSIVE_API_KEY", "")
BACKFILL_START_DATE       = os.environ.get("BACKFILL_START_DATE", "2024-01-02")
BACKFILL_END_DATE         = os.environ.get("BACKFILL_END_DATE", "")
BACKFILL_ARCHIVE_DIR      = Path(os.environ.get("BACKFILL_ARCHIVE_DIR", "artifacts/news/archive"))
BACKFILL_LIMIT_PER_SYMBOL = int(os.environ.get("BACKFILL_LIMIT_PER_SYMBOL", "10"))
BACKFILL_REQUEST_DELAY_MS = int(os.environ.get("BACKFILL_REQUEST_DELAY_MS", "300"))
BACKFILL_SKIP_EXISTING    = os.environ.get("BACKFILL_SKIP_EXISTING", "1") == "1"
BACKFILL_SKIP_WEEKENDS    = os.environ.get("BACKFILL_SKIP_WEEKENDS", "1") == "1"

DEFAULT_UNIVERSE_PATH = ROOT / "artifacts/daily_cycle/universe/universe_snapshot.parquet"
BASE_URL = "https://api.massive.com/v2/reference/news"

# US market holidays 2024-2026 (NYSE)
_HOLIDAYS = {
    "2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29",
    "2024-05-27", "2024-06-19", "2024-07-04", "2024-09-02",
    "2024-11-28", "2024-12-25",
    "2025-01-01", "2025-01-20", "2025-02-17", "2025-04-18",
    "2025-05-26", "2025-06-19", "2025-07-04", "2025-09-01",
    "2025-11-27", "2025-12-25",
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03",
    "2026-05-25", "2026-06-19", "2026-07-03", "2026-09-07",
    "2026-11-26", "2026-12-25",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _get_headers() -> dict:
    return {"Authorization": f"Bearer {MASSIVE_API_KEY}"}


def _is_trading_day(dt: datetime) -> bool:
    if BACKFILL_SKIP_WEEKENDS and dt.weekday() >= 5:
        return False
    if dt.strftime("%Y-%m-%d") in _HOLIDAYS:
        return False
    return True


def _trading_days(start: datetime, end: datetime) -> list[datetime]:
    days = []
    cur  = start
    while cur <= end:
        if _is_trading_day(cur):
            days.append(cur)
        cur += timedelta(days=1)
    return days


def _load_symbols() -> list[str]:
    env_syms = os.environ.get("BACKFILL_SYMBOLS", "").strip()
    if env_syms:
        syms = [s.strip().upper() for s in env_syms.split("|") if s.strip()]
        print(f"  [INFO] символи з env: {len(syms)}")
        return syms

    if DEFAULT_UNIVERSE_PATH.exists():
        try:
            import pandas as pd
            df  = pd.read_parquet(DEFAULT_UNIVERSE_PATH)
            col = next((c for c in ["symbol", "ticker"] if c in df.columns), None)
            if col:
                syms = sorted(df[col].dropna().str.upper().unique().tolist())
                print(f"  [INFO] символи з universe: {len(syms)}")
                return syms
        except Exception as e:
            print(f"  [WARN] помилка читання universe: {e}")

    raise RuntimeError(
        "Символи не знайдено. Передайте $env:BACKFILL_SYMBOLS = 'AAPL|MSFT|...' "
        "або переконайтесь що artifacts/daily_cycle/universe/universe_snapshot.parquet існує."
    )


def _fetch_day(ticker: str, date_str: str) -> list[dict]:
    """
    Fetch новин для тікера за один торговий день.
    published_utc від початку дня до кінця наступного (щоб захопити after-hours).
    """
    dt      = datetime.strptime(date_str, "%Y-%m-%d")
    gte_str = dt.strftime("%Y-%m-%dT00:00:00Z")
    lte_str = (dt + timedelta(days=1)).strftime("%Y-%m-%dT23:59:59Z")

    params = {
        "ticker":                ticker,
        "limit":                 BACKFILL_LIMIT_PER_SYMBOL,
        "order":                 "desc",
        "published_utc.gte":     gte_str,
        "published_utc.lte":     lte_str,
    }
    try:
        resp = requests.get(BASE_URL, params=params, headers=_get_headers(), timeout=20)
        if resp.status_code != 200:
            return []
        data    = resp.json()
        results = data.get("results", [])
        articles = []
        for art in results:
            insights = art.get("insights") or []
            matched  = [i for i in insights if i.get("ticker", "").upper() == ticker.upper()]
            if not matched:
                matched = insights[:1]
            sentiment        = matched[0].get("sentiment", "unknown") if matched else "unknown"
            sentiment_reason = matched[0].get("sentiment_reasoning", "") if matched else ""
            articles.append({
                "title":               art.get("title", ""),
                "published_utc":       art.get("published_utc", ""),
                "sentiment":           sentiment,
                "sentiment_reasoning": sentiment_reason,
            })
        return articles
    except Exception:
        return []


def _build_day_result(date_str: str, symbols: list[str]) -> dict:
    symbols_out: dict[str, dict] = {}
    flagged_count = 0

    for i, sym in enumerate(symbols):
        arts  = _fetch_day(sym, date_str)
        n_neg = sum(1 for a in arts if a["sentiment"] == "negative")
        n_pos = sum(1 for a in arts if a["sentiment"] == "positive")
        n_neu = sum(1 for a in arts if a["sentiment"] == "neutral")
        flagged = n_neg >= 1

        if flagged:
            flagged_count += 1

        symbols_out[sym] = {
            "n_articles": len(arts),
            "n_negative": n_neg,
            "n_positive": n_pos,
            "n_neutral":  n_neu,
            "flagged":    flagged,
            "articles":   arts,
        }
        if i < len(symbols) - 1:
            time.sleep(BACKFILL_REQUEST_DELAY_MS / 1000.0)

    return {
        "generated_utc":      datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "trade_date":         date_str,
        "lookback_hours":     24,
        "since_utc":          f"{date_str}T00:00:00Z",
        "symbol_count":       len(symbols),
        "flagged_count":      flagged_count,
        "negative_threshold": 1,
        "symbols":            symbols_out,
        "source":             "backfill",
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    if not MASSIVE_API_KEY:
        raise RuntimeError(
            "MASSIVE_API_KEY не знайдено. "
            "Встановіть: $env:MASSIVE_API_KEY = 'your_key'"
        )

    # resolve dates
    start_dt = datetime.strptime(BACKFILL_START_DATE, "%Y-%m-%d")
    if BACKFILL_END_DATE:
        end_dt = datetime.strptime(BACKFILL_END_DATE, "%Y-%m-%d")
    else:
        end_dt = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=1)
        end_dt = end_dt.replace(hour=0, minute=0, second=0, microsecond=0)

    arch_dir = ROOT / BACKFILL_ARCHIVE_DIR
    arch_dir.mkdir(parents=True, exist_ok=True)

    symbols = _load_symbols()

    trading_days = _trading_days(start_dt, end_dt)

    print(f"\n{'='*60}")
    print(f"  run_news_backfill")
    print(f"{'='*60}")
    print(f"  start:          {start_dt.date()}")
    print(f"  end:            {end_dt.date()}")
    print(f"  trading days:   {len(trading_days)}")
    print(f"  symbols:        {len(symbols)}")
    print(f"  skip_existing:  {BACKFILL_SKIP_EXISTING}")
    print(f"  delay_ms:       {BACKFILL_REQUEST_DELAY_MS}")
    print(f"  archive_dir:    {arch_dir}")
    print(f"\n  Оцінка часу: {len(trading_days)} днів × {len(symbols)} символів × "
          f"{BACKFILL_REQUEST_DELAY_MS}ms = "
          f"~{len(trading_days) * len(symbols) * BACKFILL_REQUEST_DELAY_MS / 1000 / 60:.0f} хвилин")
    print(f"{'='*60}\n")

    skipped  = 0
    fetched  = 0
    errors   = 0

    for day_i, dt in enumerate(trading_days):
        date_str  = dt.strftime("%Y-%m-%d")
        out_path  = arch_dir / f"news_risk_flags_{date_str}.json"

        if BACKFILL_SKIP_EXISTING and out_path.exists():
            skipped += 1
            print(f"  [{day_i+1:4d}/{len(trading_days)}] {date_str}  SKIP (вже існує)")
            continue

        print(f"  [{day_i+1:4d}/{len(trading_days)}] {date_str}  fetching {len(symbols)} symbols...")

        try:
            result = _build_day_result(date_str, symbols)

            n_with_news = sum(1 for d in result["symbols"].values() if d["n_articles"] > 0)
            n_flagged   = result["flagged_count"]
            total_arts  = sum(d["n_articles"] for d in result["symbols"].values())

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            fetched += 1
            print(f"             → articles={total_arts}  with_news={n_with_news}  flagged={n_flagged}  ✓")

        except Exception as e:
            errors += 1
            print(f"             → [ERROR] {type(e).__name__}: {e}")

    print(f"\n{'='*60}")
    print(f"  Backfill завершено")
    print(f"  fetched:  {fetched}")
    print(f"  skipped:  {skipped}")
    print(f"  errors:   {errors}")
    print(f"  archive:  {arch_dir}")
    print(f"{'='*60}")
    print(f"\n  Тепер запустіть: python run_news_ic.py")

    return 0 if errors == 0 else 1


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        print("\n[CRASHED]")
    finally:
        print()
        input("Press Enter to exit...")
    sys.exit(rc)
