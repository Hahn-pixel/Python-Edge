"""
fetch_massive_news.py
─────────────────────────────────────────────────────────────────────
Отримує новинний sentiment по всьому universe з massive.com
і зберігає результат у news_risk_flags.json.

Використання (standalone):
    python fetch_massive_news.py

Використання з launch_daily_update.py:
    from fetch_massive_news import run_news_fetch
    run_news_fetch(symbols=list_of_symbols, output_path=Path("..."))

Env vars:
    MASSIVE_API_KEY          — обов'язково
    NEWS_LOOKBACK_HOURS      — скільки годин назад шукати (default: 24)
    NEWS_LIMIT_PER_SYMBOL    — max статей на символ (default: 10, max 1000)
    NEWS_NEGATIVE_THRESHOLD  — мін кількість negative insights щоб флагнути (default: 1)
    NEWS_BATCH_MODE          — "1" = один великий запит без ticker (default: 0)
    NEWS_REQUEST_DELAY_MS    — затримка між запитами в ms (default: 200)

Output: artifacts/news/news_risk_flags.json
    {
      "generated_utc": "2026-04-25T14:00:00Z",
      "lookback_hours": 24,
      "symbol_count": 229,
      "flagged_count": 3,
      "symbols": {
        "AAPL": {
          "n_articles": 3,
          "n_negative": 0,
          "n_positive": 2,
          "n_neutral": 1,
          "flagged": false,
          "articles": [
            {
              "title": "...",
              "published_utc": "...",
              "sentiment": "positive",
              "sentiment_reasoning": "..."
            }
          ]
        },
        ...
      }
    }
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


# ── конфігурація ──────────────────────────────────────────────────────────────

MASSIVE_API_KEY       = os.environ.get("MASSIVE_API_KEY", "")
NEWS_LOOKBACK_HOURS   = int(os.environ.get("NEWS_LOOKBACK_HOURS", "24"))
NEWS_LIMIT_PER_SYMBOL = int(os.environ.get("NEWS_LIMIT_PER_SYMBOL", "10"))
NEWS_NEGATIVE_THRESHOLD = int(os.environ.get("NEWS_NEGATIVE_THRESHOLD", "1"))
NEWS_BATCH_MODE       = os.environ.get("NEWS_BATCH_MODE", "0") == "1"
NEWS_REQUEST_DELAY_MS = int(os.environ.get("NEWS_REQUEST_DELAY_MS", "200"))

BASE_URL = "https://api.massive.com/v2/reference/news"

# Дефолтний universe якщо запускається standalone
DEFAULT_UNIVERSE_PATH = Path("artifacts/daily_cycle/universe/universe_snapshot.parquet")
DEFAULT_OUTPUT_PATH   = Path("artifacts/news/news_risk_flags.json")

# debug counters
_counters = {
    "requests_sent": 0,
    "requests_failed": 0,
    "articles_total": 0,
    "symbols_flagged": 0,
    "symbols_no_news": 0,
}


# ── HTTP helper ───────────────────────────────────────────────────────────────

def _get_headers() -> dict:
    return {"Authorization": f"Bearer {MASSIVE_API_KEY}"}


def _fetch_news_for_ticker(ticker: str, since_utc: str) -> list[dict]:
    """
    Повертає список статей з insights для заданого тікера.
    Кожен елемент: {"title", "published_utc", "sentiment", "sentiment_reasoning"}
    """
    params = {
        "ticker": ticker,
        "limit":  NEWS_LIMIT_PER_SYMBOL,
        "order":  "desc",
        "published_utc.gte": since_utc,
    }
    try:
        _counters["requests_sent"] += 1
        resp = requests.get(
            BASE_URL,
            params=params,
            headers=_get_headers(),
            timeout=15,
        )
        if resp.status_code != 200:
            print(f"  [WARN] {ticker}: HTTP {resp.status_code} — {resp.text[:200]}")
            _counters["requests_failed"] += 1
            return []

        data = resp.json()
        results = data.get("results", [])
        articles = []
        for art in results:
            insights = art.get("insights") or []
            # знайти insight для нашого тікера
            matched = [
                ins for ins in insights
                if ins.get("ticker", "").upper() == ticker.upper()
            ]
            if not matched:
                # якщо немає точного матчу — беремо перший available
                matched = insights[:1]

            sentiment         = matched[0].get("sentiment", "unknown") if matched else "unknown"
            sentiment_reason  = matched[0].get("sentiment_reasoning", "") if matched else ""

            articles.append({
                "title":               art.get("title", ""),
                "published_utc":       art.get("published_utc", ""),
                "sentiment":           sentiment,
                "sentiment_reasoning": sentiment_reason,
            })
            _counters["articles_total"] += 1

        return articles

    except requests.exceptions.Timeout:
        print(f"  [WARN] {ticker}: timeout")
        _counters["requests_failed"] += 1
        return []
    except Exception as e:
        print(f"  [WARN] {ticker}: {type(e).__name__}: {e}")
        _counters["requests_failed"] += 1
        return []


def _fetch_news_batch(since_utc: str) -> dict[str, list[dict]]:
    """
    Batch режим: один запит з limit=1000, без ticker фільтра.
    Повертає dict: ticker -> [articles].
    """
    params = {
        "limit": 1000,
        "order": "desc",
        "published_utc.gte": since_utc,
    }
    try:
        _counters["requests_sent"] += 1
        resp = requests.get(
            BASE_URL,
            params=params,
            headers=_get_headers(),
            timeout=30,
        )
        if resp.status_code != 200:
            print(f"  [WARN] batch: HTTP {resp.status_code} — {resp.text[:200]}")
            _counters["requests_failed"] += 1
            return {}

        data = resp.json()
        results = data.get("results", [])
        by_ticker: dict[str, list[dict]] = {}

        for art in results:
            tickers_in_art = [t.upper() for t in (art.get("tickers") or [])]
            insights       = art.get("insights") or []

            for ticker in tickers_in_art:
                matched = [i for i in insights if i.get("ticker", "").upper() == ticker]
                if not matched:
                    matched = insights[:1]

                sentiment        = matched[0].get("sentiment", "unknown") if matched else "unknown"
                sentiment_reason = matched[0].get("sentiment_reasoning", "") if matched else ""

                entry = {
                    "title":               art.get("title", ""),
                    "published_utc":       art.get("published_utc", ""),
                    "sentiment":           sentiment,
                    "sentiment_reasoning": sentiment_reason,
                }
                by_ticker.setdefault(ticker, []).append(entry)
                _counters["articles_total"] += 1

        return by_ticker

    except Exception as e:
        print(f"  [WARN] batch fetch: {type(e).__name__}: {e}")
        _counters["requests_failed"] += 1
        return {}


# ── universe loader ───────────────────────────────────────────────────────────

def _load_symbols(universe_path: Path | None) -> list[str]:
    """
    Завантажує список символів з universe_snapshot.parquet або повертає дефолтний список.
    """
    path = universe_path or DEFAULT_UNIVERSE_PATH
    if path.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(path)
            col = next((c for c in ["symbol", "ticker"] if c in df.columns), None)
            if col:
                syms = sorted(df[col].dropna().str.upper().unique().tolist())
                print(f"  [INFO] universe: {len(syms)} символів з {path}")
                return syms
        except Exception as e:
            print(f"  [WARN] не вдалося прочитати universe: {e}")

    print(f"  [WARN] universe_snapshot.parquet не знайдено — використовується порожній список")
    return []


# ── core ──────────────────────────────────────────────────────────────────────

def run_news_fetch(
    symbols: list[str] | None = None,
    output_path: Path | None  = None,
    universe_path: Path | None = None,
) -> dict:
    """
    Головна функція. Повертає dict з результатами (також зберігає JSON).
    """
    if not MASSIVE_API_KEY:
        raise RuntimeError(
            "MASSIVE_API_KEY не знайдено в середовищі. "
            "Встановіть: $env:MASSIVE_API_KEY = 'your_key'"
        )

    out_path = output_path or DEFAULT_OUTPUT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    now_utc   = datetime.now(timezone.utc)
    since_utc = (now_utc - timedelta(hours=NEWS_LOOKBACK_HOURS)).strftime("%Y-%m-%dT%H:%M:%SZ")
    now_str   = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

    print(f"\n[NEWS] fetch_massive_news — старт")
    print(f"  lookback:  {NEWS_LOOKBACK_HOURS}h  (since {since_utc})")
    print(f"  mode:      {'batch (1 запит)' if NEWS_BATCH_MODE else 'per-ticker'}")
    print(f"  threshold: flagged якщо negative >= {NEWS_NEGATIVE_THRESHOLD}")

    # завантажити символи
    if symbols is None:
        symbols = _load_symbols(universe_path)
    if not symbols:
        print("  [ERROR] символи не передані і universe не знайдено — виходимо")
        return {}

    print(f"  symbols:   {len(symbols)}")

    # отримати новини
    symbol_articles: dict[str, list[dict]] = {}

    if NEWS_BATCH_MODE:
        print(f"\n  → batch запит...")
        batch_result = _fetch_news_batch(since_utc)
        for sym in symbols:
            symbol_articles[sym] = batch_result.get(sym.upper(), [])
    else:
        print(f"\n  → per-ticker запити (затримка {NEWS_REQUEST_DELAY_MS}ms між запитами)...")
        for i, sym in enumerate(symbols):
            arts = _fetch_news_for_ticker(sym, since_utc)
            symbol_articles[sym] = arts
            status = f"n={len(arts)}" if arts else "no_news"
            print(f"  [{i+1:3d}/{len(symbols)}] {sym:<8s}  {status}")
            if i < len(symbols) - 1:
                time.sleep(NEWS_REQUEST_DELAY_MS / 1000.0)

    # агрегація і флаги
    symbols_out: dict[str, dict] = {}
    flagged_count = 0

    for sym in symbols:
        arts = symbol_articles.get(sym, [])
        n_neg  = sum(1 for a in arts if a["sentiment"] == "negative")
        n_pos  = sum(1 for a in arts if a["sentiment"] == "positive")
        n_neu  = sum(1 for a in arts if a["sentiment"] == "neutral")
        flagged = n_neg >= NEWS_NEGATIVE_THRESHOLD

        if not arts:
            _counters["symbols_no_news"] += 1
        if flagged:
            flagged_count += 1
            _counters["symbols_flagged"] += 1

        symbols_out[sym] = {
            "n_articles": len(arts),
            "n_negative": n_neg,
            "n_positive": n_pos,
            "n_neutral":  n_neu,
            "flagged":    flagged,
            "articles":   arts,
        }

    result = {
        "generated_utc":     now_str,
        "lookback_hours":    NEWS_LOOKBACK_HOURS,
        "since_utc":         since_utc,
        "symbol_count":      len(symbols),
        "flagged_count":     flagged_count,
        "negative_threshold": NEWS_NEGATIVE_THRESHOLD,
        "symbols":           symbols_out,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # summary
    print(f"\n[NEWS] summary")
    print(f"  requests_sent:    {_counters['requests_sent']}")
    print(f"  requests_failed:  {_counters['requests_failed']}")
    print(f"  articles_total:   {_counters['articles_total']}")
    print(f"  symbols_no_news:  {_counters['symbols_no_news']}")
    print(f"  symbols_flagged:  {_counters['symbols_flagged']}")
    print(f"  output:           {out_path}")

    if flagged_count:
        print(f"\n  ⚠  FLAGGED ({flagged_count}):")
        for sym, d in symbols_out.items():
            if d["flagged"]:
                neg_arts = [a for a in d["articles"] if a["sentiment"] == "negative"]
                for a in neg_arts:
                    print(f"    {sym}: [{a['published_utc'][:16]}] {a['title'][:80]}")
                    print(f"      → {a['sentiment_reasoning'][:120]}")
    else:
        print(f"\n  ✓  Жодного символу не флагнуто (negative < {NEWS_NEGATIVE_THRESHOLD})")

    return result


# ── standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    rc = 1
    try:
        run_news_fetch()
        rc = 0
    except Exception:
        traceback.print_exc()
        print("\n[CRASHED]")
    finally:
        print()
        input("Press Enter to exit...")
    sys.exit(rc)
