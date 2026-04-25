"""
check_massive_news.py
Діагностика: перевіряє endpoint /v2/reference/news на massive.com
для заданого тікера і виводить повну відповідь.
"""

import os
import json
import requests
import sys

MASSIVE_API_KEY = os.environ.get("MASSIVE_API_KEY", "")

def check_news(ticker: str, limit: int = 3):
    if not MASSIVE_API_KEY:
        print("[ERROR] MASSIVE_API_KEY не знайдено в середовищі.")
        print("  Встановіть змінну: $env:MASSIVE_API_KEY = 'your_key_here'")
        return

    url = "https://api.massive.com/v2/reference/news"
    params = {
        "ticker": ticker,
        "limit": limit,
        "order": "desc",
    }
    headers = {
        "Authorization": f"Bearer {MASSIVE_API_KEY}",
    }

    print(f"\n{'='*60}")
    print(f"  massive /v2/reference/news  |  ticker={ticker}  limit={limit}")
    print(f"{'='*60}")
    print(f"  URL: {url}")
    print(f"  Params: {params}\n")

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        print(f"  HTTP status: {resp.status_code}")
        print(f"  Content-Type: {resp.headers.get('Content-Type', '?')}\n")

        try:
            data = resp.json()
        except Exception:
            print("[ERROR] Відповідь не є валідним JSON:")
            print(resp.text[:2000])
            return

        # Загальна структура
        print("--- Верхній рівень відповіді ---")
        for k, v in data.items():
            if k != "results":
                print(f"  {k}: {v}")

        results = data.get("results", [])
        print(f"\n  results count: {len(results)}")

        if not results:
            print("\n[!] results порожній — новини не повернуто.")
            print("    Повна відповідь:")
            print(json.dumps(data, indent=2, ensure_ascii=False))
            return

        for i, article in enumerate(results):
            print(f"\n{'─'*50}")
            print(f"  Стаття [{i+1}]")
            print(f"{'─'*50}")
            for field, val in article.items():
                if field == "insights":
                    print(f"  insights ({len(val)} шт.):")
                    for ins in val:
                        print(f"    {json.dumps(ins, ensure_ascii=False)}")
                elif isinstance(val, (list, dict)):
                    print(f"  {field}: {json.dumps(val, ensure_ascii=False)}")
                else:
                    # обрізаємо довгі рядки для читабельності
                    s = str(val)
                    print(f"  {field}: {s[:200]}{'...' if len(s) > 200 else ''}")

    except requests.exceptions.ConnectionError as e:
        print(f"[ERROR] ConnectionError: {e}")
    except requests.exceptions.Timeout:
        print("[ERROR] Timeout (15s)")
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    limit  = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    check_news(ticker, limit)
    print(f"\n{'='*60}")
    input("\nНатисніть Enter для виходу...")
