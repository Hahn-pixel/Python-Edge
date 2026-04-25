"""
diag_universe_symbols.py
Показує скільки символів є в universe_snapshot.parquet і порівнює з news_risk_flags.json
"""
import sys
import json
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def main():
    # 1. universe_snapshot.parquet
    parquet_path = ROOT / "artifacts/daily_cycle/universe/universe_snapshot.parquet"
    print(f"\n{'='*60}")
    print(f"  universe_snapshot.parquet")
    print(f"{'='*60}")
    if not parquet_path.exists():
        print(f"  [!] NOT FOUND: {parquet_path}")
    else:
        try:
            import pandas as pd
            df = pd.read_parquet(parquet_path)
            print(f"  shape:   {df.shape}")
            print(f"  columns: {list(df.columns)}")
            col = next((c for c in ["symbol","ticker"] if c in df.columns), None)
            if col:
                syms = sorted(df[col].dropna().str.upper().unique().tolist())
                print(f"  unique symbols ({col}): {len(syms)}")
                print(f"  first 20: {syms[:20]}")
                print(f"  last  10: {syms[-10:]}")
            else:
                print(f"  [!] немає колонки symbol/ticker")
                print(df.head(3).to_string())
        except Exception as e:
            print(f"  [ERROR] {e}")

    # 2. news_risk_flags.json
    flags_path = ROOT / "artifacts/news/news_risk_flags.json"
    print(f"\n{'='*60}")
    print(f"  news_risk_flags.json")
    print(f"{'='*60}")
    if not flags_path.exists():
        print(f"  [!] NOT FOUND — запустіть fetch_massive_news.py спочатку")
    else:
        with open(flags_path, encoding="utf-8") as f:
            flags = json.load(f)
        print(f"  generated_utc:  {flags.get('generated_utc')}")
        print(f"  symbol_count:   {flags.get('symbol_count')}")
        print(f"  flagged_count:  {flags.get('flagged_count')}")
        syms_in_flags = sorted(flags.get("symbols", {}).keys())
        print(f"  symbols in JSON: {len(syms_in_flags)}")

        # символи з no_news
        no_news = [s for s,d in flags["symbols"].items() if d["n_articles"] == 0]
        print(f"  symbols_no_news: {len(no_news)}")
        print(f"  no_news list: {sorted(no_news)}")

    # 3. порівняння якщо обидва є
    if parquet_path.exists() and flags_path.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(parquet_path)
            col = next((c for c in ["symbol","ticker"] if c in df.columns), None)
            if col:
                universe_syms = set(df[col].dropna().str.upper().unique())
                flags_syms    = set(flags.get("symbols", {}).keys())
                missing_in_flags   = sorted(universe_syms - flags_syms)
                extra_in_flags     = sorted(flags_syms - universe_syms)
                print(f"\n{'='*60}")
                print(f"  Порівняння universe vs news_risk_flags")
                print(f"{'='*60}")
                print(f"  universe: {len(universe_syms)} символів")
                print(f"  flags:    {len(flags_syms)} символів")
                print(f"  в universe але НЕ в flags ({len(missing_in_flags)}): {missing_in_flags}")
                print(f"  в flags але НЕ в universe ({len(extra_in_flags)}): {extra_in_flags}")
        except Exception as e:
            print(f"  [ERROR] порівняння: {e}")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
    finally:
        print()
        input("Press Enter to exit...")
    sys.exit(0)
