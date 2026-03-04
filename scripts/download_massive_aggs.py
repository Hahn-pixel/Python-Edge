# scripts/download_massive_aggs.py
# Double-click runnable. Never auto-closes.
# massive (massive.com) data downloader (Polygon-compatible agg endpoints).
# Features:
# - full pagination via next_url (ALL pages)
# - sharded date ranges per TF (stable downloads, resume-friendly)
# - QA: any 'next_url' present in saved file => FAIL
# - resume: skip existing shard files that pass QA
# - optional merge per TF into a single FULL file
#
# Data output:
#   data/raw/massive_dataset/<TICKER>/aggs_<tf>_<from>_<to>.json
#   data/raw/massive_dataset/<TICKER>/aggs_<tf>_<START>_<END>__FULL.json  (optional)
#
# Universe:
#   - DATA_TICKERS env or data/universe_etf_first_30.txt (recommended)

from __future__ import annotations

import os
import json
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone, date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


def _press_enter_exit(code: int) -> None:
    try:
        print(f"\n[EXIT] code={code}")
        input("Press Enter to exit...")
    except Exception:
        pass
    raise SystemExit(code)


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _env_str(name: str, default: str = "") -> str:
    v = os.environ.get(name, default)
    return (v if v is not None else default).strip()


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    s = str(v).strip().lower()
    return s in ("1", "true", "yes", "y", "on")


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or not str(v).strip():
        return default
    try:
        return int(float(str(v).strip()))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None or not str(v).strip():
        return default
    try:
        return float(str(v).strip())
    except Exception:
        return default


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_json_atomic(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _find_key_anywhere(obj: Any, key: str) -> List[str]:
    hits: List[str] = []

    def walk(x: Any, path: str) -> None:
        if isinstance(x, dict):
            for k, v in x.items():
                p2 = f"{path}.{k}" if path else k
                if k == key:
                    hits.append(p2)
                walk(v, p2)
        elif isinstance(x, list):
            for i, v in enumerate(x):
                walk(v, f"{path}[{i}]")

    walk(obj, "")
    return hits


def _parse_ymd(s: str) -> date:
    y, m, d = [int(x) for x in s.split("-")]
    return date(y, m, d)


def _fmt_ymd(d: date) -> str:
    return d.isoformat()


def _split_range_days(start_ymd: str, end_ymd: str, chunk_days: int) -> List[Tuple[str, str]]:
    a = _parse_ymd(start_ymd)
    b = _parse_ymd(end_ymd)
    if b < a:
        return []
    out: List[Tuple[str, str]] = []
    cur = a
    while cur <= b:
        nxt = min(b, cur + timedelta(days=chunk_days - 1))
        out.append((_fmt_ymd(cur), _fmt_ymd(nxt)))
        cur = nxt + timedelta(days=1)
    return out


class PermanentHTTPError(RuntimeError):
    pass


@dataclass(frozen=True)
class Config:
    root: Path
    out_dir: Path
    tickers: List[str]
    start: str
    end: str
    api_key: str

    # TF switches
    fetch_1d: bool
    fetch_4h: bool
    fetch_1h: bool
    fetch_15m: bool

    # sharding
    chunk_days_15m: int
    chunk_days_1h: int
    chunk_days_4h: int
    chunk_days_1d: int

    # paging/safety
    limit: int
    max_pages: int
    max_bars: int

    # network
    timeout_sec: int
    max_retries: int
    backoff_base: float

    # behavior
    merge_full: bool
    resume_skip_ok: bool

    # QA hard fail
    qa_fail_on_next_url: bool


class MassiveClient:
    def __init__(self, api_key: str, timeout_sec: int, max_retries: int, backoff_base: float):
        self.api_key = api_key
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.s = requests.Session()

    def get_json(self, url: str, params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if params is None:
            params = {}
        params = dict(params)
        params["apiKey"] = self.api_key

        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                r = self.s.get(url, params=params, timeout=self.timeout_sec)

                if r.status_code in (401, 403):
                    raise PermanentHTTPError(f"{r.status_code} Forbidden/Unauthorized for url={r.url}")

                if r.status_code == 429:
                    sleep_s = min(60.0, (self.backoff_base ** attempt))
                    _log(f"[NET] 429 rate-limit -> sleep {sleep_s:.1f}s")
                    time.sleep(sleep_s)
                    continue

                r.raise_for_status()
                return r.json()

            except PermanentHTTPError:
                raise
            except Exception as e:
                last_err = e
                sleep_s = min(60.0, (self.backoff_base ** attempt))
                _log(f"[NET] error attempt={attempt}/{self.max_retries}: {e!r} -> sleep {sleep_s:.1f}s")
                time.sleep(sleep_s)

        raise RuntimeError(f"GET failed after retries. url={url} last_err={last_err!r}")


def _project_root_from_this_file() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_tickers(cfg_root: Path) -> List[str]:
    # 1) env
    for k in ("DATA_TICKERS", "LIVE_SYMBOLS", "WF_SYMBOLS"):
        v = _env_str(k, "")
        if v:
            return [x.strip().upper() for x in v.split(",") if x.strip()]

    # 2) file: ETF-first default
    p = cfg_root / "data" / "universe_etf_first_30.txt"
    if p.exists():
        txt = p.read_text(encoding="utf-8", errors="ignore")
        out: List[str] = []
        for line in txt.splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(s.upper())
        return out

    return []


def load_config() -> Config:
    root = _project_root_from_this_file()

    # output dir under data/raw/massive_dataset by default
    out_dir = Path(_env_str("DATA_OUT_DIR", str(root / "data" / "raw" / "massive_dataset")))

    tickers = _load_tickers(root)
    if not tickers:
        raise RuntimeError("No tickers. Set DATA_TICKERS=... or create data/universe_etf_first_30.txt")

    start = _env_str("DATA_START", "")
    end = _env_str("DATA_END", "")
    if not start or not end:
        raise RuntimeError("Missing DATA_START/DATA_END (YYYY-MM-DD). Example: DATA_START=2023-01-01 DATA_END=2026-02-28")

    # naming: massive
    api_key = _env_str("MASSIVE_API_KEY", "") or _env_str("POLYGON_API_KEY", "") or _env_str("API_KEY", "")
    if not api_key:
        raise RuntimeError("Missing API key. Set MASSIVE_API_KEY (preferred) or API_KEY")

    return Config(
        root=root,
        out_dir=out_dir,
        tickers=tickers,
        start=start,
        end=end,
        api_key=api_key,

        fetch_1d=_env_bool("DATA_1D", True),
        fetch_4h=_env_bool("DATA_4H", True),
        fetch_1h=_env_bool("DATA_1H", True),
        fetch_15m=_env_bool("DATA_15M", False),  # off by default for MVP

        chunk_days_15m=_env_int("DATA_CHUNK_DAYS_15M", 31),
        chunk_days_1h=_env_int("DATA_CHUNK_DAYS_1H", 120),
        chunk_days_4h=_env_int("DATA_CHUNK_DAYS_4H", 180),
        chunk_days_1d=_env_int("DATA_CHUNK_DAYS_1D", 366),

        limit=_env_int("DATA_AGGS_LIMIT", 50000),
        max_pages=_env_int("DATA_MAX_PAGES", 2000),
        max_bars=_env_int("DATA_MAX_BARS", 2_000_000),

        timeout_sec=_env_int("DATA_TIMEOUT", 30),
        max_retries=_env_int("DATA_RETRIES", 6),
        backoff_base=_env_float("DATA_BACKOFF", 1.6),

        merge_full=_env_bool("DATA_MERGE_FULL", True),
        resume_skip_ok=_env_bool("DATA_RESUME_SKIP_OK", True),

        qa_fail_on_next_url=_env_bool("DATA_QA_FAIL_ON_NEXT_URL", True),
    )


def _aggs_base_url(ticker: str, multiplier: int, timespan: str, start: str, end: str) -> str:
    # Endpoint is Polygon-style; we refer to vendor as "massive" in code/docs.
    return f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start}/{end}"


def fetch_aggs_all_pages(
    client: MassiveClient,
    base_url: str,
    params0: Dict[str, Any],
    max_pages: int,
    max_bars: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    pages = 0
    merged_results: List[Dict[str, Any]] = []
    first: Optional[Dict[str, Any]] = None

    url = base_url
    params = params0

    while True:
        js = client.get_json(url, params=params)
        if first is None:
            first = js

        pages += 1
        res = js.get("results") or []
        if res:
            merged_results.extend(res)

        if len(merged_results) > max_bars:
            raise RuntimeError(f"Safety cap exceeded: bars={len(merged_results)} > DATA_MAX_BARS={max_bars}")

        next_url = js.get("next_url")
        if not next_url:
            break

        if pages >= max_pages:
            raise RuntimeError(f"Safety cap exceeded: pages={pages} >= DATA_MAX_PAGES={max_pages} (still had next_url)")

        url = next_url
        params = None  # next_url already has cursor; client appends apiKey
        time.sleep(0.05)

    if first is None:
        first = {"status": "OK", "results": []}

    merged = dict(first)
    merged["results"] = merged_results
    merged["resultsCount"] = len(merged_results)
    merged["count"] = len(merged_results)
    merged.pop("next_url", None)
    merged["_pagination"] = {"pages": pages, "limit": params0.get("limit"), "base_url": base_url}
    merged["_written_utc"] = _utc_now()

    stats = {"pages": pages, "bars": len(merged_results)}
    return merged, stats


def qa_no_next_url_in_file(path: Path) -> Tuple[bool, List[str]]:
    obj = _read_json(path)
    hits = _find_key_anywhere(obj, "next_url")
    return (len(hits) == 0), hits


def _dedup_sort_results_by_t(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for r in sorted(rows, key=lambda x: int(x.get("t", 0))):
        t = r.get("t")
        if t is None:
            continue
        if t in seen:
            continue
        seen.add(t)
        out.append(r)
    return out


def merge_shards_to_full(out_path: Path, shard_paths: List[Path], meta: Dict[str, Any], qa_fail_on_next_url: bool) -> None:
    all_results: List[Dict[str, Any]] = []
    first: Optional[Dict[str, Any]] = None

    for p in shard_paths:
        obj = _read_json(p)
        if first is None:
            first = obj
        all_results.extend(obj.get("results") or [])

    if first is None:
        first = {"status": "OK", "results": []}

    merged = dict(first)
    merged["results"] = _dedup_sort_results_by_t(all_results)
    merged["resultsCount"] = len(merged["results"])
    merged["count"] = len(merged["results"])
    merged.pop("next_url", None)
    merged["_merged_from"] = [str(p) for p in shard_paths]
    merged["_meta"] = meta
    merged["_written_utc"] = _utc_now()

    _write_json_atomic(out_path, merged)

    if qa_fail_on_next_url:
        ok, hits = qa_no_next_url_in_file(out_path)
        if not ok:
            raise RuntimeError(f"QA FAIL after merge: next_url present in FULL file: {hits[:8]}")


def main() -> int:
    cfg = load_config()
    _ensure_dir(cfg.out_dir)

    client = MassiveClient(cfg.api_key, cfg.timeout_sec, cfg.max_retries, cfg.backoff_base)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "written_utc": _utc_now(),
        "vendor": "massive",
        "start": cfg.start,
        "end": cfg.end,
        "tickers": cfg.tickers,
        "cfg": {
            "fetch": {"1d": cfg.fetch_1d, "4h": cfg.fetch_4h, "1h": cfg.fetch_1h, "15m": cfg.fetch_15m},
            "chunk_days": {"1d": cfg.chunk_days_1d, "4h": cfg.chunk_days_4h, "1h": cfg.chunk_days_1h, "15m": cfg.chunk_days_15m},
            "merge_full": cfg.merge_full,
            "resume_skip_ok": cfg.resume_skip_ok,
            "limit": cfg.limit,
            "qa_fail_on_next_url": cfg.qa_fail_on_next_url,
        },
        "files": [],
        "full_files": [],
        "errors": [],
        "qa_failures": [],
        "stats": [],
    }

    _log(f"[CFG] vendor=massive out_dir={cfg.out_dir}")
    _log(f"[CFG] tickers={len(cfg.tickers)} range={cfg.start}..{cfg.end}")
    _log(f"[CFG] TFs: 1d={cfg.fetch_1d} 4h={cfg.fetch_4h} 1h={cfg.fetch_1h} 15m={cfg.fetch_15m}")
    _log(f"[CFG] chunk_days: 1d={cfg.chunk_days_1d} 4h={cfg.chunk_days_4h} 1h={cfg.chunk_days_1h} 15m={cfg.chunk_days_15m}")
    _log(f"[CFG] merge_full={cfg.merge_full} resume_skip_ok={cfg.resume_skip_ok} QA_fail_on_next_url={cfg.qa_fail_on_next_url}")

    tf_specs: List[Tuple[str, int, str, int]] = []
    if cfg.fetch_15m:
        tf_specs.append(("15m", 15, "minute", cfg.chunk_days_15m))
    if cfg.fetch_1h:
        tf_specs.append(("1h", 1, "hour", cfg.chunk_days_1h))
    if cfg.fetch_4h:
        tf_specs.append(("4h", 4, "hour", cfg.chunk_days_4h))
    if cfg.fetch_1d:
        tf_specs.append(("1d", 1, "day", cfg.chunk_days_1d))

    for i, tkr in enumerate(cfg.tickers, 1):
        _log(f"[TICKER] {tkr} ({i}/{len(cfg.tickers)})")
        tdir = cfg.out_dir / tkr
        _ensure_dir(tdir)

        for tag, mult, span, chunk_days in tf_specs:
            shards = _split_range_days(cfg.start, cfg.end, chunk_days)
            shard_paths: List[Path] = []
            _log(f"[TF] {tkr} {tag} shards={len(shards)} chunk_days={chunk_days}")

            for (a, b) in shards:
                out_path = tdir / f"aggs_{tag}_{a}_{b}.json"

                if cfg.resume_skip_ok and out_path.exists():
                    try:
                        if cfg.qa_fail_on_next_url:
                            ok, hits = qa_no_next_url_in_file(out_path)
                            if ok:
                                shard_paths.append(out_path)
                                _log(f"[SKIP] {tkr} {tag} {a}..{b} (exists, QA ok)")
                                continue
                            else:
                                _log(f"[RETRY] {tkr} {tag} {a}..{b} (exists but QA FAIL: {hits[:3]})")
                        else:
                            shard_paths.append(out_path)
                            _log(f"[SKIP] {tkr} {tag} {a}..{b} (exists, QA not enforced)")
                            continue
                    except Exception:
                        _log(f"[RETRY] {tkr} {tag} {a}..{b} (exists but unreadable/QA error)")

                base_url = _aggs_base_url(tkr, mult, span, a, b)
                params0 = {"adjusted": "true", "sort": "asc", "limit": int(cfg.limit)}

                try:
                    merged, stats = fetch_aggs_all_pages(client, base_url, params0, cfg.max_pages, cfg.max_bars)
                    _write_json_atomic(out_path, merged)

                    if cfg.qa_fail_on_next_url:
                        ok, hits = qa_no_next_url_in_file(out_path)
                        if not ok:
                            manifest["qa_failures"].append({"ticker": tkr, "tf": tag, "file": str(out_path), "hits": hits})
                            _log(f"[FAIL] {tkr} {tag} {a}..{b} QA: next_url present at {hits[:5]}")
                            continue

                    shard_paths.append(out_path)
                    manifest["files"].append(str(out_path))
                    manifest["stats"].append({"ticker": tkr, "tf": tag, "from": a, "to": b, **stats})
                    _log(f"[OK] {tkr} {tag} {a}..{b} bars={stats['bars']} pages={stats['pages']} (QA ok)")

                except PermanentHTTPError as e:
                    manifest["errors"].append({"ticker": tkr, "tf": tag, "from": a, "to": b, "error": repr(e)})
                    _log(f"[ERR] {tkr} {tag} {a}..{b} {e!r}")
                except Exception as e:
                    manifest["errors"].append({"ticker": tkr, "tf": tag, "from": a, "to": b, "error": repr(e)})
                    _log(f"[ERR] {tkr} {tag} {a}..{b} {e!r}")

            if cfg.merge_full and shard_paths:
                try:
                    full_path = tdir / f"aggs_{tag}_{cfg.start}_{cfg.end}__FULL.json"
                    merge_shards_to_full(
                        out_path=full_path,
                        shard_paths=shard_paths,
                        meta={"ticker": tkr, "tf": tag, "start": cfg.start, "end": cfg.end, "vendor": "massive"},
                        qa_fail_on_next_url=cfg.qa_fail_on_next_url,
                    )
                    manifest["full_files"].append(str(full_path))
                    _log(f"[FULL] {tkr} {tag} -> {full_path.name}")
                except Exception as e:
                    manifest["errors"].append({"ticker": tkr, "tf": tag, "kind": "merge_full", "error": repr(e)})
                    _log(f"[ERR] {tkr} {tag} merge_full: {e!r}")

    mp = cfg.out_dir / f"manifest__massive_aggs__{run_id}.json"
    _write_json_atomic(mp, manifest)
    _log(f"[MANIFEST] wrote {mp}")
    _log(f"[SUMMARY] tickers={len(cfg.tickers)} files={len(manifest['files'])} full_files={len(manifest['full_files'])} errors={len(manifest['errors'])} qa_failures={len(manifest['qa_failures'])}")

    if manifest["qa_failures"]:
        return 2
    if manifest["errors"]:
        return 1
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        _log("[FATAL] unhandled exception")
        traceback.print_exc()
        rc = 99
    _press_enter_exit(int(rc))