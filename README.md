# Python-Edge

MVP: rule-mining + purged walk-forward + portfolio overlay (3/5 concurrent positions) to discover structural edge.
Data vendor naming: **massive** (massive.com) everywhere (code/docs/env vars).

## Layout
- src/python_edge: core library
- scripts: runnable entry points
- data/raw/massive_dataset: downloaded OHLCV shards
- outputs/reports: generated reports

## First steps
1) Download massive aggs (1D/1H/4H; 15m optional later)
2) Run QA (no next_url left in saved files)
3) Build features
4) Mine rules (hundreds)
5) Purged walk-forward evaluate rules
6) Portfolio overlay + report