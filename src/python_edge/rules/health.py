from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple

@dataclass(frozen=True)
class RetiredRule:
    signature: str
    support: int
    median_signed: float
    es5_signed: float

def _extract_rule_obj(item: Any) -> Any | None:
    if hasattr(item, "conds") and hasattr(item, "meta"):
        return item
    if isinstance(item, (tuple, list)):
        for x in item:
            if hasattr(x, "conds") and hasattr(x, "meta"):
                return x
    if isinstance(item, dict):
        for k in ("rule", "r", "obj"):
            x = item.get(k)
            if hasattr(x, "conds") and hasattr(x, "meta"):
                return x
        for x in item.values():
            if hasattr(x, "conds") and hasattr(x, "meta"):
                return x
    return None

def health_filter_rules(
    df_tr,
    rules_in: Sequence[Any],
    *,
    direction: str,
    cfg,
    health_win: int,
    health_min_n: int,
    health_med_min: float,
    health_es5_min: float,
) -> Tuple[List[Any], List[RetiredRule]]:
    if not rules_in:
        return list(rules_in), []
    if "date" not in df_tr.columns:
        raise RuntimeError("health_filter_rules requires 'date' column in df_tr")

    dts = sorted(df_tr["date"].unique().tolist())
    if not dts:
        return list(rules_in), []
    win_dts = dts[-max(1, int(health_win)):]
    df_h = df_tr[df_tr["date"].isin(win_dts)].copy()

    from python_edge.rules.event_mining import score_rule_event

    fwd_col = f"fwd_{cfg.fwd_days}d_ret"
    event_col = "event_up" if direction == "long" else "event_dn"

    kept: List[Any] = []
    retired: List[RetiredRule] = []

    for item in rules_in:
        r = _extract_rule_obj(item)
        if r is None:
            kept.append(item)
            continue

        st = score_rule_event(df_h, r, event_col=event_col, fwd_col=fwd_col)
        if (st is None) or (int(st.support) < int(health_min_n)):
            kept.append(item)
            continue

        med = float(st.median_signed)
        es5 = float(st.es5_signed)

        if (med <= float(health_med_min)) or (es5 <= float(health_es5_min)):
            retired.append(RetiredRule(str(st.signature), int(st.support), med, es5))
        else:
            kept.append(item)

    retired.sort(key=lambda x: (x.median_signed, x.es5_signed))
    return kept, retired
