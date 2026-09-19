"""Critic checks (SPEC §10.4). Checks 1-3 are deterministic; check 4 asks the LLM whether the
note is clear, correctly prioritized, and actionable. Each issue names the agent that must fix it.

1. Evidence: every value is re-fetched through its result_id (tool_results) and must match a
   number in that result (relative 1e-6, or the rounding of the value as written).
2. Citations: every (doc_id, section_id) must exist in the corpus and have been retrieved by
   search_policy during this investigation.
3. Rule consistency with the policy's evidence requirements (MRLP-5, MRLP-6.7).
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from decimal import Decimal
from typing import Any

from riskgraph.agents.tools import Toolbox, fetch_result
from riskgraph.pricing.market import FACTORS

ACTION = {  # MRLP-6.7 routing table
    "position_change": "escalate_to_risk_manager",
    "market_move": "escalate_to_risk_manager",
    "bad_market_data": "route_to_data_ops",
    "no_true_breach": "no_action",
    "unknown": "escalate_to_risk_manager",
}
SCALE = {"%": 0.01, "percent": 0.01, "pct": 0.01, "bp": 1.0, "k": 1e3, "thousand": 1e3}
SCALE |= {"m": 1e6, "mm": 1e6, "mn": 1e6, "million": 1e6, "bn": 1e9, "billion": 1e9}
ROUNDING_CAP = 0.005  # a rounded value may differ from the source by at most 0.5%


def numbers(obj: Any) -> Iterator[float]:
    """Every numeric leaf of a JSON-like value."""
    if isinstance(obj, bool):
        return
    if isinstance(obj, int | float):
        yield float(obj)
    elif isinstance(obj, Mapping):
        for v in obj.values():
            yield from numbers(v)
    elif isinstance(obj, list | tuple):
        for v in obj:
            yield from numbers(v)


def matches(value: float, unit: str, source: float) -> bool:
    """value (as written, in `unit`) equals source within 1e-6 or its own display rounding."""
    words = unit.lower().replace("usd", " ").replace("$", " ").split()
    scale = math.prod(SCALE.get(w, 1.0) for w in words) if words else 1.0
    if math.isclose(value * scale, source, rel_tol=1e-6, abs_tol=1e-9):
        return True
    exp = Decimal(repr(abs(value))).normalize().as_tuple().exponent
    half_ulp = 0.5 * 10.0 ** int(exp) * scale
    return abs(value * scale - source) <= min(half_ulp, ROUNDING_CAP * abs(source))


def issue(check: str, agent: str, message: str) -> dict[str, str]:
    return {"check": check, "agent": agent, "message": message}


def contributing(box: Toolbox, scope: str, k: int = 3) -> list[str]:
    rows = box.run["explain"][scope]["top_factors"]
    return [r["factor"] for r in rows if r["factor"] in FACTORS and r["contribution"] > 0][:k]


def check_report(
    report: Mapping[str, Any],
    alert: Mapping[str, Any],
    calls: Sequence[Mapping[str, Any]],
    retrieved: Sequence[Sequence[str]],
    box: Toolbox,
    sections: set[tuple[str, str]],
) -> list[dict[str, str]]:
    """Deterministic checks 1-3. Returns issues; empty means pass."""
    issues = []
    producer: dict[str, str] = {}
    for c in calls:
        producer.setdefault(c["result_id"], "attribution" if c["agent"] == "intake" else c["agent"])

    for i, ev in enumerate(report["evidence"], 1):
        rid = ev["result_id"]
        row = fetch_result(box.engine, rid)
        if row is None or rid not in producer or row["run_id"] != box.case.run_id:
            msg = f"evidence {i} cites result_id {rid}, which no tool returned in this run"
            issues.append(issue("evidence", "writer", msg))
        elif not any(matches(ev["value"], ev["unit"], x) for x in numbers(row["result"])):
            msg = (
                f"evidence {i} ({ev['claim']!r}): {ev['value']} {ev['unit']} matches no number in "
                f"{row['tool']} result {rid}; call the tool again and copy the exact value"
            )
            issues.append(issue("evidence", producer[rid], msg))

    got = {tuple(r) for r in retrieved}
    for c in report["policy_citations"]:
        key = (c["doc_id"], c["section_id"])
        if key not in sections:
            issues.append(issue("citation", "policy", f"{key[1]} ({key[0]}) is not in the corpus"))
        elif key not in got:
            msg = f"{key[1]} was not retrieved by search_policy; retrieve it or do not cite it"
            issues.append(issue("citation", "policy", msg))

    rc, scope, util = report["root_cause"], alert["scope"], float(alert["utilization"])
    top = contributing(box, scope)
    date = box.case.as_of_date
    findings = box.compute("get_dq_findings", {"date": date})[1]["findings"]
    flagged = sorted({f["factor"] for f in findings} & set(top))
    if rc == "no_true_breach" and util >= 1:
        msg = f"utilization is {util:.1%}: the limit is breached, so no_true_breach cannot hold"
        issues.append(issue("rule", "attribution", msg))
    if rc in ("position_change", "market_move") and util < 1:
        msg = f"utilization is {util:.1%} (below 100%): this is not a limit breach (MRLP-4.4)"
        issues.append(issue("rule", "attribution", msg))
    if rc == "position_change" and box.run["explain"][scope]["position"] <= 0:
        msg = "position_change needs a positive VaR explain position effect (MRLP-5.1)"
        issues.append(issue("rule", "attribution", msg))
    if rc == "bad_market_data" and not flagged:
        msg = f"bad_market_data needs a data-quality finding on a contributing factor {top}"
        issues.append(issue("rule", "data_quality", msg + " (MRLP-5.3)"))
    if rc == "market_move" and flagged:
        msg = f"data-quality findings on contributing factors {flagged}: rule out bad data"
        msg += " (MRLP-5.2)"
        issues.append(issue("rule", "data_quality", msg))
    if report["recommended_action"] != ACTION[rc]:
        msg = f"{rc} requires recommended_action {ACTION[rc]} (MRLP-6.7)"
        issues.append(issue("rule", "writer", msg))
    return issues
