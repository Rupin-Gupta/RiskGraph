"""Independent checker for agent reports (SPEC §11.2), deliberately separate from the critic so
the system does not grade itself.

Numeric faithfulness: each evidence value must match a number the cited tool returns when it is
re-executed now from the engine's stored outputs (not the logged copy the critic reads), either
exactly (relative 1e-6) or as that number rounded to 3 or more significant figures, in the unit
the evidence states. Citation validity: the section exists in the corpus manifest and appeared
in a search_policy result logged for this run.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from sqlalchemy import select
from sqlalchemy.engine import Engine

from riskgraph.agents.tools import Toolbox
from riskgraph.db.tables import tool_results

UNIT_SCALE = {"%": 1e-2, "percent": 1e-2, "k": 1e3, "m": 1e6, "mn": 1e6, "mm": 1e6}
UNIT_SCALE |= {"million": 1e6, "bn": 1e9, "billion": 1e9}


def scale_of(unit: str) -> float:
    s = 1.0
    for word in unit.lower().replace("$", " ").replace("usd", " ").split():
        s *= UNIT_SCALE.get(word, 1.0)
    return s


def flat(obj: Any) -> Iterable[float]:
    stack = [obj]
    while stack:
        x = stack.pop()
        if isinstance(x, dict):
            stack.extend(x.values())
        elif isinstance(x, list):
            stack.extend(x)
        elif isinstance(x, int | float) and not isinstance(x, bool):
            yield float(x)


def agree(value: float, unit: str, source: float) -> bool:
    """value (in `unit`) is the source number, exactly or rounded to >= 3 significant figures."""
    x = source / scale_of(unit)
    if math.isclose(value, x, rel_tol=1e-6, abs_tol=1e-12):
        return True
    return any(float(f"{x:.{sig}g}") == value for sig in range(3, 16))


def _row(engine: Engine, rid: str) -> Any:
    t = tool_results.c
    with engine.connect() as conn:
        q = select(t.run_id, t.tool, t.args, t.result).where(t.result_id == rid)
        return conn.execute(q).first()


def evidence_confirmed(
    box: Toolbox, evidence: Sequence[Mapping[str, Any]], calls: Sequence[Mapping[str, Any]]
) -> list[bool]:
    """One flag per evidence item."""
    ran = {c["result_id"] for c in calls}
    out = []
    for ev in evidence:
        row = _row(box.engine, ev["result_id"])
        if row is None or ev["result_id"] not in ran or row.run_id != box.case.run_id:
            out.append(False)
            continue
        fresh = box.compute(row.tool, row.args)[1]
        out.append(any(agree(float(ev["value"]), ev["unit"], x) for x in flat(fresh)))
    return out


def citations_valid(
    engine: Engine,
    citations: Sequence[Mapping[str, str]],
    calls: Sequence[Mapping[str, Any]],
    corpus: set[tuple[str, str]],
) -> list[bool]:
    """One flag per citation: in the corpus and returned by this run's search_policy calls."""
    seen: set[tuple[str, str]] = set()
    for c in calls:
        if c["tool"] == "search_policy":
            row = _row(engine, c["result_id"])
            hits = row.result["results"] if row else []
            seen |= {(r["doc_id"], r["section_id"]) for r in hits}
    valid = corpus & seen
    return [(c["doc_id"], c["section_id"]) in valid for c in citations]
