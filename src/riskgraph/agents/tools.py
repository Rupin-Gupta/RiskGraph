"""Agent tools (SPEC §10.2): typed, deterministic, read-only except dispatch.

A Toolbox is bound to one incident's risk run. The LLM passes dates and scopes; the run_id stays
hidden, so it cannot reveal whether the case carries overrides. Every successful call is logged
to `tool_results` under a result_id hashed from (run_id, tool, args): a repeated call maps to the
same row, and evidence can always be re-fetched by its result_id.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import insert, select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError

from riskgraph import cli
from riskgraph.book.generate import load_book
from riskgraph.book.schema import Trade
from riskgraph.db.tables import dq_findings, limit_status, risk_results, tool_results
from riskgraph.marketdata.controls import cross_gap_bp
from riskgraph.pricing.market import FACTORS
from riskgraph.risk.var import SCOPES

SNIPPET = 700  # characters of each retrieved section shown to the agent
Search = Callable[[str, Mapping[str, str]], list[dict[str, Any]]]
UNITS = (
    "USD unless the metric name says otherwise: sens_<factor> is USD P&L per +1% move (equities,"
    " FX, VIX) or per +1bp (DGS2/5/10); dv01 per +1bp parallel; utilization is a fraction."
)


class ToolError(Exception):
    """A tool call the LLM can fix (bad date, unknown scope or trade)."""


@dataclass(frozen=True)
class Case:
    """One incident as the tools see it (incident.json)."""

    incident_id: str
    as_of_date: str
    run_id: str
    alert: dict[str, str]
    book_override: Path | None = None
    market_override: Path | None = None

    @classmethod
    def load(cls, incident_dir: Path) -> Case:
        doc = json.loads((incident_dir / "incident.json").read_text())
        keys = ("book_override", "market_override")
        paths = {k: incident_dir / doc[k] if doc[k] else None for k in keys}
        return cls(doc["incident_id"], doc["as_of_date"], doc["run_id"], doc["alert"], **paths)


# -- argument schemas (what the LLM sees) ---------------------------------------------------
DATE = Field(description="As-of date, YYYY-MM-DD.")
DESK = Field(description="fx, rates, equity_derivatives, or firm.")


class LimitArgs(BaseModel):
    date: str = DATE
    scope: str = Field("all", description="fx, rates, equity_derivatives, firm, or all.")


class DeskArgs(BaseModel):
    date: str = DATE
    desk: str = DESK


class TopArgs(BaseModel):
    date: str = DATE
    desk: str = DESK
    k: int = Field(5, ge=1, le=10, description="How many trades and factors to return.")


class NewTradeArgs(BaseModel):
    date: str = DATE
    desk: str = Field("all", description="fx, rates, equity_derivatives, or all.")


class TradeArgs(BaseModel):
    trade_id: str


class StressArgs(BaseModel):
    date: str = DATE
    scenario: str = Field("all", description="Scenario name from configs/scenarios.yaml, or all.")


class FindingArgs(BaseModel):
    date: str = DATE
    factor: str = Field("all", description="Risk factor (e.g. SPY, DGS10, EURUSD=X) or all.")


class SourceArgs(BaseModel):
    factor: str = Field(description="Risk factor, e.g. EURUSD=X.")
    date: str = DATE


class PolicyArgs(BaseModel):
    query: str = Field(description="What to look up in the limit policy, procedures, or Basel.")
    filters: dict[str, str] = Field(
        default_factory=dict,
        description='Optional, e.g. {"doc_id": "MRLP"} (MRLP, MDCP, MAR, CRE).',
    )


# -- the tools --------------------------------------------------------------------------------
def _day(box: Toolbox, date: str) -> None:
    if date != box.case.as_of_date:
        raise ToolError(f"only {box.case.as_of_date} is available for this incident")


def _scope(scope: str, allow_all: bool = False) -> None:
    if scope not in SCOPES and not (allow_all and scope == "all"):
        raise ToolError(f"unknown scope {scope!r}; use one of {', '.join(SCOPES)}")


def get_limit_status(box: Toolbox, date: str, scope: str = "all") -> dict[str, Any]:
    """Limit utilization and status (ok < 90% <= warning < 100% <= breach) for the run."""
    _day(box, date)
    _scope(scope, allow_all=True)
    t = limit_status.c
    q = select(t.scope, t.metric, t.value, t.limit, t.utilization, t.status)
    q = q.where(t.run_id == box.case.run_id).order_by(t.scope, t.metric)
    rows = [r._asdict() for r in box.rows(q) if scope in ("all", r.scope)]
    units = "value and limit in USD; utilization = value / limit"
    return {"date": date, "units": units, "rows": rows}


def get_risk_results(box: Toolbox, date: str, desk: str) -> dict[str, Any]:
    """Every risk metric for one desk or the firm: VaR/ES (HS, MC, parametric), PV, DV01,
    vega, theta, factor sensitivities, stress P&L, and VaR explain effects."""
    _day(box, date)
    _scope(desk)
    t = risk_results.c
    q = select(t.metric, t.value).where(t.run_id == box.case.run_id, t.desk == desk)
    metrics = dict(box.rows(q.order_by(t.metric)))
    return {"date": date, "desk": desk, "units": UNITS, "metrics": metrics}


def run_var_explain(box: Toolbox, date: str, desk: str) -> dict[str, Any]:
    """Day-over-day change in 99% HS VaR (USD) split into the position effect (new or changed
    trades at yesterday's market), market effect (yesterday's trades at today's market), and
    interaction."""
    _day(box, date)
    _scope(desk)
    e = box.run["explain"][desk]
    effects = {k: e[k] for k in ("total", "position", "market", "interaction")}
    return {"date": date, "prev_date": box.run["prev_date"], "desk": desk, "units": "USD"} | effects


def get_top_contributors(box: Toolbox, date: str, desk: str, k: int = 5) -> dict[str, Any]:
    """Largest contributors to 99% HS VaR (USD, mean loss over the tail scenarios) by trade and
    by risk factor. cross_effects is the part no single factor explains."""
    _day(box, date)
    _scope(desk)
    e = box.run["explain"][desk]
    return {
        "date": date,
        "desk": desk,
        "units": "USD",
        "top_trades": e["top_trades"][:k],
        "top_factors": e["top_factors"][:k],
    }


def list_new_trades(box: Toolbox, date: str, desk: str = "all") -> dict[str, Any]:
    """Trades in today's book that were not in the previous day's book."""
    _day(box, date)
    if desk != "all":
        _scope(desk)
    base = {t.trade_id for t in box.base_book}
    new = [t for t in box.book if t.trade_id not in base and desk in ("all", t.desk)]
    return {"date": date, "desk": desk, "new_trades": [t.model_dump(mode="json") for t in new]}


def get_trade(box: Toolbox, trade_id: str) -> dict[str, Any]:
    """Terms of one trade, with its PV (USD) and option Greeks from today's run."""
    trade = next((t for t in box.book if t.trade_id == trade_id), None)
    if trade is None:
        raise ToolError(f"unknown trade_id {trade_id!r}")
    row = next(r for r in box.run["trades"] if r["trade_id"] == trade_id)
    risk = {k: v for k, v in row.items() if k not in ("trade_id", "desk", "instrument_type")}
    return {"trade": trade.model_dump(mode="json"), "risk_usd": risk}


def get_greeks(box: Toolbox, date: str, desk: str) -> dict[str, Any]:
    """Option Greeks in USD for a desk: totals and per option (delta_1pct, gamma_1pct per 1% move,
    vega_1pt per vol point, theta_1d per day)."""
    _day(box, date)
    _scope(desk)
    names = ("delta_1pct", "gamma_1pct", "vega_1pt", "theta_1d")
    opts = [r for r in box.run["trades"] if "vega_1pt" in r and desk in ("firm", r["desk"])]
    per = [{"trade_id": r["trade_id"]} | {g: r[g] for g in names} for r in opts]
    totals = {g: sum(r[g] for r in opts) for g in names}
    return {"date": date, "desk": desk, "units": "USD", "totals": totals, "options": per}


def run_stress(box: Toolbox, date: str, scenario: str = "all") -> dict[str, Any]:
    """Stress-test P&L (USD, negative = loss) per scope for one scenario or all of them."""
    _day(box, date)
    stress = box.run["stress"]
    if scenario != "all" and scenario not in stress:
        raise ToolError(f"unknown scenario {scenario!r}; use one of {', '.join(stress)}")
    names = list(stress) if scenario == "all" else [scenario]
    return {"date": date, "units": "USD", "scenarios": {n: stress[n] for n in names}}


def get_dq_findings(box: Toolbox, date: str, factor: str = "all") -> dict[str, Any]:
    """Market data control findings for the run date. critical: rule checks, factor held flat
    (zero shocks) in the run. warning: anomaly model, factor stays live in the run."""
    _day(box, date)
    t = dq_findings.c
    q = select(t.factor, t.check, t.severity, t.detail).where(t.run_id == box.case.run_id)
    rows = [r._asdict() for r in box.rows(q.order_by(t.factor, t.check))]
    rows = [r for r in rows if factor in ("all", r["factor"])]
    held = box.run["controls"]["excluded_factors"]
    return {"date": date, "factor": factor, "findings": rows, "held_flat": held}


def compare_sources(box: Toolbox, factor: str, date: str) -> dict[str, Any]:
    """Primary (yfinance) level vs the second source (FRED, previous noon print) in bp. Only
    EURUSD=X and INR=X have a second source; tolerance 200bp."""
    _day(box, date)
    if factor not in FACTORS:
        raise ToolError(f"unknown factor {factor!r}; use one of {', '.join(FACTORS)}")
    cc = box.cfg.risk["controls"]["cross_source"]
    source = cc["pairs"].get(factor)
    if source is None:
        return {"factor": factor, "date": date, "second_source": None, "note": "no second source"}
    panel = cli.market_panel(box.case.market_override)
    ref = panel[source].shift(1).ffill(limit=3)
    return {
        "factor": factor,
        "date": date,
        "second_source": source,
        "primary_level": float(panel.at[date, factor]),
        "second_source_level": float(ref.loc[date]),
        "gap_bp": float(cross_gap_bp(panel, factor, source).loc[date]),
        "tolerance_bp": float(cc["tolerance_bp"]),
    }


def search_policy(
    box: Toolbox, query: str, filters: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """Search the Meridian Bank limit policy (MRLP), market data controls procedure (MDCP), and
    Basel MAR / CRE50-55. Returns sections with doc_id and section_id to cite."""
    if box.search is None:
        raise ToolError("policy search is not available")
    try:
        hits = box.search(query, dict(filters or {}))
    except ValueError as e:
        raise ToolError(str(e)) from None
    # ponytail: excerpts bound the token budget; revisit with re-ranking in phase 05a
    return {"query": query, "results": [h | {"text": h["text"][:SNIPPET]} for h in hits]}


@dataclass(frozen=True)
class Spec:
    fn: Callable[..., dict[str, Any]]
    args: type[BaseModel]


TOOLS: dict[str, Spec] = {
    "get_limit_status": Spec(get_limit_status, LimitArgs),
    "get_risk_results": Spec(get_risk_results, DeskArgs),
    "run_var_explain": Spec(run_var_explain, DeskArgs),
    "get_top_contributors": Spec(get_top_contributors, TopArgs),
    "list_new_trades": Spec(list_new_trades, NewTradeArgs),
    "get_trade": Spec(get_trade, TradeArgs),
    "get_greeks": Spec(get_greeks, DeskArgs),
    "run_stress": Spec(run_stress, StressArgs),
    "get_dq_findings": Spec(get_dq_findings, FindingArgs),
    "compare_sources": Spec(compare_sources, SourceArgs),
    "search_policy": Spec(search_policy, PolicyArgs),
}


def result_id(run_id: str, tool: str, args: Mapping[str, Any]) -> str:
    key = json.dumps([run_id, tool, args], sort_keys=True, default=str)
    return "R" + hashlib.sha256(key.encode()).hexdigest()[:12]


class Toolbox:
    """The tools for one incident, with logging to tool_results."""

    def __init__(self, case: Case, engine: Engine, search: Search | None = None) -> None:
        self.case, self.engine, self.search = case, engine, search
        self.cfg = cli.configs()

    @cached_property
    def run(self) -> dict[str, Any]:
        doc: dict[str, Any] = json.loads((cli.RUNS / self.case.run_id / "run.json").read_text())
        return doc

    @cached_property
    def base_book(self) -> list[Trade]:
        return load_book(cli.BOOK)

    @cached_property
    def book(self) -> list[Trade]:
        return load_book(self.case.book_override) if self.case.book_override else self.base_book

    def rows(self, query: Any) -> list[Any]:
        with self.engine.connect() as conn:
            return list(conn.execute(query))

    def compute(self, tool: str, args: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """(validated args, result) without logging. Raises ToolError or ValidationError."""
        spec = TOOLS[tool]
        parsed = spec.args.model_validate(dict(args)).model_dump()
        return parsed, spec.fn(self, **parsed)

    def call(self, tool: str, args: Mapping[str, Any]) -> dict[str, Any]:
        """Run a tool and log it. Errors come back as {"error": ...} for the LLM to fix."""
        if tool not in TOOLS:
            return {"error": f"unknown tool {tool!r}"}
        try:
            parsed, result = self.compute(tool, args)
        except (ToolError, ValidationError) as e:
            return {"error": str(e)}
        rid = result_id(self.case.run_id, tool, parsed)
        row = {"result_id": rid, "run_id": self.case.run_id, "tool": tool}
        try:
            with self.engine.begin() as conn:
                conn.execute(insert(tool_results).values(row | {"args": parsed, "result": result}))
        except IntegrityError:
            pass  # the same call was logged before; its result is identical
        return {"result_id": rid} | result


def fetch_result(engine: Engine, rid: str) -> dict[str, Any] | None:
    """A logged tool result by result_id: {run_id, tool, args, result}, or None."""
    t = tool_results.c
    with engine.connect() as conn:
        q = select(t.run_id, t.tool, t.args, t.result).where(t.result_id == rid)
        row = conn.execute(q).first()
    return row._asdict() if row else None


# -- dispatch (not available to any LLM) ------------------------------------------------------
def approval_token(thread_id: str, incident_id: str) -> str:
    """Server-issued approval token: HMAC of the thread and incident with APPROVAL_SECRET."""
    secret = os.environ.get("APPROVAL_SECRET")
    if not secret:
        raise RuntimeError("APPROVAL_SECRET is not set (see .env.example)")
    msg = f"{thread_id}:{incident_id}".encode()
    return hmac.new(secret.encode(), msg, hashlib.sha256).hexdigest()


def send_escalation_email(
    incident_id: str, approval_token: str, *, thread_id: str, as_of_date: str, note: str
) -> Path:
    """Phase 02 dispatch stub: write the approved note to data/runs/<date>/escalations/
    <incident_id>.md instead of emailing (SES arrives in phase 03)."""
    if not hmac.compare_digest(approval_token, globals()["approval_token"](thread_id, incident_id)):
        raise PermissionError("invalid approval token")
    out = cli.RUNS / as_of_date / "escalations" / f"{incident_id}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(note.rstrip() + "\n")
    return out
