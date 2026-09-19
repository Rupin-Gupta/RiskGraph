"""Agent test helpers: a tiny risk run in SQLite + run.json, and a scripted fake LLM."""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from riskgraph import cli
from riskgraph.agents import schemas
from riskgraph.agents.tools import Toolbox
from riskgraph.db.tables import limit_status, metadata, replace_rows, risk_results

DAY = "2024-03-15"
RUN = "2024-03-15+abcd1234"
CASE = {
    "incident_id": "INC-T01",
    "as_of_date": DAY,
    "run_id": RUN,
    "alert": {"scope": "fx", "metric": "var_99_1d"},
    "book_override": None,
    "market_override": None,
}
SECTIONS = {("MRLP", "MRLP-6.5"), ("MRLP", "MRLP-5.2")}
TRADE: dict[str, Any] = {"trade_id": "FXF001", "desk": "fx", "instrument_type": "fx_forward"}
TRADE |= {"counterparty_id": "CP01", "currency": "EUR", "notional": 1e7, "trade_date": "2024-01-02"}
TRADE |= {"maturity_date": "2024-12-02", "currency_pair": "EURUSD", "forward_rate": 1.1}
EXPLAIN = {
    "total": 12000.0,
    "position": 0.0,
    "market": 12000.0,
    "interaction": 0.0,
    "top_trades": [{"trade_id": "FXF001", "contribution": 2100000.5}],
    "top_factors": [
        {"factor": "EURUSD=X", "contribution": 2050000.25},
        {"factor": "INR=X", "contribution": 40000.0},
        {"factor": "cross_effects", "contribution": 10000.0},
    ],
}


def search(query: str, filters: Any) -> list[dict[str, Any]]:
    hits = [("MRLP-6.5", "Market-driven breach"), ("MRLP-5.2", "Market move")]
    return [
        {"doc_id": "MRLP", "section_id": s, "section_path": p, "text": p, "score": 0.8}
        for s, p in hits
    ]


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Engine:
    """SQLite with one incident run; run.json and the base book under tmp_path."""
    monkeypatch.setattr(cli, "RUNS", tmp_path / "runs")
    book = tmp_path / "book.json"
    book.write_text(json.dumps({"trades": [TRADE]}))
    monkeypatch.setattr(cli, "BOOK", book)
    monkeypatch.setenv("APPROVAL_SECRET", "test-secret")
    run = {
        "run_id": RUN,
        "prev_date": "2024-03-14",
        "controls": {"findings": [], "excluded_factors": []},
        "explain": {s: EXPLAIN for s in ("fx", "rates", "equity_derivatives", "firm")},
        "stress": {"rates_up_200bp": {"pnl": {"fx": -1.0e5, "firm": -2.5e6}}},
        "trades": [TRADE | {"pv": 1234.5}],
    }
    (tmp_path / "runs" / RUN).mkdir(parents=True)
    (tmp_path / "runs" / RUN / "run.json").write_text(json.dumps(run))
    engine = create_engine(f"sqlite:///{tmp_path / 't.db'}")  # threads share one file
    metadata.create_all(engine)
    lim = {"scope": "fx", "metric": "var_99_1d", "value": 2.2e6, "limit": 2.18e6}
    lim |= {"utilization": 2.2e6 / 2.18e6, "status": "breach", "date": date(2024, 3, 15)}
    risk = {"desk": "fx", "metric": "var_99_1d_hs", "value": 2.2e6, "date": date(2024, 3, 15)}
    replace_rows(engine, RUN, {limit_status: [lim], risk_results: [risk]})
    return engine


@pytest.fixture
def box(env: Engine) -> Toolbox:
    from riskgraph.agents.graph import to_case

    return Toolbox(to_case(CASE), env, search)


def blocks(msgs: list[BaseMessage]) -> list[tuple[str, Any]]:
    """(source, JSON body) of every <untrusted_data> block in the messages."""
    out = []
    for m in msgs:
        for src, body in re.findall(
            r'<untrusted_data source="([^"]+)">\n(.*?)\n</untrusted_data>', str(m.content), re.S
        ):
            out.append((src, json.loads(body)))
    return out


USAGE = {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110}
SCRIPTS: dict[
    str, list[tuple[str, dict[str, Any]]]
] = {  # role keyword in the system prompt -> tool calls on the first turn
    "attribution specialist": [("get_top_contributors", {"date": DAY, "desk": "fx", "k": 5})],
    "data-quality specialist": [
        ("get_top_contributors", {"date": DAY, "desk": "fx", "k": 3}),
        ("get_dq_findings", {"date": DAY}),
    ],
    "policy specialist": [("search_policy", {"query": "market-driven breach escalation"})],
    "end to end": [
        ("get_top_contributors", {"date": DAY, "desk": "fx", "k": 3}),
        ("search_policy", {"query": "market-driven breach escalation"}),
    ],
}


class FakeLLM:
    """Scripted stand-in for ChatBedrockConverse (bind_tools / with_structured_output)."""

    def __init__(self, root_cause: str = "market_move", corrupt_first: bool = False) -> None:
        self.root_cause, self.corrupt_first = root_cause, corrupt_first
        self.turns: Counter[str] = Counter()
        self.writes = 0

    @staticmethod
    def role(msgs: list[BaseMessage]) -> str:
        return next((r for r in SCRIPTS if r in str(msgs[0].content)), "")

    def bind_tools(self, tools: Any) -> Any:
        llm = self

        class Bound:
            def invoke(self, msgs: list[BaseMessage]) -> AIMessage:
                if not isinstance(msgs[-1], HumanMessage):
                    return AIMessage(content="done", usage_metadata=USAGE)
                role = llm.role(msgs)
                llm.turns[role] += 1
                script = enumerate(SCRIPTS[role])
                calls = [{"name": n, "args": a, "id": f"call{i}"} for i, (n, a) in script]
                return AIMessage(content="", tool_calls=calls, usage_metadata=USAGE)

        return Bound()

    def with_structured_output(self, schema: Any, include_raw: bool = False) -> Any:
        llm = self

        class Structured:
            def invoke(self, msgs: list[BaseMessage]) -> dict[str, Any]:
                raw = AIMessage(content="", usage_metadata=USAGE)
                return {"raw": raw, "parsed": llm.answer(schema, msgs), "parsing_error": None}

        return Structured()

    def answer(self, schema: Any, msgs: list[BaseMessage]) -> Any:
        found = blocks(msgs)
        if schema is schemas.Plan:
            agents = ("attribution", "data_quality", "policy")
            return schemas.Plan.model_validate(
                {"tasks": [{"agent": a, "question": "look"} for a in agents]}
            )
        if schema is schemas.NoteReview:
            return schemas.NoteReview(ok=True, feedback="")
        tops = [b for _, b in found if "top_factors" in b]
        hits = [r for _, b in found if "results" in b for r in b["results"]]
        cites = [{"doc_id": r["doc_id"], "section_id": r["section_id"]} for r in hits]
        if schema is schemas.Findings:
            ev = [
                {"claim": "top factor", "value": t["top_factors"][0]["contribution"], "unit": "USD"}
                | {"result_id": t["result_id"]}
                for t in tops[:1]
            ]
            return schemas.Findings.model_validate(
                {"summary": "checked", "evidence": ev, "citations": cites}
            )
        # ReportDraft: the writer reads findings; the baseline reads raw tool results
        dq = [b for s, b in found if s == "agent:data_quality"]
        pol = [b for s, b in found if s == "agent:policy"]
        ev = (
            dq[0]["evidence"]
            if dq
            else [
                {"claim": "top factor", "value": t["top_factors"][0]["contribution"], "unit": "USD"}
                | {"result_id": t["result_id"]}
                for t in tops[:1]
            ]
        )
        self.writes += 1
        if self.corrupt_first and self.writes == 1:
            ev = [e | {"value": e["value"] * 1.5} for e in ev]
        action = "no_action" if self.root_cause == "no_true_breach" else "escalate_to_risk_manager"
        return schemas.ReportDraft.model_validate(
            {
                "root_cause": self.root_cause,
                "confidence": 0.8,
                "evidence": ev,
                "recommended_action": action,
                "policy_citations": pol[0]["citations"] if pol else cites[:1],
                "draft_note": "**Action: escalate to the risk manager.** FX VaR is over its limit.",
            }
        )
