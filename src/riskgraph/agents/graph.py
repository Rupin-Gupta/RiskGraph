"""Multi-agent investigation graph (SPEC §10.1):

intake -> supervisor -> Send fan-out: attribution | data_quality | policy -> writer -> critic
  -> [issues: back to the named specialist (or writer), at most critic.max_loops times]
  -> human_approval (interrupt) -> dispatch | close

intake, human_approval, dispatch, and close are shared with the baseline (agents/baseline.py).
A run that exceeds its token or tool-call budget, or fails structured output twice, stops with
status needs_human.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt

from riskgraph.agents.critic import check_report
from riskgraph.agents.llm import (
    MAX_STEPS,
    Budget,
    BudgetExceeded,
    SchemaFailure,
    ask,
    config,
    react,
    wrap,
)
from riskgraph.agents.schemas import Breach, Findings, IncidentReport, NoteReview, Plan, ReportDraft
from riskgraph.agents.tools import Case, Toolbox, send_escalation_email

MakeBox = Callable[[Mapping[str, Any]], Toolbox]
POLICY_TURNS = 2  # one round of searches, then findings: retrieval was the largest token cost
SPECIALISTS = ("attribution", "data_quality", "policy")

ROLE = "You work in Market Risk Management at Meridian Bank, a fictional bank."
SCHEMA_GUIDE = """Report fields:
- root_cause: position_change (new or changed trades drive the metric: positive VaR explain
  position effect); market_move (unchanged positions, market moves or volatility drive it, no
  data-quality finding on a contributing risk factor); bad_market_data (a data-quality finding of
  any severity, critical or warning, sits on a risk factor that contributes to the metric; a
  source gap inside tolerance does not clear a warning); no_true_breach (utilization below 100%:
  a near miss or a false alert); unknown (the evidence is insufficient).
- recommended_action: escalate_to_risk_manager, route_to_data_ops, or no_action, following the
  routing in the Meridian Bank limit policy.
- confidence: the probability that root_cause is right.
- evidence: 2-6 items. Copy each value exactly from a number in a tool result, with that
  result's result_id and the unit it uses (USD, fraction, bp). Never compute, count, round, or
  convert values. State absences (no new trades, no findings) in the note, not as evidence.
- policy_citations: 2-4 sections, only from search_policy results, with exact doc_id and
  section_id: the section defining the chosen root-cause category, the section setting its
  escalation or routing path, and (for a breach) the escalation rule for the alerted limit type.
- draft_note: Markdown for a risk manager, under 250 words. Lead with the required action, then
  the limit, value, utilization, root cause, key evidence, and the cited sections. In the note,
  round numbers for reading (USD 1.35mn, 103.5%); evidence values stay exact."""

SUPERVISOR = f"""{ROLE} You supervise the investigation of a limit alert. Plan it: pick the
specialists to call and write one specific question for each.
- attribution: why the metric moved: VaR explain (position vs market effect), top contributing
  trades and risk factors, new trades, Greeks, stress results, limit status.
- data_quality: whether bad market data drives the move: data-quality findings on the date,
  contributing risk factors, cross-source comparison (FX only).
- policy: which policy and Basel sections apply: status definitions, root-cause categories,
  escalation paths and routing.
Call every specialist whose answer could change the root cause or the recommended action."""

PROMPTS = {
    "attribution": f"""{ROLE} You are the attribution specialist. With your tools, explain why
the alerted metric is where it is: confirm the limit status and utilization, split the
day-over-day VaR change into position and market effects, find the top contributing trades and
risk factors, and check for new trades. Work only from tool results.""",
    "data_quality": f"""{ROLE} You are the data-quality specialist. Decide whether bad market
data drives the alerted metric. Get the data-quality findings for the date (critical: rule check,
the factor is held flat; warning: anomaly model, the factor stays live and can inflate VaR), find
the risk factors that contribute most to the alerted scope, and compare sources for FX factors.
Say clearly whether any finding sits on a contributing factor.""",
    "policy": f"""{ROLE} You are the policy specialist. Use search_policy to find the sections
that govern this alert: status definitions and alert validation, root-cause categories and their
evidence requirements, escalation paths and routing (including data-issue routing and near
misses), and relevant Basel paragraphs. Run at most 3 specific searches, all in one turn. Cite
only sections that search_policy returned, with their exact doc_id and section_id.""",
}
FINISH = (
    "Stop using tools. Report your findings: a short summary and evidence items whose values are"
    " numbers copied exactly from tool results, each with its result_id and unit. State absences"
    " (no new trades, no findings) in the summary, never as evidence values."
)
FINISH_POLICY = FINISH + " Add the citations (doc_id, section_id) of the sections that apply."
WRITER = f"{ROLE} You write the incident report for a risk manager from the specialists' findings."
CRITIC = f"""{ROLE} Review an escalation note written for a risk manager. Is it clear, correctly
prioritized (it leads with the required action), and actionable? Set ok=true if it is. Otherwise
list specific fixes. Judge only the writing; the numbers are checked elsewhere."""


def merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    return a | b


def sticky(a: str, b: str) -> str:
    """needs_human, once set, survives later writes from parallel branches."""
    return a if a == "needs_human" else b


def first(a: str, b: str) -> str:
    """Keep the first reason when parallel branches stop at once."""
    return a or b


class State(TypedDict, total=False):
    case: dict[str, Any]  # incident.json with resolved override paths
    variant: str
    alert: dict[str, Any]  # limit status row of the alert, with its result_id
    plan: list[dict[str, str]]
    task: dict[str, str]  # Send payload for one specialist
    findings: Annotated[dict[str, Any], merge]
    calls: Annotated[list[dict[str, Any]], operator.add]  # {agent, tool, args, result_id}
    retrieved: Annotated[list[list[str]], operator.add]  # [doc_id, section_id]
    report: dict[str, Any]
    critique: dict[str, Any]
    feedback: dict[str, str]  # agent -> critic feedback for its next pass
    loops: int
    status: Annotated[str, sticky]
    reason: Annotated[str, first]
    decision: dict[str, Any]
    note_path: str


OVERRIDES = ("book_override", "market_override")


def case_dict(incident_dir: Path) -> dict[str, Any]:
    """incident.json with resolved override paths, as stored in the graph state."""
    c = Case.load(incident_dir)
    paths = {k: str(getattr(c, k)) if getattr(c, k) else None for k in OVERRIDES}
    ids = {"incident_id": c.incident_id, "as_of_date": c.as_of_date, "run_id": c.run_id}
    return ids | {"alert": c.alert} | paths


def to_case(d: Mapping[str, Any]) -> Case:
    paths = {k: Path(d[k]) if d.get(k) else None for k in OVERRIDES}
    return Case(d["incident_id"], d["as_of_date"], d["run_id"], dict(d["alert"]), **paths)


def budget_of(cfg: RunnableConfig) -> Budget:
    b = cfg.get("configurable", {}).get("budget")
    return b if isinstance(b, Budget) else Budget()


def intro(state: Mapping[str, Any]) -> str:
    c, a = state["case"], state["alert"]
    return (
        f"Incident {c['incident_id']}, as-of date {c['as_of_date']}. Alert on limit "
        f"{a['scope']}/{a['metric']}. Limit status at intake (already fetched):\n"
        f"{wrap('tool:get_limit_status', a)}"
    )


def slim(calls: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[list[str]]]:
    """Calls for the state (without results) and the sections search_policy returned."""
    retrieved = [
        [r["doc_id"], r["section_id"]]
        for c in calls
        if c["tool"] == "search_policy"
        for r in c["result"]["results"]
    ]
    return [{k: c[k] for k in ("agent", "tool", "args", "result_id")} for c in calls], retrieved


def stop(e: Exception) -> dict[str, Any]:
    kind = "schema" if isinstance(e, SchemaFailure) else "budget"
    return {"status": "needs_human", "reason": f"{kind}: {e}"}


def assemble(state: Mapping[str, Any], draft: ReportDraft) -> dict[str, Any]:
    """IncidentReport = intake facts (ID, date, breach) + the generated draft."""
    c, a = state["case"], state["alert"]
    breach = Breach(**{k: a[k] for k in ("scope", "metric", "value", "limit", "utilization")})
    facts = {"incident_id": c["incident_id"], "as_of_date": c["as_of_date"], "breach": breach}
    report = IncidentReport(**facts, **draft.model_dump())
    return report.model_dump(mode="json")


# -- nodes shared with the baseline ---------------------------------------------------------
def intake_node(make_box: MakeBox) -> Callable[..., dict[str, Any]]:
    def intake(state: State, config: RunnableConfig) -> dict[str, Any]:
        c = state["case"]
        budget_of(config).tool()
        args = {"date": c["as_of_date"], "scope": c["alert"]["scope"]}
        out = make_box(c).call("get_limit_status", args)
        row = next(r for r in out["rows"] if r["metric"] == c["alert"]["metric"])
        call = {"agent": "intake", "tool": "get_limit_status", "args": args}
        return {
            "alert": row | {"result_id": out["result_id"]},
            "calls": [call | {"result_id": out["result_id"]}],
            "status": "running",
            "loops": 0,
        }

    return intake


def human_approval(state: State) -> dict[str, Any]:
    """Pause for a human decision (SPEC §10.5). Resume with {decision, edits, token}."""
    answer = interrupt(
        {"incident_id": state["case"]["incident_id"], "report": state["report"]}
        | {"critique": state.get("critique")}
    )
    if answer.get("decision") != "approve":
        return {"decision": {"decision": "reject", "reason": answer.get("reason", "")}}
    report = dict(state["report"])
    if answer.get("edits"):
        report["draft_note"] = answer["edits"]
    return {"decision": {"decision": "approve", "token": answer.get("token", "")}, "report": report}


def after_approval(state: State) -> str:
    return "dispatch" if state["decision"]["decision"] == "approve" else "close"


def dispatch(state: State, config: RunnableConfig) -> dict[str, Any]:
    r, c = state["report"], state["case"]
    thread = str(config.get("configurable", {}).get("thread_id", ""))
    cites = ", ".join(x["section_id"] for x in r["policy_citations"]) or "none"
    b = r["breach"]
    note = (
        f"# Incident {c['incident_id']}: {b['scope']}/{b['metric']} as of {c['as_of_date']}\n\n"
        f"{r['draft_note']}\n\n---\nRoot cause: {r['root_cause']} "
        f"(confidence {r['confidence']:.0%}). Action: {r['recommended_action']}. "
        f"Citations: {cites}. Approved in thread {thread}.\n"
    )
    path = send_escalation_email(
        c["incident_id"],
        state["decision"]["token"],
        thread_id=thread,
        as_of_date=c["as_of_date"],
        note=note,
    )
    return {"status": "dispatched", "note_path": str(path)}


def close(state: State) -> dict[str, Any]:
    return {"status": "closed", "reason": state["decision"].get("reason", "")}


# -- the multi-agent graph -----------------------------------------------------------------
def build_graph(
    llm: BaseChatModel,
    make_box: MakeBox,
    sections: set[tuple[str, str]],
    checkpointer: Any = None,
) -> Any:
    cfg = config()
    allow = cfg["tools"]
    max_loops = int(cfg["critic"]["max_loops"])

    def supervisor(state: State, config: RunnableConfig) -> dict[str, Any]:
        try:
            msgs = [SystemMessage(SUPERVISOR), HumanMessage(intro(state))]
            plan = ask(llm, Plan, msgs, budget_of(config))
        except (BudgetExceeded, SchemaFailure) as e:
            return stop(e)
        tasks: dict[str, str] = {t.agent: t.question for t in plan.tasks}
        default = "Investigate the alert from your angle."
        tasks = tasks or dict.fromkeys(SPECIALISTS, default)
        return {"plan": [{"agent": a, "question": q} for a, q in tasks.items()]}

    def payload(state: State, agent: str, question: str) -> dict[str, Any]:
        keys = ("case", "alert", "findings", "feedback")
        return {k: state.get(k) for k in keys} | {"task": {"agent": agent, "question": question}}

    def fan_out(state: State) -> Any:
        if state.get("status") == "needs_human":
            return END
        return [Send(t["agent"], payload(state, t["agent"], t["question"])) for t in state["plan"]]

    def specialist(agent: str) -> Callable[..., dict[str, Any]]:
        def node(state: State, config: RunnableConfig) -> dict[str, Any]:
            budget = budget_of(config)
            task = f"{intro(state)}\n\nYour task: {state['task']['question']}"
            fb = (state.get("feedback") or {}).get(agent)
            if fb:
                prev = (state.get("findings") or {}).get(agent)
                task += f"\n\nA reviewer rejected your previous findings:\n{fb}"
                task += f"\nPrevious findings:\n{wrap(f'agent:{agent}', prev)}\nFix every point."
            try:
                box = make_box(state["case"])
                turns = POLICY_TURNS if agent == "policy" else MAX_STEPS
                msgs, calls = react(
                    llm, box, allow[agent], PROMPTS[agent], task, budget, agent, turns
                )
                done = FINISH_POLICY if agent == "policy" else FINISH
                findings = ask(llm, Findings, [*msgs, HumanMessage(done)], budget, agent)
            except (BudgetExceeded, SchemaFailure) as e:
                return stop(e)
            slim_calls, retrieved = slim(calls)
            out = {"calls": slim_calls, "retrieved": retrieved}
            return out | {"findings": {agent: findings.model_dump()}}

        return node

    def writer(state: State, config: RunnableConfig) -> dict[str, Any]:
        if state.get("status") == "needs_human":
            return {}
        parts = [intro(state), "Specialist findings:"]
        for agent, f in (state.get("findings") or {}).items():
            parts.append(wrap(f"agent:{agent}", f))
        got = sorted({f"{d} {s}" for d, s in state.get("retrieved", [])})
        parts.append("Sections returned by search_policy (cite only these): " + ", ".join(got))
        fb = (state.get("feedback") or {}).get("writer")
        if fb and state.get("report"):
            parts.append(f"A reviewer rejected your previous report:\n{fb}\nFix every point.")
            parts.append("Previous report:\n" + wrap("agent:writer", state["report"]))
        msgs = [SystemMessage(f"{WRITER}\n\n{SCHEMA_GUIDE}"), HumanMessage("\n\n".join(parts))]
        try:
            draft = ask(llm, ReportDraft, msgs, budget_of(config))
        except (BudgetExceeded, SchemaFailure) as e:
            return stop(e)
        return {"report": assemble(state, draft)}

    def critic(state: State, config: RunnableConfig) -> dict[str, Any]:
        if state.get("status") == "needs_human":
            return {}
        box = make_box(state["case"])
        r = state["report"]
        calls, got = state.get("calls", []), state.get("retrieved", [])
        issues = check_report(r, state["alert"], calls, got, box, sections)
        if not issues:  # check 4 only once the facts hold
            note = wrap("agent:writer", {"draft_note": r["draft_note"]})
            try:
                review = ask(
                    llm, NoteReview, [SystemMessage(CRITIC), HumanMessage(note)], budget_of(config)
                )
            except (BudgetExceeded, SchemaFailure) as e:
                return stop(e)
            if not review.ok:
                issues = [{"check": "note", "agent": "writer", "message": review.feedback}]
        loops = state.get("loops", 0) + bool(issues)
        feedback: dict[str, str] = {}
        for i in issues:
            feedback[i["agent"]] = (feedback.get(i["agent"], "") + f"- {i['message']}\n").strip()
        if issues:  # the writer redoes the report in every case, so it sees every issue
            feedback["writer"] = "\n".join(f"- {i['message']}" for i in issues)
        return {
            "critique": {"passed": not issues, "issues": issues, "loops": loops},
            "feedback": feedback,
            "loops": loops,
        }

    def after_critic(state: State) -> Any:
        if state.get("status") == "needs_human":
            return END
        c = state["critique"]
        if c["passed"] or state["loops"] > max_loops:
            return "human_approval"
        agents = sorted({i["agent"] for i in c["issues"]} - {"writer"})
        if not agents:
            return "writer"
        questions = {t["agent"]: t["question"] for t in state.get("plan", [])}
        default = "Investigate the alert from your angle."
        return [Send(a, payload(state, a, questions.get(a, default))) for a in agents]

    g = StateGraph(State)
    g.add_node("intake", intake_node(make_box))
    g.add_node("supervisor", supervisor)
    for a in SPECIALISTS:
        g.add_node(a, specialist(a))
        g.add_edge(a, "writer")
    g.add_node("writer", writer)
    g.add_node("critic", critic)
    g.add_node("human_approval", human_approval)
    g.add_node("dispatch", dispatch)
    g.add_node("close", close)
    g.add_edge(START, "intake")
    g.add_edge("intake", "supervisor")
    g.add_conditional_edges("supervisor", fan_out, [*SPECIALISTS, END])
    g.add_edge("writer", "critic")
    g.add_conditional_edges("critic", after_critic, [*SPECIALISTS, "writer", "human_approval", END])
    g.add_conditional_edges("human_approval", after_approval, ["dispatch", "close"])
    g.add_edge("dispatch", END)
    g.add_edge("close", END)
    return g.compile(checkpointer=checkpointer)


def resume(decision: str, token: str, edits: str | None = None, reason: str = "") -> Command:
    return Command(resume={"decision": decision, "token": token, "edits": edits, "reason": reason})
