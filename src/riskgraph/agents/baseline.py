"""Single-agent ReAct baseline (SPEC §10.9): one agent with every tool, the same budget, the same
output schema, and the same intake, approval, and dispatch nodes as the multi-agent graph.

intake -> agent -> human_approval (interrupt) -> dispatch | close
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from riskgraph.agents.graph import (
    ROLE,
    SCHEMA_GUIDE,
    MakeBox,
    State,
    after_approval,
    assemble,
    budget_of,
    close,
    dispatch,
    human_approval,
    intake_node,
    intro,
    slim,
    stop,
)
from riskgraph.agents.llm import BudgetExceeded, SchemaFailure, ask, react
from riskgraph.agents.schemas import ReportDraft
from riskgraph.agents.tools import TOOLS

SYSTEM = f"""{ROLE} You investigate a limit alert end to end. Use the tools to establish the
limit status, why the metric moved (VaR explain, top contributors, new trades), whether bad market
data is involved (data-quality findings, source comparison), and which policy sections apply
(search_policy). Then write the incident report.

{SCHEMA_GUIDE}"""
WRITE = "Stop using tools. Write the incident report now."


def build_baseline(llm: BaseChatModel, make_box: MakeBox, checkpointer: Any = None) -> Any:
    def agent(state: State, config: RunnableConfig) -> dict[str, Any]:
        budget = budget_of(config)
        try:
            box = make_box(state["case"])
            msgs, calls = react(llm, box, list(TOOLS), SYSTEM, intro(state), budget, "baseline")
            draft = ask(llm, ReportDraft, [*msgs, HumanMessage(WRITE)], budget)
        except (BudgetExceeded, SchemaFailure) as e:
            return stop(e)
        slim_calls, retrieved = slim(calls)
        return {"calls": slim_calls, "retrieved": retrieved, "report": assemble(state, draft)}

    def after_agent(state: State) -> str:
        return END if state.get("status") == "needs_human" else "human_approval"

    g = StateGraph(State)
    g.add_node("intake", intake_node(make_box))
    g.add_node("agent", agent)
    g.add_node("human_approval", human_approval)
    g.add_node("dispatch", dispatch)
    g.add_node("close", close)
    g.add_edge(START, "intake")
    g.add_edge("intake", "agent")
    g.add_conditional_edges("agent", after_agent, ["human_approval", END])
    g.add_conditional_edges("human_approval", after_approval, ["dispatch", "close"])
    g.add_edge("dispatch", END)
    g.add_edge("close", END)
    return g.compile(checkpointer=checkpointer)
