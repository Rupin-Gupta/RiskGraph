"""Wiring for the CLI and the evaluation: database, retriever, checkpointer, LLM, graphs, and
per-run config (budget, thread, Langfuse tags)."""

from __future__ import annotations

import contextlib
import io
import os
from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from sqlalchemy import func, select
from sqlalchemy.engine import Engine

from riskgraph import cli
from riskgraph.agents.baseline import build_baseline
from riskgraph.agents.graph import build_graph, to_case
from riskgraph.agents.llm import Budget, make_llm
from riskgraph.agents.tools import Toolbox
from riskgraph.db.tables import database_url, get_engine, limit_status, metadata
from riskgraph.rag.chunking import known_sections

INCIDENTS = Path("data/synthetic/incidents")
VARIANTS = ("baseline", "multi")


def ensure_run(case: Mapping[str, Any], engine: Engine, seed: int = 42) -> None:
    """Make sure the incident's risk run is in Postgres (incident generation validates runs
    without writing to the database)."""
    t = limit_status.c
    with engine.connect() as conn:
        n = conn.execute(select(func.count()).where(t.run_id == case["run_id"])).scalar()
    if n:
        return
    paths = [Path(case[k]) if case.get(k) else None for k in ("book_override", "market_override")]
    with contextlib.redirect_stdout(io.StringIO()):
        doc = cli.daily_run(case["as_of_date"], paths[0], paths[1], seed, db=True)
    if doc["run_id"] != case["run_id"]:
        raise RuntimeError(f"run_id mismatch: {doc['run_id']} != {case['run_id']}")


def run_config(thread_id: str, budget: Budget, tags: Mapping[str, str]) -> RunnableConfig:
    """Budget and thread for the graph; Langfuse tracing when its keys are set (SPEC §10.10)."""
    callbacks: list[Any] = []
    if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
        from langfuse.langchain import CallbackHandler

        callbacks.append(CallbackHandler())
    meta = {
        "langfuse_session_id": thread_id,
        "langfuse_trace_name": f"investigate:{tags.get('variant', '')}",
        "langfuse_tags": [f"{k}:{v}" for k, v in tags.items()],
    } | dict(tags)
    return {
        "configurable": {"thread_id": thread_id, "budget": budget},
        "callbacks": callbacks,
        "metadata": meta,
        "recursion_limit": 60,
    }


def flush_traces() -> None:
    if os.environ.get("LANGFUSE_PUBLIC_KEY"):
        from langfuse import get_client

        get_client().flush()


class Runtime:
    """Everything a graph needs, opened once. checkpointer: "postgres" (CLI, survives restarts)
    or "memory" (evaluation)."""

    def __init__(self, checkpointer: str = "postgres") -> None:
        self.kind = checkpointer
        self._stack = ExitStack()

    def __enter__(self) -> Runtime:
        from riskgraph.rag.store import Retriever, connect

        self.engine = get_engine()
        metadata.create_all(self.engine)  # tool_results is new in phase 02; idempotent
        client = self._stack.enter_context(connect())
        self.search = Retriever(client)
        self.sections = known_sections()
        self.llm = make_llm()
        if self.kind == "postgres":
            from langgraph.checkpoint.postgres import PostgresSaver

            url = database_url().replace("postgresql+psycopg://", "postgresql://", 1)
            self.saver: Any = self._stack.enter_context(PostgresSaver.from_conn_string(url))
            self.saver.setup()
        else:
            from langgraph.checkpoint.memory import InMemorySaver

            self.saver = InMemorySaver()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stack.close()

    def make_box(self, case: Mapping[str, Any]) -> Toolbox:
        return Toolbox(to_case(case), self.engine, self.search)

    def graph(self, variant: str) -> Any:
        if variant == "multi":
            return build_graph(self.llm, self.make_box, self.sections, self.saver)
        if variant == "baseline":
            return build_baseline(self.llm, self.make_box, self.saver)
        raise ValueError(f"unknown variant {variant!r}; use one of {VARIANTS}")
