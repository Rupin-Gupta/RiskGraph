"""HTTP API (SPEC §10.5, §13): the daily run, risk and limit views, and the approval queue.

Each limit breach from a daily run becomes an incident, registered in the `incidents` table and
investigated in the background by the multi-agent graph. The investigation's state lives in its
LangGraph thread (PostgresSaver), which approve and reject resume.

Run with `uvicorn riskgraph.api.app:app`; docs at /docs.
"""

from __future__ import annotations

import datetime as dt
import hmac
import json
import logging
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import ExitStack, asynccontextmanager
from typing import Annotated, Any
from urllib.parse import quote

import pandas as pd
import typer
from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy import Column, Date, DateTime, String, Table, func, insert, select, text, update
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError

from riskgraph import cli
from riskgraph.agents.graph import resume
from riskgraph.agents.llm import Budget
from riskgraph.agents.runner import flush_traces, run_config
from riskgraph.agents.tools import approval_token
from riskgraph.db.tables import limit_status, metadata, risk_results

log = logging.getLogger("riskgraph.api")
VARIANT = "multi"
VAR_METRICS = ("var_99_1d_hs", "var_99_1d_mc", "var_99_1d_param", "es_975_1d_hs")

# status: queued | investigating | awaiting_approval | needs_human | dispatched | closed | failed
# (investigation error; a rerun of the date retries it) | dispatch_failed (approve retries the send)
incidents = Table(
    "incidents",
    metadata,
    Column("incident_id", String, primary_key=True),
    Column("thread_id", String, nullable=False),
    Column("as_of_date", Date, nullable=False, index=True),
    Column("run_id", String, nullable=False),
    Column("scope", String, nullable=False),
    Column("metric", String, nullable=False),
    Column("status", String, nullable=False),
    Column("reason", String, nullable=False, default=""),
    Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
    Column("updated_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
)


class DailyRun(BaseModel):
    date: dt.date | None = Field(
        None,
        description="Valuation date; omit to refresh market data and run the panel's latest date.",
    )
    seed: int = 42


class Approval(BaseModel):
    edits: str | None = Field(None, description="Markdown replacing the draft note.")


class Rejection(BaseModel):
    reason: str = Field(min_length=1)


# -- helpers --------------------------------------------------------------------------------
def incident_id(day: str, scope: str, metric: str) -> str:
    return f"INC-{day.replace('-', '')}-{scope}-{metric}"


def thread_for(iid: str) -> str:
    return f"{iid}:{VARIANT}:{uuid.uuid4().hex[:8]}"  # the CLI's format: incident:variant:id


def set_status(engine: Engine, iid: str, status: str, reason: str = "", **extra: Any) -> None:
    values = {"status": status, "reason": reason[:500], "updated_at": func.now()} | extra
    with engine.begin() as conn:
        conn.execute(update(incidents).where(incidents.c.incident_id == iid).values(values))


def get_row(engine: Engine, iid: str) -> dict[str, Any]:
    with engine.connect() as conn:
        row = conn.execute(select(incidents).where(incidents.c.incident_id == iid)).first()
    if row is None:
        raise HTTPException(404, f"unknown incident {iid!r}")
    return row._asdict()


def thread_cfg(thread: str, iid: str) -> Any:
    return run_config(thread, Budget(), {"incident_id": iid, "variant": VARIANT})


def outcome(graph: Any, cfg: Any) -> tuple[str, str]:
    """(incident status, reason) from the thread's latest checkpoint."""
    snap = graph.get_state(cfg)
    if "human_approval" in snap.next:
        return "awaiting_approval", ""
    return snap.values.get("status", "failed"), snap.values.get("reason", "")


def base_dates(engine: Engine) -> list[str]:
    """Dates with a base run (run_id == date; incident runs carry a +hash suffix)."""
    t = limit_status.c
    q = select(t.run_id).where(~t.run_id.contains("+")).distinct().order_by(t.run_id)
    with engine.connect() as conn:
        return list(conn.execute(q).scalars())


def pick_date(engine: Engine, day: dt.date | None) -> str:
    dates = base_dates(engine)
    if not dates:
        raise HTTPException(404, "no risk run yet: POST /runs/daily first")
    d = day.isoformat() if day else dates[-1]
    if d not in dates:
        raise HTTPException(404, f"no risk run for {d}")
    return d


def limit_rows(engine: Engine, run_id: str) -> list[dict[str, Any]]:
    t = limit_status.c
    q = select(t.scope, t.metric, t.value, t.limit, t.utilization, t.status)
    with engine.connect() as conn:
        rows = conn.execute(q.where(t.run_id == run_id).order_by(t.scope, t.metric))
        return [r._asdict() for r in rows]


def trace_url(thread: str) -> str | None:
    pid = os.environ.get("LANGFUSE_PROJECT_ID")
    host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com").rstrip("/")
    return f"{host}/project/{pid}/sessions/{quote(thread, safe='')}" if pid else None


def _engine(request: Request) -> Engine:
    return request.app.state.engine  # type: ignore[no-any-return]


def _graph(request: Request) -> Any:
    return request.app.state.graph


Eng = Annotated[Engine, Depends(_engine)]
Graph = Annotated[Any, Depends(_graph)]
Day = Annotated[dt.date | None, Query(alias="date", description="Base-run date; default latest.")]


def require_token(authorization: str = Header("")) -> None:
    expected = os.environ.get("API_TOKEN")
    if not expected:
        raise HTTPException(503, "API_TOKEN is not set")
    if not hmac.compare_digest(authorization.encode(), f"Bearer {expected}".encode()):
        raise HTTPException(401, "invalid token")


# -- the app ----------------------------------------------------------------------------------
def create_app(runtime: Any = None) -> FastAPI:
    """runtime: anything with .engine and .graph(variant); None opens the Postgres Runtime."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        with ExitStack() as stack:
            rt = runtime
            if rt is None:
                from riskgraph.agents.runner import Runtime

                load_dotenv()
                rt = stack.enter_context(Runtime("postgres"))
            app.state.engine, app.state.graph = rt.engine, rt.graph(VARIANT)
            incidents.create(rt.engine, checkfirst=True)
            with rt.engine.begin() as conn:  # a restart kills background investigations
                stuck = incidents.c.status.in_(["queued", "investigating"])
                conn.execute(
                    update(incidents)
                    .where(stuck)
                    .values(status="failed", reason="interrupted by an API restart")
                )
            yield

    app = FastAPI(
        title="RiskGraph API",
        description="Daily market-risk run, limit monitoring, and the incident approval queue.",
        lifespan=lifespan,
    )

    def investigate(eng: Engine, g: Any, ids: list[str]) -> None:
        """Background task: run each queued incident until it pauses for approval or stops."""
        for iid in ids:
            row = get_row(eng, iid)
            set_status(eng, iid, "investigating")
            case = {"incident_id": iid, "as_of_date": row["as_of_date"].isoformat()}
            case |= {"run_id": row["run_id"], "book_override": None, "market_override": None}
            case |= {"alert": {"scope": row["scope"], "metric": row["metric"]}}
            cfg = thread_cfg(row["thread_id"], iid)
            try:
                g.invoke({"case": case, "variant": VARIANT}, cfg)
                status, reason = outcome(g, cfg)
            except Exception as e:  # recorded on the incident; the next run can retry it
                log.exception("investigation_failed incident=%s", iid)
                status, reason = "failed", f"{type(e).__name__}: {e}"
            set_status(eng, iid, status, reason)
            log.info("investigation incident=%s status=%s", iid, status)
        flush_traces()

    @app.get("/health")
    def health(eng: Eng) -> dict[str, str]:
        with eng.connect() as conn:
            conn.execute(text("select 1"))
        return {"status": "ok"}

    @app.post("/runs/daily", dependencies=[Depends(require_token)])
    def run_daily(body: DailyRun, tasks: BackgroundTasks, eng: Eng, g: Graph) -> dict[str, Any]:
        """Market data controls, then the risk run, then an investigation per limit breach.
        Investigations run in the background; rerunning a date skips incidents it already has
        (failed ones are retried)."""
        day = body.date.isoformat() if body.date else None
        if day is None:
            try:
                cli.download(refresh=True, seed=body.seed)
            except typer.Exit as e:
                raise HTTPException(502, "market data refresh failed") from e
            day = f"{pd.read_parquet(cli.PANEL).index.max():%Y-%m-%d}"
        try:
            doc = cli.daily_run(day, seed=body.seed)
        except (KeyError, ValueError) as e:
            raise HTTPException(422, f"risk run failed for {day}: {e}") from e
        breaches = [r for r in doc["limits"] if r["status"] == "breach"]
        queued = []
        for r in breaches:
            iid = incident_id(day, r["scope"], r["metric"])
            row = {"incident_id": iid, "thread_id": thread_for(iid), "run_id": doc["run_id"]}
            row |= {"as_of_date": dt.date.fromisoformat(day), "scope": r["scope"]}
            row |= {"metric": r["metric"], "status": "queued", "reason": ""}
            try:
                with eng.begin() as conn:
                    conn.execute(insert(incidents).values(row))
            except IntegrityError:
                if get_row(eng, iid)["status"] != "failed":
                    continue
                set_status(eng, iid, "queued", thread_id=row["thread_id"])
            queued.append(iid)
        if queued:
            tasks.add_task(investigate, eng, g, queued)
        log.info("daily_run date=%s breaches=%d queued=%d", day, len(breaches), len(queued))
        return {"date": day, "run_id": doc["run_id"], "breaches": breaches, "queued": queued}

    @app.get("/risk/summary")
    def risk_summary(eng: Eng, day: Day = None) -> dict[str, Any]:
        """Desk VaR/ES and limit utilization for a base run (default: the latest), the
        available dates, and the VaR backtest over the latest 250 days."""
        dates = base_dates(eng)
        d = pick_date(eng, day)
        t = risk_results.c
        q = select(t.desk, t.metric, t.value).where(t.run_id == d, t.metric.in_(VAR_METRICS))
        desks: dict[str, dict[str, float]] = {}
        with eng.connect() as conn:
            for desk, metric, value in conn.execute(q):
                desks.setdefault(desk, {})[metric] = value
        bt = cli.METRICS / "backtest.json"
        backtest = json.loads(bt.read_text())["window"] if bt.exists() else None
        doc = {"date": d, "dates": dates, "desks": desks, "limits": limit_rows(eng, d)}
        return doc | {"backtest": backtest}

    @app.get("/limits")
    def limits(eng: Eng, day: Day = None) -> dict[str, Any]:
        """Limit status rows (value, limit, utilization, status) for a base run."""
        d = pick_date(eng, day)
        return {"date": d, "rows": limit_rows(eng, d)}

    @app.get("/incidents")
    def list_incidents(eng: Eng) -> list[dict[str, Any]]:
        """Every incident, newest first."""
        q = select(incidents).order_by(incidents.c.created_at.desc(), incidents.c.incident_id)
        with eng.connect() as conn:
            return [r._asdict() for r in conn.execute(q)]

    @app.get("/incidents/{iid}")
    def incident(iid: str, eng: Eng, g: Graph) -> Any:
        """The incident with its report, evidence (with the tool behind each value), policy
        citations, critic result, decision, and Langfuse trace link."""
        row = get_row(eng, iid)
        snap = g.get_state(thread_cfg(row["thread_id"], iid))
        v = snap.values
        report = v.get("report")
        if report:
            tools = {c["result_id"]: c["tool"] for c in v.get("calls", [])}
            ev = [e | {"tool": tools.get(e["result_id"])} for e in report["evidence"]]
            report = report | {"evidence": ev}
        decision = {k: x for k, x in (v.get("decision") or {}).items() if k != "token"}
        return row | {
            "report": report,
            "critique": v.get("critique"),
            "decision": decision or None,
            "note_path": v.get("note_path"),
            "trace_url": trace_url(row["thread_id"]),
        }

    def decide(eng: Engine, g: Any, iid: str, command: Any) -> Any:
        row = get_row(eng, iid)
        cfg = thread_cfg(row["thread_id"], iid)
        nxt = g.get_state(cfg).next
        if nxt == ("dispatch",):
            command = None  # approved, but the send failed: retry it from the checkpoint
        elif "human_approval" not in nxt:
            raise HTTPException(409, f"incident {iid} is not waiting for approval")
        try:
            g.invoke(command, cfg)
        except Exception as e:  # e.g. SES rejects the send; the thread stays at dispatch
            log.exception("dispatch_failed incident=%s", iid)
            set_status(eng, iid, "dispatch_failed", f"{type(e).__name__}: {e}")
            raise HTTPException(502, f"dispatch failed: {type(e).__name__}") from e
        set_status(eng, iid, *outcome(g, cfg))
        flush_traces()
        return incident(iid, eng, g)

    @app.post("/incidents/{iid}/approve")
    def approve(iid: str, body: Approval, eng: Eng, g: Graph) -> Any:
        """Approve the note (optionally edited): issues the approval token and dispatches."""
        thread = get_row(eng, iid)["thread_id"]
        cmd = resume("approve", approval_token(thread, iid), body.edits or None)
        return decide(eng, g, iid, cmd)

    @app.post("/incidents/{iid}/reject")
    def reject(iid: str, body: Rejection, eng: Eng, g: Graph) -> Any:
        """Reject the note: closes the incident with the reason."""
        return decide(eng, g, iid, resume("reject", "", reason=body.reason))

    return app


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
app = create_app()
