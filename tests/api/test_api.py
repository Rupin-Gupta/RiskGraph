from collections.abc import Iterator
from datetime import date
from types import SimpleNamespace
from typing import Any, cast

import pytest
from fakes import DAY, RUN, SECTIONS, FakeLLM, search
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy.engine import Engine

from riskgraph import cli
from riskgraph.agents import tools
from riskgraph.agents.graph import build_graph, to_case
from riskgraph.agents.tools import Toolbox, approval_token
from riskgraph.api.app import create_app
from riskgraph.db.tables import limit_status, replace_rows, risk_results

IID = "INC-20240315-fx-var_99_1d"
AUTH = {"Authorization": "Bearer t0ken"}
BREACH = {"scope": "fx", "metric": "var_99_1d", "value": 2.2e6, "limit": 2.18e6}
BREACH |= {"utilization": 2.2e6 / 2.18e6, "status": "breach"}


class FakeSES:
    def __init__(self, fail: int = 0) -> None:
        self.fail = fail
        self.sent: list[dict[str, Any]] = []

    def send_email(self, **kw: Any) -> dict[str, str]:
        if self.fail:
            self.fail -= 1
            raise RuntimeError("SES throttled")
        self.sent.append(kw)
        return {"MessageId": "m1"}


@pytest.fixture
def client(env: Engine, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """The API on the SQLite risk run, with the fake LLM and a stubbed daily run (one breach)."""
    monkeypatch.setenv("API_TOKEN", "t0ken")
    monkeypatch.setattr(cli, "daily_run", lambda day, seed=42: {"run_id": RUN, "limits": [BREACH]})

    def make_box(c: Any) -> Toolbox:
        return Toolbox(to_case(c), env, search)

    graph = build_graph(cast(Any, FakeLLM()), make_box, SECTIONS, InMemorySaver())
    with TestClient(create_app(SimpleNamespace(engine=env, graph=lambda v: graph))) as c:
        yield c


def start(client: TestClient) -> dict[str, Any]:
    r = client.post("/runs/daily", json={"date": DAY}, headers=AUTH)
    assert r.status_code == 200, r.text
    return cast(dict[str, Any], r.json())


def test_daily_run_needs_the_token(client: TestClient) -> None:
    assert client.post("/runs/daily", json={"date": DAY}).status_code == 401
    bad = {"Authorization": "Bearer nope"}
    assert client.post("/runs/daily", json={"date": DAY}, headers=bad).status_code == 401
    assert client.get("/health").json() == {"status": "ok"}


def test_breach_is_investigated_then_approved_and_dispatched(client: TestClient) -> None:
    assert start(client)["queued"] == [IID]
    assert [i["status"] for i in client.get("/incidents").json()] == ["awaiting_approval"]
    d = client.get(f"/incidents/{IID}").json()
    assert d["report"]["root_cause"] == "market_move"
    assert d["report"]["evidence"][0]["tool"] == "get_top_contributors"
    assert d["report"]["policy_citations"]

    r = client.post(f"/incidents/{IID}/approve", json={"edits": "Edited note for the desk head."})
    assert r.status_code == 200 and r.json()["status"] == "dispatched"
    note = cli.RUNS / DAY / "escalations" / f"{IID}.md"
    assert "Edited note for the desk head." in note.read_text()
    assert "token" not in r.json()["decision"]

    assert start(client)["queued"] == []  # rerunning the date keeps the incident
    assert client.post(f"/incidents/{IID}/approve", json={}).status_code == 409


def test_reject_needs_a_reason_and_closes(client: TestClient) -> None:
    start(client)
    assert client.post(f"/incidents/{IID}/reject", json={"reason": ""}).status_code == 422
    r = client.post(f"/incidents/{IID}/reject", json={"reason": "duplicate alert"})
    assert r.json()["status"] == "closed"
    assert r.json()["decision"] == {"decision": "reject", "reason": "duplicate alert"}
    assert not (cli.RUNS / DAY / "escalations").exists()
    assert client.get("/incidents/INC-nope").status_code == 404


def test_a_failed_send_is_retried_by_approving_again(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    ses = FakeSES(fail=1)
    monkeypatch.setattr("boto3.client", lambda *a, **kw: ses)
    monkeypatch.setenv("DISPATCH_MODE", "ses")
    monkeypatch.setenv("SES_SENDER", "risk@example.com")
    monkeypatch.setenv("SES_RECIPIENT", "manager@example.com")
    start(client)
    assert client.post(f"/incidents/{IID}/approve", json={}).status_code == 502
    assert client.get(f"/incidents/{IID}").json()["status"] == "dispatch_failed"
    assert client.post(f"/incidents/{IID}/approve", json={}).json()["status"] == "dispatched"
    (mail,) = ses.sent
    assert mail["Destination"] == {"ToAddresses": ["manager@example.com"]}
    assert mail["Message"]["Subject"]["Data"].startswith(f"Incident {IID}: fx/var_99_1d")


def test_ses_dispatch_still_checks_the_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APPROVAL_SECRET", "s")
    monkeypatch.setenv("DISPATCH_MODE", "ses")
    monkeypatch.setattr("boto3.client", lambda *a, **kw: pytest.fail("sent without a token"))
    with pytest.raises(PermissionError):
        tools.send_escalation_email("I1", "forged", thread_id="t", as_of_date=DAY, note="# x")
    monkeypatch.setenv("DISPATCH_MODE", "smtp")
    with pytest.raises(ValueError):
        token = approval_token("t", "I1")
        tools.send_escalation_email("I1", token, thread_id="t", as_of_date=DAY, note="# x")


def test_risk_views_read_the_base_run(client: TestClient, env: Engine) -> None:
    assert client.get("/limits").status_code == 404  # only the incident run (RUN) so far
    replace_rows(
        env,
        DAY,
        {
            limit_status: [BREACH | {"date": date(2024, 3, 15)}],
            risk_results: [
                {"desk": "fx", "metric": "var_99_1d_hs", "value": 2.2e6}
                | {"date": date(2024, 3, 15)}
            ],
        },
    )
    s = client.get("/risk/summary", params={"date": DAY}).json()
    assert s["date"] == DAY and s["dates"] == [DAY]
    assert s["desks"] == {"fx": {"var_99_1d_hs": 2.2e6}}
    assert s["limits"][0]["status"] == "breach"
    assert client.get("/limits").json()["date"] == DAY
    assert client.get("/limits", params={"date": "2020-01-02"}).status_code == 404
