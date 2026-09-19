"""The weekday trigger Lambda: skip when the instance is stopped, POST with the token when
it runs."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

HANDLER = Path(__file__).parents[1] / "infra" / "lambda" / "trigger_daily" / "handler.py"


def load() -> Any:
    spec = importlib.util.spec_from_file_location("trigger_daily", HANDLER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["trigger_daily"] = module
    spec.loader.exec_module(module)
    return module


class FakeBoto:
    def __init__(self, state: str) -> None:
        self.state = state
        self.calls: list[list[str]] = []

    def client(self, name: str) -> Any:
        boto = self

        class Client:
            def describe_instances(self, InstanceIds: list[str]) -> dict[str, Any]:
                boto.calls.append(InstanceIds)
                return {"Reservations": [{"Instances": [{"State": {"Name": boto.state}}]}]}

            def get_secret_value(self, SecretId: str) -> dict[str, str]:
                return {"SecretString": "t0ken"}

        return Client()


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("INSTANCE_ID", "i-123")
    monkeypatch.setenv("TOKEN_SECRET", "riskgraph/api-token")
    monkeypatch.setenv("SITE_URL", "https://203.0.113.10/")
    return load()


def test_a_stopped_instance_is_skipped_without_an_http_call(
    env: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(env, "boto3", FakeBoto("stopped"))
    monkeypatch.setattr(env.urllib.request, "urlopen", lambda *a, **k: pytest.fail("posted"))
    assert env.handler({}, None) == {"skipped": True, "instance_state": "stopped"}


def test_a_running_instance_is_triggered_with_the_bearer_token(
    env: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent = {}

    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def read(self) -> bytes:
            body = {"date": "2026-09-18", "breaches": [{"scope": "fx"}], "queued": ["INC-1"]}
            return json.dumps(body).encode()

    def urlopen(request: Any, timeout: int = 0) -> Response:
        sent["url"] = request.full_url
        sent["auth"] = request.headers["Authorization"]
        return Response()

    monkeypatch.setattr(env, "boto3", FakeBoto("running"))
    monkeypatch.setattr(env.urllib.request, "urlopen", urlopen)
    assert env.handler({}, None) == {"date": "2026-09-18", "breaches": 1, "queued": ["INC-1"]}
    assert sent == {"url": "https://203.0.113.10/api/runs/daily", "auth": "Bearer t0ken"}
