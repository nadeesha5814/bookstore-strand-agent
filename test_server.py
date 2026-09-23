"""Tests for the HTTP layer. The agent is stubbed, so no model and no network.

    python -m pytest test_server.py -q
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

import server


class StubAgent:
    """Records the prompts it was given and echoes a canned reply."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def invoke_async(self, prompt: str):
        self.prompts.append(prompt)
        await asyncio.sleep(0.01)  # let a competing request interleave if it can
        return f"reply {len(self.prompts)}"


@pytest.fixture
def client(monkeypatch):
    built: list[StubAgent] = []

    def fake_build_agent(*a, **kw):
        agent = StubAgent()
        built.append(agent)
        return agent

    monkeypatch.setattr(server, "build_agent", fake_build_agent)
    monkeypatch.setattr(server, "store", server.SessionStore(server.MAX_SESSIONS, server.SESSION_TTL_SECONDS))
    with TestClient(server.app) as c:
        c.built = built
        yield c


def post(client, message="hello", session_id="s1", **extra):
    body = {"message": message, "session_id": session_id, **extra}
    return client.post("/chat", json=body)


def test_contract_shape(client):
    r = post(client)
    assert r.status_code == 200
    assert list(r.json()) == ["response"]
    assert isinstance(r.json()["response"], str)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_context_is_optional(client):
    assert post(client).status_code == 200


def test_same_session_reuses_one_agent(client):
    post(client, session_id="a")
    post(client, session_id="a")
    assert len(client.built) == 1
    assert len(client.built[0].prompts) == 2


def test_different_sessions_get_separate_agents(client):
    post(client, session_id="a")
    post(client, session_id="b")
    assert len(client.built) == 2


def test_context_is_prepended_to_the_prompt(client):
    post(client, message="where are my orders?", context={"customer_email": "rafa@example.com"})
    prompt = client.built[0].prompts[0]
    assert "rafa@example.com" in prompt
    assert prompt.endswith("where are my orders?")


def test_empty_context_leaves_the_message_alone(client):
    post(client, message="hi", context={})
    assert client.built[0].prompts[0] == "hi"


def test_context_rendering_is_deterministic():
    a = server.apply_context("m", {"b": 2, "a": 1})
    b = server.apply_context("m", {"a": 1, "b": 2})
    assert a == b  # sorted keys, so caching and tests stay stable


@pytest.mark.parametrize("body", [
    {"message": "hi"},                       # no session_id
    {"session_id": "s"},                     # no message
    {"message": "", "session_id": "s"},      # empty message
    {"message": "hi", "session_id": ""},     # empty session_id
])
def test_invalid_requests_are_rejected(client, body):
    assert client.post("/chat", json=body).status_code == 422


def test_agent_failure_becomes_502(client, monkeypatch):
    async def boom(prompt):
        raise RuntimeError("model exploded")

    post(client, session_id="z")
    monkeypatch.setattr(client.built[0], "invoke_async", boom)
    r = post(client, session_id="z")
    assert r.status_code == 502 and "model exploded" in r.json()["detail"]


def test_provider_misconfiguration_becomes_503(client, monkeypatch):
    def unusable(*a, **kw):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    monkeypatch.setattr(server, "build_agent", unusable)
    r = post(client, session_id="new-one")
    assert r.status_code == 503 and "OPENAI_API_KEY" in r.json()["detail"]


def test_sessions_are_capped(monkeypatch):
    monkeypatch.setattr(server, "build_agent", lambda *a, **kw: StubAgent())
    small = server.SessionStore(max_sessions=2, ttl_seconds=3600)

    async def drive():
        for sid in ("a", "b", "c"):
            await small.get(sid)
        return len(small)

    assert asyncio.run(drive()) == 2


def test_idle_sessions_expire(monkeypatch):
    monkeypatch.setattr(server, "build_agent", lambda *a, **kw: StubAgent())
    store = server.SessionStore(max_sessions=10, ttl_seconds=0)

    async def drive():
        await store.get("a")
        await store.get("b")   # evicts "a" on the way in
        return len(store)

    assert asyncio.run(drive()) == 1


def test_same_session_requests_are_serialised(monkeypatch):
    """Strands raises if one Agent is invoked concurrently, so the lock matters."""
    overlap = {"peak": 0, "live": 0}

    class Tracking(StubAgent):
        async def invoke_async(self, prompt: str):
            overlap["live"] += 1
            overlap["peak"] = max(overlap["peak"], overlap["live"])
            await asyncio.sleep(0.05)
            overlap["live"] -= 1
            return "ok"

    monkeypatch.setattr(server, "build_agent", lambda *a, **kw: Tracking())
    store = server.SessionStore(max_sessions=10, ttl_seconds=3600)

    async def one_call():
        session = await store.get("shared")
        async with session.lock:
            return await session.agent.invoke_async("hi")

    async def drive():
        await asyncio.gather(*(one_call() for _ in range(5)))

    asyncio.run(drive())
    assert overlap["peak"] == 1, f"{overlap['peak']} concurrent invocations of one Agent"


# --------------------------------------------------------------------------
# `context` accepts any JSON value, per the interface contract.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("context", [
    {"customer_email": "rafa@example.com"},   # the useful shape
    "JSON",                                   # a bare string
    "signed in as Rafa",                      # a plain note
    ["a", "b"],                               # an array
    None,                                     # null
    {},                                       # empty object
    "",                                       # empty string
    42,                                       # a number
    True,                                     # a boolean
])
def test_any_json_context_is_accepted(client, context):
    r = client.post("/chat", json={"message": "hi", "session_id": "s", "context": context})
    assert r.status_code == 200, r.text


@pytest.mark.parametrize("empty", [None, {}, "", []])
def test_empty_context_leaves_the_message_untouched(empty):
    assert server.apply_context("hi", empty) == "hi"


def test_double_encoded_context_is_unwrapped():
    """A client that JSON-encodes context twice still gets an object, not a blob."""
    once = server.apply_context("hi", {"customer_email": "rafa@example.com"})
    twice = server.apply_context("hi", '{"customer_email": "rafa@example.com"}')
    assert once == twice


def test_unparseable_string_context_is_kept_as_a_note():
    assert "signed in as Rafa" in server.apply_context("hi", "signed in as Rafa")


def test_the_exact_body_from_the_report_is_accepted(client):
    r = client.post("/chat", json={"message": "How is th", "session_id": "xx1111", "context": "JSON"})
    assert r.status_code == 200
    assert list(r.json()) == ["response"]


# --------------------------------------------------------------------------
# The interface contract is port 8000.
# --------------------------------------------------------------------------

def test_default_port_is_8000(monkeypatch):
    monkeypatch.delenv("PORT", raising=False)
    monkeypatch.delenv("BOOKSTORE_PORT", raising=False)
    monkeypatch.delenv("HOST", raising=False)
    assert server.resolve_bind() == ("0.0.0.0", 8000)


def test_platform_injected_port_is_ignored(monkeypatch):
    """WSO2 injects PORT=8080; the contract says 8000, so 8000 wins."""
    monkeypatch.delenv("BOOKSTORE_PORT", raising=False)
    monkeypatch.setenv("PORT", "8080")
    assert server.resolve_bind()[1] == 8000


def test_deliberate_override_is_honoured(monkeypatch):
    monkeypatch.setenv("BOOKSTORE_PORT", "9999")
    assert server.resolve_bind()[1] == 9999


def test_deliberate_override_beats_injected_port(monkeypatch):
    monkeypatch.setenv("PORT", "8080")
    monkeypatch.setenv("BOOKSTORE_PORT", "9999")
    assert server.resolve_bind()[1] == 9999


def test_non_numeric_override_falls_back_to_8000(monkeypatch):
    monkeypatch.delenv("PORT", raising=False)
    monkeypatch.setenv("BOOKSTORE_PORT", "not-a-port")
    assert server.resolve_bind()[1] == 8000
