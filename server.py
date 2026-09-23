#!/usr/bin/env python3
"""HTTP chat interface for the Northwind Books agent.

    POST /chat  {"message": str, "session_id": str, "context": {...}}
             -> {"response": str}

Run it:

    python server.py                      # port 8000
    PORT=9000 python server.py

One `session_id` is one conversation. The Agent instance for that id holds the
history, which is what lets follow-ups like "how much is the second one?" work.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from bookstore.agent import build_agent

logger = logging.getLogger("northwind.server")

# The interface contract is port 8000, so that is what we bind — full stop.
#
# A plain `PORT` is deliberately NOT honoured: hosting platforms (WSO2 Agent
# Manager among them) inject their own `PORT`, which would silently move the
# listener off 8000 and leave callers with a connection refused. To move the
# port on purpose, set BOOKSTORE_PORT, which nothing else writes.
DEFAULT_PORT = 8000
DEFAULT_HOST = "0.0.0.0"
PORT_OVERRIDE_VAR = "BOOKSTORE_PORT"

# Sessions are held in memory, so cap them. Oldest idle session is evicted first.
MAX_SESSIONS = int(os.environ.get("MAX_SESSIONS", "500"))
SESSION_TTL_SECONDS = int(os.environ.get("SESSION_TTL_SECONDS", str(60 * 60)))


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="What the customer said.")
    session_id: str = Field(..., min_length=1, description="Conversation id; reuse it to keep history.")
    # Any JSON value, per the interface contract — an object is the useful shape,
    # but a string, array or null are accepted rather than rejected with a 422.
    context: Any = Field(default=None, description="Optional caller-supplied facts, any JSON value.")


class ChatResponse(BaseModel):
    response: str


@dataclass
class Session:
    agent: Any
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_used: float = field(default_factory=time.monotonic)


class SessionStore:
    """LRU + TTL map of session_id -> Session, safe for concurrent requests."""

    def __init__(self, max_sessions: int, ttl_seconds: int) -> None:
        self._sessions: OrderedDict[str, Session] = OrderedDict()
        self._guard = asyncio.Lock()
        self._max = max_sessions
        self._ttl = ttl_seconds

    async def get(self, session_id: str) -> Session:
        async with self._guard:
            self._evict_expired()
            session = self._sessions.get(session_id)
            if session is None:
                # build_agent raises if the provider is misconfigured; let it
                # surface here rather than half-registering a session.
                session = Session(agent=build_agent())
                self._sessions[session_id] = session
                logger.info("new session %s (%d live)", session_id, len(self._sessions))
            session.last_used = time.monotonic()
            self._sessions.move_to_end(session_id)
            while len(self._sessions) > self._max:
                dropped, _ = self._sessions.popitem(last=False)
                logger.info("evicted session %s (over %d)", dropped, self._max)
            return session

    def _evict_expired(self) -> None:
        cutoff = time.monotonic() - self._ttl
        for sid in [s for s, v in self._sessions.items() if v.last_used < cutoff]:
            del self._sessions[sid]
            logger.info("expired session %s", sid)

    def __len__(self) -> int:
        return len(self._sessions)


store = SessionStore(MAX_SESSIONS, SESSION_TTL_SECONDS)
app = FastAPI(
    title="Northwind Books agent",
    description="Book prices, availability, order status, and recommendations.",
    version="1.0.0",
)


def apply_context(message: str, context: Any) -> str:
    """Prepend caller-supplied context so the model can use it as fact.

    Anything the caller already knows — the signed-in customer's email, their
    locale — belongs here rather than in the message, so the agent does not have
    to ask for it. The agent treats these as account facts, so only send what the
    caller has actually authenticated.

    Accepts any JSON value. An object is the useful shape; a string is passed
    through as a plain note, and a string that itself contains JSON (a
    double-encoded body, which is easy to send by accident) is unwrapped first.
    `None`, `{}`, `[]` and `""` all mean "no context".
    """
    if context is None or (isinstance(context, (dict, list, str)) and not context):
        return message

    if isinstance(context, str):
        try:
            decoded = json.loads(context)
        except (ValueError, TypeError):
            rendered = context.strip()          # a plain note
        else:
            if decoded is None or (isinstance(decoded, (dict, list, str)) and not decoded):
                return message
            rendered = _as_json(decoded)
    else:
        rendered = _as_json(context)

    return f"[session context: {rendered}]\n\n{message}"


def _as_json(value: Any) -> str:
    """Stable rendering — sorted keys so identical context yields identical prompts."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        session = await store.get(request.session_id)
    except (RuntimeError, ValueError) as exc:  # provider misconfigured
        logger.error("cannot build agent: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    prompt = apply_context(request.message, request.context)

    # One Agent cannot be invoked concurrently, and two requests may share a
    # session_id, so serialise per session rather than globally.
    async with session.lock:
        try:
            result = await session.agent.invoke_async(prompt)
        except Exception as exc:
            logger.exception("agent failed for session %s", request.session_id)
            raise HTTPException(status_code=502, detail=f"Agent error: {exc}") from exc

    return ChatResponse(response=str(result).strip())


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "sessions": len(store)}


def resolve_bind() -> tuple[str, int]:
    """Always port 8000, unless BOOKSTORE_PORT deliberately says otherwise."""
    host = os.environ.get("HOST", DEFAULT_HOST)

    injected = os.environ.get("PORT")
    if injected is not None and injected != str(DEFAULT_PORT):
        logger.warning(
            "Ignoring PORT=%s from the environment; binding %d as the interface "
            "contract requires. Set %s to move it on purpose.",
            injected, DEFAULT_PORT, PORT_OVERRIDE_VAR,
        )

    raw = os.environ.get(PORT_OVERRIDE_VAR)
    if raw is None:
        return host, DEFAULT_PORT
    try:
        port = int(raw)
    except ValueError:
        logger.warning("%s=%r is not a number; binding %d", PORT_OVERRIDE_VAR, raw, DEFAULT_PORT)
        return host, DEFAULT_PORT
    if port != DEFAULT_PORT:
        logger.warning("%s=%d moves the listener off the expected port %d", PORT_OVERRIDE_VAR, port, DEFAULT_PORT)
    return host, port


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    host, port = resolve_bind()
    logger.info("Northwind Books agent listening on http://%s:%d  (POST /chat)", host, port)
    uvicorn.run(app, host=host, port=port)
