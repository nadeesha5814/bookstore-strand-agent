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

# Sessions are held in memory, so cap them. Oldest idle session is evicted first.
MAX_SESSIONS = int(os.environ.get("MAX_SESSIONS", "500"))
SESSION_TTL_SECONDS = int(os.environ.get("SESSION_TTL_SECONDS", str(60 * 60)))


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="What the customer said.")
    session_id: str = Field(..., min_length=1, description="Conversation id; reuse it to keep history.")
    context: dict[str, Any] = Field(default_factory=dict, description="Optional caller-supplied facts.")


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


def apply_context(message: str, context: dict[str, Any]) -> str:
    """Prepend caller-supplied context so the model can use it as fact.

    Anything the caller already knows — the signed-in customer's email, their
    locale — belongs here rather than in the message, so the agent does not have
    to ask for it. Keys are passed through verbatim; the agent treats them as
    account facts, so only send what the caller has actually authenticated.
    """
    if not context:
        return message
    rendered = json.dumps(context, ensure_ascii=False, sort_keys=True, default=str)
    return f"[session context: {rendered}]\n\n{message}"


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


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    uvicorn.run(
        app,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
    )
