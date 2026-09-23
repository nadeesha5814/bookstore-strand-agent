#!/usr/bin/env python3
"""Interactive chat with the Northwind Books agent.

    python chat.py                          # interactive
    python chat.py "is Dune in stock?"      # one-shot
    python chat.py --provider openai        # use OpenAI instead of Bedrock

Bedrock needs AWS credentials with model access; OpenAI needs OPENAI_API_KEY.
To see the tools working without any model call at all, run `python demo.py`.
"""

from __future__ import annotations

import asyncio
import sys

from bookstore.agent import build_agent

BANNER = """\
Northwind Books — ask me about prices, availability, your orders, or what to read next.
Try:  "how much is Project Hail Mary?"   "where is order ORD-24801?"
      "something like The Martian"        "mysteries under $12 that ship today"
Ctrl-C or 'quit' to leave.
"""


async def stream_reply(agent, prompt: str) -> None:
    """Stream the assistant's text, announcing each tool call as it starts."""
    announced: set[str] = set()
    async for event in agent.stream_async(prompt):
        tool_use = event.get("current_tool_use")
        if tool_use and (use_id := tool_use.get("toolUseId")) and use_id not in announced:
            if name := tool_use.get("name"):
                announced.add(use_id)
                print(f"\n  [calling {name}]", flush=True)
        if text := event.get("data"):
            print(text, end="", flush=True)
    print("\n")


def parse_args(argv: list[str]) -> tuple[str | None, str]:
    """Pull an optional --provider/--model out of argv; the rest is the prompt."""
    provider, model, rest = None, None, []
    i = 0
    while i < len(argv):
        if argv[i] in ("--provider", "-p") and i + 1 < len(argv):
            provider, i = argv[i + 1], i + 2
        elif argv[i] in ("--model", "-m") and i + 1 < len(argv):
            model, i = argv[i + 1], i + 2
        else:
            rest.append(argv[i])
            i += 1
    return (provider, model), " ".join(rest)


async def main() -> int:
    (provider, model), prompt = parse_args(sys.argv[1:])
    try:
        agent = build_agent(provider=provider, model_id=model)
    except (RuntimeError, ValueError) as exc:
        print(f"[config] {exc}")
        return 1

    label = type(agent.model).__name__.replace("Model", "")
    print(f"[{label}: {agent.model.config['model_id']}]")

    if prompt:
        await stream_reply(agent, prompt)
        return 0

    print(BANNER)
    while True:
        try:
            prompt = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return 0
        if not prompt:
            continue
        if prompt.lower() in {"quit", "exit", "q"}:
            print("Bye.")
            return 0
        print("bot > ", end="", flush=True)
        try:
            await stream_reply(agent, prompt)
        except Exception as exc:  # noqa: BLE001 — keep the REPL alive
            print(f"\n[error] {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
