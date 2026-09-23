"""Assembles the Northwind Books agent.

The agent works the same whichever model backs it — Strands normalises the
provider differences, so `tools` and `system_prompt` never change. Only
`_build_model` cares which one you picked.
"""

from __future__ import annotations

import os

from strands import Agent

from .tools import BOOKSTORE_TOOLS

# Strands' own default is a Bedrock-hosted Claude Sonnet. Override either of
# these with BEDROCK_MODEL_ID / OPENAI_MODEL_ID.
DEFAULT_BEDROCK_MODEL_ID = "global.anthropic.claude-sonnet-4-6"
DEFAULT_OPENAI_MODEL_ID = "gpt-4o-mini"

DEFAULT_PROVIDER = "openai"

# The newest Claude models (Opus 5, Fable 5) reject sampling parameters outright,
# so `temperature` has to be left off for those.
_NO_SAMPLING = ("claude-opus-5", "claude-fable-5")

# OpenAI's reasoning models reject temperature the same way. gpt-4o and the
# gpt-4.x family accept it.
_OPENAI_NO_SAMPLING = ("o1", "o3", "o4", "gpt-5")

SYSTEM_PROMPT = """\
You are the assistant for Northwind Books, an online bookshop. You help customers
with three things: book prices and availability, the status of their orders, and
recommendations.

How to work:
- Answer from the tools, never from memory. If a tool says a book is out of stock
  or an order is late, say exactly that — do not soften it or invent a date.
- If a tool reports it found nothing, tell the customer plainly and offer the
  nearest thing you can: a similar title, or a different search.
- Order lookups need an order id or the account email. Ask for one; never guess
  an id or an address, and never read out an order you were not asked about.
- Prices are in USD. Quote them as the tool returns them.
- get_book_details already includes stock and shipping, so never follow it with
  check_availability for the same book. One lookup per book is enough.
- When you recommend books, give a one-line reason for each that draws on the
  `why` field, and include the price and whether it can ship now.

Style: warm and brief. A couple of sentences, or a short list when you are
showing several books. No restating the question back. Write plain prose — this
is a terminal, so no markdown, asterisks, or headings.

You can look things up and advise, but you cannot place, change, or cancel
orders, or take payment. If a customer asks for that, say so and point them to
support@northwindbooks.example.
"""


def _bedrock_model(model_id: str | None, region: str | None):
    from strands.models import BedrockModel

    resolved = model_id or os.environ.get("BEDROCK_MODEL_ID", DEFAULT_BEDROCK_MODEL_ID)
    settings: dict = {"max_tokens": 2048}
    if not any(marker in resolved for marker in _NO_SAMPLING):
        settings["temperature"] = 0.3

    return BedrockModel(
        model_id=resolved,
        region_name=region or os.environ.get("AWS_REGION", "us-west-2"),
        **settings,
    )


def _openai_model(model_id: str | None):
    try:
        from strands.models.openai import OpenAIModel
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise RuntimeError(
            "The OpenAI provider needs its extra: pip install 'strands-agents[openai]'"
        ) from exc

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export your key, then rerun:\n"
            "    export OPENAI_API_KEY=sk-...\n"
            "(Use `python list_openai_models.py` to see which models that key can reach.)"
        )

    resolved = model_id or os.environ.get("OPENAI_MODEL_ID", DEFAULT_OPENAI_MODEL_ID)
    params: dict = {"max_completion_tokens": 2048}
    if not resolved.startswith(_OPENAI_NO_SAMPLING):
        params["temperature"] = 0.3

    return OpenAIModel(
        client_args={"api_key": api_key},
        model_id=resolved,
        params=params,
    )


def build_agent(
    provider: str | None = None,
    model_id: str | None = None,
    region: str | None = None,
) -> Agent:
    """Build the bookstore agent against Bedrock or OpenAI.

    Args:
        provider: "openai" or "bedrock". Defaults to $BOOKSTORE_PROVIDER, else openai.
        model_id: Model id for that provider. Defaults to the provider's env var.
        region: AWS region, Bedrock only. Defaults to $AWS_REGION, else us-west-2.
    """
    chosen = (provider or os.environ.get("BOOKSTORE_PROVIDER", DEFAULT_PROVIDER)).lower()

    if chosen == "bedrock":
        model = _bedrock_model(model_id, region)
    elif chosen == "openai":
        model = _openai_model(model_id)
    else:
        raise ValueError(f"Unknown provider '{chosen}'. Use 'bedrock' or 'openai'.")

    return Agent(
        model=model,
        tools=BOOKSTORE_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        # The caller renders the stream; without this the SDK also prints it.
        callback_handler=None,
        name="northwind-books",
        description="Conversational assistant for the Northwind Books catalog and order book.",
    )
