# Northwind Books — a Strands conversational agent

A sample customer-service chatbot for an online bookshop, built with the
[Strands Agents SDK](https://strandsagents.com). It answers three kinds of
question against seed data in `data/`:

- **Price & availability** — "how much is Project Hail Mary?", "is Snow Crash in stock?"
- **Order status** — "where is ORD-24801?", "I forgot my order number, my email is …"
- **Recommendations** — "I loved The Martian, what next?"

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # includes the OpenAI extra
```

The agent runs on either **OpenAI** (default) or **Amazon Bedrock**.

```bash
# OpenAI — the default; needs an API key
export OPENAI_API_KEY=sk-...
python chat.py
python chat.py "is Dune in stock?"
python list_openai_models.py        # which models can this key reach?

# Bedrock — needs AWS credentials with model access
python chat.py --provider bedrock
```

Switch the default with `BOOKSTORE_PROVIDER=bedrock`, or pick per-run with
`--provider` / `--model`. Everything else — the six tools, the system prompt,
the conversation handling — is identical either way.

No AWS access yet? `python demo.py` calls every tool directly with no model
involved, so you can see the data and the tool contracts first.

## Configuration

| Variable | Default | Notes |
|---|---|---|
| `BOOKSTORE_PROVIDER` | `openai` | `openai` or `bedrock` |
| `BEDROCK_MODEL_ID` | `global.anthropic.claude-sonnet-4-6` | Any Claude profile your account has enabled |
| `AWS_REGION` | `us-west-2` | Bedrock only; must have model access |
| `OPENAI_API_KEY` | — | Required for the OpenAI provider |
| `OPENAI_MODEL_ID` | `gpt-4o-mini` | Must support tool calling; run `list_openai_models.py` to see your options |

Model availability differs per account on both providers, which is why the
defaults are overridable and both have a way to list what you can actually use.

`build_agent` adjusts sampling parameters to the model: it drops `temperature`
for models that reject it (Claude Opus 5 / Fable 5, and OpenAI's `o*` and
`gpt-5` reasoning models) and keeps it for the rest, so switching models does
not produce a confusing 400.

To see what your account can actually use:

```bash
aws bedrock list-inference-profiles --region us-west-2 --query "inferenceProfileSummaries[?contains(inferenceProfileId,'claude')].inferenceProfileId"
```

A listed profile is not necessarily an enabled one — request access in the
Bedrock console if you get `AccessDeniedException`. If you point
`BEDROCK_MODEL_ID` at Opus 5 or Fable 5, note those models reject sampling
parameters; `build_agent` already drops `temperature` for them.

## Layout

```
data/catalog.json      25 books: price, stock, restock date, genres, tags
data/orders.json       8 orders across 4 customers, every status
bookstore/store.py     domain logic — search, similarity, order views
bookstore/tools.py     the six @tool functions the model can call
bookstore/agent.py     provider selection + system prompt
chat.py                streaming CLI, --provider / --model
demo.py                every tool, no model, no credentials
list_openai_models.py  what your OpenAI key can reach
test_bookstore.py      34 tests, no network
```

## How it works

`bookstore/store.py` holds all the logic and imports nothing from Strands, so it
is testable on its own. `bookstore/tools.py` wraps it in six `@tool` functions —
Strands turns each signature and docstring into the JSON schema the model sees,
which is why the docstrings spell out *when* to use each tool:

```python
@tool
def check_availability(book: str) -> dict[str, Any]:
    """Check only stock and delivery timing for one book — no blurb, no metadata.

    Prefer this over get_book_details when the customer asks purely about
    availability ("is it in stock?", "when can you get it?").

    Args:
        book: Title, catalog id, or ISBN of the book to check.
    """
```

Then `Agent(model=..., tools=BOOKSTORE_TOOLS, system_prompt=...)` and the SDK
runs the loop: the model picks tools, Strands executes them, results go back,
and it keeps going until it has an answer.

Swapping providers only changes the `model` argument. Strands translates the
same tool specs into whichever wire format the provider wants — Bedrock's
`toolSpec` blocks or OpenAI's `type: "function"` definitions — so the tools and
prompt are untouched. That translation is what `_bedrock_model` and `_openai_model` in
`bookstore/agent.py` is hiding.

**Recommendations** are computed, not improvised. `store.similar_to` scores
every other book by Jaccard overlap on genres (0.45) and tags (0.35), a
same-author bonus (0.12), and rating (0.08), and returns a `why` string the
model is told to base its wording on — so the reasons it gives trace back to
real catalog data.

**Conversation state** lives on the `Agent` instance. One `Agent` is one
conversation; `agent.messages` is the history, which is what makes follow-ups
like "how much is the second one?" resolve.

## Tools

| Tool | Answers |
|---|---|
| `search_books` | free text / author / genre / price ceiling / in-stock filter |
| `get_book_details` | everything about one book, including stock |
| `check_availability` | stock and shipping only |
| `recommend_similar_books` | ranked similar titles, each with a reason |
| `get_order_status` | one order by id (`ORD-` prefix optional) |
| `find_orders_by_email` | every order for a customer, newest first |

Lookups accept a title, catalog id, or ISBN. An exact title always wins over a
fragment, and an ambiguous fragment comes back as an error naming the
candidates, which the agent is instructed to relay as a clarifying question.

## Seed data to try

Books: `Dune`, `Dune Messiah` (out of stock, restock 2026-10-14), `Snow Crash`
(out of stock), `1984` (3 left), `Project Hail Mary`, `Atomic Habits`.

Orders: `ORD-24801` shipped with UPS tracking · `ORD-24803` processing, no
tracking yet · `ORD-24804` backordered · `ORD-24806` cancelled · `ORD-24807`
awaiting payment · `ORD-24808` returned and refunded.

Customers: `ada@example.com`, `rafa@example.com`, `yuki@example.com`,
`priya@example.com`.

## Tests

```bash
python -m pytest test_bookstore.py -q
```

Covers the domain logic and also checks the seed data is internally consistent —
every order line references a real book, at the catalog price, and every order
total adds up.

## Scope

The agent is read-only by design. It cannot place, change, or cancel orders or
take payment; the system prompt tells it to refer those to support instead. That
keeps the sample safe to point at real-looking data.
