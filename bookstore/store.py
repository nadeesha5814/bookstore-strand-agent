"""Domain layer for the bookstore.

Loads the seed JSON and answers questions about it. Deliberately free of any
Strands or model imports so it can be exercised straight from a REPL or a test.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Weights for the similarity score. They sum to 1.0.
_W_GENRE = 0.45
_W_TAG = 0.35
_W_AUTHOR = 0.12
_W_RATING = 0.08


class BookNotFound(LookupError):
    """Raised when a title/id/ISBN doesn't resolve to a catalog entry."""


class OrderNotFound(LookupError):
    """Raised when an order id isn't in the order book."""


@dataclass(frozen=True)
class Catalog:
    store: dict[str, Any]
    books: list[dict[str, Any]]

    def by_id(self, book_id: str) -> dict[str, Any] | None:
        key = book_id.strip().upper()
        return next((b for b in self.books if b["id"].upper() == key), None)


@lru_cache(maxsize=1)
def catalog() -> Catalog:
    raw = json.loads((DATA_DIR / "catalog.json").read_text(encoding="utf-8"))
    return Catalog(store=raw["store"], books=raw["books"])


@lru_cache(maxsize=1)
def _orders_file() -> dict[str, Any]:
    return json.loads((DATA_DIR / "orders.json").read_text(encoding="utf-8"))


def orders() -> list[dict[str, Any]]:
    return _orders_file()["orders"]


def customers() -> list[dict[str, Any]]:
    return _orders_file()["customers"]


# --------------------------------------------------------------------------
# Lookup
# --------------------------------------------------------------------------

# Words too common to carry any signal in a bookshop query.
_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "have", "has", "was",
    "are", "any", "all", "about", "into", "like", "under", "over", "something",
    "anything", "book", "books", "please", "want", "need", "looking", "give",
    "recommend", "show", "find", "got", "you", "your", "can", "could", "would",
}


def _norm(text: str) -> str:
    return " ".join(text.lower().replace("&", "and").split())


def _tokens(text: str) -> set[str]:
    """Split on anything that is not a letter or digit."""
    out, current = set(), []
    for char in text:
        if char.isalnum():
            current.append(char)
        elif current:
            out.add("".join(current))
            current = []
    if current:
        out.add("".join(current))
    return out


def _stem(word: str) -> str:
    """Crude suffix stripping, enough to tie "mysteries" to "mystery"."""
    for suffix, replacement in (("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)] + replacement
    return word


def _hits(term: str, tokens: set[str]) -> bool:
    """Match a whole word, its stem, or — for longer terms — a word it prefixes.

    The prefix rule catches "hack"/"hacking" without letting a three-letter
    fragment match half the catalog.
    """
    if term in tokens:
        return True
    stem = _stem(term)
    if any(stem == _stem(tok) for tok in tokens):
        return True
    return len(term) >= 4 and any(tok.startswith(term) for tok in tokens)


def resolve_book(reference: str) -> dict[str, Any]:
    """Resolve an id, ISBN, exact title, or a distinctive fragment of a title.

    Raises BookNotFound when nothing matches, or when a fragment is ambiguous
    (the message then lists the candidates so the caller can disambiguate).
    """
    ref = reference.strip()
    if not ref:
        raise BookNotFound("No book reference was given.")

    books = catalog().books
    ref_norm = _norm(ref)
    ref_digits = "".join(c for c in ref if c.isdigit())

    for book in books:
        if book["id"].upper() == ref.upper():
            return book
        if ref_digits and book["isbn"] == ref_digits:
            return book
        if _norm(book["title"]) == ref_norm:
            return book

    partial = [b for b in books if ref_norm in _norm(b["title"])]
    if len(partial) == 1:
        return partial[0]
    if len(partial) > 1:
        names = ", ".join(f"{b['title']} ({b['id']})" for b in partial)
        raise BookNotFound(f"'{reference}' matches more than one book: {names}")

    author_match = [b for b in books if ref_norm in _norm(b["author"])]
    if len(author_match) == 1:
        return author_match[0]

    raise BookNotFound(f"No book in the catalog matches '{reference}'.")


def resolve_order(order_id: str) -> dict[str, Any]:
    key = order_id.strip().upper()
    for order in orders():
        if order["order_id"].upper() == key:
            return order
    raise OrderNotFound(f"No order with id '{order_id}'.")


# --------------------------------------------------------------------------
# Availability & pricing
# --------------------------------------------------------------------------

def availability(book: dict[str, Any]) -> dict[str, Any]:
    """Turn a raw stock count into something a customer can act on."""
    stock = int(book["stock"])
    if stock <= 0:
        state, message = "out_of_stock", "Out of stock."
        if book.get("restock_date"):
            message = f"Out of stock; expected back on {book['restock_date']}."
    elif stock <= 10:
        state = "low_stock"
        message = f"Only {stock} left in stock."
    else:
        state = "in_stock"
        message = f"In stock ({stock} available)."

    ships_in = None
    if stock > 0:
        ships_in = f"{catalog().store['default_shipping_days']} days"

    return {
        "state": state,
        "units_in_stock": stock,
        "message": message,
        "restock_date": book.get("restock_date"),
        "estimated_shipping": ships_in,
    }


def summarize(book: dict[str, Any], *, full: bool = False) -> dict[str, Any]:
    """The shape every tool hands back to the model for a single book."""
    out = {
        "id": book["id"],
        "title": book["title"],
        "author": book["author"],
        "price": f"{book['price']:.2f} {catalog().store['currency']}",
        "availability": availability(book),
        "rating": book["rating"],
        "genres": book["genres"],
    }
    if full:
        out |= {
            "isbn": book["isbn"],
            "format": book["format"],
            "pages": book["pages"],
            "published_year": book["published_year"],
            "tags": book["tags"],
            "description": book["description"],
        }
    return out


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------

def search(
    query: str | None = None,
    author: str | None = None,
    genre: str | None = None,
    max_price: float | None = None,
    in_stock_only: bool = False,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Score every book against the supplied filters and return the best few."""
    results: list[tuple[float, dict[str, Any]]] = []

    for book in catalog().books:
        if author and _norm(author) not in _norm(book["author"]):
            continue
        if genre and not any(_norm(genre) in _norm(g) for g in book["genres"]):
            continue
        if max_price is not None and book["price"] > max_price:
            continue
        if in_stock_only and book["stock"] <= 0:
            continue

        score = 0.0
        if query:
            q = _norm(query)
            terms = [t for t in q.split() if len(t) > 2 and t not in _STOPWORDS]
            title = _norm(book["title"])
            fields = [
                (3.0, title),
                (2.0, _norm(book["author"])),
                (1.5, " ".join(_norm(g) for g in book["genres"])),
                (1.0, " ".join(_norm(t) for t in book["tags"])),
                (0.5, _norm(book["description"])),
            ]
            if q in title:
                score += 6.0
            for weight, text in fields:
                tokens = _tokens(text)
                score += weight * sum(1 for t in terms if _hits(t, tokens))
            if score == 0.0:
                continue

        # Nudge well-rated, purchasable books up when scores are otherwise level.
        score += book["rating"] / 10.0
        if book["stock"] > 0:
            score += 0.2
        results.append((score, book))

    results.sort(key=lambda pair: (-pair[0], pair[1]["title"]))
    return [summarize(b) for _, b in results[: max(1, limit)]]


# --------------------------------------------------------------------------
# Recommendations
# --------------------------------------------------------------------------

def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = {_norm(x) for x in a}, {_norm(x) for x in b}
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _explain(seed: dict[str, Any], other: dict[str, Any]) -> str:
    reasons = []
    if seed["author"] == other["author"]:
        reasons.append(f"also by {other['author']}")
    shared_genres = [g for g in other["genres"] if g in seed["genres"]]
    if shared_genres:
        reasons.append("same " + " / ".join(shared_genres))
    shared_tags = [t for t in other["tags"] if t in seed["tags"]]
    if shared_tags:
        reasons.append("shares " + ", ".join(shared_tags[:3]))
    return "; ".join(reasons) or "generally well rated in a nearby genre"


def similar_to(reference: str, limit: int = 3, in_stock_only: bool = False) -> dict[str, Any]:
    """Rank the catalog by similarity to one book the customer already likes."""
    seed = resolve_book(reference)
    scored: list[tuple[float, dict[str, Any]]] = []

    for book in catalog().books:
        if book["id"] == seed["id"]:
            continue
        if in_stock_only and book["stock"] <= 0:
            continue
        score = (
            _W_GENRE * _jaccard(seed["genres"], book["genres"])
            + _W_TAG * _jaccard(seed["tags"], book["tags"])
            + _W_AUTHOR * (1.0 if book["author"] == seed["author"] else 0.0)
            + _W_RATING * (book["rating"] / 5.0)
        )
        if score > 0.0:
            scored.append((score, book))

    scored.sort(key=lambda pair: (-pair[0], pair[1]["title"]))

    picks = []
    for score, book in scored[: max(1, limit)]:
        entry = summarize(book)
        entry["match_score"] = round(score, 3)
        entry["why"] = _explain(seed, book)
        picks.append(entry)

    return {"based_on": {"id": seed["id"], "title": seed["title"], "author": seed["author"]},
            "recommendations": picks}


# --------------------------------------------------------------------------
# Orders
# --------------------------------------------------------------------------

_STATUS_NEXT_STEP = {
    "pending_payment": "Nothing to do yet — we release it the moment the card clears.",
    "processing": "No tracking number yet; you'll get one by email when it leaves the warehouse.",
    "backordered": "You can split the order to ship the in-stock items now.",
    "shipped": "Track it with the carrier reference below.",
    "delivered": "Delivered. Returns are open for 30 days from delivery.",
    "cancelled": "Refunded in full. Nothing further is owed.",
    "returned": "Return received and refunded.",
}


def order_view(order: dict[str, Any]) -> dict[str, Any]:
    view = {
        "order_id": order["order_id"],
        "status": order["status"],
        "detail": order["status_detail"],
        "next_step": _STATUS_NEXT_STEP.get(order["status"], ""),
        "placed_at": order["placed_at"],
        "estimated_delivery": order["estimated_delivery"],
        "delivered_at": order["delivered_at"],
        "shipping_city": order["shipping_city"],
        "items": [
            {"title": i["title"], "quantity": i["quantity"],
             "unit_price": f"{i['unit_price']:.2f}"} for i in order["items"]
        ],
        "total": f"{order['total']:.2f} {catalog().store['currency']}",
    }
    if order.get("tracking_number"):
        view["tracking"] = {"carrier": order["carrier"], "number": order["tracking_number"]}

    eta, today = order.get("estimated_delivery"), date.today().isoformat()
    if eta and order["status"] in {"processing", "shipped", "backordered", "pending_payment"}:
        view["is_late"] = eta < today
    return view


def orders_for(email: str) -> list[dict[str, Any]]:
    key = email.strip().lower()
    matched = [o for o in orders() if o["customer_email"].lower() == key]
    matched.sort(key=lambda o: o["placed_at"], reverse=True)
    return [order_view(o) for o in matched]
