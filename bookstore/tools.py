"""The tools the agent is allowed to call.

Each one is a thin, well-described wrapper over `bookstore.store`. The docstring
and type hints are what the model actually sees, so they carry the guidance
about *when* to reach for each tool.
"""

from __future__ import annotations

from typing import Any, Optional

from strands import tool

from . import store


@tool
def search_books(
    query: Optional[str] = None,
    author: Optional[str] = None,
    genre: Optional[str] = None,
    max_price: Optional[float] = None,
    in_stock_only: bool = False,
    limit: int = 5,
) -> dict[str, Any]:
    """Search the catalog by free text, author, genre, and/or price ceiling.

    Use this when the customer describes what they want rather than naming one
    exact book ("something like a space opera", "anything by Agatha Christie",
    "a mystery under $12"). Every result already carries its price and stock
    level, so you usually do not need a follow-up call.

    Args:
        query: Free text matched against title, author, genre, tags, and blurb.
        author: Restrict to authors whose name contains this text.
        genre: Restrict to a genre, e.g. "fantasy", "mystery", "non-fiction".
        max_price: Only return books at or below this price, in USD.
        in_stock_only: Drop anything that is currently out of stock.
        limit: Maximum number of books to return (default 5).
    """
    results = store.search(
        query=query, author=author, genre=genre, max_price=max_price,
        in_stock_only=in_stock_only, limit=limit,
    )
    return {"count": len(results), "results": results}


@tool
def get_book_details(book: str) -> dict[str, Any]:
    """Look up one specific book's price, stock, and full details.

    Use this when the customer names a book. Accepts a title, a catalog id such
    as "BK-1003", or an ISBN. If the reference is ambiguous the error names the
    candidates — relay those to the customer and ask which they meant.

    Args:
        book: Title, catalog id, or ISBN of the book to look up.
    """
    try:
        found = store.resolve_book(book)
    except store.BookNotFound as exc:
        return {"found": False, "error": str(exc)}
    return {"found": True, "book": store.summarize(found, full=True)}


@tool
def check_availability(book: str) -> dict[str, Any]:
    """Check only stock and delivery timing for one book — no blurb, no metadata.

    Prefer this over get_book_details when the customer asks purely about
    availability ("is it in stock?", "when can you get it?", "how fast can it
    ship?").

    Args:
        book: Title, catalog id, or ISBN of the book to check.
    """
    try:
        found = store.resolve_book(book)
    except store.BookNotFound as exc:
        return {"found": False, "error": str(exc)}
    return {
        "found": True,
        "id": found["id"],
        "title": found["title"],
        "price": f"{found['price']:.2f} USD",
        **store.availability(found),
    }


@tool
def recommend_similar_books(
    book: str,
    limit: int = 3,
    in_stock_only: bool = False,
) -> dict[str, Any]:
    """Recommend books similar to one the customer already likes.

    Similarity is computed from shared genres and themes, with a bonus for the
    same author. Each recommendation comes back with a `why` field explaining
    the match — use that wording when you present the picks, and mention the
    price and stock state alongside.

    Args:
        book: The book to base recommendations on (title, catalog id, or ISBN).
        limit: How many recommendations to return (default 3).
        in_stock_only: Only recommend books that can ship today.
    """
    try:
        return {"found": True, **store.similar_to(book, limit=limit, in_stock_only=in_stock_only)}
    except store.BookNotFound as exc:
        return {"found": False, "error": str(exc)}


@tool
def get_order_status(order_id: str) -> dict[str, Any]:
    """Look up one order by its id, e.g. "ORD-24801".

    Returns the status, what it means, the next step for the customer, the
    items, the total, and a tracking reference once the order has shipped.

    Args:
        order_id: The order identifier, with or without the "ORD-" prefix.
    """
    key = order_id.strip().upper()
    if key and not key.startswith("ORD-") and key.replace("-", "").isdigit():
        key = f"ORD-{key.lstrip('-')}"
    try:
        found = store.resolve_order(key)
    except store.OrderNotFound as exc:
        return {"found": False, "error": str(exc)}
    return {"found": True, "order": store.order_view(found)}


@tool
def find_orders_by_email(email: str) -> dict[str, Any]:
    """List every order placed by one customer, newest first.

    Use this when the customer cannot remember their order id. Ask for the email
    on the account first — do not guess one.

    Args:
        email: The email address on the customer's account.
    """
    matched = store.orders_for(email)
    if not matched:
        return {"count": 0, "orders": [],
                "note": f"No orders found for {email}. Check the address, or ask whether a different one was used at checkout."}
    return {"count": len(matched), "orders": matched}


BOOKSTORE_TOOLS = [
    search_books,
    get_book_details,
    check_availability,
    recommend_similar_books,
    get_order_status,
    find_orders_by_email,
]
