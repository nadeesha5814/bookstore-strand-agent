#!/usr/bin/env python3
"""Exercise every tool directly, with no model and no AWS credentials.

Useful for checking the seed data and the tool contracts before you point the
agent at Bedrock. Each tool below is a plain Python function as well as a
Strands tool, so it can be called straight.
"""

from __future__ import annotations

import json

from bookstore import tools

CASES = [
    ("Price and stock for a named book",
     lambda: tools.get_book_details("Project Hail Mary")),
    ("Availability only, for a book that is out of stock",
     lambda: tools.check_availability("Snow Crash")),
    ("Search: mysteries under $12 that can ship today",
     lambda: tools.search_books(genre="mystery", max_price=12.0, in_stock_only=True)),
    ("Search: free text",
     lambda: tools.search_books(query="cyberpunk hacking", limit=3)),
    ("Recommendations from a book the customer liked",
     lambda: tools.recommend_similar_books("The Martian", limit=3)),
    ("Recommendations, in-stock only",
     lambda: tools.recommend_similar_books("Dune", limit=3, in_stock_only=True)),
    ("Order status: shipped, with tracking",
     lambda: tools.get_order_status("ORD-24801")),
    ("Order status: backordered",
     lambda: tools.get_order_status("24804")),
    ("All orders for a customer",
     lambda: tools.find_orders_by_email("yuki@example.com")),
    ("Unknown book — the error the model sees",
     lambda: tools.get_book_details("The Silmarillion")),
    ("Ambiguous fragment — the model is told to disambiguate",
     lambda: tools.get_book_details("murder")),
]


def main() -> None:
    for title, call in CASES:
        print(f"\n\033[1m{title}\033[0m")
        print(json.dumps(call(), indent=2, default=str))


if __name__ == "__main__":
    main()
