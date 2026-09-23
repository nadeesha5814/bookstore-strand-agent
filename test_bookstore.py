"""Tests for the domain layer and tool contracts. No model, no network.

    python -m pytest test_bookstore.py -q
"""

from __future__ import annotations

import pytest

from bookstore import store, tools


def test_catalog_and_orders_load():
    assert len(store.catalog().books) == 25
    assert len(store.orders()) == 8


def test_every_order_item_points_at_a_real_book():
    for order in store.orders():
        for item in order["items"]:
            assert store.catalog().by_id(item["book_id"]), item


def test_order_item_prices_match_the_catalog():
    for order in store.orders():
        for item in order["items"]:
            book = store.catalog().by_id(item["book_id"])
            assert item["unit_price"] == book["price"], (order["order_id"], item["title"])


def test_order_totals_add_up():
    for order in store.orders():
        expected = sum(i["unit_price"] * i["quantity"] for i in order["items"])
        assert round(expected, 2) == order["total"], order["order_id"]


@pytest.mark.parametrize("ref", ["BK-1003", "bk-1003", "The Hobbit", "the hobbit", "9780547928227"])
def test_resolve_book_accepts_id_title_and_isbn(ref):
    assert store.resolve_book(ref)["id"] == "BK-1003"


def test_exact_title_beats_a_fragment_of_a_longer_title():
    assert store.resolve_book("Dune")["id"] == "BK-1001"


def test_ambiguous_fragment_names_the_candidates():
    with pytest.raises(store.BookNotFound, match="more than one"):
        store.resolve_book("murder")


def test_unknown_book_raises():
    with pytest.raises(store.BookNotFound):
        store.resolve_book("The Silmarillion")


def test_availability_states():
    assert store.availability({"stock": 0, "restock_date": "2026-10-02"})["state"] == "out_of_stock"
    assert store.availability({"stock": 7, "restock_date": None})["state"] == "low_stock"
    assert store.availability({"stock": 99, "restock_date": None})["state"] == "in_stock"


def test_out_of_stock_has_no_shipping_estimate():
    snow_crash = store.resolve_book("Snow Crash")
    avail = store.availability(snow_crash)
    assert avail["estimated_shipping"] is None
    assert avail["restock_date"] == "2026-10-02"


def test_search_respects_price_and_stock_filters():
    results = store.search(genre="mystery", max_price=12.0, in_stock_only=True)
    assert results
    for r in results:
        assert float(r["price"].split()[0]) <= 12.0
        assert r["availability"]["units_in_stock"] > 0


def test_search_with_no_match_returns_nothing():
    assert store.search(query="quantum gardening for cats") == []


def test_recommendations_exclude_the_seed_book():
    out = store.similar_to("The Martian", limit=5)
    assert "BK-1013" not in {r["id"] for r in out["recommendations"]}


def test_recommendations_rank_the_same_author_and_themes_first():
    out = store.similar_to("The Martian", limit=1)
    top = out["recommendations"][0]
    assert top["title"] == "Project Hail Mary"
    assert "Andy Weir" in top["why"]


def test_recommendations_are_sorted_by_score():
    picks = store.similar_to("A Game of Thrones", limit=5)["recommendations"]
    scores = [p["match_score"] for p in picks]
    assert scores == sorted(scores, reverse=True)


def test_in_stock_only_recommendations_skip_backordered_titles():
    picks = store.similar_to("Dune", limit=5, in_stock_only=True)["recommendations"]
    assert "BK-1002" not in {p["id"] for p in picks}  # Dune Messiah is out of stock


def test_order_id_prefix_is_optional():
    assert tools.get_order_status("24804")["order"]["order_id"] == "ORD-24804"
    assert tools.get_order_status("ord-24804")["order"]["order_id"] == "ORD-24804"


def test_shipped_order_exposes_tracking():
    view = tools.get_order_status("ORD-24801")["order"]
    assert view["status"] == "shipped"
    assert view["tracking"]["carrier"] == "UPS"


def test_processing_order_has_no_tracking_yet():
    assert "tracking" not in tools.get_order_status("ORD-24803")["order"]


def test_unknown_order_returns_a_readable_error():
    out = tools.get_order_status("ORD-99999")
    assert out["found"] is False and "ORD-99999" in out["error"]


def test_orders_for_customer_are_newest_first():
    found = store.orders_for("YUKI@example.com")  # lookup is case-insensitive
    assert [o["order_id"] for o in found] == ["ORD-24805", "ORD-24806"]


def test_unknown_email_returns_a_note_not_an_error():
    out = tools.find_orders_by_email("nobody@example.com")
    assert out["count"] == 0 and "note" in out


def test_every_tool_exposes_a_described_schema():
    for t in tools.BOOKSTORE_TOOLS:
        schema = t.tool_spec["inputSchema"]["json"]
        assert t.tool_spec["description"].strip()
        for name, prop in schema["properties"].items():
            assert prop.get("description"), f"{t.tool_spec['name']}.{name} has no description"


@pytest.mark.parametrize("query,expected", [
    ("mysteries", "Murder on the Orient Express"),   # plural -> singular genre
    ("hacking", "Neuromancer"),                       # tag match
    ("tolkien", "The Hobbit"),                        # author match
    ("surveillance", "1984"),                         # blurb/tag match
])
def test_search_matches_natural_phrasing(query, expected):
    titles = [r["title"] for r in store.search(query=query, limit=3)]
    assert expected in titles


def test_short_stopwords_do_not_match_everything():
    """'for' and 'the' must not drag in the whole catalog."""
    assert store.search(query="for the with") == []


def test_unknown_provider_is_rejected():
    from bookstore import agent as agent_mod
    with pytest.raises(ValueError, match="Unknown provider"):
        agent_mod.build_agent(provider="anthropic")


def test_openai_provider_without_a_key_explains_itself(monkeypatch):
    from bookstore import agent as agent_mod
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        agent_mod.build_agent(provider="openai")
