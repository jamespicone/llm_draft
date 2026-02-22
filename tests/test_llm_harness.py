"""Tests for LLM harness — tool dispatch and fuzzy card name matching."""
from __future__ import annotations

import pytest

from card_database import Card, CardDatabase
from draft_engine import SeatState
from llm_harness import LLMHarness
from llm_provider import LLMProvider, LLMResponse, TextBlock, ToolUseBlock


def make_card(
    name: str,
    rarity: str = "common",
    colors: list[str] | None = None,
    oracle_text: str = "",
    mana_cost: str = "{2}",
    cmc: float = 2.0,
    type_line: str = "Creature — Human",
    power: str = "2",
    toughness: str = "2",
) -> Card:
    return Card(
        name=name,
        mana_cost=mana_cost,
        cmc=cmc,
        type_line=type_line,
        oracle_text=oracle_text,
        colors=colors or [],
        color_identity=colors or [],
        rarity=rarity,
        power=power,
        toughness=toughness,
        set_code="TST",
        collector_number="1",
        scryfall_id=name,
        card_faces=None,
        keywords=[],
    )


class MockCardDatabase:
    def __init__(self, cards: list[Card]) -> None:
        self._cards = cards
        self._set_cards = {"TST": cards}

    def search_cards(self, query: str, set_code: str) -> list[Card]:
        q = query.lower()
        return [c for c in self._cards if q in c.name.lower() or q in c.oracle_text.lower()]

    def format_card_for_llm(self, card: Card) -> str:
        rarity_map = {"common": "C", "uncommon": "U", "rare": "R", "mythic": "M"}
        ri = rarity_map.get(card.rarity, "?")
        header = f"[{ri}] {card.name} {card.mana_cost}"
        type_str = f"    {card.type_line} ({card.power}/{card.toughness})"
        lines = [header, type_str]
        if card.oracle_text:
            lines.append(f"    {card.oracle_text}")
        return "\n".join(lines)


class MockProvider:
    """Provider that never actually sends messages — for unit testing dispatch only."""

    async def send_message(self, system, messages, tools, tool_choice, max_tokens):
        raise NotImplementedError("MockProvider should not be called in unit tests")


def make_harness(cards: list[Card] | None = None) -> tuple[LLMHarness, list[Card]]:
    if cards is None:
        cards = [
            make_card("Sheoldred, the Apocalypse", rarity="mythic", colors=["B"]),
            make_card("Go for the Throat", rarity="uncommon", colors=["B"]),
            make_card("Llanowar Elves", rarity="common", colors=["G"]),
        ]
    db = MockCardDatabase(cards)
    harness = LLMHarness(MockProvider(), db, "system prompt")  # type: ignore[arg-type]
    harness.set_set_code("TST")
    return harness, cards


# ── Fuzzy card name matching ─────────────────────────────────────────────────

class TestFindCardInPack:
    def test_exact_match(self):
        harness, cards = make_harness()
        result = harness._find_card_in_pack("Go for the Throat", cards)
        assert result is not None
        assert result.name == "Go for the Throat"

    def test_case_insensitive_match(self):
        harness, cards = make_harness()
        result = harness._find_card_in_pack("go for the throat", cards)
        assert result is not None
        assert result.name == "Go for the Throat"

    def test_case_insensitive_mixed(self):
        harness, cards = make_harness()
        result = harness._find_card_in_pack("LLANOWAR ELVES", cards)
        assert result is not None
        assert result.name == "Llanowar Elves"

    def test_levenshtein_typo_match(self):
        """Small typo should still match via Levenshtein."""
        harness, cards = make_harness()
        # "Sheoldred the Apocalypse" is missing the comma but otherwise identical
        result = harness._find_card_in_pack("Sheoldred the Apocalypse", cards)
        assert result is not None
        assert result.name == "Sheoldred, the Apocalypse"

    def test_levenshtein_misspelling(self):
        harness, cards = make_harness()
        # "Sheoldred, the Apocolypse" — single character swap
        result = harness._find_card_in_pack("Sheoldred, the Apocolypse", cards)
        assert result is not None
        assert result.name == "Sheoldred, the Apocalypse"

    def test_invalid_name_returns_none(self):
        harness, cards = make_harness()
        result = harness._find_card_in_pack("Totally Fake Card XYZ123", cards)
        assert result is None


# ── Tool dispatch ────────────────────────────────────────────────────────────

class TestToolPickCard:
    def test_valid_name_returns_card(self):
        harness, cards = make_harness()
        block = ToolUseBlock(
            id="test-1",
            name="pick_card",
            input={"card_name": "Llanowar Elves", "reasoning": "Good mana dork."},
        )
        result, card = harness._dispatch_tool(block, cards, SeatState())
        assert card is not None
        assert card.name == "Llanowar Elves"
        assert "Picked" in result

    def test_invalid_name_returns_error_with_available_cards(self):
        harness, cards = make_harness()
        block = ToolUseBlock(
            id="test-2",
            name="pick_card",
            input={"card_name": "Nonexistent Card", "reasoning": "..."},
        )
        result, card = harness._dispatch_tool(block, cards, SeatState())
        assert card is None
        assert "not found" in result.lower()
        # Available cards should be listed
        assert "Llanowar Elves" in result

    def test_case_insensitive_pick(self):
        harness, cards = make_harness()
        block = ToolUseBlock(
            id="test-3",
            name="pick_card",
            input={"card_name": "llanowar elves", "reasoning": "Mana."},
        )
        result, card = harness._dispatch_tool(block, cards, SeatState())
        assert card is not None
        assert card.name == "Llanowar Elves"


class TestToolViewCurrentPack:
    def test_returns_card_list(self):
        harness, cards = make_harness()
        block = ToolUseBlock(id="t", name="view_current_pack", input={})
        result, card = harness._dispatch_tool(block, cards, SeatState())
        assert card is None
        assert "Sheoldred" in result
        assert "Llanowar Elves" in result

    def test_empty_pack(self):
        harness, _ = make_harness()
        block = ToolUseBlock(id="t", name="view_current_pack", input={})
        result, card = harness._dispatch_tool(block, [], SeatState())
        assert card is None
        assert "empty" in result.lower()


class TestToolViewMyPicks:
    def _make_seat_with_picks(self) -> SeatState:
        seat = SeatState()
        seat.picks = [
            make_card("Black Creature", colors=["B"], rarity="common"),
            make_card("White Instant", colors=["W"], type_line="Instant", power=None, toughness=None),
            make_card("Green Land", colors=["G"], type_line="Basic Land — Forest"),
        ]
        return seat

    def _dispatch_view(self, harness, cards, seat, group_by):
        block = ToolUseBlock(
            id="t", name="view_my_picks", input={"group_by": group_by}
        )
        result, card = harness._dispatch_tool(block, cards, seat)
        return result, card

    def test_pick_order_returns_nonempty_string(self):
        harness, cards = make_harness()
        seat = self._make_seat_with_picks()
        result, card = self._dispatch_view(harness, cards, seat, "pick_order")
        assert card is None
        assert len(result) > 0
        assert "Black Creature" in result

    def test_color_grouping_returns_nonempty_string(self):
        harness, cards = make_harness()
        seat = self._make_seat_with_picks()
        result, card = self._dispatch_view(harness, cards, seat, "color")
        assert card is None
        assert len(result) > 0

    def test_type_grouping_returns_nonempty_string(self):
        harness, cards = make_harness()
        seat = self._make_seat_with_picks()
        result, card = self._dispatch_view(harness, cards, seat, "type")
        assert card is None
        assert len(result) > 0
        assert "Creature" in result or "Instant" in result

    def test_cmc_grouping_returns_nonempty_string(self):
        harness, cards = make_harness()
        seat = self._make_seat_with_picks()
        result, card = self._dispatch_view(harness, cards, seat, "cmc")
        assert card is None
        assert len(result) > 0

    def test_no_picks_returns_empty_message(self):
        harness, cards = make_harness()
        result, card = self._dispatch_view(harness, cards, SeatState(), "pick_order")
        assert card is None
        assert "no cards" in result.lower()


class TestToolAddNote:
    def test_note_is_saved(self):
        harness, cards = make_harness()
        seat = SeatState()
        block = ToolUseBlock(
            id="t", name="add_note", input={"note": "Going black-blue."}
        )
        result, card = harness._dispatch_tool(block, cards, seat)
        assert card is None
        assert "Going black-blue." in seat.notes
        assert "saved" in result.lower()

    def test_empty_note_rejected(self):
        harness, cards = make_harness()
        seat = SeatState()
        block = ToolUseBlock(id="t", name="add_note", input={"note": "   "})
        result, card = harness._dispatch_tool(block, cards, seat)
        assert card is None
        assert len(seat.notes) == 0


class TestToolMoveCard:
    def test_move_to_sideboard(self):
        harness, cards = make_harness()
        seat = SeatState()
        card = make_card("Llanowar Elves", colors=["G"])
        seat.picks.append(card)

        block = ToolUseBlock(
            id="t",
            name="move_card",
            input={"card_name": "Llanowar Elves", "destination": "sideboard"},
        )
        result, picked = harness._dispatch_tool(block, cards, seat)
        assert picked is None
        assert len(seat.picks) == 0
        assert len(seat.sideboard) == 1
        assert "sideboard" in result.lower()

    def test_move_back_to_deck(self):
        harness, cards = make_harness()
        seat = SeatState()
        card = make_card("Llanowar Elves", colors=["G"])
        seat.sideboard.append(card)

        block = ToolUseBlock(
            id="t",
            name="move_card",
            input={"card_name": "Llanowar Elves", "destination": "deck"},
        )
        result, picked = harness._dispatch_tool(block, cards, seat)
        assert picked is None
        assert len(seat.sideboard) == 0
        assert len(seat.picks) == 1

    def test_move_nonexistent_returns_error(self):
        harness, cards = make_harness()
        seat = SeatState()
        block = ToolUseBlock(
            id="t",
            name="move_card",
            input={"card_name": "Fake Card", "destination": "sideboard"},
        )
        result, picked = harness._dispatch_tool(block, cards, seat)
        assert picked is None
        assert "not found" in result.lower()


class TestFormatPickMessage:
    def test_includes_pack_and_round_info(self):
        harness, cards = make_harness()
        seat = SeatState()
        msg = harness.format_pick_message(cards, seat, round_num=0, pick_num=0)
        assert "Pack 1, Pick 1" in msg

    def test_includes_card_names(self):
        harness, cards = make_harness()
        seat = SeatState()
        msg = harness.format_pick_message(cards, seat, round_num=1, pick_num=5)
        assert "Sheoldred" in msg
        assert "Llanowar Elves" in msg

    def test_shows_no_notes_when_empty(self):
        harness, cards = make_harness()
        seat = SeatState()
        msg = harness.format_pick_message(cards, seat, round_num=0, pick_num=0)
        assert "No notes yet" in msg

    def test_shows_notes_when_present(self):
        harness, cards = make_harness()
        seat = SeatState()
        seat.notes.append("Going black.")
        msg = harness.format_pick_message(cards, seat, round_num=0, pick_num=0)
        assert "Going black." in msg
