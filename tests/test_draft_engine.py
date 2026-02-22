"""Tests for draft engine — pack rotation and bot pick logic."""
from __future__ import annotations

import pytest

from card_database import Card
from draft_engine import SeatState, _rotate_packs
from ratings import Ratings


def make_card(
    name: str,
    rarity: str = "common",
    colors: list[str] | None = None,
) -> Card:
    return Card(
        name=name,
        mana_cost="",
        cmc=2.0,
        type_line="Creature",
        oracle_text="",
        colors=colors or [],
        color_identity=colors or [],
        rarity=rarity,
        power="2",
        toughness="2",
        set_code="TST",
        collector_number="1",
        scryfall_id=name,
        card_faces=None,
        keywords=[],
    )


def make_packs(n: int = 8) -> list[list[Card]]:
    """Create n distinct packs each containing a single unique card."""
    return [[make_card(f"Card-{i}")] for i in range(n)]


# ── Pack rotation ────────────────────────────────────────────────────────────

class TestPackRotationLeft:
    def test_seat0_receives_seat7_pack(self):
        packs = make_packs(8)
        original_seat7_card = packs[7][0]

        rotated = _rotate_packs(packs, pass_left=True)

        assert rotated[0][0].name == original_seat7_card.name

    def test_seat_n_receives_seat_n_minus_1_pack(self):
        packs = make_packs(8)
        originals = [p[0].name for p in packs]

        rotated = _rotate_packs(packs, pass_left=True)

        # After pass-left rotation, seat i gets what was seat i-1's pack
        # (seat 0 gets seat 7, seat 1 gets seat 0, ...)
        assert rotated[1][0].name == originals[0]
        assert rotated[2][0].name == originals[1]
        assert rotated[7][0].name == originals[6]

    def test_all_packs_seen_after_7_rotations(self):
        """After 7 left-rotations seat 0 has seen all 8 original packs."""
        packs = make_packs(8)
        original_names = {p[0].name for p in packs}

        seen_by_seat0: set[str] = {packs[0][0].name}
        current = packs
        for _ in range(7):
            current = _rotate_packs(current, pass_left=True)
            seen_by_seat0.add(current[0][0].name)

        assert seen_by_seat0 == original_names

    def test_rotation_preserves_pack_count(self):
        packs = make_packs(8)
        rotated = _rotate_packs(packs, pass_left=True)
        assert len(rotated) == 8


class TestPackRotationRight:
    def test_seat0_receives_seat1_pack(self):
        packs = make_packs(8)
        original_seat1_card = packs[1][0]

        rotated = _rotate_packs(packs, pass_left=False)

        assert rotated[0][0].name == original_seat1_card.name

    def test_seat_n_receives_seat_n_plus_1_pack(self):
        packs = make_packs(8)
        originals = [p[0].name for p in packs]

        rotated = _rotate_packs(packs, pass_left=False)

        # Pass right: seat i gets what was seat i+1's pack
        # (seat 0 gets seat 1, seat 1 gets seat 2, ..., seat 7 gets seat 0)
        assert rotated[0][0].name == originals[1]
        assert rotated[1][0].name == originals[2]
        assert rotated[7][0].name == originals[0]

    def test_all_packs_seen_after_7_rotations(self):
        packs = make_packs(8)
        original_names = {p[0].name for p in packs}

        seen_by_seat0: set[str] = {packs[0][0].name}
        current = packs
        for _ in range(7):
            current = _rotate_packs(current, pass_left=False)
            seen_by_seat0.add(current[0][0].name)

        assert seen_by_seat0 == original_names

    def test_left_and_right_are_inverses(self):
        """One left rotation followed by one right rotation returns to original."""
        packs = make_packs(8)
        originals = [p[0].name for p in packs]

        rotated = _rotate_packs(_rotate_packs(packs, pass_left=True), pass_left=False)

        assert [p[0].name for p in rotated] == originals


# ── Bot pick logic ───────────────────────────────────────────────────────────

class MockRatings:
    def __init__(self, ratings: dict[str, float]) -> None:
        self._data = ratings

    def get_rating(self, card_name: str, set_code: str) -> float | None:
        return self._data.get(card_name)

    def get_all_ratings(self, set_code: str) -> dict[str, float]:
        return dict(self._data)


def make_draft_engine_bot(ratings_data: dict[str, float]):
    """Create a minimal DraftEngine-like object with bot_pick accessible."""
    from draft_engine import DraftEngine, DraftState

    mock_ratings = MockRatings(ratings_data)

    class _FakeHarness:
        def set_set_code(self, _): pass
        def start_round(self, _): pass
        async def make_pick(self, *a, **kw): raise NotImplementedError

    state = DraftState(
        set_code="TST",
        seats=[SeatState()],
        packs=[[[]]],
    )
    engine = DraftEngine(state, mock_ratings, _FakeHarness())  # type: ignore[arg-type]
    return engine


class TestBotPick:
    def test_picks_highest_rated_with_no_affinity(self):
        pack = [
            make_card("Good Card", rarity="rare"),
            make_card("Bad Card", rarity="common"),
            make_card("Medium Card", rarity="uncommon"),
        ]
        ratings_data = {
            "Good Card": 0.65,
            "Bad Card": 0.48,
            "Medium Card": 0.55,
        }
        engine = make_draft_engine_bot(ratings_data)
        seat = SeatState()

        picked = engine._bot_pick(pack, seat, "TST")

        assert picked.name == "Good Card"

    def test_picks_using_fallback_when_no_ratings(self):
        """Without ratings, falls back to rarity: mythic > rare > uncommon > common."""
        pack = [
            make_card("A Common", rarity="common"),
            make_card("A Rare", rarity="rare"),
            make_card("An Uncommon", rarity="uncommon"),
        ]
        engine = make_draft_engine_bot({})
        seat = SeatState()

        picked = engine._bot_pick(pack, seat, "TST")

        assert picked.name == "A Rare"

    def test_color_affinity_updates_after_pick(self):
        red_card = make_card("Red Card", colors=["R"])
        pack = [red_card]
        engine = make_draft_engine_bot({"Red Card": 0.55})
        seat = SeatState()

        engine._bot_pick(pack, seat, "TST")

        assert seat.color_affinity.get("R", 0) > 0

    def test_color_affinity_boosts_on_color_card(self):
        """Bot with red affinity should prefer a weaker red card over a stronger off-color one."""
        red_card = make_card("Red Beater", colors=["R"], rarity="common")
        blue_card = make_card("Blue Flier", colors=["U"], rarity="common")
        pack = [red_card, blue_card]

        ratings_data = {
            "Red Beater": 0.54,
            "Blue Flier": 0.56,
        }
        engine = make_draft_engine_bot(ratings_data)

        # Build significant red affinity (affinity only applied when seat has prior picks)
        seat = SeatState()
        seat.picks = [make_card("Prior Pick")]  # triggers affinity logic
        seat.color_affinity["R"] = 10.0

        # With high red affinity, even a slightly weaker red card should beat off-color
        # red_score = 0.54 + (10 * 0.15) = 0.54 + 1.50 = 2.04
        # blue_score = 0.56 + 0 = 0.56
        picked = engine._bot_pick(pack, seat, "TST")

        assert picked.name == "Red Beater"

    def test_colorless_gets_small_bonus(self):
        """Colourless cards get a flat 0.01 affinity bonus when bot has picks."""
        colorless = make_card("Artifact", colors=[], rarity="common")
        colored = make_card("Colored Card", colors=["G"], rarity="common")
        pack = [colorless, colored]

        # Artifact rated slightly lower; colored card has no affinity bonus
        ratings_data = {"Artifact": 0.549, "Colored Card": 0.54}
        engine = make_draft_engine_bot(ratings_data)

        seat = SeatState()
        # Give seat a prior pick so affinity bonus kicks in
        seat.picks = [make_card("Prior Pick")]

        # Artifact score = 0.549 + 0.01 * 0.15 = 0.5505
        # Colored score  = 0.54 + 0 * 0.15 = 0.54
        picked = engine._bot_pick(pack, seat, "TST")

        assert picked.name == "Artifact"
