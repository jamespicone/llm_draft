"""Tests for pack generator."""
from __future__ import annotations

import random

import pytest

from card_database import Card, CardDatabase
from pack_generator import PackGenerator


def make_card(
    name: str,
    rarity: str = "common",
    colors: list[str] | None = None,
    type_line: str = "Creature",
) -> Card:
    return Card(
        name=name,
        mana_cost="{2}",
        cmc=2.0,
        type_line=type_line,
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


def make_card_pool(
    n_commons: int = 60,
    n_uncommons: int = 30,
    n_rares: int = 15,
    n_mythics: int = 10,
    n_basics: int = 5,
) -> list[Card]:
    cards: list[Card] = []
    for i in range(n_commons):
        cards.append(make_card(f"Common-{i}", rarity="common"))
    for i in range(n_uncommons):
        cards.append(make_card(f"Uncommon-{i}", rarity="uncommon"))
    for i in range(n_rares):
        cards.append(make_card(f"Rare-{i}", rarity="rare"))
    for i in range(n_mythics):
        cards.append(make_card(f"Mythic-{i}", rarity="mythic"))
    for i in range(n_basics):
        cards.append(
            make_card(f"Plains-{i}", rarity="common", type_line="Basic Land — Plains")
        )
    return cards


class MockCardDatabase:
    def __init__(self, cards: list[Card]) -> None:
        self._cards = cards
        self._mtgjson_cache: dict = {}
        self._mtgjson_uuid_to_card: dict = {}

    def get_cards_for_set(self, set_code: str) -> list[Card]:
        return self._cards


def make_generator(cards: list[Card] | None = None) -> PackGenerator:
    if cards is None:
        cards = make_card_pool()
    db = MockCardDatabase(cards)
    return PackGenerator(db)  # type: ignore[arg-type]


# ── Fallback pack generation ─────────────────────────────────────────────────

class TestFallbackPackGeneration:
    def test_pack_has_expected_card_count(self):
        gen = make_generator()
        rng = random.Random(42)
        pack = gen._generate_fallback_pack(make_card_pool(), rng)
        # 10C + 3U + 1R-or-M + 1 basic = 15 (when all pools are big enough)
        assert len(pack) == 15

    def test_no_duplicate_cards_in_pack(self):
        gen = make_generator()
        rng = random.Random(1)
        pack = gen._generate_fallback_pack(make_card_pool(), rng)
        names = [c.name for c in pack]
        assert len(names) == len(set(names))

    def test_rarity_distribution(self):
        """Each fallback pack has 10C, 3U, and 1 rare-or-mythic."""
        gen = make_generator()
        rng = random.Random(99)
        cards = make_card_pool()

        n_trials = 100
        for _ in range(n_trials):
            pack = gen._generate_fallback_pack(cards, rng)
            commons = [c for c in pack if c.rarity == "common" and "Basic" not in c.type_line]
            uncommons = [c for c in pack if c.rarity == "uncommon"]
            rares = [c for c in pack if c.rarity in ("rare", "mythic")]
            basics = [c for c in pack if "Basic" in c.type_line]

            assert len(commons) == 10, f"Expected 10 commons, got {len(commons)}"
            assert len(uncommons) == 3, f"Expected 3 uncommons, got {len(uncommons)}"
            assert len(rares) == 1, f"Expected 1 rare/mythic, got {len(rares)}"
            assert len(basics) == 1, f"Expected 1 basic, got {len(basics)}"

    def test_mythic_rate_approximately_correct(self):
        """Mythics should appear at roughly 1/8 = 12.5% of packs."""
        gen = make_generator()
        rng = random.Random(7)
        cards = make_card_pool()

        n_trials = 800
        mythic_count = sum(
            1
            for _ in range(n_trials)
            for c in gen._generate_fallback_pack(cards, rng)
            if c.rarity == "mythic"
        )
        mythic_rate = mythic_count / n_trials
        # Allow generous margin for randomness
        assert 0.05 <= mythic_rate <= 0.25, f"Mythic rate {mythic_rate:.2%} out of expected range"

    def test_fallback_works_with_only_commons(self):
        """Gracefully handles a pool with only commons (no rares/mythics)."""
        cards = [make_card(f"Common-{i}", rarity="common") for i in range(15)]
        gen = make_generator(cards)
        rng = random.Random(0)
        pack = gen._generate_fallback_pack(cards, rng)
        assert len(pack) > 0

    def test_all_packs_generated(self):
        gen = make_generator()
        packs = gen.generate_all_packs("TST", num_seats=8, num_rounds=3, seed=42)
        assert len(packs) == 3
        for round_packs in packs:
            assert len(round_packs) == 8
            for pack in round_packs:
                assert len(pack) > 0

    def test_reproducible_with_seed(self):
        cards = make_card_pool()
        gen1 = make_generator(cards)
        gen2 = make_generator(cards)

        packs1 = gen1.generate_all_packs("TST", seed=123)
        packs2 = gen2.generate_all_packs("TST", seed=123)

        names1 = [[c.name for c in pack] for pack in packs1[0]]
        names2 = [[c.name for c in pack] for pack in packs2[0]]
        assert names1 == names2

    def test_different_seeds_produce_different_packs(self):
        cards = make_card_pool()
        gen1 = make_generator(cards)
        gen2 = make_generator(cards)

        packs1 = gen1.generate_all_packs("TST", seed=1)
        packs2 = gen2.generate_all_packs("TST", seed=2)

        names1 = {c.name for c in packs1[0][0]}
        names2 = {c.name for c in packs2[0][0]}
        # Very unlikely to be identical with different seeds and a large pool
        assert names1 != names2
