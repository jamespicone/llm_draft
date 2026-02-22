"""Pack generator — builds booster packs from MTGJSON configuration."""
from __future__ import annotations

import logging
import random
from typing import Any

from card_database import Card, CardDatabase

logger = logging.getLogger(__name__)


class PackGenerator:
    def __init__(self, card_db: CardDatabase) -> None:
        self.card_db = card_db

    def generate_all_packs(
        self,
        set_code: str,
        num_seats: int = 8,
        num_rounds: int = 3,
        seed: int | None = None,
    ) -> list[list[list[Card]]]:
        """Generate all packs as packs[round][seat] = list[Card]."""
        rng = random.Random(seed)
        set_code = set_code.upper()

        mtgjson_data = self.card_db._mtgjson_cache.get(set_code, {})
        booster_config = mtgjson_data.get("data", {}).get("booster", {})

        # Try "default" booster type, then "arena", then first available
        booster_type: dict | None = None
        if booster_config:
            booster_type = (
                booster_config.get("default")
                or booster_config.get("arena")
                or next(iter(booster_config.values()), None)
            )

        all_cards = self.card_db.get_cards_for_set(set_code)
        if not all_cards:
            logger.warning(f"No cards found for set {set_code}")

        packs: list[list[list[Card]]] = []
        for _round_idx in range(num_rounds):
            round_packs: list[list[Card]] = []
            for _seat_idx in range(num_seats):
                if booster_type and self.card_db._mtgjson_uuid_to_card:
                    pack = self._generate_mtgjson_pack(booster_type, set_code, rng)
                else:
                    pack = self._generate_fallback_pack(all_cards, rng)
                # Fall back if MTGJSON pack is empty
                if not pack and all_cards:
                    pack = self._generate_fallback_pack(all_cards, rng)
                round_packs.append(pack)
            packs.append(round_packs)

        return packs

    def _generate_mtgjson_pack(
        self, booster_type: dict, set_code: str, rng: random.Random
    ) -> list[Card]:
        boosters: list[dict] = booster_type.get("boosters", [])
        total_weight: int = booster_type.get("boostersTotalWeight", 0)
        sheets: dict[str, Any] = booster_type.get("sheets", {})

        if not boosters or total_weight == 0:
            return self._generate_fallback_pack(
                self.card_db.get_cards_for_set(set_code), rng
            )

        # Select pack structure by weighted random
        structure = self._weighted_choice(
            boosters, [b.get("weight", 1) for b in boosters], rng
        )

        contents: dict[str, int] = structure.get("contents", {})
        pack: list[Card] = []
        used_scryfall_ids: set[str] = set()

        for sheet_name, count in contents.items():
            sheet = sheets.get(sheet_name, {})
            cards_map: dict[str, int] = sheet.get("cards", {})
            sheet_total: int = sheet.get("totalWeight", 0)
            balance_colors: bool = sheet.get("balanceColors", False)

            if not cards_map or sheet_total == 0:
                continue

            uuid_list = list(cards_map.keys())
            weight_list = [cards_map[u] for u in uuid_list]

            selected = self._pick_from_sheet(
                uuid_list,
                weight_list,
                count,
                balance_colors,
                used_scryfall_ids,
                rng,
            )
            pack.extend(selected)
            for card in selected:
                used_scryfall_ids.add(card.scryfall_id)

        return pack

    def _pick_from_sheet(
        self,
        uuid_list: list[str],
        weight_list: list[int],
        count: int,
        balance_colors: bool,
        used_scryfall_ids: set[str],
        rng: random.Random,
    ) -> list[Card]:
        uuid_to_card = self.card_db._mtgjson_uuid_to_card

        def try_pick(n: int) -> list[Card] | None:
            available: list[tuple[str, int]] = [
                (u, weight_list[i])
                for i, u in enumerate(uuid_list)
                if u in uuid_to_card
                and uuid_to_card[u].scryfall_id not in used_scryfall_ids
            ]
            if len(available) < n:
                return None

            selected_cards: list[Card] = []
            remaining = list(available)

            for _ in range(n):
                if not remaining:
                    break
                uuids = [x[0] for x in remaining]
                weights = [x[1] for x in remaining]
                chosen_uuid = rng.choices(uuids, weights=weights, k=1)[0]
                idx = uuids.index(chosen_uuid)
                selected_cards.append(uuid_to_card[chosen_uuid])
                remaining.pop(idx)

            return selected_cards

        if balance_colors:
            mono_colors = {"W", "U", "B", "R", "G"}
            for _attempt in range(10):
                cards = try_pick(count)
                if cards is None:
                    break
                present = {c for card in cards for c in card.colors if len(card.colors) == 1}
                if mono_colors.issubset(present):
                    return cards
            # Last attempt without balance check
            result = try_pick(count)
            return result if result is not None else []
        else:
            result = try_pick(count)
            return result if result is not None else []

    def _weighted_choice(
        self, items: list, weights: list, rng: random.Random
    ) -> Any:
        return rng.choices(items, weights=weights, k=1)[0]

    def _generate_fallback_pack(
        self, all_cards: list[Card], rng: random.Random
    ) -> list[Card]:
        """Fallback: 10C / 3U / 1R-or-M / 1 basic land."""
        commons = [
            c for c in all_cards
            if c.rarity == "common" and "Basic" not in c.type_line
        ]
        uncommons = [c for c in all_cards if c.rarity == "uncommon"]
        rares = [c for c in all_cards if c.rarity == "rare"]
        mythics = [c for c in all_cards if c.rarity == "mythic"]
        basics = [
            c for c in all_cards if "Basic" in c.type_line and "Land" in c.type_line
        ]

        pack: list[Card] = []
        used_names: set[str] = set()

        def pick_n(pool: list[Card], n: int) -> list[Card]:
            available = [c for c in pool if c.name not in used_names]
            chosen = rng.sample(available, min(n, len(available)))
            for c in chosen:
                used_names.add(c.name)
            return chosen

        pack.extend(pick_n(commons, 10))
        pack.extend(pick_n(uncommons, 3))

        # 1/8 mythic, else rare
        if mythics and rng.random() < 0.125:
            pack.extend(pick_n(mythics, 1))
        elif rares:
            pack.extend(pick_n(rares, 1))
        elif mythics:
            pack.extend(pick_n(mythics, 1))

        if basics:
            pack.extend(pick_n(basics, 1))

        return pack
