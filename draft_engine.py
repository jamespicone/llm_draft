"""Draft engine — state machine, pack rotation, and bot pick logic."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from card_database import Card
from ratings import Ratings

if TYPE_CHECKING:
    from llm_harness import LLMHarness

logger = logging.getLogger(__name__)

FALLBACK_RATINGS = {"mythic": 0.62, "rare": 0.58, "uncommon": 0.54, "common": 0.52}


@dataclass
class SeatState:
    picks: list[Card] = field(default_factory=list)
    sideboard: list[Card] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    color_affinity: dict[str, float] = field(default_factory=dict)


@dataclass
class DraftState:
    set_code: str
    seats: list[SeatState]
    packs: list[list[list[Card]]]  # [round][seat] = remaining cards
    current_round: int = 0
    current_pick: int = 0
    pack_size: int = 15


class DraftEngine:
    def __init__(
        self, state: DraftState, ratings: Ratings, llm_harness: "LLMHarness"
    ) -> None:
        self.state = state
        self.ratings = ratings
        self.llm_harness = llm_harness

    async def run(self) -> DraftState:
        state = self.state
        self.llm_harness.set_set_code(state.set_code)

        for round_idx in range(3):
            state.current_round = round_idx
            # Rounds 0 and 2 pass left (+1), round 1 passes right (-1)
            pass_left = round_idx % 2 == 0

            self.llm_harness.start_round(round_idx)

            # Determine picks from actual pack size
            if not state.packs[round_idx] or not state.packs[round_idx][0]:
                continue
            pack_size = len(state.packs[round_idx][0])

            for pick_idx in range(pack_size):
                state.current_pick = pick_idx

                # LLM picks from seat 0
                llm_pack = state.packs[round_idx][0]
                if llm_pack:
                    try:
                        picked_card = await self.llm_harness.make_pick(
                            llm_pack, state.seats[0], round_idx, pick_idx
                        )
                    except Exception as e:
                        logger.error(f"LLM pick failed: {e}. Using first card.")
                        picked_card = llm_pack[0]
                    llm_pack.remove(picked_card)
                    state.seats[0].picks.append(picked_card)

                # Bots pick from their packs
                num_seats = len(state.seats)
                for seat_idx in range(1, num_seats):
                    bot_pack = state.packs[round_idx][seat_idx]
                    if bot_pack:
                        picked = self._bot_pick(
                            bot_pack, state.seats[seat_idx], state.set_code
                        )
                        bot_pack.remove(picked)
                        state.seats[seat_idx].picks.append(picked)

                # Rotate packs (not needed after the last pick)
                if pick_idx < pack_size - 1:
                    state.packs[round_idx] = _rotate_packs(
                        state.packs[round_idx], pass_left
                    )

            # Context compression between rounds
            if round_idx < 2:
                self.llm_harness.compress_round(round_idx, state.seats[0])

        return state

    def _bot_pick(
        self, pack: list[Card], seat: SeatState, set_code: str
    ) -> Card:
        def score(card: Card) -> float:
            base = self.ratings.get_rating(card.name, set_code) or FALLBACK_RATINGS.get(
                card.rarity, 0.50
            )
            affinity_bonus = 0.0
            if seat.picks:
                card_colors = card.colors or []
                if not card_colors:
                    affinity_bonus = 0.01
                else:
                    affinity_bonus = (
                        sum(seat.color_affinity.get(c, 0.0) for c in card_colors)
                        / len(card_colors)
                    )
            return base + affinity_bonus * 0.15

        best_card = max(pack, key=score)

        for color in best_card.colors or []:
            seat.color_affinity[color] = seat.color_affinity.get(color, 0.0) + 1.0

        return best_card


def _rotate_packs(packs: list[list[Card]], pass_left: bool) -> list[list[Card]]:
    """
    Pass left (+1): seat N's pack goes to seat N+1 (seat 7 wraps to seat 0).
    After rotation, seat 0 receives what was seat 7's pack.
    Implementation: last element moves to front.

    Pass right (-1): seat N's pack goes to seat N-1 (seat 0 wraps to seat 7).
    After rotation, seat 0 receives what was seat 1's pack.
    Implementation: first element moves to back.
    """
    if pass_left:
        return [packs[-1]] + packs[:-1]
    else:
        return packs[1:] + [packs[0]]
