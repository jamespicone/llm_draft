"""LLM harness — tool definitions, tool dispatch loop, and context management."""
from __future__ import annotations

import logging
from collections import defaultdict

from card_database import Card, CardDatabase
from draft_engine import SeatState
from llm_provider import LLMProvider, TextBlock, ToolUseBlock

logger = logging.getLogger(__name__)

try:
    from Levenshtein import ratio as levenshtein_ratio
except ImportError:
    logger.warning("python-Levenshtein not installed; fuzzy matching will be limited")

    def levenshtein_ratio(a: str, b: str) -> float:  # type: ignore[misc]
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        longer = max(len(a), len(b))
        matches = sum(1 for c1, c2 in zip(a, b) if c1 == c2)
        return matches / longer


FUZZY_THRESHOLD = 0.85
MAX_ITERATIONS = 15
MAX_DECKBUILD_ITERATIONS = 30

BASIC_LAND_TYPES = ["Plains", "Island", "Swamp", "Mountain", "Forest"]

_BASIC_LAND_COLOR_IDENTITY: dict[str, list[str]] = {
    "Plains": ["W"], "Island": ["U"], "Swamp": ["B"],
    "Mountain": ["R"], "Forest": ["G"],
}


def _make_basic_land(land_type: str) -> Card:
    return Card(
        name=land_type, mana_cost="", cmc=0,
        type_line=f"Basic Land — {land_type}", oracle_text="",
        colors=[], color_identity=_BASIC_LAND_COLOR_IDENTITY[land_type],
        rarity="common", power=None, toughness=None,
        set_code="", collector_number="", scryfall_id="",
        card_faces=None, keywords=[],
    )


TOOLS: list[dict] = [
    {
        "name": "pick_card",
        "description": (
            "Pick a card from the current booster pack to add to your drafted card pool. "
            "You must call this exactly once per pack to make your selection. This is irreversible."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "card_name": {
                    "type": "string",
                    "description": "The exact name of the card to pick, as shown in the pack contents.",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation of why you chose this card (1-3 sentences).",
                },
            },
            "required": ["card_name", "reasoning"],
        },
    },
    {
        "name": "view_current_pack",
        "description": (
            "View the full details of all cards currently in the pack you are picking from. "
            "Returns each card with its complete oracle text, type, mana cost, and rarity."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "view_my_picks",
        "description": (
            "View all cards you have drafted so far, organised by category. "
            "Shows your current deck and sideboard grouped by the specified category."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "group_by": {
                    "type": "string",
                    "enum": ["color", "type", "cmc", "pick_order"],
                    "description": (
                        "How to group the cards. 'color' groups by card colour, "
                        "'type' by card type (creature/noncreature/land), "
                        "'cmc' by mana value, 'pick_order' shows cards in the order drafted."
                    ),
                },
            },
            "required": ["group_by"],
        },
    },
    {
        "name": "lookup_card",
        "description": (
            "Look up the full details of any card in the current set by name. "
            "Use this to check what a card does. Supports partial name matching."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "card_name": {
                    "type": "string",
                    "description": "The name (or partial name) of the card to look up.",
                },
            },
            "required": ["card_name"],
        },
    },
    {
        "name": "move_card",
        "description": (
            "Move a card between your main deck and sideboard. "
            "Use this for cards you have hate-drafted or do not expect to play."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "card_name": {
                    "type": "string",
                    "description": "The exact name of the card to move.",
                },
                "destination": {
                    "type": "string",
                    "enum": ["sideboard", "deck"],
                    "description": "Where to move the card.",
                },
            },
            "required": ["card_name", "destination"],
        },
    },
    {
        "name": "add_note",
        "description": (
            "Save a strategic note for yourself. "
            "These notes persist across picks and are shown at the start of each pick. "
            "Use this to track your draft plan, colour commitment, signals, and card targets."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "note": {
                    "type": "string",
                    "description": "The note text. Keep it concise — a sentence or two.",
                },
            },
            "required": ["note"],
        },
    },
]

DECKBUILDING_TOOLS: list[dict] = [
    {
        "name": "view_my_picks",
        "description": (
            "View all cards in your deck and sideboard, organised by category. "
            "Use this to review your pool and decide which lands to add and which spells to cut."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "group_by": {
                    "type": "string",
                    "enum": ["color", "type", "cmc", "pick_order"],
                    "description": "How to group the cards.",
                },
            },
            "required": ["group_by"],
        },
    },
    {
        "name": "lookup_card",
        "description": "Look up the full oracle text of any card in the set by name.",
        "input_schema": {
            "type": "object",
            "properties": {
                "card_name": {"type": "string", "description": "Card name to look up."},
            },
            "required": ["card_name"],
        },
    },
    {
        "name": "move_card",
        "description": "Move a drafted card between your main deck and sideboard.",
        "input_schema": {
            "type": "object",
            "properties": {
                "card_name": {"type": "string", "description": "Card to move."},
                "destination": {
                    "type": "string",
                    "enum": ["sideboard", "deck"],
                    "description": "Where to move the card.",
                },
            },
            "required": ["card_name", "destination"],
        },
    },
    {
        "name": "add_note",
        "description": "Add a note about your deck-building decisions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "note": {"type": "string", "description": "Note text."},
            },
            "required": ["note"],
        },
    },
    {
        "name": "add_basic_land",
        "description": (
            "Add one or more basic lands of a given type to your main deck. "
            "Standard Limited decks run approximately 17 lands in a 40-card deck."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "land_type": {
                    "type": "string",
                    "enum": ["Plains", "Island", "Swamp", "Mountain", "Forest"],
                    "description": "The basic land type to add.",
                },
                "count": {
                    "type": "integer",
                    "description": "Number of lands to add (default 1).",
                    "minimum": 1,
                },
            },
            "required": ["land_type"],
        },
    },
    {
        "name": "remove_basic_land",
        "description": "Remove one or more basic lands of a given type from your main deck.",
        "input_schema": {
            "type": "object",
            "properties": {
                "land_type": {
                    "type": "string",
                    "enum": ["Plains", "Island", "Swamp", "Mountain", "Forest"],
                    "description": "The basic land type to remove.",
                },
                "count": {
                    "type": "integer",
                    "description": "Number of lands to remove (default 1).",
                    "minimum": 1,
                },
            },
            "required": ["land_type"],
        },
    },
    {
        "name": "finalize_deck",
        "description": (
            "Lock in your deck and sideboard. Call this when you are satisfied with "
            "your 40-card main deck (approximately 17 lands + 23 spells)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "notes": {
                    "type": "string",
                    "description": "Optional notes about your final deck (strategy, key synergies, etc.).",
                },
            },
            "required": [],
        },
    },
]


class LLMHarness:
    def __init__(
        self, provider: LLMProvider, card_db: CardDatabase, system_prompt: str
    ) -> None:
        self.provider = provider
        self.card_db = card_db
        self.system_prompt = system_prompt
        self.messages: list[dict] = []

        # Token / call tracking
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_api_calls: int = 0

        # Per-pick data (set after each make_pick call)
        self._last_reasoning: str = ""
        self._last_tool_call_count: int = 0

        # Round tracking for context compression
        self._current_set_code: str = ""
        self._current_round_num: int = 0
        self._round_picks_data: list[tuple[str, str]] = []  # (card_name, reasoning)
        self._round_msg_start: int = 0  # message index where current round started

    def set_set_code(self, set_code: str) -> None:
        self._current_set_code = set_code.upper()

    def start_round(self, round_num: int) -> None:
        """Mark the start of a new round for context compression purposes."""
        self._current_round_num = round_num
        self._round_picks_data = []
        self._round_msg_start = len(self.messages)

    async def make_pick(
        self,
        pack: list[Card],
        seat_state: SeatState,
        round_num: int,
        pick_num: int,
    ) -> Card:
        user_msg = self.format_pick_message(pack, seat_state, round_num, pick_num)
        self.messages.append({"role": "user", "content": user_msg})

        picked_card: Card | None = None
        reasoning: str = ""
        tool_call_count: int = 0

        for iteration in range(MAX_ITERATIONS):
            tool_choice = "any" if iteration == MAX_ITERATIONS - 1 else "auto"

            response = await self.provider.send_message(
                system=self.system_prompt,
                messages=self.messages,
                tools=TOOLS,
                tool_choice=tool_choice,
                max_tokens=2048,
            )

            self.total_api_calls += 1
            self.total_input_tokens += response.usage.get("input_tokens", 0)
            self.total_output_tokens += response.usage.get("output_tokens", 0)

            # Store assistant response as dicts for message history
            assistant_content: list[dict] = []
            for block in response.content:
                if isinstance(block, TextBlock):
                    assistant_content.append({"type": "text", "text": block.text})
                elif isinstance(block, ToolUseBlock):
                    assistant_content.append(
                        {
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        }
                    )
            self.messages.append({"role": "assistant", "content": assistant_content})

            # Dispatch tool calls
            tool_results: list[dict] = []
            for block in response.content:
                if isinstance(block, ToolUseBlock):
                    tool_call_count += 1
                    result, card = self._dispatch_tool(block, pack, seat_state)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        }
                    )
                    if card is not None:
                        picked_card = card
                        reasoning = block.input.get("reasoning", "")

            if tool_results:
                self.messages.append({"role": "user", "content": tool_results})

            if picked_card is not None:
                self._last_reasoning = reasoning
                self._last_tool_call_count = tool_call_count
                self._round_picks_data.append((picked_card.name, reasoning))
                return picked_card

            if response.stop_reason == "end_turn":
                self.messages.append(
                    {
                        "role": "user",
                        "content": "You need to pick a card. Please call the pick_card tool.",
                    }
                )

        # Fallback: pick first card
        logger.warning(
            f"LLM failed to pick after {MAX_ITERATIONS} iterations. Using first card."
        )
        fallback = pack[0]
        self._last_reasoning = "Fallback: auto-picked first card."
        self._last_tool_call_count = tool_call_count
        self._round_picks_data.append((fallback.name, self._last_reasoning))
        return fallback

    def _dispatch_tool(
        self, block: ToolUseBlock, pack: list[Card], seat_state: SeatState
    ) -> tuple[str, Card | None]:
        name = block.name
        inp = block.input

        if name == "pick_card":
            return self._tool_pick_card(inp, pack)
        elif name == "view_current_pack":
            return self._tool_view_current_pack(pack), None
        elif name == "view_my_picks":
            return self._tool_view_my_picks(seat_state, inp.get("group_by", "pick_order")), None
        elif name == "lookup_card":
            return self._tool_lookup_card(inp.get("card_name", "")), None
        elif name == "move_card":
            return (
                self._tool_move_card(
                    seat_state,
                    inp.get("card_name", ""),
                    inp.get("destination", "sideboard"),
                ),
                None,
            )
        elif name == "add_note":
            return self._tool_add_note(seat_state, inp.get("note", "")), None
        else:
            return f"Unknown tool: {name}", None

    def _find_card_in_pack(self, card_name: str, pack: list[Card]) -> Card | None:
        name_lower = card_name.lower().strip()

        # Case-insensitive exact match
        for card in pack:
            if card.name.lower() == name_lower:
                return card

        # Levenshtein fuzzy match
        best_card: Card | None = None
        best_ratio = 0.0
        for card in pack:
            r = levenshtein_ratio(name_lower, card.name.lower())
            if r > best_ratio:
                best_ratio = r
                best_card = card

        if best_ratio >= FUZZY_THRESHOLD:
            return best_card

        return None

    def _tool_pick_card(
        self, inp: dict, pack: list[Card]
    ) -> tuple[str, Card | None]:
        card_name = inp.get("card_name", "")
        card = self._find_card_in_pack(card_name, pack)

        if card is None:
            available = ", ".join(c.name for c in pack)
            return (
                f"Card '{card_name}' not found in pack. Available cards: {available}",
                None,
            )

        return f"Picked: {card.name}", card

    def _tool_view_current_pack(self, pack: list[Card]) -> str:
        if not pack:
            return "Pack is empty."
        lines = [f"Current pack ({len(pack)} cards):"]
        for i, card in enumerate(pack, 1):
            lines.append(f"{i}. {self.card_db.format_card_for_llm(card)}")
        return "\n".join(lines)

    def _tool_view_my_picks(self, seat_state: SeatState, group_by: str) -> str:
        picks = seat_state.picks
        sideboard = seat_state.sideboard

        if not picks and not sideboard:
            return "No cards drafted yet."

        lines = [
            f"Your picks ({len(picks)} cards in deck, {len(sideboard)} in sideboard):"
        ]

        if group_by == "color":
            color_groups: dict[str, list[Card]] = defaultdict(list)
            for card in picks:
                if not card.colors:
                    color_groups["Colorless"].append(card)
                else:
                    for color in card.colors:
                        color_groups[color].append(card)

            color_names = {
                "W": "White", "U": "Blue", "B": "Black", "R": "Red", "G": "Green"
            }
            for code in ["W", "U", "B", "R", "G", "Colorless"]:
                group = color_groups.get(code, [])
                if group:
                    label = color_names.get(code, code)
                    lines.append(f"\n{label} ({len(group)}):")
                    for card in group:
                        lines.append(f"  {card.name} {card.mana_cost}")

        elif group_by == "type":
            creatures = [c for c in picks if "Creature" in c.type_line]
            instants_sorcs = [
                c for c in picks if "Instant" in c.type_line or "Sorcery" in c.type_line
            ]
            lands = [c for c in picks if "Land" in c.type_line]
            other = [
                c for c in picks
                if c not in creatures and c not in instants_sorcs and c not in lands
            ]
            for group_name, group in [
                ("Creatures", creatures),
                ("Instants/Sorceries", instants_sorcs),
                ("Other Spells", other),
                ("Lands", lands),
            ]:
                if group:
                    lines.append(f"\n{group_name} ({len(group)}):")
                    for card in group:
                        lines.append(f"  {card.name} {card.mana_cost}")

        elif group_by == "cmc":
            cmc_groups: dict[str, list[Card]] = defaultdict(list)
            for card in picks:
                if "Land" in card.type_line:
                    cmc_groups["Land"].append(card)
                elif card.cmc <= 1:
                    cmc_groups["0-1"].append(card)
                elif card.cmc == 2:
                    cmc_groups["2"].append(card)
                elif card.cmc == 3:
                    cmc_groups["3"].append(card)
                elif card.cmc == 4:
                    cmc_groups["4"].append(card)
                else:
                    cmc_groups["5+"].append(card)
            for bucket in ["0-1", "2", "3", "4", "5+", "Land"]:
                group = cmc_groups.get(bucket, [])
                if group:
                    lines.append(f"\nCMC {bucket} ({len(group)}):")
                    for card in group:
                        lines.append(f"  {card.name} {card.mana_cost}")

        else:  # pick_order
            lines.append("\nPick order:")
            for i, card in enumerate(picks, 1):
                lines.append(f"  {i}. {card.name} {card.mana_cost}")

        if sideboard:
            lines.append(f"\nSideboard ({len(sideboard)}):")
            for card in sideboard:
                lines.append(f"  {card.name} {card.mana_cost}")

        return "\n".join(lines)

    def _tool_lookup_card(self, card_name: str) -> str:
        if not self._current_set_code:
            return "Set code not configured."
        results = self.card_db.search_cards(card_name, self._current_set_code)
        if not results:
            return f"No cards found matching '{card_name}' in {self._current_set_code}."
        lines = []
        for card in results[:5]:
            lines.append(self.card_db.format_card_for_llm(card))
            lines.append("")
        return "\n".join(lines).strip()

    def _tool_move_card(
        self, seat_state: SeatState, card_name: str, destination: str
    ) -> str:
        name_lower = card_name.lower().strip()
        if destination == "sideboard":
            card = next(
                (c for c in seat_state.picks if c.name.lower() == name_lower), None
            )
            if card is None:
                return f"'{card_name}' not found in your deck."
            seat_state.picks.remove(card)
            seat_state.sideboard.append(card)
            return f"Moved {card.name} to sideboard."
        elif destination == "deck":
            card = next(
                (c for c in seat_state.sideboard if c.name.lower() == name_lower), None
            )
            if card is None:
                return f"'{card_name}' not found in your sideboard."
            seat_state.sideboard.remove(card)
            seat_state.picks.append(card)
            return f"Moved {card.name} to deck."
        return f"Unknown destination: {destination}"

    def _tool_add_note(self, seat_state: SeatState, note: str) -> str:
        if not note.strip():
            return "Note is empty."
        seat_state.notes.append(note.strip())
        return f"Note saved: {note.strip()}"

    def format_pick_message(
        self,
        pack: list[Card],
        seat_state: SeatState,
        round_num: int,
        pick_num: int,
    ) -> str:
        lines = [f"=== Pack {round_num + 1}, Pick {pick_num + 1} ===", ""]

        # Notes
        if seat_state.notes:
            lines.append("Your notes:")
            for note in seat_state.notes:
                lines.append(f"  - {note}")
        else:
            lines.append("No notes yet.")
        lines.append("")

        # Compact picks summary
        picks = seat_state.picks
        lines.append(f"Cards drafted so far ({len(picks)} cards):")

        if picks:
            # Color counts
            color_counts: dict[str, int] = defaultdict(int)
            for card in picks:
                if not card.colors:
                    color_counts["Colorless"] += 1
                else:
                    for c in card.colors:
                        color_counts[c] += 1

            color_names = {
                "W": "White", "U": "Blue", "B": "Black",
                "R": "Red", "G": "Green", "Colorless": "Colorless",
            }
            color_str = ", ".join(
                f"{count} {color_names.get(c, c)}"
                for c, count in sorted(color_counts.items(), key=lambda x: -x[1])
            )
            lines.append(f"  Colors: {color_str}")

            creatures = sum(1 for c in picks if "Creature" in c.type_line)
            instants_sorcs = sum(
                1 for c in picks if "Instant" in c.type_line or "Sorcery" in c.type_line
            )
            other = len(picks) - creatures - instants_sorcs
            lines.append(
                f"  Types: {creatures} creatures, {instants_sorcs} instants/sorceries, {other} other"
            )

            # CMC curve (non-lands only)
            nonlands = [c for c in picks if "Land" not in c.type_line]
            cmc_curve: dict[str, int] = defaultdict(int)
            for card in nonlands:
                if card.cmc <= 1:
                    cmc_curve["0-1"] += 1
                elif card.cmc == 2:
                    cmc_curve["2"] += 1
                elif card.cmc == 3:
                    cmc_curve["3"] += 1
                elif card.cmc == 4:
                    cmc_curve["4"] += 1
                else:
                    cmc_curve["5+"] += 1
            curve_str = ", ".join(
                f"{k}: {v}" for k, v in sorted(cmc_curve.items()) if v > 0
            )
            lines.append(f"  CMC curve: {curve_str or 'none'}")

            # Notable rares/mythics
            notable = [c for c in picks if c.rarity in ("rare", "mythic")]
            if notable:
                lines.append(f"  Notable: {'; '.join(c.name for c in notable[:5])}")

        lines.append("")

        # Pack contents
        lines.append(f"Current pack ({len(pack)} cards remaining):")
        for i, card in enumerate(pack, 1):
            lines.append(f"{i}. {self.card_db.format_card_for_llm(card)}")
        lines.append("")
        lines.append("Make your pick.")

        return "\n".join(lines)

    def compress_round(self, round_num: int, seat_state: SeatState) -> None:
        """Replace round's conversation history with a compact summary."""
        picks_text = "; ".join(
            f"{name} ({reason[:80]})" if reason else name
            for name, reason in self._round_picks_data
        )
        notes_text = "; ".join(seat_state.notes) or "None"

        summary = (
            f"[Summary of Pack {round_num + 1}]\n"
            f"You drafted: {picks_text}\n"
            f"Your notes at end of pack: {notes_text}\n"
            f"(Use view_my_picks to see your full card pool.)"
        )

        # Replace all messages from this round's start with the summary
        self.messages = self.messages[: self._round_msg_start]
        self.messages.append({"role": "user", "content": summary})

    async def build_deck(self, seat_state: SeatState) -> None:
        """Run the deckbuilding phase after drafting is complete."""
        pool_size = len(seat_state.picks)
        lands_in_pool = sum(1 for c in seat_state.picks if "Land" in c.type_line)
        spells_in_pool = pool_size - lands_in_pool

        user_msg = (
            f"=== Deckbuilding Phase ===\n\n"
            f"The draft is complete. You have {pool_size} cards in your pool "
            f"({spells_in_pool} spells, {lands_in_pool} non-basic lands).\n\n"
            f"Your goal is to build a 40-card main deck:\n"
            f"  • Cut weaker cards to sideboard with move_card\n"
            f"  • Add basic lands with add_basic_land (aim for ~17 lands total)\n"
            f"  • Review your pool with view_my_picks\n"
            f"  • Call finalize_deck when satisfied with your 40-card deck"
        )
        self.messages.append({"role": "user", "content": user_msg})

        for iteration in range(MAX_DECKBUILD_ITERATIONS):
            tool_choice = "any" if iteration == MAX_DECKBUILD_ITERATIONS - 1 else "auto"

            response = await self.provider.send_message(
                system=self.system_prompt,
                messages=self.messages,
                tools=DECKBUILDING_TOOLS,
                tool_choice=tool_choice,
                max_tokens=2048,
            )

            self.total_api_calls += 1
            self.total_input_tokens += response.usage.get("input_tokens", 0)
            self.total_output_tokens += response.usage.get("output_tokens", 0)

            assistant_content: list[dict] = []
            for block in response.content:
                if isinstance(block, TextBlock):
                    assistant_content.append({"type": "text", "text": block.text})
                elif isinstance(block, ToolUseBlock):
                    assistant_content.append(
                        {
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        }
                    )
            self.messages.append({"role": "assistant", "content": assistant_content})

            tool_results: list[dict] = []
            is_done = False
            for block in response.content:
                if isinstance(block, ToolUseBlock):
                    result, done = self._dispatch_deckbuild_tool(block, seat_state)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        }
                    )
                    if done:
                        is_done = True

            if tool_results:
                self.messages.append({"role": "user", "content": tool_results})

            if is_done:
                logger.info("Deckbuilding finalized.")
                return

            if response.stop_reason == "end_turn":
                deck_size = len(seat_state.picks)
                lands_now = sum(1 for c in seat_state.picks if "Land" in c.type_line)
                self.messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Your deck currently has {deck_size} cards ({lands_now} lands). "
                            f"Call finalize_deck when you are satisfied with your 40-card deck."
                        ),
                    }
                )

        logger.warning(
            f"Deckbuilding did not finalize after {MAX_DECKBUILD_ITERATIONS} iterations."
        )

    def _dispatch_deckbuild_tool(
        self, block: ToolUseBlock, seat_state: SeatState
    ) -> tuple[str, bool]:
        name = block.name
        inp = block.input

        if name == "finalize_deck":
            notes = inp.get("notes", "")
            if notes:
                seat_state.notes.append(f"[Deckbuilding] {notes}")
            deck_size = len(seat_state.picks)
            lands_now = sum(1 for c in seat_state.picks if "Land" in c.type_line)
            return (
                f"Deck finalized: {deck_size} cards "
                f"({lands_now} lands, {deck_size - lands_now} spells).",
                True,
            )
        elif name == "add_basic_land":
            land_type = inp.get("land_type", "")
            count = max(1, int(inp.get("count", 1)))
            return self._tool_add_basic_land(seat_state, land_type, count), False
        elif name == "remove_basic_land":
            land_type = inp.get("land_type", "")
            count = max(1, int(inp.get("count", 1)))
            return self._tool_remove_basic_land(seat_state, land_type, count), False
        elif name in ("view_my_picks", "lookup_card", "move_card", "add_note"):
            result, _ = self._dispatch_tool(block, [], seat_state)
            return result, False
        else:
            return f"Unknown tool: {name}", False

    def _tool_add_basic_land(
        self, seat_state: SeatState, land_type: str, count: int
    ) -> str:
        if land_type not in BASIC_LAND_TYPES:
            return (
                f"Invalid land type '{land_type}'. "
                f"Must be one of: {', '.join(BASIC_LAND_TYPES)}"
            )
        for _ in range(count):
            seat_state.picks.append(_make_basic_land(land_type))
        lands_now = sum(1 for c in seat_state.picks if "Land" in c.type_line)
        return (
            f"Added {count} {land_type}{'s' if count > 1 else ''}. "
            f"Deck: {len(seat_state.picks)} cards, {lands_now} lands."
        )

    def _tool_remove_basic_land(
        self, seat_state: SeatState, land_type: str, count: int
    ) -> str:
        if land_type not in BASIC_LAND_TYPES:
            return (
                f"Invalid land type '{land_type}'. "
                f"Must be one of: {', '.join(BASIC_LAND_TYPES)}"
            )
        removed = 0
        for _ in range(count):
            card = next(
                (
                    c for c in reversed(seat_state.picks)
                    if c.name == land_type and "Basic Land" in c.type_line
                ),
                None,
            )
            if card is None:
                break
            seat_state.picks.remove(card)
            removed += 1
        if removed == 0:
            return f"No {land_type}s found in your deck to remove."
        lands_now = sum(1 for c in seat_state.picks if "Land" in c.type_line)
        msg = f"Removed {removed} {land_type}{'s' if removed > 1 else ''}."
        if removed < count:
            msg += f" (Only {removed} found.)"
        msg += f" Deck: {len(seat_state.picks)} cards, {lands_now} lands."
        return msg
