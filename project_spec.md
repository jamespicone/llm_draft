# MTG Draft LLM Harness — Project Specification

## Purpose

Build a Python application that evaluates how well LLMs can draft Magic: The Gathering. The system runs a complete 8-player booster draft where one seat is controlled by an LLM via tool calling, and the other seven seats are controlled by algorithmic bots. The LLM receives no human-derived card evaluations (no 17Lands data, no tier lists). It must draft purely from card text, set mechanics knowledge, and its own strategic reasoning. 17Lands data is used only for bot pick logic and post-hoc evaluation of the LLM's picks.

The system should be set-agnostic: given a set code (e.g. `DSK`, `MKM`, `BLB`), it downloads the necessary data and runs a draft for that set.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────┐
│                    main.py (CLI)                     │
│  Parses args, orchestrates setup and draft execution │
└──────────────┬──────────────────────────┬───────────┘
               │                          │
    ┌──────────▼──────────┐    ┌──────────▼──────────┐
    │   Draft Engine       │    │   LLM Harness        │
    │ (draft_engine.py)    │    │ (llm_harness.py)     │
    │                      │    │                      │
    │ - Pack generation    │    │ - Tool definitions   │
    │ - Seat management    │    │ - Tool dispatch loop │
    │ - Pack rotation      │    │ - Context management │
    │ - Bot pick logic     │    │ - System prompt      │
    └──────────┬──────────┘    └──────────┬───────────┘
               │                          │
    ┌──────────▼──────────┐    ┌──────────▼──────────┐
    │   Card Database      │    │   LLM Provider       │
    │ (card_database.py)   │    │ (llm_provider.py)    │
    │                      │    │                      │
    │ - Scryfall bulk data │    │ - Anthropic SDK      │
    │ - MTGJSON booster    │    │ - (OpenAI optional)  │
    │   composition data   │    │ - API abstraction    │
    │ - Card lookup/search │    │                      │
    └─────────────────────┘    └──────────────────────┘
               │
    ┌──────────▼──────────┐
    │   17Lands Ratings    │
    │ (ratings.py)         │
    │                      │
    │ - GIH WR per card    │
    │ - Used by bots ONLY  │
    │ - Used for evaluation│
    └─────────────────────┘

    ┌─────────────────────┐
    │   Evaluation         │
    │ (evaluation.py)      │
    │                      │
    │ - Pick-by-pick log   │
    │ - Accuracy metrics   │
    │ - Draft report       │
    └─────────────────────┘
```

---

## Component Specifications

### 1. Card Database (`card_database.py`)

**Responsibilities**: Download, cache, and serve card data for any requested set.

**Data sources**:

- **Scryfall bulk data**: Download the `oracle_cards` bulk file from `https://api.scryfall.com/bulk-data/oracle-cards`. This gives one entry per unique card (not per printing). Cache locally at `~/.mtg-draft-harness/scryfall_oracle_cards.json`. Re-download if the file is older than 7 days. Include a `User-Agent` header identifying the application, per Scryfall's API guidelines. The Scryfall bulk data is used for card text, types, mana costs, and other oracle-level information available to the LLM.

- **Scryfall set-specific data**: For set-specific card data (collector numbers, rarities, set membership), use the Scryfall search API: `GET https://api.scryfall.com/cards/search?q=set:{SET_CODE}+is:booster` with pagination. This gives all cards that appear in booster packs for a given set. Cache per-set at `~/.mtg-draft-harness/sets/{SET_CODE}/scryfall_cards.json`. Respect Scryfall's rate limit of 10 requests per second (add a 100ms delay between paginated requests).

- **MTGJSON set data**: Download the set JSON from `https://mtgjson.com/api/v5/{SET_CODE}.json`. This contains the `booster` object with sheet and weight configurations needed for pack generation. Cache at `~/.mtg-draft-harness/sets/{SET_CODE}/mtgjson.json`. Re-download if older than 7 days.

**Card representation**:

```python
@dataclass
class Card:
    name: str
    mana_cost: str            # e.g. "{2}{B}{B}"
    cmc: float                # converted mana cost
    type_line: str            # e.g. "Creature — Zombie Knight"
    oracle_text: str          # rules text
    colors: list[str]         # e.g. ["B"]
    color_identity: list[str] # e.g. ["B"]
    rarity: str               # "common", "uncommon", "rare", "mythic"
    power: str | None         # e.g. "3" or None
    toughness: str | None     # e.g. "4" or None
    set_code: str             # e.g. "DSK"
    collector_number: str     # e.g. "42"
    scryfall_id: str          # UUID
    card_faces: list[dict] | None  # for DFCs, split cards, adventures
    keywords: list[str]       # e.g. ["Flying", "Deathtouch"]
```

For double-faced cards, split cards, and adventure cards: store the full `card_faces` data from Scryfall. When presenting cards to the LLM, format both faces. Top-level `oracle_text`, `mana_cost`, etc. may be empty for multi-face cards — always check `card_faces` and merge the information.

**Public interface**:

```python
class CardDatabase:
    def __init__(self, cache_dir: str = "~/.mtg-draft-harness")
    async def ensure_set_data(self, set_code: str) -> None
    def get_card_by_name(self, name: str, set_code: str) -> Card | None
    def get_cards_for_set(self, set_code: str) -> list[Card]
    def search_cards(self, query: str, set_code: str) -> list[Card]
    def get_set_mechanics(self, set_code: str) -> list[str]  # keywords in set
    def format_card_for_llm(self, card: Card) -> str  # compact text repr
```

**`format_card_for_llm`** should produce a compact but complete representation:

```
[R] Sheoldred, the Apocalypse {2}{B}{B}
    Legendary Creature — Phyrexian Praetor (4/5)
    Deathtouch
    Whenever you draw a card, you gain 2 life.
    Whenever an opponent draws a card, they lose 2 life.
```

Format: `[{rarity initial}] {name} {mana_cost}\n    {type_line} ({P}/{T} if creature)\n    {oracle_text, each ability on new indented line}`

Rarity initials: C = common, U = uncommon, R = rare, M = mythic.

### 2. Pack Generation (within `card_database.py` or separate `pack_generator.py`)

**Responsibilities**: Generate booster packs that match the set's actual booster composition.

Use the MTGJSON `booster` data for the set. The booster configuration contains:

- `boosters`: an array of possible pack structures, each with a `contents` map (sheet name → count) and a `weight`.
- `sheets`: a map of sheet names to `{ cards: {uuid: weight}, totalWeight, foil, balanceColors }`.

**Pack generation algorithm**:

1. Select a pack structure by weighted random choice from `boosters` (using `boostersTotalWeight`).
2. For each sheet in the selected structure's `contents`, pick the required number of cards from that sheet using weighted random selection (card UUID → weight, total weight from `totalWeight`).
3. Map MTGJSON UUIDs to Scryfall card data. MTGJSON UUIDs are their own format — match cards by `name` + `set_code` + `collector_number` between MTGJSON's `cards` array and Scryfall data. Build this mapping during `ensure_set_data`.
4. If `balanceColors` is true on a sheet, ensure the selected cards include at least one card of each mono colour (W, U, B, R, G) where possible. Retry selection up to 10 times to achieve this.
5. Cards should not repeat within a single pack.

**Fallback**: If MTGJSON booster data is unavailable or unparseable for a set, fall back to a simple heuristic: 10 commons, 3 uncommons, 1 rare (7/8 chance) or mythic (1/8 chance), 1 basic land. Select randomly by rarity from the set's card pool.

**Pack generation should produce 24 packs total**: 3 packs per seat × 8 seats. Generate all packs up front before the draft begins. Store as `list[list[list[Card]]]` — `packs[pack_number][seat_index]` → list of Card.

### 3. Draft Engine (`draft_engine.py`)

**Responsibilities**: Manage the draft state machine — pack distribution, pick tracking, pack rotation, and bot logic.

**Draft parameters**:

- 8 seats (configurable, but default 8)
- 3 rounds of packs
- Pack 1 passes left (seat index +1), Pack 2 passes right (seat index -1), Pack 3 passes left (+1)
- The LLM controls seat 0
- Seats 1-7 are bots

**State**:

```python
@dataclass
class SeatState:
    picks: list[Card]           # all cards drafted, in pick order
    sideboard: list[Card]       # cards moved to sideboard by the LLM
    notes: list[str]            # strategic notes (LLM only)
    color_affinity: dict[str, float]  # bots only: running color preference

@dataclass  
class DraftState:
    set_code: str
    seats: list[SeatState]      # index 0 = LLM
    packs: list[list[list[Card]]]  # packs[round][seat] = remaining cards
    current_round: int          # 0, 1, 2
    current_pick: int           # 0 through (pack_size - 1)
    pack_size: int              # typically 14 (15 minus basic land) or 15
```

**Draft loop** (pseudocode):

```
for round in 0..2:
    direction = +1 if round is even else -1
    for pick in 0..(pack_size - 1):
        # LLM picks from seat 0's current pack
        llm_pack = packs[round][0]
        picked_card = await llm_harness.make_pick(llm_pack, seats[0], round, pick)
        remove picked_card from llm_pack
        add picked_card to seats[0].picks

        # Bots pick from their respective packs
        for seat_idx in 1..7:
            bot_pack = packs[round][seat_idx]
            picked_card = bot_pick(bot_pack, seats[seat_idx])
            remove picked_card from bot_pack
            add picked_card to seats[seat_idx].picks

        # Rotate packs
        if direction == +1:  # pass left
            packs[round] = [packs[round][-1]] + packs[round][:-1]
            # i.e. each seat's pack goes to the next seat
        else:  # pass right
            packs[round] = packs[round][1:] + [packs[round][0]]
```

Wait — the rotation here needs care. "Pass left" means seat N passes to seat N+1. After rotation, seat 0 should receive what was seat 7's pack (in an 8-seat draft passing left). Verify the rotation logic with a concrete example and add a unit test for it.

**Bot pick logic**:

Bots use a score-based system combining 17Lands GIH WR with colour affinity:

```python
def bot_pick(pack: list[Card], seat: SeatState, ratings: dict[str, float]) -> Card:
    def score(card: Card) -> float:
        base = ratings.get(card.name, 0.50)  # default to 50% if no data
        
        # Colour affinity bonus: increases as bot commits to colours
        affinity_bonus = 0.0
        if seat.picks:  # skip for first pick
            card_colors = card.colors or []
            if not card_colors:
                # Colourless cards: small flat bonus (always playable)
                affinity_bonus = 0.01
            else:
                # Average affinity across the card's colours
                affinity_bonus = sum(
                    seat.color_affinity.get(c, 0.0) for c in card_colors
                ) / len(card_colors)
        
        return base + affinity_bonus * 0.15  # tune this weight
    
    best_card = max(pack, key=score)
    
    # Update colour affinity after picking
    for color in best_card.colors or []:
        seat.color_affinity[color] = seat.color_affinity.get(color, 0.0) + 1.0
    
    return best_card
```

The `0.15` weight on `affinity_bonus` means a card in the bot's primary colour (say, affinity 10 after many picks) gets `10 * 0.15 = 1.5` percentage points of bonus. This is a tunable parameter. The effect is that bots start mostly picking on raw card quality, then increasingly prefer on-colour cards as they commit, which produces realistic signal dynamics.

If 17Lands data is unavailable for the set, fall back to a simple rarity-based rating: mythic = 0.62, rare = 0.58, uncommon = 0.54, common = 0.52. This is crude but ensures the system works for any set.

### 4. 17Lands Ratings (`ratings.py`)

**Responsibilities**: Fetch and serve per-card GIH WR data for a set. Used by bot logic and evaluation only — never exposed to the LLM.

**Data source**: 17Lands provides a card ratings endpoint. The URL pattern used by community tools is:

```
https://www.17lands.com/card_ratings/data?expansion={SET_CODE}&format=PremierDraft
```

This returns JSON with per-card statistics including `ever_drawn_win_rate` (GIH WR). However, this endpoint is undocumented and may be rate-limited or require specific headers.

**Fallback approach**: If the API endpoint is unreliable, 17Lands publishes public datasets (CSV) at `https://www.17lands.com/public_datasets`. These contain per-game data from which GIH WR can be computed, but they are large files. For simplicity, implement the API approach first, with the rarity-based fallback if it fails.

**Interface**:

```python
class Ratings:
    def __init__(self, cache_dir: str = "~/.mtg-draft-harness")
    async def ensure_ratings(self, set_code: str) -> None
    def get_rating(self, card_name: str, set_code: str) -> float | None
    def get_all_ratings(self, set_code: str) -> dict[str, float]
```

Cache ratings at `~/.mtg-draft-harness/sets/{SET_CODE}/17lands_ratings.json`. Re-download if older than 24 hours (ratings stabilise quickly but do shift early in a set's lifecycle).

### 5. LLM Harness (`llm_harness.py`)

**Responsibilities**: Manage the LLM conversation, define tools, execute the tool-calling loop, and extract pick decisions.

#### 5.1 LLM Provider Abstraction (`llm_provider.py`)

Support Anthropic's Claude API as the primary provider. Optionally support OpenAI's API behind a common interface.

```python
class LLMProvider(Protocol):
    async def send_message(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict],
        tool_choice: dict | str,
        max_tokens: int,
    ) -> LLMResponse

@dataclass
class LLMResponse:
    content: list[ContentBlock]  # text blocks and tool_use blocks
    stop_reason: str             # "end_turn", "tool_use", etc.
    usage: dict                  # input_tokens, output_tokens

@dataclass
class TextBlock:
    text: str

@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict
```

For Anthropic, use the `anthropic` Python SDK. The model should be configurable (default: `claude-sonnet-4-5-20250929`). The API key comes from the `ANTHROPIC_API_KEY` environment variable.

For OpenAI (optional, lower priority), use the `openai` Python SDK. Translate tool definitions between the two formats (Anthropic uses `input_schema`, OpenAI wraps in `{"type": "function", "function": {...}}` with `parameters`). The model should be configurable (default: `gpt-4o`). The API key comes from the `OPENAI_API_KEY` environment variable.

#### 5.2 Tool Definitions

Define six tools available to the LLM:

**`pick_card`** (required action — every pick must end with this)

```json
{
    "name": "pick_card",
    "description": "Pick a card from the current booster pack to add to your drafted card pool. You must call this exactly once per pack to make your selection. This is irreversible.",
    "input_schema": {
        "type": "object",
        "properties": {
            "card_name": {
                "type": "string",
                "description": "The exact name of the card to pick, as shown in the pack contents."
            },
            "reasoning": {
                "type": "string",
                "description": "Brief explanation of why you chose this card (1-3 sentences)."
            }
        },
        "required": ["card_name", "reasoning"]
    }
}
```

**`view_current_pack`**

```json
{
    "name": "view_current_pack",
    "description": "View the full details of all cards currently in the pack you are picking from. Returns each card with its complete oracle text, type, mana cost, and rarity.",
    "input_schema": {
        "type": "object",
        "properties": {},
        "required": []
    }
}
```

**`view_my_picks`**

```json
{
    "name": "view_my_picks",
    "description": "View all cards you have drafted so far, organised by category. Shows your current deck and sideboard, grouped by colour and card type, with mana curve information.",
    "input_schema": {
        "type": "object",
        "properties": {
            "group_by": {
                "type": "string",
                "enum": ["color", "type", "cmc", "pick_order"],
                "description": "How to group the cards. 'color' groups by card colour, 'type' by card type (creature/noncreature/land), 'cmc' by mana value, 'pick_order' shows cards in the order drafted."
            }
        },
        "required": ["group_by"]
    }
}
```

**`lookup_card`**

```json
{
    "name": "lookup_card",
    "description": "Look up the full details of any card in the current set by name. Use this to check what a card does if you are unsure. Supports partial name matching.",
    "input_schema": {
        "type": "object",
        "properties": {
            "card_name": {
                "type": "string",
                "description": "The name (or partial name) of the card to look up."
            }
        },
        "required": ["card_name"]
    }
}
```

**`move_card`**

```json
{
    "name": "move_card",
    "description": "Move a card between your main deck and sideboard. Use this for cards you have hate-drafted or do not expect to play in your final deck.",
    "input_schema": {
        "type": "object",
        "properties": {
            "card_name": {
                "type": "string",
                "description": "The exact name of the card to move."
            },
            "destination": {
                "type": "string",
                "enum": ["sideboard", "deck"],
                "description": "Where to move the card. 'sideboard' to sideboard it, 'deck' to move it back to the main deck."
            }
        },
        "required": ["card_name", "destination"]
    }
}
```

**`add_note`**

```json
{
    "name": "add_note",
    "description": "Save a strategic note for yourself. These notes persist across picks and are shown to you at the start of each pick. Use this to track your draft plan, colour commitment, cards you are looking for, signals you have read, or anything else you want to remember.",
    "input_schema": {
        "type": "object",
        "properties": {
            "note": {
                "type": "string",
                "description": "The note text. Keep it concise — a sentence or two."
            }
        },
        "required": ["note"]
    }
}
```

#### 5.3 System Prompt

The system prompt should be stored in a separate file (`system_prompt.txt` or similar) for easy iteration. It should contain:

1. **Role and objective**: "You are an expert Magic: The Gathering drafter. Your goal is to draft the strongest possible 40-card limited deck from the cards available to you."

2. **Draft rules**: Explain the booster draft format — 3 packs of N cards, pick 1 card per pack and pass the rest, packs 1 and 3 pass left, pack 2 passes right, goal is to build a 40-card deck with approximately 17 lands and 23 nonland cards.

3. **Strategic guidance** (keep this concise):
   - Prioritise removal spells, bombs, and efficient creatures
   - Stay open in pack 1; read signals from late-pick quality cards
   - Commit to two colours (possibly splashing a third) by mid-pack 2
   - Value mana curve: need a good distribution of 2, 3, and 4-drops
   - Creature count: aim for 14-18 creatures in most decks
   - Synergy matters: cards that work together are worth more than individually strong cards that don't fit your deck

4. **Set-specific context**: Dynamically generated from the card database. Include the set's keyword mechanics and a brief description of the set's colour-pair archetypes if known. This section is generated at runtime.

5. **Tool usage instructions**: "You have tools to view the pack, view your picks, look up cards, save notes, and make your pick. You MUST call pick_card exactly once to make your selection. You may call other tools first to gather information. Use add_note to record your draft strategy — your notes are shown to you each pick."

6. **Format of the set archetypes**: If the set's archetype data can be determined (from Scryfall's set data or a hardcoded map for popular sets), include it. Otherwise omit it — the LLM should be able to infer archetypes from the cards it sees.

#### 5.4 Per-Pick User Message

At the start of each pick, construct a user message containing:

```
=== Pack {round+1}, Pick {pick+1} ===

Your notes:
{newline-joined list of notes, or "No notes yet."}

Cards drafted so far ({len(picks)} cards):
{compact summary: colour counts, creature count, average CMC, notable cards}

Current pack ({len(pack)} cards remaining):
{each card formatted with format_card_for_llm, numbered 1-N}

Make your pick.
```

The "cards drafted so far" summary should be compact — not the full card list. Something like:

```
Cards drafted so far (12 cards):
  Colours: 5 Black, 4 Blue, 2 Red, 1 Colourless
  Types: 7 creatures, 3 instants/sorceries, 2 other
  CMC curve: 0-1: 1, 2: 3, 3: 4, 4: 2, 5+: 2
  Notable: Sheoldred, the Apocalypse; Go for the Throat; Preacher of the Schism
```

The LLM can use `view_my_picks` if it wants the full list.

#### 5.5 Tool-Calling Loop

```python
async def make_pick(self, pack, seat_state, round_num, pick_num) -> Card:
    # Build the user message
    user_msg = self.format_pick_message(pack, seat_state, round_num, pick_num)
    
    # Append to conversation history
    self.messages.append({"role": "user", "content": user_msg})
    
    max_iterations = 15  # safety limit
    for iteration in range(max_iterations):
        response = await self.provider.send_message(
            system=self.system_prompt,
            messages=self.messages,
            tools=self.tools,
            tool_choice="any" if iteration == max_iterations - 1 else "auto",
            max_tokens=2048,
        )
        
        # Append assistant response to history
        self.messages.append({"role": "assistant", "content": response.content})
        
        # Process each content block
        tool_results = []
        picked_card = None
        
        for block in response.content:
            if isinstance(block, ToolUseBlock):
                result, card = self.dispatch_tool(block, pack, seat_state)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })
                if card is not None:
                    picked_card = card
        
        if tool_results:
            self.messages.append({"role": "user", "content": tool_results})
        
        if picked_card is not None:
            return picked_card
        
        if response.stop_reason == "end_turn" and picked_card is None:
            # LLM responded with text but didn't pick — prompt it to pick
            self.messages.append({
                "role": "user",
                "content": "You need to pick a card. Please call the pick_card tool."
            })
    
    # Fallback: pick the first card in the pack
    logging.warning(f"LLM failed to pick after {max_iterations} iterations. Picking first card.")
    return pack[0]
```

**Tool dispatch**: Match the tool name, validate inputs, execute, return a string result. For `pick_card`, validate that `card_name` matches a card in the current pack. Use fuzzy matching (case-insensitive, strip punctuation) since LLMs sometimes slightly misname cards. If no match, return an error message naming the available cards so the LLM can retry.

#### 5.6 Context Management

Over 45 picks, the conversation history grows large. Implement a context compression strategy:

- **Between rounds** (after picks 1-N of pack 1, before pack 2 starts; and after pack 2 before pack 3): Truncate the conversation history. Replace all the per-pick messages from the completed round with a single summary message:
  ```
  [Summary of Pack {N}]
  You drafted the following cards: {list of card names with brief reasons}
  Your notes at end of pack: {notes}
  ```
  This preserves the essential information (what was picked and why) without the full pack contents and tool-call transcripts.

- **Within a round**: Keep the full conversation. A single round is at most ~15 picks, each with a few hundred tokens of pack content plus tool calls. This should stay well within context limits.

- **Track token usage**: Log input and output token counts from each API call. Sum them per draft for cost reporting.

### 6. Evaluation (`evaluation.py`)

**Responsibilities**: Log every pick with context, compute accuracy metrics, and produce a draft report.

**Pick log**: For each pick, record:

```python
@dataclass
class PickRecord:
    round_num: int              # 0-2
    pick_num: int               # 0-N
    pack_contents: list[str]    # card names in pack
    picked_card: str            # card name chosen
    reasoning: str              # LLM's stated reasoning
    llm_tool_calls: int         # number of tool calls before picking
    notes_at_time: list[str]    # LLM's notes at time of pick
    
    # Evaluation fields (filled in post-hoc)
    card_ratings: dict[str, float]  # 17Lands GIH WR for each card in pack
    best_available: str         # highest-rated card in pack by GIH WR
    pick_was_best: bool         # did the LLM pick the highest-rated card?
    pick_rank_in_pack: int      # rank of picked card among pack by GIH WR (1 = best)
```

**Metrics** to compute after the draft:

- **Top-1 accuracy**: Fraction of picks where the LLM chose the highest GIH WR card in the pack.
- **Top-3 accuracy**: Fraction of picks where the LLM's choice was among the top 3 by GIH WR.
- **Average pick rank**: Mean rank of the LLM's picks within each pack (1.0 = always picked the best card).
- **Colour coherence**: What fraction of the LLM's picks share its two most-drafted colours? Computed as: count of picks matching the top 2 colours / total picks (excluding colourless).
- **Mana curve score**: How well does the final deck's mana curve match a standard limited curve (roughly: 1-2 one-drops, 4-5 two-drops, 4-5 three-drops, 3-4 four-drops, 2-3 five+). Compute as negative mean squared deviation from an ideal curve — details are secondary, just implement something reasonable.
- **Total API cost**: Sum of token costs across all API calls, computed from token counts and per-model pricing.
- **Total API calls**: Number of messages.create calls made during the draft.

**Draft report**: Output as both a JSON file (machine-readable, containing all PickRecords and aggregate metrics) and a human-readable text/markdown summary. The summary should show:

- Each pick: "P1P1: [pack contents...] → Picked {card} (rank {N}/{total}, reason: {reason})"
- Aggregate metrics at the end
- Final deck list (main deck and sideboard)
- Total cost and API call count

Write the JSON log to `./drafts/{timestamp}_{set_code}.json` and the summary to `./drafts/{timestamp}_{set_code}.md`.

### 7. CLI (`main.py`)

**Usage**:

```
python main.py draft --set DSK [options]

Required:
  --set SET_CODE          Set code to draft (e.g. DSK, MKM, BLB)

Options:
  --provider anthropic|openai    LLM provider (default: anthropic)
  --model MODEL_NAME             Model to use (default: provider-dependent)
  --seats N                      Number of seats (default: 8)
  --output-dir DIR               Output directory for logs (default: ./drafts)
  --verbose                      Print detailed output during draft
  --dry-run                      Generate packs and print them without running LLM
  --seed N                       Random seed for reproducible pack generation
```

**Workflow**:

1. Parse arguments
2. Initialise CardDatabase, ensure set data is downloaded
3. Initialise Ratings, ensure 17Lands data is downloaded
4. Generate packs
5. Build system prompt with set-specific context
6. Initialise LLM harness
7. Run draft loop
8. Compute evaluation metrics
9. Write output files
10. Print summary to stdout

---

## Project Structure

```
mtg-draft-harness/
├── main.py                  # CLI entry point
├── card_database.py         # Scryfall + MTGJSON data management
├── pack_generator.py        # Booster pack generation from MTGJSON data
├── draft_engine.py          # Draft state machine, bot logic, pack rotation
├── llm_harness.py           # Tool definitions, tool dispatch, pick loop
├── llm_provider.py          # Anthropic/OpenAI API abstraction
├── ratings.py               # 17Lands GIH WR data
├── evaluation.py            # Pick logging, metrics, report generation
├── system_prompt.txt        # System prompt template (with {set_context} placeholder)
├── requirements.txt         # anthropic, httpx (or aiohttp), python-Levenshtein (for fuzzy matching)
├── tests/
│   ├── test_pack_generator.py   # Pack composition tests
│   ├── test_draft_engine.py     # Rotation logic, bot picks
│   ├── test_llm_harness.py      # Tool dispatch, fuzzy matching
│   └── test_evaluation.py       # Metric calculations
└── README.md
```

---

## Dependencies

```
anthropic>=0.40.0
openai>=1.50.0        # optional, for OpenAI provider
httpx>=0.27.0         # async HTTP client for data downloads
python-Levenshtein>=0.25.0   # fuzzy card name matching (or thefuzz)
```

Use Python 3.11+ (for `asyncio` improvements and type syntax).

All code should use `async`/`await` for I/O operations (API calls, file downloads). The draft loop itself is sequential (one pick at a time), but downloads and API calls benefit from async.

---

## Key Design Decisions and Constraints

1. **The LLM gets NO 17Lands data.** No win rates, no pick rates, no tier lists. The LLM drafts purely from card text, set mechanics, and its own knowledge of Magic. This is the core experimental constraint.

2. **17Lands data drives bot behaviour and evaluation.** Bots pick using GIH WR + colour affinity. Evaluation compares LLM picks against the GIH WR ranking.

3. **The LLM can look up any card in the set via `lookup_card`.** This simulates having access to a spoiler/card database, which a human drafter would also have. The LLM can look up cards it remembers from previous packs, cards referenced in synergy with its picks, etc.

4. **Notes are the LLM's external memory.** Since conversation history is compressed between rounds, notes are the LLM's persistent strategic state. Encourage their use in the system prompt.

5. **Bot strategy is pluggable.** The bot pick function takes `(pack, seat_state, ratings)` and returns a `Card`. This can be swapped out for more sophisticated strategies later.

6. **Fuzzy card name matching is important.** LLMs frequently produce slight variations on card names (wrong capitalisation, missing commas, "the" vs "The", split card name conventions). Implement case-insensitive matching first, then Levenshtein distance as a fallback, with a similarity threshold of 0.85.

7. **Graceful degradation.** If 17Lands data is unavailable, bots fall back to rarity-based ratings. If MTGJSON booster data is unavailable, fall back to simple rarity-based pack generation. If the LLM fails to pick after max iterations, pick the first card. Log all fallbacks.

---

## Testing Strategy

Write unit tests for:

- **Pack generation**: Verify that generated packs have the correct number of cards, correct rarity distribution (within statistical expectation), and no duplicate cards within a pack.
- **Pack rotation**: Verify that after rotation, seat 0 receives the pack that was at seat 7 (passing left) or seat 1 (passing right). Verify all 8 seats see all 8 original packs over the course of a round.
- **Bot pick logic**: Verify that bots pick the highest-rated card when they have no colour preference, and prefer on-colour cards as affinity builds.
- **Tool dispatch**: Verify that `pick_card` with a valid name returns the correct card, that fuzzy matching works for common misspellings, and that invalid names return an error.
- **Evaluation metrics**: Verify top-1 and top-3 accuracy calculations with known pick logs.

Integration tests (require API key):

- **Single-pick test**: Present a hardcoded pack to the LLM and verify it calls `pick_card` with a valid card name.
- **Full draft test**: Run a complete draft and verify the output log is well-formed.

---

## Future Extensions (Out of Scope for Initial Implementation)

- Multiple LLM seats (LLM vs LLM drafting)
- Deck building phase (the LLM decides which 23 cards to play and which to sideboard)
- Game simulation (evaluate draft quality by playing games)
- Batch evaluation (run N drafts and aggregate metrics)
- Web UI for viewing draft logs
- Support for Cube drafts (custom card lists)
- Signal analysis (track what the LLM infers about open colours vs what is actually open)