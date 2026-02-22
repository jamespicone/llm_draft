# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MTG Draft LLM Harness — evaluates how well LLMs can draft Magic: The Gathering. One LLM seat (seat 0) runs a complete 8-player booster draft against 7 algorithmic bots. The LLM drafts purely from card text with no win-rate data; 17Lands data is used only for bot logic and post-hoc evaluation.

## Commands

```bash
# Run a draft
python main.py draft --set DSK [--provider anthropic|openai] [--model MODEL] [--seats N] [--verbose] [--dry-run] [--seed N] [--output-dir DIR]

# Run tests
python -m pytest tests/

# Run a single test file
python -m pytest tests/test_pack_generator.py

# Run a single test
python -m pytest tests/test_draft_engine.py::test_pack_rotation_left
```

## Architecture

```
main.py → draft_engine.py + llm_harness.py
                ↓                   ↓
         card_database.py    llm_provider.py
         pack_generator.py   (Anthropic/OpenAI)
         ratings.py (bots + eval only)
         evaluation.py
```

**Critical constraint**: 17Lands GIH WR data must NEVER be exposed to the LLM. It flows only to `draft_engine.py` (bot picks) and `evaluation.py` (post-hoc scoring).

## Key Components

**`card_database.py`** — Downloads and caches Scryfall oracle bulk data (`~/.mtg-draft-harness/scryfall_oracle_cards.json`, refreshed every 7 days) and per-set Scryfall data. Provides `format_card_for_llm()` which renders cards as `[R] Name {cost}\n    Type (P/T)\n    Oracle text`.

**`pack_generator.py`** — Generates packs from MTGJSON booster sheet/weight configuration. Falls back to simple rarity heuristic (10C/3U/1R-or-M/1 land) if MTGJSON data is absent. Generates all 24 packs (3 rounds × 8 seats) up front as `list[list[list[Card]]]` indexed `[round][seat]`.

**`draft_engine.py`** — Pack rotation: pack 1 and 3 pass left (`seat N → seat N+1`), pack 2 passes right. After rotation, seat 0 receives what was seat 7's pack (passing left). Bot scoring: `GIH_WR + color_affinity * 0.15`. Fallback ratings: mythic=0.62, rare=0.58, uncommon=0.54, common=0.52.

**`llm_harness.py`** — Tool-calling loop (max 15 iterations per pick). Six tools: `pick_card` (required once per pick), `view_current_pack`, `view_my_picks`, `lookup_card`, `move_card`, `add_note`. Card name validation uses fuzzy matching (case-insensitive → Levenshtein with 0.85 threshold). Context compression between rounds: replace per-pick history with a single summary of picks + reasons.

**`llm_provider.py`** — `LLMProvider` Protocol with Anthropic (primary) and OpenAI (optional) implementations. Default models: `claude-sonnet-4-5-20250929` / `gpt-4o`. API keys from `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` env vars.

**`ratings.py`** — Fetches `https://www.17lands.com/card_ratings/data?expansion={SET_CODE}&format=PremierDraft`. Cached at `~/.mtg-draft-harness/sets/{SET_CODE}/17lands_ratings.json`, refreshed every 24 hours.

**`evaluation.py`** — Records `PickRecord` per pick (pack contents, card picked, reasoning, tool call count, notes). Computes: top-1/top-3 accuracy, average pick rank, colour coherence, mana curve score, token costs. Writes `./drafts/{timestamp}_{set_code}.json` and `.md`.

**`system_prompt.txt`** — Template with `{set_context}` placeholder for set mechanics/archetypes injected at runtime.

## Data Flow

1. Scryfall bulk data → card oracle text (LLM-visible)
2. Scryfall set search → set membership, rarities
3. MTGJSON set JSON → booster sheet/weight config for pack generation
4. 17Lands API → GIH WR per card → bots + evaluation only

## Data Model

```python
Card: name, mana_cost, cmc, type_line, oracle_text, colors, color_identity,
      rarity, power, toughness, set_code, collector_number, scryfall_id,
      card_faces (for DFCs/split/adventure), keywords

SeatState: picks, sideboard, notes, color_affinity
DraftState: set_code, seats, packs[round][seat], current_round, current_pick, pack_size
PickRecord: round_num, pick_num, pack_contents, picked_card, reasoning,
            llm_tool_calls, notes_at_time, card_ratings, best_available,
            pick_was_best, pick_rank_in_pack
```

## Dependencies

```
anthropic>=0.40.0
openai>=1.50.0        # optional
httpx>=0.27.0
python-Levenshtein>=0.25.0
```

Python 3.11+ required. All I/O uses `async`/`await`.

## Testing Priorities

Unit tests cover pack rotation (critical edge case — verify seat 0 receives seat 7's pack when passing left), bot pick scoring with colour affinity, fuzzy card name matching for `pick_card` dispatch, and evaluation metric calculations. Integration tests require an API key.
