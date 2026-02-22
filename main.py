#!/usr/bin/env python3
"""MTG Draft LLM Harness — CLI entry point."""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from card_database import CardDatabase
from draft_engine import DraftEngine, DraftState, SeatState
from evaluation import Evaluator
from llm_harness import LLMHarness
from llm_provider import AnthropicProvider, HumanProvider, OpenAIProvider
from pack_generator import PackGenerator
from ratings import Ratings

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def load_system_prompt(set_code: str, card_db: CardDatabase) -> str:
    prompt_path = Path(__file__).parent / "system_prompt.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"system_prompt.txt not found at {prompt_path}")

    template = prompt_path.read_text(encoding="utf-8")

    mechanics = card_db.get_set_mechanics(set_code)
    if mechanics:
        set_context = (
            f"This draft uses the **{set_code.upper()}** set. "
            f"Key mechanics present in this set: {', '.join(mechanics)}."
        )
    else:
        set_context = f"This draft uses the **{set_code.upper()}** set."

    return template.replace("{set_context}", set_context)


async def run_draft(args: argparse.Namespace) -> None:
    set_code = args.set.upper()

    print(f"MTG Draft LLM Harness — Set: {set_code}")
    print("=" * 50)

    # Card database
    print("Loading card database...")
    card_db = CardDatabase()
    await card_db.ensure_set_data(set_code)
    cards = card_db.get_cards_for_set(set_code)
    print(f"Loaded {len(cards)} cards for {set_code}")

    # Ratings (17Lands)
    print("Loading 17Lands ratings...")
    ratings = Ratings()
    await ratings.ensure_ratings(set_code)

    # Pack generation
    num_seats = args.seats
    print(f"Generating {num_seats * 3} packs...")
    generator = PackGenerator(card_db)
    packs = generator.generate_all_packs(
        set_code, num_seats=num_seats, num_rounds=3, seed=args.seed
    )

    if args.dry_run:
        print("\n--- DRY RUN: Generated Packs ---")
        for round_idx, round_packs in enumerate(packs):
            print(f"\nRound {round_idx + 1}:")
            for seat_idx, pack in enumerate(round_packs):
                card_names = ", ".join(c.name for c in pack)
                print(f"  Seat {seat_idx} ({len(pack)} cards): {card_names}")
        print("\nDry run complete.")
        return

    # System prompt
    system_prompt = load_system_prompt(set_code, card_db)

    # Human provider gets verbose feedback by default
    if args.provider == "human" and not args.verbose:
        args.verbose = True

    # LLM provider
    if args.provider == "human":
        model = "human"
        provider = HumanProvider()
    elif args.provider == "openai":
        model = args.model or "gpt-4o"
        provider = OpenAIProvider(model=model)
    else:
        model = args.model or "claude-sonnet-4-6"
        provider = AnthropicProvider(model=model)
    print(f"Provider: {args.provider}, Model: {model}")

    # Draft state
    seats = [SeatState() for _ in range(num_seats)]
    pack_size = len(packs[0][0]) if packs and packs[0] else 15
    state = DraftState(
        set_code=set_code,
        seats=seats,
        packs=packs,
        pack_size=pack_size,
    )

    # Harness + evaluator
    harness = LLMHarness(provider, card_db, system_prompt)
    evaluator = Evaluator(ratings)

    # Instrument make_pick to record evaluation data
    _original_make_pick = harness.make_pick

    async def instrumented_make_pick(
        pack, seat_state, round_num, pick_num
    ):
        pack_snapshot = list(pack)  # snapshot before pick removes card
        card = await _original_make_pick(pack, seat_state, round_num, pick_num)
        evaluator.record_pick(
            round_num=round_num,
            pick_num=pick_num,
            pack=pack_snapshot,
            picked_card=card,
            reasoning=harness._last_reasoning,
            tool_calls=harness._last_tool_call_count,
            notes=list(seat_state.notes),
            set_code=set_code,
        )
        if args.verbose:
            rec = evaluator.pick_records[-1] if evaluator.pick_records else None
            rank_str = f" (rank {rec.pick_rank_in_pack}/{len(pack_snapshot)})" if rec else ""
            print(f"  P{round_num+1}P{pick_num+1}: {card.name}{rank_str}")
        return card

    harness.make_pick = instrumented_make_pick  # type: ignore[method-assign]

    # Run draft
    print("\nStarting draft...")
    engine = DraftEngine(state, ratings, harness)
    await engine.run()
    print("Draft complete!")

    # Deckbuilding phase
    print("\nDeckbuilding phase...")
    await harness.build_deck(state.seats[0])
    if args.verbose:
        deck = state.seats[0].picks
        lands = sum(1 for c in deck if "Land" in c.type_line)
        print(f"  Final deck: {len(deck)} cards, {lands} lands")

    # Evaluate
    metrics = evaluator.compute_metrics(state.seats[0], set_code, model)
    output_dir = args.output_dir or "./drafts"
    evaluator.write_report(
        output_dir=output_dir,
        set_code=set_code,
        metrics=metrics,
        seat_state=state.seats[0],
        total_input_tokens=harness.total_input_tokens,
        total_output_tokens=harness.total_output_tokens,
        total_api_calls=harness.total_api_calls,
        model=model,
    )

    # Summary
    print("\n--- Draft Summary ---")
    print(f"Top-1 Accuracy:    {metrics.get('top1_accuracy', 0):.1%}")
    print(f"Top-3 Accuracy:    {metrics.get('top3_accuracy', 0):.1%}")
    print(f"Avg Pick Rank:     {metrics.get('avg_pick_rank', 0):.2f}")
    print(f"Color Coherence:   {metrics.get('color_coherence', 0):.1%}")
    print(f"Total API Calls:   {harness.total_api_calls}")
    print(f"Input Tokens:      {harness.total_input_tokens:,}")
    print(f"Output Tokens:     {harness.total_output_tokens:,}")

    deck = state.seats[0].picks
    sideboard = state.seats[0].sideboard
    print(f"\nMain Deck ({len(deck)} cards):")
    for card in deck:
        print(f"  {card.name} {card.mana_cost}")
    if sideboard:
        print(f"\nSideboard ({len(sideboard)} cards):")
        for card in sideboard:
            print(f"  {card.name} {card.mana_cost}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MTG Draft LLM Harness — evaluates LLM drafting performance"
    )
    subparsers = parser.add_subparsers(dest="command")

    draft_parser = subparsers.add_parser("draft", help="Run a booster draft")
    draft_parser.add_argument(
        "--set", required=True, metavar="SET_CODE",
        help="Set code to draft (e.g. DSK, MKM, BLB)"
    )
    draft_parser.add_argument(
        "--provider", choices=["anthropic", "openai", "human"], default="anthropic",
        help="LLM provider (default: anthropic)"
    )
    draft_parser.add_argument(
        "--model", default=None,
        help="Model name (default: provider-dependent)"
    )
    draft_parser.add_argument(
        "--seats", type=int, default=8,
        help="Number of draft seats (default: 8)"
    )
    draft_parser.add_argument(
        "--output-dir", default="./drafts",
        help="Output directory for draft logs (default: ./drafts)"
    )
    draft_parser.add_argument(
        "--verbose", action="store_true",
        help="Print pick-by-pick output during draft"
    )
    draft_parser.add_argument(
        "--dry-run", action="store_true",
        help="Generate packs and print them without running LLM"
    )
    draft_parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducible pack generation"
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command is None:
        print("Usage: python main.py draft --set DSK [options]")
        print("       python main.py draft --help")
        sys.exit(1)

    setup_logging(getattr(args, "verbose", False))
    asyncio.run(run_draft(args))


if __name__ == "__main__":
    main()
