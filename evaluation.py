"""Evaluation — pick logging, accuracy metrics, and draft report generation."""
from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from card_database import Card
from draft_engine import SeatState
from ratings import Ratings

logger = logging.getLogger(__name__)

# Ideal limited mana curve (non-lands by CMC bucket)
IDEAL_CURVE: dict[str, float] = {
    "0-1": 1.5,
    "2": 4.5,
    "3": 4.5,
    "4": 3.5,
    "5+": 2.5,
}

# Per-model pricing in USD per 1M tokens
MODEL_PRICING: dict[str, dict[str, float]] = {
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    "claude-sonnet-4-5": {"input": 3.0, "output": 15.0},
    "claude-opus-4-6": {"input": 15.0, "output": 75.0},
    "gpt-4o": {"input": 5.0, "output": 15.0},
}
DEFAULT_PRICING: dict[str, float] = {"input": 3.0, "output": 15.0}


@dataclass
class PickRecord:
    round_num: int
    pick_num: int
    pack_contents: list[str]
    picked_card: str
    reasoning: str
    llm_tool_calls: int
    notes_at_time: list[str]
    card_ratings: dict[str, float] = field(default_factory=dict)
    best_available: str = ""
    pick_was_best: bool = False
    pick_rank_in_pack: int = 0


class Evaluator:
    def __init__(self, ratings: Ratings) -> None:
        self.ratings = ratings
        self.pick_records: list[PickRecord] = []

    def record_pick(
        self,
        round_num: int,
        pick_num: int,
        pack: list[Card],
        picked_card: Card,
        reasoning: str,
        tool_calls: int,
        notes: list[str],
        set_code: str,
    ) -> None:
        all_ratings = self.ratings.get_all_ratings(set_code)

        card_ratings = {
            c.name: all_ratings[c.name]
            for c in pack
            if c.name in all_ratings
        }

        rated_cards = sorted(
            [(c.name, card_ratings.get(c.name, 0.0)) for c in pack],
            key=lambda x: -x[1],
        )

        best_available = rated_cards[0][0] if rated_cards else ""
        pick_was_best = picked_card.name == best_available

        pick_rank = 1
        for i, (name, _) in enumerate(rated_cards):
            if name == picked_card.name:
                pick_rank = i + 1
                break

        record = PickRecord(
            round_num=round_num,
            pick_num=pick_num,
            pack_contents=[c.name for c in pack],
            picked_card=picked_card.name,
            reasoning=reasoning,
            llm_tool_calls=tool_calls,
            notes_at_time=list(notes),
            card_ratings=card_ratings,
            best_available=best_available,
            pick_was_best=pick_was_best,
            pick_rank_in_pack=pick_rank,
        )
        self.pick_records.append(record)

    def compute_metrics(
        self,
        seat_state: SeatState,
        set_code: str,
        model: str = "",
    ) -> dict:
        if not self.pick_records:
            return {}

        total_picks = len(self.pick_records)
        top1_correct = sum(1 for r in self.pick_records if r.pick_was_best)
        top3_correct = sum(1 for r in self.pick_records if r.pick_rank_in_pack <= 3)

        top1_accuracy = top1_correct / total_picks
        top3_accuracy = top3_correct / total_picks
        avg_pick_rank = sum(r.pick_rank_in_pack for r in self.pick_records) / total_picks

        # Colour coherence: fraction of coloured picks matching top-2 colours
        color_counts: Counter = Counter()
        for card in seat_state.picks:
            for c in (card.colors or []):
                color_counts[c] += 1

        total_colored = sum(color_counts.values())
        if total_colored > 0 and len(color_counts) >= 2:
            top2_count = sum(v for _, v in color_counts.most_common(2))
            color_coherence = top2_count / total_colored
        elif total_colored > 0:
            color_coherence = 1.0
        else:
            color_coherence = 0.0

        # Mana curve score
        nonlands = [c for c in seat_state.picks if "Land" not in c.type_line]
        actual_curve: dict[str, int] = {k: 0 for k in IDEAL_CURVE}
        for card in nonlands:
            if card.cmc <= 1:
                actual_curve["0-1"] += 1
            elif card.cmc == 2:
                actual_curve["2"] += 1
            elif card.cmc == 3:
                actual_curve["3"] += 1
            elif card.cmc == 4:
                actual_curve["4"] += 1
            else:
                actual_curve["5+"] += 1

        mse = (
            sum((actual_curve[k] - IDEAL_CURVE[k]) ** 2 for k in IDEAL_CURVE)
            / len(IDEAL_CURVE)
        )
        mana_curve_score = -mse

        return {
            "top1_accuracy": top1_accuracy,
            "top3_accuracy": top3_accuracy,
            "avg_pick_rank": avg_pick_rank,
            "color_coherence": color_coherence,
            "mana_curve_score": mana_curve_score,
            "total_picks": total_picks,
        }

    def write_report(
        self,
        output_dir: str,
        set_code: str,
        metrics: dict,
        seat_state: SeatState,
        total_input_tokens: int = 0,
        total_output_tokens: int = 0,
        total_api_calls: int = 0,
        model: str = "",
    ) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{timestamp}_{set_code.upper()}"

        # Compute token cost
        pricing = MODEL_PRICING.get(model, DEFAULT_PRICING)
        token_cost = (
            total_input_tokens * pricing["input"] / 1_000_000
            + total_output_tokens * pricing["output"] / 1_000_000
        )

        metrics_full = {
            **metrics,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_api_calls": total_api_calls,
            "estimated_cost_usd": round(token_cost, 4),
            "model": model,
        }

        # JSON
        json_data = {
            "set_code": set_code.upper(),
            "model": model,
            "timestamp": timestamp,
            "metrics": metrics_full,
            "main_deck": [c.name for c in seat_state.picks],
            "sideboard": [c.name for c in seat_state.sideboard],
            "pick_records": [
                {
                    "round": r.round_num + 1,
                    "pick": r.pick_num + 1,
                    "pack": r.pack_contents,
                    "picked": r.picked_card,
                    "reasoning": r.reasoning,
                    "tool_calls": r.llm_tool_calls,
                    "rank": r.pick_rank_in_pack,
                    "best_available": r.best_available,
                    "was_best": r.pick_was_best,
                    "ratings": r.card_ratings,
                }
                for r in self.pick_records
            ],
        }

        json_path = output_path / f"{base_name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)

        # Markdown
        md_lines = [
            f"# MTG Draft Report — {set_code.upper()}",
            f"",
            f"**Model**: {model}  ",
            f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"",
            f"## Pick-by-Pick Log",
            f"",
        ]

        for r in self.pick_records:
            pack_size = len(r.pack_contents)
            md_lines.append(
                f"**P{r.round_num + 1}P{r.pick_num + 1}**: "
                f"Picked **{r.picked_card}** "
                f"(rank {r.pick_rank_in_pack}/{pack_size}, "
                f"best: {r.best_available})"
            )
            md_lines.append(f"  *Reason*: {r.reasoning}")
            md_lines.append(f"  *Tool calls*: {r.llm_tool_calls}")
            md_lines.append("")

        md_lines += [
            f"## Final Deck",
            f"",
            f"**Main Deck** ({len(seat_state.picks)} cards):",
        ]
        for card in seat_state.picks:
            md_lines.append(f"- {card.name} {card.mana_cost}")

        if seat_state.sideboard:
            md_lines.append("")
            md_lines.append(f"**Sideboard** ({len(seat_state.sideboard)} cards):")
            for card in seat_state.sideboard:
                md_lines.append(f"- {card.name} {card.mana_cost}")

        md_lines += [
            "",
            "## Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Top-1 Accuracy | {metrics.get('top1_accuracy', 0):.1%} |",
            f"| Top-3 Accuracy | {metrics.get('top3_accuracy', 0):.1%} |",
            f"| Avg Pick Rank | {metrics.get('avg_pick_rank', 0):.2f} |",
            f"| Color Coherence | {metrics.get('color_coherence', 0):.1%} |",
            f"| Mana Curve Score | {metrics.get('mana_curve_score', 0):.2f} |",
            f"| Total API Calls | {total_api_calls} |",
            f"| Input Tokens | {total_input_tokens:,} |",
            f"| Output Tokens | {total_output_tokens:,} |",
            f"| Estimated Cost | ${token_cost:.4f} |",
        ]

        md_path = output_path / f"{base_name}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))

        logger.info(f"Report written: {json_path} and {md_path}")
        print(f"\nDraft report: {json_path}")
        print(f"Markdown report: {md_path}")
