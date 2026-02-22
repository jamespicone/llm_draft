"""Tests for evaluation metrics."""
from __future__ import annotations

import pytest

from card_database import Card
from draft_engine import SeatState
from evaluation import Evaluator, PickRecord
from ratings import Ratings


def make_card(
    name: str,
    rarity: str = "common",
    colors: list[str] | None = None,
    cmc: float = 2.0,
    type_line: str = "Creature",
) -> Card:
    return Card(
        name=name,
        mana_cost="{2}",
        cmc=cmc,
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


class MockRatings:
    def __init__(self, ratings: dict[str, float]) -> None:
        self._data = ratings

    def get_rating(self, card_name: str, set_code: str) -> float | None:
        return self._data.get(card_name)

    def get_all_ratings(self, set_code: str) -> dict[str, float]:
        return dict(self._data)


def make_evaluator(ratings: dict[str, float] | None = None) -> Evaluator:
    return Evaluator(MockRatings(ratings or {}))  # type: ignore[arg-type]


def make_pick_record(
    picked: str,
    pack: list[str],
    was_best: bool,
    rank: int,
    round_num: int = 0,
    pick_num: int = 0,
) -> PickRecord:
    return PickRecord(
        round_num=round_num,
        pick_num=pick_num,
        pack_contents=pack,
        picked_card=picked,
        reasoning="Test reasoning.",
        llm_tool_calls=2,
        notes_at_time=[],
        card_ratings={name: 0.5 for name in pack},
        best_available=pack[0],
        pick_was_best=was_best,
        pick_rank_in_pack=rank,
    )


# ── record_pick ──────────────────────────────────────────────────────────────

class TestRecordPick:
    def test_records_pick_with_correct_rank(self):
        ratings = {"Best Card": 0.65, "Middle Card": 0.55, "Worst Card": 0.45}
        evaluator = make_evaluator(ratings)

        pack = [
            make_card("Best Card"),
            make_card("Middle Card"),
            make_card("Worst Card"),
        ]
        picked = pack[1]  # Middle Card
        seat = SeatState()
        seat.notes.append("Test note")

        evaluator.record_pick(
            round_num=0,
            pick_num=0,
            pack=pack,
            picked_card=picked,
            reasoning="It's medium.",
            tool_calls=3,
            notes=seat.notes,
            set_code="TST",
        )

        assert len(evaluator.pick_records) == 1
        rec = evaluator.pick_records[0]
        assert rec.picked_card == "Middle Card"
        assert rec.best_available == "Best Card"
        assert rec.pick_was_best is False
        assert rec.pick_rank_in_pack == 2  # rank 2 of 3
        assert rec.llm_tool_calls == 3

    def test_records_best_pick_correctly(self):
        ratings = {"Best Card": 0.65, "Other Card": 0.50}
        evaluator = make_evaluator(ratings)

        pack = [make_card("Best Card"), make_card("Other Card")]
        evaluator.record_pick(
            round_num=0, pick_num=0,
            pack=pack, picked_card=pack[0],
            reasoning="Best card.", tool_calls=1,
            notes=[], set_code="TST",
        )

        rec = evaluator.pick_records[0]
        assert rec.pick_was_best is True
        assert rec.pick_rank_in_pack == 1


# ── compute_metrics ──────────────────────────────────────────────────────────

class TestComputeMetrics:
    def test_top1_accuracy_two_thirds(self):
        evaluator = make_evaluator()
        # 3 picks, 2 were best
        evaluator.pick_records = [
            make_pick_record("A", ["A", "B", "C"], was_best=True, rank=1),
            make_pick_record("B", ["A", "B", "C"], was_best=True, rank=1),
            make_pick_record("C", ["A", "B", "C"], was_best=False, rank=2),
        ]
        seat = SeatState()
        metrics = evaluator.compute_metrics(seat, "TST")

        assert abs(metrics["top1_accuracy"] - 2 / 3) < 0.001

    def test_top1_accuracy_perfect(self):
        evaluator = make_evaluator()
        evaluator.pick_records = [
            make_pick_record("A", ["A", "B"], was_best=True, rank=1)
            for _ in range(5)
        ]
        seat = SeatState()
        metrics = evaluator.compute_metrics(seat, "TST")
        assert metrics["top1_accuracy"] == 1.0

    def test_top3_accuracy_all_in_top3(self):
        evaluator = make_evaluator()
        # All picks rank 1, 2, or 3 in a 5-card pack
        evaluator.pick_records = [
            make_pick_record("X", ["A", "B", "C", "D", "E"], was_best=False, rank=1),
            make_pick_record("X", ["A", "B", "C", "D", "E"], was_best=False, rank=2),
            make_pick_record("X", ["A", "B", "C", "D", "E"], was_best=False, rank=3),
        ]
        seat = SeatState()
        metrics = evaluator.compute_metrics(seat, "TST")
        assert metrics["top3_accuracy"] == 1.0

    def test_top3_accuracy_partial(self):
        evaluator = make_evaluator()
        evaluator.pick_records = [
            make_pick_record("X", ["A", "B", "C", "D", "E"], was_best=False, rank=2),
            make_pick_record("X", ["A", "B", "C", "D", "E"], was_best=False, rank=4),
        ]
        seat = SeatState()
        metrics = evaluator.compute_metrics(seat, "TST")
        assert metrics["top3_accuracy"] == 0.5

    def test_avg_pick_rank(self):
        evaluator = make_evaluator()
        evaluator.pick_records = [
            make_pick_record("X", ["A", "B"], was_best=True, rank=1),
            make_pick_record("X", ["A", "B", "C"], was_best=False, rank=3),
            make_pick_record("X", ["A", "B"], was_best=False, rank=2),
        ]
        seat = SeatState()
        metrics = evaluator.compute_metrics(seat, "TST")
        expected_avg = (1 + 3 + 2) / 3
        assert abs(metrics["avg_pick_rank"] - expected_avg) < 0.001

    def test_color_coherence_two_dominant_colors(self):
        """8 black, 7 blue, 1 red → coherence = 15/16."""
        evaluator = make_evaluator()
        seat = SeatState()
        seat.picks = (
            [make_card(f"B-{i}", colors=["B"]) for i in range(8)]
            + [make_card(f"U-{i}", colors=["U"]) for i in range(7)]
            + [make_card("R-1", colors=["R"])]
        )
        evaluator.pick_records = [make_pick_record("X", ["X"], was_best=True, rank=1)]
        metrics = evaluator.compute_metrics(seat, "TST")
        assert abs(metrics["color_coherence"] - 15 / 16) < 0.001

    def test_color_coherence_single_color(self):
        """All cards one colour → coherence = 1.0."""
        evaluator = make_evaluator()
        seat = SeatState()
        seat.picks = [make_card(f"B-{i}", colors=["B"]) for i in range(10)]
        evaluator.pick_records = [make_pick_record("X", ["X"], was_best=True, rank=1)]
        metrics = evaluator.compute_metrics(seat, "TST")
        assert metrics["color_coherence"] == 1.0

    def test_mana_curve_score_perfect_curve(self):
        """A deck matching the ideal curve should score better than a random one."""
        evaluator_perfect = make_evaluator()
        evaluator_random = make_evaluator()

        seat_perfect = SeatState()
        # Approximate ideal curve: 0-1: 1.5, 2: 4.5, 3: 4.5, 4: 3.5, 5+: 2.5
        seat_perfect.picks = (
            [make_card(f"C1-{i}", cmc=1.0) for i in range(2)]  # 0-1: 2
            + [make_card(f"C2-{i}", cmc=2.0) for i in range(4)]  # 2: 4
            + [make_card(f"C3-{i}", cmc=3.0) for i in range(5)]  # 3: 5
            + [make_card(f"C4-{i}", cmc=4.0) for i in range(4)]  # 4: 4
            + [make_card(f"C5-{i}", cmc=5.0) for i in range(2)]  # 5+: 2
        )

        seat_random = SeatState()
        # All 5+ drops — a terrible curve
        seat_random.picks = [make_card(f"C5-{i}", cmc=6.0) for i in range(17)]

        evaluator_perfect.pick_records = [make_pick_record("X", ["X"], True, 1)]
        evaluator_random.pick_records = [make_pick_record("X", ["X"], True, 1)]

        metrics_perfect = evaluator_perfect.compute_metrics(seat_perfect, "TST")
        metrics_random = evaluator_random.compute_metrics(seat_random, "TST")

        assert metrics_perfect["mana_curve_score"] > metrics_random["mana_curve_score"]

    def test_empty_records_returns_empty_dict(self):
        evaluator = make_evaluator()
        metrics = evaluator.compute_metrics(SeatState(), "TST")
        assert metrics == {}
