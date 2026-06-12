"""
Tests for the scoring engine.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

import sys
import os

# Ensure the parent directory is on the path so `app` is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.models import (
    BracketPrediction,
    Match,
    MatchStatus,
    RoundType,
    ScoringWeights,
    Team,
)
from app.scoring import (
    calculate_bracket_score,
    get_scoring_summary,
    score_match,
)


def _make_match(
    home: str,
    away: str,
    home_score: int,
    away_score: int,
    round_type: RoundType = RoundType.ROUND_OF_16,
    label: str = "R16_1",
    status: MatchStatus = MatchStatus.COMPLETED,
) -> Match:
    return Match(
        match_id="test-001",
        home_team=Team(name=home),
        away_team=Team(name=away),
        round_type=round_type,
        match_datetime=datetime.now(tz=timezone.utc),
        status=status,
        home_score=home_score,
        away_score=away_score,
        match_label=label,
    )


def _default_bracket() -> BracketPrediction:
    return BracketPrediction(
        winner="Argentina",
        runner_up="France",
        semifinals=["Argentina", "Brazil", "France", "Spain"],
        quarterfinals=["Argentina", "Netherlands", "Brazil", "England", "France", "Portugal", "Spain", "Germany"],
        match_predictions={
            "R16_1": "Argentina",
            "R16_2": "Brazil",
            "QF_1": "Argentina",
            "SF_1": "Argentina",
            "F_1": "Argentina",
        },
    )


class TestScoringWeights:
    def test_default_weights(self):
        w = ScoringWeights()
        assert w.group_stage == 1
        assert w.round_of_16 == 2
        assert w.quarterfinal == 4
        assert w.semifinal == 8
        assert w.final == 16
        assert w.champion == 32

    def test_custom_weights(self):
        w = ScoringWeights(group_stage=2, round_of_16=5)
        assert w.group_stage == 2
        assert w.round_of_16 == 5

    def test_for_round(self):
        w = ScoringWeights()
        assert w.for_round(RoundType.GROUP_STAGE) == 1
        assert w.for_round(RoundType.ROUND_OF_16) == 2
        assert w.for_round(RoundType.QUARTERFINAL) == 4
        assert w.for_round(RoundType.SEMIFINAL) == 8
        assert w.for_round(RoundType.FINAL) == 16


class TestScoreMatch:
    def test_correct_pick_earns_points(self):
        match = _make_match("Argentina", "Netherlands", 3, 0)
        bracket = _default_bracket()
        result = score_match(match, bracket, ScoringWeights())
        assert result is not None
        assert result.is_correct is True
        assert result.points_earned == 2  # R16 weight

    def test_incorrect_pick_earns_zero(self):
        match = _make_match("Argentina", "Netherlands", 0, 2, label="R16_1")
        bracket = _default_bracket()
        result = score_match(match, bracket, ScoringWeights())
        assert result is not None
        assert result.is_correct is False
        assert result.points_earned == 0

    def test_no_prediction_returns_none(self):
        match = _make_match("Iran", "Saudi Arabia", 1, 0, label="R16_99")
        bracket = BracketPrediction()
        result = score_match(match, bracket, ScoringWeights())
        assert result is None

    def test_pending_match_has_no_score(self):
        match = _make_match("Argentina", "Netherlands", 0, 0, status=MatchStatus.SCHEDULED)
        bracket = _default_bracket()
        result = score_match(match, bracket, ScoringWeights())
        assert result is not None
        assert result.points_earned == 0

    def test_draw_returns_no_winner(self):
        match = _make_match(
            "France", "Spain", 1, 1,
            round_type=RoundType.GROUP_STAGE,
            label="GS_1",
        )
        assert match.winner is None


class TestCalculateBracketScore:
    def test_single_correct_match(self):
        matches = [_make_match("Argentina", "Netherlands", 3, 0)]
        bracket = _default_bracket()
        score = calculate_bracket_score(matches, bracket, ScoringWeights())
        assert score.current_score == 2
        assert len(score.correct_picks) == 1
        assert len(score.incorrect_picks) == 0

    def test_single_incorrect_match(self):
        matches = [_make_match("Argentina", "Netherlands", 0, 2, label="R16_1")]
        bracket = _default_bracket()
        score = calculate_bracket_score(matches, bracket, ScoringWeights())
        assert score.current_score == 0
        assert len(score.incorrect_picks) == 1

    def test_mixed_results(self):
        matches = [
            _make_match("Argentina", "Netherlands", 3, 0, label="R16_1"),
            _make_match("Brazil", "England", 0, 1, label="R16_2"),
        ]
        bracket = _default_bracket()
        score = calculate_bracket_score(matches, bracket, ScoringWeights())
        assert score.current_score == 2  # Argentina correct (2pts), Brazil wrong (0pts)
        assert len(score.correct_picks) == 1
        assert len(score.incorrect_picks) == 1

    def test_empty_matches(self):
        score = calculate_bracket_score([], BracketPrediction(), ScoringWeights())
        assert score.current_score == 0
        assert len(score.correct_picks) == 0


class TestGetScoringSummary:
    def test_summary_keys(self):
        matches = [_make_match("Argentina", "Netherlands", 3, 0)]
        bracket = _default_bracket()
        score = calculate_bracket_score(matches, bracket, ScoringWeights())
        summary = get_scoring_summary(score)
        assert "current_score" in summary
        assert "max_possible_score" in summary
        assert "health_score" in summary
        assert "correct_picks" in summary
        assert "incorrect_picks" in summary

    def test_health_score_between_0_and_100(self):
        matches = [_make_match("Argentina", "Netherlands", 3, 0)]
        bracket = _default_bracket()
        score = calculate_bracket_score(matches, bracket, ScoringWeights())
        assert 0.0 <= score.health_score <= 100.0
