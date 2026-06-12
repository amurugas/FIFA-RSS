"""
Tests for the probability engine.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timezone

from app.models import (
    BracketPrediction,
    Match,
    MatchStatus,
    RoundType,
    ScoringWeights,
    Team,
)
from app.probabilities import (
    build_rooting_guide,
    calculate_probabilities,
    elo_win_probability,
    upset_probability,
)


def _make_today_match(home: str, away: str, status: MatchStatus = MatchStatus.SCHEDULED) -> Match:
    return Match(
        match_id="today-001",
        home_team=Team(name=home),
        away_team=Team(name=away),
        round_type=RoundType.QUARTERFINAL,
        match_datetime=datetime.now(tz=timezone.utc),
        status=status,
    )


def _default_bracket() -> BracketPrediction:
    return BracketPrediction(
        winner="Argentina",
        runner_up="France",
        semifinals=["Argentina", "Brazil", "France", "Spain"],
        quarterfinals=["Argentina", "Netherlands", "Brazil", "England",
                       "France", "Portugal", "Spain", "Germany"],
        match_predictions={"QF_1": "Argentina"},
    )


class TestEloWinProbability:
    def test_probabilities_sum_to_one(self):
        p_home, p_away = elo_win_probability("Argentina", "Saudi Arabia")
        assert abs(p_home + p_away - 1.0) < 1e-9

    def test_stronger_team_has_higher_probability(self):
        p_arg, p_sau = elo_win_probability("Argentina", "Saudi Arabia")
        assert p_arg > p_sau

    def test_equal_teams_close_to_fifty_fifty(self):
        # Two unknown teams should be close to 50/50 (with slight home advantage)
        p_home, p_away = elo_win_probability("Unknown A", "Unknown B")
        # Home advantage moves it slightly above 0.5 for home
        assert p_home > p_away
        assert 0.5 <= p_home <= 0.6

    def test_home_advantage_applied(self):
        # When we swap home/away the probabilities should differ
        p1, _ = elo_win_probability("France", "Brazil")
        p2, _ = elo_win_probability("Brazil", "France")
        # France as home should have higher probability than France as away
        assert p1 != p2


class TestUpsetProbability:
    def test_upset_prob_is_probability_of_losing_team(self):
        # If Argentina is predicted to win, upset = Netherlands wins
        p = upset_probability("Argentina", "Netherlands", predicted_winner="Argentina")
        assert 0.0 < p < 1.0

    def test_no_prediction_returns_half(self):
        p = upset_probability("France", "Spain", predicted_winner=None)
        assert p == 0.5

    def test_upset_is_complementary(self):
        home = "Argentina"
        away = "Netherlands"
        p_home, p_away = elo_win_probability(home, away)
        upset_p = upset_probability(home, away, predicted_winner=home)
        assert abs(upset_p - p_away) < 1e-9


class TestBuildRootingGuide:
    def test_returns_entry_per_unfinished_match(self):
        matches = [
            _make_today_match("Argentina", "Netherlands"),
            _make_today_match("France", "Brazil"),
        ]
        bracket = _default_bracket()
        guide = build_rooting_guide(matches, bracket, ScoringWeights())
        assert len(guide) == 2

    def test_completed_matches_excluded(self):
        matches = [
            _make_today_match("Argentina", "Netherlands", status=MatchStatus.COMPLETED),
            _make_today_match("France", "Brazil"),
        ]
        bracket = _default_bracket()
        guide = build_rooting_guide(matches, bracket, ScoringWeights())
        assert len(guide) == 1

    def test_guide_has_required_keys(self):
        matches = [_make_today_match("Argentina", "Netherlands")]
        bracket = _default_bracket()
        guide = build_rooting_guide(matches, bracket, ScoringWeights())
        assert len(guide) == 1
        entry = guide[0]
        assert "match" in entry
        assert "root_for" in entry
        assert "reason" in entry
        assert "impact" in entry
        assert "win_probability" in entry

    def test_root_for_bracket_pick(self):
        matches = [_make_today_match("Argentina", "Netherlands")]
        bracket = _default_bracket()
        guide = build_rooting_guide(matches, bracket, ScoringWeights())
        # Argentina is champion pick, so should root for Argentina
        assert guide[0]["root_for"] == "Argentina"


class TestCalculateProbabilities:
    def test_returns_two_teams_per_match(self):
        matches = [_make_today_match("Argentina", "Netherlands")]
        bracket = _default_bracket()
        results = calculate_probabilities(matches, bracket, ScoringWeights())
        assert len(results) == 2

    def test_probabilities_valid_range(self):
        matches = [
            _make_today_match("Argentina", "France"),
            _make_today_match("Brazil", "Spain"),
        ]
        bracket = _default_bracket()
        results = calculate_probabilities(matches, bracket, ScoringWeights())
        for r in results:
            assert 0.0 <= r.win_probability <= 1.0
            assert 0.0 <= r.elimination_probability <= 1.0

    def test_win_and_elim_sum_to_one(self):
        matches = [_make_today_match("Argentina", "Brazil")]
        bracket = _default_bracket()
        results = calculate_probabilities(matches, bracket, ScoringWeights())
        for r in results:
            total = r.win_probability + r.elimination_probability
            assert abs(total - 1.0) < 1e-9
