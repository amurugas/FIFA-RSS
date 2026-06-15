"""
Tests for Bracket Impact Score, Future Health, and Team Value Ranking (Features 1, 3, 4).
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timezone

import pytest

from app.models import (
    BracketPrediction,
    BracketScore,
    Match,
    MatchStatus,
    PickResult,
    RoundType,
    ScoringWeights,
    Team,
)
from app.probabilities import (
    DRAW_ADV_FACTOR,
    DRAW_BASE_PROB,
    ROOTING_INTEREST_HIGH,
    ROOTING_INTEREST_VERY_HIGH,
    _classify_rooting_interest,
    compute_future_health,
    compute_match_impact_scores,
    compute_team_value_ranking,
    elo_win_probability_with_draw,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _match(home: str, away: str, status: MatchStatus = MatchStatus.SCHEDULED) -> Match:
    return Match(
        match_id="test-001",
        home_team=Team(name=home),
        away_team=Team(name=away),
        round_type=RoundType.GROUP_STAGE,
        match_datetime=datetime.now(tz=timezone.utc),
        status=status,
    )


def _strong_bracket() -> BracketPrediction:
    """A bracket where France is champion, England runner-up, Spain+Argentina semis."""
    return BracketPrediction(
        winner="France",
        runner_up="England",
        semifinals=["France", "Spain", "England", "Argentina"],
        quarterfinals=["France", "Morocco", "Spain", "USA",
                       "Brazil", "England", "Argentina", "Portugal"],
        match_predictions={},
    )


# ---------------------------------------------------------------------------
# Draw probability model
# ---------------------------------------------------------------------------

class TestDrawProbability:
    def test_three_outcomes_sum_to_one(self):
        h, d, a = elo_win_probability_with_draw("France", "Brazil")
        assert abs(h + d + a - 1.0) < 1e-9

    def test_draw_higher_for_even_teams(self):
        # Two unknown/equal teams should produce a higher draw probability
        _, d_even, _ = elo_win_probability_with_draw("Unknown A", "Unknown B")
        _, d_uneven, _ = elo_win_probability_with_draw("France", "Qatar")
        assert d_even > d_uneven

    def test_draw_probability_bounded(self):
        for home, away in [("Argentina", "Saudi Arabia"), ("Spain", "Spain"),
                           ("Unknown", "Unknown")]:
            h, d, a = elo_win_probability_with_draw(home, away)
            assert 0.0 <= d <= DRAW_BASE_PROB + 1e-9, f"draw out of range for {home} vs {away}"

    def test_stronger_home_team_wins_more_often(self):
        h, d, a = elo_win_probability_with_draw("France", "Saudi Arabia")
        assert h > a  # France is much stronger

    def test_draw_probability_near_base_for_equal_teams(self):
        h, d, a = elo_win_probability_with_draw("Unknown A", "Unknown B")
        # For identical Elo + home advantage: home is slightly stronger, so draw < DRAW_BASE_PROB
        assert d > 0.0
        assert d <= DRAW_BASE_PROB


# ---------------------------------------------------------------------------
# Bracket Impact Score (Feature 1)
# ---------------------------------------------------------------------------

class TestComputeMatchImpactScores:
    def test_completed_matches_excluded(self):
        bracket = _strong_bracket()
        matches = [
            _match("France", "Morocco", MatchStatus.COMPLETED),
            _match("England", "Argentina"),
        ]
        scores = compute_match_impact_scores(matches, bracket, ScoringWeights())
        # Only the scheduled match should be processed
        assert len(scores) == 1
        assert "England" in scores[0].match

    def test_always_three_outcomes(self):
        bracket = _strong_bracket()
        m = _match("France", "Morocco")
        scores = compute_match_impact_scores([m], bracket, ScoringWeights())
        assert len(scores) == 1
        assert len(scores[0].outcomes) == 3

    def test_outcomes_ranked_best_to_worst(self):
        bracket = _strong_bracket()
        m = _match("France", "Morocco")  # France: champion (60pts), Morocco: QF (4pts)
        scores = compute_match_impact_scores([m], bracket, ScoringWeights())
        evs = [o.ev for o in scores[0].outcomes]
        assert evs == sorted(evs, reverse=True), "Outcomes should be sorted best → worst"

    def test_champion_pick_win_is_best_outcome(self):
        bracket = _strong_bracket()
        m = _match("France", "Morocco")
        scores = compute_match_impact_scores([m], bracket, ScoringWeights())
        assert scores[0].outcomes[0].outcome == "France win"

    def test_baseline_delta_is_zero(self):
        bracket = _strong_bracket()
        m = _match("France", "Morocco")
        scores = compute_match_impact_scores([m], bracket, ScoringWeights())
        worst = scores[0].outcomes[-1]
        assert worst.delta == 0.0, "Baseline (lowest EV) outcome should have delta=0"

    def test_better_outcomes_have_positive_delta(self):
        bracket = _strong_bracket()
        m = _match("France", "Morocco")
        scores = compute_match_impact_scores([m], bracket, ScoringWeights())
        for o in scores[0].outcomes[:-1]:
            assert o.delta > 0.0, f"Non-baseline outcome {o.outcome!r} should have positive delta"

    def test_ev_delta_math(self):
        """delta == outcome.ev - baseline.ev"""
        bracket = _strong_bracket()
        m = _match("England", "Argentina")
        scores = compute_match_impact_scores([m], bracket, ScoringWeights())
        baseline_ev = scores[0].outcomes[-1].ev
        for o in scores[0].outcomes:
            assert abs(o.delta - (o.ev - baseline_ev)) < 0.01

    def test_draw_ev_between_win_and_loss(self):
        """Draw EV should fall between home-win and away-win EVs when teams differ in bracket value."""
        bracket = _strong_bracket()
        m = _match("France", "Morocco")
        scores = compute_match_impact_scores([m], bracket, ScoringWeights())
        evs_by_label = {o.outcome: o.ev for o in scores[0].outcomes}
        assert evs_by_label["France win"] > evs_by_label["Draw"] >= evs_by_label["Morocco win"]

    def test_high_rooting_interest_for_champion_match(self):
        bracket = _strong_bracket()
        m = _match("France", "Morocco")
        scores = compute_match_impact_scores([m], bracket, ScoringWeights())
        interest = scores[0].rooting_interest
        assert interest in ("HIGH", "VERY HIGH")

    def test_neutral_rooting_interest_for_non_bracket_teams(self):
        bracket = _strong_bracket()
        m = _match("Belgium", "Egypt")  # neither team is in the bracket
        scores = compute_match_impact_scores([m], bracket, ScoringWeights())
        assert scores[0].rooting_interest == "NEUTRAL"

    def test_base_ev_includes_other_bracket_teams(self):
        """Base EV should reflect all teams not playing in this match."""
        bracket = _strong_bracket()
        weights = ScoringWeights()
        m = _match("France", "Morocco")
        scores = compute_match_impact_scores([m], bracket, weights)
        # Morocco (worst outcome for this bracket) gives base_ev + Morocco's QF value (4)
        worst_ev = scores[0].outcomes[-1].ev
        assert worst_ev > 0  # should include value from other bracket teams

    def test_current_score_adds_to_base_ev(self):
        bracket = _strong_bracket()
        weights = ScoringWeights()
        m = _match("France", "Morocco")
        scores_0 = compute_match_impact_scores([m], bracket, weights, current_score=0)
        scores_10 = compute_match_impact_scores([m], bracket, weights, current_score=10)
        # All EVs should be exactly 10 higher
        for o0, o10 in zip(scores_0[0].outcomes, scores_10[0].outcomes):
            assert abs(o10.ev - o0.ev - 10.0) < 0.01

    def test_empty_match_list(self):
        bracket = _strong_bracket()
        scores = compute_match_impact_scores([], bracket, ScoringWeights())
        assert scores == []

    def test_match_name_format(self):
        bracket = _strong_bracket()
        m = _match("France", "Morocco")
        scores = compute_match_impact_scores([m], bracket, ScoringWeights())
        assert scores[0].match == "France vs Morocco"
        assert scores[0].home_team == "France"
        assert scores[0].away_team == "Morocco"

    def test_outcome_labels(self):
        bracket = _strong_bracket()
        m = _match("France", "Morocco")
        scores = compute_match_impact_scores([m], bracket, ScoringWeights())
        labels = {o.outcome for o in scores[0].outcomes}
        assert labels == {"France win", "Draw", "Morocco win"}


# ---------------------------------------------------------------------------
# Rooting-interest classification
# ---------------------------------------------------------------------------

class TestClassifyRootingInterest:
    def test_very_high(self):
        assert _classify_rooting_interest(ROOTING_INTEREST_VERY_HIGH) == "VERY HIGH"
        assert _classify_rooting_interest(ROOTING_INTEREST_VERY_HIGH + 10) == "VERY HIGH"

    def test_high(self):
        assert _classify_rooting_interest(ROOTING_INTEREST_HIGH) == "HIGH"

    def test_medium(self):
        assert _classify_rooting_interest(8) == "MEDIUM"

    def test_low(self):
        assert _classify_rooting_interest(3) == "LOW"

    def test_neutral(self):
        assert _classify_rooting_interest(0) == "NEUTRAL"
        assert _classify_rooting_interest(2) == "NEUTRAL"


# ---------------------------------------------------------------------------
# Future Health (Feature 3)
# ---------------------------------------------------------------------------

class TestComputeFutureHealth:
    def test_unscored_bracket_has_positive_expected_future_score(self):
        """A perfect, unscored bracket on Day 3 should NOT show 0 expected future score."""
        bracket = _strong_bracket()
        bs = BracketScore(current_score=0, max_possible_score=128)
        fh = compute_future_health(bracket, ScoringWeights(), bs)
        assert fh.expected_future_score > 0.0

    def test_unscored_bracket_has_high_survival(self):
        """An unscored bracket should show meaningful survival probability."""
        bracket = _strong_bracket()
        bs = BracketScore(current_score=0, max_possible_score=128)
        fh = compute_future_health(bracket, ScoringWeights(), bs)
        # With no eliminations, survival should be > 0 (strong teams have good Elo)
        assert fh.bracket_survival > 0.0

    def test_unscored_bracket_has_nonzero_elite_chance(self):
        """Elite finish chance should be positive with no eliminations."""
        bracket = _strong_bracket()
        bs = BracketScore(current_score=0, max_possible_score=128)
        fh = compute_future_health(bracket, ScoringWeights(), bs)
        assert fh.elite_finish_chance > 0.0

    def test_current_score_reflected(self):
        bracket = _strong_bracket()
        bs = BracketScore(current_score=42, max_possible_score=128)
        fh = compute_future_health(bracket, ScoringWeights(), bs)
        assert fh.current_score == 42

    def test_eliminated_champion_drops_survival_to_zero(self):
        """If the champion pick is eliminated, survival should drop to 0."""
        bracket = _strong_bracket()
        weights = ScoringWeights()
        # Simulate France being an incorrect pick
        from app.models import Match as M
        france_match = _match("France", "Germany", MatchStatus.COMPLETED)
        france_match.home_score = 0
        france_match.away_score = 1
        eliminated_pick = PickResult(
            match=france_match,
            predicted="France",
            actual="Germany",
            is_correct=False,
            points_earned=0,
            points_possible=4,
            future_value=0,
        )
        bs = BracketScore(
            current_score=0,
            max_possible_score=100,
            incorrect_picks=[eliminated_pick],
        )
        fh = compute_future_health(bracket, weights, bs)
        assert fh.bracket_survival == 0.0
        assert fh.elite_finish_chance == 0.0

    def test_probabilities_are_percentages(self):
        bracket = _strong_bracket()
        bs = BracketScore(current_score=0, max_possible_score=128)
        fh = compute_future_health(bracket, ScoringWeights(), bs)
        assert 0.0 <= fh.bracket_survival <= 100.0
        assert 0.0 <= fh.elite_finish_chance <= 100.0

    def test_empty_bracket_does_not_crash(self):
        bracket = BracketPrediction()
        bs = BracketScore(current_score=0, max_possible_score=0)
        fh = compute_future_health(bracket, ScoringWeights(), bs)
        assert fh.expected_future_score == 0.0

    def test_strong_bracket_beats_weak_bracket_on_expected_score(self):
        """A bracket with deeper picks should have higher expected future score."""
        strong = _strong_bracket()
        weak = BracketPrediction(winner="Qatar", runner_up="New Zealand",
                                 semifinals=["Qatar", "New Zealand", "Saudi Arabia", "Jamaica"])
        weights = ScoringWeights()
        bs = BracketScore(current_score=0, max_possible_score=128)
        fh_strong = compute_future_health(strong, weights, bs)
        fh_weak = compute_future_health(weak, weights, bs)
        assert fh_strong.expected_future_score > fh_weak.expected_future_score


# ---------------------------------------------------------------------------
# Team Value Ranking (Feature 4)
# ---------------------------------------------------------------------------

class TestComputeTeamValueRanking:
    def test_champion_pick_ranks_first(self):
        bracket = _strong_bracket()
        ranking = compute_team_value_ranking(bracket, ScoringWeights())
        assert ranking[0].team == "France"

    def test_runner_up_ranks_above_semis(self):
        bracket = _strong_bracket()
        ranking = compute_team_value_ranking(bracket, ScoringWeights())
        teams = [tv.team for tv in ranking]
        runner_up_idx = teams.index("England")
        semi_idx = teams.index("Spain")
        assert runner_up_idx < semi_idx

    def test_values_descending(self):
        bracket = _strong_bracket()
        ranking = compute_team_value_ranking(bracket, ScoringWeights())
        values = [tv.bracket_value for tv in ranking]
        assert values == sorted(values, reverse=True)

    def test_zero_value_teams_excluded(self):
        bracket = _strong_bracket()
        ranking = compute_team_value_ranking(bracket, ScoringWeights())
        for tv in ranking:
            assert tv.bracket_value > 0, f"{tv.team} has zero value and should be excluded"

    def test_empty_bracket_returns_empty(self):
        bracket = BracketPrediction()
        ranking = compute_team_value_ranking(bracket, ScoringWeights())
        assert ranking == []

    def test_bracket_value_matches_expected(self):
        """France (champion) should have QF + SF + F + Champion points."""
        bracket = _strong_bracket()
        weights = ScoringWeights()
        ranking = compute_team_value_ranking(bracket, weights)
        france = next(tv for tv in ranking if tv.team == "France")
        expected = weights.quarterfinal + weights.semifinal + weights.final + weights.champion
        assert france.bracket_value == expected

    def test_all_bracket_teams_with_value_included(self):
        bracket = _strong_bracket()
        ranking = compute_team_value_ranking(bracket, ScoringWeights())
        team_names = {tv.team for tv in ranking}
        # These all have bracket value (QF or deeper)
        for team in ("France", "England", "Spain", "Argentina", "Morocco",
                     "USA", "Brazil", "Portugal"):
            assert team in team_names, f"{team} should be in ranking"
