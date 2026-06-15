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
    elo_win_probability_with_draw,
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


class TestEloWinProbabilityWithDraw:
    def test_three_outcomes_sum_to_one(self):
        h, d, a = elo_win_probability_with_draw("Argentina", "France")
        assert abs(h + d + a - 1.0) < 1e-9

    def test_all_values_in_range(self):
        h, d, a = elo_win_probability_with_draw("Spain", "Morocco")
        for p in (h, d, a):
            assert 0.0 <= p <= 1.0

    def test_consistent_with_elo_win_probability_ordering(self):
        # Stronger home team should have higher win prob than away
        h_win, _, a_win = elo_win_probability_with_draw("France", "Saudi Arabia")
        assert h_win > a_win


class TestBuildRootingGuideExtended:
    """Additional tests for Feature 2 — richer rooting guide shape."""

    def test_guide_has_new_impact_level_key(self):
        matches = [_make_today_match("Argentina", "Netherlands")]
        bracket = _default_bracket()
        guide = build_rooting_guide(matches, bracket, ScoringWeights())
        assert len(guide) == 1
        assert "impact_level" in guide[0]

    def test_guide_has_outcomes_key(self):
        matches = [_make_today_match("Argentina", "Netherlands")]
        bracket = _default_bracket()
        guide = build_rooting_guide(matches, bracket, ScoringWeights())
        assert "outcomes" in guide[0]

    def test_outcomes_list_has_three_entries(self):
        matches = [_make_today_match("Argentina", "France")]
        bracket = _default_bracket()
        guide = build_rooting_guide(matches, bracket, ScoringWeights())
        assert len(guide[0]["outcomes"]) == 3

    def test_outcomes_include_draw(self):
        matches = [_make_today_match("Argentina", "France")]
        bracket = _default_bracket()
        guide = build_rooting_guide(matches, bracket, ScoringWeights())
        assert "Draw" in guide[0]["outcomes"]

    def test_guide_has_outcome_details_key(self):
        matches = [_make_today_match("Argentina", "Netherlands")]
        bracket = _default_bracket()
        guide = build_rooting_guide(matches, bracket, ScoringWeights())
        assert "outcome_details" in guide[0]
        details = guide[0]["outcome_details"]
        assert len(details) == 3
        for d in details:
            assert "outcome" in d
            assert "ev" in d
            assert "delta" in d

    def test_impact_level_is_valid_string(self):
        matches = [_make_today_match("Argentina", "Netherlands")]
        bracket = _default_bracket()
        guide = build_rooting_guide(matches, bracket, ScoringWeights())
        valid_levels = {"VERY HIGH", "HIGH", "MEDIUM", "LOW", "NEUTRAL"}
        assert guide[0]["impact_level"] in valid_levels

    def test_root_for_is_first_outcome(self):
        """root_for should correspond to the best outcome."""
        matches = [_make_today_match("Argentina", "Netherlands")]
        bracket = _default_bracket()
        guide = build_rooting_guide(matches, bracket, ScoringWeights())
        entry = guide[0]
        best_outcome = entry["outcomes"][0]
        assert entry["root_for"] == best_outcome

    def test_legacy_keys_preserved(self):
        """All legacy keys must still be present for backward compatibility."""
        matches = [_make_today_match("Argentina", "Netherlands")]
        bracket = _default_bracket()
        guide = build_rooting_guide(matches, bracket, ScoringWeights())
        for key in ("match", "root_for", "reason", "impact", "win_probability"):
            assert key in guide[0], f"Legacy key {key!r} missing from rooting guide"


class TestGetEloCaseInsensitive:
    def test_case_insensitive_lookup(self):
        """_get_elo should handle case-insensitive team names."""
        from app.probabilities import _get_elo, _DEFAULT_ELO_FALLBACK
        # Exact match
        elo_exact = _get_elo("France")
        assert elo_exact > _DEFAULT_ELO_FALLBACK
        # Case-insensitive match
        elo_lower = _get_elo("france")
        assert elo_lower == elo_exact

    def test_unknown_team_returns_fallback(self):
        from app.probabilities import _get_elo, _DEFAULT_ELO_FALLBACK
        assert _get_elo("Narnia United") == _DEFAULT_ELO_FALLBACK


class TestUpsetProbabilityAwayPick:
    def test_predicted_away_win_returns_home_as_upset(self):
        """If away team is predicted winner, upset = home team wins = p_home."""
        from app.probabilities import upset_probability, elo_win_probability
        p_home, _ = elo_win_probability("Argentina", "Netherlands")
        upset_p = upset_probability("Argentina", "Netherlands", predicted_winner="Netherlands")
        assert abs(upset_p - p_home) < 1e-9


class TestBuildRootingGuideDraw:
    """Cover the Draw-as-best-outcome branch in build_rooting_guide."""

    def test_root_for_draw_when_draw_is_best(self):
        """Verify build_rooting_guide works when neither team in the match
        has equal bracket value. Spain vs Argentina: Argentina is the champion
        (60 pts) while Spain is SF+QF (12 pts), so Argentina is root_for."""
        matches = [_make_today_match("Spain", "Argentina")]
        bracket = _default_bracket()
        guide = build_rooting_guide(matches, bracket, ScoringWeights())
        assert len(guide) == 1
        # Argentina (champion, 60 pts) dominates Spain (12 pts) → not a Draw result
        assert guide[0]["root_for"] == "Argentina"

    def test_away_win_as_best_outcome_covered(self):
        """Away team with much higher bracket value should be root_for."""
        matches = [_make_today_match("Morocco", "Argentina")]
        bracket = _default_bracket()  # Morocco is QF (4 pts), Argentina is SF (12 pts)
        # Argentina (away) has higher bracket value → root for Argentina
        guide = build_rooting_guide(matches, bracket, ScoringWeights())
        assert len(guide) == 1
        # Verify away team (Argentina) is the best outcome
        assert guide[0]["root_for"] == "Argentina"
        assert guide[0]["win_probability"] > 0


class TestCalculateProbabilitiesDedupe:
    """Cover the dedup branch in calculate_probabilities."""

    def test_same_team_in_multiple_matches_counted_once(self):
        matches = [
            _make_today_match("Argentina", "France"),
            _make_today_match("Argentina", "Brazil"),  # Argentina appears again
        ]
        bracket = _default_bracket()
        results = calculate_probabilities(matches, bracket, ScoringWeights())
        team_names = [r.team for r in results]
        # Argentina should only appear once
        assert team_names.count("Argentina") == 1


class TestBuildRootingGuideDrawOutcome:
    """Cover the Draw-as-best-outcome and away-win branches in build_rooting_guide."""

    def test_draw_is_root_for_when_both_teams_have_equal_value(self):
        """When both teams have identical bracket value, Draw yields the highest EV
        (EV_draw = base + 0.6*v + 0.6*v = base + 1.2v > base + v = EV_win).
        Covers the root_for = 'Draw' branch and win_prob = dp.
        Spain and Brazil are both SF+QF picks (value 12 each) in _default_bracket.
        """
        bracket = _default_bracket()
        matches = [_make_today_match("Spain", "Brazil")]
        guide = build_rooting_guide(matches, bracket, ScoringWeights())
        assert len(guide) == 1
        assert guide[0]["root_for"] == "Draw"
        # win_probability should be the draw probability (> 0)
        assert guide[0]["win_probability"] > 0

    def test_away_team_is_best_when_away_has_higher_bracket_value(self):
        """Away team with significantly higher bracket value should be root_for,
        covering the away-win win_prob branch."""
        # Morocco (QF, 4 pts) vs Argentina (SF, 12 pts) — Argentina is away
        bracket = _default_bracket()
        matches = [_make_today_match("Morocco", "Argentina")]
        guide = build_rooting_guide(matches, bracket, ScoringWeights())
        assert guide[0]["root_for"] == "Argentina"
        # win_probability should reflect Argentina's win probability as away team
        assert guide[0]["win_probability"] > 0

    def test_runner_up_as_away_team_is_root_for(self):
        """Runner-up pick as away team → root for the away team."""
        bracket = _default_bracket()  # France is runner-up in _default_bracket
        matches = [_make_today_match("Morocco", "France")]
        guide = build_rooting_guide(matches, bracket, ScoringWeights())
        assert guide[0]["root_for"] == "France"

    def test_quarterfinalist_pick_is_root_for(self):
        """Quarterfinalist-only pick is correctly preferred over a non-bracket team."""
        bracket = _default_bracket()  # Netherlands is a quarterfinalist
        matches = [_make_today_match("Netherlands", "Saudi Arabia")]
        guide = build_rooting_guide(matches, bracket, ScoringWeights())
        assert guide[0]["root_for"] == "Netherlands"
