"""Scoring engine for FIFA bracket predictions."""

from __future__ import annotations

import logging
from typing import Optional

from .models import (
    BracketPrediction,
    BracketScore,
    Match,
    MatchStatus,
    PickResult,
    RoundType,
    ScoringWeights,
    Team,
)

logger = logging.getLogger(__name__)


def _team_name(team: Optional[Team]) -> str:
    return team.name if team else ""


def score_match(
    match: Match,
    bracket: BracketPrediction,
    weights: ScoringWeights,
) -> Optional[PickResult]:
    """
    Score a single match against bracket predictions.

    Returns None if the match has no bracket prediction.
    """
    predicted = bracket.match_predictions.get(match.match_label)
    if not predicted:
        # Try champion/runner_up/semi/QF based on round
        predicted = _infer_prediction(match, bracket)

    if not predicted:
        return None

    actual_winner = _team_name(match.winner) if match.status == MatchStatus.COMPLETED else None
    is_correct = bool(actual_winner and actual_winner.lower() == predicted.lower())
    points_earned = weights.for_round(match.round_type) if is_correct else 0

    # Calculate future value preserved or lost
    future_value = _calculate_future_value(predicted, match, bracket, weights)

    return PickResult(
        match=match,
        predicted=predicted,
        actual=actual_winner,
        is_correct=is_correct,
        points_earned=points_earned,
        points_possible=weights.for_round(match.round_type),
        future_value=future_value,
    )


def _infer_prediction(match: Match, bracket: BracketPrediction) -> Optional[str]:
    """
    Infer a prediction when match_label is absent by checking whether
    either team appears in higher-round bracket slots.
    """
    home = match.home_team.name
    away = match.away_team.name

    advanced_teams = (
        [bracket.winner, bracket.runner_up]
        + bracket.semifinals
        + bracket.quarterfinals
    )
    for team in advanced_teams:
        if team.lower() in (home.lower(), away.lower()):
            return team
    return None


def _calculate_future_value(
    team: str,
    match: Match,
    bracket: BracketPrediction,
    weights: ScoringWeights,
) -> int:
    """
    Points that can still be earned if *team* advances from this match.
    Based on how far the bracket has the team going.
    """
    future = 0
    t = team.lower()

    if t in [s.lower() for s in bracket.semifinals]:
        future += weights.semifinal
    if t in (bracket.winner.lower(), bracket.runner_up.lower()):
        future += weights.final
    if t == bracket.winner.lower():
        future += weights.champion

    return future


def calculate_bracket_score(
    matches: list[Match],
    bracket: BracketPrediction,
    weights: ScoringWeights,
) -> BracketScore:
    """
    Calculate the full bracket score from a list of matches.
    """
    correct: list[PickResult] = []
    incorrect: list[PickResult] = []
    pending: list[PickResult] = []

    for match in matches:
        result = score_match(match, bracket, weights)
        if result is None:
            continue

        if match.status == MatchStatus.COMPLETED:
            if result.is_correct:
                correct.append(result)
            else:
                incorrect.append(result)
        else:
            pending.append(result)

    current_score = sum(r.points_earned for r in correct)
    # Max possible = current + pending possible + future values from surviving picks
    surviving_future = sum(r.future_value for r in correct)
    pending_possible = sum(r.points_possible for r in pending)
    max_possible = current_score + pending_possible + surviving_future

    return BracketScore(
        current_score=current_score,
        max_possible_score=max_possible - current_score,
        correct_picks=correct,
        incorrect_picks=incorrect,
        pending_picks=pending,
    )


def get_scoring_summary(score: BracketScore) -> dict:
    """Return a human-readable summary dict."""
    return {
        "current_score": score.current_score,
        "max_possible_score": score.current_score + score.max_possible_score,
        "health_score": score.health_score,
        "correct_picks": len(score.correct_picks),
        "incorrect_picks": len(score.incorrect_picks),
        "pending_picks": len(score.pending_picks),
    }
