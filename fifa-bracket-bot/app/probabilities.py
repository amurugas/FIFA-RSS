"""
Probability engine.

Calculates:
  - Remaining maximum possible score
  - Eliminated picks
  - Surviving champion paths
  - Expected score estimate
  - Bracket health score (0–100)
  - Per-match win/upset probabilities (Elo-based heuristic)
  - Draw probability model for group-stage matches
  - Rooting guide (which team to support for bracket benefit)
  - Bracket Impact Score (per-match EV for each outcome)
  - Future Health metrics (expected future score, survival, elite finish)
  - Team Value Ranking
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from .models import (
    BracketPrediction,
    BracketScore,
    FutureHealth,
    Match,
    MatchImpactScore,
    MatchStatus,
    OutcomeImpact,
    ProbabilityResult,
    RoundType,
    ScoringWeights,
    Team,
    TeamValue,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Simple Elo-based win probability
# ---------------------------------------------------------------------------

# Baseline Elo ratings for 2026 WC era (approximate FIFA rankings proxy)
_DEFAULT_ELO: dict[str, float] = {
    "Argentina": 2050,
    "France": 2000,
    "Brazil": 1990,
    "England": 1970,
    "Spain": 1960,
    "Portugal": 1940,
    "Netherlands": 1930,
    "Germany": 1920,
    "Belgium": 1890,
    "Uruguay": 1870,
    "Croatia": 1850,
    "Colombia": 1840,
    "Italy": 1830,
    "Mexico": 1810,
    "United States": 1790,
    "Canada": 1770,
    "Morocco": 1760,
    "Senegal": 1750,
    "Japan": 1740,
    "South Korea": 1720,
    "Australia": 1700,
    "Saudi Arabia": 1680,
    "Iran": 1670,
    "Poland": 1660,
    "Denmark": 1720,
    "Sweden": 1710,
    "Switzerland": 1700,
    "Serbia": 1690,
    "Ukraine": 1680,
    "Czech Republic": 1670,
    "Austria": 1660,
    "Hungary": 1640,
    "Ecuador": 1660,
    "Peru": 1650,
    "Chile": 1640,
    "Venezuela": 1600,
    "Paraguay": 1590,
    "Costa Rica": 1580,
    "Honduras": 1550,
    "Panama": 1540,
    "Jamaica": 1520,
    "Algeria": 1660,
    "Nigeria": 1650,
    "Cameroon": 1630,
    "Ghana": 1620,
    "Egypt": 1610,
    "Tunisia": 1600,
    "Ivory Coast": 1590,
    "South Africa": 1560,
    "Qatar": 1540,
    "New Zealand": 1510,
}

_DEFAULT_ELO_FALLBACK = 1600.0

# ---------------------------------------------------------------------------
# Draw probability model constants (group stage)
# ---------------------------------------------------------------------------

DRAW_BASE_PROB: float = 0.25
"""Base draw probability for a perfectly even group-stage match.
Scales down as the Elo gap between teams widens."""

DRAW_ADV_FACTOR: float = 0.6
"""Fraction of a team's future bracket value credited when the match ends in a draw.
A draw partially preserves advancement chances (~60% of a win's value)."""

# Rooting-interest classification thresholds (EV spread between best and worst outcome)
ROOTING_INTEREST_VERY_HIGH: int = 30
ROOTING_INTEREST_HIGH: int = 15
ROOTING_INTEREST_MEDIUM: int = 8
ROOTING_INTEREST_LOW: int = 3

# Average Elo of a World Cup participant — used in advancement probability estimates
_AVG_WC_ELO: float = 1700.0


def _get_elo(team_name: str) -> float:
    # Try exact match, then case-insensitive
    if team_name in _DEFAULT_ELO:
        return _DEFAULT_ELO[team_name]
    for key, val in _DEFAULT_ELO.items():
        if key.lower() == team_name.lower():
            return val
    return _DEFAULT_ELO_FALLBACK


def elo_win_probability(home: str, away: str) -> tuple[float, float]:
    """
    Return (home_win_prob, away_win_prob) based on Elo ratings.
    Draw probability is excluded; used for knockout rounds.
    For group stage where draws are possible this still works as an
    approximation of "home scores more / away scores more".
    """
    r_home = _get_elo(home) + 50  # home advantage
    r_away = _get_elo(away)
    expected_home = 1.0 / (1.0 + 10 ** ((r_away - r_home) / 400.0))
    expected_away = 1.0 - expected_home
    return expected_home, expected_away


def elo_win_probability_with_draw(home: str, away: str) -> tuple[float, float, float]:
    """
    Return (home_win, draw, away_win) for a group-stage match including draw probability.

    Draw probability is highest when Elo ratings are close and decreases as the gap
    widens. The three probabilities always sum to 1.0.
    """
    r_home = _get_elo(home) + 50  # home advantage
    r_away = _get_elo(away)
    raw_home = 1.0 / (1.0 + 10 ** ((r_away - r_home) / 400.0))
    raw_away = 1.0 - raw_home

    # evenness ∈ [0, 1]: 1 = perfectly equal, 0 = maximally uneven
    evenness = 1.0 - abs(raw_home - 0.5) * 2.0
    draw_prob = DRAW_BASE_PROB * evenness

    # Renormalize win probabilities over the remaining probability mass
    remaining = 1.0 - draw_prob
    home_win = raw_home * remaining
    away_win = raw_away * remaining
    return home_win, draw_prob, away_win


def upset_probability(home: str, away: str, predicted_winner: Optional[str]) -> float:
    """
    Probability that the *unpredicted* team wins.
    """
    if not predicted_winner:
        return 0.5
    p_home, p_away = elo_win_probability(home, away)
    if predicted_winner.lower() == home.lower():
        return p_away  # upset = away wins
    return p_home


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _all_bracket_teams(bracket: BracketPrediction) -> set[str]:
    """Return all teams mentioned anywhere in the bracket predictions."""
    teams: set[str] = set()
    if bracket.winner:
        teams.add(bracket.winner)
    if bracket.runner_up:
        teams.add(bracket.runner_up)
    teams.update(bracket.semifinals)
    teams.update(bracket.quarterfinals)
    teams.update(v for v in bracket.match_predictions.values() if v)
    return teams


def _elo_advancement_probability(team: str, rounds_remaining: int = 3) -> float:
    """
    Estimate the probability that *team* survives *rounds_remaining* more matches,
    assuming Elo-based win probability against an average World Cup opponent each round.

    Uses _AVG_WC_ELO as the benchmark opponent. Result is clamped to [0.01, 0.99].
    """
    elo = _get_elo(team)
    p_win = 1.0 / (1.0 + 10 ** ((_AVG_WC_ELO - elo) / 400.0))
    return max(0.01, min(0.99, p_win ** rounds_remaining))


def _classify_rooting_interest(ev_spread: float) -> str:
    """Classify rooting interest based on EV spread between best and worst outcome."""
    if ev_spread >= ROOTING_INTEREST_VERY_HIGH:
        return "VERY HIGH"
    if ev_spread >= ROOTING_INTEREST_HIGH:
        return "HIGH"
    if ev_spread >= ROOTING_INTEREST_MEDIUM:
        return "MEDIUM"
    if ev_spread >= ROOTING_INTEREST_LOW:
        return "LOW"
    return "NEUTRAL"


def _build_impact_reason(
    home: str,
    away: str,
    home_value: int,
    away_value: int,
    bracket: BracketPrediction,
) -> str:
    """Build an advancement-based reason string for a match impact score."""
    parts = []
    for team, value in ((home, home_value), (away, away_value)):
        if value == 0:
            continue
        t = team.lower()
        if bracket.winner and t == bracket.winner.lower():
            parts.append(f"{team} is your predicted champion")
        elif bracket.runner_up and t == bracket.runner_up.lower():
            parts.append(f"{team} is your predicted runner-up")
        elif t in [s.lower() for s in bracket.semifinals]:
            parts.append(f"{team} is a predicted semifinalist")
        elif t in [q.lower() for q in bracket.quarterfinals]:
            parts.append(f"{team} is a predicted quarterfinalist")

    if parts:
        return ". ".join(parts) + "."
    if home_value > 0 or away_value > 0:
        high = home if home_value >= away_value else away
        return f"{high} advancing preserves bracket value."
    return "Neither team has direct bracket significance."


# ---------------------------------------------------------------------------
# Bracket Impact Score (Feature 1)
# ---------------------------------------------------------------------------

def compute_match_impact_scores(
    today_matches: list[Match],
    bracket: BracketPrediction,
    weights: ScoringWeights,
    current_score: int = 0,
) -> list[MatchImpactScore]:
    """
    For each unfinished group-stage match, compute the expected bracket value (EV)
    under each of the three possible outcomes: home win, draw, and away win.

    EV(outcome) = base_ev + home_value × home_factor + away_value × away_factor

    where base_ev is the sum of future bracket values of all bracket teams NOT
    playing in this specific match (constant across outcomes), and the factors are:
      - Win  → 1.0 (full advancement value)
      - Draw → DRAW_ADV_FACTOR (~0.6, partial advancement value)
      - Loss → 0.0

    The outcomes are ranked best → worst by EV. Deltas are reported relative to
    the lowest-EV (baseline) outcome.
    """
    all_teams = _all_bracket_teams(bracket)
    scores: list[MatchImpactScore] = []

    for match in today_matches:
        if match.status == MatchStatus.COMPLETED:
            continue

        home = match.home_team.name
        away = match.away_team.name

        home_value = _future_bracket_value(home, bracket, weights)
        away_value = _future_bracket_value(away, bracket, weights)

        # Base EV: sum of future values for bracket teams NOT in this match
        other_teams = all_teams - {home, away}
        base_ev = float(current_score) + sum(
            float(_future_bracket_value(t, bracket, weights)) for t in other_teams
        )

        # Per-outcome EVs
        ev_home_win = base_ev + float(home_value)
        ev_draw = base_ev + float(home_value) * DRAW_ADV_FACTOR + float(away_value) * DRAW_ADV_FACTOR
        ev_away_win = base_ev + float(away_value)

        # Rank best → worst
        raw_outcomes = [
            (f"{home} win", ev_home_win),
            ("Draw", ev_draw),
            (f"{away} win", ev_away_win),
        ]
        raw_outcomes.sort(key=lambda x: x[1], reverse=True)

        baseline_ev = raw_outcomes[-1][1]
        outcomes = [
            OutcomeImpact(
                outcome=label,
                ev=round(ev, 1),
                delta=round(ev - baseline_ev, 1),
            )
            for label, ev in raw_outcomes
        ]

        ev_spread = outcomes[0].ev - outcomes[-1].ev
        rooting_interest = _classify_rooting_interest(ev_spread)
        reason = _build_impact_reason(home, away, home_value, away_value, bracket)

        scores.append(MatchImpactScore(
            match=f"{home} vs {away}",
            home_team=home,
            away_team=away,
            outcomes=outcomes,
            rooting_interest=rooting_interest,
            reason=reason,
        ))

    return scores


# ---------------------------------------------------------------------------
# Rooting guide (Feature 2 — extended from legacy)
# ---------------------------------------------------------------------------

def build_rooting_guide(
    today_matches: list[Match],
    bracket: BracketPrediction,
    weights: ScoringWeights,
) -> list[dict]:
    """
    For each today's match, determine the ranked outcome list and explain why
    it matters for the bracket.

    Legacy keys (preserved for backwards compatibility):
        match, root_for, reason, impact, win_probability

    New keys added for Feature 2:
        impact_level   – "VERY HIGH" / "HIGH" / "MEDIUM" / "LOW" / "NEUTRAL"
        outcomes       – ordered list of team names / "Draw" best → worst
        outcome_details – list of {outcome, ev, delta} dicts
    """
    guide: list[dict] = []
    impact_scores = compute_match_impact_scores(today_matches, bracket, weights)

    for mis in impact_scores:
        home = mis.home_team
        away = mis.away_team

        # Best outcome's display name
        best_label = mis.outcomes[0].outcome  # e.g., "Belgium win" or "Draw"
        if best_label == "Draw":
            root_for = "Draw"
        else:
            root_for = best_label[: -len(" win")]  # strip trailing " win"

        # Ordered list of names (best → worst) for Root-for display
        def _label_to_name(label: str) -> str:
            return label if label == "Draw" else label[: -len(" win")]

        outcomes_ordered = [_label_to_name(o.outcome) for o in mis.outcomes]

        # Legacy impact string
        ev_spread = mis.outcomes[0].ev - mis.outcomes[-1].ev
        impact_legacy = (
            f"+{int(ev_spread)} expected bracket points" if ev_spread > 0 else "Neutral"
        )

        # Win probability for the best outcome (for legacy display)
        hw, dp, aw = elo_win_probability_with_draw(home, away)
        if root_for == home:
            win_prob = hw
        elif root_for == away:
            win_prob = aw
        else:
            win_prob = dp

        guide.append({
            # Legacy keys
            "match": mis.match,
            "root_for": root_for,
            "reason": mis.reason,
            "impact": impact_legacy,
            "win_probability": round(win_prob * 100, 1),
            # New keys
            "impact_level": mis.rooting_interest,
            "outcomes": outcomes_ordered,
            "outcome_details": [
                {"outcome": o.outcome, "ev": o.ev, "delta": o.delta}
                for o in mis.outcomes
            ],
        })

    return guide


def _future_bracket_value(team: str, bracket: BracketPrediction, weights: ScoringWeights) -> int:
    """Total future points if *team* wins every round they're predicted to."""
    value = 0
    t = team.lower()

    if t in [s.lower() for s in bracket.quarterfinals]:
        value += weights.quarterfinal
    if t in [s.lower() for s in bracket.semifinals]:
        value += weights.semifinal
    if bracket.winner and t == bracket.winner.lower():
        value += weights.final
    elif bracket.runner_up and t == bracket.runner_up.lower():
        value += weights.final
    if bracket.winner and t == bracket.winner.lower():
        value += weights.champion
    return value


# ---------------------------------------------------------------------------
# Future Health (Feature 3)
# ---------------------------------------------------------------------------

def compute_future_health(
    bracket: BracketPrediction,
    weights: ScoringWeights,
    bracket_score: BracketScore,
) -> FutureHealth:
    """
    Compute forward-looking bracket health metrics.

    Expected Future Score: weighted sum of each bracket team's future value
        multiplied by an Elo-based probability of surviving ~3 more rounds.
        Teams that already have an incorrect pick are treated as eliminated
        (probability = 0). This ensures an unscored bracket on Day 3 shows
        a meaningful positive value rather than 0.

    Bracket Survival: P(champion still on track to reach SF) ×
        P(runner-up still on track to reach QF) × 100.

    Elite Finish Chance: P(champion reaches the final) ×
        P(runner-up reaches the final) × 100.
        When either finalist pick is already eliminated this drops to 0.
    """
    eliminated = {r.predicted.lower() for r in bracket_score.incorrect_picks if r.predicted}

    def _adv(team: str, rounds: int) -> float:
        return 0.0 if team.lower() in eliminated else _elo_advancement_probability(team, rounds)

    # Expected future score: sum over all bracket teams
    all_teams = _all_bracket_teams(bracket)
    expected_future = sum(
        float(_future_bracket_value(t, bracket, weights)) * _adv(t, 3)
        for t in all_teams
    )

    champion = bracket.winner or ""
    runner_up = bracket.runner_up or ""

    # Bracket survival
    champ_survival = _adv(champion, 4) if champion else 0.0
    ru_survival = _adv(runner_up, 3) if runner_up else 0.0
    bracket_survival = round(champ_survival * ru_survival * 100.0, 1)

    # Elite finish chance
    champ_elite = _adv(champion, 5) if champion else 0.0
    ru_elite = _adv(runner_up, 4) if runner_up else 0.0
    elite_finish_chance = round(champ_elite * ru_elite * 100.0, 1)

    return FutureHealth(
        current_score=bracket_score.current_score,
        expected_future_score=round(expected_future, 1),
        bracket_survival=bracket_survival,
        elite_finish_chance=elite_finish_chance,
    )


# ---------------------------------------------------------------------------
# Team Value Ranking (Feature 4)
# ---------------------------------------------------------------------------

def compute_team_value_ranking(
    bracket: BracketPrediction,
    weights: ScoringWeights,
) -> list[TeamValue]:
    """
    Return bracket teams sorted by net future bracket value, descending.

    Champion pick ranks highest (holds the most future value).
    Teams with zero bracket significance are omitted.
    """
    all_teams = _all_bracket_teams(bracket)
    ranking = [
        TeamValue(team=t, bracket_value=_future_bracket_value(t, bracket, weights))
        for t in all_teams
    ]
    ranking.sort(key=lambda tv: tv.bracket_value, reverse=True)
    return [tv for tv in ranking if tv.bracket_value > 0]


# ---------------------------------------------------------------------------
# Probability results per team
# ---------------------------------------------------------------------------

def calculate_probabilities(
    today_matches: list[Match],
    bracket: BracketPrediction,
    weights: ScoringWeights,
) -> list[ProbabilityResult]:
    results: list[ProbabilityResult] = []
    seen: set[str] = set()
    for match in today_matches:
        for team in (match.home_team, match.away_team):
            if team.name in seen:
                continue
            seen.add(team.name)
            other = match.away_team if team == match.home_team else match.home_team
            win_p, _ = elo_win_probability(team.name, other.name)
            elim_p = 1.0 - win_p
            expected_pts = _future_bracket_value(team.name, bracket, weights) * win_p
            results.append(
                ProbabilityResult(
                    team=team.name,
                    win_probability=round(win_p, 3),
                    elimination_probability=round(elim_p, 3),
                    expected_additional_points=round(expected_pts, 1),
                    surviving_paths=1 if win_p > 0.5 else 0,
                )
            )
    return results
