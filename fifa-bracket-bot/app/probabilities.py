"""
Probability engine.

Calculates:
  - Remaining maximum possible score
  - Eliminated picks
  - Surviving champion paths
  - Expected score estimate
  - Bracket health score (0–100)
  - Per-match win/upset probabilities (Elo-based heuristic)
  - Rooting guide (which team to support for bracket benefit)
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from .models import (
    BracketPrediction,
    BracketScore,
    Match,
    MatchStatus,
    ProbabilityResult,
    RoundType,
    ScoringWeights,
    Team,
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
# Rooting guide
# ---------------------------------------------------------------------------

def build_rooting_guide(
    today_matches: list[Match],
    bracket: BracketPrediction,
    weights: ScoringWeights,
) -> list[dict]:
    """
    For each today's match, determine which team winning is better for
    the bracket and explain why.
    """
    guide: list[dict] = []
    for match in today_matches:
        if match.status == MatchStatus.COMPLETED:
            continue

        home = match.home_team.name
        away = match.away_team.name

        home_value = _future_bracket_value(home, bracket, weights)
        away_value = _future_bracket_value(away, bracket, weights)

        if home_value == away_value:
            root_for = "Either"
            reason = "Neither team has special significance to your bracket."
            impact = "Neutral"
        elif home_value > away_value:
            root_for = home
            reason = _build_reason(home, away, home_value, away_value, bracket)
            impact = f"+{home_value - away_value} expected bracket points"
        else:
            root_for = away
            reason = _build_reason(away, home, away_value, home_value, bracket)
            impact = f"+{away_value - home_value} expected bracket points"

        win_prob_home, win_prob_away = elo_win_probability(home, away)
        win_prob = win_prob_home if root_for == home else win_prob_away

        guide.append(
            {
                "match": f"{home} vs {away}",
                "root_for": root_for,
                "reason": reason,
                "impact": impact,
                "win_probability": round(win_prob * 100, 1),
            }
        )
    return guide


def _future_bracket_value(team: str, bracket: BracketPrediction, weights: ScoringWeights) -> int:
    """Total future points if *team* wins every round they're predicted to."""
    value = 0
    t = team.lower()
    mp = bracket.match_predictions

    # check match-level predictions
    for _label, predicted in mp.items():
        if predicted.lower() == t:
            # Already counted when scoring; just count future advanced rounds
            pass

    if t in [s.lower() for s in bracket.quarterfinals]:
        value += weights.quarterfinal
    if t in [s.lower() for s in bracket.semifinals]:
        value += weights.semifinal
    if t in (bracket.winner.lower(), bracket.runner_up.lower()):
        value += weights.final
    if t == bracket.winner.lower():
        value += weights.champion
    return value


def _build_reason(
    preferred: str,
    other: str,
    preferred_value: int,
    other_value: int,
    bracket: BracketPrediction,
) -> str:
    reasons: list[str] = []
    p = preferred.lower()
    if p == bracket.winner.lower():
        reasons.append(f"you picked {preferred} as champion")
    elif p == bracket.runner_up.lower():
        reasons.append(f"you picked {preferred} as runner-up")
    elif p in [s.lower() for s in bracket.semifinals]:
        reasons.append(f"you picked {preferred} as a semifinalist")
    elif p in [s.lower() for s in bracket.quarterfinals]:
        reasons.append(f"you picked {preferred} as a quarterfinalist")

    if reasons:
        base = f"A {other} victory would eliminate one of your key picks. " + " and ".join(reasons).capitalize() + "."
    else:
        base = f"{preferred} advancing preserves more bracket value than {other}."

    if preferred_value > 0:
        base += f" Bracket value at stake: {preferred_value} points."
    return base


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
