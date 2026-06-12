"""Bracket prediction management."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from .models import BracketPrediction

logger = logging.getLogger(__name__)


def load_bracket(path: str | Path) -> BracketPrediction:
    """Load bracket predictions from a JSON file."""
    p = Path(path)
    if not p.exists():
        logger.warning("Bracket file not found at %s; using empty prediction", path)
        return BracketPrediction()

    with p.open(encoding="utf-8") as fh:
        data = json.load(fh)

    return BracketPrediction(
        winner=data.get("winner", ""),
        runner_up=data.get("runner_up", ""),
        semifinals=data.get("semifinals", []),
        quarterfinals=data.get("quarterfinals", []),
        match_predictions=data.get("match_predictions", {}),
    )


def save_bracket(bracket: BracketPrediction, path: str | Path) -> None:
    """Persist bracket predictions to a JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "winner": bracket.winner,
                "runner_up": bracket.runner_up,
                "semifinals": bracket.semifinals,
                "quarterfinals": bracket.quarterfinals,
                "match_predictions": bracket.match_predictions,
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )
    logger.info("Bracket saved to %s", p)


def get_teams_in_bracket(bracket: BracketPrediction) -> set[str]:
    """Return all team names referenced in the bracket."""
    teams: set[str] = set()
    if bracket.winner:
        teams.add(bracket.winner)
    if bracket.runner_up:
        teams.add(bracket.runner_up)
    teams.update(bracket.semifinals)
    teams.update(bracket.quarterfinals)
    teams.update(bracket.match_predictions.values())
    return teams


def is_team_in_bracket(team_name: str, bracket: BracketPrediction) -> bool:
    return team_name in get_teams_in_bracket(bracket)


def get_future_value(team: str, bracket: BracketPrediction, scoring_weights) -> int:  # type: ignore[type-arg]
    """
    Estimate the maximum future points still reachable for a team
    based on remaining bracket predictions.
    """
    from .models import RoundType  # avoid circular import

    future = 0
    mp = bracket.match_predictions

    # Remaining rounds that haven't been scored yet
    # This is a simplified upper-bound calculation
    if team in bracket.semifinals:
        future += scoring_weights.for_round(RoundType.SEMIFINAL)
    if team in [bracket.winner, bracket.runner_up]:
        future += scoring_weights.for_round(RoundType.FINAL)
    if team == bracket.winner:
        future += scoring_weights.champion

    return future
