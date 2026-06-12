"""Data models for the FIFA Bracket Bot."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class RoundType(str, Enum):
    GROUP_STAGE = "group_stage"
    ROUND_OF_16 = "round_of_16"
    QUARTERFINAL = "quarterfinal"
    SEMIFINAL = "semifinal"
    FINAL = "final"


class MatchStatus(str, Enum):
    SCHEDULED = "scheduled"
    LIVE = "live"
    COMPLETED = "completed"
    POSTPONED = "postponed"
    CANCELLED = "cancelled"


@dataclass
class Team:
    name: str
    code: str = ""
    group: str = ""

    def __str__(self) -> str:
        return self.name


@dataclass
class MatchEvent:
    minute: int
    event_type: str  # "goal", "yellow_card", "red_card", "substitution"
    player: str
    team: str
    detail: str = ""


@dataclass
class Match:
    match_id: str
    home_team: Team
    away_team: Team
    round_type: RoundType
    match_datetime: datetime
    status: MatchStatus = MatchStatus.SCHEDULED
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    events: list[MatchEvent] = field(default_factory=list)
    venue: str = ""
    match_label: str = ""  # e.g. "R16_1", "QF_2"

    @property
    def winner(self) -> Optional[Team]:
        if self.status != MatchStatus.COMPLETED:
            return None
        if self.home_score is None or self.away_score is None:
            return None
        if self.home_score > self.away_score:
            return self.home_team
        if self.away_score > self.home_score:
            return self.away_team
        return None  # draw (group stage)

    @property
    def score_display(self) -> str:
        if self.home_score is None or self.away_score is None:
            return "vs"
        return f"{self.home_score}–{self.away_score}"

    @property
    def is_upset(self) -> bool:
        """Simple heuristic: no real odds, so marked externally."""
        return False


@dataclass
class BracketPrediction:
    winner: str = ""
    runner_up: str = ""
    semifinals: list[str] = field(default_factory=list)
    quarterfinals: list[str] = field(default_factory=list)
    match_predictions: dict[str, str] = field(default_factory=dict)

    def predicted_winner_for(self, match_label: str) -> Optional[str]:
        return self.match_predictions.get(match_label)


@dataclass
class ScoringWeights:
    group_stage: int = 1
    round_of_16: int = 2
    quarterfinal: int = 4
    semifinal: int = 8
    final: int = 16
    champion: int = 32

    def for_round(self, round_type: RoundType) -> int:
        mapping = {
            RoundType.GROUP_STAGE: self.group_stage,
            RoundType.ROUND_OF_16: self.round_of_16,
            RoundType.QUARTERFINAL: self.quarterfinal,
            RoundType.SEMIFINAL: self.semifinal,
            RoundType.FINAL: self.final,
        }
        return mapping.get(round_type, 0)


@dataclass
class PickResult:
    match: Match
    predicted: str
    actual: Optional[str]
    is_correct: bool
    points_earned: int
    points_possible: int
    future_value: int  # points still reachable via this pick


@dataclass
class BracketScore:
    current_score: int = 0
    max_possible_score: int = 0
    correct_picks: list[PickResult] = field(default_factory=list)
    incorrect_picks: list[PickResult] = field(default_factory=list)
    pending_picks: list[PickResult] = field(default_factory=list)

    @property
    def health_score(self) -> float:
        """Return a 0–100 bracket health score."""
        if self.max_possible_score == 0:
            return 0.0
        # max_possible_score is the total achievable score (current + remaining)
        overall_max = self._compute_overall_max()
        if overall_max == 0:
            return 0.0
        return round((self.max_possible_score / overall_max) * 100, 1)

    def _compute_overall_max(self) -> int:
        weights = ScoringWeights()
        # Theoretical perfect bracket for the 2026 FIFA World Cup (48-team format)
        # 104 group stage matches + 8 R32 + 8 R16 + 4 QF + 2 SF + 1 F + champion bonus
        return (
            104 * weights.group_stage
            + 8 * weights.round_of_16   # Round of 32 (use R16 weight as proxy)
            + 8 * weights.round_of_16
            + 4 * weights.quarterfinal
            + 2 * weights.semifinal
            + 1 * weights.final
            + weights.champion
        )


@dataclass
class ProbabilityResult:
    team: str
    win_probability: float  # 0–1
    elimination_probability: float  # 0–1
    expected_additional_points: float
    surviving_paths: int


@dataclass
class DailyReport:
    generated_at: datetime
    yesterday_matches: list[Match]
    today_matches: list[Match]
    bracket_score: BracketScore
    pick_results: list[PickResult]
    probability_results: list[ProbabilityResult]
    rooting_guide: list[dict]  # {"team": ..., "reason": ..., "impact": ...}
    ai_analysis: dict  # structured sections from LLM
    error_messages: list[str] = field(default_factory=list)
