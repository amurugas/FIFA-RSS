"""
Tests for HTML email rendering.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.models import (
    BracketPrediction,
    BracketScore,
    DailyReport,
    Match,
    MatchStatus,
    PickResult,
    RoundType,
    Team,
)
from app.report_generator import render_html
from app.scoring import get_scoring_summary


TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "..", "templates", "email_template.html")


def _make_match(home: str, away: str, home_score: int = 2, away_score: int = 1) -> Match:
    return Match(
        match_id="render-001",
        home_team=Team(name=home),
        away_team=Team(name=away),
        round_type=RoundType.QUARTERFINAL,
        match_datetime=datetime.now(tz=timezone.utc),
        status=MatchStatus.COMPLETED,
        home_score=home_score,
        away_score=away_score,
        match_label="QF_1",
    )


def _make_report(
    *,
    correct_picks=None,
    incorrect_picks=None,
    yesterday_matches=None,
    today_matches=None,
    ai_analysis=None,
    error_messages=None,
) -> DailyReport:
    if correct_picks is None:
        correct_picks = []
    if incorrect_picks is None:
        incorrect_picks = []
    if yesterday_matches is None:
        yesterday_matches = []
    if today_matches is None:
        today_matches = []
    if ai_analysis is None:
        ai_analysis = {}
    if error_messages is None:
        error_messages = []

    bracket_score = BracketScore(
        current_score=sum(p.points_earned for p in correct_picks),
        max_possible_score=100,
        correct_picks=correct_picks,
        incorrect_picks=incorrect_picks,
    )

    return DailyReport(
        generated_at=datetime.now(tz=timezone.utc),
        yesterday_matches=yesterday_matches,
        today_matches=today_matches,
        bracket_score=bracket_score,
        pick_results=correct_picks + incorrect_picks,
        probability_results=[],
        rooting_guide=[],
        ai_analysis=ai_analysis,
        error_messages=error_messages,
    )


class TestRenderHtml:
    @pytest.fixture(autouse=True)
    def skip_if_no_template(self):
        if not Path(TEMPLATE_PATH).exists():
            pytest.skip("Template file not found")

    def test_renders_without_error(self):
        report = _make_report()
        html = render_html(report, TEMPLATE_PATH)
        assert isinstance(html, str)
        assert len(html) > 100

    def test_contains_required_sections(self):
        report = _make_report()
        html = render_html(report, TEMPLATE_PATH)
        assert "FIFA World Cup Bracket Intelligence" in html
        assert "Bracket Score" in html

    def test_renders_yesterday_matches(self):
        match = _make_match("Argentina", "Netherlands")
        report = _make_report(yesterday_matches=[match])
        html = render_html(report, TEMPLATE_PATH)
        assert "Argentina" in html
        assert "Netherlands" in html

    def test_renders_correct_picks(self):
        match = _make_match("Argentina", "Netherlands")
        pick = PickResult(
            match=match,
            predicted="Argentina",
            actual="Argentina",
            is_correct=True,
            points_earned=4,
            points_possible=4,
            future_value=40,
        )
        report = _make_report(correct_picks=[pick])
        html = render_html(report, TEMPLATE_PATH)
        assert "Argentina" in html
        assert "+4 pts" in html

    def test_renders_incorrect_picks(self):
        match = _make_match("Argentina", "Netherlands", home_score=0, away_score=2)
        pick = PickResult(
            match=match,
            predicted="Argentina",
            actual="Netherlands",
            is_correct=False,
            points_earned=0,
            points_possible=4,
            future_value=0,
        )
        report = _make_report(incorrect_picks=[pick])
        html = render_html(report, TEMPLATE_PATH)
        assert "Incorrect Picks" in html

    def test_renders_error_messages(self):
        report = _make_report(error_messages=["API unavailable", "No data for yesterday"])
        html = render_html(report, TEMPLATE_PATH)
        assert "API unavailable" in html
        assert "No data for yesterday" in html

    def test_renders_ai_analysis(self):
        ai = {
            "executive_summary": ["Argentina leads the bracket", "France on track"],
            "biggest_winners": ["Argentina", "Spain"],
            "biggest_threats": ["Brazil"],
            "what_to_watch": ["Argentina vs Netherlands – must win"],
            "outlook": "Strong chance of hitting 120+ points.",
        }
        report = _make_report(ai_analysis=ai)
        html = render_html(report, TEMPLATE_PATH)
        assert "Argentina leads the bracket" in html
        assert "Biggest Winners" in html

    def test_renders_rooting_guide(self):
        report = _make_report()
        report.rooting_guide = [
            {
                "match": "Argentina vs Netherlands",
                "root_for": "Argentina",
                "reason": "Champion pick",
                "impact": "+32 pts",
                "win_probability": 72.5,
            }
        ]
        html = render_html(report, TEMPLATE_PATH)
        assert "Who To Root For" in html
        assert "Argentina" in html

    def test_is_valid_html(self):
        report = _make_report()
        html = render_html(report, TEMPLATE_PATH)
        assert html.strip().startswith("<!DOCTYPE html>") or "<html" in html
        assert "</html>" in html
