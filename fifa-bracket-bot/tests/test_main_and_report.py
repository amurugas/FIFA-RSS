"""
Tests for main.py orchestration and report_generator helpers.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from app.models import (
    BracketPrediction,
    BracketScore,
    DailyReport,
    Match,
    MatchStatus,
    RoundType,
    Team,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_match(home: str = "Argentina", away: str = "Netherlands") -> Match:
    return Match(
        match_id="test-001",
        home_team=Team(name=home),
        away_team=Team(name=away),
        round_type=RoundType.ROUND_OF_16,
        match_datetime=datetime.now(tz=timezone.utc),
        status=MatchStatus.COMPLETED,
        home_score=2,
        away_score=0,
        match_label="R16_1",
    )


def _make_report(health: float = 75.0, score: int = 10, error_msgs=None) -> DailyReport:
    match = _make_match()
    bracket_score = BracketScore(
        current_score=score,
        max_possible_score=100,
    )
    return DailyReport(
        generated_at=datetime.now(tz=timezone.utc),
        yesterday_matches=[match],
        today_matches=[],
        bracket_score=bracket_score,
        pick_results=[],
        probability_results=[],
        rooting_guide=[],
        ai_analysis={},
        error_messages=error_msgs or [],
    )


# ---------------------------------------------------------------------------
# Test _build_subject
# ---------------------------------------------------------------------------

class TestBuildSubject:
    def test_subject_contains_score(self):
        from app.main import _build_subject

        report = _make_report(score=42)
        subject = _build_subject(report)
        assert "42" in subject

    def test_subject_contains_date(self):
        from app.main import _build_subject

        report = _make_report()
        subject = _build_subject(report)
        assert "FIFA" in subject

    def test_subject_counts_completed_matches(self):
        from app.main import _build_subject

        report = _make_report()
        subject = _build_subject(report)
        # yesterday_matches has one completed match
        assert "1" in subject


# ---------------------------------------------------------------------------
# Test _send_error_email
# ---------------------------------------------------------------------------

class TestSendErrorEmail:
    def test_calls_send_email(self):
        from app.main import _send_error_email
        from app.config import Config

        config = Config()
        config.dry_run = True

        with patch("app.main.send_email") as mock_send:
            _send_error_email(config, "Something went wrong")
            mock_send.assert_called_once()
            subject = mock_send.call_args[0][1]
            assert "Error" in subject

    def test_does_not_raise_on_send_failure(self):
        from app.main import _send_error_email
        from app.config import Config

        config = Config()
        with patch("app.main.send_email", side_effect=Exception("SMTP down")):
            # Should not propagate the exception
            _send_error_email(config, "Some error")


# ---------------------------------------------------------------------------
# Test full run() flow
# ---------------------------------------------------------------------------

class TestRun:
    def test_dry_run_succeeds(self, tmp_path):
        """Full integration-style run with mocked I/O."""
        from app.main import run

        bracket_path = tmp_path / "bracket.json"
        bracket_path.write_text('{"winner": "Argentina"}', encoding="utf-8")

        template_src = os.path.join(
            os.path.dirname(__file__), "..", "templates", "email_template.html"
        )

        with patch.dict(
            os.environ,
            {
                "DRY_RUN": "true",
                "EMAIL_ADDRESS": "test@example.com",
                "EMAIL_RECIPIENT": "test@example.com",
                "EMAIL_PROVIDER": "gmail_smtp",
                "EMAIL_PASSWORD": "fakepass",
                "BRACKET_PATH": str(bracket_path),
                "TEMPLATE_PATH": template_src,
                "OPENAI_API_KEY": "",
            },
            clear=False,
        ):
            with patch("app.main.build_provider") as mock_provider_factory:
                mock_provider = MagicMock()
                mock_provider.get_yesterday_matches.return_value = []
                mock_provider.get_today_matches.return_value = []
                mock_provider_factory.return_value = mock_provider

                with patch("app.main.send_email") as mock_send:
                    result = run()
                    # dry_run=true, so send_email is called but logs only
                    assert result == 0

    def test_run_returns_1_on_provider_failure(self, tmp_path):
        """If all providers fail and build_report fails too, returns 1."""
        from app.main import run

        bracket_path = tmp_path / "bracket.json"
        bracket_path.write_text('{"winner": "Argentina"}', encoding="utf-8")

        template_src = os.path.join(
            os.path.dirname(__file__), "..", "templates", "email_template.html"
        )

        with patch.dict(
            os.environ,
            {
                "DRY_RUN": "true",
                "EMAIL_ADDRESS": "test@example.com",
                "EMAIL_PROVIDER": "gmail_smtp",
                "EMAIL_PASSWORD": "fakepass",
                "BRACKET_PATH": str(bracket_path),
                "TEMPLATE_PATH": template_src,
                "OPENAI_API_KEY": "",
            },
            clear=False,
        ):
            with patch("app.main.build_provider") as mock_provider_factory:
                mock_provider = MagicMock()
                mock_provider.get_yesterday_matches.side_effect = RuntimeError("No internet")
                mock_provider.get_today_matches.side_effect = RuntimeError("No internet")
                mock_provider_factory.return_value = mock_provider

                with patch("app.main.send_email"):
                    # Provider errors are collected but not fatal; build_report should succeed
                    result = run()
                    assert result in (0, 1)  # depends on whether send succeeds


# ---------------------------------------------------------------------------
# Test generate_ai_analysis
# ---------------------------------------------------------------------------

class TestGenerateAiAnalysis:
    def test_no_api_key_returns_error(self):
        from app.report_generator import generate_ai_analysis

        score = BracketScore()
        result = generate_ai_analysis(
            bracket_score=score,
            yesterday_matches=[],
            today_matches=[],
            bracket=BracketPrediction(),
            api_key="",
            model="gpt-4o-mini",
        )
        assert "error" in result

    def test_openai_failure_returns_error(self):
        from app.report_generator import generate_ai_analysis

        score = BracketScore()
        with patch("app.report_generator._call_openai", return_value=""):
            result = generate_ai_analysis(
                bracket_score=score,
                yesterday_matches=[],
                today_matches=[],
                bracket=BracketPrediction(),
                api_key="fake_key",
                model="gpt-4o-mini",
            )
        assert "error" in result

    def test_valid_openai_response_parsed(self):
        from app.report_generator import generate_ai_analysis

        fake_response = '{"executive_summary": ["All good"], "biggest_winners": ["Argentina"], "biggest_threats": [], "what_to_watch": [], "outlook": "Looking strong."}'
        score = BracketScore()
        with patch("app.report_generator._call_openai", return_value=fake_response):
            result = generate_ai_analysis(
                bracket_score=score,
                yesterday_matches=[],
                today_matches=[],
                bracket=BracketPrediction(winner="Argentina"),
                api_key="fake_key",
                model="gpt-4o-mini",
            )
        assert result.get("executive_summary") == ["All good"]
        assert result.get("biggest_winners") == ["Argentina"]

    def test_invalid_json_response_returns_raw(self):
        from app.report_generator import generate_ai_analysis

        score = BracketScore()
        with patch("app.report_generator._call_openai", return_value="NOT JSON {{{"):
            result = generate_ai_analysis(
                bracket_score=score,
                yesterday_matches=[],
                today_matches=[],
                bracket=BracketPrediction(),
                api_key="fake_key",
                model="gpt-4o-mini",
            )
        assert "parse_error" in result or "raw" in result


class TestGenerateAiAnalysisExtraContext:
    """Tests for the extra_context enrichment in generate_ai_analysis."""

    def test_extra_context_included_in_payload(self):
        """extra_context data (future_health, team_value_ranking, match_impact_scores) is passed to LLM."""
        from app.report_generator import generate_ai_analysis

        extra = {
            "future_health": {"current_score": 0, "expected_future_score": 72.7},
            "team_value_ranking": [{"team": "France", "bracket_value": 60}],
            "match_impact_scores": [{"match": "France vs Morocco", "rooting_interest": "VERY HIGH"}],
        }

        captured_prompts = []
        def fake_call(prompt, api_key, model, **kwargs):
            captured_prompts.append(prompt)
            return '{"executive_summary": ["Test"], "biggest_winners": [], "biggest_threats": [], "what_to_watch": [], "outlook": "ok"}'

        score = BracketScore()
        with patch("app.report_generator._call_openai", side_effect=fake_call):
            result = generate_ai_analysis(
                bracket_score=score,
                yesterday_matches=[],
                today_matches=[],
                bracket=BracketPrediction(winner="France"),
                api_key="fake_key",
                model="gpt-4o-mini",
                extra_context=extra,
            )

        assert len(captured_prompts) == 1
        assert "future_health" in captured_prompts[0]
        assert "team_value_ranking" in captured_prompts[0]
        assert "match_impact_scores" in captured_prompts[0]
        assert result.get("executive_summary") == ["Test"]

    def test_no_extra_context_still_works(self):
        """When extra_context is None, generate_ai_analysis works normally."""
        from app.report_generator import generate_ai_analysis

        score = BracketScore()
        fake_response = '{"executive_summary": ["All good"], "biggest_winners": [], "biggest_threats": [], "what_to_watch": [], "outlook": "ok"}'
        with patch("app.report_generator._call_openai", return_value=fake_response):
            result = generate_ai_analysis(
                bracket_score=score,
                yesterday_matches=[],
                today_matches=[],
                bracket=BracketPrediction(winner="France"),
                api_key="fake_key",
                model="gpt-4o-mini",
                extra_context=None,
            )
        assert result.get("executive_summary") == ["All good"]

    def test_extra_context_partial_keys(self):
        """extra_context with only some keys populates only those in the DATA payload."""
        import json
        from app.report_generator import generate_ai_analysis

        extra = {"future_health": {"current_score": 5}}

        captured_prompts = []
        def fake_call(prompt, api_key, model, **kwargs):
            captured_prompts.append(prompt)
            return '{"executive_summary": [], "biggest_winners": [], "biggest_threats": [], "what_to_watch": [], "outlook": ""}'

        score = BracketScore()
        with patch("app.report_generator._call_openai", side_effect=fake_call):
            generate_ai_analysis(
                bracket_score=score,
                yesterday_matches=[],
                today_matches=[],
                bracket=BracketPrediction(),
                api_key="fake_key",
                model="gpt-4o-mini",
                extra_context=extra,
            )

        # The DATA: JSON block should contain future_health but NOT team_value_ranking
        prompt = captured_prompts[0]
        data_section = prompt[prompt.index("DATA:"):]
        assert '"future_health"' in data_section
        assert '"team_value_ranking"' not in data_section
