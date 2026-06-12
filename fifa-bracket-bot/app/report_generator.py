"""
Report generator.

Combines scoring, probabilities, and LLM analysis into a structured
DailyReport object, then renders the HTML email body.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .bracket import load_bracket
from .config import Config
from .models import (
    BracketPrediction,
    BracketScore,
    DailyReport,
    Match,
    MatchStatus,
    PickResult,
    ProbabilityResult,
)
from .probabilities import build_rooting_guide, calculate_probabilities
from .scoring import calculate_bracket_score, get_scoring_summary

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM integration
# ---------------------------------------------------------------------------

def _call_openai(
    prompt: str,
    api_key: str,
    model: str = "gpt-4o-mini",
    max_tokens: int = 1000,
) -> str:
    """Call OpenAI chat completions. Returns empty string on failure."""
    try:
        import openai  # lazy import

        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a FIFA World Cup bracket analysis assistant. "
                        "You MUST only use the data provided in the user message. "
                        "Never hallucinate match results, scores, or statistics. "
                        "If data is missing, say so clearly."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return response.choices[0].message.content or ""
    except Exception as exc:  # noqa: BLE001
        logger.warning("OpenAI call failed: %s", exc)
        return ""


def generate_ai_analysis(
    bracket_score: BracketScore,
    yesterday_matches: list[Match],
    today_matches: list[Match],
    bracket: BracketPrediction,
    api_key: str,
    model: str,
) -> dict:
    """Return structured AI analysis sections."""
    if not api_key:
        logger.info("No OpenAI API key configured; skipping AI analysis")
        return {"error": "OpenAI API key not configured"}

    payload = {
        "bracket": {
            "winner": bracket.winner,
            "runner_up": bracket.runner_up,
            "semifinals": bracket.semifinals,
            "quarterfinals": bracket.quarterfinals,
            "match_predictions": bracket.match_predictions,
        },
        "bracket_score": {
            "current": bracket_score.current_score,
            "max_possible": bracket_score.current_score + bracket_score.max_possible_score,
            "health": bracket_score.health_score,
        },
        "yesterday_results": [
            {
                "match": f"{m.home_team} vs {m.away_team}",
                "score": m.score_display,
                "winner": str(m.winner) if m.winner else "draw",
                "round": m.round_type.value,
            }
            for m in yesterday_matches
            if m.status == MatchStatus.COMPLETED
        ],
        "today_matches": [
            {
                "match": f"{m.home_team} vs {m.away_team}",
                "round": m.round_type.value,
            }
            for m in today_matches
        ],
        "correct_picks": [r.predicted for r in bracket_score.correct_picks],
        "incorrect_picks": [r.predicted for r in bracket_score.incorrect_picks],
    }

    prompt = f"""Analyze this FIFA World Cup bracket data and produce JSON with these keys:
- executive_summary: list of 3-5 bullet strings
- biggest_winners: list of team names that helped my bracket most
- biggest_threats: list of team names most dangerous to my bracket
- what_to_watch: list of today's most important match strings
- outlook: 2-3 sentence forecast

DATA:
{json.dumps(payload, indent=2)}

Respond ONLY with valid JSON. Do not include markdown code fences."""

    raw = _call_openai(prompt, api_key, model)
    if not raw:
        return {"error": "LLM returned empty response"}

    try:
        # Strip markdown fences if present
        cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse LLM JSON response: %s", exc)
        return {"raw": raw, "parse_error": str(exc)}


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

def build_report(
    yesterday_matches: list[Match],
    today_matches: list[Match],
    config: Config,
) -> DailyReport:
    bracket = load_bracket(config.bracket_path)
    all_matches = yesterday_matches + today_matches
    bracket_score = calculate_bracket_score(all_matches, bracket, config.scoring_weights)
    pick_results = (
        bracket_score.correct_picks
        + bracket_score.incorrect_picks
        + bracket_score.pending_picks
    )
    probability_results = calculate_probabilities(today_matches, bracket, config.scoring_weights)
    rooting_guide = build_rooting_guide(today_matches, bracket, config.scoring_weights)
    ai_analysis = generate_ai_analysis(
        bracket_score=bracket_score,
        yesterday_matches=yesterday_matches,
        today_matches=today_matches,
        bracket=bracket,
        api_key=config.openai_api_key,
        model=config.openai_model,
    )

    return DailyReport(
        generated_at=datetime.now(tz=timezone.utc),
        yesterday_matches=yesterday_matches,
        today_matches=today_matches,
        bracket_score=bracket_score,
        pick_results=pick_results,
        probability_results=probability_results,
        rooting_guide=rooting_guide,
        ai_analysis=ai_analysis,
    )


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

def render_html(report: DailyReport, template_path: str) -> str:
    """Render the report to an HTML string using Jinja2."""
    template_dir = str(Path(template_path).parent)
    template_file = Path(template_path).name

    env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(["html", "xml"]),
    )
    env.filters["format_dt"] = lambda dt, fmt="%B %d, %Y %H:%M UTC": (
        dt.strftime(fmt) if dt else ""
    )

    template = env.get_template(template_file)
    return template.render(
        report=report,
        scoring_summary=get_scoring_summary(report.bracket_score),
        generated_at=report.generated_at,
    )
