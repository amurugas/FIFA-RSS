"""
Entry point for the FIFA Bracket Bot.

Orchestrates:
  1. Load configuration
  2. Fetch yesterday's and today's matches
  3. Build the daily report
  4. Render the HTML email
  5. Send the email
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone

from .config import load_config
from .email_sender import EmailSendError, send_email
from .fifa_api import build_provider
from .report_generator import build_report, render_html


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


def run() -> int:
    """Main execution flow. Returns exit code (0=success, 1=error)."""
    config = load_config()
    setup_logging(config.log_level)
    logger = logging.getLogger(__name__)

    logger.info("FIFA Bracket Bot starting – %s", datetime.now(tz=timezone.utc).isoformat())

    # Validate config (warn but don't abort – we want to send an error email)
    errors = config.validate()
    if errors:
        for err in errors:
            logger.warning("Config warning: %s", err)

    error_messages: list[str] = list(errors)

    # --- Fetch match data ---
    provider = build_provider(config)
    yesterday_matches = []
    today_matches = []

    try:
        yesterday_matches = provider.get_yesterday_matches()
        logger.info("Fetched %d yesterday matches", len(yesterday_matches))
    except Exception as exc:  # noqa: BLE001
        msg = f"Failed to fetch yesterday's matches: {exc}"
        logger.error(msg)
        error_messages.append(msg)

    try:
        today_matches = provider.get_today_matches()
        logger.info("Fetched %d today matches", len(today_matches))
    except Exception as exc:  # noqa: BLE001
        msg = f"Failed to fetch today's matches: {exc}"
        logger.error(msg)
        error_messages.append(msg)

    # --- Build report ---
    try:
        report = build_report(yesterday_matches, today_matches, config)
        report.error_messages.extend(error_messages)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to build report: %s", exc)
        # Send a minimal error email
        _send_error_email(config, str(exc))
        return 1

    # --- Render HTML ---
    try:
        html_body = render_html(report, config.template_path)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to render HTML: %s", exc)
        _send_error_email(config, f"Template rendering error: {exc}")
        return 1

    # --- Send email ---
    subject = _build_subject(report)
    try:
        send_email(config, subject, html_body)
        logger.info("Daily report sent successfully")
    except EmailSendError as exc:
        logger.error("Email delivery failed: %s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error during email delivery: %s", exc)
        return 1

    return 0


def _build_subject(report) -> str:  # type: ignore[type-arg]
    from .models import MatchStatus

    completed = sum(
        1 for m in report.yesterday_matches if m.status == MatchStatus.COMPLETED
    )
    score = report.bracket_score.current_score
    health = report.bracket_score.health_score
    date_str = report.generated_at.strftime("%b %d")
    return (
        f"⚽ FIFA Bracket Report – {date_str} | "
        f"Score: {score} | Health: {health:.0f}% | "
        f"{completed} result(s) yesterday"
    )


def _send_error_email(config, error_detail: str) -> None:
    """Send a minimal plain-text error notification."""
    logger = logging.getLogger(__name__)
    subject = "⚠️ FIFA Bracket Bot Error"
    body = f"""<html><body>
<h2>FIFA Bracket Bot encountered an error</h2>
<p>{error_detail}</p>
<p>Please check the GitHub Actions logs for more details.</p>
</body></html>"""
    try:
        send_email(config, subject, body)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to send error notification: %s", exc)


if __name__ == "__main__":
    sys.exit(run())
