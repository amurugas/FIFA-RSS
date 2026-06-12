"""Configuration management for the FIFA Bracket Bot."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from zoneinfo import ZoneInfo

from .models import ScoringWeights


@dataclass
class Config:
    # --- Email ---
    email_provider: str = field(
        default_factory=lambda: os.getenv("EMAIL_PROVIDER", "gmail_smtp")
    )
    email_address: str = field(
        default_factory=lambda: os.getenv("EMAIL_ADDRESS", "")
    )
    email_password: str = field(
        default_factory=lambda: os.getenv("EMAIL_PASSWORD", "")
    )
    email_recipient: str = field(
        default_factory=lambda: os.getenv("EMAIL_RECIPIENT", os.getenv("EMAIL_ADDRESS", ""))
    )
    gmail_client_id: str = field(
        default_factory=lambda: os.getenv("GMAIL_CLIENT_ID", "")
    )
    gmail_client_secret: str = field(
        default_factory=lambda: os.getenv("GMAIL_CLIENT_SECRET", "")
    )
    gmail_refresh_token: str = field(
        default_factory=lambda: os.getenv("GMAIL_REFRESH_TOKEN", "")
    )

    # --- OpenAI ---
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    openai_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    )

    # --- Data sources ---
    # Comma-separated priority list
    data_providers: list[str] = field(
        default_factory=lambda: os.getenv(
            "DATA_PROVIDERS", "api_football,fotmob,espn"
        ).split(",")
    )
    api_football_key: str = field(
        default_factory=lambda: os.getenv("API_FOOTBALL_KEY", "")
    )
    fotmob_base_url: str = field(
        default_factory=lambda: os.getenv(
            "FOTMOB_BASE_URL", "https://www.fotmob.com/api"
        )
    )
    espn_base_url: str = field(
        default_factory=lambda: os.getenv(
            "ESPN_BASE_URL",
            "https://site.api.espn.com/apis/site/v2/sports/soccer",
        )
    )

    # --- Scoring ---
    scoring_weights: ScoringWeights = field(default_factory=ScoringWeights)

    # --- Paths ---
    bracket_path: str = field(
        default_factory=lambda: os.getenv("BRACKET_PATH", "data/bracket.json")
    )
    template_path: str = field(
        default_factory=lambda: os.getenv(
            "TEMPLATE_PATH", "templates/email_template.html"
        )
    )

    # --- Timezone ---
    timezone: ZoneInfo = field(
        default_factory=lambda: ZoneInfo(
            os.getenv("TIMEZONE", "America/Los_Angeles")
        )
    )

    # --- Misc ---
    dry_run: bool = field(
        default_factory=lambda: os.getenv("DRY_RUN", "false").lower() == "true"
    )
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )

    def validate(self) -> list[str]:
        """Return a list of validation error messages."""
        errors: list[str] = []
        if not self.email_address:
            errors.append("EMAIL_ADDRESS is not set")
        if self.email_provider == "gmail_smtp" and not self.email_password:
            errors.append("EMAIL_PASSWORD is required for gmail_smtp provider")
        if self.email_provider == "gmail_api":
            for var in ("gmail_client_id", "gmail_client_secret", "gmail_refresh_token"):
                if not getattr(self, var):
                    errors.append(f"{var.upper()} is required for gmail_api provider")
        return errors


def load_config() -> Config:
    return Config()
