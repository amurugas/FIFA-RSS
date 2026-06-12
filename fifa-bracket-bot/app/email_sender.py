"""
Email delivery module.

Supports:
  - gmail_smtp  – standard SMTP via smtplib
  - gmail_api   – Gmail REST API via OAuth2
"""

from __future__ import annotations

import base64
import logging
import smtplib
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_GMAIL_TOKEN_URL = "https://oauth2.googleapis.com/token"
_GMAIL_SEND_URL = "https://gmail.googleapis.com/gmail/v1/users/me/messages/send"
_RETRIES = 3
_BACKOFF = 2.0


class EmailSendError(Exception):
    """Raised when email delivery fails permanently."""


# ---------------------------------------------------------------------------
# Gmail SMTP
# ---------------------------------------------------------------------------

def send_via_smtp(
    sender: str,
    password: str,
    recipient: str,
    subject: str,
    html_body: str,
) -> None:
    """Send HTML email via Gmail SMTP with SSL."""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = recipient
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    last_exc: Optional[Exception] = None
    for attempt in range(1, _RETRIES + 1):
        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(sender, password)
                server.sendmail(sender, [recipient], msg.as_string())
            logger.info("Email sent via SMTP to %s", recipient)
            return
        except smtplib.SMTPException as exc:
            logger.warning("SMTP attempt %d/%d failed: %s", attempt, _RETRIES, exc)
            last_exc = exc
            if attempt < _RETRIES:
                time.sleep(_BACKOFF**attempt)

    raise EmailSendError(
        f"SMTP delivery failed after {_RETRIES} attempts: {last_exc}"
    ) from last_exc


# ---------------------------------------------------------------------------
# Gmail API (OAuth2)
# ---------------------------------------------------------------------------

def _refresh_access_token(client_id: str, client_secret: str, refresh_token: str) -> str:
    """Exchange a refresh token for a short-lived access token."""
    resp = requests.post(
        _GMAIL_TOKEN_URL,
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        },
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def send_via_gmail_api(
    sender: str,
    client_id: str,
    client_secret: str,
    refresh_token: str,
    recipient: str,
    subject: str,
    html_body: str,
) -> None:
    """Send HTML email via Gmail REST API using OAuth2."""
    access_token = _refresh_access_token(client_id, client_secret, refresh_token)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = recipient
    msg.attach(MIMEText(html_body, "html", "utf-8"))
    raw_bytes = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")

    bearer = "Bearer " + access_token
    headers = {"Authorization": bearer, "Content-Type": "application/json"}
    payload = {"raw": raw_bytes}

    last_exc: Optional[Exception] = None
    for attempt in range(1, _RETRIES + 1):
        try:
            resp = requests.post(_GMAIL_SEND_URL, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            logger.info("Email sent via Gmail API to %s", recipient)
            return
        except requests.RequestException as exc:
            logger.warning("Gmail API attempt %d/%d failed: %s", attempt, _RETRIES, exc)
            last_exc = exc
            if attempt < _RETRIES:
                time.sleep(_BACKOFF**attempt)

    raise EmailSendError(
        f"Gmail API delivery failed after {_RETRIES} attempts: {last_exc}"
    ) from last_exc


# ---------------------------------------------------------------------------
# Unified send interface
# ---------------------------------------------------------------------------

def send_email(
    config,  # Config object
    subject: str,
    html_body: str,
    recipient: Optional[str] = None,
) -> None:
    """
    Send an email using the provider specified in config.

    Logs the email body without sending if dry_run is True.
    """
    to = recipient or config.email_recipient or config.email_address

    if config.dry_run:
        logger.info(
            "[DRY RUN] Would send email to %s | Subject: %s | Body length: %d chars",
            to,
            subject,
            len(html_body),
        )
        return

    provider = config.email_provider.lower().strip()
    if provider == "gmail_api":
        send_via_gmail_api(
            sender=config.email_address,
            client_id=config.gmail_client_id,
            client_secret=config.gmail_client_secret,
            refresh_token=config.gmail_refresh_token,
            recipient=to,
            subject=subject,
            html_body=html_body,
        )
    else:
        send_via_smtp(
            sender=config.email_address,
            **{"password": config.email_password},
            recipient=to,
            subject=subject,
            html_body=html_body,
        )
