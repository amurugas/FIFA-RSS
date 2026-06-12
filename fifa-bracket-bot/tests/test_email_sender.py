"""
Tests for email_sender module.

Uses mocking to avoid real network/SMTP calls.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import MagicMock, patch, call

import pytest

from app.email_sender import (
    EmailSendError,
    _refresh_access_token,
    send_email,
    send_via_gmail_api,
    send_via_smtp,
)


class FakeConfig:
    def __init__(self, provider="gmail_smtp", dry_run=False):
        self.email_provider = provider
        self.email_address = "sender@example.com"
        self.email_password = "fakepassword"
        self.email_recipient = "recipient@example.com"
        self.gmail_client_id = "fake_client_id"
        self.gmail_client_secret = "fake_client_secret"
        self.gmail_refresh_token = "fake_refresh_token"
        self.dry_run = dry_run


class TestSendViaSMTP:
    def test_sends_successfully(self):
        with patch("app.email_sender.smtplib.SMTP_SSL") as mock_smtp_cls:
            mock_server = MagicMock()
            mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_server)
            mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

            send_via_smtp("sender@example.com", "pass", "recv@example.com", "Subject", "<html/>")

            mock_server.login.assert_called_once()
            mock_server.sendmail.assert_called_once()

    def test_retries_on_failure_then_raises(self):
        import smtplib

        with patch("app.email_sender.smtplib.SMTP_SSL") as mock_smtp_cls:
            mock_smtp_cls.side_effect = smtplib.SMTPException("connection refused")

            with pytest.raises(EmailSendError):
                send_via_smtp("sender@example.com", "pass", "recv@example.com", "Subject", "<html/>")


class TestRefreshAccessToken:
    def test_returns_access_token(self):
        with patch("app.email_sender.requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"access_token": "my_token_xyz"}
            mock_post.return_value.raise_for_status = MagicMock()

            token = _refresh_access_token("client_id", "client_secret", "refresh_token")

            assert token == "my_token_xyz"

    def test_raises_on_http_error(self):
        import requests as req_lib

        with patch("app.email_sender.requests.post") as mock_post:
            mock_post.return_value.raise_for_status.side_effect = req_lib.HTTPError("401")

            with pytest.raises(req_lib.HTTPError):
                _refresh_access_token("id", "secret", "refresh")


class TestSendViaGmailApi:
    def test_sends_successfully(self):
        with patch("app.email_sender._refresh_access_token", return_value="tok"):
            with patch("app.email_sender.requests.post") as mock_post:
                mock_post.return_value.raise_for_status = MagicMock()
                mock_post.return_value.status_code = 200

                send_via_gmail_api(
                    sender="s@x.com",
                    client_id="id",
                    client_secret="secret",
                    refresh_token="rt",
                    recipient="r@x.com",
                    subject="Hi",
                    html_body="<p>Hello</p>",
                )

                mock_post.assert_called_once()
                call_kwargs = mock_post.call_args
                auth_header = call_kwargs.kwargs.get("headers", {}).get("Authorization", "")
                assert auth_header.startswith("Bearer ")

    def test_retries_on_failure_then_raises(self):
        import requests as req_lib

        with patch("app.email_sender._refresh_access_token", return_value="tok"):
            with patch("app.email_sender.requests.post") as mock_post:
                mock_post.side_effect = req_lib.RequestException("network error")

                with pytest.raises(EmailSendError):
                    send_via_gmail_api(
                        sender="s@x.com",
                        client_id="id",
                        client_secret="secret",
                        refresh_token="rt",
                        recipient="r@x.com",
                        subject="Hi",
                        html_body="<p>Hello</p>",
                    )


class TestSendEmail:
    def test_dry_run_does_not_call_smtp(self):
        config = FakeConfig(dry_run=True)
        with patch("app.email_sender.send_via_smtp") as mock_smtp:
            send_email(config, "Subject", "<p>Body</p>")
            mock_smtp.assert_not_called()

    def test_smtp_provider_calls_send_via_smtp(self):
        config = FakeConfig(provider="gmail_smtp")
        with patch("app.email_sender.send_via_smtp") as mock_smtp:
            send_email(config, "Subject", "<p>Body</p>")
            mock_smtp.assert_called_once()

    def test_gmail_api_provider_calls_send_via_gmail_api(self):
        config = FakeConfig(provider="gmail_api")
        with patch("app.email_sender.send_via_gmail_api") as mock_api:
            send_email(config, "Subject", "<p>Body</p>")
            mock_api.assert_called_once()

    def test_uses_email_recipient(self):
        config = FakeConfig(provider="gmail_smtp")
        config.email_recipient = "custom@example.com"
        with patch("app.email_sender.send_via_smtp") as mock_smtp:
            send_email(config, "Subject", "<html/>")
            call_args = mock_smtp.call_args
            # recipient passed as keyword arg
            recipient_val = call_args.kwargs.get("recipient") or (
                call_args.args[2] if len(call_args.args) > 2 else None
            )
            assert recipient_val == "custom@example.com"
