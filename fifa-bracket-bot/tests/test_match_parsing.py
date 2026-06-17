"""
Tests for match parsing utilities and data provider helpers.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import date, datetime, timezone

import pytest

from app.fifa_api import (
    EspnProvider,
    FallbackProvider,
    FotMobProvider,
    _parse_round,
)
from app.models import MatchStatus, RoundType


class TestParseRound:
    def test_group_stage_variants(self):
        for raw in ("group stage", "Group Stage", "GROUP STAGE", "group"):
            assert _parse_round(raw) == RoundType.GROUP_STAGE

    def test_round_of_16_variants(self):
        for raw in ("round of 16", "Round of 16", "last 16"):
            assert _parse_round(raw) == RoundType.ROUND_OF_16

    def test_quarterfinal_variants(self):
        for raw in ("quarterfinals", "Quarterfinal", "quarter-finals"):
            assert _parse_round(raw) == RoundType.QUARTERFINAL

    def test_semifinal_variants(self):
        for raw in ("semifinals", "Semifinal", "semi-finals"):
            assert _parse_round(raw) == RoundType.SEMIFINAL

    def test_final(self):
        assert _parse_round("final") == RoundType.FINAL

    def test_unknown_defaults_to_group_stage(self):
        assert _parse_round("unknown round") == RoundType.GROUP_STAGE


class TestFallbackProvider:
    def test_raises_on_empty_providers(self):
        with pytest.raises(ValueError):
            FallbackProvider([])

    def test_returns_first_success(self):
        """Provider that raises first, then successful second."""

        class FailProvider:
            def get_matches_for_date(self, target_date):
                raise RuntimeError("fail")

        class SuccessProvider:
            def get_matches_for_date(self, target_date):
                return []

        provider = FallbackProvider([FailProvider(), SuccessProvider()])
        result = provider.get_matches_for_date(date.today())
        assert result == []

    def test_raises_when_all_fail(self):
        class FailProvider:
            def get_matches_for_date(self, target_date):
                raise RuntimeError("fail")

        provider = FallbackProvider([FailProvider(), FailProvider()])
        with pytest.raises(RuntimeError):
            provider.get_matches_for_date(date.today())


class TestEspnProviderParsing:
    """Test ESPN event parsing without network calls."""

    def test_parse_completed_event(self):
        provider = EspnProvider()
        event = {
            "id": "espn-001",
            "date": "2026-07-10T18:00:00Z",
            "competitions": [
                {
                    "competitors": [
                        {
                            "homeAway": "home",
                            "score": "3",
                            "team": {
                                "displayName": "Argentina",
                                "abbreviation": "ARG",
                            },
                        },
                        {
                            "homeAway": "away",
                            "score": "0",
                            "team": {
                                "displayName": "Netherlands",
                                "abbreviation": "NED",
                            },
                        },
                    ],
                    "status": {
                        "type": {"name": "STATUS_FINAL"}
                    },
                    "venue": {"fullName": "MetLife Stadium"},
                    "notes": [],
                }
            ],
        }
        match = provider._parse_event(event, date(2026, 7, 10))
        assert match.home_team.name == "Argentina"
        assert match.away_team.name == "Netherlands"
        assert match.home_score == 3
        assert match.away_score == 0
        assert match.status == MatchStatus.COMPLETED
        assert match.venue == "MetLife Stadium"

    def test_parse_scheduled_event(self):
        provider = EspnProvider()
        event = {
            "id": "espn-002",
            "date": "2026-07-11T21:00:00Z",
            "competitions": [
                {
                    "competitors": [
                        {
                            "homeAway": "home",
                            "score": "",
                            "team": {"displayName": "France", "abbreviation": "FRA"},
                        },
                        {
                            "homeAway": "away",
                            "score": "",
                            "team": {"displayName": "Brazil", "abbreviation": "BRA"},
                        },
                    ],
                    "status": {"type": {"name": "STATUS_SCHEDULED"}},
                    "venue": {},
                    "notes": [],
                }
            ],
        }
        match = provider._parse_event(event, date(2026, 7, 11))
        assert match.status == MatchStatus.SCHEDULED
        assert match.home_score is None
        assert match.away_score is None


class TestFotMobProviderParsing:
    """Test FotMob match parsing without network calls."""

    def test_parse_completed_match(self):
        provider = FotMobProvider()
        raw = {
            "id": 12345,
            "time": {"utcTime": "2026-07-10T18:00:00Z"},
            "home": {"name": "Spain", "score": 2},
            "away": {"name": "Germany", "score": 1},
            "status": {"finished": True, "started": True},
            "roundName": "Quarterfinals",
        }
        match = provider._parse_match(raw, date(2026, 7, 10))
        assert match.home_team.name == "Spain"
        assert match.away_team.name == "Germany"
        assert match.home_score == 2
        assert match.away_score == 1
        assert match.status == MatchStatus.COMPLETED
        assert match.round_type == RoundType.QUARTERFINAL

    def test_parse_live_match(self):
        provider = FotMobProvider()
        raw = {
            "id": 12346,
            "time": {"utcTime": "2026-07-10T19:00:00Z"},
            "home": {"name": "Brazil", "score": 1},
            "away": {"name": "England", "score": 1},
            "status": {"finished": False, "started": True},
            "roundName": "Semifinals",
        }
        match = provider._parse_match(raw, date(2026, 7, 10))
        assert match.status == MatchStatus.LIVE
        assert match.round_type == RoundType.SEMIFINAL
