"""
Tests for bracket loading and management.
"""

from __future__ import annotations

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from app.bracket import (
    get_teams_in_bracket,
    is_team_in_bracket,
    load_bracket,
    save_bracket,
)
from app.models import BracketPrediction


@pytest.fixture
def sample_bracket_data():
    return {
        "winner": "Argentina",
        "runner_up": "France",
        "semifinals": ["Argentina", "Brazil", "France", "Spain"],
        "quarterfinals": ["Argentina", "Netherlands"],
        "match_predictions": {"R16_1": "Argentina", "R16_2": "Brazil"},
    }


@pytest.fixture
def bracket_file(tmp_path, sample_bracket_data):
    p = tmp_path / "bracket.json"
    p.write_text(json.dumps(sample_bracket_data), encoding="utf-8")
    return p


class TestLoadBracket:
    def test_loads_valid_file(self, bracket_file, sample_bracket_data):
        bracket = load_bracket(bracket_file)
        assert bracket.winner == sample_bracket_data["winner"]
        assert bracket.runner_up == sample_bracket_data["runner_up"]
        assert bracket.semifinals == sample_bracket_data["semifinals"]
        assert bracket.quarterfinals == sample_bracket_data["quarterfinals"]
        assert bracket.match_predictions == sample_bracket_data["match_predictions"]

    def test_missing_file_returns_empty(self, tmp_path):
        bracket = load_bracket(tmp_path / "nonexistent.json")
        assert bracket.winner == ""
        assert bracket.match_predictions == {}

    def test_partial_file(self, tmp_path):
        p = tmp_path / "partial.json"
        p.write_text(json.dumps({"winner": "Brazil"}), encoding="utf-8")
        bracket = load_bracket(p)
        assert bracket.winner == "Brazil"
        assert bracket.runner_up == ""
        assert bracket.semifinals == []


class TestSaveBracket:
    def test_round_trip(self, tmp_path):
        original = BracketPrediction(
            winner="Spain",
            runner_up="Germany",
            semifinals=["Spain", "Brazil", "Germany", "England"],
            quarterfinals=["Spain", "France"],
            match_predictions={"R16_1": "Spain"},
        )
        path = tmp_path / "bracket.json"
        save_bracket(original, path)
        loaded = load_bracket(path)
        assert loaded.winner == original.winner
        assert loaded.runner_up == original.runner_up
        assert loaded.semifinals == original.semifinals
        assert loaded.match_predictions == original.match_predictions

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "subdir" / "deep" / "bracket.json"
        bracket = BracketPrediction(winner="Brazil")
        save_bracket(bracket, path)
        assert path.exists()


class TestGetTeamsInBracket:
    def test_collects_all_teams(self):
        bracket = BracketPrediction(
            winner="Argentina",
            runner_up="France",
            semifinals=["Argentina", "Brazil", "France", "Spain"],
            quarterfinals=["Argentina"],
            match_predictions={"R16_1": "Netherlands"},
        )
        teams = get_teams_in_bracket(bracket)
        assert "Argentina" in teams
        assert "France" in teams
        assert "Brazil" in teams
        assert "Spain" in teams
        assert "Netherlands" in teams

    def test_empty_bracket(self):
        teams = get_teams_in_bracket(BracketPrediction())
        assert len(teams) == 0


class TestIsTeamInBracket:
    def test_team_in_bracket(self):
        bracket = BracketPrediction(winner="Argentina")
        assert is_team_in_bracket("Argentina", bracket) is True

    def test_team_not_in_bracket(self):
        bracket = BracketPrediction(winner="Argentina")
        assert is_team_in_bracket("Iran", bracket) is False
