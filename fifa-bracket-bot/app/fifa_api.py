"""
FIFA match data provider abstraction.

Implements:
  - MatchProvider (ABC)
  - ApiFootballProvider
  - FotMobProvider
  - EspnProvider
  - FallbackProvider  (tries providers in order)
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import requests

from .models import Match, MatchEvent, MatchStatus, RoundType, Team

logger = logging.getLogger(__name__)

_KNOCKOUT_START_DATE = date(2026, 6, 28)

_ROUND_MAP: dict[str, RoundType] = {
    "group stage": RoundType.GROUP_STAGE,
    "group": RoundType.GROUP_STAGE,
    "round of 16": RoundType.ROUND_OF_16,
    "last 16": RoundType.ROUND_OF_16,
    "quarterfinals": RoundType.QUARTERFINAL,
    "quarter-finals": RoundType.QUARTERFINAL,
    "quarterfinal": RoundType.QUARTERFINAL,
    "semifinals": RoundType.SEMIFINAL,
    "semi-finals": RoundType.SEMIFINAL,
    "semifinal": RoundType.SEMIFINAL,
    "final": RoundType.FINAL,
}

_REQUEST_TIMEOUT = 15  # seconds


def _get_with_retry(
    url: str,
    headers: Optional[dict] = None,
    params: Optional[dict] = None,
    retries: int = 3,
    backoff: float = 2.0,
) -> requests.Response:
    """HTTP GET with exponential-backoff retry."""
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(
                url, headers=headers, params=params, timeout=_REQUEST_TIMEOUT
            )
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            logger.warning("Request to %s failed (attempt %d/%d): %s", url, attempt, retries, exc)
            if attempt < retries:
                time.sleep(backoff**attempt)
    raise RuntimeError(f"All {retries} attempts to {url} failed")


def _parse_round(raw: str) -> RoundType:
    return _ROUND_MAP.get(raw.lower().strip(), RoundType.GROUP_STAGE)


def _normalize_round_for_date(target_date: date, parsed_round: RoundType) -> RoundType:
    """Force pre-knockout matches to remain group-stage through June 27, 2026."""
    if target_date < _KNOCKOUT_START_DATE:
        return RoundType.GROUP_STAGE
    return parsed_round


class MatchProvider(ABC):
    """Abstract base class for FIFA match data providers."""

    @abstractmethod
    def get_matches_for_date(self, target_date: date) -> list[Match]:
        """Return all matches on *target_date* (local calendar date)."""

    def get_yesterday_matches(self) -> list[Match]:
        yesterday = date.today() - timedelta(days=1)
        return self.get_matches_for_date(yesterday)

    def get_today_matches(self) -> list[Match]:
        return self.get_matches_for_date(date.today())


# ---------------------------------------------------------------------------
# API-Football (api-football.com / RapidAPI)
# ---------------------------------------------------------------------------

class ApiFootballProvider(MatchProvider):
    """
    Provider backed by api-football.com (available via RapidAPI).

    Requires:
      api_key  – X-RapidAPI-Key or X-Auth-Token
    """

    BASE_URL = "https://v3.football.api-sports.io"
    WORLD_CUP_ID = 1  # FIFA World Cup league id on api-football

    def __init__(self, api_key: str) -> None:
        self._headers = {
            "x-rapidapi-host": "v3.football.api-sports.io",
            "x-rapidapi-key": api_key,
        }

    def get_matches_for_date(self, target_date: date) -> list[Match]:
        params = {
            "league": self.WORLD_CUP_ID,
            "season": target_date.year,
            "date": target_date.isoformat(),
        }
        resp = _get_with_retry(
            f"{self.BASE_URL}/fixtures",
            headers=self._headers,
            params=params,
        )
        data = resp.json()
        matches: list[Match] = []
        for fixture in data.get("response", []):
            try:
                matches.append(self._parse_fixture(fixture, target_date))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to parse fixture: %s", exc)
        return matches

    def _parse_fixture(self, fixture: dict, target_date: date) -> Match:
        f = fixture["fixture"]
        league = fixture["league"]
        teams = fixture["teams"]
        goals = fixture["goals"]

        match_dt = datetime.fromisoformat(f["date"]) if f.get("date") else datetime.now(tz=timezone.utc)
        status_raw = f.get("status", {}).get("short", "NS")
        status = {
            "FT": MatchStatus.COMPLETED,
            "1H": MatchStatus.LIVE,
            "2H": MatchStatus.LIVE,
            "HT": MatchStatus.LIVE,
            "NS": MatchStatus.SCHEDULED,
            "PST": MatchStatus.POSTPONED,
            "CANC": MatchStatus.CANCELLED,
        }.get(status_raw, MatchStatus.SCHEDULED)
        parsed_round = _parse_round(league.get("round", ""))

        return Match(
            match_id=str(f["id"]),
            home_team=Team(name=teams["home"]["name"], code=teams["home"].get("code", "")),
            away_team=Team(name=teams["away"]["name"], code=teams["away"].get("code", "")),
            round_type=_normalize_round_for_date(target_date, parsed_round),
            match_datetime=match_dt,
            status=status,
            home_score=goals.get("home"),
            away_score=goals.get("away"),
            venue=f.get("venue", {}).get("name", ""),
            match_label=league.get("round", ""),
        )


# ---------------------------------------------------------------------------
# FotMob (unofficial public API)
# ---------------------------------------------------------------------------

class FotMobProvider(MatchProvider):
    """Provider backed by the FotMob public API."""

    WORLD_CUP_LEAGUE_ID = 77  # FotMob league id for FIFA World Cup

    def __init__(self, base_url: str = "https://www.fotmob.com/api") -> None:
        self._base_url = base_url.rstrip("/")

    def get_matches_for_date(self, target_date: date) -> list[Match]:
        params = {"date": target_date.strftime("%Y%m%d")}
        resp = _get_with_retry(f"{self._base_url}/matches", params=params)
        data = resp.json()
        matches: list[Match] = []
        for league in data.get("leagues", []):
            if league.get("id") != self.WORLD_CUP_LEAGUE_ID:
                continue
            for match_raw in league.get("matches", []):
                try:
                    matches.append(self._parse_match(match_raw, target_date))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("FotMob parse error: %s", exc)
        return matches

    def _parse_match(self, raw: dict, target_date: date) -> Match:
        status_raw = raw.get("status", {})
        finished = status_raw.get("finished", False)
        started = status_raw.get("started", False)

        if finished:
            status = MatchStatus.COMPLETED
        elif started:
            status = MatchStatus.LIVE
        else:
            status = MatchStatus.SCHEDULED

        utc_ts = raw.get("time", {}).get("utcTime", "")
        match_dt = datetime.fromisoformat(utc_ts.replace("Z", "+00:00")) if utc_ts else datetime.now(tz=timezone.utc)

        score_home = raw.get("home", {}).get("score")
        score_away = raw.get("away", {}).get("score")
        parsed_round = _parse_round(raw.get("roundName", ""))

        return Match(
            match_id=str(raw.get("id", "")),
            home_team=Team(name=raw.get("home", {}).get("name", "Unknown")),
            away_team=Team(name=raw.get("away", {}).get("name", "Unknown")),
            round_type=_normalize_round_for_date(target_date, parsed_round),
            match_datetime=match_dt,
            status=status,
            home_score=int(score_home) if score_home is not None else None,
            away_score=int(score_away) if score_away is not None else None,
        )


# ---------------------------------------------------------------------------
# ESPN (public API)
# ---------------------------------------------------------------------------

class EspnProvider(MatchProvider):
    """Provider backed by ESPN's public soccer API."""

    def __init__(self, base_url: str = "https://site.api.espn.com/apis/site/v2/sports/soccer") -> None:
        self._base_url = base_url.rstrip("/")

    def get_matches_for_date(self, target_date: date) -> list[Match]:
        date_str = target_date.strftime("%Y%m%d")
        url = f"{self._base_url}/fifa.world/scoreboard"
        params = {"dates": date_str}
        resp = _get_with_retry(url, params=params)
        data = resp.json()
        matches: list[Match] = []
        for event in data.get("events", []):
            try:
                matches.append(self._parse_event(event, target_date))
            except Exception as exc:  # noqa: BLE001
                logger.warning("ESPN parse error: %s", exc)
        return matches

    def _parse_event(self, event: dict, target_date: date) -> Match:
        competition = event.get("competitions", [{}])[0]
        competitors = competition.get("competitors", [])
        home = next((c for c in competitors if c.get("homeAway") == "home"), {})
        away = next((c for c in competitors if c.get("homeAway") == "away"), {})

        status_type = competition.get("status", {}).get("type", {}).get("name", "")
        if status_type in ("STATUS_FINAL", "STATUS_FULL_TIME"):
            status = MatchStatus.COMPLETED
        elif status_type in ("STATUS_IN_PROGRESS", "STATUS_HALFTIME"):
            status = MatchStatus.LIVE
        else:
            status = MatchStatus.SCHEDULED

        date_str = event.get("date", "")
        match_dt = datetime.fromisoformat(date_str.replace("Z", "+00:00")) if date_str else datetime.now(tz=timezone.utc)

        try:
            home_score = int(home.get("score", "")) if home.get("score") else None
        except ValueError:
            home_score = None
        try:
            away_score = int(away.get("score", "")) if away.get("score") else None
        except ValueError:
            away_score = None

        round_raw = competition.get("notes", [{}])[0].get("headline", "") if competition.get("notes") else ""
        parsed_round = _parse_round(round_raw)

        return Match(
            match_id=str(event.get("id", "")),
            home_team=Team(
                name=home.get("team", {}).get("displayName", "Unknown"),
                code=home.get("team", {}).get("abbreviation", ""),
            ),
            away_team=Team(
                name=away.get("team", {}).get("displayName", "Unknown"),
                code=away.get("team", {}).get("abbreviation", ""),
            ),
            round_type=_normalize_round_for_date(target_date, parsed_round),
            match_datetime=match_dt,
            status=status,
            home_score=home_score,
            away_score=away_score,
            venue=competition.get("venue", {}).get("fullName", ""),
        )


# ---------------------------------------------------------------------------
# Fallback / chain provider
# ---------------------------------------------------------------------------

class FallbackProvider(MatchProvider):
    """
    Tries providers in priority order.

    Returns results from the first provider that succeeds without error.
    """

    def __init__(self, providers: list[MatchProvider]) -> None:
        if not providers:
            raise ValueError("At least one provider must be supplied")
        self._providers = providers

    def get_matches_for_date(self, target_date: date) -> list[Match]:
        last_exc: Optional[Exception] = None
        for provider in self._providers:
            try:
                results = provider.get_matches_for_date(target_date)
                logger.info(
                    "Provider %s returned %d matches for %s",
                    type(provider).__name__,
                    len(results),
                    target_date,
                )
                return results
            except Exception as exc:  # noqa: BLE001
                logger.warning("Provider %s failed: %s", type(provider).__name__, exc)
                last_exc = exc
        raise RuntimeError(
            f"All data providers failed for {target_date}. Last error: {last_exc}"
        )


def build_provider(config) -> MatchProvider:  # type: ignore[type-arg]
    """Factory that builds a FallbackProvider from config."""
    providers: list[MatchProvider] = []
    for name in config.data_providers:
        name = name.strip().lower()
        if name == "api_football" and config.api_football_key:
            providers.append(ApiFootballProvider(config.api_football_key))
        elif name == "fotmob":
            providers.append(FotMobProvider(config.fotmob_base_url))
        elif name == "espn":
            providers.append(EspnProvider(config.espn_base_url))
        else:
            logger.debug("Skipping unknown or unconfigured provider: %s", name)

    if not providers:
        # Default to ESPN which requires no API key
        providers.append(EspnProvider())

    return FallbackProvider(providers)
