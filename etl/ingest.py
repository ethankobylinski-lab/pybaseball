"""Tools for ingesting MLB data from StatsAPI and Statcast.

This module exposes a small facade around the MLB StatsAPI so the rest of
this project can request schedules, box scores and play-by-play feeds without
having to know about HTTP details.  The ingestion layer is intentionally kept
thin: it performs HTTP calls, normalises responses and writes cached copies to
`data/raw` so repeated runs avoid hammering the upstream services.  The
functions accept start/end dates to make it easy to backfill or incrementally
update a data warehouse.

The module is written with loose coupling in mind.  The :class:`StatsAPIClient`
only depends on :mod:`requests` and uses dataclasses for configuration, which
means that swapping it out for a different provider (college, youth baseball)
just requires implementing the same public methods.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import json
import logging
import time

import requests


LOGGER = logging.getLogger(__name__)
RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class IngestionConfig:
    """Configuration for Stats API pulls.

    Attributes
    ----------
    base_url:
        Base URL for the Stats API.  The default points to the public MLB
        endpoint, but tests can override it.
    retry_attempts:
        Number of retry attempts for transient network errors.
    retry_backoff_seconds:
        Backoff between retries.  The implementation uses exponential
        backoff to avoid overwhelming the API.
    use_cache:
        When ``True`` responses are cached on disk and reused on subsequent
        calls.
    """

    base_url: str = "https://statsapi.mlb.com/api/v1"
    retry_attempts: int = 3
    retry_backoff_seconds: float = 1.5
    use_cache: bool = True


class StatsAPIClient:
    """Simple MLB StatsAPI wrapper with on-disk caching."""

    def __init__(self, config: Optional[IngestionConfig] = None) -> None:
        self.config = config or IngestionConfig()

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------
    def _cache_path(self, endpoint: str, params: Optional[Dict[str, Any]]) -> Path:
        key = endpoint.strip("/").replace("/", "_")
        if params:
            sorted_items = sorted(params.items())
            suffix = "_" + "_".join(f"{k}-{v}" for k, v in sorted_items)
        else:
            suffix = ""
        return RAW_DATA_DIR / f"{key}{suffix}.json"

    def _get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if self.config.use_cache:
            cache_path = self._cache_path(endpoint, params)
            if cache_path.exists():
                LOGGER.debug("Reading %s from cache", cache_path)
                return json.loads(cache_path.read_text())

        url = f"{self.config.base_url}/{endpoint.lstrip('/')}"
        attempt = 0
        while True:
            try:
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                payload = response.json()
                if self.config.use_cache:
                    cache_path = self._cache_path(endpoint, params)
                    cache_path.write_text(json.dumps(payload))
                return payload
            except requests.RequestException as exc:  # pragma: no cover - network
                attempt += 1
                if attempt > self.config.retry_attempts:
                    LOGGER.error("StatsAPI request failed after retries: %s", exc)
                    raise
                sleep_for = self.config.retry_backoff_seconds * attempt
                LOGGER.warning("StatsAPI error. Retrying in %.1fs", sleep_for)
                time.sleep(sleep_for)

    # ------------------------------------------------------------------
    # Public endpoints
    # ------------------------------------------------------------------
    def fetch_schedule(self, start_date: str, end_date: str, league_ids: Optional[Iterable[int]] = None) -> Dict[str, Any]:
        """Fetch the MLB schedule between two dates."""
        params: Dict[str, Any] = {"startDate": start_date, "endDate": end_date}
        if league_ids:
            params["leagueId"] = ",".join(str(i) for i in league_ids)
        return self._get("schedule", params=params)

    def fetch_game(self, game_pk: int) -> Dict[str, Any]:
        """Fetch the full game feed for a single game."""

        return self._get(f"game/{game_pk}/feed/live")

    def fetch_boxscore(self, game_pk: int) -> Dict[str, Any]:
        """Fetch the box score for a game."""

        return self._get(f"game/{game_pk}/boxscore")

    def fetch_play_by_play(self, game_pk: int) -> Dict[str, Any]:
        """Fetch the play-by-play data for a game."""

        return self._get(f"game/{game_pk}/playByPlay")


def fetch_games_between_dates(
    start_date: str,
    end_date: str,
    league_ids: Optional[Iterable[int]] = None,
    client: Optional[StatsAPIClient] = None,
) -> Dict[str, Dict[str, Any]]:
    """Collect game feeds for a date range.

    The function first requests the schedule and then pulls the live feed for
    each game.  Results are returned as a mapping ``{game_pk: feed}``, which is a
    convenient structure for downstream normalisation.
    """

    client = client or StatsAPIClient()
    schedule = client.fetch_schedule(start_date=start_date, end_date=end_date, league_ids=league_ids)
    games: Dict[str, Dict[str, Any]] = {}
    for date_info in schedule.get("dates", []):
        for game in date_info.get("games", []):
            game_pk = game["gamePk"]
            games[str(game_pk)] = client.fetch_game(game_pk)
    return games


__all__ = ["IngestionConfig", "StatsAPIClient", "fetch_games_between_dates"]
