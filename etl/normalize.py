"""Normalisation utilities for canonical MLB tables."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import polars as pl


@dataclass(slots=True)
class NormalisedData:
    """Container holding canonical tables as Polars DataFrames."""

    games: pl.DataFrame
    pbp: pl.DataFrame
    box_team: pl.DataFrame
    box_batting: pl.DataFrame
    box_pitching: pl.DataFrame


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_int(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _base_out_state(play: Dict[str, object]) -> Tuple[int, int, int]:
    """Return the base occupancy (first, second, third) for a play."""

    runners = play.get("runners", []) or []
    occupied = {runner.get("startingBase") for runner in runners if runner.get("movement")}
    return (1 if 1 in occupied else 0, 1 if 2 in occupied else 0, 1 if 3 in occupied else 0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def normalise_games(game_feeds: Dict[str, Dict[str, object]]) -> NormalisedData:
    """Normalise StatsAPI game feeds into canonical tables."""

    games: List[pl.DataFrame] = []
    pbp_rows: List[Dict[str, object]] = []
    team_rows: List[Dict[str, object]] = []
    batting_rows: List[Dict[str, object]] = []
    pitching_rows: List[Dict[str, object]] = []

    for game_pk, feed in game_feeds.items():
        game_data = feed.get("gameData", {})
        datetime = game_data.get("datetime", {})
        status = feed.get("gameData", {}).get("status", {})
        venue = game_data.get("venue", {})
        weather = game_data.get("weather", {})
        teams = feed.get("liveData", {}).get("boxscore", {}).get("teams", {})

        games.append(
            pl.DataFrame(
                {
                    "game_pk": [int(game_pk)],
                    "date": [datetime.get("officialDate")],
                    "season": [_safe_int(game_data.get("game", {}).get("season"))],
                    "home_id": [_safe_int(game_data.get("teams", {}).get("home", {}).get("id"))],
                    "away_id": [_safe_int(game_data.get("teams", {}).get("away", {}).get("id"))],
                    "venue": [venue.get("name")],
                    "doubleheader_flag": [status.get("isDoubleHeader", "N") == "Y"],
                    "day_night": [game_data.get("datetime", {}).get("dayNight")],
                    "temperature": [weather.get("temp")],
                }
            )
        )

        # Play-by-play
        all_plays: Iterable[Dict[str, object]] = feed.get("liveData", {}).get("plays", {}).get("allPlays", [])
        for play in all_plays:
            runners = play.get("runners", [])
            before = _base_out_state(play)
            after = tuple(1 if play.get("about", {}).get("isTopInning") else 0 for _ in range(3))
            pbp_rows.append(
                {
                    "game_pk": int(game_pk),
                    "inning": play.get("about", {}).get("inning"),
                    "top_bottom": "top" if play.get("about", {}).get("isTopInning") else "bottom",
                    "half_inning": f"{play.get('about', {}).get('inning')}" + ("T" if play.get("about", {}).get("isTopInning") else "B"),
                    "at_bat_index": play.get("atBatIndex"),
                    "batter_id": _safe_int(play.get("matchup", {}).get("batter", {}).get("id")),
                    "pitcher_id": _safe_int(play.get("matchup", {}).get("pitcher", {}).get("id")),
                    "event": play.get("result", {}).get("event"),
                    "description": play.get("result", {}).get("description"),
                    "runs_scored": play.get("result", {}).get("awayScore", 0) + play.get("result", {}).get("homeScore", 0),
                    "outs_before": play.get("count", {}).get("outs"),
                    "before_base_state": before,
                    "rbi": sum(1 for runner in runners if runner.get("details", {}).get("isScoringEvent")),
                }
            )

        for side in ("home", "away"):
            team = teams.get(side, {})
            team_rows.append(
                {
                    "game_pk": int(game_pk),
                    "team_id": _safe_int(team.get("team", {}).get("id")),
                    "is_home": side == "home",
                    "R": team.get("teamStats", {}).get("batting", {}).get("runs"),
                    "H": team.get("teamStats", {}).get("batting", {}).get("hits"),
                    "BB": team.get("teamStats", {}).get("batting", {}).get("baseOnBalls"),
                    "SO": team.get("teamStats", {}).get("batting", {}).get("strikeOuts"),
                    "HR": team.get("teamStats", {}).get("batting", {}).get("homeRuns"),
                    "2B": team.get("teamStats", {}).get("batting", {}).get("doubles"),
                    "3B": team.get("teamStats", {}).get("batting", {}).get("triples"),
                    "SB": team.get("teamStats", {}).get("batting", {}).get("stolenBases"),
                    "CS": team.get("teamStats", {}).get("batting", {}).get("caughtStealing"),
                    "E": team.get("teamStats", {}).get("fielding", {}).get("errors"),
                    "LOB": team.get("teamStats", {}).get("batting", {}).get("leftOnBase"),
                }
            )

            people = team.get("players", {})
            for player_id, player in people.items():
                stats = player.get("stats", {})
                batting = stats.get("batting", {})
                pitching = stats.get("pitching", {})
                if batting:
                    batting_rows.append(
                        {
                            "game_pk": int(game_pk),
                            "team_id": _safe_int(team.get("team", {}).get("id")),
                            "player_id": _safe_int(player.get("person", {}).get("id")),
                            "is_home": side == "home",
                            "position": player.get("position", {}).get("abbreviation"),
                            "starter": player.get("gameStatus", {}).get("isCurrentBatter"),
                            "AB": batting.get("atBats"),
                            "H": batting.get("hits"),
                            "BB": batting.get("baseOnBalls"),
                            "SO": batting.get("strikeOuts"),
                            "HR": batting.get("homeRuns"),
                        }
                    )
                if pitching:
                    pitching_rows.append(
                        {
                            "game_pk": int(game_pk),
                            "team_id": _safe_int(team.get("team", {}).get("id")),
                            "player_id": _safe_int(player.get("person", {}).get("id")),
                            "is_home": side == "home",
                            "starter": player.get("gameStatus", {}).get("isCurrentPitcher"),
                            "IP": pitching.get("inningsPitched"),
                            "BF": pitching.get("battersFaced"),
                            "SO": pitching.get("strikeOuts"),
                            "BB": pitching.get("baseOnBalls"),
                            "HR": pitching.get("homeRuns"),
                        }
                    )

    return NormalisedData(
        games=pl.concat(games) if games else pl.DataFrame(),
        pbp=pl.DataFrame(pbp_rows),
        box_team=pl.DataFrame(team_rows),
        box_batting=pl.DataFrame(batting_rows),
        box_pitching=pl.DataFrame(pitching_rows),
    )


__all__ = ["NormalisedData", "normalise_games"]
