from __future__ import annotations

import pytest

pl = pytest.importorskip("polars")

from features.pipeline import build_features
from features.registry import FeatureDefinition, FeatureRegistry


def _mock_tables() -> dict[str, pl.DataFrame]:
    games = pl.DataFrame(
        {
            "game_pk": [1],
            "date": ["2024-04-01"],
            "season": [2024],
            "home_id": [100],
            "away_id": [200],
            "venue": ["Demo Park"],
            "doubleheader_flag": [False],
            "day_night": ["day"],
            "temperature": [70],
        }
    )

    box_team = pl.DataFrame(
        {
            "game_pk": [1, 1],
            "team_id": [200, 100],
            "is_home": [False, True],
            "R": [5, 4],
            "H": [8, 7],
            "BB": [3, 2],
            "SO": [6, 7],
            "HR": [1, 0],
            "2B": [2, 1],
            "3B": [0, 0],
            "SB": [1, 0],
            "CS": [0, 0],
            "E": [1, 2],
            "LOB": [6, 7],
        }
    )

    pbp = pl.DataFrame(
        {
            "game_pk": [1, 1, 1, 1, 1],
            "inning": [1, 1, 2, 3, 3],
            "top_bottom": ["top", "bottom", "top", "bottom", "bottom"],
            "half_inning": ["1T", "1B", "2T", "3B", "3B"],
            "at_bat_index": [1, 1, 1, 1, 2],
            "batter_id": [1001, 2001, 1002, 2002, 2003],
            "pitcher_id": [5001, 6001, 5002, 6002, 6002],
            "event": ["Home Run", "Sac Fly", "Groundout", "Double", "Single"],
            "description": ["3-run HR", "Sac fly", "Groundout", "RBI Double", "RBI Single"],
            "runs_scored": [3, 1, 1, 2, 1],
            "outs_before": [2, 0, 0, 1, 1],
            "before_base_state": [
                (1, 1, 0),
                (0, 0, 0),
                (0, 0, 0),
                (0, 1, 0),
                (0, 0, 0),
            ],
            "rbi": [3, 1, 0, 2, 1],
        }
    )

    return {"games": games, "box_team": box_team, "pbp": pbp}


def test_feature_views_produce_expected_columns():
    frames = build_features(_mock_tables())

    assert set(frames.scored_first.columns) == {"game_pk", "team_id", "scored_first"}
    assert set(frames.big_inning.columns) == {"game_pk", "team_id", "big_inning"}
    assert set(frames.answered_back.columns) == {"game_pk", "team_id", "answered_back_next_half"}
    assert set(frames.shutdown_after.columns) == {"game_pk", "team_id", "shutdown_inning_after_scoring"}
    assert set(frames.two_out_runs.columns) == {"game_pk", "team_id", "two_out_runs", "total_runs", "two_out_run_share"}
    assert set(frames.boxscore_diffs.columns) == {
        "game_pk",
        "team_id",
        "hits_diff",
        "bb_diff",
        "k_diff",
        "hr_diff",
        "xbh_diff",
        "sb_diff",
        "e_diff",
        "lob_diff",
    }


def test_scored_first_logic():
    frames = build_features(_mock_tables())
    scored = frames.scored_first.to_dicts()
    away = next(row for row in scored if row["team_id"] == 200)
    home = next(row for row in scored if row["team_id"] == 100)
    assert away["scored_first"] == 1
    assert home["scored_first"] == 0


def test_big_inning_and_two_out_share():
    frames = build_features(_mock_tables())
    big = frames.big_inning.to_dicts()
    away_big = next(row for row in big if row["team_id"] == 200)
    assert away_big["big_inning"] == 1
    two_out = frames.two_out_runs.to_dicts()
    away_two_out = next(row for row in two_out if row["team_id"] == 200)
    assert away_two_out["two_out_runs"] == 3
    assert away_two_out["two_out_run_share"] == 0.75


def test_shutdown_and_answered_back():
    frames = build_features(_mock_tables())
    answered = frames.answered_back.to_dicts()
    away_answered = next(row for row in answered if row["team_id"] == 200)["answered_back_next_half"]
    home_answered = next(row for row in answered if row["team_id"] == 100)["answered_back_next_half"]
    assert away_answered == 1
    assert home_answered == 1


def test_include_allows_subset_materialisation():
    frames = build_features(_mock_tables(), include=["scored_first"])

    assert "scored_first" in frames
    assert "big_inning" not in list(frames)


def test_custom_registry_can_alias_features():
    registry = FeatureRegistry()
    registry.register(
        FeatureDefinition(
            key="score_first_alias",
            view_name="feature_scored_first",
            description="Alias for scored first metric",
        )
    )
    frames = build_features(_mock_tables(), include=["score_first_alias"], registry=registry)
    assert "score_first_alias" in frames
    assert set(frames["score_first_alias"].columns) == {"game_pk", "team_id", "scored_first"}
