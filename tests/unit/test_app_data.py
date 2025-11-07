from __future__ import annotations

import pytest

pl = pytest.importorskip("polars")

from app.data import WarehouseRepository


def test_repository_lists_and_loads(tmp_path):
    base = tmp_path / "warehouse"
    base.mkdir()

    games = pl.DataFrame({"game_pk": [1], "season": [2023], "date": ["2023-04-01"], "home_id": [1], "away_id": [2]})
    pbp = pl.DataFrame({"game_pk": [1], "inning": [1], "top_bottom": ["top"], "half_inning": ["1T"], "at_bat_index": [1], "runs_scored": [1]})
    box_team = pl.DataFrame({"game_pk": [1, 1], "team_id": [1, 2], "is_home": [True, False], "R": [3, 2], "H": [6, 5]})

    games.write_parquet(base / "games_2023.parquet")
    pbp.write_parquet(base / "pbp_2023.parquet")
    box_team.write_parquet(base / "box_team_2023.parquet")

    repo = WarehouseRepository(base_path=base)
    seasons = repo.available_seasons()
    assert seasons == ["2023"]

    tables = repo.load_tables(["2023"])
    assert set(tables.keys()) == {"games", "pbp", "box_team"}
    assert tables["games"].height == 1


def test_repository_handles_csv_snapshots(tmp_path):
    base = tmp_path / "warehouse"
    base.mkdir()

    games = pl.DataFrame({"game_pk": [2], "season": [2022], "date": ["2022-04-02"], "home_id": [3], "away_id": [4]})
    pbp = pl.DataFrame({
        "game_pk": [2],
        "inning": [1],
        "top_bottom": ["top"],
        "half_inning": ["1T"],
        "at_bat_index": [1],
        "runs_scored": [0],
    })
    box_team = pl.DataFrame({"game_pk": [2], "team_id": [3], "is_home": [True], "R": [5]})

    games.write_csv(base / "games_sample.csv")
    pbp.write_csv(base / "pbp_sample.csv")
    box_team.write_csv(base / "box_team_sample.csv")

    repo = WarehouseRepository(base_path=base)
    assert repo.available_seasons() == ["sample"]

    tables = repo.load_tables(["sample"])
    assert tables["games"].height == 1
