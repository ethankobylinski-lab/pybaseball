from __future__ import annotations

import pandas as pd
import pytest
import requests

from pybaseball.statcast_fielding import (
    statcast_catcher_framing,
    statcast_catcher_poptime,
    statcast_fielding_run_value,
    statcast_outfield_catch_prob,
    statcast_outfield_directional_oaa,
    statcast_outfielder_jump,
    statcast_outs_above_average,
)


def _run_or_skip(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except requests.exceptions.RequestException as exc:  # pragma: no cover - depends on network
        pytest.skip(f"Statcast service unavailable during test: {exc}")


def test_statcast_outs_above_average() -> None:
    min_att = 50
    pos = "of"
    result: pd.DataFrame = _run_or_skip(statcast_outs_above_average, 2019, pos, min_att)

    assert result is not None
    assert not result.empty
    print(result)

    assert [result.year == 2019]
    assert len(result) > 0


def test_statcast_outs_above_average_view() -> None:
    min_att = 50
    pos = "of"
    view = "Pitcher"
    result: pd.DataFrame = _run_or_skip(statcast_outs_above_average, 2019, pos, min_att, view)

    assert result is not None
    assert not result.empty

    assert len(result) > 0


def test_statcast_outfield_directional_oaa() -> None:
    min_opp = 50
    result: pd.DataFrame = _run_or_skip(statcast_outfield_directional_oaa, 2019, min_opp)

    assert result is not None
    assert not result.empty

    # assert len(result.columns) == 13
    assert len(result) > 0
    assert len(result.loc[result.attempts < min_opp]) == 0


def test_statcast_outfield_catch_prob() -> None:
    min_opp = 25
    result: pd.DataFrame = _run_or_skip(statcast_outfield_catch_prob, 2019, min_opp)

    assert result is not None
    assert not result.empty

    assert len(result) > 0


def test_statcast_outfielder_jump() -> None:
    min_att = 50
    result: pd.DataFrame = _run_or_skip(statcast_outfielder_jump, 2019, min_att)

    assert result is not None
    assert not result.empty

    # assert len(result.columns) == 13
    assert len(result) > 0
    assert len(result.loc[result.n < min_att]) == 0


def test_statcast_catcher_poptime() -> None:
    min_2b_att = 5
    min_3b_att = 0
    result: pd.DataFrame = _run_or_skip(statcast_catcher_poptime, 2019, min_2b_att, min_3b_att)

    assert result is not None
    assert not result.empty

    assert len(result) > 0
    assert len(result.loc[result.pop_2b_sba_count < min_2b_att]) == 0
    assert len(result.loc[result.pop_3b_sba_count < min_3b_att]) == 0


def test_statcast_catcher_framing() -> None:
    min_called_p = 100
    result: pd.DataFrame = _run_or_skip(statcast_catcher_framing, 2019, min_called_p)

    assert result is not None
    assert not result.empty

    assert len(result) > 0
    assert len(result.loc[result.pitches < min_called_p]) == 0


def test_statcast_fielding_run_value() -> None:
    min_inn = 50
    pos = 4
    result: pd.DataFrame = _run_or_skip(statcast_fielding_run_value, 2019, pos, min_inn)
    assert result is not None
    assert not result.empty

    assert len(result) > 0


