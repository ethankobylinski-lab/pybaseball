"""Feature computation using DuckDB views defined in ``etl/warehouse.sql``."""
from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping
from pathlib import Path
from typing import Dict, Optional

import duckdb
import polars as pl

from .registry import FeatureRegistry, default_registry

WAREHOUSE_SQL = Path(__file__).resolve().parent.parent / "etl" / "warehouse.sql"


class FeatureCollection(Mapping[str, pl.DataFrame]):
    """Mapping-style container holding team-game base table plus feature frames."""

    def __init__(
        self,
        team_games: pl.DataFrame,
        features: MutableMapping[str, pl.DataFrame],
        definitions: Dict[str, str],
    ) -> None:
        self.team_games = team_games
        self._features = dict(features)
        self._definitions = definitions

    def __getitem__(self, key: str) -> pl.DataFrame:
        if key == "team_games":
            return self.team_games
        return self._features[key]

    def __iter__(self):
        yield "team_games"
        yield from self._features

    def __len__(self) -> int:
        return 1 + len(self._features)

    def __getattr__(self, item: str) -> pl.DataFrame:
        try:
            return self[item]
        except KeyError as exc:  # pragma: no cover - defensive branch
            raise AttributeError(item) from exc

    @property
    def metadata(self) -> Dict[str, str]:
        """Return mapping of feature key to description for UI tooling."""

        return self._definitions

    def to_dict(self) -> Dict[str, pl.DataFrame]:
        payload = {"team_games": self.team_games}
        payload.update(self._features)
        return payload


def _register_frame(con: duckdb.DuckDBPyConnection, name: str, frame: pl.DataFrame) -> None:
    con.register(name, frame.to_pandas())


def build_features(
    tables: Mapping[str, pl.DataFrame],
    include: Optional[Iterable[str]] = None,
    registry: Optional[FeatureRegistry] = None,
) -> FeatureCollection:
    """Build feature tables from canonical inputs.

    Parameters
    ----------
    tables:
        Mapping of table name to Polars DataFrame for the canonical tables
        ``games``, ``pbp`` and ``box_team``.
    include:
        Optional iterable of feature keys to materialise.  When ``None`` the
        default bundle from :func:`features.registry.default_registry` is used.
        The special token ``"*"`` includes every registered feature.
    registry:
        Custom registry instance.  When omitted the default registry is used.
    """

    con = duckdb.connect(database=":memory:")
    for table_name, frame in tables.items():
        _register_frame(con, table_name, frame)

    statements = WAREHOUSE_SQL.read_text().split(";\n")
    for statement in statements:
        stmt = statement.strip()
        if stmt:
            con.execute(stmt)

    registry = registry or default_registry()
    view_names = ["team_games", *registry.view_names(include=include)]

    outputs: Dict[str, pl.DataFrame] = {}
    for view in view_names:
        outputs[view] = pl.from_pandas(con.execute(f"SELECT * FROM {view}").df())

    feature_frames: Dict[str, pl.DataFrame] = {}
    metadata = registry.metadata(include=include)
    for key, definition in metadata.items():
        feature_frames[key] = outputs[definition.view_name]

    descriptions = {key: definition.description for key, definition in metadata.items()}

    return FeatureCollection(team_games=outputs["team_games"], features=feature_frames, definitions=descriptions)


__all__ = ["FeatureCollection", "build_features"]
