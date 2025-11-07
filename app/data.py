"""Data access helpers for the interactive application."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, MutableMapping, Sequence, Tuple

import polars as pl


WAREHOUSE_DIR = Path(__file__).resolve().parent.parent / "data" / "warehouse"
WAREHOUSE_EXTENSIONS = (".parquet", ".csv")


def _season_token_from_path(path: Path, prefix: str) -> str:
    stem = path.stem
    if stem == prefix:
        return "all"
    suffix = stem.replace(f"{prefix}_", "", 1)
    return suffix


def _season_sort_key(token: str) -> Tuple[int, str]:
    if token.isdigit():
        return (0, f"{int(token):04d}")
    if token == "sample":
        return (1, token)
    return (2, token)


@dataclass(slots=True)
class WarehouseRepository:
    """Load canonical tables for one or more seasons from disk."""

    base_path: Path = WAREHOUSE_DIR
    tables: Sequence[str] = ("games", "pbp", "box_team")
    _cache: MutableMapping[Tuple[str, ...], Dict[str, pl.DataFrame]] = field(default_factory=dict)

    def available_seasons(self) -> List[str]:
        tokens = set()
        if not self.base_path.exists():
            return []
        for table in self.tables:
            for ext in WAREHOUSE_EXTENSIONS:
                pattern = f"{table}_*{ext}"
                for path in self.base_path.glob(pattern):
                    tokens.add(_season_token_from_path(path, table))
                single = self.base_path / f"{table}{ext}"
                if single.exists():
                    tokens.add("all")
        if not tokens:
            # fall back to historical sample naming convention
            for table in self.tables:
                for ext in WAREHOUSE_EXTENSIONS:
                    sample_path = self.base_path / f"{table}_sample{ext}"
                    if sample_path.exists():
                        tokens.add("sample")
                        break
        return sorted(tokens, key=_season_sort_key)

    def _resolve_path(self, table: str, token: str) -> Path:
        if token == "all":
            for ext in WAREHOUSE_EXTENSIONS:
                candidate = self.base_path / f"{table}{ext}"
                if candidate.exists():
                    return candidate
        for ext in WAREHOUSE_EXTENSIONS:
            path = self.base_path / f"{table}_{token}{ext}"
            if path.exists():
                return path
        raise FileNotFoundError(f"No {table} dataset available for token '{token}' in {self.base_path}")

    def load_tables(self, seasons: Sequence[str] | None = None) -> Dict[str, pl.DataFrame]:
        """Load canonical tables for the selected seasons.

        Parameters
        ----------
        seasons:
            Sequence of season tokens returned by :meth:`available_seasons`.  When
            ``None`` the most complete dataset is used (preferring ``all`` then
            numeric years and finally the sample snapshot).
        """

        if seasons is None or len(seasons) == 0:
            seasons = self.default_selection()

        key = tuple(sorted(seasons, key=_season_sort_key))
        if key in self._cache:
            return {name: frame.clone() for name, frame in self._cache[key].items()}

        tables: Dict[str, pl.DataFrame] = {}
        for table in self.tables:
            frames: List[pl.DataFrame] = []
            for season in key:
                try:
                    path = self._resolve_path(table, season)
                except FileNotFoundError:
                    continue
                if path.suffix == ".parquet":
                    frame = pl.read_parquet(path)
                elif path.suffix == ".csv":
                    frame = pl.read_csv(path)
                else:
                    raise ValueError(f"Unsupported warehouse file type: {path.suffix}")
                frames.append(frame)
            if not frames:
                tables[table] = pl.DataFrame()
            else:
                tables[table] = pl.concat(frames, how="vertical")

        self._cache[key] = tables
        return {name: frame.clone() for name, frame in tables.items()}

    def default_selection(self) -> List[str]:
        seasons = self.available_seasons()
        if "all" in seasons:
            return ["all"]
        numeric = [token for token in seasons if token.isdigit()]
        if numeric:
            return numeric
        if seasons:
            return [seasons[0]]
        return []


__all__ = ["WarehouseRepository"]
