"""Registry utilities for managing feature view definitions.

The previous revision of the project returned a static :class:`FeatureFrames`
dataclass from :func:`features.pipeline.build_features`.  That approach made it
painful to introduce new metrics because every addition required touching the
dataclass, the pipeline implementation and the calling code.  The
``FeatureRegistry`` introduced here keeps feature metadata in one place and
provides a lightweight plug-in system so downstream consumers can request
arbitrary view combinations without modifying the pipeline internals.

Users can register new :class:`FeatureDefinition` objects at runtime, point the
registry at alternative DuckDB view names, or disable defaults when they want
to materialise only a subset of features.  The interactive app uses the
metadata to render friendly descriptions and groups, which keeps the UI in sync
with the analytical layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence


@dataclass(slots=True)
class FeatureDefinition:
    """Describe a single feature view available in the warehouse.

    Attributes
    ----------
    key:
        Unique identifier for the feature.  This is the handle callers will use
        when requesting frames from :func:`features.pipeline.build_features`.
    view_name:
        Name of the DuckDB view defined in ``etl/warehouse.sql``.
    description:
        Human-friendly summary of what the metric represents.  The UI surfaces
        this text directly, so keep it concise but descriptive.
    tags:
        Optional labels (e.g. ``["run-scoring"]``) that allow grouping related
        metrics in the interface.
    default:
        When ``True`` the feature is materialised as part of the default bundle.
        Users can opt out by passing ``include`` to ``build_features``.
    """

    key: str
    view_name: str
    description: str
    tags: Sequence[str] = field(default_factory=tuple)
    default: bool = True


class FeatureRegistry:
    """Container mapping feature keys to DuckDB view metadata."""

    def __init__(self, definitions: Optional[Iterable[FeatureDefinition]] = None) -> None:
        self._definitions: Dict[str, FeatureDefinition] = {}
        if definitions:
            self.register_many(definitions)

    # ------------------------------------------------------------------
    # Registration utilities
    # ------------------------------------------------------------------
    def register(self, definition: FeatureDefinition) -> None:
        """Register a single feature definition.

        If the key already exists it is overwritten, which makes it trivial to
        replace built-in metrics with custom versions.
        """

        self._definitions[definition.key] = definition

    def register_many(self, definitions: Iterable[FeatureDefinition]) -> None:
        for definition in definitions:
            self.register(definition)

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    def keys(self) -> List[str]:
        return list(self._definitions.keys())

    def definitions(self) -> List[FeatureDefinition]:
        return list(self._definitions.values())

    def get(self, key: str) -> FeatureDefinition:
        return self._definitions[key]

    def view_names(self, include: Optional[Iterable[str]] = None) -> List[str]:
        """Return DuckDB view names for selected features.

        Parameters
        ----------
        include:
            Iterable of feature keys to materialise.  When ``None`` all default
            features are returned.  The special token ``"*"`` requests every
            registered feature.
        """

        if include is None:
            definitions = [d for d in self._definitions.values() if d.default]
        else:
            include_list = list(include)
            if not include_list:
                return []
            if any(token == "*" for token in include_list):
                definitions = list(self._definitions.values())
            else:
                definitions = [self.get(key) for key in include_list]
        return [definition.view_name for definition in definitions]

    def metadata(self, include: Optional[Iterable[str]] = None) -> Dict[str, FeatureDefinition]:
        """Return mapping of feature key to definitions respecting ``include``."""

        if include is None:
            selected = [d for d in self._definitions.values() if d.default]
        else:
            include_list = list(include)
            if any(token == "*" for token in include_list):
                selected = list(self._definitions.values())
            else:
                selected = [self.get(key) for key in include_list]
        return {definition.key: definition for definition in selected}


def default_registry() -> FeatureRegistry:
    """Return the built-in feature registry used throughout the project."""

    registry = FeatureRegistry(
        definitions=[
            FeatureDefinition(
                key="scored_first",
                view_name="feature_scored_first",
                description="Indicator if the team scored before its opponent",
                tags=("game-flow",),
            ),
            FeatureDefinition(
                key="big_inning",
                view_name="feature_big_inning",
                description="Flag for any inning with at least three runs scored",
                tags=("game-flow",),
            ),
            FeatureDefinition(
                key="answered_back",
                view_name="feature_answered_back",
                description="Did the team score in the half-inning immediately after conceding",
                tags=("game-flow",),
            ),
            FeatureDefinition(
                key="shutdown_after",
                view_name="feature_shutdown_after",
                description="Rate of zero-run innings immediately following the team scoring",
                tags=("pitching", "game-flow"),
            ),
            FeatureDefinition(
                key="two_out_runs",
                view_name="feature_two_out_runs",
                description="Runs scored with two outs and their share of total production",
                tags=("situational",),
            ),
            FeatureDefinition(
                key="boxscore_diffs",
                view_name="feature_boxscore_diffs",
                description="Differentials derived from the traditional box score",
                tags=("boxscore",),
            ),
        ]
    )
    return registry


__all__ = ["FeatureDefinition", "FeatureRegistry", "default_registry"]
