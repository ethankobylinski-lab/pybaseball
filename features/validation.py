"""Simple data validations for canonical tables."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import polars as pl


@dataclass(slots=True)
class ValidationRule:
    """Definition for a validation rule."""

    column: str
    predicate: callable
    description: str


@dataclass(slots=True)
class ValidationResult:
    """Result of applying validation rules to a DataFrame."""

    passed: bool
    failing_rules: list[str]


def validate(frame: pl.DataFrame, rules: Iterable[ValidationRule]) -> ValidationResult:
    """Apply validation rules and return a structured result."""

    failing: list[str] = []
    for rule in rules:
        series = frame.get_column(rule.column) if rule.column in frame.columns else pl.Series(rule.column, [])
        if not rule.predicate(series):
            failing.append(rule.description)
    return ValidationResult(passed=not failing, failing_rules=failing)


__all__ = ["ValidationRule", "ValidationResult", "validate"]
