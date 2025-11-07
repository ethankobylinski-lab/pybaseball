from __future__ import annotations

import pytest

pl = pytest.importorskip("polars")

from features.validation import ValidationRule, validate


def test_validate_success_and_failure():
    frame = pl.DataFrame({"value": [1, 2, 3]})
    rules = [
        ValidationRule("value", lambda s: s.min() >= 0, "values must be non-negative"),
        ValidationRule("value", lambda s: s.max() <= 3, "values must not exceed 3"),
    ]
    result = validate(frame, rules)
    assert result.passed
    assert result.failing_rules == []

    failing_rules = [ValidationRule("value", lambda s: s.max() < 3, "max under 3")]
    failed = validate(frame, failing_rules)
    assert not failed.passed
    assert failed.failing_rules == ["max under 3"]
