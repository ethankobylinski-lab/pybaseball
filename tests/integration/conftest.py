"""Pytest configuration for integration tests.

Automatically skips integration tests that require external baseball data
services when those services are unreachable (e.g., in offline CI).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import pytest
import requests

CHECK_URLS: tuple[str, ...] = (
    "https://statsapi.mlb.com/api/v1/teams?sportId=1",
    "https://www.fangraphs.com/",
)
TIMEOUT_SECONDS = 3


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip integration tests when required external services are unavailable."""
    if _services_available(CHECK_URLS):
        return

    skip_marker = pytest.mark.skip(reason="External baseball data services unavailable")
    for item in items:
        if item.nodeid.startswith("tests/integration/"):
            item.add_marker(skip_marker)


@lru_cache(maxsize=1)
def _services_available(urls: Iterable[str]) -> bool:
    for url in urls:
        try:
            response = requests.get(url, timeout=TIMEOUT_SECONDS)
        except requests.RequestException:
            return False
        if response.status_code != requests.codes.ok:
            return False
    return True
