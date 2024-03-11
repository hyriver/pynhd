"""Configuration for pytest."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _add_standard_imports(doctest_namespace):
    """Add pynhd namespace for doctest."""
    import pynhd as nhd

    doctest_namespace["nhd"] = nhd
