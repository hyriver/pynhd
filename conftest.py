"""Configuration for pytest."""

import pytest


@pytest.fixture(autouse=True)
def add_standard_imports(doctest_namespace):
    """Add pynhd namespace for doctest."""
    import pynhd as nhd

    doctest_namespace["nhd"] = nhd
