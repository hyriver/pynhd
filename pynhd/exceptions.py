"""Customized PyNHD exceptions."""
from typing import List


class ZeroMatched(ValueError):
    """Exception raised when a function doesn't return any feature."""


class MissingItems(Exception):
    """Exception raised when a required item is missing.

    Parameters
    ----------
    missing : tuple
        The server url
    """

    def __init__(self, missing: List[str]) -> None:
        self.message = "The following items are missing:\n" + f"{', '.join(missing)}"
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class InvalidInputRange(ValueError):
    """Exception raised when a function argument is not in the valid range."""
