"""Customized PyNHD exceptions."""
from typing import List

import async_retriever as ar


class ServiceError(ar.ServiceError):
    """Exception raised when the requested data is not available on the server.

    Parameters
    ----------
    err : str
        Service error message.
    """


class InvalidInputValue(ar.InvalidInputValue):
    """Exception raised for invalid input.

    Parameters
    ----------
    inp : str
        Name of the input parameter
    valid_inputs : tuple
        List of valid inputs
    """


class MissingItems(Exception):
    """Exception raised when a required item is missing.

    Parameters
    ----------
    missing : tuple
        The missing items.
    """

    def __init__(self, missing: List[str]) -> None:
        self.message = "The following items are missing:\n" + f"{', '.join(missing)}"
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class InvalidInputRange(Exception):
    """Exception raised when a function argument is not in the valid range.

    Parameters
    ----------
    variable : str
        Variable with invalid value
    valid_range : str
        Valid range
    """

    def __init__(self, variable: str, valid_range: str) -> None:
        self.message = f"Valid range for {variable} is {valid_range}."
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class MissingCRS(Exception):
    """Exception raised when CRS is not given."""

    def __init__(self) -> None:
        self.message = "CRS of the input geometry is missing."
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message
