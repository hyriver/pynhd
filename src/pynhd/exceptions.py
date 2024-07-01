"""Customized PyNHD exceptions."""

from __future__ import annotations

from typing import Generator

import async_retriever as ar
import pygeoogc as ogc
import pygeoutils as pgu


class ZeroMatchedError(ogc.ZeroMatchedError):
    """Exception raised when a function argument is missing.

    Parameters
    ----------
    msg : str
        The exception error message
    """


class MissingColumnError(pgu.MissingColumnError):
    """Exception raised when a required column is missing from a dataframe.

    Parameters
    ----------
    missing : list
        List of missing columns.
    """


class ServiceError(ar.ServiceError):
    """Exception raised when the requested data is not available on the server.

    Parameters
    ----------
    err : str
        Service error message.
    """


class InputValueError(ar.InputValueError):
    """Exception raised for invalid input.

    Parameters
    ----------
    inp : str
        Name of the input parameter
    valid_inputs : tuple
        List of valid inputs
    given : str, optional
        The given input, defaults to None.
    """


class InputTypeError(ar.InputTypeError):
    """Exception raised when a function argument type is invalid.

    Parameters
    ----------
    arg : str
        Name of the function argument
    valid_type : str
        The valid type of the argument
    example : str, optional
        An example of a valid form of the argument, defaults to None.
    """


class MissingItemError(Exception):
    """Exception raised when a required item is missing.

    Parameters
    ----------
    missing : tuple
        The missing items.
    """

    def __init__(self, missing: list[str]) -> None:
        self.message = f"The following items are missing:\n{', '.join(missing)}"
        super().__init__(self.message)

    def __str__(self) -> str:
        """Return the error message."""
        return self.message


class InputRangeError(Exception):
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
        """Return the error message."""
        return self.message


class MissingCRSError(Exception):
    """Exception raised when CRS is not given."""

    def __init__(self) -> None:
        self.message = "CRS of the input geometry is missing."
        super().__init__(self.message)

    def __str__(self) -> str:
        """Return the error message."""
        return self.message


class NoTerminalError(Exception):
    """Exception raised when no terminal COMID is found."""

    def __init__(self) -> None:
        self.message = "No terminal COMID was found."
        super().__init__(self.message)

    def __str__(self) -> str:
        """Return the error message."""
        return self.message


class DependencyError(Exception):
    """Exception raised when a dependencies are not met.

    Parameters
    ----------
    libraries : tuple
        List of valid inputs
    """

    def __init__(self, func: str, libraries: str | list[str] | Generator[str, None, None]) -> None:
        libraries = [libraries] if isinstance(libraries, str) else libraries
        self.message = f"The following dependencies are missing for running {func}:\n"
        self.message += ", ".join(libraries)
        super().__init__(self.message)

    def __str__(self) -> str:
        """Return the error message."""
        return self.message
