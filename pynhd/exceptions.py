"""Customized PyNHD exceptions."""
from typing import Generator, List, Optional, Union


class ZeroMatched(ValueError):
    """Exception raised when a function argument is missing."""


class InvalidInputValue(Exception):
    """Exception raised for invalid input.

    Parameters
    ----------
    inp : str
        Name of the input parameter
    valid_inputs : tuple
        List of valid inputs
    """

    def __init__(
        self, inp: str, valid_inputs: Union[List[str], Generator[str, None, None]]
    ) -> None:
        self.message = f"Given {inp} is invalid. Valid options are:\n" + "\n".join(
            str(i) for i in valid_inputs
        )
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class InvalidInputType(Exception):
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

    def __init__(self, arg: str, valid_type: str, example: Optional[str] = None) -> None:
        self.message = f"The {arg} argument should be of type {valid_type}"
        if example is not None:
            self.message += f":\n{example}"
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


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
