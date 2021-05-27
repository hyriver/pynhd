import pytest

from pynhd import InvalidInputType, InvalidInputValue, MissingItems, ZeroMatched, InvalidInputRange


def missing_items():
    raise MissingItems(["tmin", "dayl"])


def test_missing_items():
    with pytest.raises(MissingItems):
        missing_items()


def zero_matched():
    raise ZeroMatched("Query returned no matched objects.")


def test_zero_matched():
    with pytest.raises(ZeroMatched):
        zero_matched()


def invalid_value():
    raise InvalidInputValue("outFormat", ["json", "geojson"])


def test_invalid_value():
    with pytest.raises(InvalidInputValue):
        invalid_value()


def invalid_type():
    raise InvalidInputType("coords", "tuple", "(lon, lat)")


def test_invalid_type():
    with pytest.raises(InvalidInputType):
        invalid_type()


def invalid_range():
    raise InvalidInputRange("Input is out of range.")


def test_invalid_range():
    with pytest.raises(InvalidInputRange):
        invalid_range()
