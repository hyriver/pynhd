import pytest

import pynhd
from pynhd import InputRangeError, ZeroMatchedError


def test_basin_empty():
    with pytest.raises(ZeroMatchedError) as ex:
        _ = pynhd.NLDI().get_basins(["04253294", "04253296"])
        assert "no features" in str(ex.value)


class TestGCXException:
    def test_wrong_bounds(self):
        with pytest.raises(InputRangeError) as ex:
            _ = pynhd.geoconnex(
                item="gages",
                query={"geometry": (1350626.862, 1687621.573, 1996597.949, 2557120.139)},
            )
        assert "(-170, 15, -51, 72)" in str(ex.value)
