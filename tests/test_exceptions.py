import pytest

import pynhd
from pynhd import InvalidInputRange


class TestGCXException:
    def test_wrong_bounds(self):
        with pytest.raises(InvalidInputRange) as ex:
            _ = pynhd.geoconnex(
                item="gages",
                query={"geometry": (1350626.862, 1687621.573, 1996597.949, 2557120.139)},
            )
        assert "(-170, 15, -51, 72)" in str(ex.value)
