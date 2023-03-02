import pytest

import pynhd
from pynhd import NLDI, InputRangeError, InputValueError, ZeroMatchedError


class TestNLDI:
    nldi: NLDI = NLDI()

    def test_feature_missing(self, recwarn: pytest.WarningsRecorder):
        _ = self.nldi.feature_byloc([(45.2, -69.3), (-69.3, 45.2)])
        w = recwarn.pop(UserWarning)
        assert "[0]" in str(w.message)

    def test_basin_missing(self, recwarn: pytest.WarningsRecorder):
        _ = self.nldi.get_basins(["01031500", "00000000"])
        w = recwarn.pop(UserWarning)
        assert "[1]" in str(w.message)

    def test_basin_empty(self):
        with pytest.raises(ZeroMatchedError) as ex:
            _ = self.nldi.get_basins(["04253294", "04253296"])
            assert "no features" in str(ex.value)


class TestGCX:
    def test_no_item(self):
        with pytest.raises(InputValueError) as ex:
            _ = pynhd.geoconnex("wrong")
        assert "gages" in str(ex.value)

    def test_wrong_bounds(self):
        with pytest.raises(InputRangeError) as ex:
            _ = pynhd.geoconnex(
                item="gages",
                geometry1=(1350626.862, 1687621.573, 1996597.949, 2557120.139),
            )
        assert "(-170, 15, -51, 72)" in str(ex.value)
