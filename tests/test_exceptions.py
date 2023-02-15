import pytest

import pynhd
from pynhd import NLDI, InputRangeError, ZeroMatchedError


class TestNLDI:
    nldi: NLDI = NLDI()

    def test_feature_missing(self, recwarn: pytest.WarningsRecorder):
        _ = self.nldi.feature_byloc([(45.2, -69.3), (-69.3, 45.2)])
        w = recwarn.pop(UserWarning)
        assert "coords=POINT(45.200000 -69.300000)" in str(w.message)

    def test_basin_missing(self, recwarn: pytest.WarningsRecorder):
        _ = self.nldi.get_basins(["01031500", "00000000"])
        w = recwarn.pop(UserWarning)
        assert "USGS-00000000" in str(w.message)

    def test_basin_empty(self):
        with pytest.raises(ZeroMatchedError) as ex:
            _ = self.nldi.get_basins(["04253294", "04253296"])
            assert "no features" in str(ex.value)


class TestGCXException:
    def test_wrong_bounds(self):
        with pytest.raises(InputRangeError) as ex:
            _ = pynhd.geoconnex(
                item="gages",
                query={"geometry": (1350626.862, 1687621.573, 1996597.949, 2557120.139)},
            )
        assert "(-170, 15, -51, 72)" in str(ex.value)
