"""Tests for PyNHD package."""
import io

import pytest

import pynhd as nhd
from pynhd import NLDI, WaterData

STA_ID = "01031500"
station_id = f"USGS-{STA_ID}"
site = "nwissite"
UM = "upstreamMain"
UT = "upstreamTributaries"
nldi = NLDI()


def test_nldi_urlonly():
    fsource = "comid"
    fid = "1722317"
    url_box = nldi.getfeature_byid(fsource, fid, url_only=True)
    url_nav = nldi.navigate_byid(fsource, fid, navigation="upstreamMain", url_only=True)
    assert (
        url_box == "https://labs.waterdata.usgs.gov/api/nldi/linked-data/comid/1722317"
        and url_nav
        == "https://labs.waterdata.usgs.gov/api/nldi/linked-data/comid/1722317/navigate/UM"
    )


@pytest.mark.flaky(max_runs=3)
def test_nldi_navigate():
    stm = nldi.navigate_byid(site, station_id, navigation=UM, source=site)
    st100 = nldi.navigate_byid(site, station_id, navigation=UM, source=site, distance=100)
    pp = nldi.navigate_byid(site, station_id, navigation=UT, source="huc12pp")
    assert st100.shape[0] == 2 and stm.shape[0] == 2 and pp.shape[0] == 12


@pytest.mark.flaky(max_runs=3)
def test_waterdata_byid():
    comids = nldi.navigate_byid(site, station_id, navigation=UT)
    comid_list = comids.nhdplus_comid.tolist()

    wd = WaterData("nhdflowline_network")
    trib = wd.byid("comid", comid_list)

    wd = WaterData("catchmentsp")
    ct = wd.byid("featureid", comid_list)

    assert (
        trib.shape[0] == 432
        and abs(trib.lengthkm.sum() - 565.755) < 1e-3
        and abs(ct.areasqkm.sum() - 773.954) < 1e-3
    )


@pytest.mark.flaky(max_runs=3)
def test_nldi_feature():
    basin = nldi.getfeature_byid(site, station_id, basin=True)
    eck4 = "+proj=eck4 +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    assert abs(basin.to_crs(eck4).area.iloc[0] - 774170100.273) < 1e-3


@pytest.mark.flaky(max_runs=3)
def test_waterdata_bybox():
    wd = WaterData("nhdwaterbody")
    print(wd)
    wb = wd.bybox((-69.7718, 45.0742, -69.3141, 45.4534))
    assert abs(wb.areasqkm.sum() - 87.181) < 1e-3


@pytest.mark.flaky(max_runs=3)
def test_acc():
    wd = WaterData("nhdflowline_network")
    comids = nldi.navigate_byid("nwissite", "USGS-11092450", UT)
    comid_list = comids.nhdplus_comid.tolist()
    trib = wd.byid("comid", comid_list)

    nhd.prepare_nhdplus(trib, 0, 0, 0, False, False)
    flw = nhd.prepare_nhdplus(trib, 1, 1, 1, True, True)

    def routing(qin, q):
        return qin + q

    qsim = nhd.vector_accumulation(
        flw[["comid", "tocomid", "lengthkm"]], routing, "lengthkm", ["lengthkm"],
    )
    flw = flw.merge(qsim, on="comid")
    diff = flw.arbolatesu - flw.acc

    assert diff.abs().sum() < 1e-5


def test_show_versions():
    f = io.StringIO()
    nhd.show_versions(file=f)
    assert "INSTALLED VERSIONS" in f.getvalue()
