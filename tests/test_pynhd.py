"""Tests for PyNHD package."""
import io

import pytest
from shapely.geometry import box

import pynhd as nhd
from pynhd import NLDI, WaterData

STA_ID = "01031500"
station_id = f"USGS-{STA_ID}"
site = "nwissite"
UM = "upstreamMain"
UT = "upstreamTributaries"
nldi = NLDI()


@pytest.mark.flaky(max_runs=3)
def test_nldi_navigate():
    stm = nldi.navigate_byid(site, station_id, UM, site)
    st100 = nldi.navigate_byid(site, station_id, UM, site, distance=100)
    pp = nldi.navigate_byid(site, station_id, UT, "huc12pp")
    wqp = nldi.navigate_byloc((-70, 44), UT, "wqp")
    assert (
        st100.shape[0] == 2
        and stm.shape[0] == 2
        and pp.shape[0] == 12
        and wqp.comid.iloc[0] == "6710923"
    )


@pytest.mark.flaky(max_runs=3)
def test_nldi_feature():
    basin = nldi.getfeature_byid(site, station_id, basin=True)
    assert abs(basin.area.iloc[0] - 0.0887) < 1e-3


@pytest.mark.flaky(max_runs=3)
def test_nldi_char():
    tot, prc = nldi.getcharacteristic_byid("6710923", "tot", char_ids="TOT_BFI", values_only=False)
    assert abs(tot.TOT_BFI.values[0] - 57) < 1e-3 and prc.TOT_BFI.values[0] == 0


@pytest.mark.flaky(max_runs=3)
def test_nldi_chardf():
    bfi = nldi.characteristics_dataframe("tot", "TOT_BFI", "BFI_CONUS.zip")
    meta = nldi.characteristics_dataframe("tot", "TOT_BFI", "BFI_CONUS.zip", metadata=True)
    assert (
        abs(bfi[bfi.ACC_BFI > 0].ACC_BFI.sum() - 116653087.67) < 1e-3
        and meta["id"] == "5669a8e3e4b08895842a1d4f"
    )


@pytest.mark.flaky(max_runs=3)
def test_waterdata_byid():
    comids = nldi.navigate_byid(site, station_id, UT, "flowlines")
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
def test_waterdata_bybox():
    wd = WaterData("nhdwaterbody")
    print(wd)
    wb_g = wd.bygeom(box(-69.7718, 45.0742, -69.3141, 45.4534), predicate="INTERSECTS", xy=True)
    wb_b = wd.bybox((-69.7718, 45.0742, -69.3141, 45.4534))
    assert abs(wb_b.areasqkm.sum() - wb_g.areasqkm.sum()) < 1e-3


@pytest.mark.flaky(max_runs=3)
def test_waterdata_byfilter():
    wd = WaterData("huc12", "epsg:3857")
    wb = wd.byfilter(f"{wd.layer} LIKE '17030001%'")
    assert wb.shape[0] == 52


@pytest.mark.flaky(max_runs=3)
def test_nhdphr():
    hr = nhd.NHDPlusHR("networknhdflowline")
    flwb = hr.bygeom((-69.77, 45.07, -69.31, 45.45))
    flwi = hr.byids("NHDPLUSID", ["5000500013223", "5000400039708", "5000500004825"])
    flwf = hr.bysql("NHDPLUSID IN (5000500013223, 5000400039708, 5000500004825)")
    assert flwb.shape[0] == 3887 and flwi.OBJECTID.tolist() == flwf.OBJECTID.tolist()


@pytest.mark.flaky(max_runs=3)
def test_acc():
    wd = WaterData("nhdflowline_network")
    comids = nldi.navigate_byid("nwissite", "USGS-11092450", UT, "flowlines")
    comid_list = comids.nhdplus_comid.tolist()
    trib = wd.byid("comid", comid_list)

    nhd.prepare_nhdplus(trib, 0, 0, 0, False, False)
    flw = nhd.prepare_nhdplus(trib, 1, 1, 1, True, True)

    def routing(qin, q):
        return qin + q

    qsim = nhd.vector_accumulation(
        flw[["comid", "tocomid", "lengthkm"]],
        routing,
        "lengthkm",
        ["lengthkm"],
    )
    flw = flw.merge(qsim, on="comid")
    diff = flw.arbolatesu - flw.acc_lengthkm

    assert diff.abs().sum() < 1e-5


def test_show_versions():
    f = io.StringIO()
    nhd.show_versions(file=f)
    assert "INSTALLED VERSIONS" in f.getvalue()
