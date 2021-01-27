"""Tests for PyNHD package."""
import io

import pytest
from shapely.geometry import box

import pynhd as nhd
from pynhd import NLDI, AGRBase, WaterData

STA_ID = "01031500"
station_id = f"USGS-{STA_ID}"
site = "nwissite"
UM = "upstreamMain"
UT = "upstreamTributaries"
nldi = NLDI()


class NHDPlusEPA(AGRBase):
    def __init__(self, layer):
        super().__init__(layer, "*", "epsg:4326")
        self.service = self._init_service(
            "https://watersgeo.epa.gov/arcgis/rest/services/NHDPlus/NHDPlus/MapServer"
        )


@pytest.mark.flaky(max_runs=3)
def test_agr():
    layer = "network flowline"
    sql_clause = "FTYPE NOT IN (420,428,566)"
    geom = [
        (-97.06138, 32.837),
        (-97.06133, 32.836),
        (-97.06124, 32.834),
        (-97.06127, 32.832),
    ]
    geo_crs = "epsg:4269"
    distance = 1500
    epa = NHDPlusEPA(layer=layer)
    df = epa.bygeom(geom, geo_crs=geo_crs, sql_clause=sql_clause, distance=distance)
    assert abs(df.LENGTHKM.sum() - 8.917) < 1e-3


@pytest.mark.flaky(max_runs=3)
def test_nldi_navigate():
    stm = nldi.navigate_byid(site, station_id, UM, site)
    st100 = nldi.navigate_byid(site, station_id, UM, site, distance=100)
    pp = nldi.navigate_byid(site, station_id, UT, "huc12pp")
    wqp = nldi.navigate_byloc((-70, 44), UT, "wqp")
    assert (
        st100.shape[0] == 3
        and stm.shape[0] == 3
        and pp.shape[0] == 12
        and wqp.comid.iloc[0] == "6710923"
    )


@pytest.mark.flaky(max_runs=3)
def test_nldi_feature():
    station = nldi.getfeature_byid(site, station_id)
    assert station.comid.values[0] == "1722317"


@pytest.mark.flaky(max_runs=3)
def test_nldi_basin():
    eck4 = "+proj=eck4 +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=km"
    basin = nldi.get_basins(STA_ID).to_crs(eck4)
    assert abs(basin.area.values[0] - 774.170) < 1e-3


@pytest.mark.flaky(max_runs=3)
def test_nldi_char():
    tot, prc = nldi.getcharacteristic_byid("6710923", "div", char_ids="ACC_BFI", values_only=False)
    assert abs(tot.ACC_BFI.values[0] - 57) < 1e-3 and prc.ACC_BFI.values[0] == 0


@pytest.mark.flaky(max_runs=3)
def test_nldi_chardf():
    bfi = nldi.characteristics_dataframe("div", "TOT_BFI", "BFI_CONUS.zip")
    meta = nldi.characteristics_dataframe("div", "TOT_BFI", "BFI_CONUS.zip", metadata=True)
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
