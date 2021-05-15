"""Tests for PyNHD package."""
import io
import tempfile
from pathlib import Path

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
    lon = round(station.geometry[0].centroid.x, 1)
    lat = round(station.geometry[0].centroid.y, 1)
    comid, missing = nldi.comid_byloc([(lon, lat), (lat, lon)])
    assert (
        station.comid.values[0] == "1722317"
        and comid.comid.values[0] == "1722211"
        and len(missing) == 1
    )


@pytest.mark.flaky(max_runs=3)
def test_nldi_basin():
    eck4 = "+proj=eck4 +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=km"
    basin, missing = nldi.get_basins([STA_ID, "00000000"])
    basin = basin.to_crs(eck4)
    assert abs(basin.area.values[0] - 774.170) < 1e-3 and len(missing) == 1


@pytest.mark.flaky(max_runs=3)
def test_nldi_char():
    tot, prc = nldi.getcharacteristic_byid(
        "6710923", "local", char_ids="CAT_BFI", values_only=False
    )
    assert abs(tot.CAT_BFI.values[0] - 57) < 1e-3 and prc.CAT_BFI.values[0] == 0


@pytest.mark.flaky(max_runs=3)
def test_nhd_attrs():
    meta = nhd.nhdplus_attrs(save_dir=".")
    cat = nhd.nhdplus_attrs("RECHG", ".")
    Path("nhdplus_attrs.feather").unlink()
    assert abs(cat[cat.COMID > 0].CAT_RECHG.sum() - 143215331.64) < 1e-3 and len(meta) == 609


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
    crs = "epsg:3857"
    wd = WaterData("huc12", crs)
    wb = wd.byfilter(f"{wd.layer} LIKE '17030001%'")
    coords = (wb.iloc[0].geometry.centroid.x, wb.iloc[0].geometry.centroid.y)
    hucs = wd.bydistance(coords, 100, crs)
    assert wb.shape[0] == 52 and hucs.huc12.values[0] == "170300010602"


@pytest.mark.flaky(max_runs=3)
def test_nhdphr():
    hr = nhd.NHDPlusHR("networknhdflowline", service="edits", auto_switch=True)
    flwb = hr.bygeom((-69.77, 45.07, -69.31, 45.45))
    flwi = hr.byids("NHDFlowline.PERMANENT_IDENTIFIER", ["103455178", "103454362", "103453218"])
    flwf = hr.bysql("NHDFlowline.PERMANENT_IDENTIFIER IN ('103455178', '103454362', '103453218')")
    assert (
        flwb.shape[0] == 3892
        and flwi["NHDFlowline.OBJECTID"].tolist() == flwf["NHDFlowline.OBJECTID"].tolist()
    )


def test_nhdplus_vaa():
    with tempfile.NamedTemporaryFile(suffix=".parquet") as f:
        vaa = nhd.nhdplus_vaa(f.name)

    assert abs(vaa.slope.max() - 4.6) < 1e-3


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
