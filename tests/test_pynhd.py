"""Tests for PyNHD package."""
import io
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
SMALL = 1e-3


@pytest.fixture
def comids():
    return NLDI().navigate_byid(site, station_id, UT, "flowlines")


class NHDPlusEPA(AGRBase):
    def __init__(self, layer):
        super().__init__(layer, "*", "epsg:4326")
        self.service = "https://watersgeo.epa.gov/arcgis/rest/services/NHDPlus/NHDPlus/MapServer"


def test_agr():
    geom = [
        (-97.06138, 32.837),
        (-97.06133, 32.836),
        (-97.06124, 32.834),
        (-97.06127, 32.832),
    ]
    epa = NHDPlusEPA(layer="network flowline")
    df = epa.bygeom(
        geom, geo_crs="epsg:4269", sql_clause="FTYPE NOT IN (420,428,566)", distance=1500
    )
    assert abs(df.LENGTHKM.sum() - 8.917) < SMALL


class TestNLDI:
    """Test NLDI service"""

    nldi: NLDI = NLDI()

    def test_navigate(self):
        stm = self.nldi.navigate_byid(site, station_id, UM, site)
        assert stm.shape[0] == 3

    def test_navigate_distance(self):
        st100 = self.nldi.navigate_byid(site, station_id, UM, site, distance=100)
        assert st100.shape[0] == 3

    def test_navigate_fsource(self):
        pp = self.nldi.navigate_byid(site, station_id, UT, "huc12pp")
        assert pp.shape[0] == 12

    def test_navigate_loc(self):
        wqp = self.nldi.navigate_byloc((-70, 44), UT, "wqp")
        assert wqp.comid.iloc[0] == "6710923"

    def test_feature(self):
        station = self.nldi.getfeature_byid(site, station_id)
        lon = round(station.geometry[0].centroid.x, 1)
        lat = round(station.geometry[0].centroid.y, 1)
        _, missing = self.nldi.comid_byloc([(lon, lat), (lat, lon)])
        comid = self.nldi.comid_byloc((lon, lat))
        assert (
            station.comid.values[0] == "1722317"
            and comid.comid.values[0] == "1722211"
            and len(missing) == 1
        )

    def test_basin(self):
        eck4 = "+proj=eck4 +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=km"
        basin = self.nldi.get_basins(STA_ID).to_crs(eck4)
        assert abs(basin.area.values[0] - 774.170) < SMALL

    def test_basin_missing(self):
        _, missing = self.nldi.get_basins([STA_ID, "00000000"])
        assert len(missing) == 1

    def test_char(self):
        tot, prc = self.nldi.getcharacteristic_byid(
            "6710923", "local", char_ids="all", values_only=False
        )
        assert abs(tot.CAT_BFI.values[0] - 57) < SMALL and prc.CAT_BFI.values[0] == 0


def test_nhd_attrs():
    meta = nhd.nhdplus_attrs(save_dir=".")
    cat = nhd.nhdplus_attrs("RECHG", ".")
    Path("nhdplus_attrs.feather").unlink()
    assert abs(cat[cat.COMID > 0].CAT_RECHG.sum() - 143215331.64) < SMALL and len(meta) == 609


class TestWaterData:
    def test_byid_flw(self, comids):
        wd = WaterData("nhdflowline_network")
        trib = wd.byid("comid", comids.nhdplus_comid.tolist())
        assert trib.shape[0] == 432 and abs(trib.lengthkm.sum() - 565.755) < SMALL

    def test_byid(self, comids):
        wd = WaterData("catchmentsp")
        ct = wd.byid("featureid", comids.nhdplus_comid.tolist())
        assert abs(ct.areasqkm.sum() - 773.954) < SMALL

    def test_bybox(self):
        wd = WaterData("nhdwaterbody")
        print(wd)
        wb_g = wd.bygeom(box(-69.7718, 45.0742, -69.3141, 45.4534), predicate="INTERSECTS", xy=True)
        wb_b = wd.bybox((-69.7718, 45.0742, -69.3141, 45.4534))
        assert abs(wb_b.areasqkm.sum() - wb_g.areasqkm.sum()) < SMALL

    def test_byfilter(self):
        crs = "epsg:3857"
        wd = WaterData("huc12", crs)
        wb = wd.byfilter(f"{wd.layer} LIKE '17030001%'")
        huc12 = wb[wb.huc12 == "170300010602"].geometry[0]
        coords = (huc12.centroid.x, huc12.centroid.y)
        hucs = wd.bydistance(coords, 100, crs)
        assert wb.shape[0] == 52 and hucs.name[0] == "Upper Wenas River"


def test_nhdphr():
    hr = nhd.NHDPlusHR("networknhdflowline", service="hydro", auto_switch=True)
    flwb = hr.bygeom((-69.77, 45.07, -69.31, 45.45))
    flwi = hr.byids("PERMANENT_IDENTIFIER", ["103455178", "103454362", "103453218"])
    flwf = hr.bysql("PERMANENT_IDENTIFIER IN ('103455178', '103454362', '103453218')")
    assert flwb.shape[0] == 3887 and flwi["OBJECTID"].tolist() == flwf["OBJECTID"].tolist()


def test_nhdplus_vaa():
    fname = Path("nhdplus_vaa.parquet")
    vaa = nhd.nhdplus_vaa(fname)
    fname.unlink()
    assert abs(vaa.slope.max() - 4.6) < SMALL


def test_acc(comids):
    wd = WaterData("nhdflowline_network")
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

    assert abs(diff.abs().sum() - 439.451) < SMALL


def test_fcode():
    fcode = nhd.nhd_fcode()
    assert fcode.loc[57100, "Feature Type"] == "DAM"


def test_show_versions():
    f = io.StringIO()
    nhd.show_versions(file=f)
    assert "INSTALLED VERSIONS" in f.getvalue()
