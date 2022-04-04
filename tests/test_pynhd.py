"""Tests for PyNHD package."""
import io
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import MultiPoint, Point, box

import pynhd as nhd
from pynhd import NHD, NLDI, NHDPlusHR, PyGeoAPI, WaterData

try:
    import typeguard  # noqa: F401
except ImportError:
    has_typeguard = False
else:
    has_typeguard = True

STA_ID = "01031500"
station_id = f"USGS-{STA_ID}"
site = "nwissite"
UM = "upstreamMain"
UT = "upstreamTributaries"
SMALL = 1e-3
DEF_CRS = "epsg:4326"


@pytest.fixture
def trib():
    comids = NLDI().navigate_byid(site, station_id, UT, "flowlines")
    wd = WaterData("nhdflowline_network")
    return wd.byid("comid", comids.nhdplus_comid.tolist())


def test_nhd_xs_resample():
    main = NLDI().navigate_byid(site, station_id, UM, "flowlines")
    flw = NHD("flowline_mr").byids("COMID", main.nhdplus_comid.tolist()).to_crs("epsg:3857")
    main_nhd = nhd.prepare_nhdplus(flw, 0, 0, 0, purge_non_dendritic=True)
    cs = nhd.network_xsection(main_nhd, 2000, 1000)
    rs = nhd.network_resample(main_nhd, 2000)
    assert len(cs) == 29 and len(rs) == 46


class TestPyGeoAPI:
    pygeoapi = PyGeoAPI()

    def test_flowtrace(self):
        coords = (1774209.63, 856381.68)
        gs = gpd.GeoDataFrame(
            {
                "direction": [
                    "none",
                ]
            },
            geometry=[Point(coords)],
            crs="ESRI:102003",
        )
        gdf = self.pygeoapi.flow_trace(coords, crs="ESRI:102003", direction="none")
        gdfb = nhd.pygeoapi(gs, "flow_trace")
        assert gdf.comid.iloc[0] == gdfb.comid.iloc[0] == 22294818

    def test_splitcatchment(self):
        coords = (-73.82705, 43.29139)
        gs = gpd.GeoDataFrame(
            {
                "upstream": [
                    False,
                ]
            },
            geometry=[Point(coords)],
            crs=DEF_CRS,
        )
        gdf = self.pygeoapi.split_catchment(coords, crs=DEF_CRS, upstream=False)
        gdfb = nhd.pygeoapi(gs, "split_catchment")
        assert gdf.catchmentID.iloc[0] == gdfb.catchmentID.iloc[0] == "22294818"

    def test_elevation_profile(self):
        coords = [(-103.801086, 40.26772), (-103.80097, 40.270568)]
        gs = gpd.GeoDataFrame(
            {
                "numpts": [
                    101,
                ],
                "dem_res": [
                    1,
                ],
            },
            geometry=[MultiPoint(coords)],
            crs=DEF_CRS,
        )
        gdf = self.pygeoapi.elevation_profile(coords, numpts=101, dem_res=1, crs=DEF_CRS)
        gdfb = nhd.pygeoapi(gs, "elevation_profile")
        assert abs(gdf.iloc[-1, 2] - 316.053) < SMALL and abs(gdfb.iloc[-1, 2] - 316.053) < SMALL

    def test_cross_section(self):
        coords = (-103.80119, 40.2684)
        gs = gpd.GeoDataFrame(
            {
                "width": [
                    1000.0,
                ],
                "numpts": [
                    101,
                ],
            },
            geometry=[Point(coords)],
            crs=DEF_CRS,
        )
        gdf = self.pygeoapi.cross_section(coords, width=1000.0, numpts=101, crs=DEF_CRS)
        gdfb = nhd.pygeoapi(gs, "cross_section")
        assert abs(gdf.iloc[-1, 2] - 767.870) < SMALL and abs(gdfb.iloc[-1, 2] - 767.870) < SMALL


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

    def test_comid_loc(self):
        station = self.nldi.getfeature_byid(site, station_id)
        lon = station.geometry[0].centroid.x
        lat = station.geometry[0].centroid.y
        comid = self.nldi.comid_byloc((lon, lat))
        assert station.comid.values[0] == comid.comid.values[0] == "1722317"

    def test_feature_loc(self):
        station = self.nldi.getfeature_byid(site, station_id)
        lon = round(station.geometry[0].centroid.x, 1)
        lat = round(station.geometry[0].centroid.y, 1)
        comid = self.nldi.feature_byloc((lon, lat))
        assert station.comid.values[0] == "1722317" and comid.comid.values[0] == "1722211"

    def test_basin(self):
        eck4 = "+proj=eck4 +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=km"
        basin = self.nldi.get_basins(STA_ID).to_crs(eck4)
        split = self.nldi.get_basins(STA_ID, split_catchment=True).to_crs(eck4)
        assert abs((split.area.values[0] - basin.area.values[0]) - 1.824) < SMALL

    def test_empty_basin(self):
        empty_ids = ["04253294", "04253296"]
        _, not_found = self.nldi.get_basins(empty_ids)
        assert not_found == empty_ids

    def test_char(self):
        tot, prc = self.nldi.getcharacteristic_byid(
            "6710923", "local", char_ids="all", values_only=False
        )
        assert abs(tot.CAT_BFI.values[0] - 57) < SMALL and prc.CAT_BFI.values[0] == 0

    @pytest.mark.skipif(has_typeguard, reason="Broken if Typeguard is enabled")
    def test_feature_missing(self):
        _, missing = self.nldi.feature_byloc([(45.2, -69.3), (-69.3, 45.2)])
        assert len(missing) == 1

    @pytest.mark.skipif(has_typeguard, reason="Broken if Typeguard is enabled")
    def test_basin_missing(self):
        _, missing = self.nldi.get_basins([STA_ID, "00000000"])
        assert len(missing) == 1


@pytest.mark.xfail(reason="ScienceBase is unstable.")
def test_nhd_attrs():
    meta = nhd.nhdplus_attrs(parquet_path="nhdplus_attrs.parquet")
    _ = nhd.nhdplus_attrs("RECHG")
    cat = nhd.nhdplus_attrs("RECHG")
    Path("nhdplus_attrs.parquet").unlink()
    assert abs(cat[cat.COMID > 0].CAT_RECHG.sum() - 143215331.64) < SMALL and len(meta) == 609


class TestWaterData:
    def test_byid_flw(self, trib):
        assert trib.shape[0] == 432 and abs(trib.lengthkm.sum() - 565.755) < SMALL

    def test_byid(self, trib):
        wd = WaterData("catchmentsp")
        ct = wd.byid("featureid", trib.comid.astype(str).to_list())
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
    hr = NHDPlusHR("flowline")
    flwb = hr.bygeom((-69.77, 45.07, -69.31, 45.45))
    flwi = hr.byids("NHDFlowline.PERMANENT_IDENTIFIER", ["103455178", "103454362", "103453218"])
    flwf = hr.bysql("NHDFlowline.PERMANENT_IDENTIFIER IN ('103455178', '103454362', '103453218')")
    assert (
        flwb.shape[0] == 3892
        and flwi["NHDFlowline.OBJECTID"].tolist() == flwf["NHDFlowline.OBJECTID"].tolist()
    )


@pytest.mark.xfail(reason="Hydroshare is unstable.")
def test_nhdplus_vaa():
    fname = Path("nhdplus_vaa.parquet")
    vaa = nhd.nhdplus_vaa(fname)
    fname.unlink()
    assert abs(vaa.slope.max() - 4.6) < SMALL


def test_use_enhd(trib):
    org_attrs = nhd.prepare_nhdplus(trib, 0, 0, 0, use_enhd_attrs=False)
    enhd_attrs_na = nhd.prepare_nhdplus(trib, 0, 0, 0, use_enhd_attrs=False, terminal2nan=False)
    enhd_attrs = nhd.prepare_nhdplus(trib, 0, 0, 0, use_enhd_attrs=True, terminal2nan=True)
    assert (enhd_attrs.tocomid != org_attrs.tocomid).sum() and (
        enhd_attrs_na.tocomid != enhd_attrs.tocomid
    ).sum()


def test_acc(trib):
    flw = nhd.prepare_nhdplus(trib, 1, 1, 1, purge_non_dendritic=True)

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
