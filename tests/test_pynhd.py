"""Tests for PyNHD package."""

from __future__ import annotations

import io
import operator
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely import LineString, MultiPoint, Point, box

import pynhd
from pynhd import HP3D, NHD, NLDI, NHDPlusHR, PyGeoAPI, WaterData

is_ci = os.environ.get("GH_CI") == "true"
STA_ID = "01031500"
station_id = f"USGS-{STA_ID}"
site = "nwissite"
UM = "upstreamMain"
UT = "upstreamTributaries"


def assert_close(a: float, b: float) -> None:
    np.testing.assert_allclose(a, b, rtol=1e-3)


@pytest.fixture()
def trib():
    comids = NLDI().navigate_byid(site, station_id, UT, "flowlines")
    return WaterData("nhdflowline_network").byid("comid", comids.nhdplus_comid.tolist())


def test_streamcat():
    nhd_area = pynhd.streamcat("fert", comids=13212248)
    assert_close(nhd_area["FERTWS"].item(), 14.358)
    nhd_area = pynhd.streamcat("inorgnwetdep_2008", comids=23783629, lakes_only=True)
    assert_close(nhd_area["INORGNWETDEP_2008WS"].item(), 1.7746)


def test_epa():
    data = pynhd.epa_nhd_catchments(9533477, "curve_number")
    assert_close(data["curve_number"].mean(axis=1), 75.576)

    data = pynhd.epa_nhd_catchments([9533477, 1440291], "comid_info")
    assert data["comid_info"].loc[1440291, "TOCOMID"] == 1439303


@pytest.mark.xfail(reason="Hydro is unstable.")
def test_nhd_xs_resample():
    main = NLDI().navigate_byid(site, station_id, UM, "flowlines")
    flw = NHD("flowline_mr").byids("COMID", main.nhdplus_comid.tolist()).to_crs(3857)
    main_nhd = pynhd.prepare_nhdplus(flw, 0, 0, 0, purge_non_dendritic=True)
    cs = pynhd.network_xsection(main_nhd, 2000, 1000)
    rs = pynhd.network_resample(main_nhd, 2000)
    assert len(cs) == 29
    assert len(rs) == 46


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
        gdfb = pynhd.pygeoapi(gs, "flow_trace")
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
            crs=4326,
        )
        gdf = self.pygeoapi.split_catchment(coords, crs=4326, upstream=False)
        gdfb = pynhd.pygeoapi(gs, "split_catchment")
        assert gdf.catchmentID.iloc[0] == gdfb.catchmentID.iloc[0] == "22294818"

    @pytest.mark.xfail(reason="The xs endpoints of PyGeoAPI are not working.")
    def test_elevation_profile(self):
        line = LineString([(-103.801086, 40.26772), (-103.80097, 40.270568)])
        gs = gpd.GeoDataFrame(
            {
                "numpts": [
                    101,
                ],
                "dem_res": [
                    1,
                ],
            },
            geometry=[line],
            crs=4326,
        )
        gdf = self.pygeoapi.elevation_profile(line, numpts=101, dem_res=1, crs=4326)
        gdfb = pynhd.pygeoapi(gs, "elevation_profile")

        expected = 1299.8728
        assert_close(gdf.iloc[-1, 2], expected)
        assert_close(gdfb.iloc[-1, 2], expected)

    @pytest.mark.xfail(reason="The xs endpoints of PyGeoAPI are not working.")
    def test_endpoints_profile(self):
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
            crs=4326,
        )
        gdf = self.pygeoapi.endpoints_profile(coords, numpts=101, dem_res=1, crs=4326)
        gdfb = pynhd.pygeoapi(gs, "endpoints_profile")

        expected = 1299.8842
        assert_close(gdf.iloc[-1, 2], expected)
        assert_close(gdfb.iloc[-1, 2], expected)

    @pytest.mark.xfail(reason="The xs endpoints of PyGeoAPI are not working.")
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
            crs=4326,
        )
        gdf = self.pygeoapi.cross_section(coords, width=1000.0, numpts=101, crs=4326)
        gdfb = pynhd.pygeoapi(gs, "cross_section")

        expected = 1301.482
        assert_close(gdf.iloc[-1, 2], expected)
        assert_close(gdfb.iloc[-1, 2], expected)


class TestNLDI:
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
        lon = station.geometry.iloc[0].centroid.x
        lat = station.geometry.iloc[0].centroid.y
        comid = self.nldi.comid_byloc((lon, lat))
        assert station.comid.values[0] == comid.comid.values[0] == "1722317"

    def test_feature_loc(self):
        station = self.nldi.getfeature_byid(site, station_id)
        lon = round(station.geometry.iloc[0].centroid.x, 1)
        lat = round(station.geometry.iloc[0].centroid.y, 1)
        comid = self.nldi.feature_byloc((lon, lat))
        assert station.comid.values[0] == "1722317"
        assert comid.comid.values[0] == "1722211"

    def test_basin(self):
        eck4 = "+proj=eck4 +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=km"
        basin = self.nldi.get_basins(STA_ID).to_crs(eck4)
        split = self.nldi.get_basins(STA_ID, split_catchment=True).to_crs(eck4)
        assert_close(split.area.values[0] - basin.area.values[0], 0.489)

    def test_char(self):
        tot, prc = self.nldi.getcharacteristic_byid(
            "6710923", "local", char_ids="all", values_only=False
        )
        assert tot.CAT_BFI.values[0] == 57
        assert prc.CAT_BFI.values[0] == 0


class TestNHDAttrs:
    def test_meta_s3(self):
        meta = pynhd.nhdplus_attrs_s3()
        assert len(meta) == 14139

    def test_s3(self):
        attr = pynhd.nhdplus_attrs_s3("CAT_RECHG")
        assert_close(attr[attr.CAT_RECHG > 0].CAT_RECHG.mean(), 132.5881)

    def test_meta(self):
        meta = pynhd.nhdplus_attrs()
        assert len(meta) == 595

    def test_sb(self):
        attr = pynhd.nhdplus_attrs("BANKFULL")
        assert_close(attr[attr.BANKFULL_WIDTH > 0].BANKFULL_WIDTH.mean(), 13.6633)


class TestWaterData:
    def test_byid_flw(self, trib):
        assert_close(trib.lengthkm.sum(), 565.755)
        assert trib.shape[0] == 432

    def test_byid(self, trib):
        wd = WaterData("catchmentsp")
        ct = wd.byid("featureid", trib.comid.astype(str).to_list())
        assert_close(ct.areasqkm.sum(), 773.954)

    def test_bybox(self):
        wd = WaterData("wbd12")
        assert "wbd12" in wd.__repr__()
        wb_g = wd.bygeom(box(-118.72, 34.118, -118.31, 34.518), predicate="intersects", xy=True)
        wb_b = wd.bybox((-118.72, 34.118, -118.31, 34.518))
        assert_close(wb_b.areasqkm.sum(), wb_g.areasqkm.sum())

    def test_byfilter(self):
        crs = 3857
        wd = WaterData("huc12", crs)
        wb = wd.byfilter("huc12 LIKE '17030001%'", sort_attr="huc12")
        huc12 = wb[wb.huc12 == "170300010602"].geometry
        coords = (huc12.centroid.x, huc12.centroid.y)
        hucs = wd.bydistance(coords, 100, crs, sort_attr="huc12")
        assert wb.shape[0] == 52
        assert hucs.name[0] == "Upper Wenas River"


class TestGCX:
    def test_single(self):
        gcx = pynhd.GeoConnex("gages")
        gauge = gcx.byid("provider_id", "01031500")
        assert (gauge["nhdpv2_comid"] == 1722317).sum() == 1

    def test_multiple(self):
        gcx = pynhd.GeoConnex()
        gcx.item = "hu02"
        h2 = gcx.byid("huc2", "02")
        h3 = gcx.byid("huc2", "03")
        assert (h2["gnis_id"] == 2730132).sum() == (h3["gnis_id"] == 2730133).sum() == 1

    def test_many_features(self):
        gcx = pynhd.GeoConnex(max_nfeatures=10)
        gcx.item = "mainstems"
        ms = gcx.bygeometry((-69.77, 45.07, -69.31, 45.45))
        assert len(ms) == 20

    def test_cql(self):
        gcx = pynhd.GeoConnex("ua10")
        awa = gcx.bycql({"gt": [{"property": "awater10"}, 100e6]})
        assert len(awa) == 14


def test_3dhp():
    nhd3d = HP3D("flowline")
    flw = nhd3d.bygeom(Point(-89.441, 43.487), distance=10)
    assert flw.shape[0] == 1


def test_nhdphr():
    hr = NHDPlusHR("flowline")
    flwb = hr.bygeom((-69.77, 45.07, -69.31, 45.45))
    ids = ["103453218", "103454362", "103455178"]
    flwi = hr.byids("permanent_identifier", ids)
    ids_str = ", ".join([f"'{i}'" for i in ids])
    flwf = hr.bysql(f"permanent_identifier IN ({ids_str})")
    assert flwb.shape[0] == 3887
    assert sorted(flwi["permanent_identifier"]) == sorted(flwf["permanent_identifier"]) == ids


@pytest.mark.xfail(reason="Hydroshare is unstable.")
def test_nhdplus_vaa():
    fname = Path("nhdplus_vaa.parquet")
    vaa = pynhd.nhdplus_vaa(fname)
    fname.unlink()
    assert_close(vaa.slope.max(), 4.6)


class TestENHD:
    def test_wo_enhd_w_nan(self, trib: gpd.GeoDataFrame):
        attrs = pynhd.prepare_nhdplus(trib, 0, 0, 0, use_enhd_attrs=False, terminal2nan=True)
        assert attrs.tocomid.isna().sum() == 1

    def test_wo_enhd_wo_nan(self, trib: gpd.GeoDataFrame):
        attrs = pynhd.prepare_nhdplus(trib, 0, 0, 0, use_enhd_attrs=False, terminal2nan=False)
        assert attrs.tocomid.isna().sum() == 0

    def test_w_enhd_w_nan(self, trib: gpd.GeoDataFrame):
        attrs = pynhd.prepare_nhdplus(trib, 0, 0, 0, use_enhd_attrs=True, terminal2nan=True)
        assert attrs.tocomid.isna().sum() == 1


def test_toposort(trib):
    flw = pynhd.prepare_nhdplus(trib, 0, 0, 0, purge_non_dendritic=True)
    _, up_nodes, _ = pynhd.topoogical_sort(flw, id_col="comid", toid_col="tocomid")
    assert up_nodes[1721025] == [1720901]


def test_acc(trib):
    flw = pynhd.prepare_nhdplus(trib, 1, 1, 1, purge_non_dendritic=True)

    qsim = pynhd.vector_accumulation(
        flw[["comid", "tocomid", "lengthkm"]],
        operator.add,
        "lengthkm",
        ["lengthkm"],
    )
    flw = flw.merge(qsim, on="comid")
    diff = flw.arbolatesu - flw.acc_lengthkm
    assert_close(diff.abs().sum(), 439.451)


def test_h12pp():
    h12pp = pynhd.nhdplus_h12pp()
    assert len(h12pp) == 78249


def test_enhd_nx():
    g, m, s = pynhd.enhd_flowlines_nx()
    assert g.number_of_nodes() == len(m)
    assert s[0] == 8318775


def test_huc12_nx():
    g, m, s = pynhd.mainstem_huc12_nx()
    assert g.number_of_nodes() == len(m)
    assert s[0] == "150301040501"


def test_fcode():
    fcode = pynhd.nhd_fcode()
    assert fcode.loc[57100, "Feature Type"] == "DAM"


def test_show_versions():
    f = io.StringIO()
    pynhd.show_versions(file=f)
    assert "SYS INFO" in f.getvalue()
