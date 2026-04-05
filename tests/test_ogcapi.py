"""Tests for OGC API module with mocked and real network calls."""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import patch

import geopandas as gpd
import pandas as pd
import pytest
from shapely import Point

import async_retriever as ar
import pynhd
from pynhd.exceptions import (
    InputRangeError,
    InputValueError,
    MissingItemError,
    ServiceError,
    ZeroMatchedError,
)
from pynhd.ogcapi import OGCAPIBase

network = pytest.mark.network

# --- Shared mock data helpers ---


def _make_collections_response(
    collection_id: str,
    description: str,
    bbox: list[float],
    queryables_url: str,
) -> list[dict[str, Any]]:
    """Build a mock /collections JSON response."""
    return [
        {
            "collections": [
                {
                    "id": collection_id,
                    "description": description,
                    "links": [
                        {
                            "href": queryables_url,
                            "rel": "http://www.opengis.net/def/rel/ogc/1.0/queryables",
                            "type": "application/schema+json",
                        },
                    ],
                    "extent": {"spatial": {"bbox": [bbox]}},
                },
            ],
        },
    ]


def _make_queryables_response(
    properties: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    """Build a mock /queryables JSON response."""
    return [{"properties": properties}]


def _make_schema_response(
    id_field: str,
    extra_fields: dict[str, dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Build a mock /schema JSON response with an x-ogc-role=id field."""
    properties: dict[str, Any] = {
        id_field: {"type": "string", "x-ogc-role": "id"},
    }
    if extra_fields:
        properties.update(extra_fields)
    return [{"properties": properties}]


def _make_feature(fid: int | str, props: dict[str, Any], geometry: dict | None = None) -> dict:
    """Build a single GeoJSON feature."""
    geom = geometry or {
        "type": "Point",
        "coordinates": [-69.5, 45.2],
    }
    return {
        "type": "Feature",
        "id": fid,
        "geometry": geom,
        "properties": props,
    }


def _make_items_response(
    features: list[dict],
    number_matched: int | None = None,
) -> list[dict[str, Any]]:
    """Build a mock /items JSON response."""
    if number_matched is None:
        number_matched = len(features)
    return [
        {
            "type": "FeatureCollection",
            "numberMatched": number_matched,
            "features": features,
        },
    ]


# ---- Fixture: reusable mock wiring ----------------------------------------

MOCK_BBOX = [-170.0, 15.0, -51.0, 72.0]
MOCK_QUERYABLES_URL = "https://example.com/collections/test_item/queryables"
MOCK_ITEMS_URL = "https://example.com/collections/test_item/items"

MOCK_PROPERTIES = {
    "name": {"title": "name", "type": "string"},
    "value": {"title": "value", "type": "number"},
    "fid": {"title": "fid", "type": "integer"},
}


def _collections_resp() -> list[dict[str, Any]]:
    return _make_collections_response(
        "test_item", "A test collection", MOCK_BBOX, MOCK_QUERYABLES_URL
    )


def _queryables_resp() -> list[dict[str, Any]]:
    return _make_queryables_response(MOCK_PROPERTIES)


def _schema_resp() -> list[dict[str, Any]]:
    return _make_schema_response("fid")


def _single_feature() -> dict:
    return _make_feature(1, {"name": "gauge1", "value": 42.0, "fid": 1})


def _multi_features(n: int = 3) -> list[dict]:
    return [_make_feature(i, {"name": f"feat{i}", "value": float(i), "fid": i}) for i in range(n)]


def _side_effect_for_init(*url_lists: list[str], **_: Any) -> list[dict[str, Any]]:
    """Return the right mock for the URL being fetched during __init__."""
    url = url_lists[0][0]
    if "queryables" in url:
        return _queryables_resp()
    if "collections" in url:
        return _collections_resp()
    if "schema" in url:
        return _schema_resp()
    msg = f"Unexpected URL: {url}"
    raise ValueError(msg)


@pytest.fixture
def base_instance():
    """Create an OGCAPIBase instance with mocked network calls."""
    with patch("pynhd.ogcapi.ar.retrieve_json", side_effect=_side_effect_for_init):
        instance = OGCAPIBase(
            prod_url="https://example.com",
            dev_url="https://dev.example.com",
            item="test_item",
        )
    return instance


# ---- Tests: OGCAPIBase init & properties ----------------------------------


class TestOGCAPIBaseInit:
    def test_endpoints_populated(self, base_instance: OGCAPIBase):
        assert "test_item" in base_instance.endpoints
        ep = base_instance.endpoints["test_item"]
        assert ep.name == "test_item"
        assert ep.description == "A test collection"
        assert ep.extent == tuple(MOCK_BBOX)
        assert set(ep.query_fields) == {"name", "value", "fid"}

    def test_dtypes_mapped(self, base_instance: OGCAPIBase):
        dtypes = base_instance.endpoints["test_item"].dtypes
        assert dtypes["name"] == "str"
        assert dtypes["value"] == "f8"
        assert dtypes["fid"] == "int64"

    def test_item_setter_invalid(self, base_instance: OGCAPIBase):
        with (
            patch("pynhd.ogcapi.ar.retrieve_json", side_effect=_side_effect_for_init),
            pytest.raises(InputValueError),
        ):
            base_instance.item = "nonexistent"

    def test_max_nfeatures_limit(self):
        with (
            pytest.raises(InputRangeError),
            patch("pynhd.ogcapi.ar.retrieve_json", side_effect=_side_effect_for_init),
        ):
            OGCAPIBase(
                prod_url="https://example.com",
                dev_url="https://dev.example.com",
                max_nfeatures=20000,
                max_nfeatures_limit=10000,
            )

    def test_repr_no_item(self):
        with patch("pynhd.ogcapi.ar.retrieve_json", side_effect=_side_effect_for_init):
            instance = OGCAPIBase(
                prod_url="https://example.com",
                dev_url="https://dev.example.com",
            )
        result = repr(instance)
        assert "Available Endpoints" in result
        assert "test_item" in result

    def test_repr_with_item(self, base_instance: OGCAPIBase):
        result = repr(base_instance)
        assert "Item: 'test_item'" in result
        assert "A test collection" in result

    def test_dev_url(self):
        with patch("pynhd.ogcapi.ar.retrieve_json", side_effect=_side_effect_for_init):
            instance = OGCAPIBase(
                prod_url="https://example.com",
                dev_url="https://dev.example.com",
                dev=True,
            )
        assert "dev.example.com" in instance.base_url

    def test_api_key_headers(self):
        with patch("pynhd.ogcapi.ar.retrieve_json", side_effect=_side_effect_for_init):
            instance = OGCAPIBase(
                prod_url="https://example.com",
                dev_url="https://dev.example.com",
                api_key="test-key-123",
            )
        assert instance._api_headers["X-Api-Key"] == "test-key-123"
        assert "User-Agent" in instance._api_headers
        assert "pynhd" in instance._api_headers["User-Agent"]

    def test_no_api_key_empty_headers(self, base_instance: OGCAPIBase):
        assert "X-Api-Key" not in base_instance._api_headers
        assert "pynhd" in base_instance._api_headers["User-Agent"]


# ---- Tests: query methods --------------------------------------------------


class TestByID:
    def test_byid_single(self, base_instance: OGCAPIBase):
        feature = _single_feature()
        with patch("pynhd.ogcapi.ar.retrieve_json") as mock_retrieve:
            mock_retrieve.side_effect = [_schema_resp(), _make_items_response([feature])]
            result = base_instance.byid("name", "gauge1")
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 1
        assert result.iloc[0]["name"] == "gauge1"

    def test_byid_multiple(self, base_instance: OGCAPIBase):
        features = _multi_features(2)
        with patch("pynhd.ogcapi.ar.retrieve_json") as mock_retrieve:
            mock_retrieve.side_effect = [_schema_resp(), _make_items_response(features)]
            result = base_instance.byid("fid", [0, 1])
        assert len(result) == 2

    def test_byid_invalid_feature_name(self, base_instance: OGCAPIBase):
        with pytest.raises(InputValueError, match="feature_name"):
            base_instance.byid("nonexistent_field", "some_id")

    def test_byid_no_item_set(self):
        with patch("pynhd.ogcapi.ar.retrieve_json", side_effect=_side_effect_for_init):
            instance = OGCAPIBase(
                prod_url="https://example.com",
                dev_url="https://dev.example.com",
            )
        with pytest.raises(MissingItemError):
            instance.byid("name", "some_id")

    def test_byid_skip_geometry(self, base_instance: OGCAPIBase):
        feature = _make_feature(1, {"name": "g1", "value": 1.0, "fid": 1})
        feature["geometry"] = None
        resp = _make_items_response([feature])
        with patch("pynhd.ogcapi.ar.retrieve_json") as mock_retrieve:
            mock_retrieve.side_effect = [_schema_resp(), resp]
            result = base_instance.byid("name", "g1", skip_geometry=True)
        assert isinstance(result, pd.DataFrame)


class TestByItem:
    def test_byitem(self, base_instance: OGCAPIBase):
        feature = _single_feature()
        with patch("pynhd.ogcapi.ar.retrieve_json") as mock_retrieve:
            mock_retrieve.return_value = [feature]
            result = base_instance.byitem("1")
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 1

    def test_byitem_error_response(self, base_instance: OGCAPIBase):
        with patch("pynhd.ogcapi.ar.retrieve_json") as mock_retrieve:
            mock_retrieve.return_value = [{"code": "NotFound", "description": "Item not found"}]
            with pytest.raises(ServiceError, match="NotFound"):
                base_instance.byitem("9999")

    def test_byitem_no_item_set(self):
        with patch("pynhd.ogcapi.ar.retrieve_json", side_effect=_side_effect_for_init):
            instance = OGCAPIBase(
                prod_url="https://example.com",
                dev_url="https://dev.example.com",
            )
        with pytest.raises(MissingItemError):
            instance.byitem("1")


class TestByBox:
    def test_bybox(self, base_instance: OGCAPIBase):
        features = _multi_features(2)
        with patch("pynhd.ogcapi.ar.retrieve_json") as mock_retrieve:
            mock_retrieve.side_effect = [_schema_resp(), _make_items_response(features)]
            result = base_instance.bybox((-69.77, 45.07, -69.31, 45.45))
        assert len(result) == 2

    def test_bybox_no_item_set(self):
        with patch("pynhd.ogcapi.ar.retrieve_json", side_effect=_side_effect_for_init):
            instance = OGCAPIBase(
                prod_url="https://example.com",
                dev_url="https://dev.example.com",
            )
        with pytest.raises(MissingItemError):
            instance.bybox((-69.77, 45.07, -69.31, 45.45))


class TestByGeometry:
    def test_bygeometry_bbox_tuple(self, base_instance: OGCAPIBase):
        features = _multi_features(3)
        with patch("pynhd.ogcapi.ar.retrieve_json") as mock_retrieve:
            mock_retrieve.side_effect = [_schema_resp(), _make_items_response(features)]
            result = base_instance.bygeometry((-69.77, 45.07, -69.31, 45.45))
        assert len(result) == 3

    def test_bygeometry_two_geoms(self, base_instance: OGCAPIBase):
        # Use a LineString that crosses geometry2 boundary (starts outside, passes through, ends outside)
        line_geom = {"type": "LineString", "coordinates": [[-69.80, 45.20], [-69.30, 45.20]]}
        features = [_make_feature(0, {"name": "feat0", "value": 0.0, "fid": 0}, geometry=line_geom)]
        with patch("pynhd.ogcapi.ar.retrieve_json") as mock_retrieve:
            mock_retrieve.side_effect = [_schema_resp(), _make_items_response(features)]
            result = base_instance.bygeometry(
                (-69.77, 45.07, -69.31, 45.45),
                (-69.70, 45.10, -69.40, 45.40),
                predicate="crosses",
            )
        assert len(result) == 1

    def test_bygeometry_invalid_predicate(self, base_instance: OGCAPIBase):
        with pytest.raises(InputValueError, match="predicate"):
            base_instance.bygeometry((-69.77, 45.07, -69.31, 45.45), predicate="invalid")

    def test_bygeometry_out_of_bounds(self, base_instance: OGCAPIBase):
        with pytest.raises(InputRangeError):
            base_instance.bygeometry((10.0, 80.0, 11.0, 81.0))

    def test_bygeometry_no_item_set(self):
        with patch("pynhd.ogcapi.ar.retrieve_json", side_effect=_side_effect_for_init):
            instance = OGCAPIBase(
                prod_url="https://example.com",
                dev_url="https://dev.example.com",
            )
        with pytest.raises(MissingItemError):
            instance.bygeometry((-69.77, 45.07, -69.31, 45.45))

    def test_bygeometry_point(self, base_instance: OGCAPIBase):
        features = _multi_features(1)
        with patch("pynhd.ogcapi.ar.retrieve_json") as mock_retrieve:
            mock_retrieve.side_effect = [_schema_resp(), _make_items_response(features)]
            result = base_instance.bygeometry(Point(-69.5, 45.2))
        assert len(result) == 1


class TestByCQL:
    def test_bycql(self, base_instance: OGCAPIBase):
        features = _multi_features(2)
        with patch("pynhd.ogcapi.ar.retrieve_json") as mock_retrieve:
            mock_retrieve.side_effect = [_schema_resp(), _make_items_response(features)]
            result = base_instance.bycql({"op": "gt", "args": [{"property": "value"}, 0]})
        assert len(result) == 2

    def test_bycql_no_item(self):
        with patch("pynhd.ogcapi.ar.retrieve_json", side_effect=_side_effect_for_init):
            instance = OGCAPIBase(
                prod_url="https://example.com",
                dev_url="https://dev.example.com",
            )
        with pytest.raises(MissingItemError):
            instance.bycql({"op": "gt", "args": [{"property": "value"}, 0]})


class TestByFilter:
    def test_byfilter(self, base_instance: OGCAPIBase):
        features = _multi_features(2)
        with patch("pynhd.ogcapi.ar.retrieve_json") as mock_retrieve:
            mock_retrieve.side_effect = [_schema_resp(), _make_items_response(features)]
            result = base_instance.byfilter("name = 'feat1'")
        assert len(result) == 2

    def test_byfilter_no_item(self):
        with patch("pynhd.ogcapi.ar.retrieve_json", side_effect=_side_effect_for_init):
            instance = OGCAPIBase(
                prod_url="https://example.com",
                dev_url="https://dev.example.com",
            )
        with pytest.raises(MissingItemError):
            instance.byfilter("name = 'feat1'")


# ---- Tests: pagination -----------------------------------------------------


class TestPagination:
    def test_pagination_multiple_pages(self, base_instance: OGCAPIBase):
        """When numberMatched > len(features), pagination fetches more."""
        page1_features = _multi_features(2)
        all_features = _multi_features(4)
        with patch("pynhd.ogcapi.ar.retrieve_json") as mock_retrieve:
            mock_retrieve.side_effect = [
                _schema_resp(),
                _make_items_response(page1_features, number_matched=4),
                _make_items_response(all_features, number_matched=4),
            ]
            result = base_instance.bybox((-69.77, 45.07, -69.31, 45.45))
        assert len(result) == 4


# ---- Tests: error responses ------------------------------------------------


class TestErrorResponses:
    def test_service_error_code(self, base_instance: OGCAPIBase):
        with patch("pynhd.ogcapi.ar.retrieve_json") as mock_retrieve:
            mock_retrieve.side_effect = [
                _schema_resp(),
                [{"code": "InvalidParameterValue", "description": "bad param"}],
            ]
            with pytest.raises(ServiceError, match="InvalidParameterValue"):
                base_instance.bybox((-69.77, 45.07, -69.31, 45.45))

    def test_zero_matched(self, base_instance: OGCAPIBase):
        with patch("pynhd.ogcapi.ar.retrieve_json") as mock_retrieve:
            mock_retrieve.side_effect = [
                _schema_resp(),
                [{"type": "FeatureCollection", "numberMatched": 0, "features": []}],
            ]
            with pytest.raises(ZeroMatchedError):
                base_instance.bybox((-69.77, 45.07, -69.31, 45.45))

    def test_sort_attr_fallback(self):
        """When schema has no x-ogc-role=id field, fall back to first queryable field."""
        schema_no_id: list[dict[str, Any]] = [
            {"properties": {"some_field": {"type": "string"}}},
        ]

        def side_effect(*url_lists: list[str], **_: Any) -> list[dict[str, Any]]:
            url = url_lists[0][0]
            if "queryables" in url:
                return _queryables_resp()
            if "collections" in url:
                return _collections_resp()
            if "schema" in url:
                return schema_no_id
            msg = f"Unexpected URL: {url}"
            raise ValueError(msg)

        with patch("pynhd.ogcapi.ar.retrieve_json", side_effect=side_effect):
            instance = OGCAPIBase(
                prod_url="https://example.com",
                dev_url="https://dev.example.com",
                item="test_item",
            )
        with patch("pynhd.ogcapi.ar.retrieve_json", return_value=schema_no_id):
            attr = instance._get_sort_attr("test_item")
        # Falls back to first queryable field
        assert attr in instance.endpoints["test_item"].query_fields

    def test_error_response_evicts_cache_query(self, base_instance: OGCAPIBase):
        """Error responses (e.g., rate limit) must be deleted from cache."""
        error_resp = [{"error": {"code": "OVER_RATE_LIMIT", "message": "rate limited"}}]
        with (
            patch("pynhd.ogcapi.ar.retrieve_json") as mock_retrieve,
            patch("pynhd.ogcapi.ar.delete_url_cache") as mock_delete,
        ):
            mock_retrieve.side_effect = [_schema_resp(), error_resp]
            with pytest.raises(ServiceError, match="OVER_RATE_LIMIT"):
                base_instance.bybox((-69.77, 45.07, -69.31, 45.45))
            mock_delete.assert_called_once()

    def test_error_response_evicts_cache_byitem(self, base_instance: OGCAPIBase):
        """Error responses from byitem must be deleted from cache."""
        error_resp = [{"error": {"code": "OVER_RATE_LIMIT", "message": "rate limited"}}]
        with (
            patch("pynhd.ogcapi.ar.retrieve_json", return_value=error_resp),
            patch("pynhd.ogcapi.ar.delete_url_cache") as mock_delete,
        ):
            with pytest.raises(ServiceError, match="OVER_RATE_LIMIT"):
                base_instance.byitem("123")
            mock_delete.assert_called_once()

    def test_legacy_error_evicts_cache(self, base_instance: OGCAPIBase):
        """Legacy error format (code/description at top level) also evicts cache."""
        error_resp = [{"code": "ServerError", "description": "something broke"}]
        with (
            patch("pynhd.ogcapi.ar.retrieve_json") as mock_retrieve,
            patch("pynhd.ogcapi.ar.delete_url_cache") as mock_delete,
        ):
            mock_retrieve.side_effect = [_schema_resp(), error_resp]
            with pytest.raises(ServiceError, match="ServerError"):
                base_instance.bybox((-69.77, 45.07, -69.31, 45.45))
            mock_delete.assert_called_once()

    def test_sort_attr_missing_properties(self):
        """When schema has no properties key, fall back to first queryable field."""
        schema_empty: list[dict[str, Any]] = [{"type": "object"}]

        def side_effect(*url_lists: list[str], **_: Any) -> list[dict[str, Any]]:
            url = url_lists[0][0]
            if "queryables" in url:
                return _queryables_resp()
            if "collections" in url:
                return _collections_resp()
            if "schema" in url:
                return schema_empty
            msg = f"Unexpected URL: {url}"
            raise ValueError(msg)

        with patch("pynhd.ogcapi.ar.retrieve_json", side_effect=side_effect):
            instance = OGCAPIBase(
                prod_url="https://example.com",
                dev_url="https://dev.example.com",
                item="test_item",
            )
        with patch("pynhd.ogcapi.ar.retrieve_json", return_value=schema_empty):
            attr = instance._get_sort_attr("test_item")
        assert attr in instance.endpoints["test_item"].query_fields


# ---- Tests: GeoConnex subclass ---------------------------------------------

GCX_BBOX = [-170.0, 15.0, -51.0, 72.0]
GCX_QUERYABLES_URL = "https://reference.geoconnex.us/collections/gages/queryables"


def _gcx_side_effect(*url_lists: list[str], **_: Any) -> list[dict[str, Any]]:
    url = url_lists[0][0]
    if "queryables" in url:
        return _make_queryables_response(
            {
                "uri": {"title": "uri", "type": "string"},
                "name": {"title": "name", "type": "string"},
                "provider_id": {"title": "provider_id", "type": "string"},
                "nhdpv2_comid": {"title": "nhdpv2_comid", "type": "integer"},
            },
        )
    if "collections" in url:
        return _make_collections_response(
            "gages",
            "US Reference Stream Gauge Monitoring Locations",
            GCX_BBOX,
            GCX_QUERYABLES_URL,
        )
    msg = f"Unexpected URL: {url}"
    raise ValueError(msg)


class TestGeoConnex:
    def test_init(self):
        with patch("pynhd.ogcapi.ar.retrieve_json", side_effect=_gcx_side_effect):
            gcx = pynhd.GeoConnex("gages")
        assert "gages" in gcx.endpoints
        assert gcx.item == "gages"

    def test_sort_attr_returns_uri(self):
        """GeoConnex overrides _get_sort_attr to always return 'uri'."""
        with patch("pynhd.ogcapi.ar.retrieve_json", side_effect=_gcx_side_effect):
            gcx = pynhd.GeoConnex("gages")
        assert gcx._get_sort_attr("gages") == "uri"

    def test_invalid_item(self):
        with (
            patch("pynhd.ogcapi.ar.retrieve_json", side_effect=_gcx_side_effect),
            pytest.raises(InputValueError),
        ):
            pynhd.GeoConnex("wrong")

    def test_wrong_bounds(self):
        with patch("pynhd.ogcapi.ar.retrieve_json", side_effect=_gcx_side_effect):
            gcx = pynhd.GeoConnex("gages")
        with pytest.raises(InputRangeError):
            gcx.bygeometry((100.0, 80.0, 101.0, 81.0))

    def test_byid(self):
        with patch("pynhd.ogcapi.ar.retrieve_json", side_effect=_gcx_side_effect):
            gcx = pynhd.GeoConnex("gages")
        feature = _make_feature(
            1,
            {
                "uri": "https://geoconnex.us/ref/gages/01031500",
                "name": "gauge1",
                "provider_id": "01031500",
                "nhdpv2_comid": 1722317,
            },
        )
        with patch("pynhd.ogcapi.ar.retrieve_json") as mock_retrieve:
            mock_retrieve.return_value = _make_items_response([feature])
            result = gcx.byid("provider_id", "01031500")
        assert len(result) == 1
        assert result.iloc[0]["nhdpv2_comid"] == 1722317

    def test_bybox(self):
        with patch("pynhd.ogcapi.ar.retrieve_json", side_effect=_gcx_side_effect):
            gcx = pynhd.GeoConnex("gages")
        features = [
            _make_feature(
                i, {"uri": f"uri{i}", "name": f"g{i}", "provider_id": f"id{i}", "nhdpv2_comid": i}
            )
            for i in range(5)
        ]
        with patch("pynhd.ogcapi.ar.retrieve_json") as mock_retrieve:
            mock_retrieve.return_value = _make_items_response(features)
            result = gcx.bybox((-69.77, 45.07, -69.31, 45.45))
        assert len(result) == 5


# ---- Tests: FabricData subclass --------------------------------------------

FABRIC_BBOX = [-170.0, 15.0, -51.0, 72.0]
FABRIC_QUERYABLES_URL = "https://api.water.usgs.gov/fabric/pygeoapi/collections/gagesii/queryables"


def _fabric_side_effect(*url_lists: list[str], **_: Any) -> list[dict[str, Any]]:
    url = url_lists[0][0]
    if "queryables" in url:
        return _make_queryables_response(
            {
                "staid": {"title": "staid", "type": "string"},
                "staname": {"title": "staname", "type": "string"},
                "drain_sqkm": {"title": "drain_sqkm", "type": "number"},
            },
        )
    if "schema" in url:
        return _make_schema_response("ogc_fid")
    if "collections" in url:
        return _make_collections_response(
            "gagesii",
            "Geospatial Attributes of Gages for Evaluating Streamflow, Version II",
            FABRIC_BBOX,
            FABRIC_QUERYABLES_URL,
        )
    msg = f"Unexpected URL: {url}"
    raise ValueError(msg)


class TestFabricData:
    def test_init(self):
        with patch("pynhd.ogcapi.ar.retrieve_json", side_effect=_fabric_side_effect):
            hf = pynhd.FabricData("gagesii")
        assert "gagesii" in hf.endpoints
        assert hf.item == "gagesii"

    def test_api_key_from_env(self):
        with (
            patch.dict("os.environ", {"USGS_API_KEY": "env-key-123"}),
            patch("pynhd.ogcapi.ar.retrieve_json", side_effect=_fabric_side_effect),
        ):
            hf = pynhd.FabricData("gagesii")
        assert hf.api_key == "env-key-123"
        assert hf._api_headers["api_key"] == "env-key-123"
        assert "pynhd" in hf._api_headers["User-Agent"]

    def test_explicit_api_key_overrides_env(self):
        with (
            patch.dict("os.environ", {"USGS_API_KEY": "env-key"}),
            patch("pynhd.ogcapi.ar.retrieve_json", side_effect=_fabric_side_effect),
        ):
            hf = pynhd.FabricData("gagesii", api_key="explicit-key")
        assert hf.api_key == "explicit-key"

    def test_max_nfeatures_limit(self):
        with (
            pytest.raises(InputRangeError),
            patch("pynhd.ogcapi.ar.retrieve_json", side_effect=_fabric_side_effect),
        ):
            pynhd.FabricData("gagesii", max_nfeatures=2000)

    def test_byid(self):
        with patch("pynhd.ogcapi.ar.retrieve_json", side_effect=_fabric_side_effect):
            hf = pynhd.FabricData("gagesii")
        feature = _make_feature(
            1,
            {"staid": "01031500", "staname": "Test Gauge", "drain_sqkm": 123.4},
        )
        with patch("pynhd.ogcapi.ar.retrieve_json") as mock_retrieve:
            mock_retrieve.side_effect = [_schema_resp(), _make_items_response([feature])]
            result = hf.byid("staid", "01031500")
        assert len(result) == 1
        assert result.iloc[0]["staid"] == "01031500"

    def test_bycql(self):
        with patch("pynhd.ogcapi.ar.retrieve_json", side_effect=_fabric_side_effect):
            hf = pynhd.FabricData("gagesii")
        features = _multi_features(3)
        with patch("pynhd.ogcapi.ar.retrieve_json") as mock_retrieve:
            mock_retrieve.side_effect = [_schema_resp(), _make_items_response(features)]
            result = hf.bycql({"op": "gt", "args": [{"property": "drain_sqkm"}, 45000]})
        assert len(result) == 3

    def test_bygeometry(self):
        with patch("pynhd.ogcapi.ar.retrieve_json", side_effect=_fabric_side_effect):
            hf = pynhd.FabricData("gagesii")
        features = _multi_features(2)
        with patch("pynhd.ogcapi.ar.retrieve_json") as mock_retrieve:
            mock_retrieve.side_effect = [_schema_resp(), _make_items_response(features)]
            # Use a bbox that contains the mock feature coordinates at (-69.5, 45.2)
            result = hf.bygeometry((-69.77, 45.07, -69.31, 45.45))
        assert len(result) == 2

    def test_byitem(self):
        with patch("pynhd.ogcapi.ar.retrieve_json", side_effect=_fabric_side_effect):
            hf = pynhd.FabricData("gagesii")
        feature = _make_feature(
            1,
            {"staid": "01031500", "staname": "Test Gauge", "drain_sqkm": 123.4},
        )
        with patch("pynhd.ogcapi.ar.retrieve_json") as mock_retrieve:
            mock_retrieve.return_value = [feature]
            result = hf.byitem("01031500")
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 1


# ---- Tests: NWIS subclass --------------------------------------------------

NWIS_BBOX = [-180.0, -90.0, 180.0, 90.0]
NWIS_QUERYABLES_URL = (
    "https://api.waterdata.usgs.gov/ogcapi/v0/collections/monitoring-locations/queryables"
)


def _nwis_side_effect(*url_lists: list[str], **_: Any) -> list[dict[str, Any]]:
    url = url_lists[0][0]
    if "queryables" in url:
        return _make_queryables_response(
            {
                "agency_code": {"title": "agency_code", "type": "string"},
                "monitoring_location_number": {
                    "title": "monitoring_location_number",
                    "type": "string",
                },
                "monitoring_location_name": {
                    "title": "monitoring_location_name",
                    "type": "string",
                },
                "state_name": {"title": "state_name", "type": "string"},
                "site_type": {"title": "site_type", "type": "string"},
                "altitude": {"title": "altitude", "type": "number"},
            },
        )
    if "schema" in url:
        return _make_schema_response("id")
    if "collections" in url:
        return _make_collections_response(
            "monitoring-locations",
            "Basic monitoring location information",
            NWIS_BBOX,
            NWIS_QUERYABLES_URL,
        )
    msg = f"Unexpected URL: {url}"
    raise ValueError(msg)


class TestNWIS:
    def test_init(self):
        with patch("pynhd.ogcapi.ar.retrieve_json", side_effect=_nwis_side_effect):
            nwis = pynhd.NWIS("monitoring-locations")
        assert "monitoring-locations" in nwis.endpoints
        assert nwis.item == "monitoring-locations"

    def test_no_dev_url(self):
        with patch("pynhd.ogcapi.ar.retrieve_json", side_effect=_nwis_side_effect):
            nwis = pynhd.NWIS("monitoring-locations")
        with pytest.raises(InputValueError, match="dev"):
            nwis.dev = True

    def test_api_key_header_name(self):
        with patch("pynhd.ogcapi.ar.retrieve_json", side_effect=_nwis_side_effect):
            nwis = pynhd.NWIS("monitoring-locations", api_key="test-key")
        assert nwis._api_headers["api_key"] == "test-key"
        assert "pynhd" in nwis._api_headers["User-Agent"]

    def test_no_api_key(self):
        with (
            patch.dict("os.environ", {}, clear=False),
            patch("pynhd.ogcapi.ar.retrieve_json", side_effect=_nwis_side_effect),
        ):
            os.environ.pop("USGS_API_KEY", None)
            nwis = pynhd.NWIS("monitoring-locations")
        assert "api_key" not in nwis._api_headers
        assert "pynhd" in nwis._api_headers["User-Agent"]

    def test_api_key_from_env(self):
        with (
            patch.dict("os.environ", {"USGS_API_KEY": "env-key-456"}),
            patch("pynhd.ogcapi.ar.retrieve_json", side_effect=_nwis_side_effect),
        ):
            nwis = pynhd.NWIS("monitoring-locations")
        assert nwis.api_key == "env-key-456"
        assert nwis._api_headers["api_key"] == "env-key-456"
        assert "pynhd" in nwis._api_headers["User-Agent"]

    def test_explicit_api_key_overrides_env(self):
        with (
            patch.dict("os.environ", {"USGS_API_KEY": "env-key"}),
            patch("pynhd.ogcapi.ar.retrieve_json", side_effect=_nwis_side_effect),
        ):
            nwis = pynhd.NWIS("monitoring-locations", api_key="explicit-key")
        assert nwis.api_key == "explicit-key"

    def test_max_nfeatures_limit(self):
        with (
            pytest.raises(InputRangeError),
            patch("pynhd.ogcapi.ar.retrieve_json", side_effect=_nwis_side_effect),
        ):
            pynhd.NWIS("monitoring-locations", max_nfeatures=60000)

    def test_byid(self):
        with patch("pynhd.ogcapi.ar.retrieve_json", side_effect=_nwis_side_effect):
            nwis = pynhd.NWIS("monitoring-locations")
        feature = _make_feature(
            1,
            {
                "agency_code": "USGS",
                "monitoring_location_number": "01031500",
                "monitoring_location_name": "Test River",
                "state_name": "Maine",
                "site_type": "Stream",
                "altitude": 100.5,
            },
        )
        with patch("pynhd.ogcapi.ar.retrieve_json") as mock_retrieve:
            mock_retrieve.side_effect = [
                _make_schema_response("id"),
                _make_items_response([feature]),
            ]
            result = nwis.byid("monitoring_location_number", "01031500")
        assert len(result) == 1
        assert result.iloc[0]["agency_code"] == "USGS"

    def test_bybox(self):
        with patch("pynhd.ogcapi.ar.retrieve_json", side_effect=_nwis_side_effect):
            nwis = pynhd.NWIS("monitoring-locations")
        features = [
            _make_feature(
                i,
                {
                    "agency_code": "USGS",
                    "monitoring_location_number": f"0103150{i}",
                    "monitoring_location_name": f"River {i}",
                    "state_name": "Maine",
                    "site_type": "Stream",
                    "altitude": float(i * 10),
                },
            )
            for i in range(3)
        ]
        with patch("pynhd.ogcapi.ar.retrieve_json") as mock_retrieve:
            mock_retrieve.side_effect = [
                _make_schema_response("id"),
                _make_items_response(features),
            ]
            result = nwis.bybox((-69.77, 45.07, -69.31, 45.45))
        assert len(result) == 3

    def test_bycql(self):
        with patch("pynhd.ogcapi.ar.retrieve_json", side_effect=_nwis_side_effect):
            nwis = pynhd.NWIS("monitoring-locations")
        features = _multi_features(5)
        with patch("pynhd.ogcapi.ar.retrieve_json") as mock_retrieve:
            mock_retrieve.side_effect = [
                _make_schema_response("id"),
                _make_items_response(features),
            ]
            result = nwis.bycql(
                {"op": "eq", "args": [{"property": "state_name"}, "Maine"]},
            )
        assert len(result) == 5

    def test_byitem(self):
        with patch("pynhd.ogcapi.ar.retrieve_json", side_effect=_nwis_side_effect):
            nwis = pynhd.NWIS("monitoring-locations")
        feature = _make_feature(
            1,
            {
                "agency_code": "USGS",
                "monitoring_location_number": "01031500",
                "monitoring_location_name": "Test River",
                "state_name": "Maine",
                "site_type": "Stream",
                "altitude": 100.5,
            },
        )
        with patch("pynhd.ogcapi.ar.retrieve_json") as mock_retrieve:
            mock_retrieve.return_value = [feature]
            result = nwis.byitem("USGS-01031500")
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 1


# ============================================================================
# Network tests — real calls to external services
# Run with: pytest -m network
# Skip with: pytest -m "not network"
# ============================================================================


def _fetch_items(service: OGCAPIBase, item: str, n: int = 10) -> list[dict[str, Any]]:
    """Fetch the first ``n`` items from a collection's items endpoint."""
    url = f"{service.endpoints[item].url}?f=json&limit={n}"
    kwds = [{"headers": service._api_headers}] if service._api_headers else None
    resp = ar.retrieve_json([url], kwds)
    return resp  # type: ignore[return-value]


# ---- GeoConnex network tests -----------------------------------------------

GCX_ENDPOINTS = [
    "hu02",
    "hu04",
    "hu06",
    "hu08",
    "hu10",
    "hu12",
    "nat_aq",
    "principal_aq",
    "sec_hydrg_reg",
    "gages",
    "mainstems",
    "dams",
    "pws",
    "states",
    "counties",
    "aiannh",
    "cbsa",
    "ua10",
]


@network
class TestGeoConnexNetwork:
    def test_all_endpoints_discovered(self):
        gcx = pynhd.GeoConnex()
        for ep in GCX_ENDPOINTS:
            assert ep in gcx.endpoints, f"Endpoint '{ep}' not found"

    @pytest.mark.parametrize("item", GCX_ENDPOINTS)
    def test_endpoint_items(self, item: str):
        gcx = pynhd.GeoConnex(item, max_nfeatures=10)
        resp = _fetch_items(gcx, item)
        assert resp[0].get("features"), f"No features returned for '{item}'"
        assert len(resp[0]["features"]) > 0


# ---- FabricData network tests ----------------------------------------------

FABRIC_ENDPOINTS = [
    "catchmentsp",
    "nhdarea",
    "nhdflowline_network",
    "nhdflowline_nonnetwork",
    "nhdwaterbody",
    "gagesii",
    "gagesii-basins",
    "nhdplusv2-huc02",
    "nhdplusv2-huc04",
    "nhdplusv2-huc06",
    "nhdplusv2-huc08",
    "nhdplusv2-huc10",
    "nhdplusv2-huc12",
    "nhdplushr-huc02",
    "nhdplushr-huc04",
    "nhdplushr-huc06",
    "nhdplushr-huc08",
    "nhdplushr-huc10",
    "nhdplushr-huc12",
    "wbd02_20201026",
    "wbd04_20201026",
    "wbd06_20201026",
    "wbd08_20201026",
    "wbd10_20201026",
    "wbd12_20201026",
    "wbd02_20250107",
    "wbd04_20250107",
    "wbd06_20250107",
    "wbd08_20250107",
    "wbd10_20250107",
    "wbd12_20250107",
]


@network
class TestFabricDataNetwork:
    def test_all_endpoints_discovered(self):
        hf = pynhd.FabricData()
        for ep in FABRIC_ENDPOINTS:
            assert ep in hf.endpoints, f"Endpoint '{ep}' not found"

    @pytest.mark.parametrize("item", FABRIC_ENDPOINTS)
    def test_endpoint_items(self, item: str):
        hf = pynhd.FabricData(item, max_nfeatures=10)
        resp = _fetch_items(hf, item)
        assert resp[0].get("features"), f"No features returned for '{item}'"
        assert len(resp[0]["features"]) > 0


# ---- NWIS network tests ----------------------------------------------------

NWIS_ALL_ENDPOINTS = [
    "monitoring-locations",
    "daily",
    "latest-daily",
    "continuous",
    "latest-continuous",
    "field-measurements",
    "channel-measurements",
    "combined-metadata",
    "field-measurements-metadata",
    "time-series-metadata",
    "agency-codes",
    "altitude-datums",
    "aquifer-codes",
    "aquifer-types",
    "coordinate-accuracy-codes",
    "coordinate-datum-codes",
    "coordinate-method-codes",
    "counties",
    "hydrologic-unit-codes",
    "medium-codes",
]


@network
class TestNWISNetwork:
    def test_all_endpoints_discovered(self):
        nwis = pynhd.NWIS()
        for ep in NWIS_ALL_ENDPOINTS:
            assert ep in nwis.endpoints, f"Endpoint '{ep}' not found"

    @pytest.mark.parametrize("item", NWIS_ALL_ENDPOINTS)
    def test_endpoint_items(self, item: str):
        nwis = pynhd.NWIS(item, max_nfeatures=10)
        resp = _fetch_items(nwis, item)
        assert resp[0].get("features"), f"No features returned for '{item}'"
        assert len(resp[0]["features"]) > 0
