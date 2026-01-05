"""Base classes for PyNHD functions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast, overload

import cytoolz.curried as tlz
import geopandas as gpd
import pandas as pd
import shapely
from shapely import MultiPoint, MultiPolygon, Polygon
from shapely import box as shapely_box
from shapely.geometry import mapping as shapely_mapping

import async_retriever as ar
import pygeoutils as geoutils
from pygeoogc import ServiceURL
from pygeoutils.exceptions import EmptyResponseError
from pynhd.exceptions import (
    InputRangeError,
    InputValueError,
    MissingItemError,
    ServiceError,
    ZeroMatchedError,
)

if TYPE_CHECKING:

    from pyproj import CRS
    from shapely import LineString, Point

    CRSType = int | str | CRS
    GeoType = (
        Polygon | MultiPolygon | MultiPoint | LineString | Point | tuple[float, float, float, float]
    )

__all__ = ["GeoConnex", "HydroFabric"]


@dataclass(frozen=True)
class EndPoints:
    name: str
    description: str
    url: str
    query_fields: list[str]
    extent: tuple[float, float, float, float]
    dtypes: dict[str, str]


class GeoConnex:
    """Access to the GeoConnex API.

    Notes
    -----
    The ``geometry`` field of the query can be a Polygon, MultiPolygon,
    or tuple/list of length 4 (bbox) in ``EPSG:4326`` CRS. They should
    be within the extent of the GeoConnex endpoint.

    Parameters
    ----------
    The item (service endpoint) to query. Valid endpoints are:

        - ``hu02`` for Two-digit Hydrologic Regions
        - ``hu04`` for Four-digit Hydrologic Subregion
        - ``hu06`` for Six-digit Hydrologic Basins
        - ``hu08`` for Eight-digit Hydrologic Subbasins
        - ``hu10`` for Ten-digit Watersheds
        - ``nat_aq`` for National Aquifers of the United States from
            USGS National Water Information System National Aquifer code list.
        - ``principal_aq`` for Principal Aquifers of the United States from
            2003 USGS data release
        - ``sec_hydrg_reg`` for Secondary Hydrogeologic Regions of the
            Conterminous United States from 2018 USGS data release
        - ``gages`` for US Reference Stream Gauge Monitoring Locations
        - ``mainstems`` for US Reference Mainstem Rivers
        - ``states`` for U.S. States
        - ``counties`` for U.S. Counties
        - ``aiannh`` for Native American Lands
        - ``cbsa`` for U.S. Metropolitan and Micropolitan Statistical Areas
        - ``ua10`` for Urbanized Areas and Urban Clusters (2010 Census)
        - ``places`` for U.S. legally incororated and Census designated places
        - ``pws`` for U.S. Public Water Systems
        - ``dams`` for US Reference Dams

    dev : bool, optional
        Whether to use the development endpoint, defaults to ``False``.
    max_nfeatures : int, optional
        The maximum number of features to request from the service,
        defaults to 10000.
    """

    def __init__(
        self, item: str | None = None, dev: bool = False, max_nfeatures: int = 10000
    ) -> None:
        if max_nfeatures > 1000:
            raise InputRangeError("max_nfeatures", "<= 1000")
        self.max_nfeatures = max_nfeatures
        self.dev_url = ServiceURL().restful.geoconnex.replace('.us', '.dev')
        self.prod_url = ServiceURL().restful.geoconnex
        self.dev = dev
        self.item = item

    @property
    def dev(self) -> bool:
        """Return the name of the endpoint."""
        return self._dev

    @dev.setter
    def dev(self, value: bool) -> None:
        self._dev = value
        if value:
            self.base_url = f"{self.dev_url}/collections"
        else:
            self.base_url = f"{self.prod_url}/collections"

        resp = ar.retrieve_json([self.base_url], [{"params": {"f": "json"}}])
        resp = cast("list[dict[str, Any]]", resp)
        pluck = tlz.partial(tlz.pluck, seqs=resp[0]["collections"])

        def get_links(
            links: list[dict[str, Any]],
        ) -> dict[str, str | tuple[str, ...] | dict[str, str]]:
            """Get links."""
            query_url = next(
                (
                    lk["href"]
                    for lk in links
                    if lk["type"] == "application/schema+json" and "queryables" in lk["rel"]
                ),
                None,
            )
            if query_url is None:
                raise ServiceError("The service doesn't have a valid queryable link.")
            resp = ar.retrieve_json([query_url])
            resp = cast("list[dict[str, Any]]", resp)
            prop: dict[str, dict[str, str]] = resp[0]["properties"]
            _ = prop.pop("geometry", None)
            type_map = {
                "integer": "int64",
                "string": "str",
                "number": "f8",
                "timestamp": "datetime64[ns]",
                "boolean": "bool",
            }
            dtypes = {v["title"]: type_map[v["type"]] for v in prop.values()}

            return {
                "url": f"{query_url.rsplit('/', 1)[0]}/items",
                "query_fields": tuple(prop),
                "dtypes": dtypes,
            }

        eps = zip(
            pluck(ind="id"), pluck(ind="description"), pluck(ind="links"), pluck(ind="extent")
        )
        self.endpoints = {
            ep[0]: EndPoints(
                name=ep[0],
                description=ep[1],
                **get_links(ep[2]),  # pyright: ignore[reportArgumentType]
                extent=tuple(ep[3]["spatial"]["bbox"][0]),
            )
            for ep in eps
        }

    @property
    def item(self) -> str | None:
        """Return the name of the endpoint."""
        return self._item

    @item.setter
    def item(self, value: str | None) -> None:
        self._item = value
        if value is not None:
            self.dev = self._dev
            valid = [f"- ``{i}`` for {e.description}" for i, e in self.endpoints.items()]
            if value not in self.endpoints:
                raise InputValueError("item", valid)
            self.query_url = self.endpoints[value].url
            self.item_extent = self.endpoints[value].extent
        else:
            self.query_url = None
    
    def _get_sort_attr(self, endpoint: str | None) -> str:
        """Get the schema of the endpoint."""
        if endpoint not in self.endpoints:
            raise InputValueError("endpoint", list(self.endpoints.keys()))
        return "uri"

    def _query(self, kwds: dict[str, Any], skip_geometry: bool) -> gpd.GeoDataFrame | pd.DataFrame:
        """Get response as a ``geopandas.GeoDataFrame``."""
        url_kwds = {
            "f": "json",
            "limit": self.max_nfeatures,
            "sortby": self._get_sort_attr(self.item),
            "skipGeometry": str(skip_geometry).lower(),
        }
        if "filter" in kwds:
            url_kwds["filter"] = kwds.pop("filter")
        if "bbox" in kwds:
            url_kwds["bbox"] = kwds.pop("bbox")

        if "json" in kwds:
            url_kwds["filter-lang"] = "cql-json"
            params = kwds | {"headers": {"Content-Type": "application/query-cql-json"}}
        elif "data" in kwds:
            params = kwds | {"headers": {"Content-Type": "application/json"}}
        else:
            params = kwds

        if len(json.dumps(params)) > 1000 or "json" in kwds or "data" in kwds:
            request_method = "post"
        else:
            request_method = "get"

        def _get_url(offset: int) -> str:
            url_kwds["offset"] = offset
            return f"{self.query_url}?{'&'.join([f'{k}={v}' for k, v in url_kwds.items()])}"

        resp = ar.retrieve_json([_get_url(0)], [params], request_method=request_method)
        resp = cast("list[dict[str, Any]]", resp)
        if resp[0].get("code") is not None:
            msg = f"{resp[0]['code']}: {resp[0]['description']}"
            raise ServiceError(msg, _get_url(0))
        try:
            gdf = geoutils.json2geodf(resp)
            n_matched = int(resp[0]["numberMatched"])
        except EmptyResponseError as ex:
            raise ZeroMatchedError from ex

        if len(gdf) != n_matched:
            urls = [_get_url(i) for i in range(0, n_matched, self.max_nfeatures)]
            resp = ar.retrieve_json(urls, [params] * len(urls), request_method=request_method)
            resp = cast("list[dict[str, Any]]", resp)
            gdf = geoutils.json2geodf(resp)

        if len(gdf) == 0:
            raise ZeroMatchedError

        if "geometry" in gdf:
            return gdf.reset_index(drop=True)
        return pd.DataFrame(gdf.reset_index(drop=True))

    @overload
    def bygeometry(
        self,
        geometry1: GeoType,
        geometry2: GeoType | None = ...,
        predicate: str = ...,
        crs: CRSType = ...,
        skip_geometry: Literal[False] = False,
    ) -> gpd.GeoDataFrame: ...

    @overload
    def bygeometry(
        self,
        geometry1: GeoType,
        geometry2: GeoType | None = ...,
        predicate: str = ...,
        crs: CRSType = ...,
        skip_geometry: Literal[True] = True,
    ) -> pd.DataFrame: ...

    def bygeometry(
        self,
        geometry1: GeoType,
        geometry2: GeoType | None = None,
        predicate: str = "intersects",
        crs: CRSType = 4326,
        skip_geometry: bool = False,
    ) -> gpd.GeoDataFrame | pd.DataFrame:
        """Query the GeoConnex endpoint by geometry.

        Parameters
        ----------
        geometry1 : Polygon or tuple of float
            The first geometry or bounding boxes to query. A bounding box is
            a tuple of length 4 in the form of ``(xmin, ymin, xmax, ymax)``.
            For example, an spatial query for a single geometry would be
            ``INTERSECTS(geom, geometry1)``.
        geometry2 : Polygon or tuple of float, optional
            The second geometry or bounding boxes to query. A bounding box is
            a tuple of length 4 in the form of ``(xmin, ymin, xmax, ymax)``.
            Default is ``None``. For example, an spatial query for a two
            geometries would be ``CROSSES(geometry1, geometry2)``.
        predicate : str, optional
            The predicate to use, by default ``intersects``. Supported
            predicates are ``intersects``, ``equals``, ``disjoint``, ``touches``,
            ``within``, ``overlaps``, ``crosses`` and ``contains``.
        crs : int or str or pyproj.CRS, optional
            The CRS of the polygon, by default ``EPSG:4326``. If the input
            is a ``geopandas.GeoDataFrame`` or ``geopandas.GeoSeries``,
            this argument will be ignored.
        skip_geometry: bool, optional
            If ``True``, no geometry will not be returned.

        Returns
        -------
        geopandas.GeoDataFrame
            The query result as a ``geopandas.GeoDataFrame``.
        """
        if self.item is None or self.query_url is None:
            raise MissingItemError(["item"])

        valid_predicates = (
            "intersects",
            "equals",
            "disjoint",
            "touches",
            "within",
            "overlaps",
            "crosses",
            "contains",
        )
        if predicate.lower() not in valid_predicates:
            raise InputValueError("predicate", valid_predicates)

        geom1 = geoutils.geo2polygon(geometry1, crs, 4326)
        geom1 = shapely.set_precision(geom1, 1e-6)
        if not geom1.intersects(shapely_box(*self.item_extent)):
            raise InputRangeError("geometry", f"within {self.item_extent}")
        try:
            geom1_json = json.loads(shapely.to_geojson(geom1))
        except AttributeError:
            geom1_json = shapely_mapping(geom1)

        if geometry2 is None:
            return self._query(
                {
                    "json": {
                        "op": f"s_{predicate.lower()}",
                        "args": [{"property": "geom"}, geom1_json],
                    },
                },
                skip_geometry,
            )

        geom2 = geoutils.geo2polygon(geometry2, crs, 4326)
        geom2 = shapely.set_precision(geom2, 1e-6)
        if not geom2.intersects(shapely_box(*self.item_extent)):
            raise InputRangeError("geometry", f"within {self.item_extent}")
        try:
            geom2_json = json.loads(shapely.to_geojson(geom2))
        except AttributeError:
            geom2_json = shapely_mapping(geom2)
        return self._query(
            {
                "json": {"op": f"s_{predicate.lower()}", "args": [geom1_json, geom2_json]},
            },
            skip_geometry,
        )

    @overload
    def byid(
        self,
        feature_name: str,
        feature_ids: list[str] | str,
        skip_geometry: Literal[False] = False,
    ) -> gpd.GeoDataFrame: ...

    @overload
    def byid(
        self,
        feature_name: str,
        feature_ids: list[str] | str,
        skip_geometry: Literal[True],
    ) -> pd.DataFrame: ...

    def byid(
        self,
        feature_name: str,
        feature_ids: list[str] | str,
        skip_geometry: bool = False,
    ) -> gpd.GeoDataFrame | pd.DataFrame:
        """Query the GeoConnex endpoint.

        Parameters
        ----------
        feature_name : str
            The name of the feature to query.
        feature_ids : list or str
            The IDs of the feature to query.
        skip_geometry: bool, optional
            If ``True``, no geometry will not be returned, by default ``False``.

        Returns
        -------
        geopandas.GeoDataFrame
            The query result as a ``geopandas.GeoDataFrame`` or a ``pandas.DataFrame``.
        """
        if self.item is None or self.query_url is None:
            raise MissingItemError(["item"])

        fids = {feature_ids} if isinstance(feature_ids, (str, int)) else set(feature_ids)
        valid_keys = self.endpoints[self.item].query_fields
        if feature_name not in valid_keys:
            raise InputValueError("feature_name", valid_keys)
        dtype = self.endpoints[self.item].dtypes[feature_name]
        ftyped = pd.Series(list(fids)).astype(dtype)  # pyright: ignore[reportCallIssue,reportArgumentType]
        kwds = {"json": {"op": "in", "args": [{"property": feature_name}, ftyped.to_list()]}}
        return self._query(kwds, skip_geometry)

    def byitem(self, item_id: str) -> gpd.GeoDataFrame:
        """Query the GeoConnex endpoint by an item ID.

        Parameters
        ----------
        item_id : str
            The ID of the item to query. Note that this GeoConnex's item ID which
            is not necessarily the same as the provider's item ID. For example,
            for querying gages, the item ID is not the same as the USGS gage ID
            but for querying HUC02, the item ID is the same as the HUC02 ID.

        Returns
        -------
        geopandas.GeoDataFrame
            The query result as a ``geopandas.GeoDataFrame``.
        """
        if self.item is None or self.query_url is None:
            raise MissingItemError(["item"])
        url = f"{self.query_url}/{item_id}"
        resps = ar.retrieve_json([url], [{"params": {"f": "json"}}])
        resp = cast("dict[str, Any]", resps[0])
        if resp.get("code") is not None:
            msg = f"{resp['code']}: {resp['description']}"
            raise ServiceError(msg, url)
        return gpd.GeoDataFrame.from_features(
            {"type": "FeatureCollection", "features": [resp]},  # pyright: ignore[reportArgumentType]
            crs=4326,
        )

    @overload
    def bycql(
        self,
        cql_dict: dict[str, Any],
        skip_geometry: Literal[False] = False,
    ) -> gpd.GeoDataFrame: ...

    @overload
    def bycql(
        self,
        cql_dict: dict[str, Any],
        skip_geometry: Literal[True],
    ) -> pd.DataFrame: ...

    def bycql(
        self,
        cql_dict: dict[str, Any],
        skip_geometry: bool = False,
    ) -> gpd.GeoDataFrame | pd.DataFrame:
        """Query the GeoConnex endpoint.

        Notes
        -----
        GeoConnex only supports Basinc CQL2 queries. For more information
        and examples visit this
        `link <https://docs.ogc.org/is/21-065r2/21-065r2.html#cql2-core>`__.
        Use this for non-spatial queries, since there's a dedicated method
        for spatial queries, :meth:`.bygeometry`.

        Parameters
        ----------
        cql_dict : dict
            A valid CQL dictionary (non-spatial queries).
        skip_geometry: bool, optional
            If ``True``, no geometry will not be returned, by default ``False``.

        Returns
        -------
        geopandas.GeoDataFrame
            The query result as a ``geopandas.GeoDataFrame``.
        """
        if self.item is None or self.query_url is None:
            raise MissingItemError(["item"])
        return self._query({"json": cql_dict}, skip_geometry)

    def byfilter(
        self,
        filter_str: str,
        skip_geometry: bool = False,
    ) -> gpd.GeoDataFrame | pd.DataFrame:
        """Query the GeoConnex endpoint.

        Notes
        -----
        GeoConnex only supports simple CQL queries. For more information
        and examples visit https://portal.ogc.org/files/96288

        Parameters
        ----------
        filter_str : dict
            A valid filter string. The filter string shouldn't be long
            since a GET request is used.
        skip_geometry: bool, optional
            If ``True``, no geometry will not be returned, by default ``False``.

        Returns
        -------
        geopandas.GeoDataFrame
            The query result as a ``geopandas.GeoDataFrame``.
        """
        if self.item is None or self.query_url is None:
            raise MissingItemError(["item"])

        return self._query({"filter": filter_str}, skip_geometry)

    def bybox(
        self,
        bbox: tuple[float, float, float, float],
        skip_geometry: bool = False,
    ) -> gpd.GeoDataFrame | pd.DataFrame:
        """Query the GeoConnex endpoint by bounding box.

        Parameters
        ----------
        bbox : tuple
            A bounding box in the form of ``(xmin, ymin, xmax, ymax)``,
            in ``EPSG:4326`` CRS, i.e., decimal degrees.
        skip_geometry: bool, optional
            If ``True``, no geometry will not be returned, by default ``False``.

        Returns
        -------
        geopandas.GeoDataFrame
            The query result as a ``geopandas.GeoDataFrame``.
        """
        if self.item is None or self.query_url is None:
            raise MissingItemError(["item"])
        return self._query({"bbox": ",".join(f"{v:.6f}" for v in bbox)}, skip_geometry)

    def __repr__(self) -> str:
        if self.item is None:
            return "\n".join(
                [
                    "Available Endpoints:",
                    "\n".join(f"'{k}': {v.description}" for k, v in self.endpoints.items()),
                ]
            )
        return "\n".join(
            [
                f"Item: '{self.item}'",
                f"Description: {self.endpoints[self.item].description}",
                f"Queryable Fields: {', '.join(self.endpoints[self.item].query_fields)}",
                f"Extent: {self.endpoints[self.item].extent}",
            ],
        )


class HydroFabric(GeoConnex):
    """Access to the HydroFabric API.

    Notes
    -----
    The ``geometry`` field of the query can be a Polygon, MultiPolygon,
    or tuple/list of length 4 (bbox) in ``EPSG:4326`` CRS. They should
    be within the extent of the HydroFabric endpoint.

    Parameters
    ----------
    The item (service endpoint) to query. Valid endpoints are:

        - ``catchmentsp`` for Catchmentsp layer from NHDPlusV2 National Seamless Geodatabase
        - ``nhdarea`` for National Hydrography Dataset (NHD) Area features from the NHDPlusV2
        - ``nhdflowline_network`` for National Hydrography Dataset (NHD) Flowline Network from NHDPlusV2
        - ``nhdflowline_nonnetwork`` for National Hydrography Dataset (NHD) Flowline Nonnetwork from NHDPlusV2
        - ``nhdwaterbody`` for National Hydrography Dataset (NHD) Waterbody from NHDPlusV2
        - ``gagesii`` for Geospatial Attributes of Gages for Evaluating Streamflow, Version II - https://doi.org/10.5066/P96CPHOT
        - ``gagesii-basins`` for Basins for Geospatial Attributes of Gages for Evaluating Streamflow, version II - https://doi.org/10.5066/P96CPHOT
        - ``nhdplusv2-huc02`` for HUC02 created by aggregating NHDPlusV2 HUC12s
        - ``nhdplusv2-huc04`` for HUC04 created by aggregating NHDPlusV2 HUC12s
        - ``nhdplusv2-huc06`` for HUC06 created by aggregating NHDPlusV2 HUC12s
        - ``nhdplusv2-huc08`` for HUC08 created by aggregating NHDPlusV2 HUC12s
        - ``nhdplusv2-huc10`` for HUC10 created by aggregating NHDPlusV2 HUC12s
        - ``nhdplusv2-huc12`` for HUC12 layer from NHDPlusV2
        - ``nhdplushr-huc02`` for HUC02 created by aggregating NHDPlusHR HUC12s
        - ``nhdplushr-huc04`` for HUC04 created by aggregating NHDPlusHR HUC12s
        - ``nhdplushr-huc06`` for HUC06 created by aggregating NHDPlusHR HUC12s
        - ``nhdplushr-huc08`` for HUC08 created by aggregating NHDPlusHR HUC12s
        - ``nhdplushr-huc10`` for HUC10 created by aggregating NHDPlusHR HUC12s
        - ``nhdplushr-huc12`` for HUC12 layer from NHDPlusHR
        - ``wbd02_20201026`` for 2020 WBD Snapshot HUC02 Polygons
        - ``wbd04_20201026`` for 2020 WBD Snapshot HUC04 Polygons
        - ``wbd06_20201026`` for 2020 WBD Snapshot HUC06 Polygons
        - ``wbd08_20201026`` for 2020 WBD Snapshot HUC08 Polygons
        - ``wbd10_20201026`` for 2020 WBD Snapshot HUC10 Polygons
        - ``wbd12_20201026`` for 2020 WBD Snapshot HUC12 Polygons
        - ``wbd02_20250107`` for 2025 WBD Snapshot HUC02 Polygons
        - ``wbd04_20250107`` for 2025 WBD Snapshot HUC04 Polygons
        - ``wbd06_20250107`` for 2025 WBD Snapshot HUC06 Polygons
        - ``wbd08_20250107`` for 2025 WBD Snapshot HUC08 Polygons
        - ``wbd10_20250107`` for 2025 WBD Snapshot HUC10 Polygons
        - ``wbd12_20250107`` for 2025 WBD Snapshot HUC12 Polygons

    dev : bool, optional
        Whether to use the development endpoint, defaults to ``False``.
    max_nfeatures : int, optional
        The maximum number of features to request from the service,
        defaults to 1000.
    """

    def __init__(
        self, item: str | None = None, dev: bool = False, max_nfeatures: int = 1000
    ) -> None:
        if max_nfeatures > 1000:
            raise InputRangeError("max_nfeatures", "<= 1000")
        self.max_nfeatures = max_nfeatures
        self.dev_url = "https://labs-beta.waterdata.usgs.gov/api/fabric/pygeoapi"
        self.prod_url = "https://api.water.usgs.gov/fabric/pygeoapi"
        self.dev = dev
        self.item = item
    
    def _get_sort_attr(self, endpoint: str | None) -> str:
        """Get the schema of the endpoint."""
        if endpoint not in self.endpoints:
            raise InputValueError("endpoint", list(self.endpoints.keys()))
        url = f"{self.endpoints[endpoint].url.rsplit('/', 1)[0]}/schema"
        resp = ar.retrieve_json([url], [{"params": {"f": "json"}}])
        resp = cast("list[dict[str, dict[str, str]]]", resp)
        schema = resp[0]["properties"]
        attr = next((attr for attr, val in schema.items() if val.get("x-ogc-role") == "id"), None)
        if attr is None:
            raise ServiceError("The service doesn't have a valid id field.", url)
        return attr
