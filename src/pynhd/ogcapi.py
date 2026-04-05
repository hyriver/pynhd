"""Access to OGC API web services."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any, Literal, cast, overload

import cytoolz.curried as tlz
import geopandas as gpd
import pandas as pd
import shapely
from shapely import MultiPoint, MultiPolygon, Polygon
from shapely import box as shapely_box

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

__all__ = ["NWIS", "FabricData", "GeoConnex", "OGCAPIBase"]


@dataclass(frozen=True)
class EndPoints:
    name: str
    description: str
    url: str
    query_fields: list[str]
    extent: tuple[float, float, float, float]
    dtypes: dict[str, str]


class OGCAPIBase:
    """Base class for OGC API web services.

    Notes
    -----
    The ``geometry`` field of the query can be a Polygon, MultiPolygon,
    or tuple/list of length 4 (bbox) in ``EPSG:4326`` CRS. They should
    be within the extent of the service endpoint.

    Parameters
    ----------
    prod_url : str
        The production URL of the service.
    dev_url : str, optional
        The development URL of the service, defaults to ``None``.
        If ``None``, setting ``dev=True`` will raise an error.
    item : str, optional
        The item (service endpoint) to query, defaults to ``None``.
    dev : bool, optional
        Whether to use the development endpoint, defaults to ``False``.
    max_nfeatures : int, optional
        The maximum number of features to request from the service,
        defaults to 10000.
    max_nfeatures_limit : int, optional
        The maximum allowed value for ``max_nfeatures``,
        defaults to 10000.
    api_key : str, optional
        An API key for the service. If provided, the key is sent
        via the ``X-Api-Key`` header on all requests. This can
        increase rate limits for services that support it (e.g.,
        USGS Water Data APIs). Defaults to ``None``.
    """

    @staticmethod
    def _get_user_agent() -> str:
        """Build User-Agent string from package version."""
        try:
            ver = version("pynhd")
        except PackageNotFoundError:
            ver = "999"
        return f"pynhd/{ver} (https://github.com/hyriver/pynhd)"

    def __init__(
        self,
        prod_url: str,
        dev_url: str | None = None,
        item: str | None = None,
        dev: bool = False,
        max_nfeatures: int = 10000,
        max_nfeatures_limit: int = 10000,
        api_key: str | None = None,
    ) -> None:
        if max_nfeatures > max_nfeatures_limit:
            raise InputRangeError("max_nfeatures", f"<= {max_nfeatures_limit}")
        self.max_nfeatures = max_nfeatures
        self.dev_url = dev_url
        self.prod_url = prod_url
        self.api_key = api_key
        self._user_agent = self._get_user_agent()
        self.dev = dev
        self.item = item

    @property
    def _api_headers(self) -> dict[str, str]:
        """Return default headers including User-Agent and API key if set."""
        headers: dict[str, str] = {"User-Agent": self._user_agent}
        if self.api_key is not None:
            headers["X-Api-Key"] = self.api_key
        return headers

    @property
    def dev(self) -> bool:
        """Return the name of the endpoint."""
        return self._dev

    @dev.setter
    def dev(self, value: bool) -> None:
        if value and self.dev_url is None:
            raise InputValueError("dev", ("False (no dev URL available for this service)",))
        self._dev = value
        if value:
            self.base_url = f"{self.dev_url}/collections"
        else:
            self.base_url = f"{self.prod_url}/collections"

        resp = ar.retrieve_json(
            [self.base_url],
            [{"params": {"f": "json"}, "headers": self._api_headers}],
        )
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
            resp = ar.retrieve_json(
                [query_url],
                [{"headers": self._api_headers}] if self._api_headers else None,
            )
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
            pluck(ind="id"),
            pluck(ind="description"),
            pluck(ind="links"),
            pluck(ind="extent"),
            strict=False,
        )
        self.endpoints = {
            ep[0]: EndPoints(
                name=ep[0],
                description=ep[1],
                **get_links(ep[2]),  # ty: ignore[invalid-argument-type]  # pyright: ignore[reportArgumentType]
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
        """Get the sort attribute for the endpoint by querying its schema."""
        if endpoint not in self.endpoints:
            raise InputValueError("endpoint", list(self.endpoints.keys()))
        url = f"{self.endpoints[endpoint].url.rsplit('/', 1)[0]}/schema"
        resp = ar.retrieve_json(
            [url],
            [{"params": {"f": "json"}, "headers": self._api_headers}],
        )
        resp = cast("list[dict[str, Any]]", resp)
        schema: dict[str, dict[str, Any]] = resp[0].get("properties", {})
        attr = next((attr for attr, val in schema.items() if val.get("x-ogc-role") == "id"), None)
        if attr is None:
            # Fall back to the first queryable field if schema doesn't have x-ogc-role
            qfields = self.endpoints[endpoint].query_fields
            if qfields:
                return qfields[0]
            raise ServiceError("The service doesn't have a valid id field.", url)
        return attr

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

        headers = dict(self._api_headers)
        if "json" in kwds:
            url_kwds["filter-lang"] = "cql-json"
            headers["Content-Type"] = "application/query-cql-json"
        elif "data" in kwds:
            headers["Content-Type"] = "application/json"

        params = kwds | {"headers": headers}

        if len(json.dumps(params)) > 1000 or "json" in kwds or "data" in kwds:
            request_method = "post"
        else:
            request_method = "get"

        def _get_url(offset: int) -> str:
            url_kwds["offset"] = offset
            return f"{self.query_url}?{'&'.join([f'{k}={v}' for k, v in url_kwds.items()])}"

        url = _get_url(0)
        resp = ar.retrieve_json([url], [params], request_method=request_method)
        resp = cast("list[dict[str, Any]]", resp)
        if resp[0].get("code") is not None:
            ar.delete_url_cache(url, request_method=request_method)
            msg = f"{resp[0]['code']}: {resp[0]['description']}"
            raise ServiceError(msg, url)
        if "error" in resp[0]:
            ar.delete_url_cache(url, request_method=request_method)
            err = resp[0]["error"]
            msg = f"{err.get('code', 'Unknown')}: {err.get('message', 'Unknown error')}"
            raise ServiceError(msg, url)
        try:
            gdf = geoutils.json2geodf(resp)
        except (EmptyResponseError, AttributeError, TypeError, KeyError) as ex:
            raise ZeroMatchedError from ex

        n_matched = resp[0].get("numberMatched")
        if n_matched is not None and len(gdf) != int(n_matched):
            urls = [_get_url(i) for i in range(0, int(n_matched), self.max_nfeatures)]
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
        """Query the endpoint by geometry.

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

        bbox = geom1.bounds
        bbox_str = ",".join(f"{v:.6f}" for v in bbox)
        gdf = self._query({"bbox": bbox_str}, skip_geometry=False)

        pred_func = predicate.lower()
        if geometry2 is not None:
            geom2 = geoutils.geo2polygon(geometry2, crs, 4326)
            geom2 = shapely.set_precision(geom2, 1e-6)
            if not geom2.intersects(shapely_box(*self.item_extent)):
                raise InputRangeError("geometry", f"within {self.item_extent}")
            mask = getattr(gdf.geometry, pred_func)(geom2)
        else:
            mask = getattr(gdf.geometry, pred_func)(geom1)

        gdf = gdf.loc[mask].reset_index(drop=True)
        if len(gdf) == 0:
            raise ZeroMatchedError

        if skip_geometry:
            return pd.DataFrame(gdf.drop(columns="geometry"))
        return gdf

    @overload
    def byid(
        self,
        feature_name: str,
        feature_ids: list[str] | list[int] | str | int,
        skip_geometry: Literal[False] = False,
    ) -> gpd.GeoDataFrame: ...

    @overload
    def byid(
        self,
        feature_name: str,
        feature_ids: list[str] | list[int] | str | int,
        skip_geometry: Literal[True],
    ) -> pd.DataFrame: ...

    def byid(
        self,
        feature_name: str,
        feature_ids: list[str] | list[int] | str | int,
        skip_geometry: bool = False,
    ) -> gpd.GeoDataFrame | pd.DataFrame:
        """Query the endpoint by feature IDs.

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
        ftyped = pd.Series(list(fids), dtype=dtype)
        kwds = {"json": {"op": "in", "args": [{"property": feature_name}, ftyped.to_list()]}}
        return self._query(kwds, skip_geometry)

    def byitem(self, item_id: str | int) -> gpd.GeoDataFrame:
        """Query the endpoint by an item ID.

        Parameters
        ----------
        item_id : str or int
            The ID of the item to query. Note that ``item_id``
            is not necessarily the same as the provider's item ID.

        Returns
        -------
        geopandas.GeoDataFrame
            The query result as a ``geopandas.GeoDataFrame``.
        """
        if self.item is None or self.query_url is None:
            raise MissingItemError(["item"])
        url = f"{self.query_url}/{item_id}"
        resps = ar.retrieve_json(
            [url],
            [{"params": {"f": "json"}, "headers": self._api_headers}],
        )
        resp = cast("dict[str, Any]", resps[0])
        if resp.get("code") is not None:
            ar.delete_url_cache(url)
            msg = f"{resp['code']}: {resp['description']}"
            raise ServiceError(msg, url)
        if "error" in resp:
            ar.delete_url_cache(url)
            err = resp["error"]
            msg = f"{err.get('code', 'Unknown')}: {err.get('message', 'Unknown error')}"
            raise ServiceError(msg, url)
        return gpd.GeoDataFrame.from_features(
            cast("dict[str, Any]", {"type": "FeatureCollection", "features": [resp]}),
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
        """Query the endpoint by a CQL2 dictionary.

        Notes
        -----
        Only Basic CQL2 queries are supported. For more information
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
        """Query the endpoint by a CQL filter string.

        Notes
        -----
        Only simple CQL queries are supported. For more information
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
        """Query the endpoint by bounding box.

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


class GeoConnex(OGCAPIBase):
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
    api_key : str, optional
        An API key for the service, defaults to ``None``.
    """

    def __init__(
        self,
        item: str | None = None,
        dev: bool = False,
        max_nfeatures: int = 10000,
        api_key: str | None = None,
    ) -> None:
        prod_url = ServiceURL().restful.geoconnex
        dev_url = prod_url.replace(".us", ".dev")
        super().__init__(
            prod_url=prod_url,
            dev_url=dev_url,
            item=item,
            dev=dev,
            max_nfeatures=max_nfeatures,
            max_nfeatures_limit=10000,
            api_key=api_key,
        )

    def _get_sort_attr(self, endpoint: str | None) -> str:
        """Get the sort attribute for the endpoint."""
        if endpoint not in self.endpoints:
            raise InputValueError("endpoint", list(self.endpoints.keys()))
        return "uri"


class FabricData(OGCAPIBase):
    """Access to the FabricData API.

    Notes
    -----
    The ``geometry`` field of the query can be a Polygon, MultiPolygon,
    or tuple/list of length 4 (bbox) in ``EPSG:4326`` CRS. They should
    be within the extent of the FabricData endpoint.

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
    api_key : str, optional
        An API key for the service. If not provided, falls back to
        the ``USGS_API_KEY`` environment variable. Defaults
        to ``None``. Obtain a key `here <https://api.waterdata.usgs.gov/signup/>`__.
    """

    def __init__(
        self,
        item: str | None = None,
        dev: bool = False,
        max_nfeatures: int = 1000,
        api_key: str | None = None,
    ) -> None:
        if api_key is None:
            api_key = os.environ.get("USGS_API_KEY")
        super().__init__(
            prod_url="https://api.water.usgs.gov/fabric/pygeoapi",
            dev_url="https://labs-beta.waterdata.usgs.gov/api/fabric/pygeoapi",
            item=item,
            dev=dev,
            max_nfeatures=max_nfeatures,
            max_nfeatures_limit=1000,
            api_key=api_key,
        )

    @property
    def _api_headers(self) -> dict[str, str]:
        """Return USGS API key headers if an API key is set."""
        headers: dict[str, str] = {"User-Agent": self._user_agent}
        if self.api_key is not None:
            headers["api_key"] = self.api_key
        return headers


class NWIS(OGCAPIBase):
    """Access to the USGS Water Data OGC API.

    Notes
    -----
    The ``geometry`` field of the query can be a Polygon, MultiPolygon,
    or tuple/list of length 4 (bbox) in ``EPSG:4326`` CRS. They should
    be within the extent of the NWIS endpoint.

    For more information visit
    `USGS Water Data API <https://api.waterdata.usgs.gov/ogcapi/v0>`__.

    Parameters
    ----------
    item : str, optional
        The item (service endpoint) to query. Valid endpoints include:

        - ``monitoring-locations`` for monitoring location information
        - ``daily`` for daily statistical summaries of continuous data
        - ``latest-daily`` for most recent daily values
        - ``continuous`` for high-frequency sensor data (typically 15-min)
        - ``latest-continuous`` for most recent continuous observations
        - ``field-measurements`` for physically measured field values
        - ``channel-measurements`` for streamflow channel measurements
        - ``combined-metadata`` for consolidated metadata by site
        - ``field-measurements-metadata`` for field measurement metadata
        - ``time-series-metadata`` for time series metadata
        - ``agency-codes`` for agency identifiers
        - ``altitude-datums`` for vertical datum references
        - ``aquifer-codes`` for local aquifer identification codes
        - ``aquifer-types`` for groundwater aquifer classifications
        - ``coordinate-accuracy-codes`` for coordinate accuracy designations
        - ``coordinate-datum-codes`` for horizontal datum codes
        - ``coordinate-method-codes`` for coordinate determination methods
        - ``counties`` for county identifiers with FIPS codes
        - ``hydrologic-unit-codes`` for HUC drainage areas
        - ``medium-codes`` for environmental media types

    max_nfeatures : int, optional
        The maximum number of features to request from the service,
        defaults to 10000.
    api_key : str, optional
        An API key for the service. If not provided, falls back to
        the ``USGS_API_KEY`` environment variable. Defaults
        to ``None``. Obtain a key `here <https://api.waterdata.usgs.gov/signup/>`__.
    """

    def __init__(
        self,
        item: str | None = None,
        max_nfeatures: int = 10000,
        api_key: str | None = None,
    ) -> None:
        if api_key is None:
            api_key = os.environ.get("USGS_API_KEY")
        super().__init__(
            prod_url="https://api.waterdata.usgs.gov/ogcapi/v0",
            item=item,
            max_nfeatures=max_nfeatures,
            max_nfeatures_limit=50000,
            api_key=api_key,
        )

    @property
    def _api_headers(self) -> dict[str, str]:
        """Return USGS API key headers if an API key is set."""
        headers: dict[str, str] = {"User-Agent": self._user_agent}
        if self.api_key is not None:
            headers["api_key"] = self.api_key
        return headers
