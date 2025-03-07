"""Base classes for PyNHD functions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
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
from pygeoogc import ArcGISRESTful, ServiceURL
from pygeoogc import utils as ogc_utils
from pygeoutils.exceptions import EmptyResponseError
from pynhd.exceptions import (
    InputRangeError,
    InputValueError,
    MissingItemError,
    ServiceError,
    ZeroMatchedError,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pyproj import CRS
    from shapely import LineString, Point

    CRSType = int | str | CRS
    GeoType = (
        Polygon | MultiPolygon | MultiPoint | LineString | Point | tuple[float, float, float, float]
    )

__all__ = ["AGRBase", "GeoConnex", "ScienceBase"]


def get_parquet(parquet_path: Path | str) -> Path:
    """Get a parquet filename from a path or a string."""
    if Path(parquet_path).suffix != ".parquet":
        raise InputValueError("parquet_path", ["a filename with `.parquet` extension."])

    output = Path(parquet_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    return output


@dataclass(frozen=True)
class ServiceInfo:
    """Information about a web service."""

    url: str
    layer: str
    extent: tuple[float, float, float, float] | None
    feature_types: dict[int, str] | None


class AGRBase:
    """Base class for getting geospatial data from a ArcGISRESTful service.

    Parameters
    ----------
    base_url : str, optional
        The ArcGIS RESTful service url. The URL must either include a layer number
        after the last ``/`` in the url or the target layer must be passed as an argument.
    layer : str, optional
        A valid service layer. To see a list of available layers instantiate the class
        without passing any argument.
    outfields : str or list, optional
        Target field name(s), default to "*" i.e., all the fields.
    crs : str, int, or pyproj.CRS, optional
        Target spatial reference, default to ``EPSG:4326``
    outformat : str, optional
        One of the output formats offered by the selected layer. If not correct
        a list of available formats is shown, defaults to ``json``.
    """

    def __init__(
        self,
        base_url: str,
        layer: str | None = None,
        outfields: str | list[str] = "*",
        crs: CRSType = 4326,
        outformat: str = "json",
    ) -> None:
        if isinstance(layer, str):
            valid_layers = self.get_validlayers(base_url)
            try:
                self.client = ArcGISRESTful(
                    base_url,
                    valid_layers[layer.lower()],
                    outformat=outformat,
                    outfields=outfields,
                    crs=crs,
                )
            except KeyError as ex:
                raise InputValueError("layer", list(valid_layers)) from ex
        else:
            self.client = ArcGISRESTful(
                base_url,
                None,
                outformat=outformat,
                outfields=outfields,
                crs=crs,
            )

        full_layer = self.client.client.valid_layers[str(self.client.client.layer)]
        self._service_info = ServiceInfo(
            self.client.client.base_url,
            f"{full_layer} ({self.client.client.layer})",
            self.client.client.extent,
            self.client.client.feature_types,
        )

    @property
    def service_info(self) -> ServiceInfo:
        """Get the service information."""
        return self._service_info

    @staticmethod
    def get_validlayers(url: str) -> dict[str, int]:
        """Get a list of valid layers.

        Parameters
        ----------
        url : str
            The URL of the ArcGIS REST service.

        Returns
        -------
        dict
            A dictionary of valid layers.
        """
        rjson = ar.retrieve_json([url], [{"params": {"f": "json"}}])
        rjson = cast("list[dict[str, Any]]", rjson)
        if not rjson[0]["layers"]:
            ar.delete_url_cache(url, params={"f": "json"})
            with ogc_utils.RetrySession(disable=True) as session:
                rjson = session.get(url, params={"f": "json"}).json()
        if not rjson[0]["layers"]:
            msg = "The service doesn't have any layers."
            raise ServiceError(msg, url)
        return {lyr["name"].lower(): int(lyr["id"]) for lyr in rjson[0]["layers"]}

    def _getfeatures(
        self, oids: Iterator[tuple[str, ...]], return_m: bool = False, return_geom: bool = True
    ) -> gpd.GeoDataFrame:
        """Send a request for getting data based on object IDs.

        Parameters
        ----------
        return_m : bool
            Whether to activate the Return M (measure) in the request, defaults to False.
        return_geom : bool, optional
            Whether to return the geometry of the feature, defaults to ``True``.

        Returns
        -------
        geopandas.GeoDataFrame
            The requested features as a GeoDataFrame.
        """
        try:
            return geoutils.json2geodf(
                self.client.get_features(oids, return_m, return_geom), self.client.client.crs
            )
        except EmptyResponseError as ex:
            raise ZeroMatchedError from ex

    def bygeom(
        self,
        geom: LineString
        | Polygon
        | Point
        | MultiPoint
        | tuple[float, float]
        | list[tuple[float, float]]
        | tuple[float, float, float, float],
        geo_crs: CRSType = 4326,
        sql_clause: str = "",
        distance: int | None = None,
        return_m: bool = False,
        return_geom: bool = True,
    ) -> gpd.GeoDataFrame:
        """Get feature within a geometry that can be combined with a SQL where clause.

        Parameters
        ----------
        geom : Polygon or tuple
            A geometry (Polygon) or bounding box (tuple of length 4).
        geo_crs : str
            The spatial reference of the input geometry.
        sql_clause : str, optional
            A valid SQL 92 WHERE clause, defaults to an empty string.
        distance : int, optional
            The buffer distance for the input geometries in meters, default to None.
        return_m : bool, optional
            Whether to activate the Return M (measure) in the request, defaults to False.
        return_geom : bool, optional
            Whether to return the geometry of the feature, defaults to ``True``.

        Returns
        -------
        geopandas.GeoDataFrame
            The requested features as a GeoDataFrame.
        """
        oids = self.client.oids_bygeom(
            geom, geo_crs=geo_crs, sql_clause=sql_clause, distance=distance
        )
        return self._getfeatures(oids, return_m, return_geom)

    def byids(
        self,
        field: str,
        fids: str | list[str],
        return_m: bool = False,
        return_geom: bool = True,
    ) -> gpd.GeoDataFrame:
        """Get features based on a list of field IDs.

        Parameters
        ----------
        field : str
            Name of the target field that IDs belong to.
        fids : str or list
            A list of target field ID(s).
        return_m : bool
            Whether to activate the Return M (measure) in the request, defaults to False.
        return_geom : bool, optional
            Whether to return the geometry of the feature, defaults to ``True``.

        Returns
        -------
        geopandas.GeoDataFrame
            The requested features as a GeoDataFrame.
        """
        oids = self.client.oids_byfield(field, fids)
        return self._getfeatures(oids, return_m, return_geom)

    def bysql(
        self,
        sql_clause: str,
        return_m: bool = False,
        return_geom: bool = True,
    ) -> gpd.GeoDataFrame:
        """Get feature IDs using a valid SQL 92 WHERE clause.

        Notes
        -----
        Not all web services support this type of query. For more details look
        `here <https://developers.arcgis.com/rest/services-reference/query-feature-service-.htm#ESRI_SECTION2_07DD2C5127674F6A814CE6C07D39AD46>`__

        Parameters
        ----------
        sql_clause : str
            A valid SQL 92 WHERE clause.
        return_m : bool
            Whether to activate the measure in the request, defaults to False.
        return_geom : bool, optional
            Whether to return the geometry of the feature, defaults to ``True``.

        Returns
        -------
        geopandas.GeoDataFrame
            The requested features as a GeoDataFrame.
        """
        oids = self.client.oids_bysql(sql_clause)
        return self._getfeatures(oids, return_m, return_geom)

    def __repr__(self) -> str:
        """Print the service configuration."""
        return self.client.__repr__()


class ScienceBase:
    """Access and explore items on USGS's ScienceBase."""

    @staticmethod
    def get_children(item: str) -> dict[str, Any]:
        """Get children items of an item."""
        url = "https://www.sciencebase.gov/catalog/items"
        payload = {
            "filter": f"parentIdExcludingLinks={item}",
            "fields": "title,id",
            "format": "json",
        }
        resp = ar.retrieve_json([url], [{"params": payload}])
        resp = cast("list[dict[str, Any]]", resp)
        return resp[0]

    @staticmethod
    def get_file_urls(item: str) -> pd.DataFrame:
        """Get download and meta URLs of all the available files for an item."""
        url = "https://www.sciencebase.gov/catalog/item"
        payload = {"fields": "files,downloadUri", "format": "json"}
        resp = ar.retrieve_json([f"{url}/{item}"], [{"params": payload}])
        resp = cast("list[dict[str, Any]]", resp)
        urls = zip(
            tlz.pluck("name", resp[0]["files"]),
            tlz.pluck("url", resp[0]["files"]),
            tlz.pluck("metadataHtmlViewUri", resp[0]["files"], default=None),
        )
        return pd.DataFrame(
            urls,
            columns=["name", "url", "metadata_url"],  # pyright: ignore[reportArgumentType]
        ).set_index("name")


@dataclass(frozen=True)
class GCXURL:
    items: str
    queryables: str


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
        self.dev = dev
        self.item = item
        self.max_nfeatures = max_nfeatures

    @property
    def dev(self) -> bool:
        """Return the name of the endpoint."""
        return self._dev

    @dev.setter
    def dev(self, value: bool) -> None:
        self._dev = value
        if value:
            self.base_url = f"{ServiceURL().restful.geoconnex.replace('.us', '.dev')}/collections"
        else:
            self.base_url = f"{ServiceURL().restful.geoconnex}/collections"

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

    def _query(self, kwds: dict[str, Any], skip_geometry: bool) -> gpd.GeoDataFrame | pd.DataFrame:
        """Get response as a ``geopandas.GeoDataFrame``."""
        url_kwds = {
            "f": "json",
            "limit": self.max_nfeatures,
            "sortby": "uri",
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
