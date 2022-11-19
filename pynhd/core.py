"""Base classes for PyNHD functions."""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Iterator, NamedTuple, Union

import async_retriever as ar
import cytoolz as tlz
import geopandas as gpd
import pandas as pd
import pygeoutils as geoutils
import pyproj
import shapely.geometry as sgeom
from loguru import logger
from pygeoogc import ArcGISRESTful, ServiceURL
from pygeoogc import utils as ogc_utils

from .exceptions import (
    InputRangeError,
    InputTypeError,
    InputValueError,
    MissingColumnError,
    MissingCRSError,
    MissingItemError,
    ServiceError,
    ZeroMatchedError,
)

logger.configure(
    handlers=[
        {
            "sink": sys.stdout,
            "colorize": True,
            "format": " | ".join(
                [
                    "{time:YYYY-MM-DD at HH:mm:ss}",  # noqa: FS003
                    "{name: ^15}.{function: ^15}:{line: >3}",  # noqa: FS003
                    "{message}",  # noqa: FS003
                ]
            ),
        }
    ]
)
if os.environ.get("HYRIVER_VERBOSE", "false").lower() == "true":
    logger.enable("pynhd")
else:
    logger.disable("pynhd")

CRSTYPE = Union[int, str, pyproj.CRS]
__all__ = ["AGRBase", "ScienceBase", "stage_nhdplus_attrs", "GeoConnex"]


def get_parquet(parquet_path: Path | str) -> Path:
    """Get a parquet filename from a path or a string."""
    if Path(parquet_path).suffix != ".parquet":
        raise InputValueError("parquet_path", ["a filename with `.parquet` extension."])

    output = Path(parquet_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    return output


class GCXURL(NamedTuple):
    items: str
    queryables: str


class EndPoints(NamedTuple):
    name: str
    description: str
    url: str
    query_fields: list[str]
    extent: tuple[float, float, float, float]


class ServiceInfo(NamedTuple):
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
        crs: CRSTYPE = 4326,
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
        return geoutils.json2geodf(
            self.client.get_features(oids, return_m, return_geom), self.client.client.crs
        )

    def bygeom(
        self,
        geom: sgeom.Polygon | list[tuple[float, float]] | tuple[float, float, float, float],
        geo_crs: CRSTYPE = 4326,
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


class PyGeoAPIBase:
    """Access `PyGeoAPI <https://labs.waterdata.usgs.gov/api/nldi/pygeoapi>`__ service."""

    def __init__(self) -> None:
        self.base_url = ServiceURL().restful.pygeoapi
        self.req_idx: list[int | str] = [0]

    def _get_url(self, operation: str) -> str:
        """Set the service url."""
        return f"{self.base_url}/nldi-{operation}/execution"

    @staticmethod
    def _request_body(
        id_value: list[dict[str, Any]]
    ) -> list[dict[str, dict[str, list[dict[str, Any]]]]]:
        """Return a valid request body."""
        return [
            {
                "json": {
                    "inputs": [
                        {
                            "id": f"{i}",
                            "type": "text/plain",
                            "value": v if isinstance(v, list) else f"{v}",
                        }
                        for i, v in iv.items()
                    ]
                }
            }
            for iv in id_value
        ]

    def _get_response(
        self, url: str, payload: list[dict[str, dict[str, list[dict[str, Any]]]]]
    ) -> gpd.GeoDataFrame:
        """Post the request and return the response as a GeoDataFrame."""
        resp = ar.retrieve_json([url] * len(payload), payload, "POST")
        nfeat = len(resp)
        idx, resp = zip(*[(i, r) for i, r in enumerate(resp) if "code" not in r])

        if len(resp) == 0:
            msg = "Invalid inpute parameters, check them and retry."
            raise ServiceError(msg)

        if len(resp) < nfeat:
            logger.warning(
                " ".join(
                    [
                        f"There was {nfeat - len(resp)} requests that",
                        "didn't return any features. Check their parameters and retry.",
                    ]
                )
            )

        gdf = gpd.GeoDataFrame(
            pd.concat((geoutils.json2geodf(r) for r in resp), keys=[self.req_idx[i] for i in idx]),
            crs=4326,
        )
        drop_cols = ["level_1", "spatial_ref"] if "spatial_ref" in gdf else ["level_1"]
        return gdf.reset_index().rename(columns={"level_0": "req_idx"}).drop(columns=drop_cols)

    @staticmethod
    def _check_coords(
        coords: tuple[float, float] | list[tuple[float, float]],
        crs: CRSTYPE,
    ) -> list[tuple[float, float]]:
        """Check the coordinates."""
        _coords = [coords] if isinstance(coords, tuple) else coords

        if not isinstance(_coords, list) or any(len(c) != 2 for c in _coords):
            raise InputTypeError("coords", "tuple or list", "(lon, lat) or [(lon, lat), ...]")

        return ogc_utils.match_crs(_coords, crs, 4326)


class PyGeoAPIBatch(PyGeoAPIBase):
    """Access `PyGeoAPI <https://labs.waterdata.usgs.gov/api/nldi/pygeoapi>`__ service.

    Parameters
    ----------
    coords : geopandas.GeoDataFrame
        A GeoDataFrame containing the coordinates to query. The indices of the
        GeoDataFrame are used as the request IDs that will be returned in the
        response in a column named ``req_idx``.
        The required columns for using the class methods are:

        * ``flow_trace``: ``direction`` that indicates the direction of the flow trace.
          It can be ``up``, ``down``, or ``none``.
        * ``split_catchment``: ``upstream`` that indicates whether to return all upstream
          catchments or just the local catchment.
        * ``elevation_profile``: ``numpts`` that indicates the number of points to extract
          along the flowpath and ``dem_res`` that indicates the target resolution for
          requesting the DEM from 3DEP service.
        * ``cross_section``: ``numpts`` that indicates the number of points to extract
          along the flowpath and ``width`` that indicates the width of the cross-section
          in meters.
    """

    def __init__(self, coords: gpd.GeoDataFrame) -> None:
        super().__init__()
        if coords.crs is None:
            raise MissingCRSError

        self.coords = coords.to_crs(4326)
        self.req_idx = self.coords.index.tolist()
        self.req_cols = {
            "flow_trace": ["direction"],
            "split_catchment": ["upstream"],
            "elevation_profile": ["numpts", "dem_res"],
            "cross_section": ["numpts", "width"],
        }
        self.geo_types = {
            "flow_trace": "Point",
            "split_catchment": "Point",
            "elevation_profile": "MultiPoint",
            "cross_section": "Point",
        }
        self.service = {
            "flow_trace": "flowtrace",
            "split_catchment": "splitcatchment",
            "elevation_profile": "xsatendpts",
            "cross_section": "xsatpoint",
        }

    def check_col(self, method: str) -> None:
        """Check if the required columns are present in the GeoDataFrame."""
        missing = [c for c in self.req_cols[method] if c not in self.coords]
        if missing:
            raise MissingColumnError(missing)

    def check_geotype(self, method: str) -> None:
        """Check if the required geometry type is present in the GeoDataFrame."""
        if any(self.coords.geom_type != self.geo_types[method]):
            raise InputTypeError("coords", self.geo_types[method])

    def get_payload(self, method: str) -> list[dict[str, dict[str, list[dict[str, Any]]]]]:
        """Return the payload for a request."""
        self.check_col(method)
        self.check_geotype(method)

        attrs = self.req_cols[method]

        if "dem_res" in self.coords:
            coords = self.coords.rename(columns={"dem_res": "3dep_res"})
            attrs = ["numpts", "3dep_res"]
        else:
            coords = self.coords

        geo_iter = coords[["geometry"] + attrs].itertuples(index=False, name=None)

        if method == "elevation_profile":
            if any(len(g.geoms) != 2 for g in coords.geometry):
                raise InputTypeError("coords", "MultiPoint of length 2")

            return self._request_body(
                [
                    {
                        "lat": [g.y for g in mp.geoms],
                        "lon": [g.x for g in mp.geoms],
                        **dict(zip(attrs, list(u))),
                    }
                    for mp, *u in geo_iter
                ]
            )

        return self._request_body(
            [{"lat": g.y, "lon": g.x, **dict(zip(attrs, list(u)))} for g, *u in geo_iter]
        )


class ScienceBase:
    """Access and explore files on ScienceBase."""

    @staticmethod
    def get_children(item: str) -> dict[str, Any]:
        """Get children items of an item."""
        url = "https://www.sciencebase.gov/catalog/items"
        payload = {
            "filter": f"parentIdExcludingLinks={item}",
            "fields": "title,id",
            "format": "json",
        }
        resp = ar.retrieve_json(
            [url],
            [{"params": payload}],
        )
        return resp[0]

    @staticmethod
    def get_file_urls(item: str) -> pd.DataFrame:
        """Get download and meta URLs of all the available files for an item."""
        url = "https://www.sciencebase.gov/catalog/item"
        payload = {"fields": "files,downloadUri", "format": "json"}
        resp = ar.retrieve_json(
            [f"{url}/{item}"],
            [{"params": payload}],
        )
        urls = zip(
            tlz.pluck("name", resp[0]["files"]),
            tlz.pluck("url", resp[0]["files"]),
            tlz.pluck("metadataHtmlViewUri", resp[0]["files"], default=None),
        )
        files = pd.DataFrame(urls, columns=["name", "url", "metadata_url"])
        return files.set_index("name")


def stage_nhdplus_attrs(
    parquet_path: Path | str | None = None,
) -> pd.DataFrame:
    """Stage the NHDPlus Attributes database and save to nhdplus_attrs.parquet.

    More info can be found `here <https://www.sciencebase.gov/catalog/item/5669a79ee4b08895842a1d47>`_.

    Parameters
    ----------
    parquet_path : str or Path
        Path to a file with ``.parquet`` extension for saving the processed to disk for
        later use.

    Returns
    -------
    pandas.DataFrame
        The staged data as a DataFrame.
    """
    if parquet_path is None:
        output = get_parquet(Path("cache", "nhdplus_attrs.parquet"))
    else:
        output = get_parquet(parquet_path)

    sb = ScienceBase()
    r = sb.get_children("5669a79ee4b08895842a1d47")

    titles = tlz.pluck("title", r["items"])
    titles = tlz.concat(tlz.map(tlz.partial(re.findall, "Select(.*?)Attributes"), titles))
    titles = tlz.map(str.strip, titles)

    main_items = dict(zip(titles, tlz.pluck("id", r["items"])))

    def get_files(item: str) -> dict[str, tuple[str, str]]:
        """Get all the available zip files in an item."""
        url = "https://www.sciencebase.gov/catalog/item"
        payload = {"fields": "files,downloadUri", "format": "json"}
        resp = ar.retrieve_json(
            [f"{url}/{item}"],
            [{"params": payload}],
        )
        files_url = zip(tlz.pluck("name", resp[0]["files"]), tlz.pluck("url", resp[0]["files"]))
        meta = list(tlz.pluck("metadataHtmlViewUri", resp[0]["files"], default=""))[-1]
        return {f.replace("_CONUS.zip", ""): (u, meta) for f, u in files_url if ".zip" in f}

    files = {}
    soil = main_items.pop("Soil")
    for i, item in main_items.items():
        r = sb.get_children(item)

        titles = tlz.pluck("title", r["items"])
        titles = tlz.map(lambda s: s.split(":")[1].strip() if ":" in s else s, titles)

        child_items = dict(zip(titles, tlz.pluck("id", r["items"])))
        files[i] = {t: get_files(c) for t, c in child_items.items()}

    r = sb.get_children(soil)
    titles = tlz.pluck("title", r["items"])
    titles = tlz.map(lambda s: s.split(":")[1].strip() if ":" in s else s, titles)

    child_items = dict(zip(titles, tlz.pluck("id", r["items"])))
    stat = child_items.pop("STATSGO Soil Characteristics")
    ssur = child_items.pop("SSURGO Soil Characteristics")
    files["Soil"] = {t: get_files(c) for t, c in child_items.items()}

    r = sb.get_children(stat)
    titles = tlz.pluck("title", r["items"])
    titles = tlz.map(lambda s: s.split(":")[1].split(",")[1].strip(), titles)
    child_items = dict(zip(titles, tlz.pluck("id", r["items"])))
    files["STATSGO"] = {t: get_files(c) for t, c in child_items.items()}

    r = sb.get_children(ssur)
    titles = tlz.pluck("title", r["items"])
    titles = tlz.map(lambda s: s.split(":")[1].strip(), titles)
    child_items = dict(zip(titles, tlz.pluck("id", r["items"])))
    files["SSURGO"] = {t: get_files(c) for t, c in child_items.items()}

    chars = []
    types = {"CAT": "local", "TOT": "upstream_acc", "ACC": "div_routing"}
    for t, dd in files.items():
        for d, fd in dd.items():
            for f, u in fd.items():
                chars.append(
                    {
                        "name": f,
                        "type": types.get(f[-3:], "other"),
                        "theme": t,
                        "description": d,
                        "url": u[0],
                        "meta": u[1],
                    }
                )
    char_df = pd.DataFrame(chars)
    char_df.to_parquet(output)
    return char_df


class GeoConnex:
    """Access to the GeoConnex API.

    Notes
    -----
    The ``geometry`` field of the query can be a Polygon, MultiPolygon,
    or tuple/list of length 4 (bbox) in ``EPSG:4326`` CRS. They should
    be within the extent of the GeoConnex endpoint.

    Parameters
    ----------
    item : str, optional
        The target endpoint to query, defaults to ``None``.
    """

    @staticmethod
    def __get_url(url: str, kwds: dict[str, str] | None = None) -> dict[str, Any]:
        params = {"params": {**kwds, "f": "json"}} if kwds else {"params": {"f": "json"}}
        return ar.retrieve_json([url], [params])[0]

    def __get_endpoints(self) -> dict[str, EndPoints]:
        pluck = tlz.partial(tlz.pluck, seqs=self.__get_url(self.base_url)["collections"])

        def get_links(links: list[dict[str, Any]]) -> dict[str, str | list[str]]:
            """Get links."""
            urls = {
                lk["rel"]: lk["href"].replace("?f=json", "")
                for lk in links
                if lk["type"] == "application/json"
            }
            fields = list(self.__get_url(urls["queryables"])["properties"])
            if "geom" in fields:
                fields.remove("geom")
                if "geometry" not in fields:
                    fields.append("geometry")

            return {
                "url": f"{urls['self']}/items",
                "query_fields": fields,
            }

        eps = zip(
            pluck(ind="id"), pluck(ind="description"), pluck(ind="links"), pluck(ind="extent")
        )
        return {
            ep[0]: EndPoints(
                name=ep[0],
                description=ep[1],
                **get_links(ep[2]),  # type: ignore
                extent=tuple(ep[3]["spatial"]["bbox"][0]),  # type: ignore
            )
            for ep in eps
        }

    def __init__(self, item: str | None = None) -> None:
        self.base_url = f"{ServiceURL().restful.geoconnex}/collections"
        self.endpoints = self.__get_endpoints()
        self.query_url: str | None = None
        self.item = item

    @staticmethod
    def __get_urls(url: str, kwds: list[dict[str, str]]) -> list[dict[str, Any]]:
        params = [{"params": {**kwd, "f": "json"}} for kwd in kwds]
        return ar.retrieve_json([url] * len(kwds), params)

    @property
    def item(self) -> str | None:
        """Return the name of the endpoint."""
        return self._item

    @item.setter
    def item(self, value: str | None) -> None:
        self._item = value
        if value is not None:
            if value not in self.endpoints:
                raise InputValueError("item", list(self.endpoints))
            self.query_url = self.endpoints[value].url
        else:
            self.query_url = None

    def query(
        self,
        kwds: dict[
            str,
            (
                str
                | int
                | float
                | tuple[float, float, float, float]
                | sgeom.Polygon
                | sgeom.MultiPolygon
            ),
        ],
        skip_geometry: bool = False,
    ) -> gpd.GeoDataFrame:
        """Query the GeoConnex endpoint."""
        if self.query_url is None or self.item is None:
            raise MissingItemError(["item"])

        valid_keys = self.endpoints[self.item].query_fields
        invalid_key = [k for k in kwds if k not in valid_keys]
        if invalid_key:
            keys = ", ".join(invalid_key)
            raise InputValueError(f"query: {keys}", valid_keys)

        if skip_geometry:
            kwds["skip_geometry"] = "true"

        if "geometry" in kwds:
            geometry = geoutils.geometry_list(kwds["geometry"])

            extent = self.endpoints[self.item].extent
            if not all(g.within(sgeom.box(*extent)) for g in geometry):
                raise InputRangeError("geometry", f"within {extent}")

            _ = kwds.pop("geometry")
            param_list = [
                {**kwds, "bbox": ",".join(f"{c:.6f}" for c in g.bounds)} for g in geometry  # type: ignore[arg-type]
            ]
            gdf = geoutils.json2geodf(self.__get_urls(self.query_url, param_list))
            gdf = gdf.reset_index(drop=True)
            _, idx = gdf.sindex.query_bulk(gpd.GeoSeries(geometry, crs=4326), predicate="contains")
            if len(idx) == 0:
                raise ZeroMatchedError
            gdf = gdf.iloc[idx].reset_index(drop=True)
        else:
            gdf = geoutils.json2geodf(self.__get_url(self.query_url, kwds))  # type: ignore[arg-type]
            if len(gdf) == 0:
                raise ZeroMatchedError

        if "nhdpv2_COMID" in gdf:
            gdf["nhdpv2_COMID"] = gdf["nhdpv2_COMID"].astype("Int64")

        return gdf

    def __repr__(self) -> str:
        if self.item is None:
            return "\n".join(
                [
                    "Available Endpoints:",
                    "\n".join(f"    '{k}': {v.description}" for k, v in self.endpoints.items()),
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
