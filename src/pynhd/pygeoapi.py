"""Access NLDI and WaterData databases."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal, cast

import cytoolz.curried as tlz
import geopandas as gpd
import pandas as pd
from shapely import LineString, MultiLineString, MultiPoint

import async_retriever as ar
import pygeoutils as geoutils
from pygeoogc import ServiceURL
from pygeoutils.exceptions import EmptyResponseError
from pynhd.exceptions import (
    InputTypeError,
    MissingColumnError,
    MissingCRSError,
    ServiceError,
    ZeroMatchedError,
)

if TYPE_CHECKING:
    from pyproj import CRS
    from shapely import MultiPolygon, Point, Polygon

    CRSType = int | str | CRS
    GeoType = (
        Polygon | MultiPolygon | MultiPoint | LineString | Point | tuple[float, float, float, float]
    )

__all__ = ["PyGeoAPI", "pygeoapi"]


class PyGeoAPIBase:
    """Access `PyGeoAPI <https://api.water.usgs.gov/api/nldi/pygeoapi>`__ service."""

    def __init__(self) -> None:
        self.base_url = ServiceURL().restful.pygeoapi
        self.req_idx: list[int | str] = [0]

    def get_url(self, operation: str) -> str:
        """Set the service url."""
        return f"{self.base_url}/nldi-{operation}/execution"

    @staticmethod
    def request_body(
        id_value: list[dict[str, Any]],
    ) -> list[dict[str, dict[str, list[dict[str, Any]]]]]:
        """Return a valid request body."""
        return [
            {
                "json": {
                    "inputs": [
                        {
                            "id": i,
                            "type": "text/plain",
                            "value": list(v) if isinstance(v, (list, tuple)) else str(v),
                        }
                        for i, v in iv.items()
                    ]
                }
            }
            for iv in id_value
        ]

    def get_response(
        self, url: str, payload: list[dict[str, dict[str, list[dict[str, Any]]]]]
    ) -> gpd.GeoDataFrame:
        """Post the request and return the response as a GeoDataFrame."""
        resp = ar.retrieve_json([url] * len(payload), payload, "post")
        resp = cast("list[dict[str, Any]]", resp)
        nfeat = len(resp)
        try:
            idx, resp = zip(*((i, r) for i, r in enumerate(resp) if "code" not in r))
            idx = cast("tuple[int]", idx)
            if len(idx) != nfeat:
                _ = [
                    ar.delete_url_cache(url, **p)  # pyright: ignore[reportArgumentType]
                    for i, p in enumerate(payload)
                    if i not in idx
                ]
            resp = cast("tuple[dict[str, Any]]", resp)

            if len(resp) < nfeat:
                warnings.warn(
                    " ".join(
                        [
                            f"There are {nfeat - len(resp)} requests that",
                            "didn't return any feature. Check their parameters and retry.",
                        ]
                    ),
                    UserWarning,
                    stacklevel=2,
                )
            gdf = gpd.GeoDataFrame(  # pyright: ignore[reportCallIssue]
                pd.concat(
                    (geoutils.json2geodf(r) for r in resp), keys=[self.req_idx[i] for i in idx]
                ),
                crs=4326,
            )
        except ValueError as ex:
            msg = "Invalid inpute parameters, check them and retry."
            raise ServiceError(msg) from ex
        except EmptyResponseError as ex:
            raise ZeroMatchedError from ex
        drop_cols = ["level_1", "spatial_ref"] if "spatial_ref" in gdf else ["level_1"]
        return (  # pyright: ignore[reportReturnType]
            gdf.reset_index().rename(columns={"level_0": "req_idx"}).drop(columns=drop_cols)
        )

    @staticmethod
    def check_coords(
        coords: tuple[float, float] | list[tuple[float, float]],
        crs: CRSType,
    ) -> list[tuple[float, float]]:
        """Check the coordinates."""
        try:
            mps = MultiPoint(coords)  # pyright: ignore[reportArgumentType]
        except (TypeError, AttributeError):
            try:
                mps = MultiPoint([coords])
            except ValueError as ex:
                raise InputTypeError(
                    "coords", "tuple or list of them", "(x, y) or [(x, y), ...]"
                ) from ex
        except ValueError as ex:
            raise InputTypeError(
                "coords", "tuple or list of them", "(x, y) or [(x, y), ...]"
            ) from ex

        _coords = [(p.x, p.y) for p in mps.geoms]
        _coords = [
            (round(x, 6), round(y, 6)) for x, y in geoutils.geometry_reproject(_coords, crs, 4326)
        ]
        return _coords


class PyGeoAPIBatch(PyGeoAPIBase):
    """Access `PyGeoAPI <https://api.water.usgs.gov/api/nldi/pygeoapi>`__ service.

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
            "endpoints_profile": ["numpts", "dem_res"],
            "cross_section": ["numpts", "width"],
        }
        self.geo_types = {
            "flow_trace": "Point",
            "split_catchment": "Point",
            "elevation_profile": "LineString",
            "endpoints_profile": "MultiPoint",
            "cross_section": "Point",
        }
        self.service = {
            "flow_trace": "flowtrace",
            "split_catchment": "splitcatchment",
            "elevation_profile": "xsatpathpts",
            "endpoints_profile": "xsatendpts",
            "cross_section": "xsatpoint",
        }

    def check_col(self, method: str) -> None:
        """Check if the required columns are present in the GeoDataFrame."""
        missing = [c for c in self.req_cols[method] if c not in self.coords]
        if missing:
            raise MissingColumnError(missing)

    def check_geotype(self, method: str) -> None:
        """Check if the required geometry type is present in the GeoDataFrame."""
        if not self.coords.geom_type.eq(self.geo_types[method]).all():
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

        geo_iter = coords[["geometry", *attrs]].itertuples(index=False, name=None)

        if method == "endpoints_profile":
            if any(len(g.geoms) != 2 for g in coords.geometry):
                raise InputTypeError("coords", "MultiPoint of length 2")

            return self.request_body(
                [
                    {
                        "lat": [round(g.y, 6) for g in mp.geoms],
                        "lon": [round(g.x, 6) for g in mp.geoms],
                    }
                    | dict(zip(attrs, list(u)))
                    for mp, *u in geo_iter
                ]
            )

        if method == "elevation_profile":
            return self.request_body(
                [
                    {
                        "path": [[round(x, 6), round(y, 6)] for x, y in line.coords],
                    }
                    | dict(zip(attrs, list(u)))
                    for line, *u in geo_iter
                ]
            )

        return self.request_body(
            [
                {"lat": round(g.y, 6), "lon": round(g.x, 6)} | dict(zip(attrs, list(u)))
                for g, *u in geo_iter
            ]
        )


class PyGeoAPI(PyGeoAPIBase):
    """Access `PyGeoAPI <https://api.water.usgs.gov/api/nldi/pygeoapi>`__ service."""

    def flow_trace(
        self,
        coord: tuple[float, float],
        crs: CRSType = 4326,
        direction: Literal["down", "up", "none"] = "none",
    ) -> gpd.GeoDataFrame:
        """Return a GeoDataFrame from the flowtrace service.

        Parameters
        ----------
        coord : tuple
            The coordinate of the point to trace as a tuple,e.g., (lon, lat).
        crs : str
            The coordinate reference system of the coordinates, defaults to EPSG:4326.
        direction : str, optional
            The direction of flowpaths, either ``down``, ``up``, or ``none``.
            Defaults to ``none``.

        Returns
        -------
        geopandas.GeoDataFrame
            A GeoDataFrame containing the traced flowline.

        Examples
        --------
        >>> from pynhd import PyGeoAPI
        >>> pga = PyGeoAPI()
        >>> gdf = pga.flow_trace(
        ...     (1774209.63, 856381.68), crs="ESRI:102003", direction="none"
        ... )  # doctest: +SKIP
        >>> print(gdf.comid.iloc[0])  # doctest: +SKIP
        22294818
        """
        lon, lat = self.check_coords(coord, crs)[0]
        url = self.get_url("flowtrace")
        payload = self.request_body([{"lat": lat, "lon": lon, "direction": direction}])
        return self.get_response(url, payload)

    def split_catchment(
        self, coord: tuple[float, float], crs: CRSType = 4326, upstream: bool = False
    ) -> gpd.GeoDataFrame:
        """Return a GeoDataFrame from the splitcatchment service.

        Parameters
        ----------
        coord : tuple
            The coordinate of the point to trace as a tuple,e.g., (lon, lat).
        crs : str, int, or pyproj.CRS, optional
            The coordinate reference system of the coordinates, defaults to EPSG:4326.
        upstream : bool, optional
            If True, return all upstream catchments rather than just the local catchment,
            defaults to False.

        Returns
        -------
        geopandas.GeoDataFrame
            A GeoDataFrame containing the local catchment or the entire upstream catchments.

        Examples
        --------
        >>> from pynhd import PyGeoAPI
        >>> pga = PyGeoAPI()
        >>> gdf = pga.split_catchment((-73.82705, 43.29139), crs=4326, upstream=False)  # doctest: +SKIP
        >>> print(gdf.catchmentID.iloc[0])  # doctest: +SKIP
        22294818
        """
        lon, lat = self.check_coords(coord, crs)[0]
        url = self.get_url("splitcatchment")
        payload = self.request_body([{"lat": lat, "lon": lon, "upstream": upstream}])
        return self.get_response(url, payload)

    def elevation_profile(
        self,
        line: LineString | MultiLineString,
        numpts: int,
        dem_res: int,
        crs: CRSType = 4326,
    ) -> gpd.GeoDataFrame:
        """Return a GeoDataFrame from the xsatpathpts service.

        Parameters
        ----------
        line : shapely.LineString or shapely.MultiLineString
            The line to extract the elevation profile for.
        numpts : int
            The number of points to extract the elevation profile from the DEM.
        dem_res : int
            The target resolution for requesting the DEM from 3DEP service.
        crs : str, int, or pyproj.CRS, optional
            The coordinate reference system of the coordinates, defaults to EPSG:4326.

        Returns
        -------
        geopandas.GeoDataFrame
            A GeoDataFrame containing the elevation profile along the requested endpoints.

        Examples
        --------
        >>> from pynhd import PyGeoAPI
        >>> from shapely import LineString
        >>> pga = PyGeoAPI()
        >>> line = LineString([(-103.801086, 40.26772), (-103.80097, 40.270568)])
        >>> gdf = pga.elevation_profile(line, 101, 1, 4326)  # doctest: +SKIP
        >>> print(gdf.iloc[-1, 2])  # doctest: +SKIP
        1299.8727
        """
        if not isinstance(line, (LineString, MultiLineString)):
            raise InputTypeError("line", "LineString or MultiLineString")
        line = geoutils.geometry_reproject(line, crs, 4326)  # pyright: ignore[reportArgumentType]
        if isinstance(line, LineString):
            coords = line.coords
        else:
            coords = list(tlz.concat(c.coords for c in line.geoms))

        url = self.get_url("xsatpathpts")
        payload = self.request_body(
            [
                {
                    "path": [[round(x, 6), round(y, 6)] for x, y in coords],
                    "numpts": numpts,
                    "3dep_res": dem_res,
                }
            ]
        )
        return self.get_response(url, payload)

    def endpoints_profile(
        self,
        coords: list[tuple[float, float]],
        numpts: int,
        dem_res: int,
        crs: CRSType = 4326,
    ) -> gpd.GeoDataFrame:
        """Return a GeoDataFrame from the xsatendpts service.

        Parameters
        ----------
        coords : list
            A list of two coordinates to trace as a list of tuples, e.g.,
            [(x1, y1), (x2, y2)].
        numpts : int
            The number of points to extract the elevation profile from the DEM.
        dem_res : int
            The target resolution for requesting the DEM from 3DEP service.
        crs : str, int, or pyproj.CRS, optional
            The coordinate reference system of the coordinates, defaults to EPSG:4326.

        Returns
        -------
        geopandas.GeoDataFrame
            A GeoDataFrame containing the elevation profile along the requested endpoints.

        Examples
        --------
        >>> from pynhd import PyGeoAPI
        >>> pga = PyGeoAPI()
        >>> gdf = pga.endpoints_profile(
        ...     [(-103.801086, 40.26772), (-103.80097, 40.270568)], numpts=101, dem_res=1, crs=4326
        ... )  # doctest: +SKIP
        >>> print(gdf.iloc[-1, 1])  # doctest: +SKIP
        411.5906
        """
        lons, lats = zip(*self.check_coords(coords, crs))

        url = self.get_url("xsatendpts")
        payload = self.request_body(
            [{"lat": lats, "lon": lons, "numpts": numpts, "3dep_res": dem_res}]
        )
        return self.get_response(url, payload)

    def cross_section(
        self, coord: tuple[float, float], width: float, numpts: int, crs: CRSType = 4326
    ) -> gpd.GeoDataFrame:
        """Return a GeoDataFrame from the xsatpoint service.

        Parameters
        ----------
        coord : tuple
            The coordinate of the point to extract the cross-section as a tuple,e.g., (lon, lat).
        width : float
            The width of the cross-section in meters.
        numpts : int
            The number of points to extract the cross-section from the DEM.
        crs : str, int, or pyproj.CRS, optional
            The coordinate reference system of the coordinates, defaults to EPSG:4326.

        Returns
        -------
        geopandas.GeoDataFrame
            A GeoDataFrame containing the cross-section at the requested point.

        Examples
        --------
        >>> from pynhd import PyGeoAPI
        >>> pga = PyGeoAPI()
        >>> gdf = pga.cross_section((-103.80119, 40.2684), width=1000.0, numpts=101, crs=4326)  # doctest: +SKIP
        >>> print(gdf.iloc[-1, 1])  # doctest: +SKIP
        1000.0
        """
        lon, lat = self.check_coords(coord, crs)[0]
        url = self.get_url("xsatpoint")
        payload = self.request_body([{"lat": lat, "lon": lon, "width": width, "numpts": numpts}])
        return self.get_response(url, payload)


def pygeoapi(
    geodf: gpd.GeoDataFrame,
    service: Literal[
        "flow_trace", "split_catchment", "elevation_profile", "endpoints_profile", "cross_section"
    ],
) -> gpd.GeoDataFrame:
    """Return a GeoDataFrame from the flowtrace service.

    Parameters
    ----------
    geodf : geopandas.GeoDataFrame
        A GeoDataFrame containing geometries to query.
        The required columns for each service are:

        * ``flow_trace``: ``direction`` that indicates the direction of the flow trace.
          It can be ``up``, ``down``, or ``none`` (both directions).
        * ``split_catchment``: ``upstream`` that indicates whether to return all upstream
          catchments or just the local catchment.
        * ``elevation_profile``: ``numpts`` that indicates the number of points to extract
          along the flowpath and ``3dep_res`` that indicates the target resolution for
          requesting the DEM from 3DEP service.
        * ``endpoints_profile``: ``numpts`` that indicates the number of points to extract
          along the flowpath and ``3dep_res`` that indicates the target resolution for
          requesting the DEM from 3DEP service.
        * ``cross_section``: ``numpts`` that indicates the number of points to extract
          along the flowpath and ``width`` that indicates the width of the cross-section
          in meters.

    service : str
        The service to query, can be ``flow_trace``, ``split_catchment``, ``elevation_profile``,
        ``endpoints_profile``, or ``cross_section``.

    Returns
    -------
    geopandas.GeoDataFrame
        A GeoDataFrame containing the results of requested service.

    Examples
    --------
    >>> from shapely import Point
    >>> import geopandas as gpd
    >>> gdf = gpd.GeoDataFrame(
    ...     {
    ...         "direction": [
    ...             "none",
    ...         ]
    ...     },
    ...     geometry=[Point((1774209.63, 856381.68))],
    ...     crs="ESRI:102003",
    ... )
    >>> trace = nhd.pygeoapi(gdf, "flow_trace")
    >>> print(trace.comid.iloc[0])
    22294818
    """
    pgab = PyGeoAPIBatch(geodf)
    url = pgab.get_url(pgab.service[service])
    payload = pgab.get_payload(service)
    gdf = pgab.get_response(url, payload)
    if service == "flow_trace":
        gdf["comid"] = gdf["comid"].astype("Int64")
        feat = gdf[~gdf.comid.isna()].set_index("req_idx")
        raindrop = gdf.loc[gdf.comid.isna(), ["req_idx", "geometry"]].set_index("req_idx")
        feat["raindrop_path"] = raindrop.geometry
        return feat.reset_index()
    return gdf
