"""Access NLDI and WaterData databases."""

# pyright: reportGeneralTypeIssues=false
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Generator, Literal, Sequence, Union, cast, overload

import cytoolz.curried as tlz
import numpy as np
import pandas as pd
from shapely import LineString, MultiLineString
from yarl import URL

import async_retriever as ar
import pygeoogc as ogc
import pygeoutils as geoutils
from pygeoogc import WFS, InputValueError, ServiceURL
from pygeoutils import EmptyResponseError, InputTypeError
from pynhd.core import AGRBase, PyGeoAPIBase, PyGeoAPIBatch
from pynhd.exceptions import InputRangeError, MissingItemError, ZeroMatchedError

if TYPE_CHECKING:
    import geopandas as gpd
    import pyproj
    from shapely import MultiPoint, MultiPolygon, Point, Polygon

    CRSTYPE = Union[int, str, pyproj.CRS]
    GTYPE = Union[
        Polygon, MultiPolygon, MultiPoint, LineString, Point, "tuple[float, float, float, float]"
    ]
    NHD_LAYERS = Literal[
        "point",
        "point_event",
        "line_hr",
        "flow_direction",
        "flowline_mr",
        "flowline_hr_nonconus",
        "flowline_hr",
        "area_mr",
        "area_hr_nonconus",
        "area_hr",
        "waterbody_mr",
        "waterbody_hr_nonconus",
        "waterbody_hr",
    ]
    WD_LAYERS = Literal[
        "catchmentsp",
        "gagesii",
        "gagesii_basins",
        "huc08",
        "huc12",
        "nhdarea",
        "nhdflowline_network",
        "nhdflowline_nonnetwork",
        "nhdwaterbody",
        "wbd02",
        "wbd04",
        "wbd06",
        "wbd08",
        "wbd10",
        "wbd12",
    ]
    PREDICATES = Literal[
        "equals",
        "disjoint",
        "intersects",
        "touches",
        "crosses",
        "within",
        "contains",
        "overlaps",
        "relate",
        "beyond",
    ]
    NHDHR_LAYERS = Literal[
        "gauge",
        "sink",
        "point",
        "flowline",
        "non_network_flowline",
        "flow_direction",
        "wall",
        "line",
        "area",
        "waterbody",
        "catchment",
        "boundary_unit",
        "huc12",
    ]
    HP3D_LAYERS = Literal[
        "hydrolocation_waterbody",
        "hydrolocation_flowline",
        "hydrolocation_reach",
        "flowline",
        "waterbody",
        "drainage_area",
        "catchment",
    ]

__all__ = ["NHD", "HP3D", "PyGeoAPI", "pygeoapi", "WaterData", "NHDPlusHR", "NLDI"]


class NHD(AGRBase):
    """Access National Hydrography Dataset (NHD), both meduim and high resolution.

    Notes
    -----
    For more info visit: https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer

    Parameters
    ----------
    layer : str, optional
        A valid service layer. Layer names with ``_hr`` are high resolution and
        ``_mr`` are medium resolution. Also, layer names with ``_nonconus`` are for
        non-conus areas, i.e., Alaska, Hawaii, Puerto Rico, the Virgin Islands , and
        the Pacific Islands. Valid layers are:

        - ``point``
        - ``point_event``
        - ``line_hr``
        - ``flow_direction``
        - ``flowline_mr``
        - ``flowline_hr_nonconus``
        - ``flowline_hr``
        - ``area_mr``
        - ``area_hr_nonconus``
        - ``area_hr``
        - ``waterbody_mr``
        - ``waterbody_hr_nonconus``
        - ``waterbody_hr``

    outfields : str or list, optional
        Target field name(s), default to "*" i.e., all the fields.
    crs : str, int, or pyproj.CRS, optional
        Target spatial reference, default to ``EPSG:4326``.

    Methods
    -------
    bygeom(geom, geo_crs=4326, sql_clause="", distance=None, return_m=False, return_geom=True)
        Get features within a geometry that can be combined with a SQL where clause.
    byids(field, fids, return_m=False, return_geom=True)
        Get features by object IDs.
    bysql(sql_clause, return_m=False, return_geom=True)
        Get features using a valid SQL 92 WHERE clause.
    """

    def __init__(
        self,
        layer: NHD_LAYERS,
        outfields: str | list[str] = "*",
        crs: CRSTYPE = 4326,
    ):
        self.valid_layers = {
            "point": "point",
            "point_event": "point event",
            "line_hr": "line - large scale ",
            "flow_direction": "flow direction",
            "flowline_mr": "flowline - small scale",
            "flowline_hr_nonconus": "flowline - small scale (hi, pr, vi, pacific territories)",
            "flowline_hr": "flowline - large scale",
            "area_mr": "area - small scale",
            "area_hr_nonconus": "area - small scale (hi, pr, vi, pacific territories)",
            "area_hr": "area - large scale",
            "waterbody_mr": "waterbody - small scale",
            "waterbody_hr_nonconus": "waterbody - small scale (hi, pr, vi, pacific territories)",
            "waterbody_hr": "waterbody - large scale",
        }
        _layer = self.valid_layers.get(layer)
        if _layer is None:
            raise InputValueError("layer", list(self.valid_layers))
        super().__init__(
            ServiceURL().restful.nhd,
            _layer,
            outfields,
            crs,
        )


class HP3D(AGRBase):
    """Access USGS 3D Hydrography Program (3DHP) service.

    Notes
    -----
    For more info visit: https://hydro.nationalmap.gov/arcgis/rest/services/3DHP_all/MapServer

    Parameters
    ----------
    layer : str, optional
        A valid service layer. Layer names with ``_hr`` are high resolution and
        ``_mr`` are medium resolution. Also, layer names with ``_nonconus`` are for
        non-conus areas, i.e., Alaska, Hawaii, Puerto Rico, the Virgin Islands , and
        the Pacific Islands. Valid layers are:

        - ``hydrolocation_waterbody`` for Sink, Spring, Waterbody Outlet
        - ``hydrolocation_flowline`` for Headwater, Terminus, Divergence, Confluence, Catchment Outlet
        - ``hydrolocation_reach`` for Reach Code, External Connection
        - ``flowline`` for river flowlines
        - ``waterbody`` for waterbodies
        - ``drainage_area`` for drainage areas
        - ``catchment`` for catchments

    outfields : str or list, optional
        Target field name(s), default to "*" i.e., all the fields.
    crs : str, int, or pyproj.CRS, optional
        Target spatial reference, default to ``EPSG:4326``.

    Methods
    -------
    bygeom(geom, geo_crs=4326, sql_clause="", distance=None, return_m=False, return_geom=True)
        Get features within a geometry that can be combined with a SQL where clause.
    byids(field, fids, return_m=False, return_geom=True)
        Get features by object IDs.
    bysql(sql_clause, return_m=False, return_geom=True)
        Get features using a valid SQL 92 WHERE clause.
    """

    def __init__(
        self,
        layer: HP3D_LAYERS,
        outfields: str | list[str] = "*",
        crs: CRSTYPE = 4326,
    ):
        self.valid_layers = {
            "hydrolocation_waterbody": "Sink, Spring, Waterbody Outlet",
            "hydrolocation_flowline": "Headwater, Terminus, Divergence, Confluence, Catchment Outlet",
            "hydrolocation_reach": "Reach Code, External Connection",
            "flowline": "Flowline",
            "waterbody": "Waterbody",
            "drainage_area": "Drainage Area",
            "catchment": "Catchment",
        }
        _layer = self.valid_layers.get(layer)
        if _layer is None:
            raise InputValueError("layer", list(self.valid_layers))
        super().__init__(
            ServiceURL().restful.hp3d,
            _layer,
            outfields,
            crs,
        )


class PyGeoAPI(PyGeoAPIBase):
    """Access `PyGeoAPI <https://labs.waterdata.usgs.gov/api/nldi/pygeoapi>`__ service."""

    def flow_trace(
        self,
        coord: tuple[float, float],
        crs: CRSTYPE = 4326,
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
        self, coord: tuple[float, float], crs: CRSTYPE = 4326, upstream: bool = False
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
        crs: CRSTYPE = 4326,
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
        line = geoutils.geometry_reproject(line, crs, 4326)
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
        crs: CRSTYPE = 4326,
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
        self, coord: tuple[float, float], width: float, numpts: int, crs: CRSTYPE = 4326
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
        feat = cast("gpd.GeoDataFrame", feat.reset_index())
        return feat
    return gdf


class WaterData:
    """Access to `WaterData <https://labs.waterdata.usgs.gov/geoserver>`__ service.

    Parameters
    ----------
    layer : str
        A valid layer from the WaterData service. Valid layers are:

        - ``catchmentsp``
        - ``gagesii``
        - ``gagesii_basins``
        - ``nhdarea``
        - ``nhdflowline_network``
        - ``nhdflowline_nonnetwork``
        - ``nhdwaterbody``
        - ``wbd02``
        - ``wbd04``
        - ``wbd06``
        - ``wbd08``
        - ``wbd10``
        - ``wbd12``

        Note that the layers' namespace for the WaterData service is
        ``wmadata`` and will be added to the given ``layer`` argument
        if it is not provided.
    crs : str, int, or pyproj.CRS, optional
        The target spatial reference system, defaults to ``epsg:4326``.
    validation : bool, optional
        Whether to validate the input data, defaults to ``True``.
    """

    def __init__(
        self,
        layer: WD_LAYERS,
        crs: CRSTYPE = 4326,
    ) -> None:
        self.valid_layers = [
            "catchmentsp",
            "gagesii",
            "gagesii_basins",
            "huc08",
            "huc12",
            "nhdarea",
            "nhdflowline_network",
            "nhdflowline_nonnetwork",
            "nhdwaterbody",
            "wbd02",
            "wbd04",
            "wbd06",
            "wbd08",
            "wbd10",
            "wbd12",
        ]
        if layer not in self.valid_layers:
            raise InputValueError("layer", self.valid_layers)
        self.layer = layer if ":" in layer else f"wmadata:{layer}"
        if "wbd" in self.layer and "20201006" not in self.layer:
            self.layer = f"{self.layer}_20201006"
        self.crs = crs
        self.wfs = WFS(
            ServiceURL().wfs.waterdata,
            layer=self.layer,
            outformat="application/json",
            version="2.0.0",
            crs=4269,
            validation=False,
        )

    def _to_geodf(self, resp: list[dict[str, Any]]) -> gpd.GeoDataFrame:
        """Convert a response from WaterData to a GeoDataFrame.

        Parameters
        ----------
        resp : list of dicts
            A ``json`` response from a WaterData request.

        Returns
        -------
        geopandas.GeoDataFrame
            The requested features in a GeoDataFrames.
        """
        try:
            features = geoutils.json2geodf(resp, self.wfs.crs, self.crs)
        except EmptyResponseError as ex:
            raise ZeroMatchedError from ex

        if features.empty:
            raise ZeroMatchedError
        return features

    def bybox(
        self,
        bbox: tuple[float, float, float, float],
        box_crs: CRSTYPE = 4326,
        sort_attr: str | None = None,
    ) -> gpd.GeoDataFrame:
        """Get features within a bounding box.

        Parameters
        ----------
        bbox : tuple of floats
            A bounding box in the form of (minx, miny, maxx, maxy).
        box_crs : str, int, or pyproj.CRS, optional
            The spatial reference system of the bounding box, defaults to ``epsg:4326``.
        sort_attr : str, optional
            The column name in the database to sort request by, defaults
            to the first attribute in the schema that contains ``id`` in its name.

        Returns
        -------
        geopandas.GeoDataFrame
            The requested features in a GeoDataFrames.
        """
        resp = self.wfs.getfeature_bybox(
            bbox,
            box_crs,
            always_xy=True,
            sort_attr=sort_attr,
        )
        resp = cast("list[dict[str, Any]]", resp)
        return self._to_geodf(resp)

    def bygeom(
        self,
        geometry: Polygon | MultiPolygon,
        geo_crs: CRSTYPE = 4326,
        xy: bool = True,
        predicate: PREDICATES = "intersects",
        sort_attr: str | None = None,
    ) -> gpd.GeoDataFrame:
        """Get features within a geometry.

        Parameters
        ----------
        geometry : shapely.Polygon or shapely.MultiPolygon
            The input (multi)polygon to request the data.
        geo_crs : str, int, or pyproj.CRS, optional
            The CRS of the input geometry, default to epsg:4326.
        xy : bool, optional
            Whether axis order of the input geometry is xy or yx.
        predicate : str, optional
            The geometric prediacte to use for requesting the data, defaults to
            INTERSECTS. Valid predicates are:

            - ``equals``
            - ``disjoint``
            - ``intersects``
            - ``touches``
            - ``crosses``
            - ``within``
            - ``contains``
            - ``overlaps``
            - ``relate``
            - ``beyond``

        sort_attr : str, optional
            The column name in the database to sort request by, defaults
            to the first attribute in the schema that contains ``id`` in its name.

        Returns
        -------
        geopandas.GeoDataFrame
            The requested features in the given geometry.
        """
        resp = self.wfs.getfeature_bygeom(
            geometry, geo_crs, always_xy=not xy, predicate=predicate.upper(), sort_attr=sort_attr
        )
        resp = cast("list[dict[str, Any]]", resp)
        return self._to_geodf(resp)

    def bydistance(
        self,
        coords: tuple[float, float],
        distance: int,
        loc_crs: CRSTYPE = 4326,
        sort_attr: str | None = None,
    ) -> gpd.GeoDataFrame:
        """Get features within a radius (in meters) of a point.

        Parameters
        ----------
        coords : tuple of float
            The x, y coordinates of the point.
        distance : int
            The radius (in meters) to search within.
        loc_crs : str, int, or pyproj.CRS, optional
            The CRS of the input coordinates, default to ``epsg:4326``.
        sort_attr : str, optional
            The column name in the database to sort request by, defaults
            to the first attribute in the schema that contains ``id`` in its name.

        Returns
        -------
        geopandas.GeoDataFrame
            Requested features as a GeoDataFrame.
        """
        if not (isinstance(coords, tuple) and len(coords) == 2):
            raise InputTypeError("coods", "tuple of length 2", "(x, y)")

        x, y = ogc.match_crs([coords], loc_crs, self.wfs.crs)[0]
        geom_name = self.wfs.schema[self.layer].get("geometry_column", "the_geom")
        cql_filter = f"DWITHIN({geom_name},POINT({y:.6f} {x:.6f}),{distance},meters)"
        resp = self.wfs.getfeature_byfilter(
            cql_filter,
            "GET",
            sort_attr=sort_attr,
        )
        resp = cast("list[dict[str, Any]]", resp)
        return self._to_geodf(resp)

    def byid(
        self, featurename: str, featureids: Sequence[int | str] | int | str
    ) -> gpd.GeoDataFrame:
        """Get features based on IDs."""
        resp = self.wfs.getfeature_byid(
            featurename,
            featureids,
        )
        resp = cast("list[dict[str, Any]]", resp)
        features = self._to_geodf(resp)

        if isinstance(featureids, (str, int)):
            fids = [str(featureids)]
        else:
            fids = [str(f) for f in featureids]

        failed = set(fids).difference(set(features[featurename].astype(str)))

        if failed:
            msg = ". ".join(
                (
                    f"{len(failed)} of {len(fids)} requests failed.",
                    f"IDs of the failed requests are {list(failed)}",
                )
            )
            warnings.warn(msg, UserWarning, stacklevel=2)
        return features

    def byfilter(
        self,
        cql_filter: str,
        method: Literal["GET", "get", "POST", "post"] = "GET",
        sort_attr: str | None = None,
    ) -> gpd.GeoDataFrame:
        """Get features based on a CQL filter.

        Parameters
        ----------
        cql_filter : str
            The CQL filter to use for requesting the data.
        method : str, optional
            The HTTP method to use for requesting the data, defaults to GET.
            Allowed methods are GET and POST.
        sort_attr : str, optional
            The column name in the database to sort request by, defaults
            to the first attribute in the schema that contains ``id`` in its name.

        Returns
        -------
        geopandas.GeoDataFrame
            The requested features as a GeoDataFrames.
        """
        resp = self.wfs.getfeature_byfilter(
            cql_filter,
            method,
            sort_attr,
        )
        resp = cast("list[dict[str, Any]]", resp)
        return self._to_geodf(resp)

    def __repr__(self) -> str:
        """Print the services properties."""
        return "\n".join(
            (
                "Connected to the WaterData web service on GeoServer:",
                f"URL: {self.wfs.url}",
                f"Version: {self.wfs.version}",
                f"Layer: {self.layer}",
                f"Output Format: {self.wfs.outformat}",
                f"Output CRS: {self.crs}",
            )
        )


class NHDPlusHR(AGRBase):
    """Access National Hydrography Dataset (NHD) Plus high resolution.

    Notes
    -----
    For more info visit: https://hydro.nationalmap.gov/arcgis/rest/services/NHDPlus_HR/MapServer

    Parameters
    ----------
    layer : str, optional
        A valid service layer. Valid layers are:

        - ``gage`` for NHDPlusGage layer
        - ``sink`` for NHDPlusSink layer
        - ``point`` for NHDPoint layer
        - ``flowline`` for NetworkNHDFlowline layer
        - ``non_network_flowline`` for NonNetworkNHDFlowline layer
        - ``flow_direction`` for FlowDirection layer
        - ``wall`` for NHDPlusWall layer
        - ``line`` for NHDLine layer
        - ``area`` for NHDArea layer
        - ``waterbody`` for NHDWaterbody layer
        - ``catchment`` for NHDPlusCatchment layer
        - ``boundary_unit`` for NHDPlusBoundaryUnit layer
        - ``huc12`` for WBDHU12 layer

    outfields : str or list, optional
        Target field name(s), default to "*" i.e., all the fields.
    crs : str, int, or pyproj.CRS, optional
        Target spatial reference, default to ``EPSG:4326``.

    Methods
    -------
    bygeom(geom, geo_crs=4326, sql_clause="", distance=None, return_m=False, return_geom=True)
        Get features within a geometry that can be combined with a SQL where clause.
    byids(field, fids, return_m=False, return_geom=True)
        Get features by object IDs.
    bysql(sql_clause, return_m=False, return_geom=True)
        Get features using a valid SQL 92 WHERE clause.
    """

    def __init__(
        self,
        layer: NHDHR_LAYERS,
        outfields: str | list[str] = "*",
        crs: CRSTYPE = 4326,
    ):
        self.valid_layers = {
            "gauge": "NHDPlusGage",
            "sink": "NHDPlusSink",
            "point": "NHDPoint",
            "flowline": "NetworkNHDFlowline",
            "non_network_flowline": "NonNetworkNHDFlowline",
            "flow_direction": "FlowDirection",
            "wall": "NHDPlusWall",
            "line": "NHDLine",
            "area": "NHDArea",
            "waterbody": "NHDWaterbody",
            "catchment": "NHDPlusCatchment",
            "boundary_unit": "NHDPlusBoundaryUnit",
            "huc12": "WBDHU12",
        }
        _layer = self.valid_layers.get(layer)
        if _layer is None:
            raise InputValueError("layer", list(self.valid_layers))
        super().__init__(
            ServiceURL().restful.nhdplushr,
            _layer,
            outfields,
            crs,
        )


class NLDI:
    """Access the Hydro Network-Linked Data Index (NLDI) service."""

    def __init__(self) -> None:
        self.base_url = ServiceURL().restful.nldi

        resp = ar.retrieve_json([f"{self.base_url}/linked-data"])
        resp = cast("list[list[dict[str, Any]]]", resp)
        self.valid_fsources = {r["source"]: r["sourceName"] for r in resp[0]}

        resp = ar.retrieve_json([f"{self.base_url}/lookups"])
        resp = cast("list[list[dict[str, Any]]]", resp)
        self.valid_chartypes = {r["type"]: r["typeName"] for r in resp[0]}

        resp = ar.retrieve_json([r["characteristics"] for r in resp[0]])
        resp = cast("list[dict[str, Any]]", resp)
        char_types = (
            ogc.traverse_json(r, ["characteristicMetadata", "characteristic"]) for r in resp
        )
        self.valid_characteristics = pd.concat(
            (pd.DataFrame(c) for c in char_types), ignore_index=True
        )

    @staticmethod
    def _check_resp(resp: dict[str, Any] | list[dict[str, Any]] | None) -> bool:
        if resp is None:
            return False
        if isinstance(resp, dict) and resp.get("type") == "error":
            return False
        if isinstance(resp, dict) and "features" in resp and not resp["features"]:
            return False
        return True

    @overload
    def _get_urls(
        self, url_parts: Generator[tuple[str, ...], None, None] | str, raw: Literal[True] = ...
    ) -> tuple[list[int], list[dict[str, Any]] | dict[str, Any]]: ...

    @overload
    def _get_urls(
        self, url_parts: Generator[tuple[str, ...], None, None] | str, raw: Literal[False] = ...
    ) -> gpd.GeoDataFrame: ...

    def _get_urls(
        self, url_parts: Generator[tuple[str, ...], None, None] | str, raw: bool = False
    ) -> tuple[list[int], list[dict[str, Any]] | dict[str, Any]] | gpd.GeoDataFrame:
        """Send a request to the service using GET method."""
        if isinstance(url_parts, str):
            urls = [URL(url_parts)]
        else:
            urls = [URL("/".join((self.base_url, *u))) for u in url_parts]
        resp = ar.retrieve_json([str(u) for u in urls], raise_status=False)

        try:
            index, resp = zip(*((i, r) for i, r in enumerate(resp) if self._check_resp(r)))
        except ValueError as ex:
            raise ZeroMatchedError from ex

        index = cast("list[int]", list(index))
        resp = cast("list[dict[str, Any]] | dict[str, Any]", list(resp))
        failed = [i for i in range(len(urls)) if i not in index]

        if failed:
            msg = ". ".join(
                (
                    f"{len(failed)} of {len(urls)} requests failed.",
                    f"Indices of the failed requests are {failed}",
                )
            )
            warnings.warn(msg, UserWarning, stacklevel=2)

        if raw:
            return index, resp

        return geoutils.json2geodf(resp, 4269, 4326)

    def _validate_fsource(self, fsource: str) -> None:
        """Check if the given feature source is valid."""
        if fsource not in self.valid_fsources:
            valids = [f'"{s}" for {d}' for s, d in self.valid_fsources.items()]
            raise InputValueError("fsource", valids)

    def getfeature_byid(self, fsource: str, fids: str | list[str]) -> gpd.GeoDataFrame:
        """Get feature(s) based ID(s).

        Parameters
        ----------
        fsource : str
            The name of feature(s) source. The valid sources are:

            * 'comid' for NHDPlus comid.
            * 'ca_gages' for Streamgage catalog for CA SB19
            * 'gfv11_pois' for USGS Geospatial Fabric V1.1 Points of Interest
            * 'huc12pp' for HUC12 Pour Points
            * 'nmwdi-st' for New Mexico Water Data Initiative Sites
            * 'nwisgw' for NWIS Groundwater Sites
            * 'nwissite' for NWIS Surface Water Sites
            * 'ref_gage' for geoconnex.us reference gages
            * 'vigil' for Vigil Network Data
            * 'wade' for Water Data Exchange 2.0 Sites
            * 'WQP' for Water Quality Portal

        fid : str or list of str
            Feature ID(s).

        Returns
        -------
        geopandas.GeoDataFrame or (geopandas.GeoDataFrame, list)
            NLDI indexed features in EPSG:4326. If some IDs don't return any features
            a list of missing ID(s) are returned as well.
        """
        self._validate_fsource(fsource)
        fids = [fids] if isinstance(fids, (int, str)) else list(fids)
        urls = (("linked-data", fsource, f) for f in fids)
        return self._get_urls(urls, False)

    def __byloc(
        self,
        source: str,
        coords: tuple[float, float] | list[tuple[float, float]],
        loc_crs: CRSTYPE = 4326,
    ) -> gpd.GeoDataFrame:
        """Get the closest feature ID(s) based on coordinates.

        Parameters
        ----------
        source : str
            The name of feature(s) source. The valid sources are ``comid`` and ``feature``.
        coords : tuple or list
            A tuple of length two (x, y) or a list of them.
        loc_crs : str, int, or pyproj.CRS, optional
            The spatial reference of the input coordinate, defaults to EPSG:4326.

        Returns
        -------
        geopandas.GeoDataFrame or (geopandas.GeoDataFrame, list)
            NLDI indexed ComID(s) in EPSG:4326. If some coords don't return any ComID
            a list of missing coords are returned as well.
        """
        _coords = geoutils.coords_list(coords)
        _coords = ogc.match_crs(_coords, loc_crs, 4326)
        endpoint = "comid/position" if source == "feature" else "hydrolocation"
        urls = (
            ("linked-data", f"{endpoint}?coords=POINT({lon:.6f} {lat:.6f})") for lon, lat in _coords
        )
        return self._get_urls(urls, False)

    def comid_byloc(
        self,
        coords: tuple[float, float] | list[tuple[float, float]],
        loc_crs: CRSTYPE = 4326,
    ) -> gpd.GeoDataFrame:
        """Get the closest ComID based on coordinates using ``hydrolocation`` endpoint.

        Notes
        -----
        This function tries to find the closest ComID based on flowline grid cells. If
        such a cell is not found, it will return the closest ComID using the flowtrace
        endpoint of the PyGeoAPI service to find the closest downstream ComID. The returned
        dataframe has a ``measure`` column that indicates the location of the input
        coordinate on the flowline as a percentage of the total flowline length.

        Parameters
        ----------
        coords : tuple or list of tuples
            A tuple of length two (x, y) or a list of them.
        loc_crs : str, int, or pyproj.CRS, optional
            The spatial reference of the input coordinate, defaults to EPSG:4326.

        Returns
        -------
        geopandas.GeoDataFrame or (geopandas.GeoDataFrame, list)
            NLDI indexed ComID(s) and points in EPSG:4326. If some coords don't return
            any ComID a list of missing coords are returned as well.
        """
        comids = self.__byloc("comid", coords, loc_crs)
        comids = comids[comids["source"] == "indexed"].reset_index(drop=True)
        comids = cast("gpd.GeoDataFrame", comids)
        return comids

    def feature_byloc(
        self,
        coords: tuple[float, float] | list[tuple[float, float]],
        loc_crs: CRSTYPE = 4326,
    ) -> gpd.GeoDataFrame:
        """Get the closest feature ID(s) based on coordinates using ``position`` endpoint.

        Parameters
        ----------
        coords : tuple or list
            A tuple of length two (x, y) or a list of them.
        loc_crs : str, int, or pyproj.CRS, optional
            The spatial reference of the input coordinate, defaults to EPSG:4326.

        Returns
        -------
        geopandas.GeoDataFrame or (geopandas.GeoDataFrame, list)
            NLDI indexed feature ID(s) and flowlines in EPSG:4326. If some coords don't
            return any IDs a list of missing coords are returned as well.
        """
        return self.__byloc("feature", coords, loc_crs)

    def get_basins(
        self,
        feature_ids: str | int | Sequence[str | int],
        fsource: str = "nwissite",
        split_catchment: bool = False,
        simplified: bool = True,
    ) -> gpd.GeoDataFrame:
        """Get basins for a list of station IDs.

        Parameters
        ----------
        feature_ids : str or list
            Target feature ID(s).
        fsource : str
            The name of feature(s) source, defaults to ``nwissite``.
            The valid sources are:

            * 'comid' for NHDPlus comid.
            * 'ca_gages' for Streamgage catalog for CA SB19
            * 'gfv11_pois' for USGS Geospatial Fabric V1.1 Points of Interest
            * 'huc12pp' for HUC12 Pour Points
            * 'nmwdi-st' for New Mexico Water Data Initiative Sites
            * 'nwisgw' for NWIS Groundwater Sites
            * 'nwissite' for NWIS Surface Water Sites
            * 'ref_gage' for geoconnex.us reference gages
            * 'vigil' for Vigil Network Data
            * 'wade' for Water Data Exchange 2.0 Sites
            * 'WQP' for Water Quality Portal

        split_catchment : bool, optional
            If ``True``, split basins at their outlet locations. Default to ``False``.
        simplified : bool, optional
            If ``True``, return a simplified version of basin geometries. Default to ``True``.

        Returns
        -------
        geopandas.GeoDataFrame or (geopandas.GeoDataFrame, list)
            NLDI indexed basins in EPSG:4326. If some IDs don't return any features
            a list of missing ID(s) are returned as well.
        """
        self._validate_fsource(fsource)
        feature_ids = [feature_ids] if isinstance(feature_ids, (str, int)) else list(feature_ids)
        feature_ids = [str(fid) for fid in feature_ids]

        if not feature_ids:
            raise InputTypeError("feature_ids", "list with at least one element")

        if fsource == "nwissite":
            feature_ids = [f"USGS-{fid.lower().replace('usgs-', '')}" for fid in feature_ids]

        payload = {}
        if split_catchment:
            payload["splitCatchment"] = "true"

        if not simplified:
            payload["simplified"] = "false"

        query = ""
        if payload:
            query = URL.build(query=payload).query_string

        urls = (("linked-data", fsource, fid, f"basin?{query}") for fid in feature_ids)
        index, resp = self._get_urls(urls, True)
        basins = geoutils.json2geodf(resp, 4269, 4326)
        basins.index = pd.Index([feature_ids[i] for i in index], name="identifier")
        basins = basins[~basins.geometry.isnull()].copy()
        basins = cast("gpd.GeoDataFrame", basins)
        return basins

    @overload
    def getcharacteristic_byid(
        self,
        feature_ids: str | int | Sequence[str | int],
        char_type: str,
        fsource: str = ...,
        char_ids: str | list[str] = ...,
        values_only: Literal[True] = ...,
    ) -> pd.DataFrame: ...

    @overload
    def getcharacteristic_byid(
        self,
        feature_ids: str | int | Sequence[str | int],
        char_type: str,
        fsource: str = ...,
        char_ids: str | list[str] = ...,
        values_only: Literal[False] = ...,
    ) -> tuple[pd.DataFrame, pd.DataFrame]: ...

    def getcharacteristic_byid(
        self,
        feature_ids: str | int | Sequence[str | int],
        char_type: str,
        fsource: str = "comid",
        char_ids: str | list[str] = "all",
        values_only: bool = True,
    ) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
        """Get characteristics using a list ComIDs.

        Parameters
        ----------
        feature_ids : str or list
            Target feature ID(s).
        char_type : str
            Type of the characteristic. Valid values are ``local`` for
            individual reach catchments, ``tot`` for network-accumulated values
            using total cumulative drainage area and ``div`` for network-accumulated values
            using divergence-routed.
        fsource : str, optional
            The name of feature(s) source, defaults to ``comid``.
            The valid sources are:

            * 'comid' for NHDPlus comid.
            * 'ca_gages' for Streamgage catalog for CA SB19
            * 'gfv11_pois' for USGS Geospatial Fabric V1.1 Points of Interest
            * 'huc12pp' for HUC12 Pour Points
            * 'nmwdi-st' for New Mexico Water Data Initiative Sites
            * 'nwisgw' for NWIS Groundwater Sites
            * 'nwissite' for NWIS Surface Water Sites
            * 'ref_gage' for geoconnex.us reference gages
            * 'vigil' for Vigil Network Data
            * 'wade' for Water Data Exchange 2.0 Sites
            * 'WQP' for Water Quality Portal

        char_ids : str or list, optional
            Name(s) of the target characteristics, default to all.
        values_only : bool, optional
            Whether to return only ``characteristic_value`` as a series, default to True.
            If is set to False, ``percent_nodata`` is returned as well.

        Returns
        -------
        pandas.DataFrame or tuple of pandas.DataFrame
            Either only ``characteristic_value`` as a dataframe or
            or if ``values_only`` is Fale return ``percent_nodata`` as well.
        """
        self._validate_fsource(fsource)
        if char_type not in self.valid_chartypes:
            valids = [f'"{s}" for {d}' for s, d in self.valid_chartypes.items()]
            raise InputValueError("char", valids)

        feature_ids = [feature_ids] if isinstance(feature_ids, (str, int)) else list(feature_ids)
        feature_ids = [str(c) for c in feature_ids]

        v_dict, nd_dict = {}, {}

        if char_ids == "all":
            payload = None
        else:
            _char_ids = [char_ids] if isinstance(char_ids, str) else list(char_ids)
            valid_charids = self.valid_characteristics["characteristic_id"].to_list()

            if any(c not in valid_charids for c in _char_ids):
                raise InputValueError("char_id", valid_charids)
            payload = {"characteristicId": ",".join(_char_ids)}

        query = URL.build(query=payload).query_string
        urls = (("linked-data", fsource, c, f"{char_type}?{query}") for c in feature_ids)
        index, resp = self._get_urls(urls, True)
        resp = cast("list[dict[str, Any]]", resp)
        for i in index:
            char = pd.DataFrame.from_dict(resp[i]["characteristics"], orient="columns").T
            char.columns = char.iloc[0]
            char = char.drop(index="characteristic_id")

            v_dict[feature_ids[i]] = char.loc["characteristic_value"]
            if values_only:
                continue

            nd_dict[feature_ids[i]] = char.loc["percent_nodata"]

        def todf(df_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
            df = pd.DataFrame.from_dict(df_dict, orient="index")
            df[df == ""] = np.nan
            df.index = df.index.astype("int64")
            return df.astype("f4")

        chars = todf(v_dict)
        if values_only:
            return chars

        return chars, todf(nd_dict)

    def navigate_byid(
        self,
        fsource: str,
        fid: str | int,
        navigation: str,
        source: str,
        distance: int = 500,
        trim_start: bool = False,
        stop_comid: str | int | None = None,
    ) -> gpd.GeoDataFrame:
        """Navigate the NHDPlus database from a single feature id up to a distance.

        Parameters
        ----------
        fsource : str
            The name of feature(s) source. The valid sources are:

            * 'comid' for NHDPlus comid.
            * 'ca_gages' for Streamgage catalog for CA SB19
            * 'gfv11_pois' for USGS Geospatial Fabric V1.1 Points of Interest
            * 'huc12pp' for HUC12 Pour Points
            * 'nmwdi-st' for New Mexico Water Data Initiative Sites
            * 'nwisgw' for NWIS Groundwater Sites
            * 'nwissite' for NWIS Surface Water Sites
            * 'ref_gage' for geoconnex.us reference gages
            * 'vigil' for Vigil Network Data
            * 'wade' for Water Data Exchange 2.0 Sites
            * 'WQP' for Water Quality Portal

        fid : str or int
            The ID of the feature.
        navigation : str
            The navigation method.
        source : str
            Return the data from another source after navigating
            features from ``fsource``.
        distance : int, optional
            Limit the search for navigation up to a distance in km,
            defaults is 500 km. Note that this is an expensive request so you
            have be mindful of the value that you provide. The value must be
            between 1 to 9999 km.
        trim_start : bool, optional
            If ``True``, trim the starting flowline at the source feature,
            defaults to ``False``.
        stop_comid : str or int, optional
            The ComID to stop the navigationation, defaults to ``None``.

        Returns
        -------
        geopandas.GeoDataFrame
            NLDI indexed features in EPSG:4326.
        """
        if not (1 <= distance <= 9999):
            raise InputRangeError("distance", "[1, 9999]")

        self._validate_fsource(fsource)

        url = "/".join((self.base_url, "linked-data", fsource, str(fid), "navigation"))
        _, resp = self._get_urls(url, True)
        resp = cast("list[dict[str, str]]", resp)
        valid_navigations = resp[0]
        if not valid_navigations:
            raise ZeroMatchedError

        if navigation not in valid_navigations:
            raise InputValueError("navigation", list(valid_navigations))

        _, resp = self._get_urls(valid_navigations[navigation], True)
        resp = cast("list[list[dict[str, Any]]]", resp)
        valid_sources = {s["source"].lower(): s["features"] for s in resp[0]}
        if source not in valid_sources:
            raise InputValueError("source", list(valid_sources))

        payload = {"distance": str(round(distance)), "trimStart": str(trim_start).lower()}
        if stop_comid:
            payload["stopComid"] = str(stop_comid)
        url = f"{valid_sources[source]}?{URL.build(query=payload).query_string}"
        return self._get_urls(url, False)

    def navigate_byloc(
        self,
        coords: tuple[float, float],
        navigation: str | None = None,
        source: str | None = None,
        loc_crs: CRSTYPE = 4326,
        distance: int = 500,
        trim_start: bool = False,
        stop_comid: str | int | None = None,
    ) -> gpd.GeoDataFrame:
        """Navigate the NHDPlus database from a coordinate.

        Notes
        -----
        This function first calls the ``feature_byloc`` function to get the
        comid of the nearest flowline and then calls the ``navigate_byid``
        function to get the features from the obtained ``comid``.

        Parameters
        ----------
        coords : tuple
            A tuple of length two (x, y).
        navigation : str, optional
            The navigation method, defaults to None which throws an exception
            if ``comid_only`` is False.
        source : str, optional
            Return the data from another source after navigating
            the features based on ``comid``, defaults to ``None`` which throws
            an exception if ``comid_only`` is False.
        loc_crs : str, int, or pyproj.CRS, optional
            The spatial reference of the input coordinate, defaults to EPSG:4326.
        distance : int, optional
            Limit the search for navigation up to a distance in km,
            defaults to 500 km. Note that this is an expensive request so you
            have be mindful of the value that you provide.
        trim_start : bool, optional
            If ``True``, trim the starting flowline at the source feature,
            defaults to ``False``.
        stop_comid : str or int, optional
            The ComID to stop the navigationation, defaults to ``None``.

        Returns
        -------
        geopandas.GeoDataFrame
            NLDI indexed features in EPSG:4326.
        """
        comid_df = self.feature_byloc(coords, loc_crs)
        comid = comid_df.comid.iloc[0]

        if navigation is None or source is None:
            raise MissingItemError(["navigation", "source"])

        return self.navigate_byid(
            "comid", comid, navigation, source, distance, trim_start, stop_comid
        )
