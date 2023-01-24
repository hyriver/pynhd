"""Access NLDI and WaterData databases."""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Mapping, Sequence, Union

import async_retriever as ar
import geopandas as gpd
import numpy as np
import pandas as pd
import pygeoogc as ogc
import pygeoutils as geoutils
import pyproj
from loguru import logger
from pygeoogc import WFS, InputValueError, ServiceUnavailableError, ServiceURL
from pygeoogc import ZeroMatchedError as ZeroMatchedErrorOGC
from pygeoutils import EmptyResponseError, InputTypeError

from pynhd.core import AGRBase, GeoConnex, PyGeoAPIBase, PyGeoAPIBatch
from pynhd.exceptions import InputRangeError, MissingItemError, ZeroMatchedError

if TYPE_CHECKING:
    from shapely.geometry import MultiPolygon, Polygon

    CRSTYPE = Union[int, str, pyproj.CRS]


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
        Target spatial reference, default to ``EPSG:4326``
    """

    def __init__(
        self,
        layer: str,
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


class PyGeoAPI(PyGeoAPIBase):
    """Access `PyGeoAPI <https://labs.waterdata.usgs.gov/api/nldi/pygeoapi>`__ service."""

    def flow_trace(
        self,
        coord: tuple[float, float],
        crs: CRSTYPE = 4326,
        direction: str = "none",
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
        >>> gdf = pga.elevation_profile(
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


def pygeoapi(coords: gpd.GeoDataFrame, service: str) -> gpd.GeoDataFrame:
    """Return a GeoDataFrame from the flowtrace service.

    Parameters
    ----------
    coords : geopandas.GeoDataFrame
        A GeoDataFrame containing the coordinates to query.
        The required columns services are:

        * ``flow_trace``: ``direction`` that indicates the direction of the flow trace.
          It can be ``up``, ``down``, or ``none``.
        * ``split_catchment``: ``upstream`` that indicates whether to return all upstream
          catchments or just the local catchment.
        * ``elevation_profile``: ``numpts`` that indicates the number of points to extract
          along the flowpath and ``3dep_res`` that indicates the target resolution for
          requesting the DEM from 3DEP service.
        * ``cross_section``: ``numpts`` that indicates the number of points to extract
          along the flowpath and ``width`` that indicates the width of the cross-section
          in meters.

    service : str
        The service to query, can be ``flow_trace``, ``split_catchment``, ``elevation_profile``,
        or ``cross_section``.

    Returns
    -------
    geopandas.GeoDataFrame
        A GeoDataFrame containing the results of requested service.

    Examples
    --------
    >>> from shapely.geometry import Point
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
    pgab = PyGeoAPIBatch(coords)
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
        layer: str,
        crs: CRSTYPE = 4326,
        validation: bool = True,
    ) -> None:
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
            validation=validation,
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
        resp: list[dict[str, Any]] = self.wfs.getfeature_bybox(  # type: ignore
            bbox,
            box_crs,
            always_xy=True,
            sort_attr=sort_attr,
        )
        return self._to_geodf(resp)

    def bygeom(
        self,
        geometry: Polygon | MultiPolygon,
        geo_crs: CRSTYPE = 4326,
        xy: bool = True,
        predicate: str = "INTERSECTS",
        sort_attr: str | None = None,
    ) -> gpd.GeoDataFrame:
        """Get features within a geometry.

        Parameters
        ----------
        geometry : shapely.geometry
            The input geometry
        geo_crs : str, int, or pyproj.CRS, optional
            The CRS of the input geometry, default to epsg:4326.
        xy : bool, optional
            Whether axis order of the input geometry is xy or yx.
        predicate : str, optional
            The geometric prediacte to use for requesting the data, defaults to
            INTERSECTS. Valid predicates are:

            - ``EQUALS``
            - ``DISJOINT``
            - ``INTERSECTS``
            - ``TOUCHES``
            - ``CROSSES``
            - ``WITHIN``
            - ``CONTAINS``
            - ``OVERLAPS``
            - ``RELATE``
            - ``BEYOND``

        sort_attr : str, optional
            The column name in the database to sort request by, defaults
            to the first attribute in the schema that contains ``id`` in its name.

        Returns
        -------
        geopandas.GeoDataFrame
            The requested features in the given geometry.
        """
        resp: list[dict[str, Any]] = self.wfs.getfeature_bygeom(  # type: ignore
            geometry, geo_crs, always_xy=not xy, predicate=predicate, sort_attr=sort_attr
        )
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
            The CRS of the input coordinates, default to epsg:4326.
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
        resp: list[dict[str, Any]] = self.wfs.getfeature_byfilter(  # type: ignore
            cql_filter,
            "GET",
            sort_attr=sort_attr,
        )
        return self._to_geodf(resp)

    def byid(self, featurename: str, featureids: list[int | str] | int | str) -> gpd.GeoDataFrame:
        """Get features based on IDs."""
        resp: list[dict[str, Any]] = self.wfs.getfeature_byid(  # type: ignore
            featurename,
            featureids,
        )
        features = self._to_geodf(resp)

        if isinstance(featureids, (str, int)):
            fids = [str(featureids)]
        else:
            fids = [str(f) for f in featureids]
        missing = set(fids).difference(set(features[featurename].astype(str)))
        if missing:
            verb = "ID was" if len(missing) == 1 else "IDs were"
            logger.warning(
                f"The following requested feature {verb} not found in WaterData:\n"
                + ", ".join(missing)
            )
        return features

    def byfilter(
        self, cql_filter: str, method: str = "GET", sort_attr: str | None = None
    ) -> gpd.GeoDataFrame:
        """Get features based on a CQL filter.

        Parameters
        ----------
        cql_filter : str
            The CQL filter to use for requesting the data.
        method : str, optional
            The HTTP method to use for requesting the data, defaults to GET.
        sort_attr : str, optional
            The column name in the database to sort request by, defaults
            to the first attribute in the schema that contains ``id`` in its name.

        Returns
        -------
        geopandas.GeoDataFrame
            The requested features as a GeoDataFrames.
        """
        resp: list[dict[str, Any]] = self.wfs.getfeature_byfilter(  # type: ignore
            cql_filter,
            method,
            sort_attr,
        )
        return self._to_geodf(resp)

    def __repr__(self) -> str:
        """Print the services properties."""
        return (
            "Connected to the WaterData web service on GeoServer:\n"
            + f"URL: {self.wfs.url}\n"
            + f"Version: {self.wfs.version}\n"
            + f"Layer: {self.layer}\n"
            + f"Output Format: {self.wfs.outformat}\n"
            + f"Output CRS: {self.crs}"
        )


class NHDPlusHR(AGRBase):
    """Access National Hydrography Dataset (NHD) high resolution.

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
        Target spatial reference, default to ``EPSG:4326``
    """

    def __init__(
        self,
        layer: str,
        outfields: str | list[str] = "*",
        crs: CRSTYPE = 4326,
    ):
        self.valid_layers = {
            "gage": "NHDPlusGage",
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

    def _get_url(
        self, url: str, payload: dict[str, str] | None = None
    ) -> list[dict[str, Any]] | dict[str, Any]:
        """Send a request to the service using GET method."""
        if payload is None:
            payload = {"f": "json"}
        else:
            payload.update({"f": "json"})

        try:
            resp = ar.retrieve_json([url], [{"params": payload}])
        except ar.ServiceError as ex:
            raise ZeroMatchedError from ex
        except ConnectionError as ex:
            raise ServiceUnavailableError(self.base_url) from ex
        else:
            if isinstance(resp[0], dict) and resp[0].get("type", "") == "error":
                raise ZeroMatchedError(resp[0].get("description", "Feature not found"))
            if resp[0] is None:
                raise ZeroMatchedError
            return resp[0]

    def __init__(self) -> None:
        self.base_url = ServiceURL().restful.nldi

        resp: list[dict[str, Any]] = self._get_url(  # type: ignore
            "/".join([self.base_url, "linked-data"])
        )
        self.valid_fsources = {r["source"]: r["sourceName"] for r in resp}

        resp: list[dict[str, Any]] = self._get_url(  # type: ignore
            "/".join([self.base_url, "lookups"])
        )
        self.valid_chartypes = {r["type"]: r["typeName"] for r in resp}

    @staticmethod
    def _missing_warning(n_miss: int, n_tot: int) -> None:
        """Show a warning if there are missing features."""
        logger.warning(
            " ".join(
                [
                    f"{n_miss} of {n_tot} inputs didn't return any features.",
                    "They are returned as a list.",
                ]
            )
        )

    def _validate_fsource(self, fsource: str) -> None:
        """Check if the given feature source is valid."""
        if fsource not in self.valid_fsources:
            valids = [f'"{s}" for {d}' for s, d in self.valid_fsources.items()]
            raise InputValueError("fsource", valids)

    def _get_urls(
        self, urls: Mapping[Any, tuple[str, dict[str, str] | None]]
    ) -> tuple[gpd.GeoDataFrame, list[str]]:
        """Get basins for a list of station IDs.

        Parameters
        ----------
        urls : dict
            A dict with keys as feature ids and values as corresponding url and payload.

        Returns
        -------
        (geopandas.GeoDataFrame, list)
            NLDI indexed features in EPSG:4326 and list of ID(s) that no feature was found.
        """
        not_found = []
        resp = []
        for f, (u, p) in urls.items():
            try:
                rjson = self._get_url(u, p)
                resp.append((f, geoutils.json2geodf(rjson, 4269, 4326)))
            except (
                ZeroMatchedErrorOGC,
                ZeroMatchedError,
                InputTypeError,
                ar.ServiceError,
                EmptyResponseError,
            ):
                not_found.append(f)

        if not resp:
            raise ZeroMatchedError

        resp_df = gpd.GeoDataFrame(pd.concat(dict(resp)), crs=4326)

        return resp_df, not_found

    def getfeature_byid(
        self, fsource: str, fid: str | list[str]
    ) -> gpd.GeoDataFrame | tuple[gpd.GeoDataFrame, list[str]]:
        """Get feature(s) based ID(s).

        Parameters
        ----------
        fsource : str
            The name of feature(s) source. The valid sources are:

            * 'comid' for NHDPlus comid.
            * 'ca_gages' for Streamgage catalog for CA SB19
            * 'gfv11_pois' for USGS Geospatial Fabric V1.1 Points of Interest
            * 'huc12pp' for HUC12 Pour Points
            * 'nmwdi-st' for New Mexico Water Data Initative Sites
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
        fid = fid if isinstance(fid, list) else [fid]
        urls = {f: ("/".join([self.base_url, "linked-data", fsource, f]), None) for f in fid}
        features, not_found = self._get_urls(urls)

        if not_found:
            self._missing_warning(len(not_found), len(fid))
            return features, not_found

        return features

    def __byloc(
        self,
        source: str,
        coords: tuple[float, float] | list[tuple[float, float]],
        loc_crs: CRSTYPE = 4326,
    ) -> gpd.GeoDataFrame | tuple[gpd.GeoDataFrame, list[tuple[float, float]]]:
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
        endpoint = "comid/position" if source == "feature" else "hydrolocation"
        base_url = "/".join([self.base_url, "linked-data", endpoint])

        _coords = [coords] if isinstance(coords, tuple) else coords

        if not isinstance(_coords, list) or any(len(c) != 2 for c in _coords):
            raise InputTypeError("coords", "list or tuple")

        _coords = ogc.match_crs(_coords, loc_crs, 4326)

        urls = {
            f"{(lon, lat)}": (base_url, {"coords": f"POINT({lon} {lat})"}) for lon, lat in _coords
        }
        comids, not_found_str = self._get_urls(urls)

        if len(comids) == 0:
            raise ZeroMatchedError

        comids = comids.reset_index(drop=True)

        if not_found_str:
            not_found = [
                tuple(float(p) for p in re.sub(r"\(|\)| ", "", m).split(",")) for m in not_found_str
            ]
            self._missing_warning(len(not_found), len(_coords))
            return comids, not_found

        return comids

    def comid_byloc(
        self,
        coords: tuple[float, float] | list[tuple[float, float]],
        loc_crs: CRSTYPE = 4326,
    ) -> gpd.GeoDataFrame | tuple[gpd.GeoDataFrame, list[tuple[float, float]]]:
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
        if isinstance(comids, tuple):
            cdf, miss = comids
            return cdf[cdf["source"] == "indexed"].reset_index(drop=True), miss
        return comids[comids["source"] == "indexed"].reset_index(drop=True)

    def feature_byloc(
        self,
        coords: tuple[float, float] | list[tuple[float, float]],
        loc_crs: CRSTYPE = 4326,
    ) -> gpd.GeoDataFrame | tuple[gpd.GeoDataFrame, list[tuple[float, float]]]:
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
    ) -> gpd.GeoDataFrame | tuple[gpd.GeoDataFrame, list[str]]:
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
            * 'nmwdi-st' for New Mexico Water Data Initative Sites
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
        if not isinstance(feature_ids, Sequence):
            raise InputTypeError("feature_ids", "str, list or tuple")

        feature_ids = [feature_ids] if isinstance(feature_ids, (str, int)) else feature_ids

        if not feature_ids:
            raise InputTypeError("feature_ids", "list with at least one element")

        if fsource == "nwissite":
            feature_ids = [f"USGS-{str(fid).lower().replace('usgs-', '')}" for fid in feature_ids]

        ftype = type(feature_ids[0])
        payload = {
            "splitCatchment": str(split_catchment).lower(),
            "simplified": str(simplified).lower(),
        }
        urls = {
            ftype(str(fid).lower().replace("usgs-", "")): (
                f"{self.base_url}/linked-data/{fsource}/{fid}/basin",
                payload,
            )
            for fid in feature_ids
        }
        basins, not_found = self._get_urls(urls)
        basins = basins.reset_index(level=1, drop=True)
        basins.index.rename("identifier", inplace=True)
        nulls = basins.geometry.isnull()
        not_found += basins[nulls].index.to_list()
        basins = basins[~nulls].copy()

        if not_found:
            self._missing_warning(len(not_found), len(feature_ids))
            return basins, not_found

        return basins

    def getcharacteristic_byid(
        self,
        comids: list[str] | str,
        char_type: str,
        char_ids: str | list[str] = "all",
        values_only: bool = True,
    ) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
        """Get characteristics using a list ComIDs.

        Parameters
        ----------
        comids : str or list
            The NHDPlus Common Identifier(s).
        char_type : str
            Type of the characteristic. Valid values are ``local`` for
            individual reach catchments, ``tot`` for network-accumulated values
            using total cumulative drainage area and ``div`` for network-accumulated values
            using divergence-routed.
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
        if char_type not in self.valid_chartypes:
            valids = [f'"{s}" for {d}' for s, d in self.valid_chartypes.items()]
            raise InputValueError("char", valids)

        comids = comids if isinstance(comids, list) else [comids]
        v_dict, nd_dict = {}, {}

        if char_ids == "all":
            payload = None
        else:
            _char_ids = char_ids if isinstance(char_ids, list) else [char_ids]
            valid_charids = self.get_validchars(char_type)

            idx = valid_charids.index
            if any(c not in idx for c in _char_ids):
                vids = valid_charids["characteristic_description"]
                raise InputValueError("char_id", [f'"{s}" for {d}' for s, d in vids.items()])
            payload = {"characteristicId": ",".join(_char_ids)}

        for comid in comids:
            url = "/".join([self.base_url, "linked-data", "comid", f"{comid}", char_type])
            rjson: dict[str, Any] = self._get_url(url, payload)  # type: ignore
            char = pd.DataFrame.from_dict(rjson["characteristics"], orient="columns").T
            char.columns = char.iloc[0]
            char = char.drop(index="characteristic_id")

            v_dict[comid] = char.loc["characteristic_value"]
            if values_only:
                continue

            nd_dict[comid] = char.loc["percent_nodata"]

        def todf(df_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
            df = pd.DataFrame.from_dict(df_dict, orient="index")
            df[df == ""] = np.nan
            df.index = df.index.astype("int64")
            return df.astype("f4")

        chars = todf(v_dict)
        if values_only:
            return chars

        return chars, todf(nd_dict)

    def get_validchars(self, char_type: str) -> pd.DataFrame:
        """Get all the available characteristics IDs for a given characteristics type."""
        if char_type not in self.valid_chartypes:
            raise InputValueError("char", list(self.valid_chartypes))

        resp = self._get_url("/".join([self.base_url, "lookups", char_type, "characteristics"]))
        c_list = ogc.traverse_json(resp, ["characteristicMetadata", "characteristic"])
        return pd.DataFrame.from_dict(
            {c.pop("characteristic_id"): c for c in c_list}, orient="index"
        )

    def navigate_byid(
        self,
        fsource: str,
        fid: str,
        navigation: str,
        source: str,
        distance: int = 500,
        trim_start: bool = False,
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
            * 'nmwdi-st' for New Mexico Water Data Initative Sites
            * 'nwisgw' for NWIS Groundwater Sites
            * 'nwissite' for NWIS Surface Water Sites
            * 'ref_gage' for geoconnex.us reference gages
            * 'vigil' for Vigil Network Data
            * 'wade' for Water Data Exchange 2.0 Sites
            * 'WQP' for Water Quality Portal

        fid : str
            The ID of the feature.
        navigation : str
            The navigation method.
        source : str, optional
            Return the data from another source after navigating
            the features using fsource, defaults to None.
        distance : int, optional
            Limit the search for navigation up to a distance in km,
            defaults is 500 km. Note that this is an expensive request so you
            have be mindful of the value that you provide. The value must be
            between 1 to 9999 km.
        trim_start : bool, optional
            If ``True``, trim the starting flowline at the source feature,
            defaults to ``False``.

        Returns
        -------
        geopandas.GeoDataFrame
            NLDI indexed features in EPSG:4326.
        """
        if not (1 <= distance <= 9999):
            raise InputRangeError("distance", "[1, 9999]")

        self._validate_fsource(fsource)

        url = "/".join([self.base_url, "linked-data", fsource, fid, "navigation"])

        valid_navigations: dict[str, Any] = self._get_url(url)  # type: ignore
        if not valid_navigations:
            raise ZeroMatchedError

        if navigation not in valid_navigations:
            raise InputValueError("navigation", list(valid_navigations))

        url = valid_navigations[navigation]

        r_json: list[dict[str, Any]] = self._get_url(url)  # type: ignore
        valid_sources = {s["source"].lower(): s["features"] for s in r_json}
        if source not in valid_sources:
            raise InputValueError("source", list(valid_sources))

        url = valid_sources[source]
        payload = {"distance": f"{round(distance)}", "trimStart": f"{trim_start}".lower()}
        try:
            return geoutils.json2geodf(self._get_url(url, payload), 4269, 4326)
        except EmptyResponseError as ex:
            raise ZeroMatchedError from ex

    def navigate_byloc(
        self,
        coords: tuple[float, float],
        navigation: str | None = None,
        source: str | None = None,
        loc_crs: CRSTYPE = 4326,
        distance: int = 500,
        trim_start: bool = False,
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
            the features based on ``comid``, defaults to None which throws an exception
            if ``comid_only`` is False.
        loc_crs : str, int, or pyproj.CRS, optional
            The spatial reference of the input coordinate, defaults to EPSG:4326.
        distance : int, optional
            Limit the search for navigation up to a distance in km,
            defaults to 500 km. Note that this is an expensive request so you
            have be mindful of the value that you provide. If you want to get
            all the available features you can pass a large distance like 9999999.
        trim_start : bool, optional
            If ``True``, trim the starting flowline at the source feature,
            defaults to ``False``.

        Returns
        -------
        geopandas.GeoDataFrame
            NLDI indexed features in EPSG:4326.
        """
        if not (isinstance(coords, tuple) and len(coords) == 2):
            raise InputTypeError("coods", "tuple of length 2", "(x, y)")
        resp = self.feature_byloc(coords, loc_crs)
        comid_df = resp[0] if isinstance(resp, tuple) else resp
        comid = comid_df.comid.iloc[0]

        if navigation is None or source is None:
            raise MissingItemError(["navigation", "source"])

        return self.navigate_byid("comid", comid, navigation, source, distance, trim_start)


def geoconnex(
    item: str | None = None,
    query: None
    | (
        dict[
            str,
            (str | int | float | tuple[float, float, float, float] | Polygon | MultiPolygon),
        ]
    ) = None,
    skip_geometry: bool = False,
) -> gpd.GeoDataFrame | None:
    """Query the GeoConnex API.

    Notes
    -----
    If you run the function without any arguments, it will print out a list
    of available endpoints. If you run the function with ``item`` but no ``query``,
    it will print out the description, queryable fields, and extent of the
    selected endpoint (``item``).

    Parameters
    ----------
    item: str, optional
        The item to query.
    query: dict, optional
        Query parameters. The ``geometry`` field can be a Polygon, MultiPolygon,
        or tuple/list of length 4 (bbox) in ``EPSG:4326`` CRS.
    skip_geometry: bool, optional
        If ``True``, the geometry will not be returned.

    Returns
    -------
    geopandas.GeoDataFrame
        The data.
    """
    gcx = GeoConnex(item)
    if item is None or query is None:
        print(gcx)
        return None

    return gcx.query(query, skip_geometry)
