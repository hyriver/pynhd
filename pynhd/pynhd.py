"""Access NLDI and WaterData databases."""
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import async_retriever as ar
import geopandas as gpd
import numpy as np
import pandas as pd
import pygeoogc as ogc
import pygeoutils as geoutils
from pygeoogc import WFS, InvalidInputValue, ServiceUnavailable, ServiceURL, ZeroMatched
from pygeoutils import InvalidInputType
from shapely.geometry import MultiPolygon, Polygon

from .core import ALT_CRS, DEF_CRS, AGRBase, PyGeoAPIBase, PyGeoAPIBatch, logger
from .exceptions import InvalidInputRange, MissingItems


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
    crs : str, optional
        Target spatial reference, default to ``EPSG:4326``
    """

    def __init__(
        self,
        layer: str,
        outfields: Union[str, List[str]] = "*",
        crs: str = DEF_CRS,
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
            raise InvalidInputValue("layer", list(self.valid_layers))
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
        coord: Tuple[float, float],
        crs: str = DEF_CRS,
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
        >>> pygeoapi = PyGeoAPI()
        >>> gdf = pygeoapi.flow_trace(
        ...     (1774209.63, 856381.68), crs="ESRI:102003", direction="none"
        ... )  # doctest: +SKIP
        >>> print(gdf.comid.iloc[0])  # doctest: +SKIP
        22294818
        """
        lon, lat = self._check_coords(coord, crs)[0]
        url = self._get_url("flowtrace")
        payload = self._request_body([{"lat": lat, "lon": lon, "direction": direction}])
        return self._get_response(url, payload)

    def split_catchment(
        self, coord: Tuple[float, float], crs: str = DEF_CRS, upstream: bool = False
    ) -> gpd.GeoDataFrame:
        """Return a GeoDataFrame from the splitcatchment service.

        Parameters
        ----------
        coord : tuple
            The coordinate of the point to trace as a tuple,e.g., (lon, lat).
        crs : str, optional
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
        >>> pygeoapi = PyGeoAPI()
        >>> gdf = pygeoapi.split_catchment((-73.82705, 43.29139), crs=DEF_CRS, upstream=False)  # doctest: +SKIP
        >>> print(gdf.catchmentID.iloc[0])  # doctest: +SKIP
        22294818
        """
        lon, lat = self._check_coords(coord, crs)[0]
        url = self._get_url("splitcatchment")
        payload = self._request_body([{"lat": lat, "lon": lon, "upstream": upstream}])
        return self._get_response(url, payload)

    def elevation_profile(
        self,
        coords: List[Tuple[float, float]],
        numpts: int,
        dem_res: int,
        crs: str = DEF_CRS,
    ) -> gpd.GeoDataFrame:
        """Return a GeoDataFrame from the xsatendpts service.

        Parameters
        ----------
        coords : list
            A list of two coordinates to trace as a list of tuples, e.g.,
            [(lon1, lat1), (lon2, lat2)].
        numpts : int
            The number of points to extract the elevation profile from the DEM.
        dem_res : int
            The target resolution for requesting the DEM from 3DEP service.
        crs : str, optional
            The coordinate reference system of the coordinates, defaults to EPSG:4326.

        Returns
        -------
        geopandas.GeoDataFrame
            A GeoDataFrame containing the elevation profile along the requested endpoints.

        Examples
        --------
        >>> from pynhd import PyGeoAPI
        >>> pygeoapi = PyGeoAPI()
        >>> gdf = pygeoapi.elevation_profile(
        ...     [(-103.801086, 40.26772), (-103.80097, 40.270568)], numpts=101, dem_res=1, crs=DEF_CRS
        ... )  # doctest: +SKIP
        >>> print(gdf.iloc[-1, 1])  # doctest: +SKIP
        411.5906
        """
        if not isinstance(coords, list) or any(len(c) != 2 for c in coords):
            raise InvalidInputType("coords", "list", "[(lon1, lat1), (lon2, lat2)]")

        lons, lats = zip(*self._check_coords(coords, crs))

        url = self._get_url("xsatendpts")
        payload = self._request_body(
            [{"lat": list(lats), "lon": list(lons), "numpts": numpts, "3dep_res": dem_res}]
        )
        return self._get_response(url, payload)

    def cross_section(
        self, coord: Tuple[float, float], width: float, numpts: int, crs: str = DEF_CRS
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
        crs : str, optional
            The coordinate reference system of the coordinates, defaults to EPSG:4326.

        Returns
        -------
        geopandas.GeoDataFrame
            A GeoDataFrame containing the cross-section at the requested point.

        Examples
        --------
        >>> from pynhd import PyGeoAPI
        >>> pygeoapi = PyGeoAPI()
        >>> gdf = pygeoapi.cross_section((-103.80119, 40.2684), width=1000.0, numpts=101, crs=DEF_CRS)  # doctest: +SKIP
        >>> print(gdf.iloc[-1, 1])  # doctest: +SKIP
        1000.0
        """
        lon, lat = self._check_coords(coord, crs)[0]
        url = self._get_url("xsatpoint")
        payload = self._request_body([{"lat": lat, "lon": lon, "width": width, "numpts": numpts}])
        return self._get_response(url, payload)


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
    >>> from pynhd import PyGeoAPI
    >>> pygeoapi = PyGeoAPI()
    >>> gdf = pygeoapi.flow_trace(
    ...     (1774209.63, 856381.68), crs="ESRI:102003", direction="none"
    ... )  # doctest: +SKIP
    >>> print(gdf.comid.iloc[0])  # doctest: +SKIP
    22294818
    """
    pgab = PyGeoAPIBatch(coords)
    url = pgab._get_url(pgab.service[service])
    payload = pgab.get_payload(service)
    gdf = pgab._get_response(url, payload)
    if service == "flow_trace":
        gdf[["comid", "reachcode"]] = gdf[["comid", "reachcode"]].astype("Int64")
        feat = gdf[~gdf.comid.isna()].set_index("req_idx")
        raindrop = gdf.loc[gdf.comid.isna(), ["req_idx", "geometry"]].set_index("req_idx")
        feat["raindrop_path"] = raindrop.geometry
        return feat.reset_index()
    return gdf


class WaterData:
    """Access to `Water Data <https://labs.waterdata.usgs.gov/geoserver>`__ service.

    Parameters
    ----------
    layer : str
        A valid layer from the WaterData service. Valid layers are:
        ``nhdarea``, ``nhdwaterbody``, ``catchmentsp``, ``nhdflowline_network``
        ``gagesii``, ``huc08``, ``huc12``, ``huc12agg``, and ``huc12all``. Note that
        the layers' worksapce for the Water Data service is ``wmadata`` which will
        be added to the given ``layer`` argument if it is not provided.
    crs : str, optional
        The target spatial reference system, defaults to ``epsg:4326``.
    """

    def __init__(
        self,
        layer: str,
        crs: str = DEF_CRS,
    ) -> None:
        self.layer = layer if ":" in layer else f"wmadata:{layer}"
        self.crs = crs
        self.wfs = WFS(
            ServiceURL().wfs.waterdata,
            layer=self.layer,
            outformat="application/json",
            version="2.0.0",
            crs=ALT_CRS,
        )

    def _to_geodf(self, resp: Union[List[Dict[str, Any]], Dict[str, Any]]) -> gpd.GeoDataFrame:
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
        features = geoutils.json2geodf(resp, ALT_CRS, self.crs)
        if features.empty:
            raise ZeroMatched
        return features

    def bybox(
        self, bbox: Tuple[float, float, float, float], box_crs: str = DEF_CRS
    ) -> gpd.GeoDataFrame:
        """Get features within a bounding box."""
        resp: Dict[str, Any] = self.wfs.getfeature_bybox(bbox, box_crs, always_xy=True)  # type: ignore
        return self._to_geodf(resp)

    def bygeom(
        self,
        geometry: Union[Polygon, MultiPolygon],
        geo_crs: str = DEF_CRS,
        xy: bool = True,
        predicate: str = "INTERSECTS",
    ) -> gpd.GeoDataFrame:
        """Get features within a geometry.

        Parameters
        ----------
        geometry : shapely.geometry
            The input geometry
        geo_crs : str, optional
            The CRS of the input geometry, default to epsg:4326.
        xy : bool, optional
            Whether axis order of the input geometry is xy or yx.
        predicate : str, optional
            The geometric prediacte to use for requesting the data, defaults to
            INTERSECTS. Valid predicates are:
            ``EQUALS``, ``DISJOINT``, ``INTERSECTS``, ``TOUCHES``, ``CROSSES``, ``WITHIN``
            ``CONTAINS``, ``OVERLAPS``, ``RELATE``, ``BEYOND``

        Returns
        -------
        geopandas.GeoDataFrame
            The requested features in the given geometry.
        """
        resp: Dict[str, Any] = self.wfs.getfeature_bygeom(  # type: ignore
            geometry, geo_crs, always_xy=not xy, predicate=predicate
        )
        return self._to_geodf(resp)

    def bydistance(
        self, coords: Tuple[float, float], distance: int, loc_crs: str = DEF_CRS
    ) -> gpd.GeoDataFrame:
        """Get features within a radius (in meters) of a point."""
        if not (isinstance(coords, tuple) and len(coords) == 2):
            raise InvalidInputType("coods", "tuple of length 2", "(x, y)")

        x, y = ogc.utils.match_crs([coords], loc_crs, ALT_CRS)[0]
        cql_filter = f"DWITHIN(the_geom,POINT({y:.6f} {x:.6f}),{distance},meters)"
        resp: Dict[str, Any] = self.wfs.getfeature_byfilter(cql_filter, "GET")  # type: ignore
        return self._to_geodf(resp)

    def byid(self, featurename: str, featureids: Union[List[str], str]) -> gpd.GeoDataFrame:
        """Get features based on IDs."""
        resp: Dict[str, Any] = self.wfs.getfeature_byid(featurename, featureids)  # type: ignore
        features = self._to_geodf(resp)

        fids = [str(f) for f in featureids] if isinstance(featureids, list) else [str(featureids)]
        missing = set(fids).difference(set(features[featurename].astype(str)))
        if missing:
            verb = "ID was" if len(missing) == 1 else "IDs were"
            logger.warning(
                f"The following requested feature {verb} not found in WaterData:\n"
                + ", ".join(missing)
            )
        return features

    def byfilter(self, cql_filter: str, method: str = "GET") -> gpd.GeoDataFrame:
        """Get features based on a CQL filter."""
        resp: Dict[str, Any] = self.wfs.getfeature_byfilter(cql_filter, method)  # type: ignore
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
    For more info visit: https://edits.nationalmap.gov/arcgis/rest/services/nhd/MapServer

    Parameters
    ----------
    layer : str, optional
        A valid service layer. Valid layers are:

        - ``point``
        - ``sink``
        - ``flowline``
        - ``non_network_flowline``
        - ``flow_direction``
        - ``line``
        - ``wall``
        - ``burn_line``
        - ``burn_waterbody``
        - ``area``
        - ``waterbody``
        - ``huc12``
        - ``catchment``

    outfields : str or list, optional
        Target field name(s), default to "*" i.e., all the fields.
    crs : str, optional
        Target spatial reference, default to ``EPSG:4326``
    """

    def __init__(
        self,
        layer: str,
        outfields: Union[str, List[str]] = "*",
        crs: str = DEF_CRS,
    ):
        self.valid_layers = {
            "point": "NHDPoint",
            "sink": "NHDPlusSink",
            "flowline": "NetworkNHDFlowline",
            "non_network_flowline": "NonNetworkNHDFlowline",
            "flow_direction": "FlowDirection",
            "line": "NHDLine",
            "wall": "NHDPlusWall",
            "burn_line": "NHDPlusBurnLineEvent",
            "burn_waterbody": "NHDPlusBurnWaterbody",
            "area": "NHDArea",
            "waterbody": "NHDWaterbody",
            "huc12": "WBDHU12",
            "catchment": "NHDPlusCatchment",
        }
        _layer = self.valid_layers.get(layer)
        if _layer is None:
            raise InvalidInputValue("layer", list(self.valid_layers))
        super().__init__(
            ServiceURL().restful.nhdplushr_edits,
            _layer,
            outfields,
            crs,
        )


class NLDI:
    """Access the Hydro Network-Linked Data Index (NLDI) service."""

    def _get_url(
        self, url: str, payload: Optional[Dict[str, str]] = None
    ) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """Send a request to the service using GET method."""
        if payload is None:
            payload = {"f": "json"}
        else:
            payload.update({"f": "json"})

        try:
            resp = ar.retrieve_json([url], [{"params": payload}])
        except ar.ServiceError as ex:
            raise ZeroMatched from ex
        except ConnectionError as ex:
            raise ServiceUnavailable(self.base_url) from ex
        else:
            return resp[0]

    def __init__(self) -> None:
        self.base_url = ServiceURL().restful.nldi

        resp = self._get_url("/".join([self.base_url, "linked-data"]))
        self.valid_fsources = {r["source"]: r["sourceName"] for r in resp}  # type: ignore

        resp = self._get_url("/".join([self.base_url, "lookups"]))
        self.valid_chartypes = {r["type"]: r["typeName"] for r in resp}  # type: ignore

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
            raise InvalidInputValue("fsource", valids)

    def _get_urls(
        self, urls: Mapping[Any, Tuple[str, Optional[Dict[str, str]]]]
    ) -> Tuple[gpd.GeoDataFrame, List[str]]:
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
                resp.append((f, geoutils.json2geodf(rjson, ALT_CRS, DEF_CRS)))
            except (ZeroMatched, InvalidInputType, ar.ServiceError):
                not_found.append(f)

        if len(resp) == 0:
            raise ZeroMatched

        resp_df = gpd.GeoDataFrame(pd.concat(dict(resp)), crs=DEF_CRS)

        return resp_df, not_found

    def getfeature_byid(
        self, fsource: str, fid: Union[str, List[str]]
    ) -> Union[gpd.GeoDataFrame, Tuple[gpd.GeoDataFrame, List[str]]]:
        """Get feature(s) based ID(s).

        Parameters
        ----------
        fsource : str
            The name of feature(s) source. The valid sources are:
            comid, huc12pp, nwissite, wade, wqp
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

        if len(not_found) > 0:
            self._missing_warning(len(not_found), len(fid))
            return features, not_found

        return features

    def __byloc(
        self,
        source: str,
        coords: Union[Tuple[float, float], List[Tuple[float, float]]],
        loc_crs: str = DEF_CRS,
    ) -> Union[gpd.GeoDataFrame, Tuple[gpd.GeoDataFrame, List[Tuple[float, float]]]]:
        """Get the closest feature ID(s) based on coordinates.

        Parameters
        ----------
        source : str
            The name of feature(s) source. The valid sources are ``comid`` and ``feature``.
        coords : tuple or list
            A tuple of length two (x, y) or a list of them.
        loc_crs : str, optional
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
            raise InvalidInputType("coords", "list or tuple")

        _coords = ogc.utils.match_crs(_coords, loc_crs, DEF_CRS)

        urls = {
            f"{(lon, lat)}": (base_url, {"coords": f"POINT({lon} {lat})"}) for lon, lat in _coords
        }
        comids, not_found = self._get_urls(urls)

        if len(comids) == 0:
            raise ZeroMatched

        comids = comids.reset_index(drop=True)

        if len(not_found) > 0:
            self._missing_warning(len(not_found), len(_coords))
            return comids, not_found

        return comids

    def comid_byloc(
        self,
        coords: Union[Tuple[float, float], List[Tuple[float, float]]],
        loc_crs: str = DEF_CRS,
    ) -> Union[gpd.GeoDataFrame, Tuple[gpd.GeoDataFrame, List[Tuple[float, float]]]]:
        """Get the closest ComID based on coordinates.

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
        loc_crs : str, optional
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
        coords: Union[Tuple[float, float], List[Tuple[float, float]]],
        loc_crs: str = DEF_CRS,
    ) -> Union[gpd.GeoDataFrame, Tuple[gpd.GeoDataFrame, List[Tuple[float, float]]]]:
        """Get the closest feature ID(s) based on coordinates.

        Parameters
        ----------
        coords : tuple or list
            A tuple of length two (x, y) or a list of them.
        loc_crs : str, optional
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
        station_ids: Union[str, List[str]],
        split_catchment: bool = False,
        simplified: bool = True,
    ) -> Union[gpd.GeoDataFrame, Tuple[gpd.GeoDataFrame, List[str]]]:
        """Get basins for a list of station IDs.

        Parameters
        ----------
        station_ids : str or list
            USGS station ID(s).
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
        station_ids = station_ids if isinstance(station_ids, list) else [station_ids]
        payload = {
            "splitCatchment": str(split_catchment).lower(),
            "simplified": str(simplified).lower(),
        }
        urls = {
            s: (f"{self.base_url}/linked-data/nwissite/USGS-{s}/basin", payload)
            for s in station_ids
        }
        basins, not_found = self._get_urls(urls)
        basins = basins.reset_index(level=1, drop=True)
        basins.index.rename("identifier", inplace=True)
        nulls = basins.geometry.isnull()
        not_found += basins[nulls].index.to_list()
        basins = basins[~nulls].copy()

        if len(not_found) > 0:
            self._missing_warning(len(not_found), len(station_ids))
            return basins, not_found

        return basins

    def getcharacteristic_byid(
        self,
        comids: Union[List[str], str],
        char_type: str,
        char_ids: Union[str, List[str]] = "all",
        values_only: bool = True,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """Get characteristics using a list ComIDs.

        Parameters
        ----------
        comids : str or list
            The ID of the feature.
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
            raise InvalidInputValue("char", valids)

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
                raise InvalidInputValue("char_id", [f'"{s}" for {d}' for s, d in vids.items()])
            payload = {"characteristicId": ",".join(_char_ids)}

        for comid in comids:
            url = "/".join([self.base_url, "linked-data", "comid", f"{comid}", char_type])
            rjson: Dict[str, Any] = self._get_url(url, payload)  # type: ignore
            char = pd.DataFrame.from_dict(rjson["characteristics"], orient="columns").T
            char.columns = char.iloc[0]
            char = char.drop(index="characteristic_id")

            v_dict[comid] = char.loc["characteristic_value"]
            if values_only:
                continue

            nd_dict[comid] = char.loc["percent_nodata"]

        def todf(df_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
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
            raise InvalidInputValue("char", list(self.valid_chartypes))

        resp = self._get_url("/".join([self.base_url, "lookups", char_type, "characteristics"]))
        c_list = ogc.utils.traverse_json(resp, ["characteristicMetadata", "characteristic"])
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
            The name of feature source. The valid sources are:
            ``comid``, ``huc12pp``, ``nwissite``, ``wade``, ``WQP``.
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
            raise InvalidInputRange("distance", "[1, 9999]")

        self._validate_fsource(fsource)

        url = "/".join([self.base_url, "linked-data", fsource, fid, "navigation"])

        valid_navigations: Dict[str, Any] = self._get_url(url)  # type: ignore
        if navigation not in valid_navigations.keys():
            raise InvalidInputValue("navigation", list(valid_navigations.keys()))

        url = valid_navigations[navigation]

        r_json = self._get_url(url)
        valid_sources = {s["source"].lower(): s["features"] for s in r_json}  # type: ignore
        if source not in valid_sources:
            raise InvalidInputValue("source", list(valid_sources.keys()))

        url = valid_sources[source]
        payload = {"distance": f"{int(distance)}", "trimStart": f"{trim_start}".lower()}

        return geoutils.json2geodf(self._get_url(url, payload), ALT_CRS, DEF_CRS)

    def navigate_byloc(
        self,
        coords: Tuple[float, float],
        navigation: Optional[str] = None,
        source: Optional[str] = None,
        loc_crs: str = DEF_CRS,
        distance: int = 500,
        trim_start: bool = False,
    ) -> gpd.GeoDataFrame:
        """Navigate the NHDPlus database from a coordinate.

        Parameters
        ----------
        coords : tuple
            A tuple of length two (x, y).
        navigation : str, optional
            The navigation method, defaults to None which throws an exception
            if comid_only is False.
        source : str, optional
            Return the data from another source after navigating
            the features using fsource, defaults to None which throws an exception
            if comid_only is False.
        loc_crs : str, optional
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
            raise InvalidInputType("coods", "tuple of length 2", "(x, y)")
        resp = self.feature_byloc(coords, loc_crs)
        comid_df = resp[0] if isinstance(resp, tuple) else resp
        comid = comid_df.comid.iloc[0]

        if navigation is None or source is None:
            raise MissingItems(["navigation", "source"])

        return self.navigate_byid("comid", comid, navigation, source, distance, trim_start)
