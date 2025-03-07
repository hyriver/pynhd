"""Access NLDI and WaterData databases."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal, cast, overload

import pandas as pd
import pyarrow.compute as pc
from yarl import URL

import async_retriever as ar
import pygeoutils as geoutils
import pynhd.nhdplus_derived as derived
from pygeoogc import ServiceURL
from pynhd.exceptions import (
    InputRangeError,
    InputTypeError,
    InputValueError,
    MissingItemError,
    ZeroMatchedError,
)

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

    import geopandas as gpd
    from pyproj import CRS

    CRSType = int | str | CRS

__all__ = ["NLDI"]


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
        self.valid_characteristics = derived.nhdplus_attrs_s3()

    @staticmethod
    def _check_resp(resp: dict[str, Any] | list[dict[str, Any]] | None) -> bool:
        if not resp:
            return False
        if isinstance(resp, dict):
            return (
                resp.get("type") != "error"
                and "error" not in resp
                and (resp.get("features") or "features" not in resp)
            )
        return True

    @overload
    def _get_urls(
        self, url_parts: Generator[tuple[str, ...], None, None] | str, raw: Literal[False] = False
    ) -> gpd.GeoDataFrame: ...

    @overload
    def _get_urls(
        self, url_parts: Generator[tuple[str, ...], None, None] | str, raw: Literal[True]
    ) -> tuple[list[int], list[dict[str, Any]] | dict[str, Any]]: ...

    def _get_urls(
        self, url_parts: Generator[tuple[str, ...], None, None] | str, raw: bool = False
    ) -> tuple[list[int], list[dict[str, Any]] | dict[str, Any]] | gpd.GeoDataFrame:
        """Send a request to the service using GET method."""
        if isinstance(url_parts, str):
            urls = [URL(url_parts).human_repr()]
        else:
            urls = [URL("/".join((self.base_url, *u))).human_repr() for u in url_parts]
        resp = ar.retrieve_json(urls, raise_status=False)

        try:
            index, resp = zip(*((i, r) for i, r in enumerate(resp) if self._check_resp(r)))
        except ValueError as ex:
            raise ZeroMatchedError from ex

        index = cast("list[int]", list(index))
        resp = cast("list[dict[str, Any]] | dict[str, Any]", list(resp))
        failed = [u for i, u in enumerate(urls) if i not in index]

        if failed:
            msg = ". ".join(
                (
                    f"{len(failed)} of {len(urls)} requests failed.",
                    f"The failed URLs are {failed}",
                )
            )
            warnings.warn(msg, UserWarning, stacklevel=2)
            _ = [ar.delete_url_cache(u) for u in failed]

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
        loc_crs: CRSType = 4326,
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
        _coords = geoutils.geometry_reproject(_coords, loc_crs, 4326)
        endpoint = "comid/position" if source == "feature" else "hydrolocation"
        urls = (
            ("linked-data", f"{endpoint}?coords=POINT({lon:.6f} {lat:.6f})") for lon, lat in _coords
        )
        return self._get_urls(urls, False)

    def comid_byloc(
        self,
        coords: tuple[float, float] | list[tuple[float, float]],
        loc_crs: CRSType = 4326,
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
        return comids

    def feature_byloc(
        self,
        coords: tuple[float, float] | list[tuple[float, float]],
        loc_crs: CRSType = 4326,
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
        return basins

    @staticmethod
    def get_characteristics(
        char_list: str | list[str],
        comids: int | list[int] | None = None,
    ) -> pd.DataFrame:
        """Get characteristics using a list ComIDs.

        Parameters
        ----------
        char_list : str or list
            The list of characteristics to get.
        comids : int or list, optional
            The list of ComIDs, defaults to None, i.e., all NHDPlus ComIDs.

        Returns
        -------
        pandas.DataFrame
            The characteristics of the requested ComIDs.
        """
        if comids is not None:
            comids = pd.Series(comids).astype(int).to_list()
        pyarrrow_filter = None if comids is None else pc.field("COMID").isin(comids)
        return derived.nhdplus_attrs_s3(char_list, pyarrrow_filter)

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
        loc_crs: CRSType = 4326,
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
