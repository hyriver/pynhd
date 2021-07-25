"""Access NLDI and WaterData databases."""
import io
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import async_retriever as ar
import cytoolz as tlz
import geopandas as gpd
import numpy as np
import pandas as pd
import pygeoogc as ogc
import pygeoutils as geoutils
from pygeoogc import (
    WFS,
    ArcGISRESTful,
    InvalidInputValue,
    ServiceError,
    ServiceUnavailable,
    ServiceURL,
    ZeroMatched,
)
from pygeoutils import InvalidInputType
from requests import Response
from shapely.geometry import MultiPolygon, Polygon
from simplejson import JSONDecodeError

from .exceptions import InvalidInputRange, MissingItems

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter(""))
logger.handlers = [handler]
logger.propagate = False
DEF_CRS = "epsg:4326"
ALT_CRS = "epsg:4269"


def nhdplus_vaa(parquet_name: Optional[Union[Path, str]] = None) -> pd.DataFrame:
    """Get NHDPlus Value Added Attributes with ComID-level roughness and slope values.

    Notes
    -----
    This downloads a 200 MB ``parquet`` file from
    `here <https://www.hydroshare.org/resource/6092c8a62fac45be97a09bfd0b0bf726>`__ .
    Although this dataframe does not include geometry, it can be linked to other geospatial
    NHDPlus dataframes through ComIDs.

    Parameters
    ----------
    parquet_name : str or Path
        Path to a file with ``.parquet`` extension for saving the processed to disk for
        later use.

    Returns
    -------
    pandas.DataFrame
        A dataframe that includes ComID-level attributes for 2.7 million NHDPlus flowlines.

    Examples
    --------
    >>> vaa = nhdplus_vaa()
    >>> print(vaa.slope.max())
    4.6
    """
    if parquet_name is None:
        output = Path("cache", "nhdplus_vaa.parquet")
    else:
        if ".parquet" not in str(parquet_name):
            raise InvalidInputValue("parquet_name", ["a filename with `.parquet` extension."])

        output = Path(parquet_name)
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists():
        return pd.read_parquet(output)

    dtypes = {
        "comid": "Int64",
        "streamleve": "Int8",
        "streamorde": "Int8",
        "streamcalc": "Int8",
        "fromnode": "Int64",
        "tonode": "Int64",
        "hydroseq": "Int64",
        "levelpathi": "Int64",
        "pathlength": "Int64",
        "terminalpa": "Int64",
        "arbolatesu": "Int64",
        "divergence": "Int8",
        "startflag": "Int8",
        "terminalfl": "Int8",
        "dnlevel": "Int16",
        "thinnercod": "Int8",
        "uplevelpat": "Int64",
        "uphydroseq": "Int64",
        "dnlevelpat": "Int64",
        "dnminorhyd": "Int64",
        "dndraincou": "Int64",
        "dnhydroseq": "Int64",
        "frommeas": "Int32",
        "tomeas": "Int32",
        "reachcode": "Int64",
        "lengthkm": "f8",
        "fcode": "Int32",
        "vpuin": "Int8",
        "vpuout": "Int8",
        "areasqkm": "f8",
        "totdasqkm": "f8",
        "divdasqkm": "f8",
        "totma": "f8",
        "wbareatype": str,
        "pathtimema": "f8",
        "slope": "f8",
        "slopelenkm": "f8",
        "ftype": str,
        "gnis_name": str,
        "gnis_id": str,
        "wbareacomi": "Int64",
        "hwnodesqkm": "f8",
        "rpuid": str,
        "vpuid": str,
        "roughness": "f8",
    }

    rid = "6092c8a62fac45be97a09bfd0b0bf726"
    fpath = "data/contents/nhdplusVAA.parquet"
    url = f"https://www.hydroshare.org/hsapi/resource/{rid}/files/{fpath}"

    resp = ar.retrieve([url], "binary")

    vaa = pd.read_parquet(io.BytesIO(resp[0]))
    vaa = vaa.astype(dtypes, errors="ignore")
    vaa.to_parquet(output)
    return vaa


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

    def bybox(
        self, bbox: Tuple[float, float, float, float], box_crs: str = DEF_CRS
    ) -> gpd.GeoDataFrame:
        """Get features within a bounding box."""
        resp = self.wfs.getfeature_bybox(bbox, box_crs, always_xy=True)
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
        resp = self.wfs.getfeature_bygeom(geometry, geo_crs, always_xy=not xy, predicate=predicate)
        return self._to_geodf(resp)

    def bydistance(
        self, coords: Tuple[float, float], distance: int, loc_crs: str = DEF_CRS
    ) -> gpd.GeoDataFrame:
        """Get features within a radius (in meters) of a point."""
        if not (isinstance(coords, tuple) and len(coords) == 2):
            raise InvalidInputType("coods", "tuple of length 2", "(x, y)")

        x, y = ogc.utils.match_crs([coords], loc_crs, ALT_CRS)[0]
        cql_filter = f"DWITHIN(the_geom,POINT({y:.6f} {x:.6f}),{distance},meters)"
        resp = self.wfs.getfeature_byfilter(cql_filter, "GET")
        return self._to_geodf(resp)

    def byid(self, featurename: str, featureids: Union[List[str], str]) -> gpd.GeoDataFrame:
        """Get features based on IDs."""
        resp = self.wfs.getfeature_byid(featurename, featureids)
        return self._to_geodf(resp)

    def byfilter(self, cql_filter: str, method: str = "GET") -> gpd.GeoDataFrame:
        """Get features based on a CQL filter."""
        resp = self.wfs.getfeature_byfilter(cql_filter, method)
        return self._to_geodf(resp)

    def _to_geodf(self, resp: Response) -> gpd.GeoDataFrame:
        """Convert a response from WaterData to a GeoDataFrame.

        Parameters
        ----------
        resp : Response
            A response from a WaterData request.

        Returns
        -------
        geopandas.GeoDataFrame
            The requested features in a GeoDataFrames.
        """
        return geoutils.json2geodf(resp, ALT_CRS, self.crs)


@dataclass
class AGRBase:
    """Base class for accessing NHD(Plus) HR database through the National Map ArcGISRESTful.

    Parameters
    ----------
    layer : str, optional
        A valid service layer. To see a list of available layers instantiate the class
        without passing any argument.
    outfields : str or list, optional
        Target field name(s), default to "*" i.e., all the fields.
    crs : str, optional
        Target spatial reference, default to EPSG:4326
    """

    layer: Optional[str] = None
    outfields: Union[str, List[str]] = "*"
    crs: str = DEF_CRS

    @property
    def service(self) -> ArcGISRESTful:
        """Connect to a RESTFul service."""
        return self._service

    @service.setter
    def service(self, value: str) -> None:
        valid_layers = self.get_validlayers(value)
        if self.layer is None:
            raise InvalidInputType("layer", "str")

        if self.layer.lower() not in valid_layers:
            raise InvalidInputValue("layer", list(valid_layers))

        self._service = ArcGISRESTful(
            value,
            valid_layers[self.layer.lower()],
            outformat="json",
            outfields=self.outfields,
            crs=self.crs,
        )

    @staticmethod
    def get_validlayers(url):
        """Get valid layer for a ArcGISREST service."""
        try:
            rjson = ar.retrieve([url], "json", [{"params": {"f": "json"}}])[0]
        except (JSONDecodeError, KeyError) as ex:
            raise ServiceError(url) from ex
        else:
            return {lyr["name"].lower(): lyr["id"] for lyr in rjson["layers"]}

    def connect_to(self, service: str, service_list: Dict[str, str], auto_switch: bool) -> None:
        """Connect to a web service.

        Parameters
        ----------
        service : str
            Name of the preferred web service to connect to from the list provided in service_list.
        service_list: dict
            A dict where keys are names of the web services and values are their URLs.
        auto_switch : bool, optional
            Automatically switch to other services' URL if the first one doesn't work, default to False.
        """
        if service not in service_list:
            raise InvalidInputValue("service", list(service_list))

        url = service_list.pop(service)
        try:
            self.service = url
        except (ServiceError, ConnectionError) as ex:
            if not auto_switch:
                raise ServiceError(url) from ex

            while len(service_list) > 0:
                next_service = next(iter(service_list.keys()))
                logger.warning(f"Connection to {url} failed. Will try {next_service} next ...")
                try:
                    url = service_list.pop(next_service)
                    self.service = url
                    logger.info(f"Connected to {url}.")
                except (ServiceError, ConnectionError):
                    continue

        if self.service is None:
            raise ServiceError(url)

    def bygeom(
        self,
        geom: Union[Polygon, List[Tuple[float, float]], Tuple[float, float, float, float]],
        geo_crs: str = DEF_CRS,
        sql_clause: str = "",
        distance: Optional[int] = None,
        return_m: bool = False,
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

        Returns
        -------
        geopandas.GeoDataFrame
            The requested features as a GeoDataFrame.
        """
        self.service.oids_bygeom(geom, geo_crs=geo_crs, sql_clause=sql_clause, distance=distance)
        return self._getfeatures(return_m)

    def byids(
        self, field: str, fids: Union[str, List[str]], return_m: bool = False
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

        Returns
        -------
        geopandas.GeoDataFrame
            The requested features as a GeoDataFrame.
        """
        self.service.oids_byfield(field, fids)
        return self._getfeatures(return_m)

    def bysql(self, sql_clause: str, return_m: bool = False) -> gpd.GeoDataFrame:
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
            Whether to activate the Return M (measure) in the request, defaults to False.

        Returns
        -------
        geopandas.GeoDataFrame
            The requested features as a GeoDataFrame.
        """
        self.service.oids_bysql(sql_clause)
        return self._getfeatures(return_m)

    def _getfeatures(self, return_m: bool = False) -> gpd.GeoDataFrame:
        """Send a request for getting data based on object IDs.

        Parameters
        ----------
        return_m : bool
            Whether to activate the Return M (measure) in the request, defaults to False.

        Returns
        -------
        geopandas.GeoDataFrame
            The requested features as a GeoDataFrame.
        """
        return geoutils.json2geodf(self.service.get_features(return_m))


class NHDPlusHR(AGRBase):
    """Access NHDPlus HR database through the National Map ArcGISRESTful.

    Parameters
    ----------
    layer : str, optional
        A valid service layer. To see a list of available layers instantiate the class
        without passing any argument like so ``NHDPlusHR()``.
    outfields : str or list, optional
        Target field name(s), default to "*" i.e., all the fields.
    crs : str, optional
        Target spatial reference, default to EPSG:4326
    service : str, optional
        Name of the web service to use, defaults to hydro. Supported web services are:

        * hydro: https://hydro.nationalmap.gov/arcgis/rest/services/NHDPlus_HR/MapServer
        * edits: https://edits.nationalmap.gov/arcgis/rest/services/NHDPlus_HR/NHDPlus_HR/MapServer

    auto_switch : bool, optional
        Automatically switch to other services' URL if the first one doesn't work, default to False.
    """

    def __init__(
        self,
        layer: Optional[str] = None,
        outfields: Union[str, List[str]] = "*",
        crs: str = DEF_CRS,
        service: str = "hydro",
        auto_switch: bool = False,
    ):
        super().__init__(layer, outfields, crs)
        service_list = {
            "hydro": ServiceURL().restful.nhdplushr,
            "edits": ServiceURL().restful.nhdplushr_edits,
        }
        if layer is None:
            raise InvalidInputValue("layer", list(self.get_validlayers(service_list[service])))
        self.connect_to(service, service_list, auto_switch)


class NLDI:
    """Access the Hydro Network-Linked Data Index (NLDI) service."""

    def __init__(self) -> None:
        self.base_url = ServiceURL().restful.nldi

        resp = ar.retrieve(["/".join([self.base_url, "linked-data"])], "json")[0]
        self.valid_fsources = {r["source"]: r["sourceName"] for r in resp}

        resp = ar.retrieve(["/".join([self.base_url, "lookups"])], "json")[0]
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

    def getfeature_byid(
        self, fsource: str, fid: Union[str, List[str]]
    ) -> Union[gpd.GeoDataFrame, Tuple[gpd.GeoDataFrame, List[str]]]:
        """Get feature(s) based ID(s).

        Parameters
        ----------
        fsource : str
            The name of feature(s) source. The valid sources are:
            comid, huc12pp, nwissite, wade, wqp
        fid : str or list
            Feature ID(s).

        Returns
        -------
        geopandas.GeoDataFrame or (geopandas.GeoDataFrame, list)
            NLDI indexed features in EPSG:4326. If some IDs don't return any features
            a list of missing ID(s) are returned as well.
        """
        self._validate_fsource(fsource)
        fid = fid if isinstance(fid, list) else [fid]
        urls = {f: "/".join([self.base_url, "linked-data", fsource, f]) for f in fid}
        features, not_found = self._get_urls(urls)

        if len(not_found) > 0:
            self._missing_warning(len(not_found), len(fid))
            return features, not_found

        return features

    def comid_byloc(
        self,
        coords: Union[Tuple[float, float], List[Tuple[float, float]]],
        loc_crs: str = DEF_CRS,
    ) -> Union[gpd.GeoDataFrame, Tuple[gpd.GeoDataFrame, List[Tuple[float, float]]]]:
        """Get the closest ComID(s) based on coordinates.

        Parameters
        ----------
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
        coords_list = coords if isinstance(coords, list) else [coords]
        coords_list = ogc.utils.match_crs(coords_list, loc_crs, DEF_CRS)

        base_url = "/".join([self.base_url, "linked-data", "comid", "position"])
        urls = {(lon, lat): f"{base_url}?coords=POINT({lon} {lat})" for lon, lat in coords_list}
        comids, not_found = self._get_urls(urls)
        comids = comids.reset_index(level=2, drop=True)

        if len(not_found) > 0:
            self._missing_warning(len(not_found), len(coords_list))
            return comids, not_found

        return comids

    def get_basins(
        self, station_ids: Union[str, List[str]]
    ) -> Union[gpd.GeoDataFrame, Tuple[gpd.GeoDataFrame, List[str]]]:
        """Get basins for a list of station IDs.

        Parameters
        ----------
        station_ids : str or list
            USGS station ID(s).

        Returns
        -------
        geopandas.GeoDataFrame or (geopandas.GeoDataFrame, list)
            NLDI indexed basins in EPSG:4326. If some IDs don't return any features
            a list of missing ID(s) are returned as well.
        """
        station_ids = station_ids if isinstance(station_ids, list) else [station_ids]
        urls = {s: f"{self.base_url}/linked-data/nwissite/USGS-{s}/basin" for s in station_ids}
        basins, not_found = self._get_urls(urls)
        basins = basins.reset_index(level=1, drop=True)
        basins.index.rename("identifier", inplace=True)

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
            rjson = self._get_url(url, payload)
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
        resp = ar.retrieve(
            ["/".join([self.base_url, "lookups", char_type, "characteristics"])], "json"
        )
        c_list = ogc.utils.traverse_json(resp[0], ["characteristicMetadata", "characteristic"])
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

        Returns
        -------
        geopandas.GeoDataFrame
            NLDI indexed features in EPSG:4326.
        """
        if not (1 <= distance <= 9999):
            raise InvalidInputRange("distance", "[1, 9999]")

        self._validate_fsource(fsource)

        url = "/".join([self.base_url, "linked-data", fsource, fid, "navigation"])

        valid_navigations = self._get_url(url)
        if navigation not in valid_navigations.keys():
            raise InvalidInputValue("navigation", list(valid_navigations.keys()))

        url = valid_navigations[navigation]

        r_json = self._get_url(url)
        valid_sources = {s["source"].lower(): s["features"] for s in r_json}  # type: ignore
        if source not in valid_sources:
            raise InvalidInputValue("source", list(valid_sources.keys()))

        url = f"{valid_sources[source]}?distance={int(distance)}"

        return geoutils.json2geodf(self._get_url(url), ALT_CRS, DEF_CRS)

    def navigate_byloc(
        self,
        coords: Tuple[float, float],
        navigation: Optional[str] = None,
        source: Optional[str] = None,
        loc_crs: str = DEF_CRS,
        distance: int = 500,
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

        Returns
        -------
        geopandas.GeoDataFrame
            NLDI indexed features in EPSG:4326.
        """
        if not (isinstance(coords, tuple) and len(coords) == 2):
            raise InvalidInputType("coods", "tuple of length 2", "(x, y)")

        lon, lat = ogc.utils.match_crs([coords], loc_crs, DEF_CRS)[0]

        url = "/".join([self.base_url, "linked-data", "comid", "position"])
        payload = {"coords": f"POINT({lon} {lat})"}
        rjson = self._get_url(url, payload)
        comid = geoutils.json2geodf(rjson, ALT_CRS, DEF_CRS).comid.iloc[0]

        if navigation is None or source is None:
            raise MissingItems(["navigation", "source"])

        return self.navigate_byid("comid", comid, navigation, source, distance)

    def _validate_fsource(self, fsource: str) -> None:
        """Check if the given feature source is valid."""
        if fsource not in self.valid_fsources:
            valids = [f'"{s}" for {d}' for s, d in self.valid_fsources.items()]
            raise InvalidInputValue("fsource", valids)

    def _get_urls(self, urls: Dict[Any, str]) -> Tuple[gpd.GeoDataFrame, List[str]]:
        """Get basins for a list of station IDs.

        Parameters
        ----------
        urls : dict
            A dict with keys as feature ids and values as corresponding url.

        Returns
        -------
        (geopandas.GeoDataFrame, list)
            NLDI indexed features in EPSG:4326 and list of ID(s) that no feature was found.
        """
        not_found = []
        resp = []
        for f, u in urls.items():
            try:
                rjson = self._get_url(u)
                resp.append((f, geoutils.json2geodf(rjson, ALT_CRS, DEF_CRS)))
            except (ZeroMatched, InvalidInputType, ar.ServiceError):
                not_found.append(f)

        if len(resp) == 0:
            raise ZeroMatched

        resp_df = gpd.GeoDataFrame(pd.concat(dict(resp)))

        return resp_df, not_found

    def _get_url(self, url: str, payload: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Send a request to the service using GET method."""
        if payload is None:
            payload = {"f": "json"}
        else:
            payload.update({"f": "json"})

        try:
            return ar.retrieve([url], "json", [{"params": payload}])[0]
        except JSONDecodeError as ex:
            raise ZeroMatched from ex
        except ConnectionError as ex:
            raise ServiceUnavailable(self.base_url) from ex


class ScienceBase:
    """Access NHDPlus V2.1 Attributes from ScienceBase over CONUS.

    More info can be found `here <https://www.sciencebase.gov/catalog/item/5669a79ee4b08895842a1d47>`_.

    Parameters
    ----------
    save_dir : str
        Directory to save the staged data frame containing metadata for the database,
        defaults to system's temp directory. The metadata dataframe is saved as a feather
        file, nhdplus_attrs.feather, in save_dir that can be loaded with Pandas.
    """

    def __init__(self, save_dir: Optional[str] = None) -> None:
        self.save_dir = Path(save_dir) if save_dir else Path("cache")
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.nhd_attr_item = "5669a79ee4b08895842a1d47"
        self.char_feather = Path(self.save_dir, "nhdplus_attrs.feather")

    @staticmethod
    def get_children(item: str) -> Dict[str, Any]:
        """Get children items of an item."""
        url = "https://www.sciencebase.gov/catalog/items"
        payload = {
            "filter": f"parentIdExcludingLinks={item}",
            "fields": "title,id",
            "format": "json",
        }
        return ar.retrieve([url], "json", [{"params": payload}])[0]

    @staticmethod
    def get_files(item: str) -> Dict[str, Tuple[str, str]]:
        """Get all the available zip files in an item."""
        url = "https://www.sciencebase.gov/catalog/item"
        payload = {"fields": "files,downloadUri", "format": "json"}
        r = ar.retrieve([f"{url}/{item}"], "json", [{"params": payload}])[0]
        files_url = zip(tlz.pluck("name", r["files"]), tlz.pluck("url", r["files"]))
        meta = list(tlz.pluck("metadataHtmlViewUri", r["files"], default=""))[-1]
        return {f.replace("_CONUS.zip", ""): (u, meta) for f, u in files_url if ".zip" in f}

    def stage_data(self) -> pd.DataFrame:
        """Stage the NHDPlus Attributes database and save to nhdplus_attrs.feather."""
        r = self.get_children(self.nhd_attr_item)

        titles = tlz.pluck("title", r["items"])
        titles = tlz.concat(tlz.map(tlz.partial(re.findall, "Select(.*?)Attributes"), titles))
        titles = tlz.map(str.strip, titles)

        main_items = dict(zip(titles, tlz.pluck("id", r["items"])))

        files = {}
        soil = main_items.pop("Soil")
        for i, item in main_items.items():
            r = self.get_children(item)

            titles = tlz.pluck("title", r["items"])
            titles = tlz.map(lambda s: s.split(":")[1].strip() if ":" in s else s, titles)

            child_items = dict(zip(titles, tlz.pluck("id", r["items"])))
            files[i] = {t: self.get_files(c) for t, c in child_items.items()}

        r = self.get_children(soil)
        titles = tlz.pluck("title", r["items"])
        titles = tlz.map(lambda s: s.split(":")[1].strip() if ":" in s else s, titles)

        child_items = dict(zip(titles, tlz.pluck("id", r["items"])))
        stat = child_items.pop("STATSGO Soil Characteristics")
        ssur = child_items.pop("SSURGO Soil Characteristics")
        files["Soil"] = {t: self.get_files(c) for t, c in child_items.items()}

        r = self.get_children(stat)
        titles = tlz.pluck("title", r["items"])
        titles = tlz.map(lambda s: s.split(":")[1].split(",")[1].strip(), titles)
        child_items = dict(zip(titles, tlz.pluck("id", r["items"])))
        files["STATSGO"] = {t: self.get_files(c) for t, c in child_items.items()}

        r = self.get_children(ssur)
        titles = tlz.pluck("title", r["items"])
        titles = tlz.map(lambda s: s.split(":")[1].strip(), titles)
        child_items = dict(zip(titles, tlz.pluck("id", r["items"])))
        files["SSURGO"] = {t: self.get_files(c) for t, c in child_items.items()}

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
        char_df.to_feather(self.char_feather)
        return char_df


def nhdplus_attrs(name: Optional[str] = None, save_dir: Optional[str] = None) -> pd.DataFrame:
    """Access NHDPlus V2.1 Attributes from ScienceBase over CONUS.

    More info can be found `here <https://www.sciencebase.gov/catalog/item/5669a79ee4b08895842a1d47>`_.

    Parameters
    ----------
    name : str, optional
        Name of the NHDPlus attribute, defaults to None which returns a dataframe containing
        metadata of all the available attributes in the database.
    save_dir : str, optional
        Directory to save the staged data frame containing metadata for the database,
        defaults to system's temp directory. The metadata dataframe is saved as a feather
        file, nhdplus_attrs.feather, in save_dir that can be loaded with Pandas.

    Returns
    -------
    pandas.DataFrame
        Either a dataframe containing the database metadata or the requested attribute over CONUS.
    """
    sb = ScienceBase(save_dir)
    if sb.char_feather.exists():
        char_df = pd.read_feather(sb.char_feather)
    else:
        char_df = sb.stage_data()

    if name is None:
        return char_df

    try:
        url = char_df[char_df.name == name].url.values[0]
    except IndexError as ex:
        raise InvalidInputValue("name", char_df.name.unique()) from ex

    return pd.read_csv(url, compression="zip")


def nhd_fcode() -> pd.DataFrame:
    """Get all the NHDPlus FCodes."""
    url = "/".join(
        [
            "https://gist.githubusercontent.com/cheginit",
            "8f9730f75b2b9918f492a236c77c180c/raw/nhd_fcode.json",
        ]
    )
    return pd.read_json(url)
