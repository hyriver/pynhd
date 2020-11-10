"""Access NLDI and WaterData databases."""
import numbers
from json import JSONDecodeError
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pygeoogc as ogc
import pygeoutils as geoutils
import shapely.ops as ops
from pandas._libs.missing import NAType
from pygeoogc import WFS, ArcGISRESTful, MatchCRS, RetrySession, ServiceURL
from requests import Response
from shapely.geometry import MultiPolygon, Polygon

from .exceptions import InvalidInputType, InvalidInputValue, MissingItems, ZeroMatched

DEF_CRS = "epsg:4326"
ALT_CRS = "epsg:4269"


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

    def bybox(
        self, bbox: Tuple[float, float, float, float], box_crs: str = DEF_CRS
    ) -> gpd.GeoDataFrame:
        """Get features within a bounding box."""
        resp = self.wfs.getfeature_bybox(bbox, box_crs, always_xy=True)
        return self.to_geodf(resp)

    def bygeom(self, geometry: Union[Polygon, MultiPolygon], xy: bool = True) -> gpd.GeoDataFrame:
        """Get features within a bounding box."""
        if not isinstance(geometry, (Polygon, MultiPolygon)):
            raise InvalidInputType("geometry", "Polygon or Multipolygon")

        g_wkt = ops.transform(lambda x, y: (y, x), geometry).wkt if xy else geometry.wkt

        return self.byfilter(f"INTERSECTS(the_geom, {g_wkt})", method="POST")

    def byid(self, featurename: str, featureids: Union[List[str], str]) -> gpd.GeoDataFrame:
        """Get features based on IDs."""
        resp = self.wfs.getfeature_byid(featurename, featureids)
        return self.to_geodf(resp)

    def byfilter(self, cql_filter: str, method: str = "GET") -> gpd.GeoDataFrame:
        """Get features based on a CQL filter."""
        resp = self.wfs.getfeature_byfilter(cql_filter, method)
        return self.to_geodf(resp)

    def to_geodf(self, resp: Response) -> gpd.GeoDataFrame:
        """Convert a response from WaterData to a GeoDataFrame.

        Parameters
        ----------
        resp : Response
            A response from a WaterData request.

        Returns
        -------
        geopandas.GeoDataFrame
            The requested features in a dataframes.
        """
        return geoutils.json2geodf(resp.json(), ALT_CRS, self.crs)


class NHDPlusHR:
    """Access NHDPlus HR database through the National Map ArcGISRESTful.

    Parameters
    ----------
    layer : str
        A valid service layer. For a list of available layers pass an empty string to
        the class.
    outfields : str or list, optional
        Target field name(s), default to "*" i.e., all the fileds.
    crs : str, optional
        Target spatial reference, default to EPSG:4326
    """

    def __init__(self, layer: str, outfields: Union[str, List[str]] = "*", crs: str = DEF_CRS):
        self.service = ArcGISRESTful(
            ServiceURL().restful.nhdplushr,
            outformat="json",
            outfields=outfields,
            crs=crs,
        )
        valid_layers = self.service.get_validlayers()
        self.valid_layers = {v.lower(): k for k, v in valid_layers.items()}
        if layer not in self.valid_layers:
            raise InvalidInputValue("layer", list(self.valid_layers))
        self.service.layer = self.valid_layers[layer]

        self.outfields = outfields
        self.crs = crs

    def bygeom(
        self,
        geom: Union[Polygon, Tuple[float, float, float, float]],
        geo_crs: str = "epsg:4326",
        sql_clause: str = "",
        return_m: bool = False,
    ) -> gpd.GeoDataFrame:
        """Get feature within a geometry that can be combined with a SQL where clause.

        Parameters
        ----------
        geom : Polygon or tuple
            A geometry (Polgon) or bounding box (tuple of length 4).
        geo_crs : str
            The spatial reference of the input geometry.
        sql_clause : str, optional
            A valid SQL 92 WHERE clause, defaults to an empty string.
        return_m : bool
            Whether to activate the Return M (measure) in the request, defaults to False.

        Returns
        -------
        geopandas.GeoDataFrame
            The requested features as a GeoDataFrame.
        """
        self.service.oids_bygeom(geom, geo_crs=geo_crs, sql_clause=sql_clause)
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

    def _getfeatures(self, return_m: bool = False):
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
        resp = self.service.get_features(return_m)
        return geoutils.json2geodf(resp)


class NLDI:
    """Access the Hydro Network-Linked Data Index (NLDI) service."""

    def __init__(self) -> None:
        self.base_url = ServiceURL().restful.nldi
        self.session = RetrySession()

        resp = self.session.get("/".join([self.base_url, "linked-data"])).json()
        self.valid_fsources = {r["source"]: r["sourceName"] for r in resp}

        resp = self.session.get("/".join([self.base_url, "lookups"])).json()
        self.valid_chartypes = {r["type"]: r["typeName"] for r in resp}

    def getfeature_byid(self, fsource: str, fid: str, basin: bool = False) -> gpd.GeoDataFrame:
        """Get features of a single id.

        Parameters
        ----------
        fsource : str
            The name of feature source. The valid sources are:
            comid, huc12pp, nwissite, wade, wqp
        fid : str
            The ID of the feature.
        basin : bool
            Whether to return the basin containing the feature.

        Returns
        -------
        geopandas.GeoDataFrame
            NLDI indexed features in EPSG:4326.
        """
        self._validate_fsource(fsource)

        url = "/".join([self.base_url, "linked-data", fsource, fid])
        if basin:
            url += "/basin"

        return geoutils.json2geodf(self._geturl(url), ALT_CRS, DEF_CRS)

    def getcharacteristic_byid(
        self,
        comids: Union[List[str], str],
        char_type: str,
        char_ids: str = "all",
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
            or if ``values_only`` is Fale return ``percent_nodata`` is well.
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
            url = "/".join([self.base_url, "linked-data", "comid", comid, char_type])
            rjson = self._geturl(url, payload)
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
        """Get all the avialable characteristics IDs for a give characteristics type."""
        resp = self.session.get("/".join([self.base_url, "lookups", char_type, "characteristics"]))
        c_list = ogc.utils.traverse_json(resp.json(), ["characteristicMetadata", "characteristic"])
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
        """Navigate the NHDPlus databse from a single feature id up to a distance.

        Parameters
        ----------
        fsource : str
            The name of feature source. The valid sources are:
            comid, huc12pp, nwissite, wade, WQP.
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
            have be mindful of the value that you provide.

        Returns
        -------
        geopandas.GeoDataFrame
            NLDI indexed features in EPSG:4326.
        """
        self._validate_fsource(fsource)

        url = "/".join([self.base_url, "linked-data", fsource, fid, "navigation"])

        valid_navigations = self._geturl(url)
        if navigation not in valid_navigations.keys():
            raise InvalidInputValue("navigation", list(valid_navigations.keys()))

        url = valid_navigations[navigation]

        r_json = self._geturl(url)
        valid_sources = {s["source"].lower(): s["features"] for s in r_json}
        if source not in valid_sources:
            raise InvalidInputValue("source", list(valid_sources.keys()))

        url = f"{valid_sources[source]}?distance={int(distance)}"

        return geoutils.json2geodf(self._geturl(url), ALT_CRS, DEF_CRS)

    def navigate_byloc(
        self,
        coords: Tuple[float, float],
        navigation: Optional[str] = None,
        source: Optional[str] = None,
        loc_crs: str = DEF_CRS,
        distance: int = 500,
        comid_only: bool = False,
    ) -> gpd.GeoDataFrame:
        """Navigate the NHDPlus databse from a coordinate.

        Parameters
        ----------
        coordinate : tuple
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
        comid_only : bool, optional
            Whether to return the nearest comid without navigation.

        Returns
        -------
        geopandas.GeoDataFrame
            NLDI indexed features in EPSG:4326.
        """
        _coords = MatchCRS().coords(((coords[0],), (coords[1],)), loc_crs, DEF_CRS)
        lon, lat = _coords[0][0], _coords[1][0]

        url = "/".join([self.base_url, "linked-data", "comid", "position"])
        payload = {"coords": f"POINT({lon} {lat})"}
        rjson = self._geturl(url, payload)
        comid = geoutils.json2geodf(rjson, ALT_CRS, DEF_CRS).comid.iloc[0]

        if comid_only:
            return comid

        if navigation is None or source is None:
            raise MissingItems(["navigation", "source"])

        return self.navigate_byid("comid", comid, navigation, source, distance)

    def characteristics_dataframe(
        self,
        char_type: str,
        char_id: str,
        filename: Optional[str] = None,
        metadata: bool = False,
    ) -> Union[Dict[str, Any], pd.DataFrame]:
        """Get a NHDPlus-based characteristic from sciencebase.gov as dataframe.

        Parameters
        ----------
        char_type : str
            Characteristic type. Valid values are ``local`` for
            individual reach catchments, ``tot`` for network-accumulated values
            using total cumulative drainage area and ``div`` for network-accumulated values
            using divergence-routed.
        char_id : str
            Characteristic ID.
        filename : str, optional
            File name, defaults to None that throws an error and shows
            a list of available files.
        metadata : bool
            Whether to only return the metadata for the selected characteristic,
            defaults to False. Useful for getting information about the dataset
            such as citation, units, column names, etc.

        Returns
        -------
        pandas.DataFrame or dict
            The requested characteristic as a dataframe or if ``metadata`` is True
            the metadata as a dictionary.
        """
        if char_type not in self.valid_chartypes:
            valids = [f'"{s}" for {d}' for s, d in self.valid_chartypes.items()]
            raise InvalidInputValue("char", valids)

        valid_charids = self.get_validchars(char_type)

        if char_id not in valid_charids.index:
            vids = valid_charids["characteristic_description"]
            raise InvalidInputValue("char_id", [f'"{s}" for {d}' for s, d in vids.items()])

        meta = self.session.get(
            valid_charids.loc[char_id, "dataset_url"], {"format": "json"}
        ).json()
        if metadata:
            return meta

        flist = {
            f["name"]: f["downloadUri"] for f in meta["files"] if f["name"].split(".")[-1] == "zip"
        }
        if filename not in flist:
            raise InvalidInputValue("filename", list(flist.keys()))

        return pd.read_csv(flist[filename], compression="zip")

    def _validate_fsource(self, fsource: str) -> None:
        """Check if the given feature source is valid."""
        if fsource not in self.valid_fsources:
            valids = [f'"{s}" for {d}' for s, d in self.valid_fsources.items()]
            raise InvalidInputValue("feature source", valids)

    def _geturl(self, url: str, payload: Optional[Dict[str, str]] = None):
        """Send a request to the service using GET method."""
        if payload is None:
            payload = {"f": "json"}
        else:
            payload.update({"f": "json"})

        try:
            return self.session.get(url, payload).json()
        except JSONDecodeError:
            raise ZeroMatched("No feature was found with the provided inputs.")


def prepare_nhdplus(
    flw: gpd.GeoDataFrame,
    min_network_size: float,
    min_path_length: float,
    min_path_size: float = 0,
    purge_non_dendritic: bool = False,
    verbose: bool = False,
) -> gpd.GeoDataFrame:
    """Clean up and fix common issues of NHDPlus flowline database.

    Ported from `nhdplusTools <https://github.com/USGS-R/nhdplusTools>`__

    Parameters
    ----------
    flw : geopandas.GeoDataFrame
        NHDPlus flowlines with at least the following columns:
        COMID, LENGTHKM, FTYPE, TerminalFl, FromNode, ToNode, TotDASqKM,
        StartFlag, StreamOrde, StreamCalc, TerminalPa, Pathlength,
        Divergence, Hydroseq, LevelPathI
    min_network_size : float
        Minimum size of drainage network in sqkm
    min_path_length : float
        Minimum length of terminal level path of a network in km.
    min_path_size : float, optional
        Minimum size of outlet level path of a drainage basin in km.
        Drainage basins with an outlet drainage area smaller than
        this value will be removed. Defaults to 0.
    purge_non_dendritic : bool, optional
        Whether to remove non dendritic paths, defaults to False
    verbose : bool, optional
        Whether to show a message about the removed features, defaults to True.

    Returns
    -------
    geopandas.GeoDataFrame
        Cleaned up flowlines. Note that all column names are converted to lower case.
    """
    flw.columns = flw.columns.str.lower()
    nrows = flw.shape[0]

    req_cols = [
        "comid",
        "terminalfl",
        "terminalpa",
        "hydroseq",
        "streamorde",
        "streamcalc",
        "divergence",
        "fromnode",
        "ftype",
    ]

    _check_requirements(req_cols, flw)
    flw[req_cols[:-1]] = flw[req_cols[:-1]].astype("Int64")

    if not any(flw.terminalfl == 1):
        if all(flw.terminalpa == flw.terminalpa.iloc[0]):
            flw.loc[flw.hydroseq == flw.hydroseq.min(), "terminalfl"] = 1
        else:
            raise ZeroMatched("No terminal flag were found in the dataframe.")

    if purge_non_dendritic:
        flw = flw[
            ((flw.ftype != "Coastline") | (flw.ftype != 566)) & (flw.streamorde == flw.streamcalc)
        ]
    else:
        flw = flw[(flw.ftype != "Coastline") | (flw.ftype != 566)]
        flw.loc[flw.divergence == 2, "fromnode"] = pd.NA

    flw = _remove_tinynetworks(flw, min_path_size, min_path_length, min_network_size)

    if verbose:
        print(f"Removed {nrows - flw.shape[0]} paths from the flowlines.")

    if flw.shape[0] > 0:
        flw = _add_tocomid(flw)

    return flw


def _remove_tinynetworks(
    flw: gpd.GeoDataFrame,
    min_path_size: float,
    min_path_length: float,
    min_network_size: float,
) -> gpd.GeoDataFrame:
    """Remove small paths in NHDPlus flowline database.

    Ported from `nhdplusTools <https://github.com/USGS-R/nhdplusTools>`__

    Parameters
    ----------
    flw : geopandas.GeoDataFrame
        NHDPlus flowlines with at least the following columns:
        levelpathi, hydroseq, totdasqkm, terminalfl, startflag,
        pathlength, terminalpa
    min_network_size : float
        Minimum size of drainage network in sqkm.
    min_path_length : float
        Minimum length of terminal level path of a network in km.
    min_path_size : float
        Minimum size of outlet level path of a drainage basin in km.
        Drainage basins with an outlet drainage area smaller than
        this value will be removed.

    Returns
    -------
    geopandas.GeoDataFrame
        Flowlines with small paths removed.
    """
    req_cols = [
        "levelpathi",
        "hydroseq",
        "terminalfl",
        "startflag",
        "terminalpa",
        "totdasqkm",
        "pathlength",
    ]
    _check_requirements(req_cols, flw)

    flw[req_cols[:-2]] = flw[req_cols[:-2]].astype("Int64")

    if min_path_size > 0:
        short_paths = flw.groupby("levelpathi").apply(
            lambda x: (x.hydroseq == x.hydroseq.min())
            & (x.totdasqkm < min_path_size)
            & (x.totdasqkm >= 0)
        )
        short_paths = short_paths.index.get_level_values("levelpathi")[short_paths].tolist()
        flw = flw[~flw.levelpathi.isin(short_paths)]

    terminal_filter = (flw.terminalfl == 1) & (flw.totdasqkm < min_network_size)
    start_filter = (flw.startflag == 1) & (flw.pathlength < min_path_length)

    if any(terminal_filter.dropna()) or any(start_filter.dropna()):
        tiny_networks = flw[terminal_filter].append(flw[start_filter])
        flw = flw[~flw.terminalpa.isin(tiny_networks.terminalpa.unique())]

    return flw


def _add_tocomid(flw: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Find the downstream comid(s) of each comid in NHDPlus flowline database.

    Ported from `nhdplusTools <https://github.com/USGS-R/nhdplusTools>`__

    Parameters
    ----------
    flw : geopandas.GeoDataFrame
        NHDPlus flowlines with at least the following columns:
        ``comid``, ``terminalpa``, ``fromnode``, ``tonode``

    Returns
    -------
    geopandas.GeoDataFrame
        The input dataframe With an additional column named ``tocomid``.
    """
    req_cols = ["comid", "terminalpa", "fromnode", "tonode"]
    _check_requirements(req_cols, flw)

    flw[req_cols] = flw[req_cols].astype("Int64")

    def tocomid(group):
        def toid(row):
            try:
                return group[group.fromnode == row.tonode].comid.to_numpy()[0]
            except IndexError:
                return pd.NA

        return group.apply(toid, axis=1)

    flw["tocomid"] = pd.concat([tocomid(g) for _, g in flw.groupby("terminalpa")])
    return flw


def topoogical_sort(
    flowlines: pd.DataFrame,
) -> Tuple[List[Union[str, NAType]], Dict[Union[str, NAType], List[str]], nx.DiGraph]:
    """Topological sorting of a river network.

    Parameters
    ----------
    flowlines : pandas.DataFrame
        A dataframe with columns ID and toID

    Returns
    -------
    (list, dict , networkx.DiGraph)
        A list of topologically sorted IDs, a dictionary
        with keys as IDs and values as its upstream nodes,
        and the generated networkx object. Note that the
        terminal node ID is set to pd.NA.
    """
    upstream_nodes = {i: flowlines[flowlines.toID == i].ID.tolist() for i in flowlines.ID.tolist()}
    upstream_nodes[pd.NA] = flowlines[flowlines.toID.isna()].ID.tolist()

    netwrok = nx.from_pandas_edgelist(
        flowlines[["ID", "toID"]],
        source="ID",
        target="toID",
        create_using=nx.DiGraph,
    )
    topo_sorted = list(nx.topological_sort(netwrok))
    return topo_sorted, upstream_nodes, netwrok


def vector_accumulation(
    flowlines: pd.DataFrame,
    func: Callable,
    attr_col: str,
    arg_cols: List[str],
    id_col: str = "comid",
    toid_col: str = "tocomid",
) -> pd.Series:
    """Flow accumulation using vector river network data.

    Parameters
    ----------
    flowlines : pandas.DataFrame
        A dataframe containing comid, tocomid, attr_col and all the columns
        that ara required for passing to ``func``.
    func : function
        The function that routes the flow in a single river segment.
        Positions of the arguments in the function should be as follows:
        ``func(qin, *arg_cols)``
        ``qin`` is computed in this function and the rest are in the order
        of the ``arg_cols``. For example, if ``arg_cols = ["slope", "roughness"]``
        then the functions is called this way:
        ``func(qin, slope, roughness)``
        where slope and roughness are elemental values read from the flowlines.
    attr_col : str
        The column name of the attribute being accumulated in the network.
        The column should contain the initial condition for the attribute for
        each river segment. It can be a scalar or an array (e.g., time series).
    arg_cols : list of strs
        List of the flowlines columns that contain all the required
        data for a routing a single river segment such as slope, length,
        lateral flow, etc.
    id_name : str, optional
        Name of the flowlines column containing IDs, defaults to ``comid``
    toid_name : str, optional
        Name of the flowlines column containing ``toIDs``, defaults to ``tocomid``

    Returns
    -------
    pandas.Series
        Accumulated flow for all the nodes. The dataframe is sorted from upstream
        to downstream (topological sorting). Depending on the given initial
        condition in the ``attr_col``, the outflow for each river segment can be
        a scalar or an array.
    """
    sorted_nodes, upstream_nodes, _ = topoogical_sort(
        flowlines[[id_col, toid_col]].rename(columns={id_col: "ID", toid_col: "toID"})
    )
    topo_sorted = sorted_nodes[:-1]

    outflow = flowlines.set_index(id_col)[attr_col].to_dict()

    init = flowlines.iloc[0][attr_col]
    if isinstance(init, numbers.Number):
        outflow["0"] = 0.0
    elif isinstance(init, (np.ndarray, list)):
        outflow["0"] = np.zeros_like(init)
    else:
        raise InvalidInputType("values of attr_col", "a scalar or an array")

    upstream_nodes.update({k: ["0"] for k, v in upstream_nodes.items() if len(v) == 0})

    for i in topo_sorted:
        outflow[i] = func(
            np.sum([outflow[u] for u in upstream_nodes[i]], axis=0),
            *flowlines.loc[flowlines[id_col] == i, arg_cols].to_numpy()[0],
        )

    outflow.pop("0")
    acc = pd.Series(outflow).loc[sorted_nodes[:-1]]
    acc = acc.rename_axis("comid").rename(f"acc_{attr_col}")
    return acc


def _check_requirements(reqs: Iterable, cols: List[str]) -> None:
    """Check for all the required data.

    Parameters
    ----------
    reqs : iterable
        A list of required data names (str)
    cols : list
        A list of variable names (str)
    """
    if not isinstance(reqs, Iterable):
        raise InvalidInputType("reqs", "iterable")

    missing = [r for r in reqs if r not in cols]
    if missing:
        raise MissingItems(missing)
