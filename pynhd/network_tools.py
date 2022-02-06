"""Access NLDI and WaterData databases."""
from typing import Callable, Iterable, List, Optional, Tuple, Union

import cytoolz as tlz
import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pyproj
from pandas._libs.missing import NAType
from pygeoutils import GeoBSpline, InvalidInputType
from pygeoutils.pygeoutils import Spline
from shapely import ops
from shapely.geometry import LineString

from . import nhdplus_derived as derived
from .core import logger
from .exceptions import MissingCRS, MissingItems

__all__ = [
    "prepare_nhdplus",
    "topoogical_sort",
    "vector_accumulation",
    "flowline_xsection",
    "network_xsection",
    "nhdflw2nx",
]


class NHDTools:
    """Prepare NHDPlus data for downstream analysis.

    Notes
    -----
    Some of these tools are ported from
    `nhdplusTools <https://github.com/USGS-R/nhdplusTools>`__.

    Parameters
    ----------
    flowlines : geopandas.GeoDataFrame
        NHDPlus flowlines with at least the following columns:
        ``comid``, ``lengthkm``, ``ftype``, ``terminalfl``, ``fromnode``, ``tonode``,
        ``totdasqkm``, ``startflag``, ``streamorde``, ``streamcalc``, ``terminalpa``,
        ``pathlength``, ``divergence``, ``hydroseq``, and ``levelpathi``.
    """

    def __init__(
        self,
        flowlines: gpd.GeoDataFrame,
    ):
        self.flw: gpd.GeoDataFrame = flowlines.copy()
        self.nrows: int = flowlines.shape[0]
        self.crs = flowlines.crs

    def clean_flowlines(self, use_enhd_attrs: bool, terminal2nan: bool) -> None:
        """Clean up flowlines."""
        self.flw.columns = self.flw.columns.str.lower()

        if not ("fcode" in self.flw and "ftype" in self.flw):
            raise MissingItems(["fcode", "ftype"])

        if use_enhd_attrs or not terminal2nan:
            if not terminal2nan and not use_enhd_attrs:
                logger.info("The use_enhd_attrs is set to True, so all attrs will be updated.")
            enhd_attrs = derived.enhd_attrs()
            self.flw["tocomid"] = self.flw["comid"].astype("Int64")
            self.flw = self.flw.reset_index().set_index("comid")
            self.flw.update(enhd_attrs.set_index("comid"))
            self.flw = self.flw.reset_index().set_index("index")

        if "fcode" in self.flw:
            self.flw = self.flw[self.flw["fcode"] != 56600].copy()
        else:
            self.flw = self.flw[
                (self.flw["ftype"] != "Coastline") | (self.flw["ftype"] != 566)
            ].copy()

        req_cols = [
            "comid",
            "terminalfl",
            "terminalpa",
            "hydroseq",
            "streamorde",
            "streamcalc",
            "divergence",
            "fromnode",
        ]

        self.check_requirements(req_cols, self.flw)
        self.flw[req_cols] = self.flw[req_cols].astype("Int64")

    def to_linestring(self) -> None:
        """Convert flowlines to shapely LineString objects."""
        self.flw["geometry"] = self.flw.geometry.apply(ops.linemerge)
        self.flw = self.flw.set_crs(self.flw.crs)

    def remove_tinynetworks(
        self,
        min_path_size: float,
        min_path_length: float,
        min_network_size: float,
    ) -> None:
        """Remove small paths in NHDPlus flowline database.

        Notes
        -----
        This functions requires the following columns:
        ``levelpathi``, ``hydroseq``, ``totdasqkm``, ``terminalfl``, ``startflag``,
        ``pathlength``, and ``terminalpa``.

        Parameters
        ----------
        min_network_size : float
            Minimum size of drainage network in sqkm.
        min_path_length : float
            Minimum length of terminal level path of a network in km.
        min_path_size : float
            Minimum size of outlet level path of a drainage basin in km.
            Drainage basins with an outlet drainage area smaller than
            this value will be removed.
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
        self.check_requirements(req_cols, self.flw)
        self.flw[req_cols[:-2]] = self.flw[req_cols[:-2]].astype("Int64")

        if min_path_size > 0:
            short_paths = self.flw.groupby("levelpathi").apply(
                lambda x: (x.hydroseq == x.hydroseq.min())
                & (x.totdasqkm < min_path_size)
                & (x.totdasqkm >= 0)
            )
            short_idx = short_paths.index.get_level_values("levelpathi")[short_paths]
            self.flw = self.flw[~self.flw.levelpathi.isin(short_idx)]

        terminal_filter = (self.flw.terminalfl == 1) & (self.flw.totdasqkm < min_network_size)
        start_filter = (self.flw.startflag == 1) & (self.flw.pathlength < min_path_length)

        if any(terminal_filter.dropna()) or any(start_filter.dropna()):
            tiny_networks = self.flw[terminal_filter].append(self.flw[start_filter])
            self.flw = self.flw[~self.flw.terminalpa.isin(tiny_networks.terminalpa.unique())]

    def add_tocomid(self) -> None:
        """Find the downstream comid(s) of each comid in NHDPlus flowline database.

        Notes
        -----
        This functions requires the following columns:
            ``comid``, ``terminalpa``, ``fromnode``, ``tonode``
        """
        req_cols = ["comid", "terminalpa", "fromnode", "tonode"]
        self.check_requirements(req_cols, self.flw)
        self.flw[req_cols] = self.flw[req_cols].astype("Int64")
        if "tocomid" in self.flw:
            self.flw = self.flw.drop(columns="tocomid")

        def tocomid(grp: pd.core.groupby.generic.DataFrameGroupBy) -> pd.DataFrame:
            def toid(tonode: pd.DataFrame) -> pd.Int64Dtype:
                try:
                    return grp[grp.fromnode == tonode].comid.iloc[0]
                except IndexError:
                    return pd.NA

            return pd.Series(
                {i: toid(n) for i, n in grp[["comid", "tonode"]].itertuples(index=False, name=None)}
            )

        toid = pd.concat(tocomid(g) for _, g in self.flw[req_cols].groupby("terminalpa"))
        toid = toid.reset_index().rename(columns={"index": "comid", 0: "tocomid"})
        self.flw = self.flw.merge(toid, on="comid")

    @staticmethod
    def check_requirements(reqs: Iterable[str], cols: Iterable[str]) -> None:
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


def prepare_nhdplus(
    flowlines: gpd.GeoDataFrame,
    min_network_size: float,
    min_path_length: float,
    min_path_size: float = 0,
    purge_non_dendritic: bool = False,
    use_enhd_attrs: bool = False,
    terminal2nan: bool = True,
) -> gpd.GeoDataFrame:
    """Clean up and fix common issues of NHDPlus flowline database.

    Ported from `nhdplusTools <https://github.com/USGS-R/nhdplusTools>`__.

    Parameters
    ----------
    flowlines : geopandas.GeoDataFrame
        NHDPlus flowlines with at least the following columns:
        ``comid``, ``lengthkm``, ``ftype``, ``terminalfl``, ``fromnode``, ``tonode``,
        ``totdasqkm``, ``startflag``, ``streamorde``, ``streamcalc``, ``terminalpa``,
        ``pathlength``, ``divergence``, ``hydroseq``, ``levelpathi``.
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
    use_enhd_attrs : bool, optional
        Whether to replace the attributes with the ENHD attributes, defaults to False.
        For more information, see
        `this <https://www.sciencebase.gov/catalog/item/60c92503d34e86b9389df1c9>`__.
    terminal2nan : bool, optional
        Whether to replace the COMID of the terminal flowline of the network with NaN,
        defaults to True. If False, the terminal COMID will be set from the
        ENHD attributes i.e. use_enhd_attrs will be set to True.

    Returns
    -------
    geopandas.GeoDataFrame
        Cleaned up flowlines. Note that all column names are converted to lower case.
    """
    nhd = NHDTools(flowlines)

    if not nhd.flw.geom_type.eq("LineString").all():
        nhd.to_linestring()

    nhd.clean_flowlines(use_enhd_attrs, terminal2nan)

    if not any(nhd.flw.terminalfl == 1):
        if len(nhd.flw.terminalpa.unique()) != 1:
            logger.error("Found no terminal flag in the dataframe.")

        nhd.flw.loc[nhd.flw.hydroseq == nhd.flw.hydroseq.min(), "terminalfl"] = 1

    if purge_non_dendritic:
        nhd.flw = nhd.flw[nhd.flw.streamorde == nhd.flw.streamcalc].copy()
    else:
        nhd.flw.loc[nhd.flw.divergence == 2, "fromnode"] = pd.NA

    nhd.remove_tinynetworks(min_path_size, min_path_length, min_network_size)

    if (nhd.nrows - nhd.flw.shape[0]) > 0:
        logger.info(f"Removed {nhd.nrows - nhd.flw.shape[0]} segments from the flowlines.")

    if nhd.flw.shape[0] > 0 and ("tocomid" not in nhd.flw or terminal2nan):
        nhd.add_tocomid()

    return nhd.flw


def nhdflw2nx(
    flowlines: pd.DataFrame,
    id_col: str = "comid",
    toid_col: str = "tocomid",
    edge_attr: Optional[Union[str, List[str]]] = None,
) -> nx.DiGraph:
    """Convert NHDPlus flowline database to networkx graph.

    Parameters
    ----------
    flowlines : geopandas.GeoDataFrame
        NHDPlus flowlines.
    id_col : str, optional
        Name of the column containing the node ID, defaults to "comid".
    toid_col : str, optional
        Name of the column containing the downstream node ID, defaults to "tocomid".
    edge_attr : str, optional
        Name of the column containing the edge attributes, defaults to ``None``.

    Returns
    -------
    nx.DiGraph
        Networkx directed graph of the NHDPlus flowlines.
    """
    return nx.from_pandas_edgelist(
        flowlines,
        source=id_col,
        target=toid_col,
        create_using=nx.DiGraph,
        edge_attr=edge_attr,
    )


def topoogical_sort(
    flowlines: pd.DataFrame, edge_attr: Optional[Union[str, List[str]]] = None
) -> Tuple[List[Union[np.int64, NAType]], pd.Series, nx.DiGraph]:
    """Topological sorting of a river network.

    Parameters
    ----------
    flowlines : pandas.DataFrame
        A dataframe with columns ID and toID
    edge_attr : str or list, optional
        Names of the columns in the dataframe to be used as edge attributes, defaults to None.

    Returns
    -------
    (list, dict , networkx.DiGraph)
        A list of topologically sorted IDs, a dictionary
        with keys as IDs and values as its upstream nodes,
        and the generated networkx object. Note that the
        terminal node ID is set to pd.NA.
    """
    up_nodes = pd.Series(
        {i: flowlines[flowlines.toID == i].ID.tolist() for i in list(flowlines.ID) + [pd.NA]}
    )

    flowlines[["ID", "toID"]] = flowlines[["ID", "toID"]].astype("Int64")
    network = nhdflw2nx(flowlines, "ID", "toID", edge_attr)
    topo_sorted = list(nx.topological_sort(network))
    return topo_sorted, up_nodes, network


def vector_accumulation(
    flowlines: pd.DataFrame,
    func: Callable[..., float],
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
    id_col : str, optional
        Name of the flowlines column containing IDs, defaults to ``comid``
    toid_col : str, optional
        Name of the flowlines column containing ``toIDs``, defaults to ``tocomid``

    Returns
    -------
    pandas.Series
        Accumulated flow for all the nodes. The dataframe is sorted from upstream
        to downstream (topological sorting). Depending on the given initial
        condition in the ``attr_col``, the outflow for each river segment can be
        a scalar or an array.
    """
    sorted_nodes, up_nodes, _ = topoogical_sort(
        flowlines[[id_col, toid_col]].rename(columns={id_col: "ID", toid_col: "toID"})
    )
    topo_sorted = sorted_nodes[:-1]

    outflow = flowlines.set_index(id_col)[attr_col].to_dict()

    for i in topo_sorted:
        outflow[i] = func(
            np.sum([outflow[u] for u in up_nodes[i]], axis=0),
            *flowlines.loc[flowlines[id_col] == i, arg_cols].to_numpy()[0],
        )

    acc = pd.Series(outflow).loc[sorted_nodes[:-1]]
    acc = acc.rename_axis("comid").rename(f"acc_{attr_col}")
    return acc


def get_spline(line: LineString, ns_pts: int, crs: Union[str, pyproj.CRS]) -> Spline:
    """Get a B-spline from a line."""
    x, y = line.xy
    pts = gpd.GeoSeries(gpd.points_from_xy(x, y, crs=crs))
    return GeoBSpline(pts, ns_pts).spline


def get_idx(d_sp: np.ndarray, distance: float) -> np.ndarray:  # type: ignore
    """Get the index of the closest point to a given distance."""
    dis = pd.DataFrame(d_sp, columns=["distance"]).reset_index()
    grouper = pd.cut(dis.distance, np.arange(0, dis.distance.max() + distance, distance))
    return dis.groupby(grouper).last()["index"].values  # type: ignore


def get_perpendicular(
    line: LineString, n_seg: int, distance: float, half_width: float, crs: Union[str, pyproj.CRS]
) -> List[LineString]:
    """Get perpendiculars to a line."""
    _n_seg = n_seg
    spline = get_spline(line, _n_seg, crs)
    idx = get_idx(spline.distance, distance)
    while np.isnan(idx).any():
        _n_seg *= 2
        spline = get_spline(line, _n_seg, crs)
        idx = get_idx(spline.distance, distance)
    x_l = spline.x[idx] - half_width * np.sin(spline.phi[idx])
    x_r = spline.x[idx] + half_width * np.sin(spline.phi[idx])
    y_l = spline.y[idx] - half_width * np.cos(spline.phi[idx])
    y_r = spline.y[idx] + half_width * np.cos(spline.phi[idx])
    return [LineString([(x1, y1), (x2, y2)]) for x1, y1, x2, y2 in zip(x_l, y_l, x_r, y_r)]


def flowline_xsection(flw: gpd.GeoDataFrame, distance: float, width: float) -> gpd.GeoDataFrame:
    """Get cross-section of a river network at a given spacing.

    Parameters
    ----------
    flw : geopandas.GeoDataFrame
        A dataframe with ``geometry`` and ``comid`` columns and CRS attribute.
    distance : float
        The distance between two consecutive cross-sections.
    width : float
        The width of the cross-section.

    Returns
    -------
    geopandas.GeoDataFrame
        A dataframe with two columns: ``geometry`` and ``comid``. The ``geometry``
        column contains the cross-section of the river network and the ``comid``
        column contains the corresponding ``comid`` from the input dataframe.
        Note that each ``comid`` can have multiple cross-sections depending on
        the given spacing distance.
    """
    if flw.crs is None:
        raise MissingCRS

    if not flw.crs.is_projected:
        raise InvalidInputType("points.crs", "projected CRS")

    req_cols = ["comid", "geometry"]
    if any(col not in flw for col in req_cols):
        raise MissingItems(req_cols)

    half_width = width * 0.5
    merged = ops.linemerge(flw.geometry.to_list())
    if isinstance(merged, LineString):
        lines = [merged]
    else:
        lines = list(merged.geoms)
    n_segments = (int(np.ceil(ln.length / distance)) * 100 for ln in lines)
    main_split = tlz.concat(
        get_perpendicular(ln, n_seg, distance, half_width, flw.crs)
        for ln, n_seg in zip(lines, n_segments)
    )

    cs = gpd.GeoDataFrame(geometry=list(main_split), crs=flw.crs)
    cs_idx, flw_idx = flw.sindex.query_bulk(cs.geometry)
    merged_idx = tlz.merge_with(list, ({t: i} for t, i in zip(flw_idx, cs_idx)))

    cs["comid"] = 0
    for fi, ci in merged_idx.items():
        cs.loc[ci, "comid"] = flw.iloc[fi].comid
    return cs.drop_duplicates()


def network_xsection(flw: gpd.GeoDataFrame, distance: float, width: float) -> gpd.GeoDataFrame:
    """Get cross-section of a river network at a given spacing.

    Parameters
    ----------
    flw : geopandas.GeoDataFrame
        A dataframe with ``geometry`` and ``comid`` columns and CRS attribute.
    distance : float
        The distance between two consecutive cross-sections.
    width : float
        The width of the cross-section.

    Returns
    -------
    geopandas.GeoDataFrame
        A dataframe with two columns: ``geometry`` and ``comid``. The ``geometry``
        column contains the cross-section of the river network and the ``comid``
        column contains the corresponding ``comid`` from the input dataframe.
        Note that each ``comid`` can have multiple cross-sections depending on
        the given spacing distance.
    """
    if flw.crs is None:
        raise MissingCRS

    if not flw.crs.is_projected:
        raise InvalidInputType("flw.crs", "projected CRS")

    req_cols = ["comid", "levelpathi", "geometry"]
    if any(col not in flw for col in req_cols):
        raise MissingItems(req_cols)
    cs = pd.concat(flowline_xsection(f, distance, width) for _, f in flw.groupby("levelpathi"))
    return gpd.GeoDataFrame(cs, crs=flw.crs).drop_duplicates()
