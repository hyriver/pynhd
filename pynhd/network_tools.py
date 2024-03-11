"""Access NLDI and WaterData databases."""

# pyright: reportGeneralTypeIssues=false, reportMissingTypeArgument=false
from __future__ import annotations

import io
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Union, cast

import cytoolz.curried as tlz
import geopandas as gpd
import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd
import shapely
from shapely import LineString, MultiLineString, ops

import async_retriever as ar
import pygeoutils as pgu
from pygeoogc import streaming_download
from pygeoutils import InputTypeError
from pynhd import nhdplus_derived as derived
from pynhd.core import ScienceBase
from pynhd.exceptions import (
    DependencyError,
    InputValueError,
    MissingCRSError,
    MissingItemError,
    NoTerminalError,
)

try:
    from pandas.errors import IntCastingNaNError
except ImportError:
    IntCastingNaNError = TypeError

if TYPE_CHECKING:
    import pyproj
    from pandas._libs.missing import NAType
    from pandas.core.groupby.generic import DataFrameGroupBy

    CRSTYPE = Union[str, int, pyproj.CRS]


__all__ = [
    "prepare_nhdplus",
    "topoogical_sort",
    "vector_accumulation",
    "flowline_resample",
    "network_resample",
    "flowline_xsection",
    "network_xsection",
    "nhdflw2nx",
    "enhd_flowlines_nx",
    "mainstem_huc12_nx",
    "nhdplus_l48",
    "NHDTools",
]


def nhdflw2nx(
    flowlines: pd.DataFrame,
    id_col: str = "comid",
    toid_col: str = "tocomid",
    edge_attr: str | list[str] | bool | None = None,
) -> nx.DiGraph:  # pyright: ignore[reportMissingTypeArgument]
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
        If ``True``, all remaining columns will be used as edge attributes.

    Returns
    -------
    nx.DiGraph
        Networkx directed graph of the NHDPlus flowlines. Note that all elements of
        the ``toid_col`` are replaced with negative values of their corresponding
        ``id_cl`` values if they are ``NaN`` or 0. This is to ensure that the generated
        nodes in the graph are unique.
    """
    flw = flowlines.copy()
    tocomid_na = flw[toid_col].isna() | (flw[toid_col] == 0)
    if tocomid_na.any():
        flw.loc[tocomid_na, toid_col] = -flw.loc[tocomid_na, id_col]

    return nx.from_pandas_edgelist(  # pyright: ignore[reportCallIssue]
        flw,
        source=id_col,
        target=toid_col,
        create_using=nx.DiGraph,
        edge_attr=edge_attr,  # pyright: ignore[reportArgumentType]
    )


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
    ) -> None:
        self.flw = cast("gpd.GeoDataFrame", flowlines.copy())
        self.nrows = flowlines.shape[0]
        self.crs = flowlines.crs
        self.is_hr = "nhdplusid" in flowlines

    def clean_flowlines(self, use_enhd_attrs: bool, terminal2nan: bool) -> None:
        """Clean up flowlines.

        Parameters
        ----------
        use_enhd_attrs : bool
            Use attributes from the ENHD database.
        terminal2nan : bool
            Convert terminal flowlines to ``NaN``.
        """
        self.flw.columns = self.flw.columns.str.lower()

        if self.is_hr:
            self.flw = self.flw.rename(columns={"nhdplusid": "comid"})  # pyright: ignore[reportCallIssue]

        if not ("fcode" in self.flw and "ftype" in self.flw):
            raise MissingItemError(["fcode", "ftype"])

        if use_enhd_attrs or not terminal2nan:
            enhd_attrs = derived.enhd_attrs().set_index("comid")
            self.flw["tocomid"] = self.flw["comid"].astype("Int64")
            self.flw = self.flw.reset_index().set_index("comid")
            cols = list(set(self.flw.columns).intersection(enhd_attrs.columns))
            self.flw[cols] = enhd_attrs.loc[self.flw.index, cols]
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

        self.check_requirements(req_cols, self.flw.columns)
        try:
            self.flw[req_cols] = self.flw[req_cols].astype("int64")
        except (ValueError, TypeError, IntCastingNaNError):
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
        self.check_requirements(req_cols, self.flw.columns)
        try:
            self.flw[req_cols[:-2]] = self.flw[req_cols[:-2]].astype("int64")
        except (TypeError, IntCastingNaNError):
            self.flw[req_cols[:-2]] = self.flw[req_cols[:-2]].astype("Int64")

        if min_path_size > 0:
            short_paths = self.flw.groupby("levelpathi")[["hydroseq", "totdasqkm"]].apply(
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

    def remove_isolated(self) -> None:
        """Remove isolated flowlines."""
        req_cols = ["comid", "tocomid"]
        self.check_requirements(req_cols, self.flw.columns)
        tocomid_na = self.flw.tocomid.isna() | (self.flw.tocomid == 0)
        if tocomid_na.any():
            self.flw.loc[tocomid_na, "tocomid"] = -self.flw.loc[tocomid_na, "comid"]

        comids = max(nx.weakly_connected_components(nhdflw2nx(self.flw)), key=len)
        self.flw = self.flw[self.flw.comid.isin(comids)].copy()
        self.flw.loc[self.flw.tocomid < 0, "tocomid"] = pd.NA
        self.flw = self.flw.reset_index(drop=True)

    def add_tocomid(self) -> None:
        """Find the downstream comid(s) of each comid in NHDPlus flowline database.

        Notes
        -----
        This functions requires the following columns:
            ``comid``, ``terminalpa``, ``fromnode``, ``tonode``
        """
        req_cols = ["comid", "terminalpa", "fromnode", "tonode"]
        self.check_requirements(req_cols, self.flw.columns)
        try:
            self.flw[req_cols] = self.flw[req_cols].astype("int64")
        except (ValueError, TypeError, IntCastingNaNError):
            self.flw[req_cols] = self.flw[req_cols].astype("Int64")

        if "tocomid" in self.flw:
            self.flw = self.flw.drop(columns="tocomid")

        def tocomid(grp: DataFrameGroupBy) -> pd.DataFrame:
            g_merged = pd.merge(
                grp[["comid", "tonode"]],  # pyright: ignore[reportArgumentType]
                grp[["comid", "fromnode"]].rename(columns={"comid": "tocomid"}),
                left_on="tonode",
                right_on="fromnode",
            )
            g_merged = g_merged[["comid", "tocomid"]].astype("Int64")
            return grp.merge(g_merged, on="comid", how="left")

        self.flw = pd.concat(  # pyright: ignore[reportCallIssue]
            tocomid(g)  # pyright: ignore[reportArgumentType]
            for _, g in self.flw.groupby("terminalpa")  # pyright: ignore[reportArgumentType]
        )
        self.flw = self.flw.reset_index(drop=True)

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
            raise InputTypeError("reqs", "iterable")

        missing = [r for r in reqs if r not in cols]
        if missing:
            raise MissingItemError(missing)


def prepare_nhdplus(
    flowlines: gpd.GeoDataFrame,
    min_network_size: float,
    min_path_length: float,
    min_path_size: float = 0,
    purge_non_dendritic: bool = False,
    remove_isolated: bool = False,
    use_enhd_attrs: bool = False,
    terminal2nan: bool = True,
) -> gpd.GeoDataFrame:
    """Clean up and fix common issues of NHDPlus MR and HR flowlines.

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
        Whether to remove non dendritic paths, defaults to ``False``.
    remove_isolated : bool, optional
        Whether to remove isolated flowlines, i.e., keep only the largest
        connected component of the flowlines. Defaults to ``False``.
    use_enhd_attrs : bool, optional
        Whether to replace the attributes with the ENHD attributes, defaults
        to ``False``. Note that this only works for NHDPlus mid-resolution (MR) and
        does not work for NHDPlus high-resolution (HR). For more information, see
        `this <https://www.sciencebase.gov/catalog/item/60c92503d34e86b9389df1c9>`__.
    terminal2nan : bool, optional
        Whether to replace the COMID of the terminal flowline of the network with NaN,
        defaults to ``True``. If ``False``, the terminal COMID will be set from the
        ENHD attributes i.e. ``use_enhd_attrs`` will be set to ``True`` which is only
        applicable to NHDPlus mid-resolution (MR).

    Returns
    -------
    geopandas.GeoDataFrame
        Cleaned up flowlines. Note that all column names are converted to lower case.
    """
    nhd = NHDTools(flowlines)

    if "geometry" in nhd.flw and not nhd.flw.geom_type.eq("LineString").all():
        nhd.to_linestring()

    if nhd.is_hr and use_enhd_attrs:
        msg = ". ".join(
            (
                "ENHD attributes are not available for NHDPlus HR.",
                "So, use_enhd_attrs is set to False.",
            )
        )
        warnings.warn(msg, UserWarning, stacklevel=2)
        use_enhd_attrs = False

    nhd.clean_flowlines(use_enhd_attrs, terminal2nan)

    if not any(nhd.flw["terminalfl"] == 1):
        if len(nhd.flw["terminalpa"].unique()) != 1:
            raise NoTerminalError

        nhd.flw.loc[nhd.flw["hydroseq"] == nhd.flw["hydroseq"].min(), "terminalfl"] = 1

    if purge_non_dendritic:
        nhd.flw = nhd.flw[nhd.flw["streamorde"] == nhd.flw["streamcalc"]].copy()
    else:
        nhd.flw.loc[nhd.flw["divergence"] == 2, "fromnode"] = pd.NA

    nhd.remove_tinynetworks(min_path_size, min_path_length, min_network_size)

    if (nhd.nrows - nhd.flw.shape[0]) > 0:
        warnings.warn(
            f"Removed {nhd.nrows - nhd.flw.shape[0]} segments from the flowlines.",
            UserWarning,
            stacklevel=2,
        )

    if nhd.flw.shape[0] > 0 and ("tocomid" not in nhd.flw or terminal2nan):
        nhd.add_tocomid()

    if remove_isolated:
        nhd.remove_isolated()
    nhd.flw = cast("gpd.GeoDataFrame", nhd.flw)
    return nhd.flw


def _create_subgraph(graph: nx.DiGraph, nodes: set[int]) -> nx.DiGraph:  # pyright: ignore[reportMissingTypeArgument]
    """Create a subgraph from a list of nodes."""
    subgraph = graph.__class__()
    subgraph.add_nodes_from((n, graph.nodes[n]) for n in nodes)
    subgraph.add_edges_from(
        (n, nbr, d)
        for n, nbrs in graph.adj.items()
        if n in nodes
        for nbr, d in nbrs.items()
        if nbr in nodes
    )
    subgraph.graph.update(graph.graph)
    return subgraph


def topoogical_sort(
    flowlines: pd.DataFrame,
    edge_attr: str | list[str] | None = None,
    largest_only: bool = False,
    id_col: str = "ID",
    toid_col: str = "toID",
) -> tuple[list[np.int64 | NAType], dict[int, list[int]], nx.DiGraph]:  # pyright: ignore[reportMissingTypeArgument]
    """Topological sorting of a river network.

    Parameters
    ----------
    flowlines : pandas.DataFrame
        A dataframe with columns ID and toID
    edge_attr : str or list, optional
        Names of the columns in the dataframe to be used as edge attributes, defaults to None.
    largest_only : bool, optional
        Whether to return only the largest network, defaults to ``False``.
    id_col : str, optional
        Name of the column containing the node ID, defaults to ``ID``.
    toid_col : str, optional
        Name of the column containing the downstream node ID, defaults to ``toID``.

    Returns
    -------
    (list, dict , networkx.DiGraph)
        A list of topologically sorted IDs, a dictionary
        with keys as IDs and values as a list of its upstream nodes,
        and the generated ``networkx.DiGraph`` object. Note that node
        IDs are associated with the input flow line IDs, but there might
        be some negative IDs in the output garph that are not present in
        the input flow line IDs. These "artificial" nodes are used to represent the
        graph outlet (the most downstream nodes) in the graph.
    """
    network = nhdflw2nx(flowlines, id_col, toid_col, edge_attr)
    if largest_only:
        nodes = max(nx.weakly_connected_components(network), key=len)  # pyright: ignore[reportArgumentType]
        network = _create_subgraph(network, nodes)
    topo_sorted = list(nx.topological_sort(network))
    up_nodes = {i: list(network.predecessors(i)) for i in network}
    return topo_sorted, up_nodes, network


def vector_accumulation(
    flowlines: pd.DataFrame,
    func: Callable[..., float],
    attr_col: str,
    arg_cols: list[str],
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
    if not isinstance(arg_cols, list):
        raise InputTypeError("arg_cols", "list of column names")

    flw = flowlines.copy()
    toid_na = flw[toid_col].isna() | (flw[toid_col] == 0)
    if toid_na.any():
        flw.loc[toid_na, toid_col] = -flw.loc[toid_na, id_col]

    graph = nx.relabel.convert_node_labels_to_integers(
        nx.from_pandas_edgelist(
            flw,
            source=id_col,
            target=toid_col,
            create_using=nx.DiGraph,
            edge_attr=attr_col,
        ),
        label_attribute="str_id",
    )

    m_int2str = nx.get_node_attributes(graph, "str_id")
    m_str2int = {v: k for k, v in m_int2str.items()}

    topo_sorted = list(nx.topological_sort(graph))

    outflow = {m_str2int[i]: a for i, a in flw.set_index(id_col)[attr_col].items()}
    attrs = {m_str2int[i]: a.tolist() for i, a in flw.set_index(id_col)[arg_cols].iterrows()}

    for n in topo_sorted:
        if n in outflow:
            outflow[n] = func(sum(outflow[i] for i in graph.predecessors(n)), *attrs[n])

    return (
        pd.Series({m_int2str[i]: outflow[i] for i in topo_sorted if i in outflow})
        .rename_axis(id_col)
        .rename(f"acc_{attr_col}")
    )


def _get_idx(d_sp: npt.NDArray[np.float64], distance: float) -> npt.NDArray[np.int64]:
    """Get the index of the closest points based on a given distance."""
    dis = pd.DataFrame(d_sp, columns=["distance"]).reset_index()  # pyright: ignore[reportArgumentType]
    bins = np.arange(0, dis["distance"].max() + distance, distance)
    grouper = pd.cut(dis["distance"], bins)
    idx = dis.groupby(grouper, observed=True).last()["index"].to_numpy("int64")
    return np.append(0, idx)


def _get_spline_params(
    line: LineString, n_seg: int, distance: float, crs: CRSTYPE, smoothing: float | None
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Get Spline parameters (x, y, phi)."""
    _n_seg = n_seg
    spline = pgu.spline_linestring(line, crs, _n_seg, smoothing=smoothing)
    idx = _get_idx(spline.distance, distance)
    while np.isnan(idx).any():
        _n_seg *= 2
        spline = pgu.spline_linestring(line, crs, _n_seg, smoothing=smoothing)
        idx = _get_idx(spline.distance, distance)
    x, y, phi = spline.x[idx], spline.y[idx], spline.phi[idx]
    spacing = np.diff(spline.distance[idx])

    if len(x) > 2 and spacing[0] < 0.25 * distance:
        x = np.delete(x, 1)
        y = np.delete(y, 1)
        phi = np.delete(phi, 1)

    if len(x) > 2 and spacing[-1] < 0.25 * distance:
        x = np.delete(x, -2)
        y = np.delete(y, -2)
        phi = np.delete(phi, -2)
    return x, y, phi


def _xs_planar(
    line: LineString,
    n_seg: int,
    half_width: float,
    distance: float,
    crs: CRSTYPE,
    smoothing: float | None,
) -> npt.NDArray[LineString]:
    """Get cross-sections along a line at a given spacing.

    Parameters
    ----------
    line : LineString
        A river centerline along which the cross-sections will be generated.
    n_seg : int
        Number of segments to use for the spline.
    half_width : float
        Half of the width of the cross-section.
    distance : float
        The distance between two consecutive cross-sections.
    crs : str or int or pyproj.CRS
        The CRS of the input line. Using projected CRS is highly recommended.
    smoothing : float or None
        Smoothing factor is used for determining the number of knots.
        This arg controls the tradeoff between closeness and smoothness of fit.
        Larger ``smoothing`` means more smoothing while smaller values of
        ``smoothing`` indicates less smoothing. If None, smoothing
        is done with all points.

    Returns
    -------
    numpy.ndarray
        Cross-sections along the line.
    """
    if not isinstance(line, LineString):
        raise InputTypeError("line", "LineString")

    if shapely.get_num_points(line) < 4:
        line = shapely.segmentize(line, line.length / 4)
    x, y, phi = _get_spline_params(line, n_seg, distance, crs, smoothing)

    xy = np.c_[x, y]
    # Normal vector of the centerline at each point.
    vn_right = np.c_[np.sin(phi), -np.cos(phi)]
    vn_left = np.c_[-np.sin(phi), np.cos(phi)]

    _lines = zip(xy + half_width * vn_left, xy + half_width * vn_right)
    return shapely.linestrings(list(_lines))


def __check_flw(flw: gpd.GeoDataFrame, req_cols: list[str]) -> None:
    """Get flowlines."""
    if flw.crs is None:
        raise MissingCRSError

    if not flw.crs.is_projected:
        warnings.warn(
            " ".join(
                (
                    "The flowlines are not in a projected CRS.",
                    "So, length computations are not accurate and all input lengths",
                    "are assumed to be in the ``flw``'s CRS unit.",
                )
            ),
            UserWarning,
            stacklevel=2,
        )

    if any(col not in flw for col in req_cols):
        raise MissingItemError(req_cols)


def __merge_flowlines(flw: gpd.GeoDataFrame) -> LineString:
    """Merge flowlines."""
    line = shapely.line_merge(flw.geometry.unary_union)

    if not isinstance(line, LineString):
        raise InputValueError("flw.geometry", "mergeable to a single line")
    return line


def flowline_resample(
    flw: gpd.GeoDataFrame, spacing: float, id_col: str = "comid", smoothing: float | None = None
) -> gpd.GeoDataFrame:
    """Resample a flowline based on a given spacing.

    Parameters
    ----------
    flw : geopandas.GeoDataFrame
        A dataframe with ``geometry`` and ``id_col`` columns and CRS attribute.
        The flowlines should be able to merged to a single ``LineString``.
        Otherwise, you should use the :func:`network_resample` function.
    spacing : float
        Spacing between the sample points in meters.
    id_col : str, optional
        Name of the flowlines column containing IDs, defaults to ``comid``.
    smoothing : float or None, optional
        Smoothing factor is used for determining the number of knots.
        This arg controls the tradeoff between closeness and smoothness of fit.
        Larger ``smoothing`` means more smoothing while smaller values of
        ``smoothing`` indicates less smoothing. If None (default), smoothing
        is done with all points.

    Returns
    -------
    geopandas.GeoDataFrame
        Resampled flowline.
    """
    __check_flw(flw, ["geometry", id_col])
    line = __merge_flowlines(flw)

    n_seg = int(np.ceil(line.length / spacing)) * 100
    line = pgu.smooth_linestring(line, smoothing, n_seg)

    lines = shapely.linestrings(list(zip(line.coords[:-1], line.coords[1:])))
    resampled = gpd.GeoDataFrame(geometry=lines, crs=flw.crs)
    rs_idx, flw_idx = flw.sindex.nearest(resampled.geometry, max_distance=spacing, return_all=False)
    merged_idx = tlz.merge_with(list, ({t: i} for t, i in zip(flw_idx, rs_idx)))

    resampled[id_col] = 0
    for fi, ci in merged_idx.items():
        resampled.loc[ci, id_col] = flw.iloc[fi][id_col]
    resampled = resampled.dissolve(by=id_col)
    resampled["geometry"] = [
        ln if isinstance(ln, LineString) else ops.linemerge(ln) for ln in resampled.geometry
    ]
    resampled = cast("gpd.GeoDataFrame", resampled)
    return resampled


def network_resample(
    flw: gpd.GeoDataFrame, spacing: float, id_col: str = "comid", smoothing: float | None = None
) -> gpd.GeoDataFrame:
    """Resample a network flowline based on a given spacing.

    Parameters
    ----------
    flw : geopandas.GeoDataFrame
        A dataframe with ``geometry`` and, ``id_col``, and ``levelpathi``
        columns and a projected CRS attribute.
    spacing : float
        Target spacing between the sample points in the length unit of the ``flw``'s CRS.
    id_col : str, optional
        Name of the flowlines column containing IDs, defaults to ``comid``.
    smoothing : float or None, optional
        Smoothing factor is used for determining the number of knots.
        This arg controls the tradeoff between closeness and smoothness of fit.
        Larger ``smoothing`` means more smoothing while smaller values of
        ``smoothing`` indicates less smoothing. If None (default), smoothing
        is done with all points.

    Returns
    -------
    geopandas.GeoDataFrame
        Resampled flowlines.
    """
    __check_flw(flw, [id_col, "levelpathi", "geometry"])
    cs = pd.concat(  # pyright: ignore[reportCallIssue]
        flowline_resample(f, spacing, id_col, smoothing)  # pyright: ignore[reportArgumentType]
        for _, f in flw.groupby("levelpathi")
    )
    return gpd.GeoDataFrame(cs, crs=flw.crs).drop_duplicates().dissolve(by=id_col)


def flowline_xsection(
    flw: gpd.GeoDataFrame,
    distance: float,
    width: float,
    id_col: str = "comid",
    smoothing: float | None = None,
) -> gpd.GeoDataFrame:
    """Get cross-section of a river network at a given spacing.

    Parameters
    ----------
    flw : geopandas.GeoDataFrame
        A dataframe with ``geometry`` and, ``id_col``, and ``levelpathi``
        columns and a projected CRS attribute.
    distance : float
        The distance between two consecutive cross-sections.
    width : float
        The width of the cross-section.
    id_col : str, optional
        Name of the flowlines column containing IDs, defaults to ``comid``.
    smoothing : float or None, optional
        Smoothing factor is used for determining the number of knots.
        This arg controls the tradeoff between closeness and smoothness of fit.
        Larger ``smoothing`` means more smoothing while smaller values of
        ``smoothing`` indicates less smoothing. If None (default), smoothing
        is done with all points.

    Returns
    -------
    geopandas.GeoDataFrame
        A dataframe with two columns: ``geometry`` and ``comid``. The ``geometry``
        column contains the cross-section of the river network and the ``comid``
        column contains the corresponding ``comid`` from the input dataframe.
        Note that each ``comid`` can have multiple cross-sections depending on
        the given spacing distance.
    """
    __check_flw(flw, ["geometry", id_col])
    half_width = width * 0.5
    lines = shapely.line_merge(flw.geometry.unary_union)
    if isinstance(lines, LineString):
        lines = [lines]
    elif isinstance(lines, MultiLineString):
        lines = list(lines.geoms)
    else:
        raise InputValueError("flw.geometry", "LineString or MultiLineString")

    n_segments = (int(np.ceil(ln.length / distance)) * 100 for ln in lines)
    main_split = tlz.concat(
        _xs_planar(ln, n_seg, half_width, distance, flw.crs, smoothing)
        for ln, n_seg in zip(lines, n_segments)
    )

    cs = gpd.GeoDataFrame(geometry=list(main_split), crs=flw.crs)
    cs_idx, flw_idx = flw.sindex.query(cs.geometry)
    merged_idx = tlz.merge_with(list, ({t: i} for t, i in zip(flw_idx, cs_idx)))

    cs[id_col] = 0
    for fi, ci in merged_idx.items():
        cs.loc[ci, id_col] = flw.iloc[fi][id_col]
    return cs.drop_duplicates().dissolve(by=id_col)


def network_xsection(
    flw: gpd.GeoDataFrame,
    distance: float,
    width: float,
    id_col: str = "comid",
    smoothing: float | None = None,
) -> gpd.GeoDataFrame:
    """Get cross-section of a river network at a given spacing.

    Parameters
    ----------
    flw : geopandas.GeoDataFrame
        A dataframe with ``geometry`` and, ``id_col``, and ``levelpathi``
        columns and a projected CRS attribute.
    distance : float
        The distance between two consecutive cross-sections.
    width : float
        The width of the cross-section.
    id_col : str, optional
        Name of the flowlines column containing IDs, defaults to ``comid``.
    smoothing : float or None, optional
        Smoothing factor is used for determining the number of knots.
        This arg controls the tradeoff between closeness and smoothness of fit.
        Larger ``smoothing`` means more smoothing while smaller values of
        ``smoothing`` indicates less smoothing. If None (default), smoothing
        is done with all points.

    Returns
    -------
    geopandas.GeoDataFrame
        A dataframe with two columns: ``geometry`` and ``comid``. The ``geometry``
        column contains the cross-section of the river network and the ``comid``
        column contains the corresponding ``comid`` from the input dataframe.
        Note that each ``comid`` can have multiple cross-sections depending on
        the given spacing distance.
    """
    __check_flw(flw, [id_col, "levelpathi", "geometry"])
    cs = pd.concat(  # pyright: ignore[reportCallIssue]
        flowline_xsection(f, distance, width, id_col, smoothing)  # pyright: ignore[reportArgumentType]
        for _, f in flw.groupby("levelpathi")
    )
    return gpd.GeoDataFrame(cs, crs=flw.crs).drop_duplicates().dissolve(by=id_col)


def enhd_flowlines_nx() -> tuple[nx.DiGraph, dict[int, int], list[int]]:
    """Get a ``networkx.DiGraph`` of the entire NHD flowlines.

    .. versionchanged:: 0.16.2

        The function now replaces all 0 values in the ``tocomid`` column of ENHD
        with the negative of their corresponding ``comid`` values. This ensures
        all sinks are unique and treated accordingly for topological sorting
        and other network analysis. The difference are in the returned
        ``label2comid`` dictionary and ``onnetwork_sorted`` which will contain
        the negative values for the sinks.

    Notes
    -----
    The graph is directed and has the all the attributes of the flowlines
    in `ENHD <https://www.sciencebase.gov/catalog/item/63cb311ed34e06fef14f40a3>`__.
    Note that COMIDs are based on the 2020 snapshot of the NHDPlusV2.1.

    Returns
    -------
    graph : networkx.DiGraph
        The generated directed graph
    label2comid : dict
        A mapping of COMIDs to the node IDs in the graph
    onnetwork_sorted : list
        A topologically sorted list of the COMIDs.
    """
    enhd = derived.enhd_attrs()
    id_col, toid_col = "comid", "tocomid"
    tocomid_na = enhd[toid_col] == 0
    enhd.loc[tocomid_na, toid_col] = -enhd.loc[tocomid_na, id_col]
    graph = nx.relabel.convert_node_labels_to_integers(
        nx.from_pandas_edgelist(
            enhd,
            source=id_col,
            target=toid_col,
            create_using=nx.DiGraph,
            edge_attr=enhd.columns.drop([id_col, toid_col]).tolist(),
        ),
        label_attribute="str_id",
    )

    label2comid = nx.get_node_attributes(graph, "str_id")
    s_map = {label2comid[i]: r for i, r in zip(nx.topological_sort(graph), range(len(graph)))}
    onnetwork_sorted = sorted(set(enhd[id_col]).intersection(s_map), key=lambda i: s_map[i])
    return graph, label2comid, onnetwork_sorted  # type: ignore[reportReturnType]


def mainstem_huc12_nx() -> tuple[nx.DiGraph, dict[int, str], list[str]]:
    """Get a ``networkx.DiGraph`` of the entire mainstem HUC12s.

    Notes
    -----
    The directed graph is generated from the ``nhdplusv2wbd.csv`` file with all
    attributes that can be found in
    `Mainstem <https://www.sciencebase.gov/catalog/item/63cb38b2d34e06fef14f40ad>`__.
    Note that HUC12s are based on the 2020 snapshot of the NHDPlusV2.1.

    Returns
    -------
    networkx.DiGraph
        The mainstem as a ``networkx.DiGraph`` with all the attributes of the
        mainstems.
    dict
        A mapping of the HUC12s to the node IDs in the graph.
    list
        A topologically sorted list of the HUC12s which strings of length 12.
    """
    resp = ar.retrieve_text(
        [ScienceBase.get_file_urls("63cb38b2d34e06fef14f40ad").loc["nhdplusv2wbd.csv"].url]
    )
    ms = pd.read_csv(io.StringIO(resp[0]))
    str_cols = ["HUC12", "TOHUC", "head_HUC12", "outlet_HUC12"]
    for col in str_cols:
        ms[col] = (
            pd.to_numeric(ms[col].astype(str), errors="coerce")
            .astype("Int64")
            .fillna(0)
            .astype(str)
            .str.zfill(12)
        )

    int_cols = ["intersected_LevelPathI", "corrected_LevelPathI"]
    ms[int_cols] = ms[int_cols].astype(int)
    zeroidx = ms["TOHUC"] == "000000000000"
    ms.loc[zeroidx, "TOHUC"] = "T" + ms.loc[zeroidx, "HUC12"]

    graph = nx.relabel.convert_node_labels_to_integers(
        nx.from_pandas_edgelist(
            ms,
            source="HUC12",
            target="TOHUC",
            create_using=nx.DiGraph,
            edge_attr=ms.columns.drop(["HUC12", "TOHUC"]).tolist(),
        ),
        label_attribute="str_id",
    )

    label2huc = nx.get_node_attributes(graph, "str_id")
    s_map = {label2huc[i]: r for i, r in zip(nx.topological_sort(graph), range(len(graph)))}
    onnetwork_sorted = sorted(set(ms.HUC12).intersection(s_map), key=lambda i: s_map[i])
    return graph, label2huc, onnetwork_sorted  # type: ignore[reportReturnType]


def nhdplus_l48(
    layer: str | None = None, data_dir: str | Path = "cache", **kwargs: Any
) -> gpd.GeoDataFrame:
    """Get the entire NHDPlus dataset.

    Notes
    -----
    The entire NHDPlus dataset for CONUS (Lower 48) is downloaded from
    `here <https://www.epa.gov/waterdata/nhdplus-national-data>`__.
    This 7.3 GB file will take a while to download, depending on your internet
    connection. The first time you run this function, the file will be downloaded
    and stored in the ``./cache`` directory. Subsequent calls will use the cached
    file. Moreover, there are two additional dependencies required to read the
    file: ``pyogrio`` and ``py7zr``. These dependencies can be installed using
    ``pip install pyogrio py7zr`` or ``conda install -c conda-forge pyogrio py7zr``.

    Parameters
    ----------
    layer : str, optional
        The layer name to be returned. Either ``layer`` should be provided or
        ``sql``. Defaults to ``None``.
        The available layers are:

        - ``Gage``
        - ``BurnAddLine``
        - ``BurnAddWaterbody``
        - ``LandSea``
        - ``Sink``
        - ``Wall``
        - ``Catchment``
        - ``CatchmentSP``
        - ``NHDArea``
        - ``NHDWaterbody``
        - ``HUC12``
        - ``NHDPlusComponentVersions``
        - ``PlusARPointEvent``
        - ``PlusFlowAR``
        - ``NHDFCode``
        - ``DivFracMP``
        - ``BurnLineEvent``
        - ``NHDFlowline_Network``
        - ``NHDFlowline_NonNetwork``
        - ``GeoNetwork_Junctions``
        - ``PlusFlow``
        - ``N_1_Desc``
        - ``N_1_EDesc``
        - ``N_1_EStatus``
        - ``N_1_ETopo``
        - ``N_1_FloDir``
        - ``N_1_JDesc``
        - ``N_1_JStatus``
        - ``N_1_JTopo``
        - ``N_1_JTopo2``
        - ``N_1_Props``

    data_dire : str or pathlib.Pathlib.Path
        Directory to store the downloaded file and use in subsequent calls,
        defaults to ``./cache``.
    **kwargs
        Keyword arguments are passed to ``pyogrio.read_dataframe``.
        For more information, visit
        `pyogrio <https://pyogrio.readthedocs.io/en/latest/introduction.html>`__.

    Returns
    -------
    geopandas.GeoDataFrame
        A dataframe with all the NHDPlus data.
    """
    try:
        import py7zr
        import pyogrio
    except ImportError as ex:
        raise DependencyError("nhdplus_l48", ["pyogrio", "py7zr"]) from ex

    layers = [
        "Gauge",
        "BurnAddLine",
        "BurnAddWaterbody",
        "LandSea",
        "Sink",
        "Wall",
        "Catchment",
        "CatchmentSP",
        "NHDArea",
        "NHDWaterbody",
        "HUC12",
        "NHDPlusComponentVersions",
        "PlusARPointEvent",
        "PlusFlowAR",
        "NHDFCode",
        "DivFracMP",
        "BurnLineEvent",
        "NHDFlowline_Network",
        "NHDFlowline_NonNetwork",
        "GeoNetwork_Junctions",
        "PlusFlow",
        "N_1_Desc",
        "N_1_EDesc",
        "N_1_EStatus",
        "N_1_ETopo",
        "N_1_FloDir",
        "N_1_JDesc",
        "N_1_JStatus",
        "N_1_JTopo",
        "N_1_JTopo2",
        "N_1_Props",
    ]
    if "sql" not in kwargs and layer is None:
        raise InputValueError("layer", layers)

    if layer is not None and layer not in layers:
        raise InputValueError("layer", layers)

    root = Path(data_dir) or Path("cache")
    nhdfile = Path(
        root, "NHDPlusNationalData", "NHDPlusV21_National_Seamless_Flattened_Lower48.gdb"
    )
    if not nhdfile.exists():
        url = "/".join(
            (
                "https://dmap-data-commons-ow.s3.amazonaws.com/NHDPlusV21/Data",
                "NationalData/NHDPlusV21_NationalData_Seamless_Geodatabase_Lower48_07.7z",
            )
        )
        nhd7z = Path(root, Path(url).name)
        _ = streaming_download(url, fnames=nhd7z)
        with py7zr.SevenZipFile(nhd7z, mode="r") as z:
            z.extractall(path=root.resolve())
        nhd7z.unlink()

    pyogrio.set_gdal_config_options({"OGR_ORGANIZE_POLYGONS": "CCW_INNER_JUST_AFTER_CW_OUTER"})
    kwargs["use_arrow"] = kwargs.pop("use_arrow", True)
    nhdp = gpd.read_file(nhdfile, layer=layer, engine="pyogrio", **kwargs)
    if layer in ("NHDFlowline_Network", "NHDFlowline_NonNetwork"):
        nhdp["geometry"] = shapely.force_2d(shapely.line_merge(nhdp.geometry))
    nhdp = cast("gpd.GeoDataFrame", nhdp)
    return nhdp
