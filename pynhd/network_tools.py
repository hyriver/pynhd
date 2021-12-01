"""Access NLDI and WaterData databases."""
import numbers
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from pandas._libs.missing import NAType
from pygeoutils import InvalidInputType

from . import nhdplus_derived as derived
from .core import logger
from .exceptions import MissingItems

__all__ = ["prepare_nhdplus", "topoogical_sort", "vector_accumulation"]


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

        Returns
        -------
        geopandas.GeoDataFrame
            The input dataframe With an additional column named ``tocomid``.
        """
        req_cols = ["comid", "terminalpa", "fromnode", "tonode"]
        self.check_requirements(req_cols, self.flw)
        self.flw[req_cols] = self.flw[req_cols].astype("Int64")

        def tocomid(group: pd.core.groupby.generic.DataFrameGroupBy) -> pd.DataFrame:
            def toid(row: pd.DataFrame) -> pd.Int64Dtype:
                try:
                    return group[group.fromnode == row.tonode].comid.to_numpy()[0]
                except IndexError:
                    return pd.NA

            return group.apply(toid, axis=1)

        self.flw["tocomid"] = pd.concat(tocomid(g) for _, g in self.flw.groupby("terminalpa"))

    @staticmethod
    def check_requirements(reqs: Iterable, cols: List[str]) -> None:
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


def topoogical_sort(
    flowlines: pd.DataFrame, edge_attr: Optional[Union[str, List[str]]] = None
) -> Tuple[List[Union[str, NAType]], Dict[Union[str, NAType], List[str]], nx.DiGraph]:
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
    upstream_nodes = {i: flowlines[flowlines.toID == i].ID.tolist() for i in flowlines.ID.tolist()}
    upstream_nodes[pd.NA] = flowlines[flowlines.toID.isna()].ID.tolist()

    network = nx.from_pandas_edgelist(
        flowlines,
        source="ID",
        target="toID",
        create_using=nx.DiGraph,
        edge_attr=edge_attr,
    )
    topo_sorted = list(nx.topological_sort(network))
    return topo_sorted, upstream_nodes, network


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
    sorted_nodes, upstream_nodes, _ = topoogical_sort(
        flowlines[[id_col, toid_col]].rename(columns={id_col: "ID", toid_col: "toID"})
    )
    topo_sorted = sorted_nodes[:-1]

    outflow = flowlines.set_index(id_col)[attr_col].to_dict()

    init = flowlines.iloc[0][attr_col]

    if isinstance(init, numbers.Number):
        outflow["0"] = 0.0
    else:
        outflow["0"] = np.zeros_like(init)

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
