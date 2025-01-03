"""Top-level package for PyNHD."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from pynhd import exceptions
from pynhd.core import AGRBase, GeoConnex, ScienceBase
from pynhd.network_tools import (
    NHDTools,
    enhd_flowlines_nx,
    flowline_resample,
    flowline_xsection,
    mainstem_huc12_nx,
    network_resample,
    network_xsection,
    nhdflw2nx,
    nhdplus_l48,
    prepare_nhdplus,
    topoogical_sort,
    vector_accumulation,
)
from pynhd.nhdplus_derived import (
    StreamCat,
    enhd_attrs,
    epa_nhd_catchments,
    nhd_fcode,
    nhdplus_attrs,
    nhdplus_attrs_s3,
    nhdplus_h12pp,
    nhdplus_vaa,
    streamcat,
)
from pynhd.print_versions import show_versions
from pynhd.pynhd import HP3D, NHD, NLDI, NHDPlusHR, PyGeoAPI, WaterData, pygeoapi

try:
    __version__ = version("pynhd")
except PackageNotFoundError:
    __version__ = "999"

__all__ = [
    "HP3D",
    "NHD",
    "NLDI",
    "AGRBase",
    "GeoConnex",
    "NHDPlusHR",
    "NHDTools",
    "PyGeoAPI",
    "ScienceBase",
    "StreamCat",
    "WaterData",
    "__version__",
    "enhd_attrs",
    "enhd_flowlines_nx",
    "epa_nhd_catchments",
    "exceptions",
    "flowline_resample",
    "flowline_xsection",
    "mainstem_huc12_nx",
    "network_resample",
    "network_xsection",
    "nhd_fcode",
    "nhdflw2nx",
    "nhdplus_attrs",
    "nhdplus_attrs_s3",
    "nhdplus_h12pp",
    "nhdplus_l48",
    "nhdplus_vaa",
    "prepare_nhdplus",
    "pygeoapi",
    "show_versions",
    "streamcat",
    "topoogical_sort",
    "vector_accumulation",
]
