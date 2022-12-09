"""Top-level package for PyNHD."""
from importlib.metadata import PackageNotFoundError, version

from .core import AGRBase, GeoConnex, ScienceBase
from .exceptions import (
    DependencyError,
    InputRangeError,
    InputTypeError,
    InputValueError,
    MissingColumnError,
    MissingCRSError,
    MissingItemError,
    ZeroMatchedError,
)
from .network_tools import (
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
from .nhdplus_derived import (
    enhd_attrs,
    epa_nhd_catchments,
    nhd_fcode,
    nhdplus_attrs,
    nhdplus_attrs_s3,
    nhdplus_vaa,
)
from .print_versions import show_versions
from .pynhd import NHD, NLDI, NHDPlusHR, PyGeoAPI, WaterData, geoconnex, pygeoapi

try:
    __version__ = version("pynhd")
except PackageNotFoundError:
    __version__ = "999"

__all__ = [
    "DependencyError",
    "InputRangeError",
    "InputValueError",
    "InputTypeError",
    "MissingItemError",
    "MissingColumnError",
    "MissingCRSError",
    "ZeroMatchedError",
    "prepare_nhdplus",
    "geoconnex",
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
    "show_versions",
    "NLDI",
    "AGRBase",
    "GeoConnex",
    "ScienceBase",
    "NHDPlusHR",
    "NHD",
    "PyGeoAPI",
    "pygeoapi",
    "WaterData",
    "nhd_fcode",
    "nhdplus_attrs",
    "nhdplus_attrs_s3",
    "epa_nhd_catchments",
    "enhd_attrs",
    "nhdplus_vaa",
]
