"""Top-level package for PyNHD."""
from importlib.metadata import PackageNotFoundError, version

from .core import AGRBase, GeoConnex
from .exceptions import (
    InputRangeError,
    InputTypeError,
    InputValueError,
    MissingColumnError,
    MissingCRSError,
    MissingItemError,
    ZeroMatchedError,
)
from .network_tools import (
    flowline_resample,
    flowline_xsection,
    network_resample,
    network_xsection,
    nhdflw2nx,
    prepare_nhdplus,
    topoogical_sort,
    vector_accumulation,
)
from .nhdplus_derived import enhd_attrs, nhd_fcode, nhdplus_attrs, nhdplus_vaa
from .print_versions import show_versions
from .pynhd import NHD, NLDI, NHDPlusHR, PyGeoAPI, WaterData, geoconnex, pygeoapi

try:
    __version__ = version("pynhd")
except PackageNotFoundError:
    __version__ = "999"

__all__ = [
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
    "show_versions",
    "NLDI",
    "AGRBase",
    "GeoConnex",
    "NHDPlusHR",
    "NHD",
    "PyGeoAPI",
    "pygeoapi",
    "WaterData",
    "nhd_fcode",
    "nhdplus_attrs",
    "enhd_attrs",
    "nhdplus_vaa",
]
