"""Top-level package for PyNHD."""
from .core import AGRBase
from .exceptions import (
    InvalidInputRange,
    InvalidInputType,
    InvalidInputValue,
    MissingColumns,
    MissingCRS,
    MissingItems,
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
from .pynhd import NHD, NLDI, NHDPlusHR, PyGeoAPI, WaterData, pygeoapi

try:
    import importlib.metadata
except ImportError:
    import importlib_metadata

    __version__ = importlib_metadata.version("pynhd")
else:
    __version__ = importlib.metadata.version("pynhd")

__all__ = [
    "InvalidInputRange",
    "InvalidInputValue",
    "InvalidInputType",
    "MissingItems",
    "MissingColumns",
    "MissingCRS",
    "prepare_nhdplus",
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
