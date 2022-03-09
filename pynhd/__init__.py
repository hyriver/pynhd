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
    flowline_xsection,
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
    import importlib.metadata as metadata
except ImportError:
    import importlib_metadata as metadata  # type: ignore[no-redef]

try:
    __version__ = metadata.version("pynhd")
except Exception:
    __version__ = "999"

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
