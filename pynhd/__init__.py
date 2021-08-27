"""Top-level package for PyNHD."""
from pkg_resources import DistributionNotFound, get_distribution

from .exceptions import InvalidInputRange, MissingItems
from .network_tools import prepare_nhdplus, topoogical_sort, vector_accumulation
from .print_versions import show_versions
from .pynhd import (
    NLDI,
    AGRBase,
    NHDPlusHR,
    PyGeoAPI,
    WaterData,
    nhd_fcode,
    nhdplus_attrs,
    nhdplus_vaa,
)

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    __version__ = "999"

__all__ = [
    "InvalidInputRange",
    "MissingItems",
    "prepare_nhdplus",
    "topoogical_sort",
    "vector_accumulation",
    "show_versions",
    "NLDI",
    "AGRBase",
    "NHDPlusHR",
    "PyGeoAPI",
    "WaterData",
    "nhd_fcode",
    "nhdplus_attrs",
    "nhdplus_vaa",
]
