"""Top-level package for PyNHD."""
from .core import AGRBase
from .exceptions import InvalidInputRange, MissingItems
from .network_tools import prepare_nhdplus, topoogical_sort, vector_accumulation
from .nhdplus_dervived import nhd_fcode, nhdplus_attrs, nhdplus_vaa
from .print_versions import show_versions
from .pynhd import NLDI, NHDPlusHR, PyGeoAPI, WaterData

try:
    import importlib.metadata as metadata
except ImportError:
    import importlib_metadata as metadata  # type: ignore[no-redef]

try:
    __version__ = metadata.version("async_retriever")
except Exception:
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
