"""Top-level package for PyNHD."""
from pkg_resources import DistributionNotFound, get_distribution

from .exceptions import InvalidInputType, InvalidInputValue, MissingItems, ZeroMatched
from .print_versions import show_versions
from .pynhd import NLDI, NHDPlusHR, WaterData, prepare_nhdplus, topoogical_sort, vector_accumulation

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass
