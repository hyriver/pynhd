"""Top-level package for PyNHD."""
from pkg_resources import DistributionNotFound, get_distribution

from .exceptions import InvalidInputRange, MissingItems, ZeroMatched
from .network_tools import prepare_nhdplus, topoogical_sort, vector_accumulation
from .print_versions import show_versions
from .pynhd import NLDI, AGRBase, NHDPlusHR, WaterData, nhd_fcode, nhdplus_attrs, nhdplus_vaa

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass
