"""Base classes for PyNHD functions."""
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import async_retriever as ar
import cytoolz as tlz
import geopandas as gpd
import pandas as pd
import pygeoutils as geoutils
from pydantic import AnyHttpUrl
from pygeoogc import ArcGISRESTful
from shapely.geometry import Polygon

from .exceptions import InvalidInputValue

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter(""))
logger.handlers = [handler]
logger.propagate = False

DEF_CRS = "epsg:4326"
ALT_CRS = "epsg:4269"
EXPIRE = -1

__all__ = ["AGRBase", "ScienceBase", "stage_nhdplus_attrs"]


def get_parquet(parquet_path: Union[Path, str]) -> Path:
    """Get a parquet filename from a path or a string."""
    if Path(parquet_path).suffix != ".parquet":
        raise InvalidInputValue("parquet_path", ["a filename with `.parquet` extension."])

    output = Path(parquet_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    return output


class AGRBase:
    """Base class for accessing NHD(Plus) HR database through the National Map ArcGISRESTful.

    Parameters
    ----------
    base_url : str, optional
        The ArcGIS RESTful service url. The URL must either include a layer number
        after the last ``/`` in the url or the target layer must be passed as an argument.
    layer : str, optional
        A valid service layer. To see a list of available layers instantiate the class
        without passing any argument.
    outfields : str or list, optional
        Target field name(s), default to "*" i.e., all the fields.
    crs : str, optional
        Target spatial reference, default to EPSG:4326
    outformat : str, optional
        One of the output formats offered by the selected layer. If not correct
        a list of available formats is shown, defaults to ``json``.
    expire_after : int, optional
        Expiration time for response caching in seconds, defaults to -1 (never expire).
    disable_caching : bool, optional
        If ``True``, disable caching requests, defaults to False.
    """

    def __init__(
        self,
        base_url: AnyHttpUrl,
        layer: Optional[str] = None,
        outfields: Union[str, List[str]] = "*",
        crs: str = DEF_CRS,
        outformat: str = "json",
        expire_after: float = EXPIRE,
        disable_caching: bool = False,
    ) -> None:
        self.expire_after = expire_after
        self.disable_caching = disable_caching
        if isinstance(layer, str):
            valid_layers = self.get_validlayers(base_url)
            try:
                self.client = ArcGISRESTful(
                    base_url,
                    valid_layers[layer.lower()],
                    outformat=outformat,
                    outfields=outfields,
                    crs=crs,
                    expire_after=expire_after,
                    disable_caching=disable_caching,
                )
            except KeyError as ex:
                raise InvalidInputValue("layer", list(valid_layers)) from ex
        else:
            self.client = ArcGISRESTful(
                base_url,
                None,
                outformat=outformat,
                outfields=outfields,
                crs=crs,
                expire_after=expire_after,
                disable_caching=disable_caching,
            )

    def get_validlayers(self, url: str) -> Dict[str, int]:
        """Get a list of valid layers.

        Parameters
        ----------
        url : str
            The URL of the ArcGIS REST service.

        Returns
        -------
        dict
            A dictionary of valid layers.
        """
        rjson = ar.retrieve_json(
            [url],
            [{"params": {"f": "json"}}],
            expire_after=self.expire_after,
            disable=self.disable_caching,
        )
        return {lyr["name"].lower(): int(lyr["id"]) for lyr in rjson[0]["layers"]}

    def bygeom(
        self,
        geom: Union[Polygon, List[Tuple[float, float]], Tuple[float, float, float, float]],
        geo_crs: str = DEF_CRS,
        sql_clause: str = "",
        distance: Optional[int] = None,
        return_m: bool = False,
        return_geom: bool = True,
    ) -> gpd.GeoDataFrame:
        """Get feature within a geometry that can be combined with a SQL where clause.

        Parameters
        ----------
        geom : Polygon or tuple
            A geometry (Polygon) or bounding box (tuple of length 4).
        geo_crs : str
            The spatial reference of the input geometry.
        sql_clause : str, optional
            A valid SQL 92 WHERE clause, defaults to an empty string.
        distance : int, optional
            The buffer distance for the input geometries in meters, default to None.
        return_m : bool, optional
            Whether to activate the Return M (measure) in the request, defaults to False.
        return_geom : bool, optional
            Whether to return the geometry of the feature, defaults to ``True``.

        Returns
        -------
        geopandas.GeoDataFrame
            The requested features as a GeoDataFrame.
        """
        oids = self.client.oids_bygeom(
            geom, geo_crs=geo_crs, sql_clause=sql_clause, distance=distance
        )
        return self._getfeatures(oids, return_m, return_geom)

    def byids(
        self,
        field: str,
        fids: Union[str, List[str]],
        return_m: bool = False,
        return_geom: bool = True,
    ) -> gpd.GeoDataFrame:
        """Get features based on a list of field IDs.

        Parameters
        ----------
        field : str
            Name of the target field that IDs belong to.
        fids : str or list
            A list of target field ID(s).
        return_m : bool
            Whether to activate the Return M (measure) in the request, defaults to False.
        return_geom : bool, optional
            Whether to return the geometry of the feature, defaults to ``True``.

        Returns
        -------
        geopandas.GeoDataFrame
            The requested features as a GeoDataFrame.
        """
        oids = self.client.oids_byfield(field, fids)
        return self._getfeatures(oids, return_m, return_geom)

    def bysql(
        self,
        sql_clause: str,
        return_m: bool = False,
        return_geom: bool = True,
    ) -> gpd.GeoDataFrame:
        """Get feature IDs using a valid SQL 92 WHERE clause.

        Notes
        -----
        Not all web services support this type of query. For more details look
        `here <https://developers.arcgis.com/rest/services-reference/query-feature-service-.htm#ESRI_SECTION2_07DD2C5127674F6A814CE6C07D39AD46>`__

        Parameters
        ----------
        sql_clause : str
            A valid SQL 92 WHERE clause.
        return_m : bool
            Whether to activate the measure in the request, defaults to False.
        return_geom : bool, optional
            Whether to return the geometry of the feature, defaults to ``True``.

        Returns
        -------
        geopandas.GeoDataFrame
            The requested features as a GeoDataFrame.
        """
        oids = self.client.oids_bysql(sql_clause)
        return self._getfeatures(oids, return_m, return_geom)

    def _getfeatures(
        self, oids: List[Tuple[str, ...]], return_m: bool = False, return_geom: bool = True
    ) -> gpd.GeoDataFrame:
        """Send a request for getting data based on object IDs.

        Parameters
        ----------
        return_m : bool
            Whether to activate the Return M (measure) in the request, defaults to False.
        return_geom : bool, optional
            Whether to return the geometry of the feature, defaults to ``True``.

        Returns
        -------
        geopandas.GeoDataFrame
            The requested features as a GeoDataFrame.
        """
        return geoutils.json2geodf(
            self.client.get_features(oids, return_m, return_geom), self.client.client.crs
        )

    def __repr__(self) -> str:
        """Print the service configuration."""
        return self.client.__repr__()


@dataclass
class ScienceBase:
    """Access and explore files on ScienceBase.

    Parameters
    ----------
    expire_after : int, optional
        Expiration time for response caching in seconds, defaults to -1 (never expire).
    disable_caching : bool, optional
        If ``True``, disable caching requests, defaults to False.
    """

    expire_after: float = EXPIRE
    disable_caching: bool = False

    def get_children(self, item: str) -> Dict[str, Any]:
        """Get children items of an item."""
        url = "https://www.sciencebase.gov/catalog/items"
        payload = {
            "filter": f"parentIdExcludingLinks={item}",
            "fields": "title,id",
            "format": "json",
        }
        resp = ar.retrieve_json(
            [url],
            [{"params": payload}],
            expire_after=self.expire_after,
            disable=self.disable_caching,
        )
        return resp[0]

    def get_file_urls(self, item: str) -> pd.DataFrame:
        """Get download and meta URLs of all the available files for an item."""
        url = "https://www.sciencebase.gov/catalog/item"
        payload = {"fields": "files,downloadUri", "format": "json"}
        resp = ar.retrieve_json(
            [f"{url}/{item}"],
            [{"params": payload}],
            expire_after=self.expire_after,
            disable=self.disable_caching,
        )
        urls = zip(
            tlz.pluck("name", resp[0]["files"]),
            tlz.pluck("url", resp[0]["files"]),
            tlz.pluck("metadataHtmlViewUri", resp[0]["files"], default=None),
        )
        files = pd.DataFrame(urls, columns=["name", "url", "metadata_url"])
        return files.set_index("name")


def stage_nhdplus_attrs(
    parquet_path: Optional[Union[Path, str]] = None,
    expire_after: float = EXPIRE,
    disable_caching: bool = False,
) -> pd.DataFrame:
    """Stage the NHDPlus Attributes database and save to nhdplus_attrs.parquet.

    More info can be found `here <https://www.sciencebase.gov/catalog/item/5669a79ee4b08895842a1d47>`_.

    Parameters
    ----------
    parquet_path : str or Path
        Path to a file with ``.parquet`` extension for saving the processed to disk for
        later use.
    expire_after : int, optional
        Expiration time for response caching in seconds, defaults to -1 (never expire).
    disable_caching : bool, optional
        If ``True``, disable caching requests, defaults to False.

    Returns
    -------
    pandas.DataFrame
        The staged data as a DataFrame.
    """
    if parquet_path is None:
        output = get_parquet(Path("cache", "nhdplus_attrs.parquet"))
    else:
        output = get_parquet(parquet_path)

    sb = ScienceBase(expire_after, disable_caching)
    r = sb.get_children("5669a79ee4b08895842a1d47")

    titles = tlz.pluck("title", r["items"])
    titles = tlz.concat(tlz.map(tlz.partial(re.findall, "Select(.*?)Attributes"), titles))
    titles = tlz.map(str.strip, titles)

    main_items = dict(zip(titles, tlz.pluck("id", r["items"])))

    def get_files(item: str) -> Dict[str, Tuple[str, str]]:
        """Get all the available zip files in an item."""
        url = "https://www.sciencebase.gov/catalog/item"
        payload = {"fields": "files,downloadUri", "format": "json"}
        resp = ar.retrieve_json(
            [f"{url}/{item}"],
            [{"params": payload}],
            expire_after=expire_after,
            disable=disable_caching,
        )
        files_url = zip(tlz.pluck("name", resp[0]["files"]), tlz.pluck("url", resp[0]["files"]))
        meta = list(tlz.pluck("metadataHtmlViewUri", resp[0]["files"], default=""))[-1]
        return {f.replace("_CONUS.zip", ""): (u, meta) for f, u in files_url if ".zip" in f}

    files = {}
    soil = main_items.pop("Soil")
    for i, item in main_items.items():
        r = sb.get_children(item)

        titles = tlz.pluck("title", r["items"])
        titles = tlz.map(lambda s: s.split(":")[1].strip() if ":" in s else s, titles)

        child_items = dict(zip(titles, tlz.pluck("id", r["items"])))
        files[i] = {t: get_files(c) for t, c in child_items.items()}

    r = sb.get_children(soil)
    titles = tlz.pluck("title", r["items"])
    titles = tlz.map(lambda s: s.split(":")[1].strip() if ":" in s else s, titles)

    child_items = dict(zip(titles, tlz.pluck("id", r["items"])))
    stat = child_items.pop("STATSGO Soil Characteristics")
    ssur = child_items.pop("SSURGO Soil Characteristics")
    files["Soil"] = {t: get_files(c) for t, c in child_items.items()}

    r = sb.get_children(stat)
    titles = tlz.pluck("title", r["items"])
    titles = tlz.map(lambda s: s.split(":")[1].split(",")[1].strip(), titles)
    child_items = dict(zip(titles, tlz.pluck("id", r["items"])))
    files["STATSGO"] = {t: get_files(c) for t, c in child_items.items()}

    r = sb.get_children(ssur)
    titles = tlz.pluck("title", r["items"])
    titles = tlz.map(lambda s: s.split(":")[1].strip(), titles)
    child_items = dict(zip(titles, tlz.pluck("id", r["items"])))
    files["SSURGO"] = {t: get_files(c) for t, c in child_items.items()}

    chars = []
    types = {"CAT": "local", "TOT": "upstream_acc", "ACC": "div_routing"}
    for t, dd in files.items():
        for d, fd in dd.items():
            for f, u in fd.items():
                chars.append(
                    {
                        "name": f,
                        "type": types.get(f[-3:], "other"),
                        "theme": t,
                        "description": d,
                        "url": u[0],
                        "meta": u[1],
                    }
                )
    char_df = pd.DataFrame(chars)
    char_df.to_parquet(output)
    return char_df
