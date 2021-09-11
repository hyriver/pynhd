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
from pygeoogc import (
    ArcGISRESTful,
    InvalidInputValue,
    ServiceError
)
from pygeoutils import InvalidInputType
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter(""))
logger.handlers = [handler]
logger.propagate = False
DEF_CRS = "epsg:4326"


@dataclass
class AGRBase:
    """Base class for accessing NHD(Plus) HR database through the National Map ArcGISRESTful.

    Parameters
    ----------
    layer : str, optional
        A valid service layer. To see a list of available layers instantiate the class
        without passing any argument.
    outfields : str or list, optional
        Target field name(s), default to "*" i.e., all the fields.
    crs : str, optional
        Target spatial reference, default to EPSG:4326
    """

    layer: Optional[str] = None
    outfields: Union[str, List[str]] = "*"
    crs: str = DEF_CRS

    @property
    def service(self) -> ArcGISRESTful:
        """Connect to a RESTFul service."""
        return self._service

    @service.setter
    def service(self, value: str) -> None:
        valid_layers = self.get_validlayers(value)
        if self.layer is None:
            raise InvalidInputType("layer", "str")

        if self.layer.lower() not in valid_layers:
            raise InvalidInputValue("layer", list(valid_layers))

        self._service = ArcGISRESTful(
            value,
            valid_layers[self.layer.lower()],
            outformat="json",
            outfields=self.outfields,
            crs=self.crs,
        )

    @staticmethod
    def get_validlayers(url):
        """Get valid layer for a ArcGISREST service."""
        try:
            rjson = ar.retrieve([url], "json", [{"params": {"f": "json"}}])[0]
        except (ar.ServiceError, KeyError) as ex:
            raise ar.ServiceError(url) from ex
        else:
            return {lyr["name"].lower(): lyr["id"] for lyr in rjson["layers"]}

    def connect_to(self, service: str, service_list: Dict[str, str], auto_switch: bool) -> None:
        """Connect to a web service.

        Parameters
        ----------
        service : str
            Name of the preferred web service to connect to from the list provided in service_list.
        service_list: dict
            A dict where keys are names of the web services and values are their URLs.
        auto_switch : bool, optional
            Automatically switch to other services' URL if the first one doesn't work, default to False.
        """
        if service not in service_list:
            raise InvalidInputValue("service", list(service_list))

        url = service_list.pop(service)
        try:
            self.service = url
        except (ServiceError, ConnectionError) as ex:
            if not auto_switch:
                raise ServiceError(url) from ex

            while len(service_list) > 0:
                next_service = next(iter(service_list.keys()))
                logger.warning(f"Connection to {url} failed. Will try {next_service} next ...")
                try:
                    url = service_list.pop(next_service)
                    self.service = url
                    logger.info(f"Connected to {url}.")
                except (ServiceError, ConnectionError):
                    continue

        if self.service is None:
            raise ServiceError(url)

    def bygeom(
        self,
        geom: Union[Polygon, List[Tuple[float, float]], Tuple[float, float, float, float]],
        geo_crs: str = DEF_CRS,
        sql_clause: str = "",
        distance: Optional[int] = None,
        return_m: bool = False,
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

        Returns
        -------
        geopandas.GeoDataFrame
            The requested features as a GeoDataFrame.
        """
        self.service.oids_bygeom(geom, geo_crs=geo_crs, sql_clause=sql_clause, distance=distance)
        return self._getfeatures(return_m)

    def byids(
        self, field: str, fids: Union[str, List[str]], return_m: bool = False
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

        Returns
        -------
        geopandas.GeoDataFrame
            The requested features as a GeoDataFrame.
        """
        self.service.oids_byfield(field, fids)
        return self._getfeatures(return_m)

    def bysql(self, sql_clause: str, return_m: bool = False) -> gpd.GeoDataFrame:
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
            Whether to activate the Return M (measure) in the request, defaults to False.

        Returns
        -------
        geopandas.GeoDataFrame
            The requested features as a GeoDataFrame.
        """
        self.service.oids_bysql(sql_clause)
        return self._getfeatures(return_m)

    def _getfeatures(self, return_m: bool = False) -> gpd.GeoDataFrame:
        """Send a request for getting data based on object IDs.

        Parameters
        ----------
        return_m : bool
            Whether to activate the Return M (measure) in the request, defaults to False.

        Returns
        -------
        geopandas.GeoDataFrame
            The requested features as a GeoDataFrame.
        """
        return geoutils.json2geodf(self.service.get_features(return_m))


class ScienceBase:
    """Access NHDPlus V2.1 Attributes from ScienceBase over CONUS.

    More info can be found `here <https://www.sciencebase.gov/catalog/item/5669a79ee4b08895842a1d47>`_.

    Parameters
    ----------
    save_dir : str
        Directory to save the staged data frame containing metadata for the database,
        defaults to system's temp directory. The metadata dataframe is saved as a feather
        file, nhdplus_attrs.feather, in save_dir that can be loaded with Pandas.
    """

    def __init__(self, save_dir: Optional[str] = None) -> None:
        self.save_dir = Path(save_dir) if save_dir else Path("cache")
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.nhd_attr_item = "5669a79ee4b08895842a1d47"
        self.char_feather = Path(self.save_dir, "nhdplus_attrs.feather")

    @staticmethod
    def get_children(item: str) -> Dict[str, Any]:
        """Get children items of an item."""
        url = "https://www.sciencebase.gov/catalog/items"
        payload = {
            "filter": f"parentIdExcludingLinks={item}",
            "fields": "title,id",
            "format": "json",
        }
        return ar.retrieve([url], "json", [{"params": payload}])[0]

    @staticmethod
    def get_files(item: str) -> Dict[str, Tuple[str, str]]:
        """Get all the available zip files in an item."""
        url = "https://www.sciencebase.gov/catalog/item"
        payload = {"fields": "files,downloadUri", "format": "json"}
        r = ar.retrieve([f"{url}/{item}"], "json", [{"params": payload}])[0]
        files_url = zip(tlz.pluck("name", r["files"]), tlz.pluck("url", r["files"]))
        meta = list(tlz.pluck("metadataHtmlViewUri", r["files"], default=""))[-1]
        return {f.replace("_CONUS.zip", ""): (u, meta) for f, u in files_url if ".zip" in f}

    def stage_data(self) -> pd.DataFrame:
        """Stage the NHDPlus Attributes database and save to nhdplus_attrs.feather."""
        r = self.get_children(self.nhd_attr_item)

        titles = tlz.pluck("title", r["items"])
        titles = tlz.concat(tlz.map(tlz.partial(re.findall, "Select(.*?)Attributes"), titles))
        titles = tlz.map(str.strip, titles)

        main_items = dict(zip(titles, tlz.pluck("id", r["items"])))

        files = {}
        soil = main_items.pop("Soil")
        for i, item in main_items.items():
            r = self.get_children(item)

            titles = tlz.pluck("title", r["items"])
            titles = tlz.map(lambda s: s.split(":")[1].strip() if ":" in s else s, titles)

            child_items = dict(zip(titles, tlz.pluck("id", r["items"])))
            files[i] = {t: self.get_files(c) for t, c in child_items.items()}

        r = self.get_children(soil)
        titles = tlz.pluck("title", r["items"])
        titles = tlz.map(lambda s: s.split(":")[1].strip() if ":" in s else s, titles)

        child_items = dict(zip(titles, tlz.pluck("id", r["items"])))
        stat = child_items.pop("STATSGO Soil Characteristics")
        ssur = child_items.pop("SSURGO Soil Characteristics")
        files["Soil"] = {t: self.get_files(c) for t, c in child_items.items()}

        r = self.get_children(stat)
        titles = tlz.pluck("title", r["items"])
        titles = tlz.map(lambda s: s.split(":")[1].split(",")[1].strip(), titles)
        child_items = dict(zip(titles, tlz.pluck("id", r["items"])))
        files["STATSGO"] = {t: self.get_files(c) for t, c in child_items.items()}

        r = self.get_children(ssur)
        titles = tlz.pluck("title", r["items"])
        titles = tlz.map(lambda s: s.split(":")[1].strip(), titles)
        child_items = dict(zip(titles, tlz.pluck("id", r["items"])))
        files["SSURGO"] = {t: self.get_files(c) for t, c in child_items.items()}

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
        char_df.to_feather(self.char_feather)
        return char_df