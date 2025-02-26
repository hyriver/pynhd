"""Access NLDI and WaterData databases."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal, cast

import pygeoogc as ogc
import pygeoutils as geoutils
from pygeoogc import WFS, ServiceURL
from pygeoutils.exceptions import EmptyResponseError
from pynhd.core import AGRBase
from pynhd.exceptions import (
    InputTypeError,
    InputValueError,
    ZeroMatchedError,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import geopandas as gpd
    from pyproj import CRS
    from shapely import LineString, MultiPoint, MultiPolygon, Point, Polygon

    CRSType = int | str | CRS
    GeoType = (
        Polygon | MultiPolygon | MultiPoint | LineString | Point | tuple[float, float, float, float]
    )
    NHDLayers = Literal[
        "point",
        "point_event",
        "line_hr",
        "flow_direction",
        "flowline_mr",
        "flowline_hr_nonconus",
        "flowline_hr",
        "area_mr",
        "area_hr_nonconus",
        "area_hr",
        "waterbody_mr",
        "waterbody_hr_nonconus",
        "waterbody_hr",
    ]
    WDLayers = Literal[
        "catchmentsp",
        "gagesii",
        "gagesii_basins",
        "huc08",
        "huc12",
        "nhdarea",
        "nhdflowline_network",
        "nhdflowline_nonnetwork",
        "nhdwaterbody",
        "wbd02",
        "wbd04",
        "wbd06",
        "wbd08",
        "wbd10",
        "wbd12",
    ]
    Predicates = Literal[
        "equals",
        "disjoint",
        "intersects",
        "touches",
        "crosses",
        "within",
        "contains",
        "overlaps",
        "relate",
        "beyond",
    ]
    NHDHRLayers = Literal[
        "gauge",
        "sink",
        "point",
        "flowline",
        "non_network_flowline",
        "flow_direction",
        "wall",
        "line",
        "area",
        "waterbody",
        "catchment",
        "boundary_unit",
        "huc12",
    ]
    HP3DLayers = Literal[
        "hydrolocation_waterbody",
        "hydrolocation_flowline",
        "hydrolocation_reach",
        "flowline",
        "waterbody",
        "drainage_area",
        "catchment",
    ]

__all__ = ["HP3D", "NHD", "NHDPlusHR", "WaterData"]


class NHD(AGRBase):
    """Access National Hydrography Dataset (NHD), both meduim and high resolution.

    Notes
    -----
    For more info visit: https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer

    Parameters
    ----------
    layer : str, optional
        A valid service layer. Layer names with ``_hr`` are high resolution and
        ``_mr`` are medium resolution. Also, layer names with ``_nonconus`` are for
        non-conus areas, i.e., Alaska, Hawaii, Puerto Rico, the Virgin Islands , and
        the Pacific Islands. Valid layers are:

        - ``point``
        - ``point_event``
        - ``line_hr``
        - ``flow_direction``
        - ``flowline_mr``
        - ``flowline_hr_nonconus``
        - ``flowline_hr``
        - ``area_mr``
        - ``area_hr_nonconus``
        - ``area_hr``
        - ``waterbody_mr``
        - ``waterbody_hr_nonconus``
        - ``waterbody_hr``

    outfields : str or list, optional
        Target field name(s), default to "*" i.e., all the fields.
    crs : str, int, or pyproj.CRS, optional
        Target spatial reference, default to ``EPSG:4326``.

    Methods
    -------
    bygeom(geom, geo_crs=4326, sql_clause="", distance=None, return_m=False, return_geom=True)
        Get features within a geometry that can be combined with a SQL where clause.
    byids(field, fids, return_m=False, return_geom=True)
        Get features by object IDs.
    bysql(sql_clause, return_m=False, return_geom=True)
        Get features using a valid SQL 92 WHERE clause.
    """

    def __init__(
        self,
        layer: NHDLayers,
        outfields: str | list[str] = "*",
        crs: CRSType = 4326,
    ):
        self.valid_layers = {
            "point": "point",
            "point_event": "point event",
            "line_hr": "line - large scale ",
            "flow_direction": "flow direction",
            "flowline_mr": "flowline - small scale",
            "flowline_hr_nonconus": "flowline - small scale (hi, pr, vi, pacific territories)",
            "flowline_hr": "flowline - large scale",
            "area_mr": "area - small scale",
            "area_hr_nonconus": "area - small scale (hi, pr, vi, pacific territories)",
            "area_hr": "area - large scale",
            "waterbody_mr": "waterbody - small scale",
            "waterbody_hr_nonconus": "waterbody - small scale (hi, pr, vi, pacific territories)",
            "waterbody_hr": "waterbody - large scale",
        }
        _layer = self.valid_layers.get(layer)
        if _layer is None:
            raise InputValueError("layer", list(self.valid_layers))
        super().__init__(
            ServiceURL().restful.nhd,
            _layer,
            outfields,
            crs,
        )


class HP3D(AGRBase):
    """Access USGS 3D Hydrography Program (3DHP) service.

    Notes
    -----
    For more info visit: https://hydro.nationalmap.gov/arcgis/rest/services/3DHP_all/MapServer

    Parameters
    ----------
    layer : str, optional
        A valid service layer. Layer names with ``_hr`` are high resolution and
        ``_mr`` are medium resolution. Also, layer names with ``_nonconus`` are for
        non-conus areas, i.e., Alaska, Hawaii, Puerto Rico, the Virgin Islands , and
        the Pacific Islands. Valid layers are:

        - ``hydrolocation_waterbody`` for Sink, Spring, Waterbody Outlet
        - ``hydrolocation_flowline`` for Headwater, Terminus, Divergence, Confluence, Catchment Outlet
        - ``hydrolocation_reach`` for Reach Code, External Connection
        - ``flowline`` for river flowlines
        - ``waterbody`` for waterbodies
        - ``drainage_area`` for drainage areas
        - ``catchment`` for catchments

    outfields : str or list, optional
        Target field name(s), default to "*" i.e., all the fields.
    crs : str, int, or pyproj.CRS, optional
        Target spatial reference, default to ``EPSG:4326``.

    Methods
    -------
    bygeom(geom, geo_crs=4326, sql_clause="", distance=None, return_m=False, return_geom=True)
        Get features within a geometry that can be combined with a SQL where clause.
    byids(field, fids, return_m=False, return_geom=True)
        Get features by object IDs.
    bysql(sql_clause, return_m=False, return_geom=True)
        Get features using a valid SQL 92 WHERE clause.
    """

    def __init__(
        self,
        layer: HP3DLayers,
        outfields: str | list[str] = "*",
        crs: CRSType = 4326,
    ):
        self.valid_layers = {
            "hydrolocation_waterbody": "Sink, Spring, Waterbody Outlet",
            "hydrolocation_flowline": "Headwater, Terminus, Divergence, Confluence, Catchment Outlet",
            "hydrolocation_reach": "Reach Code, External Connection",
            "flowline": "Flowline",
            "waterbody": "Waterbody",
            "drainage_area": "Drainage Area",
            "catchment": "Catchment",
        }
        _layer = self.valid_layers.get(layer)
        if _layer is None:
            raise InputValueError("layer", list(self.valid_layers))
        super().__init__(
            ServiceURL().restful.hp3d,
            _layer,
            outfields,
            crs,
        )


class WaterData:
    """Access `WaterData <https://api.water.usgs.gov/geoserver>`__ service.

    Parameters
    ----------
    layer : str
        A valid layer from the WaterData service. Valid layers are:

        - ``catchmentsp``
        - ``gagesii``
        - ``gagesii_basins``
        - ``nhdarea``
        - ``nhdflowline_network``
        - ``nhdflowline_nonnetwork``
        - ``nhdwaterbody``
        - ``wbd02``
        - ``wbd04``
        - ``wbd06``
        - ``wbd08``
        - ``wbd10``
        - ``wbd12``

        Note that all ``wbd*`` layers provide access to the October 2020
        snapshot of the Watershed Boundary Dataset (WBD). If you need the
        latest version, please use the ``WBD`` class from the
        `PyGeoHydro <https://docs.hyriver.io/autoapi/pygeohydro/watershed/index.html#pygeohydro.watershed.WBD>`__
        package.
    crs : str, int, or pyproj.CRS, optional
        The target spatial reference system, defaults to ``epsg:4326``.
    """

    def __init__(
        self,
        layer: WDLayers,
        crs: CRSType = 4326,
    ) -> None:
        self.valid_layers = [
            "catchmentsp",
            "gagesii",
            "gagesii_basins",
            "huc08",
            "huc12",
            "nhdarea",
            "nhdflowline_network",
            "nhdflowline_nonnetwork",
            "nhdwaterbody",
            "wbd02",
            "wbd04",
            "wbd06",
            "wbd08",
            "wbd10",
            "wbd12",
        ]
        if layer not in self.valid_layers:
            raise InputValueError("layer", self.valid_layers)
        self.layer = f"wmadata:{layer}"
        if "wbd" in self.layer:
            self.layer = f"{self.layer}_20201006"
        self.crs = crs
        self.wfs = WFS(
            ServiceURL().wfs.waterdata,
            layer=self.layer,
            outformat="application/json",
            version="2.0.0",
            crs=4269,
            validation=False,
        )

    def _to_geodf(self, resp: list[dict[str, Any]]) -> gpd.GeoDataFrame:
        """Convert a response from WaterData to a GeoDataFrame."""
        try:
            features = geoutils.json2geodf(resp, self.wfs.crs, self.crs)
        except EmptyResponseError as ex:
            raise ZeroMatchedError from ex

        if features.empty:
            raise ZeroMatchedError
        return features

    def bybox(
        self,
        bbox: tuple[float, float, float, float],
        box_crs: CRSType = 4326,
        sort_attr: str | None = None,
    ) -> gpd.GeoDataFrame:
        """Get features within a bounding box.

        Parameters
        ----------
        bbox : tuple of floats
            A bounding box in the form of (minx, miny, maxx, maxy).
        box_crs : str, int, or pyproj.CRS, optional
            The spatial reference system of the bounding box, defaults to ``epsg:4326``.
        sort_attr : str, optional
            The column name in the database to sort request by, defaults
            to the first attribute in the schema that contains ``id`` in its name.

        Returns
        -------
        geopandas.GeoDataFrame
            The requested features in a GeoDataFrames.
        """
        resp = self.wfs.getfeature_bybox(
            bbox,
            box_crs,
            always_xy=True,
            sort_attr=sort_attr,
        )
        resp = cast("list[dict[str, Any]]", resp)
        return self._to_geodf(resp)

    def bygeom(
        self,
        geometry: Polygon | MultiPolygon,
        geo_crs: CRSType = 4326,
        xy: bool = True,
        predicate: Predicates = "intersects",
        sort_attr: str | None = None,
    ) -> gpd.GeoDataFrame:
        """Get features within a geometry.

        Parameters
        ----------
        geometry : shapely.Polygon or shapely.MultiPolygon
            The input (multi)polygon to request the data.
        geo_crs : str, int, or pyproj.CRS, optional
            The CRS of the input geometry, default to epsg:4326.
        xy : bool, optional
            Whether axis order of the input geometry is xy or yx.
        predicate : str, optional
            The geometric prediacte to use for requesting the data, defaults to
            INTERSECTS. Valid predicates are:

            - ``equals``
            - ``disjoint``
            - ``intersects``
            - ``touches``
            - ``crosses``
            - ``within``
            - ``contains``
            - ``overlaps``
            - ``relate``
            - ``beyond``

        sort_attr : str, optional
            The column name in the database to sort request by, defaults
            to the first attribute in the schema that contains ``id`` in its name.

        Returns
        -------
        geopandas.GeoDataFrame
            The requested features in the given geometry.
        """
        resp = self.wfs.getfeature_bygeom(
            geometry, geo_crs, always_xy=not xy, predicate=predicate.upper(), sort_attr=sort_attr
        )
        resp = cast("list[dict[str, Any]]", resp)
        return self._to_geodf(resp)

    def bydistance(
        self,
        coords: tuple[float, float],
        distance: int,
        loc_crs: CRSType = 4326,
        sort_attr: str | None = None,
    ) -> gpd.GeoDataFrame:
        """Get features within a radius (in meters) of a point.

        Parameters
        ----------
        coords : tuple of float
            The x, y coordinates of the point.
        distance : int
            The radius (in meters) to search within.
        loc_crs : str, int, or pyproj.CRS, optional
            The CRS of the input coordinates, default to ``epsg:4326``.
        sort_attr : str, optional
            The column name in the database to sort request by, defaults
            to the first attribute in the schema that contains ``id`` in its name.

        Returns
        -------
        geopandas.GeoDataFrame
            Requested features as a GeoDataFrame.
        """
        if not (isinstance(coords, tuple) and len(coords) == 2):
            raise InputTypeError("coods", "tuple of length 2", "(x, y)")

        x, y = ogc.match_crs([coords], loc_crs, self.wfs.crs)[0]
        geom_name = self.wfs.schema[self.layer].get("geometry_column", "the_geom")
        cql_filter = f"DWITHIN({geom_name},POINT({y:.6f} {x:.6f}),{distance},meters)"
        resp = self.wfs.getfeature_byfilter(
            cql_filter,
            "GET",
            sort_attr=sort_attr,
        )
        resp = cast("list[dict[str, Any]]", resp)
        return self._to_geodf(resp)

    def byid(
        self, featurename: str, featureids: Sequence[int | str] | int | str
    ) -> gpd.GeoDataFrame:
        """Get features based on IDs."""
        resp = self.wfs.getfeature_byid(
            featurename,
            featureids,
        )
        resp = cast("list[dict[str, Any]]", resp)
        features = self._to_geodf(resp)

        if isinstance(featureids, (str, int)):
            fids = [str(featureids)]
        else:
            fids = [str(f) for f in featureids]

        failed = set(fids).difference(set(features[featurename].astype(str)))

        if failed:
            msg = ". ".join(
                (
                    f"{len(failed)} of {len(fids)} requests failed.",
                    f"IDs of the failed requests are {list(failed)}",
                )
            )
            warnings.warn(msg, UserWarning, stacklevel=2)
        return features

    def byfilter(
        self,
        cql_filter: str,
        method: Literal["GET", "get", "POST", "post"] = "GET",
        sort_attr: str | None = None,
    ) -> gpd.GeoDataFrame:
        """Get features based on a CQL filter.

        Parameters
        ----------
        cql_filter : str
            The CQL filter to use for requesting the data.
        method : str, optional
            The HTTP method to use for requesting the data, defaults to GET.
            Allowed methods are GET and POST.
        sort_attr : str, optional
            The column name in the database to sort request by, defaults
            to the first attribute in the schema that contains ``id`` in its name.

        Returns
        -------
        geopandas.GeoDataFrame
            The requested features as a GeoDataFrames.
        """
        resp = self.wfs.getfeature_byfilter(
            cql_filter,
            method,
            sort_attr,
        )
        resp = cast("list[dict[str, Any]]", resp)
        return self._to_geodf(resp)

    def __repr__(self) -> str:
        """Print the services properties."""
        return "\n".join(
            (
                "Connected to the WaterData web service on GeoServer:",
                f"URL: {self.wfs.url}",
                f"Version: {self.wfs.version}",
                f"Layer: {self.layer}",
                f"Output Format: {self.wfs.outformat}",
                f"Output CRS: {self.crs}",
            )
        )


class NHDPlusHR(AGRBase):
    """Access National Hydrography Dataset (NHD) Plus high resolution.

    Notes
    -----
    For more info visit: https://hydro.nationalmap.gov/arcgis/rest/services/NHDPlus_HR/MapServer

    Parameters
    ----------
    layer : str, optional
        A valid service layer. Valid layers are:

        - ``gage`` for NHDPlusGage layer
        - ``sink`` for NHDPlusSink layer
        - ``point`` for NHDPoint layer
        - ``flowline`` for NetworkNHDFlowline layer
        - ``non_network_flowline`` for NonNetworkNHDFlowline layer
        - ``flow_direction`` for FlowDirection layer
        - ``wall`` for NHDPlusWall layer
        - ``line`` for NHDLine layer
        - ``area`` for NHDArea layer
        - ``waterbody`` for NHDWaterbody layer
        - ``catchment`` for NHDPlusCatchment layer
        - ``boundary_unit`` for NHDPlusBoundaryUnit layer
        - ``huc12`` for WBDHU12 layer

    outfields : str or list, optional
        Target field name(s), default to "*" i.e., all the fields.
    crs : str, int, or pyproj.CRS, optional
        Target spatial reference, default to ``EPSG:4326``.

    Methods
    -------
    bygeom(geom, geo_crs=4326, sql_clause="", distance=None, return_m=False, return_geom=True)
        Get features within a geometry that can be combined with a SQL where clause.
    byids(field, fids, return_m=False, return_geom=True)
        Get features by object IDs.
    bysql(sql_clause, return_m=False, return_geom=True)
        Get features using a valid SQL 92 WHERE clause.
    """

    def __init__(
        self,
        layer: NHDHRLayers,
        outfields: str | list[str] = "*",
        crs: CRSType = 4326,
    ):
        self.valid_layers = {
            "gauge": "NHDPlusGage",
            "sink": "NHDPlusSink",
            "point": "NHDPoint",
            "flowline": "NetworkNHDFlowline",
            "non_network_flowline": "NonNetworkNHDFlowline",
            "flow_direction": "FlowDirection",
            "wall": "NHDPlusWall",
            "line": "NHDLine",
            "area": "NHDArea",
            "waterbody": "NHDWaterbody",
            "catchment": "NHDPlusCatchment",
            "boundary_unit": "NHDPlusBoundaryUnit",
            "huc12": "WBDHU12",
        }
        _layer = self.valid_layers.get(layer)
        if _layer is None:
            raise InputValueError("layer", list(self.valid_layers))
        super().__init__(
            ServiceURL().restful.nhdplushr,
            _layer,
            outfields,
            crs,
        )
