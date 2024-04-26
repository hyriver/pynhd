"""Access NLDI and WaterData databases."""

# pyright: reportGeneralTypeIssues=false
from __future__ import annotations

import contextlib
import io
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import cytoolz.curried as tlz
import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.dataset as pds
from pyarrow import ArrowInvalid, fs

import async_retriever as ar
import pygeoogc as ogc
from pynhd.core import ScienceBase, get_parquet
from pynhd.exceptions import InputTypeError, InputValueError, ServiceError

if TYPE_CHECKING:
    from pyarrow.dataset import FileSystemDataset

__all__ = [
    "enhd_attrs",
    "nhdplus_vaa",
    "nhdplus_h12pp",
    "nhdplus_attrs",
    "nhdplus_attrs_s3",
    "nhd_fcode",
    "epa_nhd_catchments",
    "StreamCat",
    "streamcat",
]

NHDP_DTYPES = {
    "comid": "int32",
    "tocomid": "int32",
    "streamleve": "int8",
    "streamorde": "int8",
    "streamcalc": "int8",
    "fromnode": "int32",
    "tonode": "int32",
    "hydroseq": "int32",
    "levelpathi": "int32",
    "pathlength": "int32",
    "terminalpa": "int32",
    "arbolatesu": "int32",
    "divergence": "int8",
    "startflag": "int8",
    "terminalfl": "int8",
    "dnlevel": "int32",
    "thinnercod": "int8",
    "uplevelpat": "int32",
    "uphydroseq": "int32",
    "dnlevelpat": "int32",
    "dnminorhyd": "int32",
    "dndraincou": "int32",
    "dnhydroseq": "int32",
    "frommeas": "f8",
    "tomeas": "f8",
    "reachcode": str,
    "lengthkm": "f8",
    "fcode": "int32",
    "vpuin": "int8",
    "vpuout": "int8",
    "areasqkm": "f8",
    "totdasqkm": "f8",
    "divdasqkm": "f8",
    "totma": "f8",
    "wbareatype": str,
    "pathtimema": "f8",
    "slope": "f8",
    "slopelenkm": "f8",
    "ftype": str,
    "gnis_name": str,
    "gnis_id": "int32",
    "wbareacomi": "int32",
    "hwnodesqkm": "f8",
    "rpuid": str,
    "vpuid": str,
    "roughness": "f8",
}


def enhd_attrs(
    parquet_path: Path | str | None = None,
) -> pd.DataFrame:
    """Get updated NHDPlus attributes from ENHD V2.0.

    Notes
    -----
    This function downloads a 160 MB ``parquet`` file from
    `here <https://doi.org/10.5066/P976XCVT>`__.
    Although this dataframe does not include geometry, it can be
    linked to other geospatial NHDPlus dataframes through ComIDs.

    Parameters
    ----------
    parquet_path : str or pathlib.Pathlib.Path, optional
        Path to a file with ``.parquet`` extension for storing the file,
        defaults to ``./cache/enhd_attrs.parquet``.

    Returns
    -------
    pandas.DataFrame
        A dataframe that includes ComID-level attributes for
        2.7 million NHDPlus flowlines.
    """
    if parquet_path is None:
        output = get_parquet(Path("cache", "enhd_attrs.parquet"))
    else:
        output = get_parquet(parquet_path)

    url = ScienceBase.get_file_urls("63cb311ed34e06fef14f40a3").loc["enhd_nhdplusatts.parquet"].url
    fname = ogc.streaming_download(url, file_extention=".parquet")

    enhd = pd.read_parquet(fname)
    dtype = {k: v for k, v in NHDP_DTYPES.items() if k in enhd.columns}
    enhd = enhd.astype(dtype, errors="ignore")
    if output.exists():
        output.unlink()
    enhd.to_parquet(output)
    return enhd


def nhdplus_vaa(
    parquet_path: Path | str | None = None,
) -> pd.DataFrame:
    """Get NHDPlus Value Added Attributes including roughness.

    Notes
    -----
    This function downloads a 245 MB ``parquet`` file from
    `here <https://www.hydroshare.org/resource/6092c8a62fac45be97a09bfd0b0bf726>`__.
    Although this dataframe does not include geometry, it can be linked
    to other geospatial NHDPlus dataframes through ComIDs.

    Parameters
    ----------
    parquet_path : str or pathlib.Pathlib.Path, optional
        Path to a file with ``.parquet`` extension for storing the file, defaults to
        ``./cache/nldplus_vaa.parquet``.

    Returns
    -------
    pandas.DataFrame
        A dataframe that includes ComID-level attributes for 2.7 million
        NHDPlus flowlines.
    """
    if parquet_path is None:
        output = get_parquet(Path("cache", "nldplus_vaa.parquet"))
    else:
        output = get_parquet(parquet_path)

    rid = "6092c8a62fac45be97a09bfd0b0bf726"
    fpath = "data/contents/nhdplusVAA.parquet"
    url = f"https://www.hydroshare.org/resource/{rid}/{fpath}"
    fname = ogc.streaming_download(url, file_extention=".parquet")

    vaa = pd.read_parquet(fname)
    dtype = {k: v for k, v in NHDP_DTYPES.items() if k in vaa.columns}
    vaa = vaa.astype(dtype, errors="ignore")
    if output.exists():
        output.unlink()
    vaa.to_parquet(output)
    return vaa


def nhdplus_attrs(attr_name: str | None = None) -> pd.DataFrame:
    """Stage the NHDPlus Attributes database and save to nhdplus_attrs.parquet.

    Notes
    -----
    More info can be found `here <https://www.sciencebase.gov/catalog/item/5669a79ee4b08895842a1d47>`__.

    Parameters
    ----------
    attr_names : str , optional
        Name of NHDPlus attribute to return, defaults to None, i.e.,
        only return a metadata dataframe that includes the attribute names
        and their description and units.

    Returns
    -------
    pandas.DataFrame
        The staged data as a DataFrame.
    """
    sb = ScienceBase()
    r = sb.get_children("5669a79ee4b08895842a1d47")

    titles = tlz.pluck("title", r["items"])
    titles = tlz.concat(tlz.map(tlz.partial(re.findall, "Select(.*?)Attributes"), titles))
    titles = tlz.map(str.strip, titles)

    main_items = dict(zip(titles, tlz.pluck("id", r["items"])))

    def get_files(item: str) -> dict[str, tuple[str, str]]:
        """Get all the available zip files in an item."""
        url = "https://www.sciencebase.gov/catalog/item"
        payload = {"fields": "files,downloadUri", "format": "json"}
        resp = ar.retrieve_json([f"{url}/{item}"], [{"params": payload}])
        resp = cast("list[dict[str, Any]]", resp)
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
    meta = pd.DataFrame(chars)
    if attr_name is None:
        return meta

    try:
        url = meta[meta.name == attr_name].url.values[0]
    except IndexError as ex:
        raise InputValueError("name", meta.name.unique().tolist()) from ex

    fname = ogc.streaming_download(url, file_extention="zip")
    return pd.read_csv(fname, engine="pyarrow")


def nhdplus_attrs_s3(
    attr_names: str | list[str] | None = None, nodata: bool = False
) -> pd.DataFrame:
    """Access NHDPlus V2.1 derived attributes over CONUS.

    Notes
    -----
    More info can be found `here <https://www.sciencebase.gov/catalog/item/5669a79ee4b08895842a1d47>`__.

    Parameters
    ----------
    attr_names : str or list of str, optional
        Names of NHDPlus attribute(s) to return, defaults to None, i.e.,
        only return a metadata dataframe that includes the attribute names
        and their description and units.
    nodata : bool
        Whether to include NODATA percentages, default is False.

    Returns
    -------
    pandas.DataFrame
        A dataframe of requested NHDPlus attributes.
    """
    bucket = "prod-is-usgs-sb-prod-publish"
    meta_url = "/".join(
        (f"https://{bucket}.s3.amazonaws.com", "5669a79ee4b08895842a1d47/metadata_table.tsv")
    )
    resp = ar.retrieve_text([meta_url])
    meta = pd.read_csv(io.StringIO(resp[0]), delimiter="\t")
    if attr_names is None:
        return meta

    urls = meta.set_index("ID").datasetURL.str.rsplit("/", n=1).str[-1].to_dict()

    attr_names = attr_names if isinstance(attr_names, (list, tuple)) else [attr_names]
    if any(a not in urls for a in attr_names):
        raise InputValueError("attr_names", list(urls))
    ids = tlz.merge_with(list, ({urls[c]: c} for c in attr_names))

    s3 = fs.S3FileSystem(anonymous=True, region="us-west-2")
    get_dataset = tlz.partial(pds.dataset, filesystem=s3, format="parquet")
    datasets = (get_dataset(f"{bucket}/{i}/{i}_{c[0][:3].lower()}.parquet") for i, c in ids.items())

    def to_pandas(ds: FileSystemDataset, cols: list[str], nodata: bool) -> pd.DataFrame:
        """Convert a pyarrow dataset to a pandas dataframe."""
        cols = ["COMID", *cols]
        if nodata:
            ds_columns = ds.schema.names
            with contextlib.suppress(StopIteration):
                cols.append(next(c for c in ds_columns if "NODATA" in c))
        return ds.to_table(columns=cols).to_pandas().set_index("COMID")

    return pd.concat((to_pandas(d, c, nodata) for d, c in zip(datasets, ids.values())), axis=1)


def nhdplus_h12pp(gpkg_path: Path | str | None = None) -> pd.DataFrame:
    """Access HUC12 Pour Points for NHDPlus V2.1 L48 (CONUS).

    Notes
    -----
    More info can be found
    `here <https://www.sciencebase.gov/catalog/item/60cb5edfd34e86b938a373f4>`__.

    Parameters
    ----------
    gpkg_path : str or pathlib.Pathlib.Path, optional
        Path to the geopackage file, defaults to None, i.e., download
        the file to the cache directory as ``102020wbd_outlets.gpkg``.

    Returns
    -------
    geopandas.GeoDataFrame
        A geodataframe of HUC12 pour points.
    """
    files = ScienceBase.get_file_urls("60cb5edfd34e86b938a373f4").loc["102020wbd_outlets.gpkg"].url
    gpkg_path = Path("cache", "102020wbd_outlets.gpkg") if gpkg_path is None else Path(gpkg_path)
    _ = ogc.streaming_download(files, fnames=gpkg_path)
    h12pp = gpd.read_file(gpkg_path, engine="pyogrio")
    h12pp = h12pp[
        ["COMID", "REACHCODE", "REACH_meas", "offset", "HUC12", "LevelPathI", "geometry"]
    ].copy()
    h12pp[["COMID", "LevelPathI"]] = h12pp[["COMID", "LevelPathI"]].astype("uint32")
    h12pp = cast("gpd.GeoDataFrame", h12pp)
    return h12pp


def nhd_fcode() -> pd.DataFrame:
    """Get all the NHDPlus FCodes."""
    url = "/".join(
        [
            "https://gist.githubusercontent.com/cheginit",
            "8f9730f75b2b9918f492a236c77c180c/raw/nhd_fcode.json",
        ]
    )
    return pd.read_json(url)


def epa_nhd_catchments(
    comids: int | str | list[int | str],
    feature: Literal["curve_number", "comid_info"],
) -> dict[str, pd.DataFrame]:
    """Get NHDPlus catchment-scale data from EPA's HMS REST API.

    Notes
    -----
    For more information about curve number please refer to the project's
    webpage on the EPA's
    `website <https://cfpub.epa.gov/si/si_public_record_Report.cfm?Lab=CEMM&dirEntryId=351307>`__.

    Parameters
    ----------
    comids : int or list of int
        ComID(s) of NHDPlus catchments.
    feature : str
        The feature of interest. Available options are:

        - ``curve_number``: 16-day average Curve Number.
        - ``comid_info``: ComID information.

    Returns
    -------
    dict of pandas.DataFrame or geopandas.GeoDataFrame
        A dict of the requested dataframes. A ``comid_info`` dataframe is
        always returned.

    Examples
    --------
    >>> import pynhd
    >>> data = pynhd.epa_nhd_catchments(9533477, "curve_number")
    >>> data["curve_number"].mean(axis=1).item()
    75.576
    """
    feature_names = {
        "curve_number": "cn",
        "comid_info": "",
    }
    if feature not in feature_names:
        raise InputValueError("feature", list(feature_names))

    clist = comids if isinstance(comids, (list, tuple)) else [comids]
    f_kwd = {feature_names[feature]: "true"} if feature != "comid_info" else {}
    urls, kwds = zip(
        *[
            (
                "https://qed.epa.gov/hms/rest/api/info/catchment",
                {"params": {**f_kwd, "comid": comid}},
            )
            for comid in clist
        ]
    )
    urls = cast("list[str]", urls)
    kwds = cast("list[dict[str, Any]]", kwds)
    resp = ar.retrieve_json(urls, kwds)
    resp = cast("list[dict[str, Any]]", resp)
    info = pd.DataFrame.from_dict(
        {i: pd.Series(r["metadata"]) for i, r in zip(clist, resp)}, orient="index"
    )
    for c in info:
        info[c] = pd.to_numeric(info[c], errors="coerce")

    if feature == "curve_number":
        data = pd.DataFrame.from_dict(
            {i: r["curve_number"] for i, r in zip(clist, resp)}, orient="index", dtype="f8"
        )
        return {"comid_info": info, "curve_number": data}

    return {"comid_info": info}


class StreamCat:
    """Get StreamCat API's properties.

    Parameters
    ----------
    lakes_only : bool, optional
        If ``True``, only return metrics for lakes and their associated catchments
        from the LakeCat dataset.

    Attributes
    ----------
    base_url : str
        The base URL of the API.
    valid_names : list of str
        The valid names of the metrics.
    alt_names : dict of str
        The alternative names of some metrics.
    valid_regions : list of str
        The valid hydro regions.
    valid_states : pandas.DataFrame
        The valid two letter states' abbreviations.
    valid_counties : pandas.DataFrame
        The valid counties' FIPS codes.
    valid_aois : list of str
        The valid types of areas of interest.
    metrics_df : pandas.DataFrame
        The metrics' metadata such as description and units.
    valid_years : dict
        A dictionary of the valid years for annual metrics.
    """

    def __init__(self, lakes_only: bool = False) -> None:
        self.lakes_only = lakes_only
        if self.lakes_only:
            self.base_url = "https://java.epa.gov/StreamCAT/LakeCat/metrics"
        else:
            self.base_url = "https://java.epa.gov/StreamCAT/metrics"
        resp = ar.retrieve_json([self.base_url])
        resp = cast("list[dict[str, Any]]", resp)
        params = resp[0]["parameters"]

        self.valid_names = cast("list[str]", params["name"]["options"])
        self.alt_names = {
            n.replace("_", ""): n for n in self.valid_names if n.split("_")[-1].isdigit()
        }
        self.alt_names.update(
            {
                "precip2008": "precip08",
                "precip2009": "precip09",
                "tmean2008": "tmean08",
                "tmean2009": "tmean09",
            }
        )
        self.valid_regions = params["region"]["options"]
        self.valid_states = pd.DataFrame.from_dict(params["state"]["options"], orient="index")
        self.valid_counties = pd.DataFrame.from_dict(params["county"]["options"], orient="index")
        self.valid_aois = params["areaOfInterest"]["options"]
        self.valid_slopes = tlz.merge_with(
            list,
            (
                {s[:-9]: f"slp{slp}" for s in self.valid_names if f"slp{slp}" in s}
                for slp in (10, 20)
            ),
        )

        url_vars = f"{self.base_url}/variable_info.csv"
        names = pd.read_csv(io.BytesIO(ar.retrieve_binary([url_vars])[0]), encoding="latin1")
        names["METRIC_NAME"] = names["METRIC_NAME"].str.replace(r"\[AOI\]|Slp[12]0", "", regex=True)
        names["SLOPE"] = [
            ", ".join(self.valid_slopes.get(m.replace("[Year]", "").lower(), []))
            for m in names.METRIC_NAME
        ]
        names.loc[names["SLOPE"] == "", "SLOPE"] = np.nan
        slp = ~names["SLOPE"].isna()
        names.loc[slp, "METRIC_NAME"] = [f"{n}[Slope]" for n in names.loc[slp, "METRIC_NAME"]]
        names = (
            names.drop_duplicates("METRIC_NAME").reset_index(drop=True).drop(columns="WEBTOOL_NAME")
        )
        self.metrics_df = names

        years = names.set_index("METRIC_NAME").YEAR.dropna()
        self.valid_years = {
            str(v): list(range(*(int(y) for y in yrs.split("-"))))
            if "-" in yrs
            else [int(y) for y in yrs.split(",")]
            for v, yrs in years.items()
        }


class StreamCatValidator(StreamCat):
    def __init__(self, lakes_only: bool = False) -> None:
        super().__init__(lakes_only)

    def validate(
        self,
        name: list[str] | None = None,
        region: list[str] | None = None,
        state: list[str] | None = None,
        county: list[str] | None = None,
        aoi: list[str] | None = None,
    ) -> None:
        """Validate input parameters."""
        if name and not set(name).issubset(self.valid_names):
            raise InputValueError("metric_names", self.valid_names)

        if region and not set(region).issubset(self.valid_regions):
            raise InputValueError("regions", self.valid_regions)

        if state and not set(state).issubset(self.valid_states.index):
            raise InputValueError("states", self.valid_states.index.to_list())

        if county and not set(county).issubset(self.valid_counties.index):
            raise InputValueError("counties", self.valid_counties.index.to_list())

        if aoi and not set(aoi).issubset(self.valid_aois):
            raise InputValueError("metric_areas", self.valid_aois)

    def id_kwds(
        self,
        comids: int | list[int] | None = None,
        regions: str | list[str] | None = None,
        states: str | list[str] | None = None,
        counties: str | list[str] | None = None,
        conus: bool = False,
    ) -> dict[str, str]:
        """Get the keyword arguments for the API's ID parameters."""
        n_args = sum(i is not None for i in (comids, regions, states, counties))
        n_args += int(conus)
        if n_args != 1:
            msg = "Exactly one of comids, regions, states, counties, or conus must be passed"
            raise ValueError(msg)

        params = {}
        if comids is not None:
            try:
                comid_arr = np.array(
                    [comids] if isinstance(comids, (str, int)) else comids, dtype=int
                )
                params["comid"] = ",".join(comid_arr.astype(str))
            except (ValueError, TypeError) as ex:
                raise InputTypeError("comids", "int or digists (str) or list of them") from ex
        elif regions is not None:
            ids = [regions] if isinstance(regions, str) else regions
            self.validate(region=ids)
            params["region"] = ",".join(ids)
        elif states is not None:
            ids = [states.upper()] if isinstance(states, str) else [s.upper() for s in states]
            self.validate(state=ids)
            params["state"] = ",".join(ids)
        elif conus:
            params["conus"] = "true"
        return params


def streamcat(
    metric_names: str | list[str],
    metric_areas: str | list[str] | None = None,
    comids: int | list[int] | None = None,
    regions: str | list[str] | None = None,
    states: str | list[str] | None = None,
    counties: str | list[str] | None = None,
    conus: bool = False,
    percent_full: bool = False,
    area_sqkm: bool = False,
    lakes_only: bool = False,
) -> pd.DataFrame:
    """Get various metrics for NHDPlusV2 catchments from EPA's StreamCat.

    Notes
    -----
    For more information about the service check its webpage
    at https://www.epa.gov/national-aquatic-resource-surveys/streamcat-dataset.


    Parameters
    ----------
    metric_names : str or list of str
        Metric name(s) to retrieve. There are 567 metrics available.
        to get a full list check out :meth:`StreamCat.valid_names`.
        To get a description of each metric, check out
        :meth:`StreamCat.metrics_df`. Some metrics require year and/or slope
        to be specified, which have ``[Year]`` and/or ``[Slope]`` in their name.
        For convenience all these variables and their years/slopes are converted
        to a dict that can be accessed via :meth:`StreamCat.valid_years` and
        :meth:`StreamCat.valid_slopes`.
    metric_areas : str or list of str, optional
        Areas to return the metrics for, defaults to ``None``, i.e. all areas.
        Valid options are: ``catchment``, ``watershed``, ``riparian_catchment``,
        ``riparian_watershed``, ``other``.
    comids : int or list of int, optional
        NHDPlus COMID(s), defaults to ``None``. Either ``comids``, ``regions``,
        ``states``, ``counties``, or ``conus`` must be passed. They are
        mutually exclusive.
    regions : str or list of str, optional
        Hydro region(s) to retrieve metrics for, defaults to ``None``. For a
        full list of valid regions check out :meth:`StreamCat.valid_regions`
        Either ``comids``, ``regions``, ``states``, ``counties``, or ``conus``
        must be passed. They are mutually exclusive.
    states : str or list of str, optional
        Two letter state abbreviation(s) to retrieve metrics for, defaults to
        ``None``. For a full list of valid states check out
        :meth:`StreamCat.valid_states` Either ``comids``, ``regions``,
        ``states``, ``counties``, or ``conus`` must be passed. They are
        mutually exclusive.
    counties : str or list of str, optional
        County FIPS codes(s) to retrieve metrics for, defaults to ``None``. For
        a full list of valid county codes check out :meth:`StreamCat.valid_counties`
        Either ``comids``, ``regions``, ``states``, ``counties``, or ``conus`` must
        be passed. They are mutually exclusive.
    conus : bool, optional
        If ``True``, ``metric_names`` of all NHDPlus COMIDs are retrieved,
        defaults ``False``. Either ``comids``, ``regions``,
        ``states``, ``counties``, or ``conus`` must be passed. They are mutually
        exclusive.
    percent_full : bool, optional
        If ``True``, return the percent of each area of interest covered by
        the metric.
    area_sqkm : bool, optional
        If ``True``, return the area in square kilometers.
    lakes_only : bool, optional
        If ``True``, only return metrics for lakes and their associated catchments
        from the LakeCat dataset.

    Returns
    -------
    pandas.DataFrame
        A dataframe with the requested metrics.
    """
    sc = StreamCatValidator(lakes_only)
    names = [metric_names] if isinstance(metric_names, str) else metric_names
    names = [sc.alt_names.get(s.lower(), s.lower()) for s in names]
    sc.validate(name=names)
    params = {"name": ",".join(n for n in names)}

    if metric_areas:
        aoi = [metric_areas] if isinstance(metric_areas, str) else metric_areas
        sc.validate(aoi=aoi)
        params["areaOfInterest"] = ",".join(aoi)

    ids = sc.id_kwds(comids, regions, states, counties, conus)
    if ids:
        params.update(ids)

    if percent_full:
        params["showPctFull"] = "true"

    if area_sqkm:
        params["showAreaSqKm"] = "true"

    params_str = "&".join(f"{k}={v}" for k, v in params.items())
    if len(params_str) < 7000:
        resp = ar.retrieve_text([f"{sc.base_url}?{params_str}"])
    else:
        resp = ar.retrieve_text([sc.base_url], [{"data": params_str}], request_method="post")

    try:
        return pd.read_csv(io.StringIO(resp[0]), engine="pyarrow")
    except ArrowInvalid as ex:
        raise ServiceError(resp[0]) from ex
