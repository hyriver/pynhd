"""Access NLDI and WaterData databases."""
from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import TYPE_CHECKING, Any

import async_retriever as ar
import cytoolz as tlz
import pandas as pd
import pyarrow.dataset as pds
from pyarrow import fs

from .core import ScienceBase, get_parquet
from .exceptions import InputValueError

if TYPE_CHECKING:
    from pyarrow.dataset import FileSystemDataset

__all__ = ["enhd_attrs", "nhdplus_vaa", "nhdplus_attrs", "nhd_fcode"]


def enhd_attrs(
    parquet_path: Path | str | None = None,
) -> pd.DataFrame:
    """Get updated NHDPlus attributes from ENHD.

    Notes
    -----
    This downloads a 160 MB ``parquet`` file from
    `here <https://www.sciencebase.gov/catalog/item/60c92503d34e86b9389df1c9>`__ .
    Although this dataframe does not include geometry, it can be linked to other geospatial
    NHDPlus dataframes through ComIDs.

    Parameters
    ----------
    parquet_path : str or Path, optional
        Path to a file with ``.parquet`` extension for storing the file, defaults to
        ``./cache/enhd_attrs.parquet``.

    Returns
    -------
    pandas.DataFrame
        A dataframe that includes ComID-level attributes for 2.7 million NHDPlus flowlines.
    """
    if parquet_path is None:
        output = get_parquet(Path("cache", "enhd_attrs.parquet"))
    else:
        output = get_parquet(parquet_path)

    if output.exists():
        return pd.read_parquet(output)

    sb = ScienceBase()
    files = sb.get_file_urls("60c92503d34e86b9389df1c9")

    ar.stream_write([files.loc["enhd_nhdplusatts.parquet"].url], [output])
    enhd = pd.read_parquet(output)
    enhd["comid"] = enhd["comid"].astype("int32")
    enhd["gnis_id"] = enhd["gnis_id"].astype("Int32")
    enhd["dnlevelpat"] = enhd["dnlevelpat"].astype("int32")
    output.unlink()
    enhd.to_parquet(output)
    return enhd


def nhdplus_vaa(
    parquet_path: Path | str | None = None,
) -> pd.DataFrame:
    """Get NHDPlus Value Added Attributes with ComID-level roughness and slope values.

    Notes
    -----
    This function downloads a 245 MB ``parquet`` file from
    `here <https://www.hydroshare.org/resource/6092c8a62fac45be97a09bfd0b0bf726>`__ .
    Although this dataframe does not include geometry, it can be linked to other geospatial
    NHDPlus dataframes through ComIDs.

    Parameters
    ----------
    parquet_path : str or Path, optional
        Path to a file with ``.parquet`` extension for storing the file, defaults to
        ``./cache/nldplus_vaa.parquet``.

    Returns
    -------
    pandas.DataFrame
        A dataframe that includes ComID-level attributes for 2.7 million NHDPlus flowlines.

    Examples
    --------
    >>> vaa = nhdplus_vaa() # doctest: +SKIP
    >>> print(vaa.slope.max()) # doctest: +SKIP
    4.6
    """
    if parquet_path is None:
        output = get_parquet(Path("cache", "nldplus_vaa.parquet"))
    else:
        output = get_parquet(parquet_path)

    if output.exists():
        return pd.read_parquet(output)

    dtypes = {
        "comid": "Int64",
        "streamleve": "Int8",
        "streamorde": "Int8",
        "streamcalc": "Int8",
        "fromnode": "Int64",
        "tonode": "Int64",
        "hydroseq": "Int64",
        "levelpathi": "Int64",
        "pathlength": "Int64",
        "terminalpa": "Int64",
        "arbolatesu": "Int64",
        "divergence": "Int8",
        "startflag": "Int8",
        "terminalfl": "Int8",
        "dnlevel": "Int16",
        "thinnercod": "Int8",
        "uplevelpat": "Int64",
        "uphydroseq": "Int64",
        "dnlevelpat": "Int64",
        "dnminorhyd": "Int64",
        "dndraincou": "Int64",
        "dnhydroseq": "Int64",
        "frommeas": "f8",
        "tomeas": "f8",
        "reachcode": str,
        "lengthkm": "f8",
        "fcode": "Int32",
        "vpuin": "Int8",
        "vpuout": "Int8",
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
        "gnis_id": str,
        "wbareacomi": "Int64",
        "hwnodesqkm": "f8",
        "rpuid": str,
        "vpuid": str,
        "roughness": "f8",
    }

    rid = "6092c8a62fac45be97a09bfd0b0bf726"
    fpath = "data/contents/nhdplusVAA.parquet"
    url = f"https://www.hydroshare.org/resource/{rid}/{fpath}"

    ar.stream_write([url], [output])
    vaa = pd.read_parquet(output)
    vaa = vaa.astype(dtypes, errors="ignore")
    output.unlink()
    vaa.to_parquet(output)
    return vaa


def nhdplus_attrs(attr_names: str | list[str] | None = None, nodata: bool = False) -> pd.DataFrame:
    """Access NHDPlus V2.1 derived attributes over CONUS.

    Notes
    -----
    More info can be found `here <https://www.sciencebase.gov/catalog/item/5669a79ee4b08895842a1d47>`_.

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
        cols = ["COMID"] + cols
        if nodata:
            ds_columns = ds.schema.names
            with contextlib.suppress(StopIteration):
                cols.append(next(c for c in ds_columns if "NODATA" in c))
        return ds.to_table(columns=cols).to_pandas().set_index("COMID")

    return pd.concat((to_pandas(d, c, nodata) for d, c in zip(datasets, ids.values())), axis=1)


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
    feature: str,
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

        - ``catchment_metrics``: 414 catchment-scale metrics.
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
    >>> data = nhd.epa_nhd_catchments(1440291, "catchment_metrics")
    >>> data["catchment_metrics"].loc[1440291, "AvgWetIndxCat"]
    579.532
    """
    feature_names = {
        "catchment_metrics": "streamcat",
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
    resp = ar.retrieve_json(urls, kwds)
    info = pd.DataFrame.from_dict(
        {i: pd.Series(r["metadata"]) for i, r in zip(clist, resp)}, orient="index"
    )
    for c in info:
        info[c] = pd.to_numeric(info[c], errors="ignore")

    if feature == "catchment_metrics":
        meta = pd.DataFrame(resp[0]["streamcat"]["metrics"]).drop(columns=["id", "metric_value"])
        meta = meta.set_index("metric_alias")

        def get_metrics(resp: dict[str, Any]) -> pd.Series:
            df = pd.DataFrame(resp["streamcat"]["metrics"])[["metric_alias", "metric_value"]]
            return df.set_index("metric_alias").metric_value

        data = pd.DataFrame.from_dict(
            {i: get_metrics(r) for i, r in zip(clist, resp)}, orient="index"
        )
        return {"comid_info": info, "catchment_metrics": data, "metadata": meta}

    if feature == "curve_number":
        data = pd.DataFrame.from_dict(
            {i: r["curve_number"] for i, r in zip(clist, resp)}, orient="index", dtype="f8"
        )
        return {"comid_info": info, "curve_number": data}

    return {"comid_info": info}
