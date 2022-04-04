"""Access NLDI and WaterData databases."""
import io
from pathlib import Path
from typing import Optional, Union

import async_retriever as ar
import pandas as pd
from pygeoogc import InvalidInputValue

from .core import ScienceBase, get_parquet, stage_nhdplus_attrs

__all__ = ["enhd_attrs", "nhdplus_vaa", "nhdplus_attrs", "nhd_fcode"]


def enhd_attrs(
    parquet_path: Optional[Union[Path, str]] = None,
) -> pd.DataFrame:
    """Get updated NHDPlus attributes from ENHD.

    Notes
    -----
    This downloads a 140 MB ``parquet`` file from
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
    resp = ar.retrieve_binary(
        [files.loc["enhd_nhdplusatts.parquet"].url],
    )
    attrs = pd.read_parquet(io.BytesIO(resp[0]))
    attrs.to_parquet(output)
    return attrs


def nhdplus_vaa(
    parquet_path: Optional[Union[Path, str]] = None,
) -> pd.DataFrame:
    """Get NHDPlus Value Added Attributes with ComID-level roughness and slope values.

    Notes
    -----
    This function downloads a 200 MB ``parquet`` file from
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
        "frommeas": "Int32",
        "tomeas": "Int32",
        "reachcode": "Int64",
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

    resp = ar.retrieve_binary([url])

    vaa = pd.read_parquet(io.BytesIO(resp[0]))
    vaa = vaa.astype(dtypes, errors="ignore")
    vaa.to_parquet(output)
    return vaa


def nhdplus_attrs(
    name: Optional[str] = None,
    parquet_path: Optional[Union[Path, str]] = None,
) -> pd.DataFrame:
    """Access NHDPlus V2.1 Attributes from ScienceBase over CONUS.

    More info can be found `here <https://www.sciencebase.gov/catalog/item/5669a79ee4b08895842a1d47>`_.

    Parameters
    ----------
    name : str, optional
        Name of the NHDPlus attribute, defaults to None which returns a dataframe containing
        metadata of all the available attributes in the database.
    parquet_path : str or Path, optional
        Path to a file with ``.parquet`` extension for saving the processed to disk for
        later use. Defaults to ``./cache/nhdplus_attrs.parquet``.

    Returns
    -------
    pandas.DataFrame
        Either a dataframe containing the database metadata or the requested attribute over CONUS.
    """
    if parquet_path is not None and Path(parquet_path).exists():
        char_df = pd.read_parquet(parquet_path)
    else:
        char_df = stage_nhdplus_attrs(parquet_path)

    if name is None:
        return char_df

    try:
        url = char_df[char_df.name == name].url.values[0]
    except IndexError as ex:
        raise InvalidInputValue("name", char_df.name.unique()) from ex
    resp = ar.retrieve_binary([url])
    return pd.read_csv(io.BytesIO(resp[0]), compression="zip")


def nhd_fcode() -> pd.DataFrame:
    """Get all the NHDPlus FCodes."""
    url = "/".join(
        [
            "https://gist.githubusercontent.com/cheginit",
            "8f9730f75b2b9918f492a236c77c180c/raw/nhd_fcode.json",
        ]
    )
    return pd.read_json(url)
