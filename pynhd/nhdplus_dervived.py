"""Access NLDI and WaterData databases."""
import io
from pathlib import Path
from typing import Optional, Union

import async_retriever as ar
import pandas as pd
from pygeoogc import InvalidInputValue

from .core import ScienceBase


def nhdplus_vaa(parquet_name: Optional[Union[Path, str]] = None) -> pd.DataFrame:
    """Get NHDPlus Value Added Attributes with ComID-level roughness and slope values.

    Notes
    -----
    This downloads a 200 MB ``parquet`` file from
    `here <https://www.hydroshare.org/resource/6092c8a62fac45be97a09bfd0b0bf726>`__ .
    Although this dataframe does not include geometry, it can be linked to other geospatial
    NHDPlus dataframes through ComIDs.

    Parameters
    ----------
    parquet_name : str or Path
        Path to a file with ``.parquet`` extension for saving the processed to disk for
        later use.

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
    if parquet_name is None:
        output = Path("cache", "nhdplus_vaa.parquet")
    else:
        if ".parquet" not in str(parquet_name):
            raise InvalidInputValue("parquet_name", ["a filename with `.parquet` extension."])

        output = Path(parquet_name)
    output.parent.mkdir(parents=True, exist_ok=True)

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

    resp = ar.retrieve([url], "binary")

    vaa = pd.read_parquet(io.BytesIO(resp[0]))
    vaa = vaa.astype(dtypes, errors="ignore")
    vaa.to_parquet(output)
    return vaa


def nhdplus_attrs(name: Optional[str] = None, save_dir: Optional[str] = None) -> pd.DataFrame:
    """Access NHDPlus V2.1 Attributes from ScienceBase over CONUS.

    More info can be found `here <https://www.sciencebase.gov/catalog/item/5669a79ee4b08895842a1d47>`_.

    Parameters
    ----------
    name : str, optional
        Name of the NHDPlus attribute, defaults to None which returns a dataframe containing
        metadata of all the available attributes in the database.
    save_dir : str, optional
        Directory to save the staged data frame containing metadata for the database,
        defaults to system's temp directory. The metadata dataframe is saved as a feather
        file, nhdplus_attrs.feather, in save_dir that can be loaded with Pandas.

    Returns
    -------
    pandas.DataFrame
        Either a dataframe containing the database metadata or the requested attribute over CONUS.
    """
    sb = ScienceBase(save_dir)
    if sb.char_feather.exists():
        char_df = pd.read_feather(sb.char_feather)
    else:
        char_df = sb.stage_data()

    if name is None:
        return char_df

    try:
        url = char_df[char_df.name == name].url.values[0]
    except IndexError as ex:
        raise InvalidInputValue("name", char_df.name.unique()) from ex

    return pd.read_csv(url, compression="zip")


def nhd_fcode() -> pd.DataFrame:
    """Get all the NHDPlus FCodes."""
    url = "/".join(
        [
            "https://gist.githubusercontent.com/cheginit",
            "8f9730f75b2b9918f492a236c77c180c/raw/nhd_fcode.json",
        ]
    )
    return pd.read_json(url)
