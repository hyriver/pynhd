=======
History
=======

0.16.3 (2024-04-26)
-------------------

New Features
~~~~~~~~~~~~
- Add support for LakeCat dataset in ``streamcat`` function. A new argument
  called ``lakes_only`` is added to the function. If set to ``True``, only
  metrics for lake and their associated catchments will be returned. The default
  is ``False`` to retain backward compatibility.

Bug Fixes
~~~~~~~~~
- Modify ``HP3D`` class based on the latest changes to the 3D Hydrography Program
  service. Hydrolocation layer has now three sub-layers:

  - ``hydrolocation_waterbody`` for Sink, Spring, Waterbody Outlet,
  - ``hydrolocation_flowline`` for Headwater, Terminus, Divergence, Confluence, Catchment Outlet,
  - ``hydrolocation_reach`` for Reach Code, External Connection.

Breaking Changes
~~~~~~~~~~~~~~~~
- EPA's HMS no longer supports the StreamCat dataset, since they have a dedicated
  service for it. Thus, the ``epa_nhd_catchments`` function no longer accepts
  "streamcat" as an input for the ``feature`` argument. For all StreamCat queries,
  use the ``streamcat`` function instead. Now, the ``epa_nhd_catchments`` function
  is essentially useful for getting Curve Number data.

0.16.2 (2024-02-12)
-------------------

Bug Fixes
~~~~~~~~~
- In ``NLDI.get_basins``, the indices used to be station IDs but in the
  previous release they were reset by mistake. This version retains the
  correct indices.

New Features
~~~~~~~~~~~~
- In ``nhdplus_l48`` function, when the layer is ``NHDFlowline_Network``
  or ``NHDFlowline_NonNetwork``, merge all ``MultiLineString`` geometries to ``LineString``.

0.16.1 (2024-01-03)
-------------------

Bug Fixes
~~~~~~~~~
- Fix an issue in ``network_xsection`` and ``flowline_xsection`` related
  to the changes in ``shapely`` 2 API. Now, these functions should return
  the correct cross-sections.

0.16.0 (2024-01-03)
-------------------

New Features
~~~~~~~~~~~~
- Add access to USGS 3D Hydrography Program (3DHP) service. The new
  class is called ``HP3D``. It can be queried by IDs, geometry, or
  SQL where clause.
- Add support for the new PyGeoAPI endpoints called ``xsatpathpts``.
  This new endpoint is useful for getting elevation profile along A
  ``shapely.LineString``. You can use ``pygeoapi`` function with
  ``service="elevation_profile"`` (or ``PyGeoAPI`` class) to access this
  new endpoint. Previously, the ``elevation_profile`` endpoint was used for
  getting elevation profile along a path from two endpoints and the input
  ``GeoDataFrame`` must have been a ``MultiPoint`` with two coordinates.
  Now, you must the input must contain ``LineString`` geometries.
- Switch to using the new smoothing algorithm from ``pygeoutils`` for
  resampling the flowlines and getting their cross-sections. This new
  algorithm is more robust, accurate, and faster. It has a new argument
  called ``smoothing`` for controlling the number knots of the spline. Higher
  values result in smoother curves. The default value is ``None`` which
  uses all the points from the input flowline.

0.15.2 (2023-09-22)
-------------------

Bug Fixes
~~~~~~~~~
- Update ``GeoConnex`` based on the latest changes in the web service.

0.15.1 (2023-09-02)
-------------------

Bug Fixes
~~~~~~~~~
- Fix HyRiver libraries requirements by specifying a range instead
  of exact version so ``conda-forge`` can resolve the dependencies.

0.15.0 (2023-05-07)
-------------------
From release 0.15 onward, all minor versions of HyRiver packages
will be pinned. This ensures that previous minor versions of HyRiver
packages cannot be installed with later minor releases. For example,
if you have ``py3dep==0.14.x`` installed, you cannot install
``pydaymet==0.15.x``. This is to ensure that the API is
consistent across all minor versions.

New Features
~~~~~~~~~~~~
- Add a new function, called ``nhdplus_h12pp``, for retrieving
  HUC12 pour points across CONUS.
- Add ``use_arrow=True`` to ``pynhd.nhdplus_l48`` when reading the NHDPlus
  dataset. This speeds up the process since ``pyarrow`` is installed.
- In ``nhdplus_l48`` make ``layer`` option so ``sql`` parameter of
  ``pyogrio.read_dataframe`` can also be used. This is necessary
  since ``pyogrio.read_dataframe`` does not support passing both
  ``layer`` and ``sql`` parameters.
- Update the mainstems dataset link to version 2.0 in ``mainstem_huc12_nx``.
- Expose ``NHDTools`` class to the public API.
- For now, retain compatibility with ``shapely<2`` while supporting
  ``shapley>=2``.

Bug Fixes
~~~~~~~~~
- Remove unnecessary conversion of ``id_col`` and ``toid_col`` to ``Int64``
  in ``nhdflw2nx`` and ``vector_accumulation``. This ensures that the input
  data types are preserved.
- Fix an issue in ``nhdplus_l48``, where if the input ``data_dir`` is not
  absolute ``py7zr`` fails to extract the file.

0.14.0 (2023-03-05)
-------------------

New Features
~~~~~~~~~~~~
- Rewrite the ``GeoConnex`` class to provide access to new capabilities
  of the web service. Support for spatial queries have been added via
  CQL queries. For more information, check out the updated GeoConnex example
  `notebook <https://github.com/hyriver/HyRiver-examples/blob/main/notebooks/geoconnex.ipynb>`__.
- Add a new property to ``StreamCat``, called ``metrics_df`` that gets
  a dataframe of metric names and their description.
- Create a new private ``StreamCatValidator`` class to avoid polluting
  the public ``StreamCat`` class with private attributes and methods.
  Moreover, add a new alternative metric names attribute to ``StreamCat``
  called ``alt_names`` for handling those metric names that do not follow
  ``METRIC+YYYY`` convention. This attribute is a dictionary that maps the
  alternative names to the actual metric names, so users can use
  ``METRIC_NAME`` column of ``metrics_df`` and add a year suffix from
  ``valid_years`` attribute of ``StreamCat`` to get the actual metric name.
- In ``navigate_by*`` functions of ``NLDI`` add ``stop_comid``,
  which is another criterion for stopping the navigation in addition
  to ``distance``.
- Improve ``UserWarning`` messages of ``NLDI`` and ``WaterData``.

Breaking Changes
~~~~~~~~~~~~~~~~
- Remove ``pynhd.geoconnex`` function since more functionality has been
  added to the GeoConnex service that existence of this function does not
  make sense anymore. All queries should be done via ``pynhd.GeoConnex``
  class.
- Rewrite ``NLDI`` to improve code readability and significantly improving
  performance. Now, its methods do now return tuples if there are failed
  requests, instead they will be shown as a ``UserWarning``.
- Bump the minimum required version of ``shapely`` to 2.0,
  and use its new API.

Internal Changes
~~~~~~~~~~~~~~~~
- Sync all minor versions of HyRiver packages to 0.14.0.

0.13.12 (2023-02-10)
--------------------

New Features
~~~~~~~~~~~~
- Update the link to version 2.0 of the ENHD dataset in ``enhd_attrs``.

Internal Changes
~~~~~~~~~~~~~~~~
- Improve columns data types in ``enhd_attrs`` and ``nhdplus_vaa`` by using
  ``int32`` instead of ``Int64``, where applicable.
- Sync all patch versions of HyRiver packages to x.x.12.

0.13.11 (2023-01-24)
--------------------

New Features
~~~~~~~~~~~~
- The ``prepare_nhdplus`` now supports NHDPlus HR in addition
  to NHDPlus MR. It automatically detects the NHDPlus version based on
  the ID column name: ``nhdplusid`` for HR and ``comid`` for MR.

Internal Changes
~~~~~~~~~~~~~~~~
- Fully migrate ``setup.cfg`` and ``setup.py`` to ``pyproject.toml``.
- Convert relative imports to absolute with ``absolufy-imports``.
- Improve performance of ``prepare_nhdplus`` by using ``pandas.merge``
  instead of applying a function to each row of the dataframe.

0.13.10 (2023-01-08)
--------------------

New Features
~~~~~~~~~~~~
- Add support for the new EPA's
  `StreamCat <https://www.epa.gov/national-aquatic-resource-surveys/streamcat-dataset>`__
  Restful API with around 600 NHDPlus
  catchment level metrics. One class is added for getting the service
  properties such as valid metrics, called ``StreamCat``. You can use
  ``streamcat`` function to get the metrics as a ``pandas.DataFrame``.
- Refactor the ``show_versions`` function to improve performance and
  print the output in a nicer table-like format.

Internal Changes
~~~~~~~~~~~~~~~~
- Skip 0.13.9 version so the minor version of all HyRiver packages become
  the same.
- Modify the codebase based on the latest changes in ``geopandas`` related
  to empty dataframes.

0.13.8 (2022-12-09)
-------------------

New Features
~~~~~~~~~~~~
- Add a new function, called ``nhdplus_attrs_s3``, for accessing the recently
  released NHDPlus derived attributes on a USGS's S3 bucket. The attributes are
  provided in parquet files, so getting them is faster than ``nhdplus_attrs``.
  Also, you can request for multiple attributes at once whereas in ``nhdplus_attrs``
  you had to request for each attribute one at a time. This function will replace
  ``nhdplus_attrs`` in a future release, as soon as all data that are available
  on the ScienceBase version are also accessible from the S3 bucket.
- Add two new functions called ``mainstem_huc12_nx`` and ``enhd_flowlines_nx``.
  These functions generate a ``networkx`` directed graph object of NHD HUC12
  water boundaries and flowlines, respectively. They also return a dictionary
  mapping of COMID and HUC12 to the corresponding ``networkx`` node.
  Additionally, a topologically sorted list of COMIDs/HUC12s are returned.
  The generated data are useful for doing US-scale network analysis and flow
  accumulation on the NHD network. The NHD graph has about 2.7 million edges
  and the mainstem HUC12 graph has about 80K edges.
- Add a new function for getting the entire NHDPlus dataset for CONUS (Lower 48),
  called ``nhdplus_l48``. The entire NHDPlus dataset is downloaded from
  `here <https://www.epa.gov/waterdata/nhdplus-national-data>`__.
  This 7.3 GB file will take a while to download, depending on your internet
  connection. The first time you run this function, the file will be downloaded
  and stored in the ``./cache`` directory. Subsequent calls will use the cached
  file. Moreover, there are two additional dependencies for using this function:
  ``pyogrio`` and ``py7zr``. These dependencies can be installed using
  ``pip install pyogrio py7zr`` or ``conda install -c conda-forge pyogrio py7zr``.

Internal Changes
~~~~~~~~~~~~~~~~
- Refactor ``vector_accumulation`` for significant performance improvements.
- Modify the codebase based on `Refurb <https://github.com/dosisod/refurb>`__
  suggestions.

0.13.7 (2022-11-04)
-------------------

New Features
~~~~~~~~~~~~
- Add a new function called ``epa_nhd_catchments`` to access one of the
  EPA's HMS endpoints called ``WSCatchment``. You can use this function to
  access 414 catchment-scale characteristics for all the NHDPlus catchments
  including 16-day average curve number. More information on the curve number
  dataset can be found at its project page
  `here <https://cfpub.epa.gov/si/si_public_record_Report.cfm?Lab=CEMM&dirEntryId=351307>`__.

Bug Fixes
~~~~~~~~~
- Fix a bug in ``NHDTools`` where due to the recent changes in ``pandas``
  exception handling, the ``NHDTools`` fails in converting columns with
  ``NaN`` values to integer type. Now, ``pandas`` throws ``IntCastingNaNError``
  instead of ``TypeError`` when using ``astype`` method on a column.

Internal Changes
~~~~~~~~~~~~~~~~
- Use ``pyupgrade`` package to update the type hinting annotations
  to Python 3.10 style.

0.13.6 (2022-08-30)
-------------------

Internal Changes
~~~~~~~~~~~~~~~~
- Add the missing PyPi classifiers for the supported Python versions.

0.13.5 (2022-08-29)
-------------------

Breaking Changes
~~~~~~~~~~~~~~~~
- Append "Error" to all exception classes for conforming to PEP-8 naming conventions.

Internal Changes
~~~~~~~~~~~~~~~~
- Bump the minimum versions of ``pygeoogc`` and ``pygeoutils`` to 0.13.5 and that of
  ``async-retriever`` to 0.3.5.

Bug Fixes
~~~~~~~~~
- Fix an issue in ``nhdplus_vaa`` and ``enhd_attrs`` functions where if ``cache`` folder
  does not exist, it would not have been created, thus resulting to an error.

0.13.3 (2022-07-31)
-------------------

Internal Changes
~~~~~~~~~~~~~~~~
- Use the new ``async_retriever.stream_write`` function to download files in
  ``nhdplus_vaa`` and ``enhd_attrs`` functions. This is more memory efficient.
- Convert the type of list of not found items in ``NLDI.comid_byloc`` and
  ``NLDI.feature_byloc`` to list of tuples of coordinates from list of strings.
  This matches the type of returned not found coordinates to that of the inputs.
- Fix an issue with NLDI that was caused by the recent changes in the NLDI web
  service's error handling. The NLDI web service now returns more descriptive
  error messages in a ``json`` format instead of returning the usual status
  errors.
- Slice the ENHD dataframe in ``NHDTools.clean_flowlines`` before updating
  the flowline dataframe to reduce the required memory for the ``update`` operation.

0.13.2 (2022-06-14)
-------------------

Breaking Changes
~~~~~~~~~~~~~~~~
- Set the minimum supported version of Python to 3.8 since many of the
  dependencies such as ``xarray``, ``pandas``, ``rioxarray`` have dropped support
  for Python 3.7.

Internal Changes
~~~~~~~~~~~~~~~~
- Use `micromamba <https://github.com/marketplace/actions/provision-with-micromamba>`__
  for running tests
  and use `nox <https://github.com/marketplace/actions/setup-nox>`__
  for linting in CI.

0.13.1 (2022-06-11)
-------------------

New Features
~~~~~~~~~~~~
- Add support for all the GeoConnex web service endpoints. There are two
  ways to use it. For a single query, you can use the ``geoconnex`` function and
  for multiple queries, it's more efficient to use the ``GeoConnex`` class.
- Add support for passing any of the supported NLDI feature sources to
  the ``get_basins`` method of the ``NLDI`` class. The default is ``nwissite``
  to retain backward compatibility.

Bug Fixes
~~~~~~~~~
- Set the type of "ReachCode" column to ``str`` instead of ``int`` in ``pygeoapi``
  and ``nhdplus_vaa`` functions.

0.13.0 (2022-04-03)
-------------------

New Features
~~~~~~~~~~~~
- Add two new functions called ``flowline_resample`` and ``network_resample`` for
  resampling a flowline or network of flowlines based on a given spacing. This is
  useful for smoothing jagged flowlines similar to those in the NHDPlus database.
- Add support for the new NLDI endpoint called "hydrolocation". The ``NLDI`` class
  now has two methods for getting features by coordinates: ``feature_byloc``
  and ``comid_byloc``. The ``feature_byloc`` method returns the flowline that is
  associated with the closest NHDPlus feature to the given coordinates. The
  ``comid_byloc`` method returns a point on the closest downstream flowline to
  the given coordinates.
- Add a new function called ``pygeoapi`` for calling the API in batch mode.
  This function accepts the input coordinates as a ``geopandas.GeoDataFrame``.
  It is more performant than calling its counteract ``PyGeoAPI`` multiple times.
  It's recommended to switch to using this new batch function instead of the
  ``PyGeoAPI`` class. Users just need to prepare an input data frame that has
  all the required service parameters as columns.
- Add a new step to ``prepare_nhdplus`` to convert ``MultiLineString`` to ``LineString``.
- Add support for the ``simplified`` flag of NLDI's ``get_basins`` function.
  The default value is ``True`` to retain the old behavior.

Breaking Changes
~~~~~~~~~~~~~~~~
- Remove caching-related arguments from all functions since now they
  can be set globally via three environmental variables:

  * ``HYRIVER_CACHE_NAME``: Path to the caching SQLite database.
  * ``HYRIVER_CACHE_EXPIRE``: Expiration time for cached requests in seconds.
  * ``HYRIVER_CACHE_DISABLE``: Disable reading/writing from/to the cache file.

  You can do this like so:

.. code-block:: python

    import os

    os.environ["HYRIVER_CACHE_NAME"] = "path/to/file.sqlite"
    os.environ["HYRIVER_CACHE_EXPIRE"] = "3600"
    os.environ["HYRIVER_CACHE_DISABLE"] = "true"

0.12.2 (2022-02-04)
-------------------

New Features
~~~~~~~~~~~~
- Add a new class called ``NHD`` for accessing the latest National Hydrography Dataset.
  More info regarding this data can be found
  `here <https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer>`__.
- Add two new functions for getting cross-sections along a single flowline via
  ``flowline_xsection`` or throughout a network of flowlines via ``network_xsection``.
  You can specify spacing and width parameters to control their location. For more
  information and examples please consult the documentation.
- Add a new property to ``AGRBase`` called ``service_info`` to include some useful info
  about the service including ``feature_types`` which can be handy for converting
  numeric values of types to their string equivalent.

Internal Changes
~~~~~~~~~~~~~~~~
- Use the new PyGeoAPI API.
- Refactor ``prepare_nhdplus`` for improving the performance and robustness of determining
  ``tocomid`` within a network of NHD flowlines.
- Add empty geometries that ``NLDI.getbasins`` returns to the list of ``not found`` IDs.
  This is because the NLDI service does not include non-network flowlines and instead returns
  an empty geometry for these flowlines. (:issue_nhd:`#48`)

0.12.1 (2021-12-31)
-------------------

Internal Changes
~~~~~~~~~~~~~~~~
- Use the three new ``ar.retrieve_*`` functions instead of the old ``ar.retrieve``
  function to improve type hinting and to make the API more consistent.
- Revert to the original PyGeoAPI base URL.

0.12.0 (2021-12-27)
-------------------

Breaking Changes
~~~~~~~~~~~~~~~~
- Rewrite ``ScienceBase`` to make it applicable for working with other ScienceBase
  items. A new function has been added for staging the Additional NHDPlus attributes items
  called ``stage_nhdplus_attrs``.
- Refactor ``AGRBase`` to remove unnecessary functions and make them more general.
- Update ``PyGeoAPI`` class to conform to the new ``pygeoapi`` API. This web service
  is undergoing some changes at the time of this release and the API is not stable,
  might not work as expected. As soon as the web service is stable, a new version
  will be released.

New Features
~~~~~~~~~~~~
- In ``WaterData.byid`` show a warning if there are any missing feature IDs that are
  requested but are not available in the dataset.
- For all ``by*`` methods of ``WaterData`` throw a ``ZeroMatched`` exception if no
  features are found.
- Add ``expire_after`` and ``disable_caching`` arguments to all functions that use
  ``async_retriever``. Set the default request caching expiration time to never expire.
  You can use ``disable_caching`` if you don't want to use the cached responses. Please
  refer to documentation of the functions for more details.

Internal Changes
~~~~~~~~~~~~~~~~
- Refactor ``prepare_nhdplus`` to reduce code complexity by grouping all the
  NHDPlus tools as a private class.
- Modify ``AGRBase`` to reflect the latest API changes in ``pygeoogc.ArcGISRESTfull``
  class.
- Refactor ``prepare_nhdplus`` by creating a private class that includes all the previously
  used private functions. This will make the code more readable and easier to maintain.
- Add all the missing types so ``mypy --strict`` passes.

0.11.4 (2021-11-12)
-------------------

New Features
~~~~~~~~~~~~
- Add a new argument to ``NLDI.get_basins`` called ``split_catchment`` that
  if is set to ``True`` will split the basin geometry at the watershed outlet.

Internal Changes
~~~~~~~~~~~~~~~~
- Catch service errors in ``PyGeoAPI`` and show useful error messages.
- Use ``importlib-metadata`` for getting the version instead of ``pkg_resources``
  to decrease import time as discussed in this
  `issue <https://github.com/pydata/xarray/issues/5676>`__.

0.11.3 (2021-09-10)
-------------------

Internal Changes
~~~~~~~~~~~~~~~~
- More robust handling of inputs and outputs of ``NLDI``'s methods.
- Use an alternative download link for NHDPlus VAA file on Hydroshare.
- Restructure the codebase to reduce the complexity of ``pynhd.py`` file
  by dividing it into three files: ``pynhd`` all classes that provide access
  to the supported web services, ``core`` that includes base classes, and
  ``nhdplus_derived`` that has functions for getting databases that provided
  additional attributes for the NHDPlus database.

0.11.2 (2021-08-26)
-------------------

New Features
~~~~~~~~~~~~
- Add support for `PyGeoAPI <https://labs.waterdata.usgs.gov/api/nldi/pygeoapi>`__. It offers
  four functionalities: ``flow_trace``, ``split_catchment``, ``elevation_profile``, and
  ``cross_section``.

0.11.1 (2021-07-31)
-------------------

New Features
~~~~~~~~~~~~
- Add a function for getting all NHD ``FCodes`` as a data frame, called ``nhd_fcode``.
- Improve ``prepare_nhdplus`` function by removing all coastlines and better detection
  of the terminal point in a network.

Internal Changes
~~~~~~~~~~~~~~~~
- Migrate to using ``AsyncRetriever`` for handling communications with web services.
- Catch the ``ConnectionError`` separately in ``NLDI`` and raise a ``ServiceError`` instead.
  So user knows that data cannot be returned due to the out of service status of the server
  not ``ZeroMatched``.

0.11.0 (2021-06-19)
-------------------

New Features
~~~~~~~~~~~~
- Add ``nhdplus_vaa`` to access NHDPlus Value Added Attributes for all its flowlines.
- To see a list of available layers in NHDPlus HR, you can instantiate its class without
  passing any argument like so ``NHDPlusHR()``.

Breaking Changes
~~~~~~~~~~~~~~~~
- Drop support for Python 3.6 since many of the dependencies such as ``xarray`` and ``pandas``
  have done so.

Internal Changes
~~~~~~~~~~~~~~~~
- Use persistent caching for all requests which can help speed up network responses significantly.
- Improve documentation and testing.

0.10.1 (2021-03-27)
-------------------

- Add an announcement regarding the new name for the software stack, HyRiver.
- Improve ``pip`` installation and release workflow.

0.10.0 (2021-03-06)
-------------------

- The first release after renaming hydrodata to PyGeoHydro.
- Make ``mypy`` checks more strict and fix all the errors and prevent possible
  bugs.
- Speed up CI testing by using ``mamba`` and caching.

0.9.0 (2021-02-14)
------------------

- Bump version to the same version as PyGeoHydro.

Breaking Changes
~~~~~~~~~~~~~~~~
- Add a new function for getting basins geometries for a list of USGS station IDs.
  The function is a method of ``NLDI`` class called ``get_basins``. So, now
  ``NLDI.getfeature_byid`` function does not have a basin flag. This change
  makes getting geometries easier and faster.
- Remove ``characteristics_dataframe`` method from ``NLDI`` and make a standalone function
  called ``nhdplus_attrs`` for accessing NHDPlus attributes directly from ScienceBase.
- Add support for using `hydro <https://hydro.nationalmap.gov/arcgis/rest/services/NHDPlus_HR/MapServer>`_
  or `edits <https://edits.nationalmap.gov/arcgis/rest/services/NHDPlus_HR/NHDPlus_HR/MapServer>`_
  webs services for getting NHDPlus High-Resolution using ``NHDPlusHR`` function. The new arguments
  are ``service`` which accepts ``hydro`` or ``edits``, and ``autos_switch`` flag for automatically
  switching to the other service if the ones passed by ``service`` fails.

New Features
~~~~~~~~~~~~
- Add a new argument to ``topoogical_sort`` called ``edge_attr`` that allows adding attribute(s) to
  the returned Networkx Graph. By default, it is ``None``.
- A new base class, ``AGRBase`` for connecting to ArcGISRESTful-based services such as National Map
  and EPA's WaterGEOS.
- Add support for setting the buffer distance for the input geometries to ``AGRBase.bygeom``.
- Add ``comid_byloc`` to ``NLDI`` class for getting ComIDs of the closest flowlines from a list of
  lon/lat coordinates.
- Add ``bydistance`` to ``WaterData`` for getting features within a given radius of a point.

0.2.0 (2020-12-06)
------------------

Breaking Changes
~~~~~~~~~~~~~~~~
- Re-wrote the ``NLDI`` function to use API v3 of the NLDI service.
- The ``crs`` argument of ``WaterData`` now is the target CRS of the output dataframe.
  The service CRS is now ``EPSG:4269`` for all the layers.
- Remove the ``url_only`` argument of ``NLDI`` since it's not applicable anymore.

New Features
~~~~~~~~~~~~
- Added support for NHDPlus High Resolution for getting features by geometry, IDs, or
  SQL where clause.
- The following functions are added to ``NLDI``:

* ``getcharacteristic_byid``: Getting characteristics of NHDPlus catchments.
* ``navigate_byloc``: Getting the nearest ComID to a coordinate and performing navigation.
* ``characteristics_dataframe``: Getting all the available catchment-scale characteristics
  as a data frame.
* ``get_validchars``: Getting a list of available characteristic IDs for a specified
  characteristic type.

- The following function is added to ``WaterData``:

* ``byfilter``: Getting data based on any valid CQL filter.
* ``bygeom``: Getting data within a geometry (polygon and multipolygon).

- Add support for Python 3.9 and tests for Windows.

Bug Fixes
~~~~~~~~~
- Refactored ``WaterData`` to fix the CRS inconsistencies (#1).

0.1.3 (2020-08-18)
------------------

- Replaced ``simplejson`` with ``orjson`` to speed-up JSON operations.

0.1.2 (2020-08-11)
------------------

- Add ``show_versions`` function for showing versions of the installed deps.
- Improve documentation

0.1.1 (2020-08-03)
------------------

- Improved documentation
- Refactored ``WaterData`` to improve readability.

0.1.0 (2020-07-23)
------------------

- First release on PyPI.
