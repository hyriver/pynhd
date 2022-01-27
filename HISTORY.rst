=======
History
=======

0.12.2 (unreleased)
-------------------

New Features
~~~~~~~~~~~~
- Add a new class called ``NHD`` for accessing the latest National Hydrography Dataset.
  More info regarding this data can be found
  `here <https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer>`__.
- Add two new functions for getting cross-sections along a single flowline via
  ``flowline_xsection`` or throughout a network of flowlines via ``network_xsection``.
  You can specify spacing and width parameters to control their location. For more
  information and example please consult the documentations.
- Add a new property to ``AGRBase`` called ``service_info`` to include some useful info
  about the service including ``feature_types`` which can be handy for converting
  numeric values of types to their string equivalent.

Internal Changes
~~~~~~~~~~~~~~~~
- Use the new PyGeoAPI API.
- Refactor ``prepare_nhdplus`` for improving the performance and robustness of determining
  ``tocomid`` within a network of NHD flowlines.

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
- Rewrite ``ScienceBase`` to make it generally usable for working with other ScienceBase
  items. A new function has been added for staging the Additional NHDPlus attributes items
  called ``stage_nhdplus_attrs``.
- Refactor ``AGRBase`` to remove unnecessary functions and make it more general.
- Update ``PyGeoAPI`` class to conform to the new ``pygeoapi`` API. This web service
  is undergoing some changes at the time of this release and API is not stable,
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
  refer to documentations of the functions for more details.

Internal Changes
~~~~~~~~~~~~~~~~
- Refactor ``prepare_nhdplus`` to reduce code complexity by grouping all the
  NHDPlus tools as a private class.
- Modify ``AGRBase`` to reflect the latest API changes in ``pygeoogc.ArcGISRESTfull``
  class.
- Refactor ``prepare_nhdplus`` by creating a private class that include all the previously
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
- Restructure the code base to reduce the complexity of ``pynhd.py`` file
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

- Add announcement regarding the new name for the software stack, HyRiver.
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
- Remove ``characteristics_dataframe`` method from ``NLDI`` and made a standalone function
  called ``nhdplus_attrs`` for accessing NHDPlus attributes directly from ScienceBase.
- Add support for using `hydro <https://hydro.nationalmap.gov/arcgis/rest/services/NHDPlus_HR/MapServer>`_
  or `edits <https://edits.nationalmap.gov/arcgis/rest/services/NHDPlus_HR/NHDPlus_HR/MapServer>`_
  webs services for getting NHDPlus High-Resolution using ``NHDPlusHR`` function. The new arguments
  are ``service`` which accepts ``hydro`` or ``edits``, and ``autos_switch`` flag for automatically
  switching to the other service if the ones passed by ``service`` fails.

New Features
~~~~~~~~~~~~
- Add a new argument to ``topoogical_sort`` called ``edge_attr`` that allows to
  add attribute(s) to the returned Networkx Graph. By default, it is ``None``.
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

* ``getcharacteristic_byid``: For getting characteristics of NHDPlus catchments.
* ``navigate_byloc``: For getting the nearest ComID to a coordinate and perform a navigation.
* ``characteristics_dataframe``: For getting all the available catchment-scale characteristics
  as a dataframe.
* ``get_validchars``: For getting a list of available characteristic IDs for a specified
  characteristic type.

- The following function is added to ``WaterData``:

* ``byfilter``: For getting data based on any valid CQL filter.
* ``bygeom``: For getting data within a geometry (polygon and multipolygon).

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
- Improve documentations

0.1.1 (2020-08-03)
------------------

- Improved documentation
- Refactored ``WaterData`` to improve readability.

0.1.0 (2020-07-23)
------------------

- First release on PyPI.
