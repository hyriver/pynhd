=======
History
=======

0.3.0 (unreleased)
------------------

Breaking Changes
~~~~~~~~~~~~~~~~
- Add a new function for getting basins geometries for a list of USGS station IDs.
  The function is a method of ``NLDI`` class called ``get_basins``. So, now
  ``NLDI.getfeature_byid`` function does not have a basin flag. This change
  makes getting geometries easier and faster.
- Remove ``characteristics_dataframe`` method from ``NLDI`` and made a standalone function
  called ``nhdplus_attrs`` for accessing NHDPlus attributes directly from ScienceBase.

New Features
~~~~~~~~~~~~
- Add a new argument to ``topoogical_sort`` called ``edge_attr`` that allows to
  add attribute(s) to the returned Networkx Graph. By default it is ``None``.
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
  The service CRS is now EPSG:4269 for all the layers.
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
  characteristic type,.

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
