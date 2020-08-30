=======
History
=======

0.2.0 (unreleased)
------------------

Breaking Changes
~~~~~~~~~~~~~~~~
- Re-wrote the ``NLDI`` function to use API v3 of the NLDI service.
- The ``crs`` argument of ``WaterData`` now is the target CRS of the output dataframe.
  The service CRS is taken care of under-the-hood so there's no need for a user input.
- Remove the ``url_only`` argument of ``NLDI`` since it's not applicable anymore.

New Features
~~~~~~~~~~~~
The following functions are added to ``NLDI``:

- ``getcharacteristic_byid``: For getting characteristics of NHDPlus catchments.
- ``navigate_byloc``: For getting the nearest ComID to a coordinate and perform a navigation.
- ``characteristics_dataframe``: For getting all the available catchment-scale characteristics
  as a dataframe.

The following function is added to ``WaterData``:

- ``byfilter``: For getting data based on any valid CQL filter.

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
