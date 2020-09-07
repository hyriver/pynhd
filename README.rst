.. image:: https://raw.githubusercontent.com/cheginit/hydrodata/master/docs/_static/pynhd_logo.png
    :target: https://github.com/cheginit/pynhd
    :align: center

|

.. |hydrodata| image:: https://github.com/cheginit/hydrodata/workflows/build/badge.svg
    :target: https://github.com/cheginit/hydrodata/actions?query=workflow%3Abuild
    :alt: Github Actions

.. |pygeoogc| image:: https://github.com/cheginit/pygeoogc/workflows/build/badge.svg
    :target: https://github.com/cheginit/pygeoogc/actions?query=workflow%3Abuild
    :alt: Github Actions

.. |pygeoutils| image:: https://github.com/cheginit/pygeoutils/workflows/build/badge.svg
    :target: https://github.com/cheginit/pygeoutils/actions?query=workflow%3Abuild
    :alt: Github Actions

.. |pynhd| image:: https://github.com/cheginit/pynhd/workflows/build/badge.svg
    :target: https://github.com/cheginit/pynhd/actions?query=workflow%3Abuild
    :alt: Github Actions

.. |py3dep| image:: https://github.com/cheginit/py3dep/workflows/build/badge.svg
    :target: https://github.com/cheginit/py3dep/actions?query=workflow%3Abuild
    :alt: Github Actions

.. |pydaymet| image:: https://github.com/cheginit/pydaymet/workflows/build/badge.svg
    :target: https://github.com/cheginit/pydaymet/actions?query=workflow%3Abuild
    :alt: Github Actions

=========== ==================================================================== ============
Package     Description                                                          Status
=========== ==================================================================== ============
Hydrodata_  Access NWIS, HCDN 2009, NLCD, and SSEBop databases                   |hydrodata|
PyGeoOGC_   Send queries to any ArcGIS RESTful-, WMS-, and WFS-based services    |pygeoogc|
PyGeoUtils_ Convert responses from PyGeoOGC's supported web services to datasets |pygeoutils|
PyNHD_      Navigate and subset NHDPlus (MR and HR) using web services           |pynhd|
Py3DEP_     Access topographic data through National Map's 3DEP web service      |py3dep|
PyDaymet_   Access Daymet for daily climate data both single pixel and gridded   |pydaymet|
=========== ==================================================================== ============

.. _Hydrodata: https://github.com/cheginit/hydrodata
.. _PyGeoOGC: https://github.com/cheginit/pygeoogc
.. _PyGeoUtils: https://github.com/cheginit/pygeoutils
.. _PyNHD: https://github.com/cheginit/pynhd
.. _Py3DEP: https://github.com/cheginit/py3dep
.. _PyDaymet: https://github.com/cheginit/pydaymet

PyNHD: Navigate and extract NHDPlus database
--------------------------------------------

.. image:: https://img.shields.io/pypi/v/pynhd.svg
    :target: https://pypi.python.org/pypi/pynhd
    :alt: PyPi

.. image:: https://img.shields.io/conda/vn/conda-forge/pynhd.svg
    :target: https://anaconda.org/conda-forge/pynhd
    :alt: Conda Version

.. image:: https://codecov.io/gh/cheginit/pynhd/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/cheginit/pynhd
    :alt: CodeCov

.. image:: https://mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/gh/cheginit/hydrodata/master?filepath=docs%2Fexamples
    :alt: Binder

|

.. image:: https://www.codefactor.io/repository/github/cheginit/pynhd/badge
   :target: https://www.codefactor.io/repository/github/cheginit/pynhd
   :alt: CodeFactor

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    :alt: black

.. image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
    :target: https://github.com/pre-commit/pre-commit
    :alt: pre-commit

|

🚨 **This package is under heavy development and breaking changes are likely to happen.** 🚨

Features
--------

PyNHD is a part of Hydrodata software stack and provides access to
`WaterData <https://labs.waterdata.usgs.gov/geoserver/web/wicket/bookmarkable/org.geoserver.web.demo.MapPreviewPage?1>`__
and `NLDI <https://labs.waterdata.usgs.gov/about-nldi/>`_ web services. These two web services
can be used to navigate and extract vector data from NHDPlus V2 database such as
catchments, HUC8, HUC12, GagesII, flowlines, and water bodies. Moreover, the NLDI service
includes more than 30 NHDPlus catchment-scale attributes that are associated with NHDPlus
ComIDs. Values of these attributes are provided in three characteristic types:

1. ``local``: For individual reach catchments,
2. ``tot``: For network-accumulated values using total cumulative drainage area,
3. ``div``: For network-accumulated values using divergence-routed.

A list of these attributes for each characteristic type can be accessed using
``NLDI().get_validchars`` class method.

Additionally, PyNHD offers some extra utilities for processing the flowlines:

- ``prepare_nhdplus``: For cleaning up the dataframe by, for example, removing tiny networks,
  adding a ``to_comid`` column, and finding a terminal flowlines if it doesn't exist.
- ``topoogical_sort``: For sorting the river network topologically which is useful for routing
  and flow accumulation.
- ``vector_accumulation``: For computing flow accumulation in a river network. This function
  is generic and any routing method can be plugged in.

These utilities are developed based on an ``R`` package called
`nhdplusTools <https://github.com/USGS-R/nhdplusTools>`__.

You can try using PyNHD without installing it on you system by clicking on the binder badge
below the PyNHD banner. A Jupyter notebook instance with the Hydrodata software stack
pre-installed will be launched in your web browser and you can start coding!

Moreover, requests for additional functionalities can be submitted via
`issue tracker <https://github.com/cheginit/pynhd/issues>`__.

Installation
------------

You can install PyNHD using ``pip`` after installing ``libgdal`` on your system
(for example, in Ubuntu run ``sudo apt install libgdal-dev``):

.. code-block:: console

    $ pip install pynhd

Alternatively, PyNHD can be installed from the ``conda-forge`` repository
using `Conda <https://docs.conda.io/en/latest/>`__:

.. code-block:: console

    $ conda install -c conda-forge pynhd

Quick start
-----------

Let's explore the capabilities of ``NLDI``. We need to instantiate the class first:

.. code:: python

    from pynhd import NLDI, WaterData, NHDPlusHR
    import pynhd as nhd

First, let’s get the watershed geometry of the contributing basin of a
USGS station using ``NLDI``:

.. code:: python

    nldi = NLDI()
    station_id = "USGS-01031500"
    ut = "upstreamTributaries"
    um = "upstreamMain"

    basin = nldi.getfeature_byid("nwissite", station_id, basin=True)

The ``navigate_byid`` class method can be used to navigate NHDPlus in
both upstream and downstream of any point in the database. Let’s get ComIDs and flowlines
of the tributaries and the main river channel in the upstream of the station.

.. code:: python

    args = {
        "fsource": "nwissite",
        "fid": station_id,
        "navigation": um,
        "source": "flowlines",
        "distance": 1000,
    }

    flw_main = nldi.navigate_byid(**args)

    args["navigation"] = ut
    flw_trib = nldi.navigate_byid(**args)

We can get other USGS stations upstream (or downstream) of the station
and even set a distance limit (in km):

.. code:: python

    args.update({
        "source" : "nwissite",
    })
    st_all = nldi.navigate_byid(**args)

    args.update({
        "distance": 20,
        "source" : "nwissite",
    })
    st_d20 = nldi.navigate_byid(**args)

Now, let’s get the `HUC12 pour
points <https://www.sciencebase.gov/catalog/item/5762b664e4b07657d19a71ea>`__:

.. code:: python

    args.update({
        "distance": 1000,
        "source" : "huc12pp",
    })
    pp = nldi.navigate_byid(**args)

.. image:: https://raw.githubusercontent.com/cheginit/hydrodata/master/docs/_static/nhdplus_12_0.png
    :target: https://raw.githubusercontent.com/cheginit/hydrodata/master/docs/_static/nhdplus_12_0.png
    :width: 400
    :align: center

Now, let's get the medium- and high-resolution flowlines within the bounding box of this watershed
and compare them.

.. code:: python

    mr = WaterData("nhdflowline_network")
    nhdp_mr = mr.bybox(basin.geometry[0].bounds)

    hr = NHDPlusHR("networknhdflowline")
    nhdp_hr = hr.bygeom(basin.geometry[0].bounds)

.. image:: https://raw.githubusercontent.com/cheginit/hydrodata/master/docs/_static/hr_mr.png
    :target: https://raw.githubusercontent.com/cheginit/hydrodata/master/docs/_static/hr_mr.png
    :width: 400
    :align: center

Since NHDPlus HR is still at the pre-release stage let's use the MR flowlines to
demonstrate the vector-based accumulation.
Based on a topological sorted river network
``pynhd.vector_accumulation`` computes flow accumulation in the network.
It returns a dataframe which is sorted from upstream to downstream that
shows the accumulated flow in each node.

PyNHD has a utility called ``prepare_nhdplus`` that identifies such
relationship among other things such as fixing some common issues with
NHDPlus flowlines. But first we need to get all the NHDPlus attributes
for each ComID since ``NLDI`` only provides the flowlines’ geometries
and ComIDs which is useful for navigating the vector river network data.
For getting the NHDPlus database we use ``WaterData``. Let’s use the
``nhdflowline_network`` layer to get required info.

.. code:: python

    wd = WaterData("nhdflowline_network")

    comids = flw_trib.nhdplus_comid.to_list()
    nhdp_trib = wd.byid("comid", comids)
    flw = nhd.prepare_nhdplus(nhdp_trib, 0, 0, purge_non_dendritic=False)

Now, let’s get Mean Annual Groundwater Recharge using ``getcharacteristic_byid``
class method and carry out the accumulation.

.. code:: python

    char = "CAT_RECHG"
    area = "areasqkm"

    local = nldi.getcharacteristic_byid(comids, "local", char_ids=char)
    flw = flw.merge(local[char], left_on="comid", right_index=True)

    def runoff_acc(qin, q, a):
        return qin + q * a

    flw_r = flw[["comid", "tocomid", char, area]]
    runoff = nhd.vector_accumulation(flw_r, runoff_acc, char, [char, area])

    def area_acc(ain, a):
        return ain + a

    flw_a = flw[["comid", "tocomid", area]]
    areasqkm = nhd.vector_accumulation(flw_a, area_acc, area, [area])

    runoff /= areasqkm

Since these are catchment-scale characteristic, let’s get the catchments
then add the accumulated characteristic as a new column and plot the
results.

.. code:: python

    wd = WaterData("catchmentsp")
    catchments = wd.byid("featureid", comids)

    c_local = catchments.merge(local, left_on="featureid", right_index=True)
    c_acc = catchments.merge(runoff, left_on="featureid", right_index=True)

.. image:: https://raw.githubusercontent.com/cheginit/hydrodata/master/docs/_static/nhdplus_19_0.png
    :target: https://raw.githubusercontent.com/cheginit/hydrodata/master/docs/_static/nhdplus_19_0.png
    :width: 600
    :align: center

More examples can be found `here <https://hydrodata.readthedocs.io/en/latest/examples.html>`__.

Contributing
------------

Contributions are very welcomed. Please read
`CONTRIBUTING.rst <https://github.com/cheginit/pynhd/blob/master/CONTRIBUTING.rst>`__
file for instructions.
