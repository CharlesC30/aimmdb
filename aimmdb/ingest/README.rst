
Ingestion steps for NSLS-II Inner Shell Spectroscopy (ISS) beamline
==================================

This ingestion pipeline reads ``.dat`` files. These files contain commented lines (denoted by #) aka metadata, and columnated data beneath the metadata. The pipeline is tested on some example data found at ``aimmdb/_tests/eli_test_file.dat``.


* For all comment lines, if they contain a colon, remove starting hash and split at colon to create key-value pairs
* The only comment line without colon should be the header for columnated data
* Place key-value pairs in temporary metadata dictionary, ``temp_md``
* Columnated data should contain energy, i0, it, ir, iff, and aux channels
* Use data in .dat file to calculate ``mu_trans``, ``mu_fluor``, and ``mu_ref`` (see equations below)
* Split the data into three entries for AIMMDB
    * One entry for each mu channel (transmission, fluorescence, reference)
    * Each entry contains DataFrame with two columns (energy, mu), plus metadata dictionary equivalent to ``temp_md`` with key added for the channel
    * Optionally include uid as third element in each entry

``mu`` transmission
-------------------

We store a dataframe with column ``mu_trans`` which is calculated via

.. code::

    mu_trans = -log(I_trans / I_0)

This is the intensity of the transmission signal, where :math:`I_0` is the background intensity.

``mu`` fluorescence
-------------------

We store a dataframe with column called ``mu_fluor`` which is calculated via

.. code::

    mu_fluor = I_fluor / I_0

This is the intensity of the fluorescence signal. Note there is no logarithmic scaling like above.

``mu`` fluorescence
-------------------

We store a dataframe with column called ``mu_ref`` which is calculated via

.. code::

    mu_ref = -log(I_ref / I_0)

