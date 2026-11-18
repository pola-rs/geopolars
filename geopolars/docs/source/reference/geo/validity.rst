===================
Validity and repair
===================

.. currentmodule:: geopolars.geo

Validity
~~~~~~~~
.. autosummary::
    :toctree: api/

    is_valid_geoarrow
    is_valid_ogc
    null_out_invalid_geoarrow
    null_out_invalid_ogc

Repair
~~~~~~
.. autosummary::
    :toctree: api/

    close_rings
    filter_coordinates
    NonFinite
    fix_out_of_bounds
    OutOfBounds
    remove_repeats_in_succession
