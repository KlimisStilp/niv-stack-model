"""Data acquisition for a stacked probabilistic GB Net Imbalance Volume (NIV) forecasting model.

The package downloads, caches and consolidates the raw inputs described in
``docs/NIV_stack_model.tex`` (Section 3, "Data resources"):

* :mod:`niv.sources.elexon`   - Elexon Insights (BMRS) API: NIV, prices, forecasts.
* :mod:`niv.sources.weather`  - Open-Meteo archived NWP forecasts.
* :mod:`niv.sources.entsoe`   - ENTSO-E Transparency Platform (neighbouring markets; needs a token).
* :mod:`niv.sources.neso`     - NESO Data Portal (CKAN) helper for datasets not on BMRS.

See ``docs/REFERENCES.md`` for the documentation each module is built on.
"""

__version__ = "0.1.0"
