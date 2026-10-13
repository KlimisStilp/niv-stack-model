"""ENTSO-E Transparency Platform data for markets coupled to GB (optional).

GB left the EU internal electricity market, so GB's own series are mostly published on
BMRS/NESO; ENTSO-E is used for *neighbouring* zones (FR, BE, NL, NO2, IE/SEM): day-ahead
prices and load / wind / solar forecasts, which proxy cross-border regimes (methodology
Table 1, block C).

Requires an API token (``ENTSOE_API_KEY``; see ``.env.example``) and the optional
dependency ``entsoe-py`` (``pip install -e .[entsoe]``). Zone codes follow ``entsoe-py``'s
``Area`` enumeration and should be verified there.

References
----------
* Transparency Platform and token request: https://transparency.entsoe.eu/
* RESTful API guide: https://documenter.getpostman.com/view/7009892/2s93JtP3F6
* entsoe-py: https://github.com/EnergieID/entsoe-py

"""

from __future__ import annotations

import logging
import os
from datetime import date

import pandas as pd

from niv.storage import Store

log = logging.getLogger(__name__)

SOURCE = "entsoe"
ZONES = ["FR", "BE", "NL", "NO_2", "IE_SEM"]
DATASETS = ("day_ahead_prices", "load_forecast", "wind_solar_forecast")


def _client():
    """Return an ``EntsoePandasClient`` or raise a helpful error."""
    try:
        from entsoe import EntsoePandasClient
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("Install the extra: pip install -e .[entsoe]") from exc
    key = os.environ.get("ENTSOE_API_KEY")
    if not key:
        raise RuntimeError("Set ENTSOE_API_KEY (see .env.example).")
    return EntsoePandasClient(api_key=key)


def _to_frame(obj: pd.Series | pd.DataFrame, zone: str) -> pd.DataFrame:
    """Normalise an entsoe-py result to a long frame with ``time`` (UTC) and ``zone``."""
    df = obj.to_frame("value") if isinstance(obj, pd.Series) else obj.copy()
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "time"
    df.columns = [str(c) if not isinstance(c, tuple) else "_".join(map(str, c)) for c in df.columns]
    df = df.reset_index()
    df.insert(1, "zone", zone)
    return df


def fetch_entsoe(
    store: Store,
    start: date,
    end: date,
    *,
    zones: list[str] | None = None,
    datasets: tuple[str, ...] = DATASETS,
    force: bool = False,
    dry_run: bool = False,
) -> int:
    """Download ENTSO-E series per zone into ``store`` (one chunk per zone and dataset).

    Parameters
    ----------
    store:
        Destination :class:`niv.storage.Store`.
    start, end:
        Inclusive start, exclusive end dates.
    zones:
        Bidding zones (default :data:`ZONES`).
    datasets:
        Subset of :data:`DATASETS`.
    force:
        Re-download chunks that are already cached.
    dry_run:
        Log the plan without calling the API.

    Returns
    -------
    int
        Number of chunks planned/processed.

    Notes
    -----
    Load/generation *forecast* series from ENTSO-E are published day-ahead; they carry no
    publication timestamp in the response, so treat them as available from ~D-1 12:00 CET
    when building point-in-time features.

    """
    client = None if dry_run else _client()
    t0 = pd.Timestamp(start, tz="UTC")
    t1 = pd.Timestamp(end, tz="UTC")
    n = 0
    for zone in zones or ZONES:
        for ds in datasets:
            part = f"{zone}__{start:%Y%m%d}__{end:%Y%m%d}"
            n += 1
            if dry_run:
                log.info("[dry-run] entsoe %s %s", ds, part)
                continue
            if store.has_chunk(SOURCE, ds, part) and not force:
                continue
            try:
                if ds == "day_ahead_prices":
                    obj = client.query_day_ahead_prices(zone, start=t0, end=t1)
                elif ds == "load_forecast":
                    obj = client.query_load_forecast(zone, start=t0, end=t1)
                else:
                    obj = client.query_wind_and_solar_forecast(zone, start=t0, end=t1)
            except Exception as exc:  # noqa: BLE001 - e.g. NoMatchingDataError for a zone
                log.warning("entsoe %s %s failed: %s", ds, zone, exc)
                obj = pd.DataFrame()
            frame = _to_frame(obj, zone) if len(obj) else pd.DataFrame()
            store.write_chunk(frame, SOURCE, ds, part)
    return n
