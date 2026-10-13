"""Archived numerical-weather-prediction (NWP) forecasts from Open-Meteo.

Two modes
---------
``historical_forecast``
    Open-Meteo *Historical Forecast API*: archived model output stitched from the first hours
    of successive model runs ("best available" analysis-like forecast). **Not** a
    lead-time-specific vintage: it understates real forecast error at 2-40 h leads.
``previous_runs``
    Open-Meteo *Previous Runs API*: the value forecast ``k`` days earlier
    (variables suffixed ``_previous_day<k>``). ``k=1,2`` approximate the 24-48 h leads of
    the project and are the better choice for leak-free training. Availability of
    variables/models and history length must be checked in the Open-Meteo documentation.

Both endpoints and parameter names follow the Open-Meteo documentation; model identifiers
(e.g. ``ecmwf_ifs025``) are passed through unchanged and should be verified there.

The point list :data:`UK_POINTS` is illustrative; replace it with capacity-weighted wind-farm
and demand-centre coordinates before serious modelling.

References
----------
* Historical Forecast API: https://open-meteo.com/en/docs/historical-forecast-api
* Previous Runs API: https://open-meteo.com/en/docs/previous-runs-api
* Licence/terms: https://open-meteo.com/en/terms
* Why reanalysis must not replace forecasts as features: see methodology Section 3.1 and
  Browell & Fasiolo (2021), IEEE Trans. Smart Grid 12(6).

"""

from __future__ import annotations

import logging
import os
from datetime import date, timedelta
from typing import Any

import pandas as pd
import requests

from niv.http import Throttle, get_json, make_session
from niv.storage import Store

log = logging.getLogger(__name__)

SOURCE = "openmeteo"
HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

#: name -> (latitude, longitude). Illustrative wind/solar/demand locations across GB.
UK_POINTS: dict[str, tuple[float, float]] = {
    "moray_firth_offshore": (58.1, -2.8),
    "scottish_borders_onshore": (55.5, -3.2),
    "central_scotland": (55.9, -4.0),
    "dogger_bank_offshore": (54.7, 2.0),
    "hornsea_offshore": (53.9, 1.8),
    "irish_sea_offshore": (53.9, -3.5),
    "east_anglia_offshore": (52.7, 2.5),
    "thames_estuary_offshore": (51.6, 1.5),
    "pennines": (53.6, -2.0),
    "wales_mid": (52.3, -3.5),
    "south_west": (50.6, -4.4),
    "london": (51.5, -0.1),
    "midlands": (52.5, -1.6),
}

DEFAULT_HOURLY = [
    "wind_speed_100m",
    "wind_direction_100m",
    "wind_gusts_10m",
    "temperature_2m",
    "shortwave_radiation",
    "cloud_cover",
]
DEFAULT_MODELS = ["ecmwf_ifs025", "gfs_seamless", "icon_eu", "ukmo_seamless"]


def build_params(
    lat: float,
    lon: float,
    start: date,
    end: date,
    *,
    hourly: list[str] | None = None,
    models: list[str] | None = None,
    previous_days: tuple[int, ...] = (),
) -> dict[str, Any]:
    """Build the query parameters for one point and date range (inclusive).

    Parameters
    ----------
    lat, lon:
        Location in degrees.
    start, end:
        Inclusive date range.
    hourly:
        Variables (default :data:`DEFAULT_HOURLY`).
    models:
        NWP models (default :data:`DEFAULT_MODELS`).
    previous_days:
        For the Previous Runs API: lead offsets in days; each variable ``v`` is requested
        as ``v_previous_day<k>``. Empty for the Historical Forecast API.

    """
    variables = hourly or DEFAULT_HOURLY
    if previous_days:
        variables = [f"{v}_previous_day{k}" for v in variables for k in previous_days]
    params: dict[str, Any] = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": ",".join(variables),
        "models": ",".join(models or DEFAULT_MODELS),
        "timezone": "UTC",
        "wind_speed_unit": "ms",
    }
    key = os.environ.get("OPENMETEO_API_KEY")
    if key:
        params["apikey"] = key
    return params


def payload_to_frame(payload: dict[str, Any], point: str) -> pd.DataFrame:
    """Convert an Open-Meteo ``hourly`` payload into a tidy DataFrame.

    Adds ``point`` and a UTC ``time`` column; value columns keep Open-Meteo's names
    (which carry the model suffix when several models are requested).
    """
    hourly = payload.get("hourly")
    if not hourly:
        return pd.DataFrame()
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df.insert(0, "point", point)
    return df


def month_windows(start: date, end: date, days: int = 31) -> list[tuple[date, date]]:
    """Split the inclusive range ``[start, end]`` into windows of at most ``days`` days."""
    out, t = [], start
    while t <= end:
        out.append((t, min(t + timedelta(days=days - 1), end)))
        t += timedelta(days=days)
    return out


def fetch_weather(
    store: Store,
    start: date,
    end: date,
    *,
    mode: str = "historical_forecast",
    points: dict[str, tuple[float, float]] | None = None,
    hourly: list[str] | None = None,
    models: list[str] | None = None,
    previous_days: tuple[int, ...] = (1, 2),
    session: requests.Session | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> int:
    """Download archived NWP forecasts for all points into ``store``.

    Parameters
    ----------
    store:
        Destination store; dataset name is ``weather_<mode>``.
    start, end:
        Inclusive date range.
    mode:
        ``"historical_forecast"`` or ``"previous_runs"`` (see module docstring).
    points:
        ``{name: (lat, lon)}`` (default :data:`UK_POINTS`).
    hourly, models:
        Variables and NWP models (defaults above).
    previous_days:
        Lead offsets used when ``mode == "previous_runs"``.
    session:
        Optional preconfigured HTTP session (tests).
    force:
        Re-download chunks that are already cached.
    dry_run:
        Log the plan without calling the API.

    Returns
    -------
    int
        Number of (point, window) chunks planned/processed.

    """
    if mode not in ("historical_forecast", "previous_runs"):
        raise ValueError(f"unknown weather mode {mode!r}")
    url = HISTORICAL_FORECAST_URL if mode == "historical_forecast" else PREVIOUS_RUNS_URL
    prev = previous_days if mode == "previous_runs" else ()
    sess = session or make_session()
    throttle = Throttle(0.3)
    dataset = f"weather_{mode}"
    n = 0
    for name, (lat, lon) in (points or UK_POINTS).items():
        for w0, w1 in month_windows(start, end):
            part = f"{name}__{w0:%Y%m%d}__{w1:%Y%m%d}"
            n += 1
            if dry_run:
                log.info("[dry-run] %s %s", dataset, part)
                continue
            if store.has_chunk(SOURCE, dataset, part) and not force:
                continue
            params = build_params(
                lat, lon, w0, w1, hourly=hourly, models=models, previous_days=prev
            )
            payload = get_json(sess, url, params, throttle=throttle)
            store.write_chunk(payload_to_frame(payload, name), SOURCE, dataset, part)
    return n
