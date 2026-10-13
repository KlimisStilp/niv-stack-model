"""Elexon Insights Solution (BMRS) client for GB balancing-market data.

Why this module exists
----------------------
It downloads the target variable (settlement-period NIV and system prices) and the
market-state / fundamentals features listed in Table 1 of ``docs/NIV_stack_model.tex``.

Key design decisions
--------------------
1. **Endpoint paths are verified, query-parameter names are discovered.**
   Paths in :data:`CATALOGUE` were checked against the endpoint list of the Elexon OpenAPI
   client (``elexonpy``) on 2026-10-13. The *names* of the time-window parameters differ
   between datasets (``from``/``to``, ``publishDateTimeFrom``/``publishDateTimeTo``,
   ``settlementDateFrom``/``settlementDateTo``), so the client reads Elexon's live OpenAPI
   specification and picks the right pair per endpoint (see :meth:`ElexonClient.time_params`).
   Run ``niv verify`` to see what was chosen.
2. **Vintages are kept.** Forecast datasets (``prefer="publish"``) are queried by *publication*
   time so every published forecast revision is retained, as required for leak-free
   training (methodology Section 3.1).
3. **Range limits are handled automatically.** If the API answers HTTP 400 for a window
   (typically a range limit), the window is bisected until it is accepted.
4. **Resumable.** Each window is cached as a Parquet chunk (:mod:`niv.storage`).

Removed after `niv verify` (2026-10-13)
---------------------------------------
* ``/generation/outturn/halfHourlyInterconnector``: no date-range parameters (latest snapshot only);
  interconnector flows come from FUELHH/FUELINST ``INT*`` fuel types instead.
* ``/datasets/PN``: only ``settlementDate``+``settlementPeriod`` filters (~17k requests/year);
  add a dedicated per-period fetcher if schedule data is needed.

Settlement-period convention
----------------------------
GB settlement periods are 30-minute blocks numbered 1-48 (46/50 on clock-change days).
Per-day endpoints use the UTC calendar date of the chunk start as ``settlementDate``;
GB local dates differ from UTC by 1 h in summer, so re-index on ``startTime`` downstream.

References
----------
* Elexon Insights Solution portal (no API key required):
  https://developer.data.elexon.co.uk/
* OpenAPI document: https://data.elexon.co.uk/swagger/v1/swagger.json
* Endpoint inventory used for verification:
  https://github.com/openclimatefix/elexonpy/blob/main/docs/DocApiEndpointsList.md
* BMRS data descriptions (dataset codes such as IMBALNGC, WINDFOR, MID):
  https://bmrs.elexon.co.uk/api-documentation
* Elexon BSC: imbalance pricing and NIV -
  https://www.elexon.co.uk/operations-settlement/balancing-and-settlement/imbalance-pricing/

"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from niv.config import ELEXON_BASE_URL, ELEXON_SWAGGER_URL
from niv.http import BadRequest, Throttle, get_json, make_session
from niv.storage import Store

log = logging.getLogger(__name__)

SOURCE = "elexon"
MIN_WINDOW = timedelta(hours=1)  # never bisect below this

#: Candidate (from, to) parameter pairs, in order of preference, per preferred time axis.
_PAIRS_BY_PREFERENCE: dict[str, list[tuple[str, str]]] = {
    "publish": [
        ("publishDateTimeFrom", "publishDateTimeTo"),
        ("from", "to"),
        ("settlementDateFrom", "settlementDateTo"),
    ],
    "settlement": [
        ("from", "to"),
        ("settlementDateFrom", "settlementDateTo"),
        ("publishDateTimeFrom", "publishDateTimeTo"),
    ],
}


@dataclass(frozen=True)
class Endpoint:
    """Description of one Elexon endpoint in the catalogue.

    Attributes
    ----------
    key:
        Short, stable dataset name used for file names and the CLI.
    path:
        Path relative to :data:`niv.config.ELEXON_BASE_URL`. ``{settlementDate}`` marks a
        per-day endpoint.
    description:
        What the dataset contains and why it matters for NIV forecasting.
    prefer:
        ``"publish"`` to window on publication time (forecast vintages) or
        ``"settlement"`` to window on the settlement/target time (outturn).
    chunk_days:
        Initial window length. Bisected automatically if the API rejects it.
    default:
        Whether the dataset is part of the default pull. Very large datasets are opt-in.
    extra_params:
        Fixed query parameters (e.g. ``processType``).
    role:
        Role in the model (target / feature family), cross-referencing the LaTeX document.

    """

    key: str
    path: str
    description: str
    prefer: str = "settlement"
    chunk_days: float = 7.0
    default: bool = True
    extra_params: Mapping[str, str] = field(default_factory=dict)
    role: str = "feature"

    @property
    def per_day(self) -> bool:
        """``True`` if the path embeds ``{settlementDate}`` (one request per day)."""
        return "{settlementDate}" in self.path


CATALOGUE: dict[str, Endpoint] = {
    e.key: e
    for e in [
        # ---- target and balancing-market state ------------------------------------------------
        Endpoint(
            "system_prices",
            "/balancing/settlement/system-prices/{settlementDate}",
            "Settlement system sell/buy price and Net Imbalance Volume per settlement period "
            "(the forecasting TARGET).",
            role="target",
        ),
        Endpoint(
            "indicated_imbalance",
            "/datasets/IMBALNGC",
            "NESO indicated imbalance volume forecast (IMBALNGC); strongest short-lead predictor.",
            prefer="publish",
            role="market state",
            chunk_days=1,  # API rejects windows > ~1 day (observed 2026-10-13)
        ),
        Endpoint(
            "market_index",
            "/datasets/MID",
            "Market Index Data: intraday/day-ahead index price and volume per provider.",
            role="market state",
        ),
        Endpoint(
            "netbsad",
            "/datasets/NETBSAD",
            "Net balancing services adjustment volumes/prices (SO actions excluded from the BM).",
            role="market state",
        ),
        Endpoint(
            "disbsad",
            "/datasets/DISBSAD",
            "Disaggregated balancing services adjustment data.",
            role="market state",
            chunk_days=1,  # API rejects windows > ~1 day (observed 2026-10-13)
        ),
        # ---- supply / demand forecasts and outturn --------------------------------------------
        Endpoint(
            "wind_forecast",
            "/datasets/WINDFOR",
            "NESO half-hourly wind generation forecast (every publication kept).",
            prefer="publish",
            role="forecast",
        ),
        Endpoint(
            "wind_solar_day_ahead",
            "/forecast/generation/wind-and-solar/day-ahead",
            "Day-ahead wind and solar generation forecast (B1440). The processType value "
            "is NOT verified; check with `niv verify` and the API documentation.",
            extra_params={"processType": "Day Ahead"},
            role="forecast",
        ),
        Endpoint(
            "demand_forecast_ndf",
            "/datasets/NDF",
            "National demand forecast (day-ahead and beyond), every publication kept.",
            prefer="publish",
            role="forecast",
            chunk_days=1,  # API rejects windows > ~1 day (observed 2026-10-13)
        ),
        Endpoint(
            "transmission_demand_forecast",
            "/datasets/TSDF",
            "Transmission system demand forecast, every publication kept.",
            prefer="publish",
            role="forecast",
            chunk_days=1,  # API rejects windows > ~1 day (observed 2026-10-13)
        ),
        Endpoint(
            "indicated_demand",
            "/datasets/INDDEM",
            "Indicated demand (forward-looking) per settlement period.",
            prefer="publish",
            role="forecast",
            chunk_days=1,  # API rejects windows > ~1 day (observed 2026-10-13)
        ),
        Endpoint(
            "indicated_generation",
            "/datasets/INDGEN",
            "Indicated generation (forward-looking) per settlement period.",
            prefer="publish",
            role="forecast",
            chunk_days=1,  # API rejects windows > ~1 day (observed 2026-10-13)
        ),
        Endpoint(
            "margin_forecast_daily",
            "/datasets/OCNMFD",
            "Day-ahead to 14-day national margin forecast (stress/regime indicator).",
            prefer="publish",
            role="regime",
        ),
        Endpoint(
            "demand_outturn_initial",
            "/datasets/INDO",
            "Initial national demand outturn.",
            role="outturn",
        ),
        Endpoint(
            "demand_outturn_transmission",
            "/datasets/ITSDO",
            "Initial transmission system demand outturn.",
            role="outturn",
        ),
        Endpoint(
            "fuel_half_hourly",
            "/datasets/FUELHH",
            "Half-hourly generation outturn by fuel type; includes the INT* interconnector fuel "
            "types (INTFR, INTIRL, INTNED, ...), i.e. interconnector flows.",
            role="outturn",
        ),
        Endpoint(
            "fuel_instant",
            "/datasets/FUELINST",
            "5-minute instantaneous generation by fuel type (ramp features); includes the INT* "
            "interconnector fuel types, i.e. interconnector flows.",
            chunk_days=3,
            role="outturn",
        ),
        # ---- heavy, opt-in datasets -----------------------------------------------------------
        Endpoint(
            "remit",
            "/datasets/REMIT",
            "REMIT urgent market messages (outages/derates), by publication time.",
            prefer="publish",
            chunk_days=1,
            default=False,
            role="outages",
        ),
        Endpoint(
            "acceptances",
            "/datasets/BOALF",
            "Bid-offer acceptance levels (BOALF): the SO's balancing actions. Large.",
            chunk_days=1,
            default=False,
            role="market state",
        ),
        Endpoint(
            "bid_offer",
            "/datasets/BOD",
            "Bid-offer data (price stack). Very large.",
            chunk_days=1,
            default=False,
            role="market state",
        ),
    ]
}


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------
def iter_windows(
    start: datetime, end: datetime, chunk_days: float
) -> Iterator[tuple[datetime, datetime]]:
    """Yield consecutive half-open windows of ``chunk_days`` covering ``[start, end)``."""
    step = timedelta(days=chunk_days)
    t = start
    while t < end:
        yield t, min(t + step, end)
        t += step


def window_label(t0: datetime, t1: datetime) -> str:
    """Return a file-name-safe label for a window."""
    return f"{t0:%Y%m%dT%H%M}__{t1:%Y%m%dT%H%M}"


def records_from_payload(payload: Any) -> list[dict[str, Any]]:
    """Extract the list of records from an Elexon JSON payload.

    Elexon returns either a bare list (``/stream`` endpoints) or an object
    ``{"metadata": ..., "data": [...]}`` when ``format=json``.
    """
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list):
            return data
        return [payload] if payload else []
    raise TypeError(f"Unexpected payload type: {type(payload)!r}")


def to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert records to a DataFrame and parse time columns to UTC timestamps.

    Columns whose name ends in ``time`` (case-insensitive; ``startTime``, ``publishTime``,
    ``publishDateTime``...) are parsed. ``settlementDate`` is deliberately left as text.
    """
    if not records:
        return pd.DataFrame()
    df = pd.json_normalize(records)
    for col in df.columns:
        if col.lower().endswith("time") and (
            pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col])
        ):
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    return df


def _fmt(value: datetime, date_only: bool) -> str:
    """Format a datetime for the API (ISO-8601 UTC, or ``YYYY-MM-DD``)."""
    return f"{value:%Y-%m-%d}" if date_only else f"{value:%Y-%m-%dT%H:%M:%SZ}"


# ---------------------------------------------------------------------------------------------
# client
# ---------------------------------------------------------------------------------------------
class ElexonClient:
    """Thin client over the Elexon Insights API with spec-driven parameter discovery."""

    def __init__(
        self,
        session: requests.Session | None = None,
        *,
        base_url: str = ELEXON_BASE_URL,
        spec: dict[str, Any] | None = None,
        throttle: Throttle | None = None,
    ) -> None:
        """Create a client.

        Parameters
        ----------
        session:
            Optional preconfigured session (useful for tests).
        base_url:
            API root.
        spec:
            Pre-loaded OpenAPI document; if omitted call :meth:`load_spec`.
        throttle:
            Request pacing; defaults to 0.2 s between calls.

        """
        self.session = session or make_session()
        self.base_url = base_url.rstrip("/")
        self.spec = spec
        self.throttle = throttle or Throttle(0.2)

    # -- OpenAPI discovery -------------------------------------------------------------------
    def load_spec(self, cache_path: str | Path | None = None) -> dict[str, Any] | None:
        """Download the OpenAPI document, falling back to a cached copy.

        Parameters
        ----------
        cache_path:
            File used to cache the spec between runs.

        Returns
        -------
        dict or None
            The spec, or ``None`` if neither the network nor a cache is available (the client
            then falls back to default parameter names).

        """
        cache = Path(cache_path) if cache_path else None
        try:
            self.spec = get_json(self.session, ELEXON_SWAGGER_URL, throttle=self.throttle)
            if cache:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(self.spec))
        except Exception as exc:  # noqa: BLE001 - any failure -> try cache
            log.warning("Could not download OpenAPI spec (%s).", exc)
            if cache and cache.is_file():
                self.spec = json.loads(cache.read_text())
                log.warning("Using cached spec %s", cache)
            else:
                self.spec = None
        return self.spec

    def query_params(self, path: str) -> dict[str, str]:
        """Return ``{lower-case name: actual name}`` of the GET parameters of ``path``.

        Empty if the spec is unavailable or does not contain the path.
        """
        if not self.spec:
            return {}
        op = self.spec.get("paths", {}).get(path, {}).get("get", {})
        return {
            p["name"].lower(): p["name"]
            for p in op.get("parameters", [])
            if isinstance(p, dict) and "name" in p
        }

    def time_params(self, ep: Endpoint) -> tuple[str, str] | None:
        """Choose the (from, to) parameter names for a range endpoint.

        Follows ``ep.prefer``; when the spec is known, the first candidate pair whose two
        names both exist is used. Without a spec the first candidate is assumed.

        Returns
        -------
        tuple[str, str] or None
            ``None`` if the spec is known and no candidate pair matches (misconfiguration).

        """
        names = self.query_params(ep.path)
        candidates = _PAIRS_BY_PREFERENCE[ep.prefer]
        if not names:
            return candidates[0]
        for lo, hi in candidates:
            if lo.lower() in names and hi.lower() in names:
                return names[lo.lower()], names[hi.lower()]
        return None

    # -- requests ---------------------------------------------------------------------------
    def _get(self, path: str, params: Mapping[str, Any]) -> pd.DataFrame:
        """GET ``path`` and return the parsed records as a DataFrame."""
        payload = get_json(self.session, f"{self.base_url}{path}", params, throttle=self.throttle)
        return to_frame(records_from_payload(payload))

    def fetch_window(self, ep: Endpoint, t0: datetime, t1: datetime) -> pd.DataFrame:
        """Fetch ``[t0, t1)`` for one endpoint, bisecting on HTTP 400 range errors.

        Raises
        ------
        ValueError
            If no time-window parameters can be determined for the endpoint.
        niv.http.BadRequest
            If the API still rejects a window of :data:`MIN_WINDOW` (a real parameter error).

        """
        params: dict[str, Any] = {"format": "json", **ep.extra_params}
        if ep.per_day:
            path = ep.path.format(settlementDate=f"{t0:%Y-%m-%d}")
            return self._get(path, params)

        pair = self.time_params(ep)
        if pair is None:
            raise ValueError(
                f"{ep.key}: none of the known time-parameter pairs exist in the OpenAPI "
                f"spec for {ep.path}; run `niv verify` and update CATALOGUE."
            )
        lo, hi = pair
        date_only = lo.lower().startswith("settlementdate")
        params[lo] = _fmt(t0, date_only)
        params[hi] = _fmt(t1, date_only)
        try:
            return self._get(ep.path, params)
        except BadRequest:
            if t1 - t0 <= MIN_WINDOW:
                raise
            mid = t0 + (t1 - t0) / 2
            log.info("%s: HTTP 400 for %s..%s, bisecting", ep.key, t0, t1)
            return pd.concat(
                [self.fetch_window(ep, t0, mid), self.fetch_window(ep, mid, t1)],
                ignore_index=True,
            )

    def fetch(
        self,
        ep: Endpoint,
        start: datetime,
        end: datetime,
        store: Store,
        *,
        force: bool = False,
        dry_run: bool = False,
    ) -> int:
        """Download ``[start, end)`` for ``ep`` into ``store``; return the number of chunks.

        Chunks already on disk are skipped unless ``force``. With ``dry_run`` nothing is
        requested; planned windows are only logged.
        """
        chunk_days = 1.0 if ep.per_day else ep.chunk_days
        n = 0
        for t0, t1 in iter_windows(start, end, chunk_days):
            part = window_label(t0, t1)
            n += 1
            if dry_run:
                log.info("[dry-run] %s %s", ep.key, part)
                continue
            if store.has_chunk(SOURCE, ep.key, part) and not force:
                continue
            store.write_chunk(self.fetch_window(ep, t0, t1), SOURCE, ep.key, part)
        return n


def verify_catalogue(client: ElexonClient) -> list[dict[str, Any]]:
    """Compare :data:`CATALOGUE` with the OpenAPI spec.

    Returns one row per endpoint with ``path_found``, the chosen time parameters and the
    full list of available parameters, so that mismatches are visible at a glance.
    """
    rows = []
    for ep in CATALOGUE.values():
        found = bool(client.spec) and ep.path in client.spec.get("paths", {})
        names = client.query_params(ep.path)
        rows.append(
            {
                "key": ep.key,
                "path": ep.path,
                "path_found": found,
                "time_params": None if ep.per_day else client.time_params(ep),
                "extra_params_present": all(k.lower() in names for k in ep.extra_params),
                "params": sorted(names.values()),
            }
        )
    return rows
