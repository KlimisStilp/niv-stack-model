"""NESO Data Portal (CKAN) helper.

Some inputs named in the methodology (embedded wind/solar estimates, constraint flows and
limits, historic generation mix, ...) are published on the NESO Data Portal rather than on
BMRS. The portal is a CKAN instance, so resources are addressed by *resource id*. Because
ids and column names change, this module offers discovery (:func:`search_packages`) and a
generic downloader (:func:`fetch_resource`) instead of a hard-coded list; pick resource ids
via ``niv neso-search <query>`` and then ``niv fetch --source neso --resource <id> --name <n>``.

References
----------
* NESO Data Portal: https://www.neso.energy/data-portal
* CKAN Action API (package_search, datastore_search):
  https://docs.ckan.org/en/latest/maintaining/datastore.html#api-reference
* Portal API base used here: https://api.neso.energy/api/3/action/

"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import requests

from niv.http import Throttle, get_json, make_session
from niv.storage import Store

log = logging.getLogger(__name__)

SOURCE = "neso"
CKAN_BASE = "https://api.neso.energy/api/3/action"
PAGE_SIZE = 10_000


def search_packages(
    query: str, rows: int = 10, session: requests.Session | None = None
) -> list[dict[str, Any]]:
    """Search datasets (CKAN ``package_search``) and return id/title/resource summaries."""
    payload = get_json(
        session or make_session(), f"{CKAN_BASE}/package_search", {"q": query, "rows": rows}
    )
    out = []
    for pkg in payload["result"]["results"]:
        out.append(
            {
                "package": pkg.get("name"),
                "title": pkg.get("title"),
                "resources": [
                    {"id": r.get("id"), "name": r.get("name"), "format": r.get("format")}
                    for r in pkg.get("resources", [])
                ],
            }
        )
    return out


def fetch_resource(
    resource_id: str,
    name: str,
    store: Store,
    *,
    session: requests.Session | None = None,
    filters_sql: str | None = None,
) -> int:
    r"""Download an entire CKAN datastore resource, paginating with ``limit``/``offset``.

    Parameters
    ----------
    resource_id:
        CKAN resource UUID.
    store:
        Destination :class:`niv.storage.Store`.
    session:
        Optional preconfigured HTTP session.
    name:
        Dataset name used for the storage folder.
    filters_sql:
        Optional SQL ``WHERE`` body (e.g. ``"\"SETTLEMENT_DATE\" >= '2025-10-01'"``); when given
        the ``datastore_search_sql`` action is used with the same pagination.

    Returns
    -------
    int
        Number of rows downloaded.

    """
    sess = session or make_session()
    throttle = Throttle(0.3)
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        if filters_sql:
            sql = (
                f'SELECT * FROM "{resource_id}" WHERE {filters_sql} '
                f"LIMIT {PAGE_SIZE} OFFSET {offset}"
            )
            payload = get_json(
                sess, f"{CKAN_BASE}/datastore_search_sql", {"sql": sql}, throttle=throttle
            )
        else:
            payload = get_json(
                sess,
                f"{CKAN_BASE}/datastore_search",
                {"resource_id": resource_id, "limit": PAGE_SIZE, "offset": offset},
                throttle=throttle,
            )
        batch = payload["result"]["records"]
        rows.extend(batch)
        if len(batch) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    df = pd.DataFrame(rows)
    store.write_chunk(df, SOURCE, name, f"resource_{resource_id}")
    return len(df)
