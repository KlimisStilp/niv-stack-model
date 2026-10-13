"""Small HTTP layer: retrying session, polite throttling, JSON helper.

Design notes
------------
* Transient failures (HTTP 429/5xx, connection resets) are retried with exponential
  back-off through :class:`urllib3.util.retry.Retry`; ``Retry-After`` is honoured.
* HTTP 400 is raised as :class:`BadRequest` so callers can react (e.g. by splitting a
  time window that exceeds a server-side range limit).

References
----------
* urllib3 Retry: https://urllib3.readthedocs.io/en/stable/reference/urllib3.util.html#urllib3.util.Retry
* requests sessions: https://requests.readthedocs.io/en/latest/user/advanced/#session-objects

"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 60  # seconds


class HttpError(RuntimeError):
    """Raised for any non-2xx response that survived retries."""

    def __init__(self, status: int, url: str, body: str = "") -> None:
        """Store status code, URL and a truncated response body."""
        super().__init__(f"HTTP {status} for {url}: {body[:300]}")
        self.status = status
        self.url = url
        self.body = body


class BadRequest(HttpError):
    """HTTP 400, typically an invalid parameter or a too-wide time range."""


def make_session(user_agent: str = "niv-stack-model/0.1 (research use)") -> requests.Session:
    """Create a :class:`requests.Session` with retries and a descriptive User-Agent."""
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({"User-Agent": user_agent, "Accept": "application/json"})
    return session


class Throttle:
    """Enforce a minimum interval between successive requests."""

    def __init__(self, min_interval: float = 0.2) -> None:
        """Set the minimum number of seconds between calls to :meth:`wait`."""
        self.min_interval = min_interval
        self._last = 0.0

    def wait(self) -> None:
        """Sleep just long enough to respect ``min_interval``."""
        delta = time.monotonic() - self._last
        if delta < self.min_interval:
            time.sleep(self.min_interval - delta)
        self._last = time.monotonic()


def get_json(
    session: requests.Session,
    url: str,
    params: Mapping[str, Any] | None = None,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    throttle: Throttle | None = None,
) -> Any:
    """GET ``url`` and return the decoded JSON body.

    Raises
    ------
    BadRequest
        On HTTP 400.
    HttpError
        On any other non-2xx status.

    """
    if throttle is not None:
        throttle.wait()
    log.debug("GET %s params=%s", url, params)
    resp = session.get(url, params=params, timeout=timeout)
    if resp.status_code == 400:
        raise BadRequest(400, url, resp.text)
    if not 200 <= resp.status_code < 300:
        raise HttpError(resp.status_code, url, resp.text)
    return resp.json()
