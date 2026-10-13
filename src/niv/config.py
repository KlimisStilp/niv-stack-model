"""Project-wide configuration: URLs, default time window and data directory.

References
----------
* Elexon Insights Solution API portal: https://developer.data.elexon.co.uk/
* Elexon OpenAPI specification: https://data.elexon.co.uk/swagger/v1/swagger.json

"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ELEXON_BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1"
ELEXON_SWAGGER_URL = "https://data.elexon.co.uk/swagger/v1/swagger.json"

#: Default look-back, in days, for the initial data pull (the project asks for one year).
DEFAULT_DAYS = 365


def data_dir() -> Path:
    """Return the root data directory.

    Taken from the ``NIV_DATA_DIR`` environment variable, defaulting to ``./data``.
    The directory is created if needed.
    """
    root = Path(os.environ.get("NIV_DATA_DIR", "data")).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root


def default_window(days: int = DEFAULT_DAYS, end: date | None = None) -> tuple[datetime, datetime]:
    """Return a half-open UTC window ``[start, end)`` covering ``days`` complete days.

    Parameters
    ----------
    days:
        Number of whole days to look back.
    end:
        Exclusive end date (UTC midnight). Defaults to *today*, so the last
        included day is yesterday, which guarantees complete settlement days.

    Returns
    -------
    tuple[datetime, datetime]
        Timezone-aware UTC ``(start, end)``.

    """
    end_date = end or datetime.now(timezone.utc).date()
    end_dt = datetime(end_date.year, end_date.month, end_date.day, tzinfo=timezone.utc)
    return end_dt - timedelta(days=days), end_dt


def load_dotenv(path: str | Path = ".env") -> None:
    """Load ``KEY=VALUE`` lines from a ``.env`` file into ``os.environ`` (no overriding).

    A tiny dependency-free replacement for python-dotenv.
    """
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
