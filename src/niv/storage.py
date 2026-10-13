"""Chunked Parquet storage that makes downloads resumable.

Layout::

    data/raw/<source>/<dataset>/<part>.parquet     one file per request chunk
    data/raw/<source>/<dataset>/<part>.empty       marker: chunk fetched, no rows
    data/processed/<source>/<dataset>.parquet      consolidated, de-duplicated table

Every raw chunk carries a ``fetched_at`` column (UTC) for provenance. Together with the
``publishTime``-style columns returned by forecast datasets this supports the
*point-in-time (vintage)* joins required in Section 3.1 of the methodology document.

References
----------
* Apache Parquet: https://parquet.apache.org/docs/
* pandas I/O: https://pandas.pydata.org/docs/user_guide/io.html#parquet

"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


class Store:
    """Filesystem store for raw chunks and consolidated tables."""

    def __init__(self, root: str | Path) -> None:
        """Create a store rooted at ``root``."""
        self.root = Path(root)

    # -- paths ---------------------------------------------------------------------------------
    def raw_dir(self, source: str, dataset: str) -> Path:
        """Directory holding raw chunks of ``dataset``."""
        return self.root / "raw" / source / dataset

    def processed_path(self, source: str, dataset: str) -> Path:
        """Path of the consolidated Parquet file of ``dataset``."""
        return self.root / "processed" / source / f"{dataset}.parquet"

    # -- chunks --------------------------------------------------------------------------------
    def has_chunk(self, source: str, dataset: str, part: str) -> bool:
        """Return ``True`` if the chunk was already fetched (with or without rows)."""
        d = self.raw_dir(source, dataset)
        return (d / f"{part}.parquet").exists() or (d / f"{part}.empty").exists()

    def write_chunk(self, df: pd.DataFrame, source: str, dataset: str, part: str) -> None:
        """Persist one chunk; an empty frame is recorded with an ``.empty`` marker."""
        d = self.raw_dir(source, dataset)
        d.mkdir(parents=True, exist_ok=True)
        if df.empty:
            (d / f"{part}.empty").touch()
            return
        out = df.copy()
        out["fetched_at"] = datetime.now(timezone.utc)
        out.to_parquet(d / f"{part}.parquet", index=False)

    def read_raw(self, source: str, dataset: str) -> pd.DataFrame:
        """Concatenate all raw chunks of ``dataset`` (empty frame if none)."""
        files = sorted(self.raw_dir(source, dataset).glob("*.parquet"))
        if not files:
            return pd.DataFrame()
        return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)

    # -- consolidation -------------------------------------------------------------------------
    def consolidate(self, source: str, dataset: str) -> Path | None:
        """Merge raw chunks into one de-duplicated, time-sorted Parquet file.

        Chunk boundaries can return the same record twice, so identical rows (ignoring
        ``fetched_at``) are dropped, keeping the most recent copy. Returns the output path,
        or ``None`` if there is nothing to consolidate.
        """
        df = self.read_raw(source, dataset)
        if df.empty:
            return None
        cols = [c for c in df.columns if c != "fetched_at"]
        try:  # fast path: all columns hashable (memory-friendly for 10M+ rows)
            dup = df.duplicated(subset=cols, keep="last")
        except TypeError:  # list/dict-valued cells
            dup = df[cols].astype(str).duplicated(keep="last")
        df = df.loc[~dup]
        for key in ("startTime", "publishTime", "time"):
            if key in df.columns:
                df = df.sort_values(key, kind="stable")
                break
        out = self.processed_path(source, dataset)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.reset_index(drop=True).to_parquet(out, index=False)
        return out

    def summary(self) -> pd.DataFrame:
        """Describe every consolidated dataset: rows, columns and time coverage.

        Returns
        -------
        pandas.DataFrame
            One row per ``source/dataset`` with ``rows``, on-disk sizes (``processed_MB``,
            ``raw_MB``), ``n_columns``, ``time_column``, ``first`` and ``last`` (UTC) taken from the first ``*time`` column found.

        """
        out = []
        for f in sorted((self.root / "processed").glob("*/*.parquet")):
            df = pd.read_parquet(f)
            tcol = next(
                (c for c in df.columns if c.lower().endswith("time") and c != "fetched_at"), None
            )
            out.append(
                {
                    "dataset": f"{f.parent.name}/{f.stem}",
                    "rows": len(df),
                    "processed_MB": round(f.stat().st_size / 1e6, 2),
                    "raw_MB": round(
                        sum(x.stat().st_size for x in self.raw_dir(f.parent.name, f.stem).glob("*"))
                        / 1e6,
                        2,
                    ),
                    "n_columns": df.shape[1],
                    "time_column": tcol,
                    "first": df[tcol].min() if tcol else None,
                    "last": df[tcol].max() if tcol else None,
                }
            )
        return pd.DataFrame(out)
