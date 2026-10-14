"""Leak-free (point-in-time) dataset builder for NIV forecasting at 2-40 h lead times.

Each row is one *(target settlement period, lead time)* pair::

    startTime  - start of the target settlement period (UTC)
    lead_h     - hours between forecast issue time and ``startTime``
    issue      - forecast issue time  tau = startTime - lead_h
    y          - realised NIV (MWh) of the target period            [label]
    <features> - everything knowable at ``issue``

The single rule that prevents leakage (methodology Section 3.1) is **as-of availability**:

* *Forecast-type datasets* (NESO indicated imbalance, demand/wind/solar forecasts, margin) carry a
  ``publishTime``. For a row we use, per target period, the **latest publication with
  publishTime <= issue** (:func:`asof_latest`).
* *Realised-type series* (NIV, system price, market index) have no publication time. A period is
  assumed known ``niv_lag_h`` hours after it ends; only such periods enter lags and rolling
  statistics (:func:`lagged_features`). The default lag is a conservative 1 h.
* Outturn datasets with a ``publishTime`` (fuel mix, demand outturn) use the as-of rule globally.

Limitations (documented, not hidden)
------------------------------------
* NIV and prices in ``system_prices`` are the *final* settlement-run values, whereas in real time
  only indicative values exist; the lagged features therefore carry slightly less noise than
  production would. Quantifying this revision noise needs the indicative-price archive.
* WINDFOR and the day-ahead wind/solar forecast are hourly: each half-hour period uses the
  forecast of the hour that contains it (no interpolation).
* Only the **national** boundary (``N``) of the boundary-indexed NESO datasets is used in v1.
* No bank-holiday calendar yet (needs an external source).

References
----------
* Methodology: ``docs/NIV_stack_model.tex`` Sections 2-3 (information set, vintage rule).
* pandas ``merge_asof``: https://pandas.pydata.org/docs/reference/api/pandas.merge_asof.html
* Elexon dataset descriptions: https://bmrs.elexon.co.uk/api-documentation

"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

#: Lead times (hours) used to build rows; spans the project range 2-40 h.
LEADS_H: tuple[float, ...] = (2, 4, 6, 8, 12, 16, 20, 24, 28, 32, 36, 40)
#: Assumed delay (h) between the end of a settlement period and availability of its NIV/price.
NIV_LAG_H: float = 1.0
KEYS = ["startTime", "issue", "lead_h", "y"]
_PERIOD_NS = 30 * 60 * 10**9
_NS = "datetime64[ns, UTC]"


def _ns(series: pd.Series) -> np.ndarray:
    """Return a datetime Series as int64 nanoseconds since epoch."""
    return series.astype(_NS).astype("int64").to_numpy()


def _read(root: Path, name: str, columns: list[str], national_only: bool = False) -> pd.DataFrame:
    """Read a consolidated Elexon table (selected columns; optionally boundary == 'N')."""
    path = Path(root) / "processed" / "elexon" / f"{name}.parquet"
    filters = [("boundary", "==", "N")] if national_only else None
    df = pd.read_parquet(path, columns=columns, filters=filters)
    for col in ("startTime", "publishTime"):
        if col in df.columns:
            df[col] = df[col].astype(_NS)
    return df


def asof_latest(
    left: pd.DataFrame,
    right: pd.DataFrame,
    value_cols: list[str],
    prefix: str,
    *,
    left_by: str = "startTime",
    right_by: str | None = "startTime",
    shift_h: float = 0.0,
    rename: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Attach, per left row, the latest right-hand record published at or before the issue time.

    Parameters
    ----------
    left:
        Frame with columns ``issue`` (UTC) and ``left_by`` (matching key, usually ``startTime``).
    right:
        Frame with ``publishTime`` (UTC), ``right_by`` and ``value_cols``.
    value_cols:
        Columns to carry over.
    prefix:
        Name stem for the age column ``<prefix>_age_h`` (hours between issue and publication).
    left_by, right_by:
        Equality keys (``right_by=None`` joins on time only, i.e. the latest record overall).
    shift_h:
        Evaluate availability at ``issue - shift_h`` (used for revision features).
    rename:
        Optional output names for ``value_cols``.

    Returns
    -------
    pandas.DataFrame
        Indexed like ``left`` with the value columns and the age column. Missing where nothing
        had been published yet.

    """
    lcols = ["issue"] + ([left_by] if right_by else [])
    ldf = left[lcols].copy()
    ldf["_t"] = ldf["issue"] - pd.Timedelta(hours=shift_h)
    ldf = ldf.reset_index(names="_row").sort_values("_t")
    rcols = ["publishTime", *value_cols] + ([right_by] if right_by else [])
    rdf = right[rcols].sort_values("publishTime")
    kw = {"left_by": left_by, "right_by": right_by} if right_by else {}
    merged = pd.merge_asof(
        ldf, rdf, left_on="_t", right_on="publishTime", direction="backward", **kw
    )
    merged[f"{prefix}_age_h"] = (merged["_t"] - merged["publishTime"]).dt.total_seconds() / 3600
    out = merged.set_index("_row").reindex(left.index)
    out = out[[*value_cols, f"{prefix}_age_h"]]
    if rename:
        out = out.rename(columns=rename)
    return out


def lagged_features(
    start: pd.Series,
    issue: pd.Series,
    series: pd.Series,
    name: str,
    *,
    lag_h: float = NIV_LAG_H,
    windows: tuple[int, ...] = (48, 336),
    seasonal: tuple[int, ...] = (336,),
) -> pd.DataFrame:
    """Build lagged / rolling features of a half-hourly realised series, respecting availability.

    A period starting at ``s`` is known at ``s + 30 min + lag_h``. For an issue time ``tau`` the
    latest known period index is ``floor((tau - lag - 30min - origin)/30min)``.

    Parameters
    ----------
    start, issue:
        Target period start and issue time for each row (UTC).
    series:
        Realised values indexed by period ``startTime`` (UTC); gaps become NaN.
    name:
        Output prefix.
    lag_h:
        Publication delay in hours.
    windows:
        Rolling window lengths (periods) for the mean/std ending at the latest known period.
    seasonal:
        Period offsets ``k`` (e.g. 336 = same period last week) for the value at ``start - k``,
        returned only if that period is already known at ``issue`` (else NaN).

    Returns
    -------
    pandas.DataFrame
        Columns ``<name>_last``, ``<name>_last_age_h``, ``<name>_m<w>``, ``<name>_s<w>`` and
        ``<name>_seas<k>``, indexed like ``start``.

    """
    grid = pd.date_range(series.index.min(), series.index.max(), freq="30min", tz="UTC")
    arr = series.reindex(grid).to_numpy(dtype=float)
    origin = int(grid[0].as_unit("ns").value)
    n = len(arr)
    k_t = ((_ns(start) - origin) // _PERIOD_NS).astype(np.int64)
    lag_ns = int(lag_h * 3600 * 10**9)
    last = np.floor_divide(_ns(issue) - lag_ns - _PERIOD_NS - origin, _PERIOD_NS).astype(np.int64)
    last = np.minimum(last, n - 1)
    valid = last >= 0
    li = np.where(valid, last, 0)

    out: dict[str, np.ndarray] = {}
    out[f"{name}_last"] = np.where(valid, arr[li], np.nan)
    age_end = (_ns(issue) - (origin + (last + 1) * _PERIOD_NS)) / 3.6e12
    out[f"{name}_last_age_h"] = np.where(valid, age_end, np.nan)

    filled = np.nan_to_num(arr)
    cnt = np.cumsum(~np.isnan(arr))
    csum = np.cumsum(filled)
    csq = np.cumsum(filled**2)

    def _cum(c: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """Cumulative value at idx, 0 for idx < 0."""
        return np.where(idx >= 0, c[np.clip(idx, 0, n - 1)], 0.0)

    for w in windows:
        a = last - w
        c = _cum(cnt, last) - _cum(cnt, a)
        s = _cum(csum, last) - _cum(csum, a)
        q = _cum(csq, last) - _cum(csq, a)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = s / c
            var = np.maximum(q / c - mean**2, 0.0)
        ok = valid & (c >= max(3, w // 4))
        out[f"{name}_m{w}"] = np.where(ok, mean, np.nan)
        out[f"{name}_s{w}"] = np.where(ok, np.sqrt(var), np.nan)

    for k in seasonal:
        idx = k_t - k
        ok = (idx >= 0) & (idx <= last) & valid
        out[f"{name}_seas{k}"] = np.where(ok, arr[np.clip(idx, 0, n - 1)], np.nan)
    return pd.DataFrame(out, index=start.index)


def _calendar(start: pd.Series) -> pd.DataFrame:
    """Calendar features from the Europe/London local time of the target period."""
    lt = start.dt.tz_convert("Europe/London")
    pod = (lt.dt.hour * 2 + lt.dt.minute // 30).astype(int)
    dow = lt.dt.dayofweek
    doy = lt.dt.dayofyear
    return pd.DataFrame(
        {
            "pod": pod,
            "pod_sin": np.sin(2 * np.pi * pod / 48),
            "pod_cos": np.cos(2 * np.pi * pod / 48),
            "dow": dow,
            "dow_sin": np.sin(2 * np.pi * dow / 7),
            "dow_cos": np.cos(2 * np.pi * dow / 7),
            "doy_sin": np.sin(2 * np.pi * doy / 365.25),
            "doy_cos": np.cos(2 * np.pi * doy / 365.25),
            "is_wkend": (dow >= 5).astype(int),
            "local_date": lt.dt.strftime("%Y-%m-%d"),
        },
        index=start.index,
    )


#: Minimum share of non-zero NETBSAD periods for its features to be added (it is near-empty).
NETBSAD_MIN_NONZERO = 0.005


def _bsad_series(root: Path, sp: pd.DataFrame) -> dict[str, pd.Series]:
    """Build half-hourly realised series from DISBSAD (and NETBSAD when informative).

    DISBSAD lists the non-BM balancing-service adjustments entering NIV, one row per action.
    Per settlement period we aggregate: net volume (``dis_vol``), volume of System-Operator
    flagged actions (``dis_sovol``), net cost (``dis_cost``) and the number of non-zero actions
    (``dis_n``). Periods without any row are zero (nothing was adjusted), not missing.
    NETBSAD is a per-period summary that is almost always zero; its price adjustment is added as
    ``nbsad_buy_adj`` only if more than :data:`NETBSAD_MIN_NONZERO` of periods are non-zero.
    Both are published after settlement, so they are used only through
    :func:`lagged_features` with the NIV publication lag (an assumption: see module docstring).

    Returns
    -------
    dict
        ``name -> Series`` indexed by period ``startTime`` (UTC) over all periods in ``sp``.

    """
    keys = sp[["startTime", "settlementDate", "settlementPeriod"]].copy()
    keys["settlementDate"] = keys["settlementDate"].astype(str)
    out: dict[str, pd.Series] = {}
    path = root / "processed/elexon/disbsad.parquet"
    if path.exists():
        d = pd.read_parquet(
            path, columns=["settlementDate", "settlementPeriod", "cost", "volume", "soFlag"]
        )
        d["settlementDate"] = d["settlementDate"].astype(str)
        d = d.drop_duplicates()
        d = d[d["volume"].fillna(0) != 0]
        d["sovol"] = np.where(d["soFlag"].astype(bool), d["volume"], 0.0)
        g = (
            d.groupby(["settlementDate", "settlementPeriod"])
            .agg(
                dis_vol=("volume", "sum"),
                dis_sovol=("sovol", "sum"),
                dis_cost=("cost", "sum"),
                dis_n=("volume", "size"),
            )
            .reset_index()
        )
        m = keys.merge(g, on=["settlementDate", "settlementPeriod"], how="left").fillna(0.0)
        for c in ("dis_vol", "dis_sovol", "dis_cost", "dis_n"):
            out[c] = m.set_index("startTime")[c]
    path = root / "processed/elexon/netbsad.parquet"
    if path.exists():
        n = pd.read_parquet(
            path, columns=["settlementDate", "settlementPeriod", "buyPricePriceAdjustment"]
        ).drop_duplicates()
        n["settlementDate"] = n["settlementDate"].astype(str)
        share = float((n["buyPricePriceAdjustment"].fillna(0) != 0).mean())
        if share > NETBSAD_MIN_NONZERO:
            m = keys.merge(n, on=["settlementDate", "settlementPeriod"], how="left").fillna(0.0)
            out["nbsad_buy_adj"] = m.set_index("startTime")["buyPricePriceAdjustment"]
        else:
            log.info("NETBSAD skipped: only %.2f%% of periods are non-zero", 100 * share)
    return out


#: Weather: Open-Meteo Previous-Runs ``_previous_day<K>`` is the run issued about K*24 h before the
#: valid time. Assuming ~8 h from run start to availability, day K is usable at lead h iff
#: ``24*K - 8 >= h``  ->  K=1 for h<=16, K=2 for 16<h<=40.  (Assumption; see methodology.)
WX_AVAIL_H: float = 8.0
_WX_RE = re.compile(r"^(?P<var>.+?)_previous_day(?P<k>\d)(?:_(?P<model>.+))?$")


def wx_day_for_lead(lead_h: float, avail_h: float = WX_AVAIL_H) -> int:
    """Smallest previous-day offset K whose run is available at ``lead_h`` (leak rule)."""
    return int(np.ceil((lead_h + avail_h) / 24.0))


def _wx_national(root: Path) -> dict[tuple[str, int], pd.DataFrame] | None:
    """Return ``{(var, K): DataFrame[time, mean, pt_std, mod_std]}`` on a half-hourly UTC grid."""
    path = root / "processed/openmeteo/weather_previous_runs.parquet"
    if not path.exists():
        return None
    w = pd.read_parquet(path)
    w["time"] = pd.to_datetime(w["time"], utc=True).astype(_NS)
    out: dict[tuple[str, int], pd.DataFrame] = {}
    groups: dict[tuple[str, int], list[str]] = {}
    for c in w.columns:
        m = _WX_RE.match(c)
        if m:
            groups.setdefault((m["var"], int(m["k"])), []).append(c)
    for (var, k), cols in groups.items():
        d = w[["point", "time", *cols]].copy()
        if var == "wind_speed_100m":  # capped cubic power-curve proxy, per point and model
            pw = (d[cols].clip(upper=13.0) / 13.0) ** 3
            pw.columns = [f"{c}__pw" for c in cols]
            d = pd.concat([d, pw], axis=1)
        pm = d.groupby("time")
        # mean over points of the per-point model mean; spreads across points and across models
        ref = [c for c in cols if "ecmwf" in c and d[c].notna().any()] or cols
        d["_pm"] = d[ref].mean(axis=1)  # one reference model -> comparable across K
        agg = pm["_pm"].agg(["mean", "std"]).rename(columns={"std": "pt_std"})
        agg["mod_std"] = d.assign(_s=d[cols].std(axis=1)).groupby("time")["_s"].mean()
        if var == "wind_speed_100m":
            agg["pow"] = (
                d.assign(_p=d[[c for c in d.columns if c.endswith("__pw")]].mean(axis=1))
                .groupby("time")["_p"]
                .mean()
            )
        agg = agg.astype(float).sort_index()
        # hourly -> half-hourly by linear interpolation within the same run
        half = agg.reindex(agg.index.union(agg.index + pd.Timedelta(minutes=30))).interpolate(
            limit=1
        )
        out[(var, k)] = half.reset_index().rename(columns={"index": "time"})
    return out


def _weather_features(left: pd.DataFrame, root: Path) -> pd.DataFrame | None:
    """Point-in-time NWP features for each (target, lead) row (``wx_*`` columns)."""
    nat = _wx_national(root)
    if nat is None:
        log.warning("no weather file; skipping wx_ features")
        return None
    kmax = max(k for _, k in nat)
    k_row = np.ceil((left["lead_h"].to_numpy() + WX_AVAIL_H) / 24.0).astype(int)
    k_row = np.minimum(np.maximum(k_row, 1), kmax)
    res = pd.DataFrame(index=left.index)
    cols_out = {
        "wind_speed_100m": ["mean", "pt_std", "mod_std", "pow"],
        "temperature_2m": ["mean"],
        "shortwave_radiation": ["mean", "pt_std"],
    }
    tgt = left["startTime"]
    for var, stats in cols_out.items():
        for stat in stats:
            name = f"wx_{var.split('_')[0] if var != 'wind_gusts_10m' else 'gust'}_{stat}"
            name = {
                "wx_wind_mean": "wx_wspd",
                "wx_wind_pt_std": "wx_wspd_ptsd",
                "wx_wind_mod_std": "wx_wspd_modsd",
                "wx_wind_pow": "wx_wpow",
                "wx_temperature_mean": "wx_temp",
                "wx_shortwave_mean": "wx_ghi",
                "wx_shortwave_pt_std": "wx_ghi_ptsd",
                "wx_cloud_mean": "wx_cloud",
            }.get(name, name)
            vals = np.full(len(left), np.nan)
            for k in range(1, kmax + 1):
                if (var, k) not in nat or stat not in nat[(var, k)]:
                    continue
                sel = np.flatnonzero(k_row == k)
                if sel.size == 0:
                    continue
                src = nat[(var, k)].set_index("time")[stat]
                vals[sel] = src.reindex(tgt.iloc[sel]).to_numpy()
            res[name] = vals
    # run-to-run revision of the wind proxy where both runs are available (short leads)
    if ("wind_speed_100m", 1) in nat and ("wind_speed_100m", 2) in nat:
        a = nat[("wind_speed_100m", 1)].set_index("time")["pow"].reindex(tgt).to_numpy()
        b = nat[("wind_speed_100m", 2)].set_index("time")["pow"].reindex(tgt).to_numpy()
        res["wx_wpow_rev"] = np.where(k_row == 1, a - b, np.nan)
    return res


def build_dataset(
    root: str | Path,
    leads: tuple[float, ...] = LEADS_H,
    niv_lag_h: float = NIV_LAG_H,
) -> pd.DataFrame:
    """Build the full (target period x lead) feature table from consolidated Elexon data.

    Parameters
    ----------
    root:
        Data directory containing ``processed/elexon/*.parquet`` (output of ``niv fetch``).
    leads:
        Lead times in hours.
    niv_lag_h:
        Assumed availability delay of realised NIV / prices / market index.

    Returns
    -------
    pandas.DataFrame
        Sorted by ``startTime`` then ``lead_h``; columns ``KEYS`` + features (see module doc).

    """
    root = Path(root)
    sp = pd.read_parquet(
        root / "processed/elexon/system_prices.parquet",
        columns=[
            "startTime",
            "settlementDate",
            "settlementPeriod",
            "netImbalanceVolume",
            "systemSellPrice",
        ],
    )
    sp["startTime"] = sp["startTime"].astype(_NS)
    sp = sp.drop_duplicates("startTime").sort_values("startTime")
    starts = sp["startTime"].reset_index(drop=True)

    frames = []
    for h in leads:
        f = pd.DataFrame(
            {"startTime": starts, "lead_h": float(h), "y": sp["netImbalanceVolume"].values}
        )
        frames.append(f)
    left = pd.concat(frames, ignore_index=True)
    left["issue"] = left["startTime"] - pd.to_timedelta(left["lead_h"], unit="h")
    left = left.sort_values(["startTime", "lead_h"]).reset_index(drop=True)
    left["hourStart"] = left["startTime"].dt.floor("h")  # WINDFOR / B1440 are hourly series
    feats: list[pd.DataFrame] = []

    # -- NESO indicated imbalance (national) -------------------------------------------------
    ii = _read(root, "indicated_imbalance", ["publishTime", "startTime", "imbalance"], True)
    f = asof_latest(left, ii, ["imbalance"], "ind_imb", rename={"imbalance": "ind_imb"})
    old = asof_latest(left, ii, ["imbalance"], "ind_imb_old", shift_h=3.0)
    f["ind_imb_rev3h"] = f["ind_imb"] - old["imbalance"]
    del ii
    feats.append(f)

    # -- NESO indicated demand / generation, TSDF, NDF (national) ---------------------------
    for name, col, out in [
        ("indicated_demand", "demand", "ind_dem"),
        ("indicated_generation", "generation", "ind_gen"),
        ("transmission_demand_forecast", "demand", "tsdf"),
        ("demand_forecast_ndf", "demand", "ndf"),
    ]:
        r = _read(root, name, ["publishTime", "startTime", col], name != "demand_forecast_ndf")
        a = asof_latest(left, r, [col], out, rename={col: out})
        feats.append(a[[out]])
        del r

    # -- wind forecast (WINDFOR) and day-ahead wind/solar (B1440) --------------------------
    wf = pd.read_parquet(
        root / "processed/elexon/wind_forecast.parquet",
        columns=["publishTime", "startTime", "generation"],
    )
    for c in ("publishTime", "startTime"):
        wf[c] = wf[c].astype(_NS)
    feats.append(
        asof_latest(
            left,
            wf,
            ["generation"],
            "wind_fc",
            left_by="hourStart",
            rename={"generation": "wind_fc"},
        )
    )
    ds = pd.read_parquet(
        root / "processed/elexon/wind_solar_day_ahead.parquet",
        columns=["publishTime", "startTime", "psrType", "quantity"],
    )
    for c in ("publishTime", "startTime"):
        ds[c] = ds[c].astype(_NS)
    for psr, out in [
        ("Solar", "da_solar"),
        ("Wind Onshore", "da_wind_on"),
        ("Wind Offshore", "da_wind_off"),
    ]:
        r = ds[ds["psrType"] == psr]
        feats.append(
            asof_latest(left, r, ["quantity"], out, left_by="hourStart", rename={"quantity": out})[
                [out]
            ]
        )

    # -- margin forecast (daily surplus for the target's local date) ------------------------
    cal = _calendar(left["startTime"])
    mg = pd.read_parquet(
        root / "processed/elexon/margin_forecast_daily.parquet",
        columns=["publishTime", "forecastDate", "surplus"],
    )
    mg["publishTime"] = mg["publishTime"].astype(_NS)
    tmp = left[["issue"]].copy()
    tmp["forecastDate"] = cal["local_date"]
    feats.append(
        asof_latest(
            tmp,
            mg,
            ["surplus"],
            "margin",
            left_by="forecastDate",
            right_by="forecastDate",
            rename={"surplus": "margin_surplus"},
        )[["margin_surplus"]]
    )

    # -- fuel mix & demand outturn: latest published overall --------------------------------
    fh = pd.read_parquet(
        root / "processed/elexon/fuel_half_hourly.parquet",
        columns=["publishTime", "startTime", "fuelType", "generation"],
    )
    fh["publishTime"] = fh["publishTime"].astype(_NS)
    fh["grp"] = np.where(fh["fuelType"].str.startswith("INT"), "IC", fh["fuelType"])
    wide = fh.pivot_table(
        index="publishTime", columns="grp", values="generation", aggfunc="sum"
    ).reset_index()
    for col, _out in [
        ("WIND", "fuel_wind"),
        ("CCGT", "fuel_ccgt"),
        ("NUCLEAR", "fuel_nuclear"),
        ("IC", "fuel_ic_net"),
    ]:
        if col not in wide.columns:
            wide[col] = np.nan
    wide = wide.rename(
        columns={
            "WIND": "fuel_wind",
            "CCGT": "fuel_ccgt",
            "NUCLEAR": "fuel_nuclear",
            "IC": "fuel_ic_net",
        }
    )
    feats.append(
        asof_latest(
            left,
            wide,
            ["fuel_wind", "fuel_ccgt", "fuel_nuclear", "fuel_ic_net"],
            "fuel",
            right_by=None,
        )
    )
    indo = pd.read_parquet(
        root / "processed/elexon/demand_outturn_initial.parquet", columns=["publishTime", "demand"]
    )
    indo["publishTime"] = indo["publishTime"].astype(_NS)
    feats.append(
        asof_latest(left, indo, ["demand"], "indo", right_by=None, rename={"demand": "indo_last"})[
            ["indo_last"]
        ]
    )

    # -- lagged realised NIV, price, market index ------------------------------------------
    niv = sp.set_index("startTime")["netImbalanceVolume"]
    feats.append(lagged_features(left["startTime"], left["issue"], niv, "niv", lag_h=niv_lag_h))
    ssp = sp.set_index("startTime")["systemSellPrice"]
    feats.append(
        lagged_features(
            left["startTime"],
            left["issue"],
            ssp,
            "ssp",
            lag_h=niv_lag_h,
            windows=(48,),
            seasonal=(),
        )[["ssp_last", "ssp_m48", "ssp_s48"]]
    )
    mi = pd.read_parquet(
        root / "processed/elexon/market_index.parquet",
        columns=["startTime", "dataProvider", "price", "volume"],
    )
    mi["startTime"] = mi["startTime"].astype(_NS)
    mi = mi[(mi["dataProvider"] == "APXMIDP") & (mi["volume"] > 0)].drop_duplicates("startTime")
    feats.append(
        lagged_features(
            left["startTime"],
            left["issue"],
            mi.set_index("startTime")["price"],
            "mid",
            lag_h=niv_lag_h,
            windows=(48,),
            seasonal=(),
        )[["mid_last"]]
    )

    for name, series in _bsad_series(root, sp).items():
        lf = lagged_features(
            left["startTime"],
            left["issue"],
            series,
            name,
            lag_h=niv_lag_h,
            windows=(48,),
            seasonal=(),
        )
        keep = [f"{name}_last", f"{name}_m48"] + ([f"{name}_s48"] if name == "dis_vol" else [])
        feats.append(lf[keep])

    wx = _weather_features(left, root)
    if wx is not None:
        feats.append(wx)

    out = pd.concat(
        [left.drop(columns=["hourStart"]), cal.drop(columns=["local_date"]), *feats], axis=1
    )

    # -- derived features -------------------------------------------------------------------
    out["ind_missing"] = out["ind_imb"].isna().astype(int)
    out["abs_ind_imb"] = out["ind_imb"].abs()
    out["ind_imb_x_lead"] = out["ind_imb"] * out["lead_h"] / 40.0
    out["ind_gap"] = out["ind_gen"] - out["ind_dem"]
    out["emb_proxy"] = out["ndf"] - out["tsdf"]
    out["resload"] = out["ndf"] - out["wind_fc"] - out["da_solar"]
    out = out.sort_values(["startTime", "lead_h"]).reset_index(drop=True)
    log.info("dataset: %d rows, %d columns", *out.shape)
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return model input columns (everything except keys, label and ``dow``/``pod`` raw ints)."""
    return [c for c in df.columns if c not in KEYS]


def save_dataset(df: pd.DataFrame, root: str | Path) -> Path:
    """Write the dataset to ``<root>/features/niv_dataset.parquet`` and return the path."""
    path = Path(root) / "features" / "niv_dataset.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_dataset(root: str | Path) -> pd.DataFrame:
    """Load the dataset written by :func:`save_dataset`."""
    return pd.read_parquet(Path(root) / "features" / "niv_dataset.parquet")
