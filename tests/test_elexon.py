from datetime import datetime, timedelta, timezone

import pandas as pd

from niv.sources import elexon
from niv.storage import Store
from tests.conftest import SPEC, FakeSession

UTC = timezone.utc
T0 = datetime(2025, 1, 1, tzinfo=UTC)


def client(handler, no_throttle, spec=SPEC):
    return elexon.ElexonClient(FakeSession(handler), spec=spec, throttle=no_throttle)


def test_time_params_follow_preference_and_spec(no_throttle):
    c = client(lambda u, p: [], no_throttle)
    assert c.time_params(elexon.CATALOGUE["indicated_imbalance"]) == (
        "publishDateTimeFrom",
        "publishDateTimeTo",
    )
    assert c.time_params(elexon.CATALOGUE["fuel_half_hourly"]) == (
        "settlementDateFrom",
        "settlementDateTo",
    )
    assert c.time_params(elexon.CATALOGUE["market_index"]) == ("from", "to")


def test_time_params_without_spec_use_first_candidate(no_throttle):
    c = client(lambda u, p: [], no_throttle, spec=None)
    assert c.time_params(elexon.CATALOGUE["wind_forecast"]) == (
        "publishDateTimeFrom",
        "publishDateTimeTo",
    )


def test_unknown_params_return_none(no_throttle):
    spec = {"paths": {"/datasets/MID": {"get": {"parameters": [{"name": "foo"}]}}}}
    c = client(lambda u, p: [], no_throttle, spec=spec)
    assert c.time_params(elexon.CATALOGUE["market_index"]) is None


def test_records_and_frame_parsing():
    recs = elexon.records_from_payload(
        {
            "metadata": {},
            "data": [{"startTime": "2025-01-01T00:00:00Z", "settlementDate": "2025-01-01", "x": 1}],
        }
    )
    df = elexon.to_frame(recs)
    assert str(df["startTime"].dtype).startswith("datetime64")
    assert df["settlementDate"].iloc[0] == "2025-01-01"
    assert elexon.records_from_payload([{"a": 1}]) == [{"a": 1}]


def test_fetch_window_formats_date_only_params(no_throttle):
    c = client(lambda u, p: {"data": [{"v": 1}]}, no_throttle)
    c.fetch_window(elexon.CATALOGUE["fuel_half_hourly"], T0, T0 + timedelta(days=2))
    _, params = c.session.calls[0]
    assert params["settlementDateFrom"] == "2025-01-01"
    assert params["settlementDateTo"] == "2025-01-03"
    assert params["format"] == "json"


def test_fetch_window_bisects_on_400(no_throttle):
    def handler(url, params):
        lo = datetime.fromisoformat(params["from"].replace("Z", "+00:00"))
        hi = datetime.fromisoformat(params["to"].replace("Z", "+00:00"))
        if hi - lo > timedelta(days=1):
            return {"error": "range"}, 400
        return {"data": [{"startTime": params["from"]}]}

    c = client(handler, no_throttle)
    df = c.fetch_window(elexon.CATALOGUE["market_index"], T0, T0 + timedelta(days=4))
    assert len(df) == 4  # 4 one-day windows after two bisections


def test_per_day_endpoint_uses_path(no_throttle):
    c = client(lambda u, p: {"data": [{"netImbalanceVolume": 1.0}]}, no_throttle)
    c.fetch_window(elexon.CATALOGUE["system_prices"], T0, T0 + timedelta(days=1))
    url, _ = c.session.calls[0]
    assert url.endswith("/balancing/settlement/system-prices/2025-01-01")


def test_fetch_is_resumable_and_dry_run(tmp_path, no_throttle):
    store = Store(tmp_path)
    c = client(lambda u, p: {"data": [{"startTime": "2025-01-01T00:00:00Z", "v": 1}]}, no_throttle)
    ep = elexon.CATALOGUE["market_index"]
    end = T0 + timedelta(days=14)
    assert c.fetch(ep, T0, end, store, dry_run=True) == 2 and not c.session.calls
    c.fetch(ep, T0, end, store)
    assert len(c.session.calls) == 2
    c.fetch(ep, T0, end, store)  # cached
    assert len(c.session.calls) == 2
    out = store.consolidate(elexon.SOURCE, ep.key)
    assert len(pd.read_parquet(out)) == 1  # identical rows de-duplicated


def test_verify_catalogue_flags_missing_paths(no_throttle):
    rows = {r["key"]: r for r in elexon.verify_catalogue(client(lambda u, p: [], no_throttle))}
    assert rows["indicated_imbalance"]["path_found"]
    assert not rows["wind_forecast"]["path_found"]
