from datetime import date, datetime, timezone

import pandas as pd

from niv.config import default_window
from niv.sources import weather
from niv.storage import Store
from tests.conftest import FakeSession


def test_default_window_is_365_complete_days():
    s, e = default_window(365, date(2026, 10, 13))
    assert e == datetime(2026, 10, 13, tzinfo=timezone.utc)
    assert (e - s).days == 365


def test_weather_params_previous_runs():
    p = weather.build_params(
        54.0,
        -2.0,
        date(2025, 1, 1),
        date(2025, 1, 31),
        hourly=["wind_speed_100m"],
        models=["ecmwf_ifs025"],
        previous_days=(1, 2),
    )
    assert p["hourly"] == "wind_speed_100m_previous_day1,wind_speed_100m_previous_day2"
    assert p["timezone"] == "UTC"


def test_month_windows_cover_range():
    w = weather.month_windows(date(2025, 1, 1), date(2025, 12, 31))
    assert w[0][0] == date(2025, 1, 1) and w[-1][1] == date(2025, 12, 31)
    assert len(w) == 12


def test_fetch_weather_writes_chunks(tmp_path):
    payload = {
        "hourly": {
            "time": ["2025-01-01T00:00", "2025-01-01T01:00"],
            "wind_speed_100m_ecmwf_ifs025": [5.0, 6.0],
        }
    }
    sess = FakeSession(lambda u, p: payload)
    store = Store(tmp_path)
    n = weather.fetch_weather(
        store, date(2025, 1, 1), date(2025, 1, 10), points={"a": (1.0, 2.0)}, session=sess
    )
    assert n == 1 and len(sess.calls) == 1
    df = store.read_raw(weather.SOURCE, "weather_historical_forecast")
    assert list(df["point"].unique()) == ["a"] and len(df) == 2


def test_store_empty_marker(tmp_path):
    s = Store(tmp_path)
    s.write_chunk(pd.DataFrame(), "x", "y", "p")
    assert s.has_chunk("x", "y", "p") and s.consolidate("x", "y") is None


def test_store_summary(tmp_path):
    s = Store(tmp_path)
    df = pd.DataFrame(
        {"startTime": pd.to_datetime(["2025-01-01", "2025-01-02"], utc=True), "v": [1, 2]}
    )
    s.write_chunk(df, "x", "y", "p")
    s.consolidate("x", "y")
    summ = s.summary()
    assert summ.loc[0, "rows"] == 2 and summ.loc[0, "time_column"] == "startTime"
