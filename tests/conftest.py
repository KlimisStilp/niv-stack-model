"""Shared test fakes: a scripted HTTP session and a miniature OpenAPI spec."""

from __future__ import annotations

import pytest

from niv.http import Throttle

SPEC = {
    "paths": {
        "/datasets/IMBALNGC": {
            "get": {
                "parameters": [
                    {"name": "publishDateTimeFrom"},
                    {"name": "publishDateTimeTo"},
                    {"name": "format"},
                ]
            }
        },
        "/datasets/FUELHH": {
            "get": {
                "parameters": [
                    {"name": "settlementDateFrom"},
                    {"name": "settlementDateTo"},
                    {"name": "format"},
                ]
            }
        },
        "/datasets/MID": {"get": {"parameters": [{"name": "from"}, {"name": "to"}]}},
        "/balancing/settlement/system-prices/{settlementDate}": {
            "get": {"parameters": [{"name": "settlementDate"}, {"name": "format"}]}
        },
    }
}


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload, self.status_code = payload, status
        self.text = str(payload)

    def json(self):
        return self._payload


class FakeSession:
    """Records calls; ``handler(url, params)`` returns payload or (payload, status)."""

    def __init__(self, handler):
        self.handler, self.calls = handler, []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        out = self.handler(url, params or {})
        return FakeResponse(*out) if isinstance(out, tuple) else FakeResponse(out)


@pytest.fixture
def no_throttle():
    return Throttle(0.0)
