import pytest

import preflight


class DummyResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.content = b"{}" if payload is not None else b""

    def json(self):
        return self._payload


def test_check_geoblock_blocked(monkeypatch):
    def fake_get(url, timeout=None):
        assert url == preflight.GEOBLOCK_URL
        return DummyResponse(
            200,
            {
                "blocked": True,
                "ip": "1.2.3.4",
                "country": "US",
                "region": "CA",
            },
        )

    monkeypatch.setattr(preflight.requests, "get", fake_get)
    result = preflight.check_geoblock()
    assert result.blocked is True
    assert result.country == "US"


def test_check_geoblock_unblocked(monkeypatch):
    def fake_get(url, timeout=None):
        assert url == preflight.GEOBLOCK_URL
        return DummyResponse(200, {"blocked": False})

    monkeypatch.setattr(preflight.requests, "get", fake_get)
    result = preflight.check_geoblock()
    assert result.blocked is False


def test_fetch_market_by_slug_success(monkeypatch):
    payload = {"clobTokenIds": "[\"1\", \"2\"]"}

    def fake_get(url, timeout=None):
        assert url == f"{preflight.GAMMA_MARKET_SLUG_URL}/test-slug"
        return DummyResponse(200, payload)

    monkeypatch.setattr(preflight.requests, "get", fake_get)
    response = preflight.fetch_market_by_slug("test-slug")
    assert response == payload


def test_fetch_market_by_slug_not_found(monkeypatch):
    def fake_get(url, timeout=None):
        return DummyResponse(404, {})

    monkeypatch.setattr(preflight.requests, "get", fake_get)
    with pytest.raises(preflight.PreflightError, match="gamma_slug_not_found"):
        preflight.fetch_market_by_slug("missing")


def test_validate_clob_token_ids_invalid(monkeypatch):
    def fake_get(url, params=None, timeout=None):
        assert url == preflight.GAMMA_MARKETS_URL
        assert params == {"clob_token_ids": "1,2"}
        return DummyResponse(400, {})

    monkeypatch.setattr(preflight.requests, "get", fake_get)
    with pytest.raises(preflight.PreflightError, match="gamma_validate"):
        preflight.validate_clob_token_ids(["1", "2"])
