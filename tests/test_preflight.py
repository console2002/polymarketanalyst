import pytest

import preflight


class DummyResponse:
    def __init__(self, status_code=200, payload=None, text=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.content = b"{}" if payload is not None else b""
        self.text = text or ""

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


def test_validate_clob_token_ids_encodes_as_array(monkeypatch):
    def fake_get(url, params=None, timeout=None):
        assert url == preflight.GAMMA_MARKETS_URL
        assert params == {"clob_token_ids": ["1", "2"]}
        return DummyResponse(200, [{"id": 1}])

    monkeypatch.setattr(preflight.requests, "get", fake_get)
    validator, count = preflight.validate_clob_token_ids(["1", "2"])
    assert validator == "gamma"
    assert count == 1


def test_validate_clob_token_ids_gamma_success(monkeypatch):
    def fake_get(url, params=None, timeout=None):
        assert url == preflight.GAMMA_MARKETS_URL
        return DummyResponse(200, [{"slug": "test"}])

    monkeypatch.setattr(preflight.requests, "get", fake_get)
    validator, count = preflight.validate_clob_token_ids(["1", "2"], slug="test")
    assert validator == "gamma"
    assert count == 1


def test_validate_clob_token_ids_fallback_to_clob(monkeypatch):
    def fake_get(url, params=None, timeout=None):
        if url == preflight.GAMMA_MARKETS_URL:
            return DummyResponse(422, {"error": "bad"}, text="bad")
        assert url == preflight.CLOB_BOOK_URL
        assert params["token_id"] in {"1", "2"}
        return DummyResponse(200, {"ok": True})

    monkeypatch.setattr(preflight.requests, "get", fake_get)
    validator, count = preflight.validate_clob_token_ids(["1", "2"])
    assert validator == "clob_book"
    assert count is None


def test_validate_clob_token_ids_fallback_fails(monkeypatch):
    def fake_get(url, params=None, timeout=None):
        if url == preflight.GAMMA_MARKETS_URL:
            return DummyResponse(422, {"error": "bad"}, text="bad")
        assert url == preflight.CLOB_BOOK_URL
        if params["token_id"] == "1":
            return DummyResponse(200, {"ok": True})
        return DummyResponse(404, {"error": "missing"}, text="missing")

    monkeypatch.setattr(preflight.requests, "get", fake_get)
    with pytest.raises(preflight.PreflightError, match="clob_book_validate"):
        preflight.validate_clob_token_ids(["1", "2"])
