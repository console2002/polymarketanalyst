import datetime

import fetch_current_polymarket as fcp


def test_resolve_market_by_slug_parses_clob_token_ids(monkeypatch):
    def fake_get_gamma(slug):
        return {"clobTokenIds": "[\"1\", \"2\"]"}, None, 200

    monkeypatch.setattr(fcp, "_get_gamma_market_by_slug", fake_get_gamma)
    result, err = fcp.resolve_market_by_slug("test-slug")

    assert err is None
    assert result["clob_token_ids"] == ["1", "2"]


def test_resolve_current_market_tries_next_window(monkeypatch):
    start_time = datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc)

    def fake_get_current_market_urls():
        return {
            "polymarket": "https://example.com/unused",
            "target_time_utc": start_time,
            "expiration_time_utc": start_time + datetime.timedelta(minutes=15),
        }

    def fake_generate_market_url(target_time):
        return f"https://example.com/btc-updown-15m-{int(target_time.timestamp())}"

    calls = []

    def fake_resolve_market_by_slug(slug):
        calls.append(slug)
        if len(calls) == 1:
            return None, f"gamma_slug_not_found slug={slug}"
        return {
            "slug": slug,
            "clob_token_ids": ["1", "2"],
            "outcomes": [],
            "start_time": None,
            "end_time": None,
            "polymarket_time_utc": None,
        }, None

    monkeypatch.setattr(fcp, "get_current_market_urls", fake_get_current_market_urls)
    monkeypatch.setattr(fcp, "generate_market_url", fake_generate_market_url)
    monkeypatch.setattr(fcp, "resolve_market_by_slug", fake_resolve_market_by_slug)

    result, err = fcp.resolve_current_market()

    assert err is None
    assert len(calls) == 2
    assert result["slug"] == calls[1]


def test_resolve_market_by_slug_malformed_clob_token_ids(monkeypatch):
    def fake_get_gamma(slug):
        return {"clobTokenIds": "not-json"}, None, 200

    monkeypatch.setattr(fcp, "_get_gamma_market_by_slug", fake_get_gamma)
    result, err = fcp.resolve_market_by_slug("test-slug")

    assert result is None
    assert "gamma_slug_invalid_clob_token_ids" in err
