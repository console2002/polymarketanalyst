"""Canonical market profile definitions for Polymarket market-window behavior.

This module is the extension point for introducing new market families
(e.g., 5-minute or non-BTC profiles) while keeping all profile-specific
configuration in one place.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class MarketProfile:
    """Configuration for one market profile.

    Keep all profile metadata here so callers can compute market windows and
    slug patterns without hard-coding market-specific constants.
    """

    key: str
    window_minutes: int
    slug_prefix: str
    data_subdir: str


_MARKET_PROFILES = {
    "btc_15m": MarketProfile(
        key="btc_15m",
        window_minutes=15,
        slug_prefix="btc-updown-15m",
        data_subdir="15min",
    ),
    "btc_5m": MarketProfile(
        key="btc_5m",
        window_minutes=5,
        slug_prefix="btc-updown-5m",
        data_subdir="5min",
    ),
}


def default_market_profile_key() -> str:
    """Return the compatibility default profile key.

    The default remains ``btc_15m`` so existing behavior is unchanged unless a
    caller explicitly selects a different profile.
    """

    return "btc_15m"


def get_market_profile(key: str) -> MarketProfile:
    """Return a market profile by key with validation.

    Args:
        key: Profile key such as ``btc_15m``.

    Raises:
        ValueError: If the key is missing, not a string, or unknown.
    """

    if not isinstance(key, str) or not key.strip():
        raise ValueError("Market profile key must be a non-empty string.")

    normalized = key.strip()
    profile = _MARKET_PROFILES.get(normalized)
    if profile is None:
        valid_keys = ", ".join(sorted(_MARKET_PROFILES.keys()))
        raise ValueError(
            f"Unknown market profile key '{normalized}'. Valid keys: {valid_keys}."
        )

    return profile
