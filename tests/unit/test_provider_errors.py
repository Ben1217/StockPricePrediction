"""
Alpha Vantage reports premium blocks and rate limiting under one "Information"
key, so the only discriminator is the message wording. These are the verbatim
messages a free key returns; they pin the classification because a mismatch
turns a retryable throttle into a permanent-looking 501 (or vice versa).
"""

import pytest

from src.data.provider_errors import (
    PremiumEndpointError,
    ProviderError,
    QuotaExceededError,
    classify_alpha_vantage_message,
)

PREMIUM_MESSAGE = (
    "Thank you for using Alpha Vantage! This is a premium endpoint. You may "
    "subscribe to any of the premium plans at https://www.alphavantage.co/premium/ "
    "to instantly unlock all premium endpoints"
)

# Returned verbatim by the live API when free-tier calls are issued back to back.
BURST_MESSAGE = (
    "Thank you for using Alpha Vantage! Please consider spreading out your free "
    "API requests more sparingly (1 request per second). You may subscribe to any "
    "of the premium plans at https://www.alphavantage.co/premium/ to lift the free "
    "key rate limit (25 requests per day), raise the per-second burst limit, and "
    "instantly unlock all premium endpoints"
)

DAILY_QUOTA_MESSAGE = (
    "We have detected your API key as demo and our standard API rate limit is 25 "
    "requests per day. Please subscribe to any of the premium plans at "
    "https://www.alphavantage.co/premium/ to instantly remove all daily rate limits."
)


def test_premium_block_is_premium_endpoint_error():
    err = classify_alpha_vantage_message(PREMIUM_MESSAGE, endpoint="TIME_SERIES_INTRADAY")
    assert isinstance(err, PremiumEndpointError)
    assert err.endpoint == "TIME_SERIES_INTRADAY"


@pytest.mark.parametrize(
    "message", [BURST_MESSAGE, DAILY_QUOTA_MESSAGE], ids=["burst", "daily_quota"]
)
def test_rate_limiting_is_quota_error_not_premium(message):
    """
    Both throttle messages end in "unlock all premium endpoints". Matching the
    bare noun "premium endpoint" would file them as premium blocks and answer a
    transient throttle with 501 Not Implemented.
    """
    err = classify_alpha_vantage_message(message, endpoint="TIME_SERIES_DAILY")
    assert isinstance(err, QuotaExceededError)
    assert not isinstance(err, PremiumEndpointError)


def test_unrecognised_message_stays_generic():
    err = classify_alpha_vantage_message("Something we have never seen.", endpoint="GLOBAL_QUOTE")
    assert type(err) is ProviderError


def test_message_and_endpoint_are_preserved():
    err = classify_alpha_vantage_message(PREMIUM_MESSAGE, endpoint="TIME_SERIES_INTRADAY")
    assert "TIME_SERIES_INTRADAY" in str(err)
    assert "This is a premium endpoint" in str(err)


def test_all_provider_errors_remain_runtime_errors():
    """Callers written before this module still catch these with RuntimeError."""
    for err in (
        ProviderError("x"),
        PremiumEndpointError("x"),
        QuotaExceededError("x"),
    ):
        assert isinstance(err, RuntimeError)
