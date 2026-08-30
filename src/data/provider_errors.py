"""
Typed errors for market-data providers.

These exist so API routes can map a provider failure onto an honest HTTP status
instead of collapsing everything into 502. A premium-only endpoint on a free key
is not an upstream failure, and an exhausted daily quota is not either — both are
deterministic and retrying them immediately is pointless.

All types subclass RuntimeError so existing `except RuntimeError` handlers and
callers written before this module keep working unchanged.
"""

from typing import Optional


class ProviderError(RuntimeError):
    """A market-data provider refused or could not serve the request."""


class PremiumEndpointError(ProviderError):
    """
    The endpoint requires a paid plan and is unavailable on the configured key.

    Deterministic: the same request will fail the same way until the plan
    changes, so callers should surface it rather than retry.
    """

    def __init__(self, message: str, endpoint: Optional[str] = None):
        super().__init__(message)
        self.endpoint = endpoint


class QuotaExceededError(ProviderError):
    """
    The provider is rate limiting: either the daily quota is spent or the
    per-second burst limit was tripped. Both map to HTTP 429; the message
    carries which one, since only the burst case is worth retrying shortly.
    """


# Alpha Vantage returns premium blocks AND rate limiting under one "Information"
# key with no machine-readable discriminator, so the wording is the only signal.
# Every one of these messages advertises the premium plans, which makes loose
# matching on the word "premium" useless. Observed free-tier messages:
#
#   premium  "Thank you for using Alpha Vantage! This is a premium endpoint. You
#             may subscribe to any of the premium plans at ... to instantly
#             unlock all premium endpoints"
#   burst    "Thank you for using Alpha Vantage! Please consider spreading out
#             your free API requests more sparingly (1 request per second). You
#             may subscribe to ... raise the per-second burst limit, and
#             instantly unlock all premium endpoints"
#   daily    "... our standard API rate limit is 25 requests per day. Please
#             subscribe to any of the premium plans at ..."
#
# The burst message ends in "unlock all premium endpoints", so the premium marker
# must be the full predicate "is a premium endpoint" — matching the bare noun
# misfiles every throttle response as a premium block.
_PREMIUM_MARKERS = ("is a premium endpoint",)
_QUOTA_MARKERS = (
    "rate limit",
    "requests per day",
    "request per second",
    "requests per second",
    "per-second burst",
    "spreading out",
    "higher api call volume",
)


def classify_alpha_vantage_message(
    message: str, endpoint: Optional[str] = None
) -> ProviderError:
    """
    Turn an Alpha Vantage "Information"/"Note" message into a typed error.

    Falls back to a plain ProviderError when the wording matches neither known
    case, so an unrecognised message is reported as an upstream problem rather
    than being silently mislabelled as premium or quota.
    """
    text = (message or "").lower()
    prefix = f"Alpha Vantage {endpoint}: " if endpoint else "Alpha Vantage: "

    if any(marker in text for marker in _PREMIUM_MARKERS):
        return PremiumEndpointError(f"{prefix}{message}", endpoint=endpoint)
    if any(marker in text for marker in _QUOTA_MARKERS):
        return QuotaExceededError(f"{prefix}{message}")
    return ProviderError(f"{prefix}{message}")
