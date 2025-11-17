"""
Debugging Requests
==================

Utilities for inspecting and working with HTTP response headers that
OpenAI returns for each API request.

From the docs, these headers include (non-exhaustive):

    # API meta information
    - openai-organization
    - openai-processing-ms
    - openai-version
    - x-request-id

    # Rate limiting information
    - x-ratelimit-limit-requests
    - x-ratelimit-limit-tokens
    - x-ratelimit-remaining-requests
    - x-ratelimit-remaining-tokens
    - x-ratelimit-reset-requests
    - x-ratelimit-reset-tokens

Merlin exposes:

- RateLimitInfo: structured view of rate limit headers
- RequestMeta: Structured metadata for a single API call
- extract_request_meta(): parse headers from a response
- make_client_request_id(): helper to generate X-Client-Request-Id values
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional
import uuid

@dataclass(frozen=True)
class RateLimitInfo:
    """Structured representation of rate limiting headers for a request."""

    limit_requests: Optional[int] = None
    limit_tokens: Optional[int] = None
    remaining_requests: Optional[int] = None
    remaining_tokens: Optional[int] = None
    reset_requests: Optional[float] = None
    reset_tokens: Optional[float] = None

@dataclass(frozen=True)
class RequestMeta:
    """
    Metadata extracted from an API response's HTTP headers.

    Attributes:
        request_id:
            The server-generated x-request-id header.
        organization:
            The openai-organization header.
        processing_ms:
            Time taken processing the API request, in milliseconds.
        api_version:
            REST API version used for this request (openai-version).
        rate_limit:
            Parsed RateLimitInfo, if present.
    """

    request_id: Optional[str]
    organization: Optional[str]
    processing_ms: Optional[int]
    api_version: Optional[str]
    rate_limit: RateLimitInfo

def _parse_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

def _parse_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def extract_request_meta(headers: Mapping[str, str]) -> RequestMeta:
    """
    Extract RequestMeta from response headers.

    This function is intentionally tolerant of missing or malformed
    headers so that it can be safely used in all environments.
    """
    # Normalize header keys to lower case for robustness.
    lower = {k.lower(): v for k, v in headers.items()}

    rate_limit = RateLimitInfo(
        limit_requests=_parse_int(lower.get("x-ratelimit-limit-requests")),
        limit_tokens=_parse_int(lower.get("x-ratelimit-limit-tokens")),
        remaining_requests=_parse_int(lower.get("x-ratelimit-remaining-requests")),
        remaining_tokens=_parse_int(lower.get("x-ratelimit-remaining-tokens")),
        reset_requests=_parse_float(lower.get("x-ratelimit-reset-requests")),
        reset_tokens=_parse_float(lower.get("x-ratelimit-reset-tokens")),
    )

    processing_ms = _parse_int(lower.get("openai-processing-ms"))

    return RequestMeta(
        request_id=lower.get("x-request-id"),
        organization=lower.get("openai-organization"),
        processing_ms=processing_ms,
        api_version=lower.get("openai-version"),
        rate_limit=rate_limit,
    )

def make_client_request_id() -> str:
    """
    Generate a value suitable for the X-Client-Request-Id header.

    OpenAI requires:
        - ASCII only
        - <= 512 characters
        - Ideally unique per request

    This helper uses a UUID4 string, which satisfies those constraints.
    """
    return str(uuid.uuid4())

__all__ = [
    "RateLimitInfo",
    "RequestMeta",
    "extract_request_meta",
    "make_client_request_id",
]
