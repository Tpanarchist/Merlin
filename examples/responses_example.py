"""
Example showing how to wire ResponsesMixin with the httpx adapter.

This is a small, non-running example (placeholder base_url and API key).
Wraps calls in try/except so importing httpx is optional during static analysis.
"""

from typing import Any

from merlin.api.responses.responses import ResponsesMixin
from merlin.http_client import HttpxMerlinHTTPClient


class MerlinClient(ResponsesMixin):
    def __init__(self, http_client: Any) -> None:
        # ResponsesMixin expects `self._http` to be a MerlinHTTPClient.
        self._http = http_client


if __name__ == "__main__":
    try:
        client = HttpxMerlinHTTPClient(
            base_url="https://api.example.com",
            headers={"Authorization": "Bearer YOUR_API_KEY"},
            timeout=10.0,
        )
    except ImportError as e:
        print("httpx not installed; install with: pip install httpx")
    else:
        api = MerlinClient(client)

        try:
            # Example call (will perform a network request if base_url is valid).
            resp = api.create_response(model="gpt-example", input="Hello")
            print("Response id:", resp.id)
            print("Status:", resp.status)
        except Exception as exc:  # pragma: no cover - runtime example
            print("Request failed:", exc)
        finally:
            client.close()
