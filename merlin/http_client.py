"""
HTTP client implementations for Merlin.

This module exposes a minimal typed interface `MerlinHTTPClient` used by the
API mixins and a concrete httpx-based adapter `HttpxMerlinHTTPClient`.

Notes:
- `HttpxMerlinHTTPClient` is a synchronous adapter using `httpx.Client`.
- httpx is an optional dependency; importing the adapter will raise an
  instructive ImportError if httpx is not installed.
- The adapter returns parsed JSON (dict/list) from requests and raises on
  non-2xx responses.
"""

from typing import Any, Mapping, Optional

class MerlinHTTPClient:
    """
    Minimal HTTP client interface used by API mixins.

    Implementations should return parsed JSON-compatible Python objects
    (usually dict/list) from these methods.
    """

    def get(self, path: str, params: Optional[Mapping[str, Any]] = None) -> Any:
        raise NotImplementedError("MerlinHTTPClient.get must be implemented by the runtime client")

    def post(
        self,
        path: str,
        *,
        json: Optional[Any] = None,
        params: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        raise NotImplementedError("MerlinHTTPClient.post must be implemented by the runtime client")

    def delete(self, path: str, *, params: Optional[Mapping[str, Any]] = None) -> Any:
        raise NotImplementedError("MerlinHTTPClient.delete must be implemented by the runtime client")


# Concrete httpx adapter ----------------------------------------------------

try:
    import httpx  # type: ignore
except Exception as exc:  # pragma: no cover - import guard
    httpx = None  # type: ignore

class HttpxMerlinHTTPClient(MerlinHTTPClient):
    """
    Synchronous httpx-based implementation of MerlinHTTPClient.

    Example:
        client = HttpxMerlinHTTPClient(base_url="https://api.example.com", headers={"Authorization": "Bearer ..."})
        data = client.post("/v1/responses", json={"model": "x", "input": "hi"})
        client.close()

    The adapter will call `response.raise_for_status()` for non-2xx responses
    and then return `response.json()`.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        headers: Optional[Mapping[str, str]] = None,
        timeout: Optional[float] = 10.0,
    ) -> None:
        if httpx is None:
            raise ImportError(
                "httpx is required for HttpxMerlinHTTPClient. Install it with `pip install httpx`."
            )
        # httpx.Client type signatures are strict about URL types in some stubs;
        # silence the arg-type error from strict type checkers here since we
        # accept Optional[str] for convenience.
        self._client = httpx.Client(base_url=base_url, headers=headers, timeout=timeout)  # type: ignore[arg-type]

    def get(self, path: str, params: Optional[Mapping[str, Any]] = None) -> Any:
        resp = self._client.get(path, params=params)
        resp.raise_for_status()
        return resp.json()

    def post(
        self,
        path: str,
        *,
        json: Optional[Any] = None,
        params: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        resp = self._client.post(path, json=json, params=params)
        resp.raise_for_status()
        return resp.json()

    def delete(self, path: str, *, params: Optional[Mapping[str, Any]] = None) -> Any:
        resp = self._client.delete(path, params=params)
        resp.raise_for_status()
        return resp.json()

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "HttpxMerlinHTTPClient":
        return self

    def __exit__(self, exc_type: Optional[type], exc: Optional[BaseException], tb: Optional[Any]) -> None:
        self.close()


__all__ = ["MerlinHTTPClient", "HttpxMerlinHTTPClient"]
