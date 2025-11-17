"""
Embeddings API
==============

High-level client for the `/v1/embeddings` endpoint.

Docs section: "Embeddings"

- POST /v1/embeddings → create embedding vectors for one or more inputs.

We wrap the raw JSON in a small, typed layer:

- EmbeddingUsage       → prompt / total token counts
- EmbeddingObject      → one embedding vector (plus index)
- EmbeddingsResponse   → top-level list response
- EmbeddingsMixin      → client mixin with `create_embeddings()` and
                         a convenience `embed_one()` helper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from merlin.http_client import MerlinHTTPClient


JSON = Dict[str, Any]


# ───────────────────────────────────────────────────────────────
# Data models
# ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EmbeddingUsage:
    """
    Usage statistics for an embeddings request.

    Docs shape:

        "usage": {
          "prompt_tokens": 8,
          "total_tokens": 8
        }
    """

    prompt_tokens: Optional[int]
    total_tokens: Optional[int]
    raw: JSON = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EmbeddingUsage":
        if not isinstance(data, Mapping):
            return cls(prompt_tokens=None, total_tokens=None, raw={})

        def _opt_int(v: Any) -> Optional[int]:
            try:
                return int(v) if v is not None else None
            except (TypeError, ValueError):
                return None

        return cls(
            prompt_tokens=_opt_int(data.get("prompt_tokens")),
            total_tokens=_opt_int(data.get("total_tokens")),
            raw=dict(data),
        )


@dataclass(frozen=True)
class EmbeddingObject:
    """
    Single embedding object from the embeddings API.

    Docs shape:

        {
          "object": "embedding",
          "embedding": [ ... floats or base64-encoded values ... ],
          "index": 0
        }

    For `encoding_format="float"`, `embedding` is a list of floats.
    For `encoding_format="base64"`, elements may be base64-encoded
    strings. We keep the vector as `List[Union[float, str]]` and
    preserve the original JSON in `raw`.
    """

    index: int
    embedding: List[Union[float, str]]
    raw: JSON = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EmbeddingObject":
        if not isinstance(data, Mapping):
            return cls(index=0, embedding=[], raw={})

        # We don't over-normalize: floats stay floats, strings stay
        # strings. Callers can coerce if they know the encoding_format.
        vec = data.get("embedding") or []
        if not isinstance(vec, Sequence):
            vec = []

        embedding: List[Union[float, str]] = []
        for v in vec:
            # Common cases: float or base64 string. Just keep as-is,
            # but try to coerce obvious numeric types to float.
            if isinstance(v, (int, float)):
                embedding.append(float(v))
            else:
                embedding.append(v)

        try:
            idx = int(data.get("index", 0))
        except (TypeError, ValueError):
            idx = 0

        return cls(
            index=idx,
            embedding=embedding,
            raw=dict(data),
        )


@dataclass(frozen=True)
class EmbeddingsResponse:
    """
    Top-level response wrapper for `/v1/embeddings`.

    Docs shape:

        {
          "object": "list",
          "data": [ {embedding objects...} ],
          "model": "text-embedding-ada-002",
          "usage": {
            "prompt_tokens": 8,
            "total_tokens": 8
          }
        }
    """

    model: Optional[str]
    data: List[EmbeddingObject]
    usage: Optional[EmbeddingUsage]
    raw: JSON

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EmbeddingsResponse":
        model = data.get("model")
        items_raw = data.get("data") or []
        items: List[EmbeddingObject] = [
            EmbeddingObject.from_dict(item)
            for item in items_raw
            if isinstance(item, Mapping)
        ]

        usage_raw = data.get("usage")
        usage = EmbeddingUsage.from_dict(usage_raw) if isinstance(usage_raw, Mapping) else None

        return cls(
            model=model,
            data=items,
            usage=usage,
            raw=dict(data),
        )

    @property
    def embeddings(self) -> List[List[Union[float, str]]]:
        """
        Convenience accessor: list of raw vectors for each embedding.
        """
        return [e.embedding for e in self.data]


# ───────────────────────────────────────────────────────────────
# Client mixin
# ───────────────────────────────────────────────────────────────


class EmbeddingsMixin:
    """
    Mixin providing convenience methods for the Embeddings API.

    Assumptions:
        - The consuming client defines `self._http` as a MerlinHTTPClient.

    This is intentionally thin: we mirror the REST surface and add a
    very small ergonomic helper for the single-input case.
    """

    _http: MerlinHTTPClient  # for type checkers

    def create_embeddings(
        self,
        *,
        input: Union[
            str,
            Sequence[str],
            Sequence[int],
            Sequence[Sequence[int]],
        ],
        model: str,
        dimensions: Optional[int] = None,
        encoding_format: Optional[str] = None,
        user: Optional[str] = None,
        **extra: Any,
    ) -> EmbeddingsResponse:
        """
        Create embeddings for one or more inputs.

        POST /v1/embeddings

        Args:
            input:
                Text or tokens to embed. Supports:
                  - str
                  - list[str]
                  - list[int] (tokens)
                  - list[list[int]] (multiple token sequences)
                Subject to model token limits and global 300k token cap.
            model:
                ID of the embeddings model, e.g. "text-embedding-3-small".
            dimensions:
                Optional number of output dimensions (text-embedding-3+ only).
            encoding_format:
                "float" (default) or "base64".
            user:
                Stable end-user identifier for abuse monitoring.
            extra:
                Any future / extra fields supported by the API.

        Returns:
            EmbeddingsResponse
        """
        payload: JSON = {
            "input": input,
            "model": model,
        }

        if dimensions is not None:
            payload["dimensions"] = dimensions
        if encoding_format is not None:
            payload["encoding_format"] = encoding_format
        if user is not None:
            payload["user"] = user

        payload.update(extra)

        resp = self._http.post(
            "/v1/embeddings",
            json=payload,
            expect_json=True,
        )
        return EmbeddingsResponse.from_dict(resp)

    # Small ergonomic helper for the "just give me a single vector" case.

    def embed_one(
        self,
        text: str,
        *,
        model: str,
        dimensions: Optional[int] = None,
        encoding_format: Optional[str] = None,
        user: Optional[str] = None,
        **extra: Any,
    ) -> List[Union[float, str]]:
        """
        Convenience wrapper around `create_embeddings` for a single string.

        Returns:
            The first embedding vector from the response, or an empty list
            if none are present.
        """
        resp = self.create_embeddings(
            input=text,
            model=model,
            dimensions=dimensions,
            encoding_format=encoding_format,
            user=user,
            **extra,
        )
        return resp.embeddings[0] if resp.embeddings else []


__all__ = [
    "EmbeddingUsage",
    "EmbeddingObject",
    "EmbeddingsResponse",
    "EmbeddingsMixin",
]
