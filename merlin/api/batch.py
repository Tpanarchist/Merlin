"""
Batch API
=========

Implements the full Batch API spec from the OpenAI documentation.

Endpoints
---------
POST   /v1/batches                 → create batch
GET    /v1/batches/{batch_id}      → retrieve batch
POST   /v1/batches/{batch_id}/cancel  → cancel batch
GET    /v1/batches                 → list batches

The Batch API processes up to 50,000 requests asynchronously for a 24h
completion window with discounted pricing.

This module provides:

- Dataclasses for Batch, BatchRequestInput, BatchRequestOutput, BatchUsage
- High-level BatchMixin for client access
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from merlin.http_client import MerlinHTTPClient


JSON = Dict[str, Any]


# ───────────────────────────────────────────────────────────────
# Dataclasses for all the Batch API objects
# ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BatchRequestCounts:
    total: int
    completed: int
    failed: int

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "BatchRequestCounts":
        return cls(
            total=int(d.get("total", 0)),
            completed=int(d.get("completed", 0)),
            failed=int(d.get("failed", 0)),
        )


@dataclass(frozen=True)
class BatchUsage:
    """
    Optional field — only appears starting Sept 2025.
    """
    input_tokens: Optional[int]
    input_tokens_details: Optional[JSON]
    output_tokens: Optional[int]
    output_tokens_details: Optional[JSON]
    total_tokens: Optional[int]

    @classmethod
    def from_dict(cls, d: Optional[Mapping[str, Any]]) -> "BatchUsage":
        if not isinstance(d, Mapping):
            return cls(None, None, None, None, None)
        return cls(
            input_tokens=d.get("input_tokens"),
            input_tokens_details=d.get("input_tokens_details"),
            output_tokens=d.get("output_tokens"),
            output_tokens_details=d.get("output_tokens_details"),
            total_tokens=d.get("total_tokens"),
        )


@dataclass(frozen=True)
class Batch:
    """
    Represents a full Batch object returned from any of the endpoints.
    """
    id: str
    object: str
    endpoint: str
    input_file_id: str
    completion_window: str
    status: str

    errors: Optional[JSON]

    output_file_id: Optional[str]
    error_file_id: Optional[str]

    created_at: Optional[int]
    in_progress_at: Optional[int]
    expires_at: Optional[int]
    finalizing_at: Optional[int]
    completed_at: Optional[int]
    failed_at: Optional[int]
    expired_at: Optional[int]
    cancelling_at: Optional[int]
    cancelled_at: Optional[int]

    request_counts: BatchRequestCounts
    model: Optional[str]
    usage: Optional[BatchUsage]
    metadata: Optional[JSON]

    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Batch":
        return cls(
            id=d["id"],
            object=d["object"],
            endpoint=d["endpoint"],
            input_file_id=d["input_file_id"],
            completion_window=d["completion_window"],
            status=d["status"],
            errors=d.get("errors"),
            output_file_id=d.get("output_file_id"),
            error_file_id=d.get("error_file_id"),
            created_at=d.get("created_at"),
            in_progress_at=d.get("in_progress_at"),
            expires_at=d.get("expires_at"),
            finalizing_at=d.get("finalizing_at"),
            completed_at=d.get("completed_at"),
            failed_at=d.get("failed_at"),
            expired_at=d.get("expired_at"),
            cancelling_at=d.get("cancelling_at"),
            cancelled_at=d.get("cancelled_at"),
            request_counts=BatchRequestCounts.from_dict(
                d.get("request_counts", {})
            ),
            model=d.get("model"),
            usage=BatchUsage.from_dict(d.get("usage")),
            metadata=d.get("metadata"),
            raw=dict(d),
        )


@dataclass(frozen=True)
class BatchList:
    """
    Represents a paginated list of Batches.
    """
    data: List[Batch]
    object: str
    first_id: Optional[str]
    last_id: Optional[str]
    has_more: bool
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "BatchList":
        return cls(
            data=[Batch.from_dict(x) for x in d.get("data", [])],
            object=d.get("object", "list"),
            first_id=d.get("first_id"),
            last_id=d.get("last_id"),
            has_more=bool(d.get("has_more", False)),
            raw=dict(d),
        )


# ───────────────────────────────────────────────────────────────
# Mixin for Batch API
# ───────────────────────────────────────────────────────────────


class BatchMixin:
    """
    High-level wrapper for:

        POST   /v1/batches
        GET    /v1/batches/{id}
        POST   /v1/batches/{id}/cancel
        GET    /v1/batches

    Exposed as:

        client.batches.create(...)
        client.batches.retrieve(id)
        client.batches.cancel(id)
        client.batches.list(...)

    """

    _http: MerlinHTTPClient

    # ── CREATE BATCH ─────────────────────────────────────────────

    def create_batch(
        self,
        *,
        input_file_id: str,
        endpoint: str,
        completion_window: str = "24h",
        metadata: Optional[JSON] = None,
        output_expires_after: Optional[JSON] = None,
    ) -> Batch:
        """
        Create a batch.

        Required:
            input_file_id     → ID of uploaded JSONL file with requests
            endpoint          → e.g. "/v1/chat/completions"
            completion_window → currently only "24h"

        Optional:
            metadata
            output_expires_after
        """
        payload: JSON = {
            "input_file_id": input_file_id,
            "endpoint": endpoint,
            "completion_window": completion_window,
        }
        if metadata is not None:
            payload["metadata"] = metadata
        if output_expires_after is not None:
            payload["output_expires_after"] = output_expires_after

        resp = self._http.post(
            "/v1/batches",
            json=payload
        )
        return Batch.from_dict(resp)

    # ── RETRIEVE ────────────────────────────────────────────────

    def retrieve_batch(self, batch_id: str) -> Batch:
        resp = self._http.get(
            f"/v1/batches/{batch_id}"
        )
        return Batch.from_dict(resp)

    # ── CANCEL ───────────────────────────────────────────────────

    def cancel_batch(self, batch_id: str) -> Batch:
        resp = self._http.post(
            f"/v1/batches/{batch_id}/cancel"
        )
        return Batch.from_dict(resp)

    # ── LIST ─────────────────────────────────────────────────────

    def list_batches(
        self,
        *,
        after: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> BatchList:
        params: JSON = {}
        if after:
            params["after"] = after
        if limit:
            params["limit"] = limit

        resp = self._http.get(
            "/v1/batches",
            params=params
        )
        return BatchList.from_dict(resp)


__all__ = [
    "Batch",
    "BatchList",
    "BatchUsage",
    "BatchRequestCounts",
    "BatchMixin",
]
