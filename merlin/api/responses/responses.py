"""
Responses API
=============

High-level client for the `/v1/responses` endpoints.

This module wraps the "Responses" section of the OpenAI API:

- POST   /v1/responses                     → create a model response
- GET    /v1/responses/{response_id}       → retrieve a model response
- DELETE /v1/responses/{response_id}       → delete a model response
- POST   /v1/responses/{response_id}/cancel
                                          → cancel a background response
- GET    /v1/responses/{response_id}/input_items
                                          → list input items for a response
- POST   /v1/responses/input_tokens        → get input token counts

Design notes
------------

- The full schema of the Response object is large and evolving.
  Merlin keeps a *stable, minimal* typed view while also exposing
  the raw JSON via the `raw` field for advanced use.
- For creation, we provide a convenience method that takes the most
  common parameters explicitly (`model`, `input`, etc.) plus an
  open-ended `**kwargs` for advanced options, matching the API docs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from merlin.http_client import MerlinHTTPClient

JSON = Dict[str, Any]
InputType = Union[str, Sequence[Any]]

# ───────────────────────────────────────────────────────────────
# Data models
# ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ResponseUsage:
    """Token usage details attached to a Response."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    details: JSON = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ResponseUsage":
        return cls(
            input_tokens=int(data.get("input_tokens", 0)),
            output_tokens=int(data.get("output_tokens", 0)),
            total_tokens=int(data.get("total_tokens", 0)),
            details=dict(data),
        )

@dataclass(frozen=True)
class ResponseObject:
    """
    Minimal typed view of a Response object.

    The full JSON is also available via `raw` for advanced use.
    """

    id: str
    status: str
    model: str
    created_at: int
    output: List[JSON]
    usage: Optional[ResponseUsage]
    raw: JSON

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ResponseObject":
        usage_raw = data.get("usage")
        usage = ResponseUsage.from_dict(usage_raw) if isinstance(usage_raw, Mapping) else None

        return cls(
            id=str(data.get("id")),
            status=str(data.get("status")),
            model=str(data.get("model")),
            created_at=int(data.get("created_at", 0)),
            output=list(data.get("output", [])),
            usage=usage,
            raw=dict(data),
        )

@dataclass(frozen=True)
class ResponseDeletionResult:
    """Result of deleting a Response."""

    id: str
    deleted: bool
    raw: JSON

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ResponseDeletionResult":
        return cls(
            id=str(data.get("id")),
            deleted=bool(data.get("deleted", False)),
            raw=dict(data),
        )

@dataclass(frozen=True)
class InputItem:
    """
    A single input item used to generate a Response.

    The schema can vary (messages, images, files, etc.), so most of the
    structure is left as raw JSON.
    """

    id: str
    type: str
    raw: JSON

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "InputItem":
        return cls(
            id=str(data.get("id")),
            type=str(data.get("type")),
            raw=dict(data),
        )

@dataclass(frozen=True)
class InputItemList:
    """A paginated list of input items for a Response."""

    data: List[InputItem]
    first_id: Optional[str]
    last_id: Optional[str]
    has_more: bool
    raw: JSON

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "InputItemList":
        items_raw = data.get("data", [])
        items = [InputItem.from_dict(item) for item in items_raw]

        return cls(
            data=items,
            first_id=data.get("first_id"),
            last_id=data.get("last_id"),
            has_more=bool(data.get("has_more", False)),
            raw=dict(data),
        )

@dataclass(frozen=True)
class InputTokenCount:
    """Result of POST /v1/responses/input_tokens."""

    input_tokens: int
    raw: JSON

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "InputTokenCount":
        return cls(
            input_tokens=int(data.get("input_tokens", 0)),
            raw=dict(data),
        )

# ───────────────────────────────────────────────────────────────
# Client mixin
# ───────────────────────────────────────────────────────────────

class ResponsesMixin:
    """
    Mixin providing convenience methods for the Responses API.

    Assumptions:
        - The consuming client defines `self._http` as a MerlinHTTPClient.
    """

    _http: MerlinHTTPClient  # for type checkers

    # ---- Core endpoints -------------------------------------------------

    def create_response(
        self,
        *,
        model: Optional[str] = None,
        input: Optional[InputType] = None,
        stream: Optional[bool] = None,
        background: Optional[bool] = None,
        conversation: Optional[Union[str, JSON]] = None,
        previous_response_id: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        tools: Optional[Sequence[JSON]] = None,
        tool_choice: Optional[Union[str, JSON]] = None,
        metadata: Optional[JSON] = None,
        **extra: Any,
    ) -> ResponseObject:
        """
        Create a model response.

        This is a high-level wrapper around POST /v1/responses. It exposes
        the most common parameters explicitly and accepts any additional
        request body fields via **extra to maintain full coverage.

        For advanced options like `reasoning`, `text`, `service_tier`,
        `stream_options`, or `include`, pass them through **extra.
        """
        payload: JSON = {}

        if model is not None:
            payload["model"] = model
        if input is not None:
            payload["input"] = input
        if stream is not None:
            payload["stream"] = stream
        if background is not None:
            payload["background"] = background
        if conversation is not None:
            payload["conversation"] = conversation
        if previous_response_id is not None:
            payload["previous_response_id"] = previous_response_id
        if max_output_tokens is not None:
            payload["max_output_tokens"] = max_output_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if tools is not None:
            payload["tools"] = list(tools)
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if metadata is not None:
            payload["metadata"] = metadata

        payload.update(extra)

        data = self._http.post("/v1/responses", json=payload)
        return ResponseObject.from_dict(data)

    def get_response(
        self,
        response_id: str,
        *,
        include: Optional[Sequence[str]] = None,
        include_obfuscation: Optional[bool] = None,
        starting_after: Optional[int] = None,
        stream: Optional[bool] = None,
    ) -> ResponseObject:
        """
        Retrieve a model response by ID.

        GET /v1/responses/{response_id}
        """
        params: JSON = {}
        if include is not None:
            params["include"] = list(include)
        if include_obfuscation is not None:
            params["include_obfuscation"] = include_obfuscation
        if starting_after is not None:
            params["starting_after"] = starting_after
        if stream is not None:
            params["stream"] = stream

        data = self._http.get(f"/v1/responses/{response_id}", params=params)
        return ResponseObject.from_dict(data)

    def delete_response(self, response_id: str) -> ResponseDeletionResult:
        """
        Delete a model response.

        DELETE /v1/responses/{response_id}
        """
        data = self._http.delete(f"/v1/responses/{response_id}")
        return ResponseDeletionResult.from_dict(data)

    def cancel_response(self, response_id: str) -> ResponseObject:
        """
        Cancel a background model response.

        POST /v1/responses/{response_id}/cancel
        """
        data = self._http.post(f"/v1/responses/{response_id}/cancel", json={})
        return ResponseObject.from_dict(data)

    # ---- Input items ----------------------------------------------------

    def list_response_input_items(
        self,
        response_id: str,
        *,
        after: Optional[str] = None,
        include: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
        order: Optional[str] = None,
    ) -> InputItemList:
        """
        List input items for a given response.

        GET /v1/responses/{response_id}/input_items
        """
        params: JSON = {}
        if after is not None:
            params["after"] = after
        if include is not None:
            params["include"] = list(include)
        if limit is not None:
            params["limit"] = limit
        if order is not None:
            params["order"] = order

        data = self._http.get(f"/v1/responses/{response_id}/input_items", params=params)
        return InputItemList.from_dict(data)

    # ---- Input token counts ---------------------------------------------

    def get_input_tokens(
        self,
        *,
        model: Optional[str] = None,
        input: Optional[InputType] = None,
        conversation: Optional[Union[str, JSON]] = None,
        previous_response_id: Optional[str] = None,
        **extra: Any,
    ) -> InputTokenCount:
        """
        Get input token counts for a hypothetical response.

        POST /v1/responses/input_tokens

        This mirrors the request body of POST /v1/responses but is used
        only to calculate token counts; no actual model response is
        generated.
        """
        payload: JSON = {}

        if model is not None:
            payload["model"] = model
        if input is not None:
            payload["input"] = input
        if conversation is not None:
            payload["conversation"] = conversation
        if previous_response_id is not None:
            payload["previous_response_id"] = previous_response_id

        payload.update(extra)

        data = self._http.post("/v1/responses/input_tokens", json=payload)
        return InputTokenCount.from_dict(data)

__all__ = [
    "ResponseObject",
    "ResponseUsage",
    "ResponseDeletionResult",
    "InputItem",
    "InputItemList",
    "InputTokenCount",
    "ResponsesMixin",
]
