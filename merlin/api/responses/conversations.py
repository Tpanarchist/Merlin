"""
Conversations API
=================

High-level client for the `/v1/conversations` endpoints.

This module wraps the "Conversations" section of the OpenAI API:

- POST   /v1/conversations                                   → create a conversation
- GET    /v1/conversations/{conversation_id}                 → retrieve a conversation
- POST   /v1/conversations/{conversation_id}                 → update a conversation's metadata
- DELETE /v1/conversations/{conversation_id}                 → delete a conversation (items remain)

- GET    /v1/conversations/{conversation_id}/items           → list items in a conversation
- POST   /v1/conversations/{conversation_id}/items           → create items in a conversation
- GET    /v1/conversations/{conversation_id}/items/{item_id} → retrieve a single item
- DELETE /v1/conversations/{conversation_id}/items/{item_id} → delete an item (conversation persists)

Design notes
------------

- The Conversation object is relatively small, so we expose a typed view.
- Conversation items are heterogeneous (messages, tool calls, etc.), so we
  keep most of their structure as raw JSON and expose only common fields.
- List responses are modeled explicitly as `ConversationItemList`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from merlin.http_client import MerlinHTTPClient


JSON = Dict[str, Any]


# ───────────────────────────────────────────────────────────────
# Data models
# ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Conversation:
    """
    Representation of a Conversation object.

    Fields mirror the documented Conversation object:
        - id: unique Conversation ID
        - object: always "conversation"
        - created_at: unix timestamp (seconds)
        - metadata: optional key-value pairs
    """

    id: str
    object: str
    created_at: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw: JSON = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Conversation":
        return cls(
            id=str(data.get("id")),
            object=str(data.get("object", "conversation")),
            created_at=int(data.get("created_at", 0)),
            metadata=dict(data.get("metadata") or {}),
            raw=dict(data),
        )


@dataclass(frozen=True)
class ConversationDeletionResult:
    """
    Result of deleting a Conversation.

    The API returns:
        {
          "id": "conv_123",
          "object": "conversation.deleted",
          "deleted": true
        }
    """

    id: str
    object: str
    deleted: bool
    raw: JSON

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ConversationDeletionResult":
        return cls(
            id=str(data.get("id")),
            object=str(data.get("object", "")),
            deleted=bool(data.get("deleted", False)),
            raw=dict(data),
        )


@dataclass(frozen=True)
class ConversationItem:
    """
    A single item in a Conversation.

    These can be messages, tool calls, etc. We surface a few common
    fields and keep everything else as raw JSON.
    """

    id: str
    type: str
    status: Optional[str]
    role: Optional[str]
    raw: JSON

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ConversationItem":
        return cls(
            id=str(data.get("id")),
            type=str(data.get("type")),
            status=data.get("status"),
            role=data.get("role"),
            raw=dict(data),
        )


@dataclass(frozen=True)
class ConversationItemList:
    """
    A list of Conversation items returned by list/create items endpoints.

    Mirrors the documented list object:
        - object: "list"
        - data: array of items
        - first_id, last_id
        - has_more
    """

    object: str
    data: List[ConversationItem]
    first_id: Optional[str]
    last_id: Optional[str]
    has_more: bool
    raw: JSON

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ConversationItemList":
        items_raw = data.get("data", [])
        items = [ConversationItem.from_dict(item) for item in items_raw]

        return cls(
            object=str(data.get("object", "list")),
            data=items,
            first_id=data.get("first_id"),
            last_id=data.get("last_id"),
            has_more=bool(data.get("has_more", False)),
            raw=dict(data),
        )


# ───────────────────────────────────────────────────────────────
# Client mixin
# ───────────────────────────────────────────────────────────────


class ConversationsMixin:
    """
    Mixin providing convenience methods for the Conversations API.

    Assumptions:
        - The consuming client defines `self._http` as a MerlinHTTPClient.
    """

    _http: MerlinHTTPClient  # for type checkers

    # ---- Conversation CRUD ---------------------------------------------

    def create_conversation(
        self,
        *,
        items: Optional[Sequence[JSON]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **extra: Any,
    ) -> Conversation:
        """
        Create a conversation.

        POST /v1/conversations

        Args:
            items:
                Optional initial items (up to 20) to include in the
                conversation context.
            metadata:
                Optional metadata (up to 16 key-value pairs).
            extra:
                Any additional fields accepted by the API in future.

        Returns:
            Conversation
        """
        payload: JSON = {}
        if items is not None:
            payload["items"] = list(items)
        if metadata is not None:
            payload["metadata"] = dict(metadata)

        payload.update(extra)

        data = self._http.post("/v1/conversations", json=payload)
        return Conversation.from_dict(data)

    def get_conversation(self, conversation_id: str) -> Conversation:
        """
        Retrieve a conversation.

        GET /v1/conversations/{conversation_id}
        """
        data = self._http.get(f"/v1/conversations/{conversation_id}")
        return Conversation.from_dict(data)

    def update_conversation(
        self,
        conversation_id: str,
        *,
        metadata: Dict[str, Any],
        **extra: Any,
    ) -> Conversation:
        """
        Update a conversation's metadata.

        POST /v1/conversations/{conversation_id}
        """
        payload: JSON = {"metadata": dict(metadata)}
        payload.update(extra)

        data = self._http.post(f"/v1/conversations/{conversation_id}", json=payload)
        return Conversation.from_dict(data)

    def delete_conversation(self, conversation_id: str) -> ConversationDeletionResult:
        """
        Delete a conversation.

        DELETE /v1/conversations/{conversation_id}

        Note: Items in the conversation are not deleted.
        """
        data = self._http.delete(f"/v1/conversations/{conversation_id}")
        return ConversationDeletionResult.from_dict(data)

    # ---- Items: list, create, retrieve, delete -------------------------

    def list_conversation_items(
        self,
        conversation_id: str,
        *,
        after: Optional[str] = None,
        include: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
        order: Optional[str] = None,
    ) -> ConversationItemList:
        """
        List all items for a conversation.

        GET /v1/conversations/{conversation_id}/items
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

        data = self._http.get(f"/v1/conversations/{conversation_id}/items", params=params)
        return ConversationItemList.from_dict(data)

    def create_conversation_items(
        self,
        conversation_id: str,
        *,
        items: Sequence[JSON],
        include: Optional[Sequence[str]] = None,
        **extra: Any,
    ) -> ConversationItemList:
        """
        Create items in a conversation.

        POST /v1/conversations/{conversation_id}/items
        """
        params: JSON = {}
        if include is not None:
            params["include"] = list(include)

        payload: JSON = {"items": list(items)}
        payload.update(extra)

        data = self._http.post(
            f"/v1/conversations/{conversation_id}/items",
            json=payload,
            params=params,
        )
        return ConversationItemList.from_dict(data)

    def get_conversation_item(
        self,
        conversation_id: str,
        item_id: str,
        *,
        include: Optional[Sequence[str]] = None,
    ) -> ConversationItem:
        """
        Retrieve a single item from a conversation.

        GET /v1/conversations/{conversation_id}/items/{item_id}
        """
        params: JSON = {}
        if include is not None:
            params["include"] = list(include)

        data = self._http.get(
            f"/v1/conversations/{conversation_id}/items/{item_id}",
            params=params,
        )
        return ConversationItem.from_dict(data)

    def delete_conversation_item(
        self,
        conversation_id: str,
        item_id: str,
    ) -> Conversation:
        """
        Delete an item from a conversation.

        DELETE /v1/conversations/{conversation_id}/items/{item_id}

        Returns the updated Conversation object.
        """
        data = self._http.delete(
            f"/v1/conversations/{conversation_id}/items/{item_id}"
        )
        return Conversation.from_dict(data)


__all__ = [
    "Conversation",
    "ConversationDeletionResult",
    "ConversationItem",
    "ConversationItemList",
    "ConversationsMixin",
]
