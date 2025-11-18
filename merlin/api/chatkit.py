"""
ChatKit API
===========

Manage ChatKit sessions, threads, and thread items.

Endpoints
---------

Sessions:
    POST /v1/chatkit/sessions
    POST /v1/chatkit/sessions/{session_id}/cancel

Threads:
    GET  /v1/chatkit/threads
    GET  /v1/chatkit/threads/{thread_id}
    DELETE /v1/chatkit/threads/{thread_id}
    GET  /v1/chatkit/threads/{thread_id}/items
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from merlin.http_client import MerlinHTTPClient

JSON = Dict[str, Any]


# ───────────────────────────────────────────────────────────────
# Session dataclasses
# ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ChatKitRateLimits:
    """
    Resolved rate limits for a ChatKit session.
    """

    max_requests_per_1_minute: Optional[int] = None
    max_requests_per_session: Optional[int] = None
    raw: JSON = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ChatKitRateLimits":
        if not d:
            return cls(raw={})
        return cls(
            max_requests_per_1_minute=d.get("max_requests_per_1_minute"),
            max_requests_per_session=d.get("max_requests_per_session"),
            raw=dict(d),
        )


@dataclass(frozen=True)
class ChatKitConfiguration:
    """
    Resolved ChatKit configuration for a session.
    """

    automatic_thread_titling: Optional[JSON] = None
    file_upload: Optional[JSON] = None
    history: Optional[JSON] = None
    raw: JSON = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ChatKitConfiguration":
        if not d:
            return cls(raw={})
        return cls(
            automatic_thread_titling=d.get("automatic_thread_titling"),
            file_upload=d.get("file_upload"),
            history=d.get("history"),
            raw=dict(d),
        )


@dataclass(frozen=True)
class ChatKitWorkflow:
    """
    Workflow metadata powering a ChatKit session.
    """

    id: str
    version: Optional[str] = None
    raw: JSON = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ChatKitWorkflow":
        return cls(id=d["id"], version=d.get("version"), raw=dict(d))


@dataclass(frozen=True)
class ChatKitSession:
    """
    Represents a ChatKit session and its resolved configuration.
    """

    id: str
    user: Optional[str]
    client_secret: Optional[str]
    expires_at: Optional[int]
    workflow: Optional[ChatKitWorkflow]
    status: str
    chatkit_configuration: ChatKitConfiguration
    rate_limits: ChatKitRateLimits
    max_requests_per_1_minute: Optional[int] = None
    object: str = "chatkit.session"
    raw: JSON = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ChatKitSession":
        workflow = d.get("workflow")
        cfg = d.get("chatkit_configuration", {})
        rl = d.get("rate_limits", {})

        return cls(
            id=d.get("id", ""),  # create endpoint may omit id in example
            user=d.get("user"),
            client_secret=d.get("client_secret"),
            expires_at=d.get("expires_at"),
            workflow=ChatKitWorkflow.from_dict(workflow) if isinstance(workflow, Mapping) else None,
            status=d.get("status", "active"),
            chatkit_configuration=ChatKitConfiguration.from_dict(cfg),
            rate_limits=ChatKitRateLimits.from_dict(rl),
            max_requests_per_1_minute=d.get("max_requests_per_1_minute"),
            object=d.get("object", "chatkit.session"),
            raw=dict(d),
        )


# ───────────────────────────────────────────────────────────────
# Thread & items dataclasses
# ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ChatKitThreadItemContent:
    """
    One content block inside a thread item (input_text, output_text, etc.).
    """

    type: str
    text: Optional[str] = None
    raw: JSON = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ChatKitThreadItemContent":
        return cls(
            type=d.get("type", ""),
            text=d.get("text"),
            raw=dict(d),
        )


@dataclass(frozen=True)
class ChatKitThreadItem:
    """
    A single item in a ChatKit thread: user_message, assistant_message, etc.
    """

    id: str
    type: str
    object: str
    content: List[ChatKitThreadItemContent]
    attachments: List[JSON] = field(default_factory=list)
    raw: JSON = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ChatKitThreadItem":
        return cls(
            id=d["id"],
            type=d.get("type", ""),
            object=d.get("object", "chatkit.thread_item"),
            content=[
                ChatKitThreadItemContent.from_dict(c)
                for c in d.get("content", [])
            ],
            attachments=list(d.get("attachments", [])),
            raw=dict(d),
        )


@dataclass(frozen=True)
class ChatKitThreadItemsPage:
    """
    Paginated list of items in a ChatKit thread.
    """

    data: List[ChatKitThreadItem]
    has_more: bool
    object: str = "list"
    first_id: Optional[str] = None
    last_id: Optional[str] = None
    raw: JSON = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ChatKitThreadItemsPage":
        return cls(
            data=[
                ChatKitThreadItem.from_dict(item)
                for item in d.get("data", [])
            ],
            has_more=bool(d.get("has_more", False)),
            object=d.get("object", "list"),
            first_id=d.get("first_id"),
            last_id=d.get("last_id"),
            raw=dict(d),
        )


@dataclass(frozen=True)
class ChatKitThread:
    """
    Represents a ChatKit thread.
    """

    id: str
    created_at: Optional[int]
    title: Optional[str]
    status: Optional[JSON]
    user: Optional[str]
    object: str = "chatkit.thread"
    items: Optional[ChatKitThreadItemsPage] = None
    raw: JSON = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ChatKitThread":
        items_block = d.get("items")
        items_page = (
            ChatKitThreadItemsPage.from_dict(items_block)
            if isinstance(items_block, Mapping)
            else None
        )
        return cls(
            id=d["id"],
            created_at=d.get("created_at"),
            title=d.get("title"),
            status=d.get("status"),
            user=d.get("user"),
            object=d.get("object", "chatkit.thread"),
            items=items_page,
            raw=dict(d),
        )


@dataclass(frozen=True)
class ChatKitThreadList:
    """
    Paginated list of threads.
    """

    data: List[ChatKitThread]
    has_more: bool
    object: str = "list"
    first_id: Optional[str] = None
    last_id: Optional[str] = None
    raw: JSON = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ChatKitThreadList":
        return cls(
            data=[ChatKitThread.from_dict(item) for item in d.get("data", [])],
            has_more=bool(d.get("has_more", False)),
            object=d.get("object", "list"),
            first_id=d.get("first_id"),
            last_id=d.get("last_id"),
            raw=dict(d),
        )


# ───────────────────────────────────────────────────────────────
# Mixin
# ───────────────────────────────────────────────────────────────

class ChatKitMixin:
    """
    High-level wrapper for ChatKit APIs.

    Usage sketch:

        # Create a session for a user/workflow
        sess = client.create_chatkit_session(
            user="user_123",
            workflow={"id": "workflow_alpha", "version": "2024-10-01"},
            expires_after=1800,
            rate_limits={"max_requests_per_1_minute": 60},
            scope={"project": "alpha", "environment": "staging"},
        )

        # Cancel it
        cancelled = client.cancel_chatkit_session(sess.id)

        # List threads for a user
        threads = client.list_chatkit_threads(user="user_123")

        # Fetch a full thread (with first page of items)
        thread = client.get_chatkit_thread("cthr_abc123")

        # Fetch items directly
        items_page = client.list_chatkit_thread_items("cthr_abc123", limit=50)
    """

    _http: MerlinHTTPClient

    # ───────── Sessions ─────────

    def create_chatkit_session(
        self,
        *,
        user: str,
        workflow: Mapping[str, Any],
        chatkit_configuration: Optional[Mapping[str, Any]] = None,
        expires_after: Optional[int] = None,
        rate_limits: Optional[Mapping[str, Any]] = None,
        **extra: Any,
    ) -> ChatKitSession:
        """
        Create a ChatKit session.

        Args:
            user: Identifier for the end-user that owns this session.
            workflow: Workflow descriptor, e.g. {"id": "workflow_alpha", "version": "2024-10-01"}.
            chatkit_configuration: Optional overrides for ChatKit runtime config.
            expires_after: Optional session TTL in seconds from creation.
            rate_limits: Optional rate limit overrides, e.g. {"max_requests_per_1_minute": 60}.
            **extra: Forward-compat fields like "scope", "max_requests_per_session", etc.
        """
        payload: JSON = {
            "user": user,
            "workflow": dict(workflow),
        }
        if chatkit_configuration is not None:
            payload["chatkit_configuration"] = dict(chatkit_configuration)
        if expires_after is not None:
            payload["expires_after"] = expires_after
        if rate_limits is not None:
            payload["rate_limits"] = dict(rate_limits)
        # allow things like scope, max_requests_per_1_minute, max_requests_per_session
        payload.update(extra)

        resp = self._http.post(
            "/v1/chatkit/sessions",
            json=payload,
            expect_json=True,
        )
        return ChatKitSession.from_dict(resp)

    def cancel_chatkit_session(self, session_id: str) -> ChatKitSession:
        """
        Cancel a ChatKit session. Prevents new requests using its client secret.
        """
        resp = self._http.post(
            f"/v1/chatkit/sessions/{session_id}/cancel",
            json={},
            expect_json=True,
        )
        return ChatKitSession.from_dict(resp)

    # ───────── Threads ─────────

    def list_chatkit_threads(
        self,
        *,
        limit: int = 20,
        after: Optional[str] = None,
        before: Optional[str] = None,
        order: Optional[str] = None,
        user: Optional[str] = None,
    ) -> ChatKitThreadList:
        """
        List ChatKit threads.

        Args:
            limit: Max number of threads to return (default 20).
            after: Return items created after this thread ID (for pagination).
            before: Return items created before this thread ID.
            order: "asc" or "desc" by creation time (default "desc").
            user: Optional user identifier to filter threads.
        """
        params: JSON = {"limit": limit}
        if after is not None:
            params["after"] = after
        if before is not None:
            params["before"] = before
        if order is not None:
            params["order"] = order
        if user is not None:
            params["user"] = user

        resp = self._http.get(
            "/v1/chatkit/threads",
            params=params,
            expect_json=True,
        )
        return ChatKitThreadList.from_dict(resp)

    def get_chatkit_thread(self, thread_id: str) -> ChatKitThread:
        """
        Retrieve a ChatKit thread by ID (may include first page of items).
        """
        resp = self._http.get(
            f"/v1/chatkit/threads/{thread_id}",
            expect_json=True,
        )
        return ChatKitThread.from_dict(resp)

    def delete_chatkit_thread(self, thread_id: str) -> bool:
        """
        Delete a ChatKit thread.

        Returns:
            True if the API indicates the thread was deleted.
        """
        resp = self._http.delete(
            f"/v1/chatkit/threads/{thread_id}",
            expect_json=True,
        )
        # Spec: "Returns a confirmation object for the deleted thread."
        # We tolerate either a "deleted" flag or just presence of an id.
        if "deleted" in resp:
            return bool(resp.get("deleted"))
        return bool(resp.get("id") == thread_id)

    def list_chatkit_thread_items(
        self,
        thread_id: str,
        *,
        limit: int = 20,
        after: Optional[str] = None,
        before: Optional[str] = None,
        order: Optional[str] = None,
    ) -> ChatKitThreadItemsPage:
        """
        List items for a given ChatKit thread.
        """
        params: JSON = {"limit": limit}
        if after is not None:
            params["after"] = after
        if before is not None:
            params["before"] = before
        if order is not None:
            params["order"] = order

        resp = self._http.get(
            f"/v1/chatkit/threads/{thread_id}/items",
            params=params,
            expect_json=True,
        )
        return ChatKitThreadItemsPage.from_dict(resp)


__all__ = [
    "ChatKitSession",
    "ChatKitRateLimits",
    "ChatKitConfiguration",
    "ChatKitWorkflow",
    "ChatKitThread",
    "ChatKitThreadItem",
    "ChatKitThreadItemContent",
    "ChatKitThreadItemsPage",
    "ChatKitThreadList",
    "ChatKitMixin",
]
