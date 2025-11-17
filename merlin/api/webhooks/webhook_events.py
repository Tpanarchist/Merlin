"""
Webhook Events
==============

Schema and helpers for OpenAI webhook events.

Webhooks are HTTP POST requests sent by OpenAI to a URL you specify when
certain events occur, such as:

- Background Responses finishing
- Batch jobs completing / failing / expiring
- Fine-tuning jobs succeeding / failing / being cancelled
- Eval runs succeeding / failing / being canceled
- Realtime incoming calls

This module provides:

- WebhookEventTypes: string constants for all documented webhook event types.
- WebhookEvent: a typed view over a webhook event payload.
- parse_webhook_event(): construct a WebhookEvent from a raw JSON dict.

The canonical webhook shape (per docs) is:

    {
      "id": "evt_abc123",
      "type": "response.completed",
      "created_at": 1719168000,
      "object": "event",
      "data": {
        "id": "resp_abc123"
      }
    }

For some events (e.g. realtime.call.incoming), `data` can contain
structured info instead of a simple `{ "id": ... }`. We keep `data`
as raw JSON and expose common convenience fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional


JSON = Dict[str, Any]


class WebhookEventTypes:
    """
    String constants for all documented webhook event types.

    These are provided to avoid typos and magic strings.
    """

    # Background response events
    RESPONSE_COMPLETED = "response.completed"
    RESPONSE_CANCELLED = "response.cancelled"
    RESPONSE_FAILED = "response.failed"
    RESPONSE_INCOMPLETE = "response.incomplete"

    # Batch events
    BATCH_COMPLETED = "batch.completed"
    BATCH_CANCELLED = "batch.cancelled"
    BATCH_EXPIRED = "batch.expired"
    BATCH_FAILED = "batch.failed"

    # Fine-tuning job events
    FT_JOB_SUCCEEDED = "fine_tuning.job.succeeded"
    FT_JOB_FAILED = "fine_tuning.job.failed"
    FT_JOB_CANCELLED = "fine_tuning.job.cancelled"

    # Eval run events
    EVAL_RUN_SUCCEEDED = "eval.run.succeeded"
    EVAL_RUN_FAILED = "eval.run.failed"
    EVAL_RUN_CANCELED = "eval.run.canceled"

    # Realtime events
    REALTIME_CALL_INCOMING = "realtime.call.incoming"


@dataclass(frozen=True)
class WebhookEvent:
    """
    Representation of a single webhook event.

    Attributes:
        id:
            Unique ID of the event (e.g. "evt_abc123").
        type:
            Event type string, e.g. "response.completed",
            "batch.failed", "fine_tuning.job.succeeded", etc.
        created_at:
            Unix timestamp (seconds) for when the event occurred.
        object:
            Object type, documented as always "event".
        data:
            The event payload (structure depends on event type).
            For many events this is a simple `{"id": "<resource_id>"}`.
        raw:
            The full original JSON payload.
    """

    id: str
    type: str
    created_at: int
    object: str
    data: JSON
    raw: JSON

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WebhookEvent":
        """
        Build a WebhookEvent from a raw webhook JSON payload.

        This is tolerant of missing or malformed fields; the intent
        is to preserve the original `payload` in `raw` even if some
        top-level fields are not present.
        """
        return cls(
            id=str(payload.get("id")),
            type=str(payload.get("type")),
            created_at=int(payload.get("created_at", 0)),
            object=str(payload.get("object", "event")),
            data=dict(payload.get("data") or {}),
            raw=dict(payload),
        )

    # Convenience predicates ------------------------------------------------

    @property
    def is_response_event(self) -> bool:
        """True if this event relates to a background Response."""
        return self.type in {
            WebhookEventTypes.RESPONSE_COMPLETED,
            WebhookEventTypes.RESPONSE_CANCELLED,
            WebhookEventTypes.RESPONSE_FAILED,
            WebhookEventTypes.RESPONSE_INCOMPLETE,
        }

    @property
    def is_batch_event(self) -> bool:
        """True if this event relates to a Batch job."""
        return self.type in {
            WebhookEventTypes.BATCH_COMPLETED,
            WebhookEventTypes.BATCH_CANCELLED,
            WebhookEventTypes.BATCH_EXPIRED,
            WebhookEventTypes.BATCH_FAILED,
        }

    @property
    def is_fine_tuning_event(self) -> bool:
        """True if this event relates to a fine-tuning job."""
        return self.type in {
            WebhookEventTypes.FT_JOB_SUCCEEDED,
            WebhookEventTypes.FT_JOB_FAILED,
            WebhookEventTypes.FT_JOB_CANCELLED,
        }

    @property
    def is_eval_event(self) -> bool:
        """True if this event relates to an eval run."""
        return self.type in {
            WebhookEventTypes.EVAL_RUN_SUCCEEDED,
            WebhookEventTypes.EVAL_RUN_FAILED,
            WebhookEventTypes.EVAL_RUN_CANCELED,
        }

    @property
    def is_realtime_call_event(self) -> bool:
        """True if this event relates to a realtime incoming call."""
        return self.type == WebhookEventTypes.REALTIME_CALL_INCOMING

    @property
    def resource_id(self) -> Optional[str]:
        """
        Best-effort extraction of the underlying resource ID.

        For many events, `data` has the shape:
            { "id": "resp_abc123" } or
            { "id": "batch_abc123" } etc.

        For realtime.call.incoming, the primary identifier is `call_id`.
        This helper checks common locations to surface a single ID.
        """
        # Common case: { "id": "<resource_id>" }
        if isinstance(self.data, dict):
            if "id" in self.data and isinstance(self.data["id"], str):
                return self.data["id"]
            # For realtime.call.incoming:
            call_id = self.data.get("call_id")
            if isinstance(call_id, str):
                return call_id
        return None


def parse_webhook_event(payload: Mapping[str, Any]) -> WebhookEvent:
    """
    Parse a raw webhook payload into a WebhookEvent.

    This is a light wrapper around `WebhookEvent.from_dict`, included for
    symmetry with other Merlin parsing helpers.
    """
    return WebhookEvent.from_dict(payload)


__all__ = [
    "WebhookEventTypes",
    "WebhookEvent",
    "parse_webhook_event",
]
