"""
Streaming Events
================

Schema and helpers for Responses API streaming events.

When you create a Response with `stream=True`, the server sends a stream of
Server-Sent Events (SSE). Each event is a small JSON object with a `type`
field (e.g. "response.created", "response.output_text.delta", "error", etc.)
and other fields depending on the event.

This module provides:

- EventTypes: string constants for all documented event types.
- StreamEvent: a generic, typed view over *any* streaming event.
- parse_stream_event(): construct a StreamEvent from a raw JSON dict.

We intentionally keep the schema flexible:

- `StreamEvent.raw` always contains the full original JSON.
- Common fields (sequence_number, item_id, delta, text, etc.) are surfaced
  as optional attributes for convenience, but not all events use all fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional


JSON = Dict[str, Any]


class EventTypes:
    """
    String constants for all documented streaming event types.

    These are provided for discoverability and to avoid typos.
    """

    # Response lifecycle
    RESPONSE_CREATED = "response.created"
    RESPONSE_IN_PROGRESS = "response.in_progress"
    RESPONSE_COMPLETED = "response.completed"
    RESPONSE_FAILED = "response.failed"
    RESPONSE_INCOMPLETE = "response.incomplete"
    RESPONSE_QUEUED = "response.queued"

    # Output items & content parts
    OUTPUT_ITEM_ADDED = "response.output_item.added"
    OUTPUT_ITEM_DONE = "response.output_item.done"

    CONTENT_PART_ADDED = "response.content_part.added"
    CONTENT_PART_DONE = "response.content_part.done"

    # Text output
    OUTPUT_TEXT_DELTA = "response.output_text.delta"
    OUTPUT_TEXT_DONE = "response.output_text.done"
    OUTPUT_TEXT_ANNOTATION_ADDED = "response.output_text.annotation.added"

    # Refusal
    REFUSAL_DELTA = "response.refusal.delta"
    REFUSAL_DONE = "response.refusal.done"

    # Function call arguments
    FUNCTION_CALL_ARGS_DELTA = "response.function_call_arguments.delta"
    FUNCTION_CALL_ARGS_DONE = "response.function_call_arguments.done"

    # File search tool calls
    FILE_SEARCH_IN_PROGRESS = "response.file_search_call.in_progress"
    FILE_SEARCH_SEARCHING = "response.file_search_call.searching"
    FILE_SEARCH_COMPLETED = "response.file_search_call.completed"

    # Web search tool calls
    WEB_SEARCH_IN_PROGRESS = "response.web_search_call.in_progress"
    WEB_SEARCH_SEARCHING = "response.web_search_call.searching"
    WEB_SEARCH_COMPLETED = "response.web_search_call.completed"

    # Reasoning summary parts
    REASONING_SUMMARY_PART_ADDED = "response.reasoning_summary_part.added"
    REASONING_SUMMARY_PART_DONE = "response.reasoning_summary_part.done"
    REASONING_SUMMARY_TEXT_DELTA = "response.reasoning_summary_text.delta"
    REASONING_SUMMARY_TEXT_DONE = "response.reasoning_summary_text.done"

    # Reasoning text (per content part)
    REASONING_TEXT_DELTA = "response.reasoning_text.delta"
    REASONING_TEXT_DONE = "response.reasoning_text.done"

    # Image generation tool calls
    IMAGE_GEN_COMPLETED = "response.image_generation_call.completed"
    IMAGE_GEN_GENERATING = "response.image_generation_call.generating"
    IMAGE_GEN_IN_PROGRESS = "response.image_generation_call.in_progress"
    IMAGE_GEN_PARTIAL_IMAGE = "response.image_generation_call.partial_image"

    # MCP tool calls
    MCP_ARGS_DELTA = "response.mcp_call_arguments.delta"
    MCP_ARGS_DONE = "response.mcp_call_arguments.done"
    MCP_COMPLETED = "response.mcp_call.completed"
    MCP_FAILED = "response.mcp_call.failed"
    MCP_IN_PROGRESS = "response.mcp_call.in_progress"

    # MCP list tools
    MCP_LIST_TOOLS_COMPLETED = "response.mcp_list_tools.completed"
    MCP_LIST_TOOLS_FAILED = "response.mcp_list_tools.failed"
    MCP_LIST_TOOLS_IN_PROGRESS = "response.mcp_list_tools.in_progress"

    # Code interpreter tool calls
    CI_IN_PROGRESS = "response.code_interpreter_call.in_progress"
    CI_INTERPRETING = "response.code_interpreter_call.interpreting"
    CI_COMPLETED = "response.code_interpreter_call.completed"
    CI_CODE_DELTA = "response.code_interpreter_call_code.delta"
    CI_CODE_DONE = "response.code_interpreter_call_code.done"

    # Custom tool call input
    CUSTOM_TOOL_INPUT_DELTA = "response.custom_tool_call_input.delta"
    CUSTOM_TOOL_INPUT_DONE = "response.custom_tool_call_input.done"

    # Generic error event
    ERROR = "error"


@dataclass(frozen=True)
class StreamEvent:
    """
    Generic representation of a single streaming event.

    Attributes expose the most common fields across all event types; for
    anything more specialized, inspect `raw` directly.

    Common patterns:
        - Lifecycle events:
            type == "response.completed", "response.failed", etc.
            response: full Response object (if present on event)
        - Text deltas:
            type == "response.output_text.delta"
            delta: text fragment (if present)
        - Reasoning deltas:
            type == "response.reasoning_text.delta"
        - Function/tool call args:
            type == "response.function_call_arguments.delta" / ".done"
        - Tool calls (file_search, web_search, MCP, code interpreter, image gen):
            item_id, output_index, sequence_number identify the call.
        - Error:
            type == "error"
            error_code, error_message, error_param populated from the event.
    """

    type: str
    sequence_number: Optional[int]

    # For full response-carrying events (created / in_progress / completed etc.)
    response: Optional[JSON]

    # Common identifiers
    item_id: Optional[str]
    output_index: Optional[int]
    content_index: Optional[int]
    summary_index: Optional[int]

    # Text / delta content
    delta: Optional[str]
    text: Optional[str]
    refusal: Optional[str]
    arguments: Optional[str]
    code: Optional[str]

    # Image generation partials
    partial_image_b64: Optional[str]
    partial_image_index: Optional[int]

    # Annotations & logprobs
    annotation: Optional[JSON]
    annotation_index: Optional[int]
    logprobs: Optional[List[JSON]]

    # Error event fields
    error_code: Optional[str]
    error_message: Optional[str]
    error_param: Optional[str]

    # The original raw event JSON
    raw: JSON

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StreamEvent":
        """
        Build a StreamEvent from a raw event dict.

        This is tolerant of missing fields; all convenience attributes
        are optional and default to None if not present.
        """
        event_type = str(data.get("type", ""))

        # Normalize frequently occurring names.
        seq = data.get("sequence_number")
        sequence_number = int(seq) if isinstance(seq, int) else None

        # Response object, when present.
        response = data.get("response")

        # Common identifiers.
        item_id = data.get("item_id")
        output_index = data.get("output_index")
        content_index = data.get("content_index")
        summary_index = data.get("summary_index")

        # Text / delta-like fields.
        delta = data.get("delta")
        text = data.get("text")
        refusal = data.get("refusal")
        arguments = data.get("arguments")
        code = data.get("code")

        # Image partials.
        partial_image_b64 = data.get("partial_image_b64")
        partial_image_index = data.get("partial_image_index")

        # Annotation & logprobs.
        annotation = data.get("annotation")
        annotation_index = data.get("annotation_index")
        logprobs = data.get("logprobs")

        # Error event fields.
        error_code = data.get("code") if event_type == EventTypes.ERROR else None
        error_message = data.get("message") if event_type == EventTypes.ERROR else None
        error_param = data.get("param") if event_type == EventTypes.ERROR else None

        return cls(
            type=event_type,
            sequence_number=sequence_number,
            response=response if isinstance(response, Mapping) else None,
            item_id=item_id,
            output_index=output_index,
            content_index=content_index,
            summary_index=summary_index,
            delta=delta,
            text=text,
            refusal=refusal,
            arguments=arguments,
            code=code,
            partial_image_b64=partial_image_b64,
            partial_image_index=partial_image_index,
            annotation=annotation if isinstance(annotation, Mapping) else None,
            annotation_index=annotation_index,
            logprobs=list(logprobs) if isinstance(logprobs, list) else None,
            error_code=error_code,
            error_message=error_message,
            error_param=error_param,
            raw=dict(data),
        )

    # Convenience predicates ------------------------------------------------

    @property
    def is_error(self) -> bool:
        """True if this event is an error event."""
        return self.type == EventTypes.ERROR

    @property
    def is_terminal(self) -> bool:
        """
        True if this event represents a terminal state for the response
        lifecycle: completed / failed / incomplete / error.
        """
        return self.type in {
            EventTypes.RESPONSE_COMPLETED,
            EventTypes.RESPONSE_FAILED,
            EventTypes.RESPONSE_INCOMPLETE,
            EventTypes.ERROR,
        }


def parse_stream_event(data: Mapping[str, Any]) -> StreamEvent:
    """
    Parse a raw event dict into a StreamEvent.

    This is a light wrapper around `StreamEvent.from_dict` for symmetry with
    other Merlin parsing helpers.
    """
    return StreamEvent.from_dict(data)


__all__ = [
    "EventTypes",
    "StreamEvent",
    "parse_stream_event",
]
