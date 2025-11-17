"""
Images API
==========

High-level client for the `/v1/images/*` endpoints.

This module wraps the "Images" section of the OpenAI API:

- POST /v1/images/generations → create images from a text prompt
- POST /v1/images/edits       → edit / extend existing images
- POST /v1/images/variations  → create variations of an image

It also defines small data models that normalize the different response
shapes into a consistent `ImageResponse` / `ImageDataItem` structure, as
well as streaming event models for image generation / editing.

Streaming events
----------------

For gpt-image-1 with `stream=true`, the API emits SSE events:

- "image_generation.partial_image"
- "image_generation.completed"
- "image_edit.partial_image"
- "image_edit.completed"

We model these as `ImageStreamEvent` plus a small `ImageStreamEventTypes`
namespace and a helper `parse_image_stream_event()`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from merlin.http_client import MerlinHTTPClient


JSON = Dict[str, Any]


# ───────────────────────────────────────────────────────────────
# Usage / core image response models
# ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ImageUsage:
    """
    Token usage information for an image generation (gpt-image-1 only).

    Docs shape (example):

        "usage": {
          "total_tokens": 100,
          "input_tokens": 50,
          "output_tokens": 50,
          "input_tokens_details": {
            "text_tokens": 10,
            "image_tokens": 40
          }
        }
    """

    total_tokens: Optional[int]
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    input_tokens_details: Optional[JSON]
    raw: JSON = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ImageUsage":
        if not isinstance(data, Mapping):
            return cls(
                total_tokens=None,
                input_tokens=None,
                output_tokens=None,
                input_tokens_details=None,
                raw={},
            )

        def _opt_int(v: Any) -> Optional[int]:
            try:
                return int(v) if v is not None else None
            except (TypeError, ValueError):
                return None

        return cls(
            total_tokens=_opt_int(data.get("total_tokens")),
            input_tokens=_opt_int(data.get("input_tokens")),
            output_tokens=_opt_int(data.get("output_tokens")),
            input_tokens_details=(
                dict(data.get("input_tokens_details"))
                if isinstance(data.get("input_tokens_details"), Mapping)
                else None
            ),
            raw=dict(data),
        )


@dataclass(frozen=True)
class ImageDataItem:
    """
    Single image entry from an images response.

    For gpt-image-1:

        { "b64_json": "..." }

    For dalle-2 / dalle-3:

        { "url": "https://..." }

    Future fields (e.g. revised_prompt) are preserved in `raw`.
    """

    url: Optional[str]
    b64_json: Optional[str]
    raw: JSON = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ImageDataItem":
        if not isinstance(data, Mapping):
            return cls(url=None, b64_json=None, raw={})
        return cls(
            url=data.get("url"),
            b64_json=data.get("b64_json"),
            raw=dict(data),
        )


@dataclass(frozen=True)
class ImageResponse:
    """
    Image generation / edit / variation response.

    Unifies:

    - "The image generation response" for gpt-image-1
    - dalle-2 / dalle-3 generations
    - image edits / variations, which all reuse the `{created, data}` shape
    """

    created: Optional[int]
    data: List[ImageDataItem]

    # gpt-image-1-specific metadata (may be absent for dalle-* / variations)
    background: Optional[str]
    output_format: Optional[str]
    size: Optional[str]
    quality: Optional[str]

    usage: Optional[ImageUsage]
    raw: JSON

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ImageResponse":
        def _opt_int(v: Any) -> Optional[int]:
            try:
                return int(v) if v is not None else None
            except (TypeError, ValueError):
                return None

        images_raw = data.get("data") or []
        items = [
            ImageDataItem.from_dict(item)
            for item in images_raw
            if isinstance(item, Mapping)
        ]

        usage_raw = data.get("usage")
        usage = ImageUsage.from_dict(usage_raw) if isinstance(usage_raw, Mapping) else None

        return cls(
            created=_opt_int(data.get("created")),
            data=items,
            background=data.get("background"),
            output_format=data.get("output_format"),
            size=data.get("size"),
            quality=data.get("quality"),
            usage=usage,
            raw=dict(data),
        )


# ───────────────────────────────────────────────────────────────
# Image streaming events (generation + edit)
# ───────────────────────────────────────────────────────────────


class ImageStreamEventTypes:
    """
    String constants for image streaming event types.

    - image_generation.partial_image
    - image_generation.completed
    - image_edit.partial_image
    - image_edit.completed
    """

    IMAGE_GENERATION_PARTIAL = "image_generation.partial_image"
    IMAGE_GENERATION_COMPLETED = "image_generation.completed"
    IMAGE_EDIT_PARTIAL = "image_edit.partial_image"
    IMAGE_EDIT_COMPLETED = "image_edit.completed"


@dataclass(frozen=True)
class ImageStreamEvent:
    """
    Generic representation of an image streaming event.

    Fields are shared by the four event types:

    Generation:

        type == "image_generation.partial_image"
            b64_json, created_at, size, quality, background,
            output_format, partial_image_index

        type == "image_generation.completed"
            b64_json, created_at, size, quality, background,
            output_format, usage

    Edit:

        type == "image_edit.partial_image"
            b64_json, created_at, size, quality, background,
            output_format, partial_image_index

        type == "image_edit.completed"
            b64_json, created_at, size, quality, background,
            output_format, usage
    """

    type: str

    # Core payload
    b64_json: Optional[str]
    created_at: Optional[int]
    size: Optional[str]
    quality: Optional[str]
    background: Optional[str]
    output_format: Optional[str]

    # Only for *.partial_image
    partial_image_index: Optional[int]

    # Only for *.completed (gpt-image-1)
    usage: Optional[ImageUsage]

    # Raw event JSON
    raw: JSON

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ImageStreamEvent":
        if not isinstance(data, Mapping):
            return cls(
                type="",
                b64_json=None,
                created_at=None,
                size=None,
                quality=None,
                background=None,
                output_format=None,
                partial_image_index=None,
                usage=None,
                raw={},
            )

        event_type = str(data.get("type", ""))

        def _opt_int(v: Any) -> Optional[int]:
            try:
                return int(v) if v is not None else None
            except (TypeError, ValueError):
                return None

        usage_raw = data.get("usage")
        usage = ImageUsage.from_dict(usage_raw) if isinstance(usage_raw, Mapping) else None

        return cls(
            type=event_type,
            b64_json=data.get("b64_json"),
            created_at=_opt_int(data.get("created_at")),
            size=data.get("size"),
            quality=data.get("quality"),
            background=data.get("background"),
            output_format=data.get("output_format"),
            partial_image_index=_opt_int(data.get("partial_image_index")),
            usage=usage,
            raw=dict(data),
        )


def parse_image_stream_event(data: Mapping[str, Any]) -> ImageStreamEvent:
    """
    Parse a raw JSON event dict into an ImageStreamEvent.

    This is intended for use inside SSE / streaming handlers, e.g.:

        for event_json in sse_client:
            evt = parse_image_stream_event(event_json)
            if evt.type == ImageStreamEventTypes.IMAGE_GENERATION_PARTIAL:
                ...
    """
    return ImageStreamEvent.from_dict(data)


# ───────────────────────────────────────────────────────────────
# Client mixin
# ───────────────────────────────────────────────────────────────


class ImagesMixin:
    """
    Mixin providing convenience methods for the Images API.

    Assumptions:
        - The consuming client defines `self._http` as a MerlinHTTPClient.
    """

    _http: MerlinHTTPClient  # for type checkers

    # ---- Create image ----------------------------------------------------

    def create_image(
        self,
        *,
        prompt: str,
        model: Optional[str] = None,
        background: Optional[str] = None,
        moderation: Optional[str] = None,
        n: Optional[int] = None,
        output_compression: Optional[int] = None,
        output_format: Optional[str] = None,
        partial_images: Optional[int] = None,
        quality: Optional[str] = None,
        response_format: Optional[str] = None,
        size: Optional[str] = None,
        stream: Optional[bool] = None,
        style: Optional[str] = None,
        user: Optional[str] = None,
        **extra: Any,
    ) -> ImageResponse:
        """
        Create one or more images from a text prompt.

        POST /v1/images/generations
        """
        payload: JSON = {
            "prompt": prompt,
        }

        if model is not None:
            payload["model"] = model
        if background is not None:
            payload["background"] = background
        if moderation is not None:
            payload["moderation"] = moderation
        if n is not None:
            payload["n"] = n
        if output_compression is not None:
            payload["output_compression"] = output_compression
        if output_format is not None:
            payload["output_format"] = output_format
        if partial_images is not None:
            payload["partial_images"] = partial_images
        if quality is not None:
            payload["quality"] = quality
        if response_format is not None:
            payload["response_format"] = response_format
        if size is not None:
            payload["size"] = size
        if stream is not None:
            payload["stream"] = stream
        if style is not None:
            payload["style"] = style
        if user is not None:
            payload["user"] = user

        payload.update(extra)

        resp = self._http.post(
            "/v1/images/generations",
            json=payload,
            expect_json=True,
        )
        return ImageResponse.from_dict(resp)

    # ---- Edit image ------------------------------------------------------

    def edit_image(
        self,
        *,
        prompt: str,
        image: Any,
        model: Optional[str] = None,
        background: Optional[str] = None,
        input_fidelity: Optional[str] = None,
        mask: Any = None,
        n: Optional[int] = None,
        output_compression: Optional[int] = None,
        output_format: Optional[str] = None,
        partial_images: Optional[int] = None,
        quality: Optional[str] = None,
        response_format: Optional[str] = None,
        size: Optional[str] = None,
        stream: Optional[bool] = None,
        user: Optional[str] = None,
        **extra: Any,
    ) -> ImageResponse:
        """
        Create an edited / extended image from one or more source images.

        POST /v1/images/edits
        """
        data: Dict[str, Any] = {
            "prompt": prompt,
        }

        if model is not None:
            data["model"] = model
        if background is not None:
            data["background"] = background
        if input_fidelity is not None:
            data["input_fidelity"] = input_fidelity
        if n is not None:
            data["n"] = n
        if output_compression is not None:
            data["output_compression"] = output_compression
        if output_format is not None:
            data["output_format"] = output_format
        if partial_images is not None:
            data["partial_images"] = partial_images
        if quality is not None:
            data["quality"] = quality
        if response_format is not None:
            data["response_format"] = response_format
        if size is not None:
            data["size"] = size
        if stream is not None:
            data["stream"] = stream
        if user is not None:
            data["user"] = user

        data.update(extra)

        files: Dict[str, Any] = {
            "image": image,
        }
        if mask is not None:
            files["mask"] = mask

        resp = self._http.post(
            "/v1/images/edits",
            data=data,
            files=files,
            expect_json=True,
        )
        return ImageResponse.from_dict(resp)

    # ---- Image variations -----------------------------------------------

    def create_image_variation(
        self,
        *,
        image: Any,
        model: Optional[str] = None,
        n: Optional[int] = None,
        response_format: Optional[str] = None,
        size: Optional[str] = None,
        user: Optional[str] = None,
        **extra: Any,
    ) -> ImageResponse:
        """
        Create one or more variations of an image.

        POST /v1/images/variations
        """
        data: Dict[str, Any] = {}
        if model is not None:
            data["model"] = model
        if n is not None:
            data["n"] = n
        if response_format is not None:
            data["response_format"] = response_format
        if size is not None:
            data["size"] = size
        if user is not None:
            data["user"] = user

        data.update(extra)

        files = {
            "image": image,
        }

        resp = self._http.post(
            "/v1/images/variations",
            data=data,
            files=files,
            expect_json=True,
        )
        return ImageResponse.from_dict(resp)


__all__ = [
    "ImageUsage",
    "ImageDataItem",
    "ImageResponse",
    "ImageStreamEventTypes",
    "ImageStreamEvent",
    "parse_image_stream_event",
    "ImagesMixin",
]
