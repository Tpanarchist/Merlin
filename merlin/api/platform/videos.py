"""
Video API
=========

High-level client for the `/v1/videos` endpoints.

This module wraps the "Videos" section of the OpenAI API:

- POST   /v1/videos                    → create a video job
- POST   /v1/videos/{video_id}/remix   → remix an existing completed video
- GET    /v1/videos                    → list video jobs
- GET    /v1/videos/{video_id}         → retrieve a video job
- DELETE /v1/videos/{video_id}         → delete a video job
- GET    /v1/videos/{video_id}/content → download rendered video content

Design notes
------------

- Video generation is asynchronous. The create/remix endpoints return a
  *video job* object whose `status` evolves from "queued" → "in_progress"
  → "completed" or "failed".
- `download_video_content()` returns raw bytes of the rendered asset
  (e.g. MP4) instead of JSON and assumes `MerlinHTTPClient.get` supports
  an `expect_json: bool = True` flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from merlin.http_client import MerlinHTTPClient


JSON = Dict[str, Any]


# ───────────────────────────────────────────────────────────────
# Data models
# ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class VideoError:
    """
    Error payload for a video job that failed.

    The docs do not fully specify the structure; we therefore keep it as
    raw JSON while providing a few convenience fields.
    """

    code: Optional[str]
    message: Optional[str]
    raw: JSON = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VideoError":
        if not isinstance(data, Mapping):
            return cls(code=None, message=None, raw={})
        return cls(
            code=str(data.get("code")) if data.get("code") is not None else None,
            message=str(data.get("message")) if data.get("message") is not None else None,
            raw=dict(data),
        )


@dataclass(frozen=True)
class VideoJob:
    """
    Structured information describing a generated video job.

    Fields mirror the "Video job" schema in the docs.
    """

    id: str
    object: str
    model: Optional[str]
    status: str

    prompt: Optional[str]
    size: Optional[str]
    seconds: Optional[str]
    quality: Optional[str]
    progress: Optional[int]

    created_at: Optional[int]
    completed_at: Optional[int]
    expires_at: Optional[int]

    remixed_from_video_id: Optional[str]
    error: Optional[VideoError]

    raw: JSON

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VideoJob":
        error_raw = data.get("error")
        error = VideoError.from_dict(error_raw) if isinstance(error_raw, Mapping) else None

        def _opt_int(key: str) -> Optional[int]:
            value = data.get(key)
            if value is None:
                return None
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        def _opt_int_direct(v: Any) -> Optional[int]:
            if v is None:
                return None
            try:
                return int(v)
            except (TypeError, ValueError):
                return None

        return cls(
            id=str(data.get("id")),
            object=str(data.get("object", "video")),
            model=data.get("model"),
            status=str(data.get("status", "")),
            prompt=data.get("prompt"),
            size=data.get("size"),
            seconds=data.get("seconds"),
            quality=data.get("quality"),
            progress=_opt_int_direct(data.get("progress")),
            created_at=_opt_int("created_at"),
            completed_at=_opt_int("completed_at"),
            expires_at=_opt_int("expires_at"),
            remixed_from_video_id=data.get("remixed_from_video_id"),
            error=error,
            raw=dict(data),
        )


@dataclass(frozen=True)
class VideoJobList:
    """
    Paginated list of video jobs.

    Mirrors the list response:

        {
          "data": [ {video_job}, ... ],
          "object": "list"
        }
    """

    object: str
    data: List[VideoJob]
    raw: JSON

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VideoJobList":
        jobs_raw = data.get("data", [])
        jobs = [VideoJob.from_dict(job) for job in jobs_raw if isinstance(job, Mapping)]
        return cls(
            object=str(data.get("object", "list")),
            data=jobs,
            raw=dict(data),
        )


# ───────────────────────────────────────────────────────────────
# Client mixin
# ───────────────────────────────────────────────────────────────


class VideosMixin:
    """
    Mixin providing convenience methods for the Video API.

    Assumptions:
        - The consuming client defines `self._http` as a MerlinHTTPClient.
    """

    _http: MerlinHTTPClient  # for type checkers

    # ---- Create / remix --------------------------------------------------

    def create_video(
        self,
        *,
        prompt: str,
        model: Optional[str] = None,
        input_reference: Any = None,
        seconds: Optional[str] = None,
        size: Optional[str] = None,
        **extra: Any,
    ) -> VideoJob:
        """
        Create a new video job.

        POST /v1/videos

        Args:
            prompt:
                Text prompt that describes the video to generate.
            model:
                Video generation model, defaults to "sora-2" on the server.
            input_reference:
                Optional file-like object or tuple referencing an image to
                guide generation (sent as multipart/form-data under the
                field name "input_reference").
            seconds:
                Clip duration in seconds (string). Defaults to "4".
            size:
                Output resolution as "WIDTHxHEIGHT", e.g. "720x1280".
            extra:
                Any additional fields accepted by the API now or in the future.

        Returns:
            VideoJob
        """
        # Multipart form: non-file fields in `data`, image (if any) in `files`.
        data: Dict[str, Any] = {
            "prompt": prompt,
        }
        if model is not None:
            data["model"] = model
        if seconds is not None:
            data["seconds"] = str(seconds)
        if size is not None:
            data["size"] = size

        data.update(extra)

        files: Dict[str, Any] = {}
        if input_reference is not None:
            files["input_reference"] = input_reference

        resp = self._http.post(
            "/v1/videos",
            data=data,
            files=files if files else None,
            expect_json=True,
        )
        return VideoJob.from_dict(resp)

    def remix_video(
        self,
        video_id: str,
        *,
        prompt: str,
        **extra: Any,
    ) -> VideoJob:
        """
        Create a video remix from a completed video.

        POST /v1/videos/{video_id}/remix

        Args:
            video_id:
                Identifier of the completed video to remix.
            prompt:
                Updated text prompt that directs the remix generation.
            extra:
                Any additional fields accepted by the API.

        Returns:
            VideoJob (for the new remix job).
        """
        payload: JSON = {"prompt": prompt}
        payload.update(extra)

        resp = self._http.post(
            f"/v1/videos/{video_id}/remix",
            json=payload,
            expect_json=True,
        )
        return VideoJob.from_dict(resp)

    # ---- List / retrieve / delete ---------------------------------------

    def list_videos(
        self,
        *,
        after: Optional[str] = None,
        limit: Optional[int] = None,
        order: Optional[str] = None,
    ) -> VideoJobList:
        """
        List video jobs for the organization.

        GET /v1/videos

        Args:
            after:
                Identifier for the last item from the previous pagination
                request (for cursor-style pagination).
            limit:
                Maximum number of items to retrieve.
            order:
                "asc" or "desc" by timestamp (server-side default is "desc").

        Returns:
            VideoJobList
        """
        params: Dict[str, Any] = {}
        if after is not None:
            params["after"] = after
        if limit is not None:
            params["limit"] = limit
        if order is not None:
            params["order"] = order

        resp = self._http.get(
            "/v1/videos",
            params=params or None,
        )
        return VideoJobList.from_dict(resp)

    def retrieve_video(self, video_id: str) -> VideoJob:
        """
        Retrieve a single video job.

        GET /v1/videos/{video_id}
        """
        resp = self._http.get(f"/v1/videos/{video_id}")
        return VideoJob.from_dict(resp)

    def delete_video(self, video_id: str) -> VideoJob:
        """
        Delete a video job.

        DELETE /v1/videos/{video_id}

        Returns:
            The deleted video job metadata (same shape as VideoJob).
        """
        resp = self._http.delete(f"/v1/videos/{video_id}")
        return VideoJob.from_dict(resp)

    # ---- Download content -----------------------------------------------

    def download_video_content(
        self,
        video_id: str,
        *,
        variant: Optional[str] = None,
    ) -> bytes:
        """
        Download rendered video content for a completed job.

        GET /v1/videos/{video_id}/content

        Args:
            video_id:
                Identifier of the video whose media to download.
            variant:
                Which downloadable asset to return. When omitted, the
                default MP4 video is returned.

        Returns:
            Raw bytes representing the requested asset (e.g. MP4).
        """
        params: Dict[str, Any] = {}
        if variant is not None:
            params["variant"] = variant

        content = self._http.get(
            f"/v1/videos/{video_id}/content",
            params=params or None,
            expect_json=False,  # require MerlinHTTPClient support
        )
        return content


__all__ = [
    "VideoError",
    "VideoJob",
    "VideoJobList",
    "VideosMixin",
]
