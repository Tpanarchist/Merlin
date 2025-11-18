"""
Files API
=========

Implements the Files API:

- Upload files for Assistants, Fine-tuning, Batch, Evals, etc.
- List, retrieve, delete, and download file contents.

Endpoints
---------
POST   /v1/files                 → upload file
GET    /v1/files                 → list files
GET    /v1/files/{file_id}       → retrieve file
DELETE /v1/files/{file_id}       → delete file
GET    /v1/files/{file_id}/content → retrieve file content
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, IO, List, Mapping, Optional, Union

from merlin.http_client import MerlinHTTPClient


JSON = Dict[str, Any]
FileLike = Union[IO[bytes], bytes, Path, str]


# ───────────────────────────────────────────────────────────────
# Dataclasses
# ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class File:
    """
    The File object represents a document uploaded to OpenAI.
    """

    id: str
    object: str
    bytes: int
    created_at: int
    expires_at: Optional[int]
    filename: str
    purpose: str

    # Deprecated, but included for completeness
    status: Optional[str] = None
    status_details: Optional[str] = None

    raw: JSON = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "File":
        return cls(
            id=d["id"],
            object=d.get("object", "file"),
            bytes=int(d["bytes"]),
            created_at=int(d["created_at"]),
            expires_at=d.get("expires_at"),
            filename=d["filename"],
            purpose=d["purpose"],
            status=d.get("status"),
            status_details=d.get("status_details"),
            raw=dict(d),
        )


@dataclass(frozen=True)
class FileList:
    """
    A paginated list of File objects.
    """

    object: str
    data: List[File]
    first_id: Optional[str]
    last_id: Optional[str]
    has_more: bool
    raw: JSON = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "FileList":
        return cls(
            object=d.get("object", "list"),
            data=[File.from_dict(x) for x in d.get("data", [])],
            first_id=d.get("first_id"),
            last_id=d.get("last_id"),
            has_more=bool(d.get("has_more", False)),
            raw=dict(d),
        )


@dataclass(frozen=True)
class FileDeleted:
    """
    Response for DELETE /v1/files/{file_id}.
    """

    id: str
    object: str
    deleted: bool
    raw: JSON = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "FileDeleted":
        return cls(
            id=d["id"],
            object=d.get("object", "file"),
            deleted=bool(d.get("deleted", False)),
            raw=dict(d),
        )


# ───────────────────────────────────────────────────────────────
# FilesMixin
# ───────────────────────────────────────────────────────────────

class FilesMixin:
    """
    High-level wrapper for the Files API:

        client.upload_file(...)
        client.list_files(...)
        client.retrieve_file(...)
        client.delete_file(...)
        client.download_file_content(...)

    Assumes `self._http` is a MerlinHTTPClient instance.
    """

    _http: MerlinHTTPClient

    # ── Helpers ─────────────────────────────────────────────────

    @staticmethod
    def _coerce_file(file: FileLike) -> IO[bytes]:
        """
        Normalize file input to a binary file-like object.

        Accepts:
          - open file objects
          - bytes
          - pathlib.Path / str paths
        """
        if hasattr(file, "read"):
            # Already a file-like object
            return file  # type: ignore[return-value]

        if isinstance(file, (str, Path)):
            return open(str(file), "rb")

        if isinstance(file, (bytes, bytearray)):
            import io
            return io.BytesIO(file)

        raise TypeError(f"Unsupported file type for upload: {type(file)!r}")

    @staticmethod
    def _flatten_expires_after(expires_after: Optional[JSON]) -> JSON:
        """
        Convert expires_after dict into multipart form fields like:

            expires_after[anchor]=created_at
            expires_after[seconds]=2592000
        """
        if not expires_after:
            return {}

        flat: JSON = {}
        for k, v in expires_after.items():
            flat[f"expires_after[{k}]"] = str(v)
        return flat

    # ── Upload file ─────────────────────────────────────────────

    def upload_file(
        self,
        *,
        file: FileLike,
        purpose: str,
        expires_after: Optional[JSON] = None,
        filename: Optional[str] = None,
    ) -> File:
        """
        Upload a file.

        Args:
            file:
                - Path, str path
                - open binary file object
                - bytes
            purpose:
                One of:
                - assistants
                - batch
                - fine-tune
                - vision
                - user_data
                - evals
            expires_after:
                Optional expiration policy:
                {"anchor": "created_at", "seconds": 2592000}
            filename:
                Optional override of filename sent to the API.
        """
        fobj = self._coerce_file(file)
        fname = filename
        if fname is None:
            # Try to pull a reasonable default from file object
            if hasattr(fobj, "name") and isinstance(fobj.name, str):
                fname = fobj.name
            else:
                fname = "upload.bin"

        data: JSON = {"purpose": purpose}
        data.update(self._flatten_expires_after(expires_after))

        files = {
            "file": (fname, fobj),
        }

        resp = self._http.post(
            "/v1/files",
            data=data,
            files=files
        )
        return File.from_dict(resp)

    # ── List files ──────────────────────────────────────────────

    def list_files(
        self,
        *,
        after: Optional[str] = None,
        limit: Optional[int] = None,
        order: Optional[str] = None,
        purpose: Optional[str] = None,
    ) -> FileList:
        """
        List files with optional filters/pagination.
        """
        params: JSON = {}
        if after is not None:
            params["after"] = after
        if limit is not None:
            params["limit"] = int(limit)
        if order is not None:
            params["order"] = order
        if purpose is not None:
            params["purpose"] = purpose

        resp = self._http.get(
            "/v1/files",
            params=params
        )
        return FileList.from_dict(resp)

    # ── Retrieve file ───────────────────────────────────────────

    def retrieve_file(self, file_id: str) -> File:
        """
        Retrieve file metadata by ID.
        """
        resp = self._http.get(
            f"/v1/files/{file_id}"
        )
        return File.from_dict(resp)

    # ── Delete file ─────────────────────────────────────────────

    def delete_file(self, file_id: str) -> FileDeleted:
        """
        Delete a file and remove it from all vector stores.
        """
        resp = self._http.delete(
            f"/v1/files/{file_id}"
        )
        return FileDeleted.from_dict(resp)

    # ── Download content ────────────────────────────────────────

    def download_file_content(self, file_id: str) -> bytes:
        """
        Download raw file content for a given file ID.

        Returns:
            bytes: raw file content (JSONL, PDF, image, etc.)
        """
        return self._http.get(
            f"/v1/files/{file_id}/content"
        )


__all__ = [
    "File",
    "FileList",
    "FileDeleted",
    "FilesMixin",
]
