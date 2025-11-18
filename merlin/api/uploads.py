"""
Uploads API
===========

Implements the multi-part Uploads API:

- Create an Upload "session" for large files
- Add Parts (chunks) to an Upload
- Complete the Upload and get a ready-to-use File object
- Cancel an Upload

Endpoints
---------
POST /v1/uploads                     → create upload
POST /v1/uploads/{upload_id}/parts   → add part
POST /v1/uploads/{upload_id}/complete → complete upload
POST /v1/uploads/{upload_id}/cancel  → cancel upload
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, IO, List, Mapping, Optional, Union

from merlin.http_client import MerlinHTTPClient
from merlin.api.files import File


JSON = Dict[str, Any]
FileLike = Union[IO[bytes], bytes, Path, str]


# ───────────────────────────────────────────────────────────────
# Dataclasses
# ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Upload:
    """
    Upload object that can accept byte chunks in the form of Parts.

    Once completed, it may contain a nested File.
    """

    id: str
    object: str
    bytes: int
    created_at: int
    filename: str
    purpose: str
    status: str
    expires_at: int

    file: Optional[File] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Upload":
        file_obj = d.get("file")
        return cls(
            id=d["id"],
            object=d.get("object", "upload"),
            bytes=int(d["bytes"]),
            created_at=int(d["created_at"]),
            filename=d["filename"],
            purpose=d["purpose"],
            status=d["status"],
            expires_at=int(d["expires_at"]),
            file=File.from_dict(file_obj) if file_obj else None,
            raw=dict(d),
        )


@dataclass(frozen=True)
class UploadPart:
    """
    Represents a single chunk (Part) added to an Upload.
    """

    id: str
    object: str
    created_at: int
    upload_id: str
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "UploadPart":
        return cls(
            id=d["id"],
            object=d.get("object", "upload.part"),
            created_at=int(d["created_at"]),
            upload_id=d["upload_id"],
            raw=dict(d),
        )


# ───────────────────────────────────────────────────────────────
# UploadsMixin
# ───────────────────────────────────────────────────────────────

class UploadsMixin:
    """
    High-level wrapper for the Uploads API:

        client.create_upload(...)
        client.add_upload_part(...)
        client.complete_upload(...)
        client.cancel_upload(...)
        client.multipart_upload(...)   # convenience helper

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
    def _infer_size(file: FileLike) -> int:
        """
        Try to infer the total byte size for a FileLike.
        Needed for create_upload(bytes=...).
        """
        if isinstance(file, (str, Path)):
            return os.path.getsize(str(file))

        if isinstance(file, (bytes, bytearray)):
            return len(file)

        if hasattr(file, "fileno"):
            try:
                # Use underlying file descriptor size when possible
                fileno = file.fileno()  # type: ignore[attr-defined]
                if isinstance(fileno, int):
                    return os.fstat(fileno).st_size
            except OSError:
                pass

        # Fallback: read into memory to determine size (last resort).
        # This is not ideal for very large inputs, but keeps the helper robust.
        if hasattr(file, "read"):
            data = file.read()  # type: ignore[call-arg]
            if not isinstance(data, (bytes, bytearray)):
                raise TypeError("File-like object did not return bytes.")
            file.seek(0)  # type: ignore[attr-defined]
            return len(data)

        raise TypeError(f"Unable to infer size for: {type(file)!r}")

    @staticmethod
    def _flatten_expires_after(expires_after: Optional[JSON]) -> JSON:
        """
        Convert expires_after dict into body fields like:

            expires_after: { "anchor": "created_at", "seconds": 3600 }
        """
        return expires_after or {}

    # ── Core endpoints ──────────────────────────────────────────

    def create_upload(
        self,
        *,
        bytes: int,
        filename: str,
        mime_type: str,
        purpose: str,
        expires_after: Optional[JSON] = None,
    ) -> Upload:
        """
        Create an intermediate Upload object.

        Args:
            bytes:      Intended total number of bytes across all parts (≤ 8 GB).
            filename:   Name of the file (e.g. "training_examples.jsonl").
            mime_type:  MIME type for the file (must be valid for the purpose).
            purpose:    File purpose (e.g. "fine-tune", "assistants", "batch", etc.).
            expires_after:
                Optional expiration policy, e.g.:
                {"anchor": "created_at", "seconds": 3600}
        """
        body: JSON = {
            "purpose": purpose,
            "filename": filename,
            "bytes": int(bytes),
            "mime_type": mime_type,
        }
        if expires_after:
            body["expires_after"] = self._flatten_expires_after(expires_after)

        resp = self._http.post(
            "/v1/uploads",
            json=body
        )
        return Upload.from_dict(resp)

    def add_upload_part(
        self,
        upload_id: str,
        *,
        data: FileLike,
    ) -> UploadPart:
        """
        Add a Part (chunk) to an existing Upload.

        Each part must be ≤ 64 MB. You can upload parts in any order.
        The final order is specified when calling complete_upload.
        """
        fobj = self._coerce_file(data)

        # The API expects multipart form data with the part as 'data'
        # We'll use 'data' param for the file content
        resp = self._http.post(
            f"/v1/uploads/{upload_id}/parts",
            data={"data": fobj}
        )
        return UploadPart.from_dict(resp)

    def complete_upload(
        self,
        upload_id: str,
        *,
        part_ids: List[str],
        md5: Optional[str] = None,
    ) -> Upload:
        """
        Complete an Upload, specifying the ordered list of part IDs.

        Args:
            upload_id: ID of the Upload to complete.
            part_ids:  Ordered list of part IDs in the final file.
            md5:       Optional MD5 checksum of the full file contents.
        """
        body: JSON = {"part_ids": part_ids}
        if md5 is not None:
            body["md5"] = md5

        resp = self._http.post(
            f"/v1/uploads/{upload_id}/complete",
            json=body
        )
        return Upload.from_dict(resp)

    def cancel_upload(self, upload_id: str) -> Upload:
        """
        Cancel an Upload so no further parts can be added.
        """
        resp = self._http.post(
            f"/v1/uploads/{upload_id}/cancel"
        )
        return Upload.from_dict(resp)

    # ── Convenience: full multi-part upload in one call ─────────

    def multipart_upload(
        self,
        *,
        file: FileLike,
        purpose: str,
        mime_type: str,
        filename: Optional[str] = None,
        part_size: int = 64 * 1024 * 1024,  # 64 MB
        expires_after: Optional[JSON] = None,
        md5: Optional[str] = None,
    ) -> File:
        """
        High-level convenience helper:

        1. Creates an Upload for the given file.
        2. Streams the file in chunks (≤ part_size, default 64MB) as Parts.
        3. Completes the Upload.
        4. Returns the resulting File object.

        NOTE: This reads and uploads synchronously in the current process.
        """
        # Infer filename & size
        size = self._infer_size(file)
        fobj = self._coerce_file(file)

        if filename is None:
            if hasattr(fobj, "name") and isinstance(fobj.name, str):
                filename = fobj.name
            else:
                filename = "upload.bin"

        # 1. Create the Upload
        upload = self.create_upload(
            bytes=size,
            filename=filename,
            mime_type=mime_type,
            purpose=purpose,
            expires_after=expires_after,
        )

        # 2. Add parts
        part_ids: List[str] = []
        while True:
            chunk = fobj.read(part_size)  # type: ignore[arg-type]
            if not chunk:
                break
            part = self.add_upload_part(upload.id, data=chunk)
            part_ids.append(part.id)

        # 3. Complete the upload
        final_upload = self.complete_upload(
            upload.id,
            part_ids=part_ids,
            md5=md5,
        )

        # 4. Return the created File
        if not final_upload.file:
            raise RuntimeError(
                f"Upload {final_upload.id} completed but no file was returned."
            )
        return final_upload.file


__all__ = [
    "Upload",
    "UploadPart",
    "UploadsMixin",
]
