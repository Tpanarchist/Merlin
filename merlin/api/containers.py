"""
Containers API
==============

Create and manage containers for use with the Code Interpreter tool.

Endpoints
---------

Containers:
    POST   /v1/containers
    GET    /v1/containers
    GET    /v1/containers/{container_id}
    DELETE /v1/containers/{container_id}

Container Files:
    POST   /v1/containers/{container_id}/files
    GET    /v1/containers/{container_id}/files
    GET    /v1/containers/{container_id}/files/{file_id}
    GET    /v1/containers/{container_id}/files/{file_id}/content
    DELETE /v1/containers/{container_id}/files/{file_id}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from merlin.http_client import MerlinHTTPClient

JSON = Dict[str, Any]


# ───────────────────────────────────────────────────────────────
# Dataclasses: Containers
# ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ContainerExpiresAfter:
    """
    Expiration policy for a container.

    Example:
        {
          "anchor": "last_active_at",
          "minutes": 20
        }
    """
    anchor: str
    minutes: int
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ContainerExpiresAfter":
        return cls(
            anchor=d.get("anchor", ""),
            minutes=d.get("minutes", 0),
            raw=dict(d),
        )


@dataclass(frozen=True)
class Container:
    """
    Container object as returned by the API.
    """

    id: str
    object: str
    created_at: int
    status: str
    name: str
    expires_after: Optional[ContainerExpiresAfter] = None
    last_active_at: Optional[int] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Container":
        expires = d.get("expires_after")
        return cls(
            id=d["id"],
            object=d.get("object", "container"),
            created_at=d["created_at"],
            status=d.get("status", ""),
            name=d.get("name", ""),
            expires_after=ContainerExpiresAfter.from_dict(expires)
            if isinstance(expires, Mapping)
            else None,
            last_active_at=d.get("last_active_at"),
            raw=dict(d),
        )


@dataclass(frozen=True)
class ContainerList:
    """
    Paginated list of containers.
    """

    data: List[Container]
    object: str
    has_more: bool
    first_id: Optional[str] = None
    last_id: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ContainerList":
        return cls(
            data=[Container.from_dict(item) for item in d.get("data", [])],
            object=d.get("object", "list"),
            has_more=bool(d.get("has_more", False)),
            first_id=d.get("first_id"),
            last_id=d.get("last_id"),
            raw=dict(d),
        )


# ───────────────────────────────────────────────────────────────
# Dataclasses: Container Files
# ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ContainerFile:
    """
    Container file object.
    """

    id: str
    object: str
    created_at: int
    bytes: int
    container_id: str
    path: str
    source: str
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ContainerFile":
        return cls(
            id=d["id"],
            object=d.get("object", "container.file"),
            created_at=d["created_at"],
            bytes=d.get("bytes", 0),
            container_id=d.get("container_id", ""),
            path=d.get("path", ""),
            source=d.get("source", ""),
            raw=dict(d),
        )


@dataclass(frozen=True)
class ContainerFileList:
    """
    Paginated list of container files.
    """

    data: List[ContainerFile]
    object: str
    has_more: bool
    first_id: Optional[str] = None
    last_id: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ContainerFileList":
        return cls(
            data=[ContainerFile.from_dict(item) for item in d.get("data", [])],
            object=d.get("object", "list"),
            has_more=bool(d.get("has_more", False)),
            first_id=d.get("first_id"),
            last_id=d.get("last_id"),
            raw=dict(d),
        )


# ───────────────────────────────────────────────────────────────
# Mixin
# ───────────────────────────────────────────────────────────────

class ContainersMixin:
    """
    High-level wrapper for Container + Container File APIs.

    Quick usage:

        # Create container
        cntr = client.create_container(
            name="My Container",
            expires_after={"anchor": "last_active_at", "minutes": 20},
            file_ids=["file-abc123"],
        )

        # List containers
        containers = client.list_containers(limit=50)

        # Get one
        cntr = client.get_container(cntr.id)

        # Delete
        ok = client.delete_container(cntr.id)

        # Attach existing File to container (JSON path)
        cfile = client.create_container_file(
            cntr.id,
            file_id="file-abc123",
        )

        # List container files
        files = client.list_container_files(cntr.id)

        # Fetch metadata and content
        meta = client.get_container_file(cntr.id, cfile.id)
        content_bytes = client.get_container_file_content(cntr.id, cfile.id)

        # Delete container file
        deleted = client.delete_container_file(cntr.id, cfile.id)
    """

    _http: MerlinHTTPClient

    # ───────── Containers ─────────

    def create_container(
        self,
        *,
        name: str,
        expires_after: Optional[Mapping[str, Any]] = None,
        file_ids: Optional[List[str]] = None,
    ) -> Container:
        """
        Create a container.

        Args:
            name: Name of the container.
            expires_after: Optional expiration policy, e.g.
                {"anchor": "last_active_at", "minutes": 20}
            file_ids: Optional list of File IDs to copy into the container.
        """
        payload: JSON = {"name": name}
        if expires_after is not None:
            payload["expires_after"] = dict(expires_after)
        if file_ids is not None:
            payload["file_ids"] = list(file_ids)

        resp = self._http.post(
            "/v1/containers",
            json=payload,
        )
        return Container.from_dict(resp)

    def list_containers(
        self,
        *,
        limit: int = 20,
        after: Optional[str] = None,
        order: Optional[str] = None,
    ) -> ContainerList:
        """
        List containers.

        Args:
            limit: Max number of containers to return (1–100).
            after: Cursor for pagination; containers created after this ID.
            order: 'asc' or 'desc' by created_at (default 'desc').
        """
        params: JSON = {"limit": limit}
        if after is not None:
            params["after"] = after
        if order is not None:
            params["order"] = order

        resp = self._http.get(
            "/v1/containers",
            params=params,
        )
        return ContainerList.from_dict(resp)

    def get_container(self, container_id: str) -> Container:
        """
        Retrieve a single container by ID.
        """
        resp = self._http.get(
            f"/v1/containers/{container_id}",
        )
        return Container.from_dict(resp)

    def delete_container(self, container_id: str) -> bool:
        """
        Delete a container.

        Returns:
            True if API reports the container as deleted.
        """
        resp = self._http.delete(
            f"/v1/containers/{container_id}",
        )
        # Spec: { "id": "...", "object": "container.deleted", "deleted": true }
        if isinstance(resp, dict):
            d: Dict[str, Any] = resp
            if "deleted" in d:
                return bool(d.get("deleted"))
            return bool(d.get("id") == container_id)
        return False

    # ───────── Container Files ─────────

    def create_container_file(
        self,
        container_id: str,
        *,
        file_id: Optional[str] = None,
    ) -> ContainerFile:
        """
        Create a container file (JSON path via existing File ID).

        NOTE:
            The API also supports multipart uploads with raw file content,
            but this wrapper only covers the JSON `file_id` variant.
            For raw-data uploads, call the underlying HTTP client directly.

        Args:
            container_id: Container ID.
            file_id: Existing File ID to attach to the container.
        """
        if file_id is None:
            raise ValueError("file_id is required for JSON-based container file creation.")

        payload: JSON = {"file_id": file_id}

        resp = self._http.post(
            f"/v1/containers/{container_id}/files",
            json=payload,
        )
        return ContainerFile.from_dict(resp)

    def list_container_files(
        self,
        container_id: str,
        *,
        limit: int = 20,
        after: Optional[str] = None,
        order: Optional[str] = None,
    ) -> ContainerFileList:
        """
        List files in a container.

        Args:
            container_id: Container ID.
            limit: Max number of files to return (1–100).
            after: Cursor for pagination; files created after this ID.
            order: 'asc' or 'desc' by created_at.
        """
        params: JSON = {"limit": limit}
        if after is not None:
            params["after"] = after
        if order is not None:
            params["order"] = order

        resp = self._http.get(
            f"/v1/containers/{container_id}/files",
            params=params,
        )
        return ContainerFileList.from_dict(resp)

    def get_container_file(
        self,
        container_id: str,
        file_id: str,
    ) -> ContainerFile:
        """
        Retrieve a container file's metadata.
        """
        resp = self._http.get(
            f"/v1/containers/{container_id}/files/{file_id}",
        )
        return ContainerFile.from_dict(resp)

    def get_container_file_content(
        self,
        container_id: str,
        file_id: str,
    ) -> bytes:
        """
        Retrieve raw contents of a container file.

        Returns:
            File content as bytes.
        """
        # Underlying MerlinHTTPClient should pass through raw bytes when
        # expect_json=False (or equivalent).
        resp = self._http.get(
            f"/v1/containers/{container_id}/files/{file_id}/content",
        )
        return resp  # type: ignore[return-value]

    def delete_container_file(
        self,
        container_id: str,
        file_id: str,
    ) -> bool:
        """
        Delete a container file.

        Returns:
            True if API reports the file as deleted.
        """
        resp = self._http.delete(
            f"/v1/containers/{container_id}/files/{file_id}",
        )
        if isinstance(resp, dict):
            d: Dict[str, Any] = resp
            if "deleted" in d:
                return bool(d.get("deleted"))
            return bool(d.get("id") == file_id)
        return False


__all__ = [
    "Container",
    "ContainerExpiresAfter",
    "ContainerList",
    "ContainerFile",
    "ContainerFileList",
    "ContainersMixin",
]
