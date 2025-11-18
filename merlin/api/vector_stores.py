"""
Vector Stores API
=================

Vector stores power semantic search for the Retrieval API and the `file_search`
tool in the Responses and Assistants APIs.

Endpoints
---------

Vector stores:
    POST   /v1/vector_stores
    GET    /v1/vector_stores
    GET    /v1/vector_stores/{vector_store_id}
    POST   /v1/vector_stores/{vector_store_id}
    DELETE /v1/vector_stores/{vector_store_id}
    POST   /v1/vector_stores/{vector_store_id}/search

Vector store files:
    POST   /v1/vector_stores/{vector_store_id}/files
    GET    /v1/vector_stores/{vector_store_id}/files
    GET    /v1/vector_stores/{vector_store_id}/files/{file_id}
    GET    /v1/vector_stores/{vector_store_id}/files/{file_id}/content
    POST   /v1/vector_stores/{vector_store_id}/files/{file_id}
    DELETE /v1/vector_stores/{vector_store_id}/files/{file_id}

Vector store file batches:
    POST   /v1/vector_stores/{vector_store_id}/file_batches
    GET    /v1/vector_stores/{vector_store_id}/file_batches/{batch_id}
    POST   /v1/vector_stores/{vector_store_id}/file_batches/{batch_id}/cancel
    GET    /v1/vector_stores/{vector_store_id}/file_batches/{batch_id}/files
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from merlin.http_client import MerlinHTTPClient

JSON = Dict[str, Any]


# ───────────────────────────────────────────────────────────────
# Core dataclasses
# ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class VectorStoreFileCounts:
    """
    File count stats for a vector store or file batch.
    """

    in_progress: int = 0
    completed: int = 0
    failed: int = 0
    cancelled: int = 0
    total: int = 0

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "VectorStoreFileCounts":
        return cls(
            in_progress=int(d.get("in_progress", 0)),
            completed=int(d.get("completed", 0)),
            failed=int(d.get("failed", 0)),
            cancelled=int(d.get("cancelled", 0)),
            total=int(d.get("total", 0)),
        )


@dataclass(frozen=True)
class VectorStore:
    """
    A vector store is a collection of processed files usable by `file_search`.
    """

    id: str
    created_at: int
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    usage_bytes: Optional[int] = None
    bytes: Optional[int] = None
    last_active_at: Optional[int] = None
    last_used_at: Optional[int] = None
    expires_at: Optional[int] = None
    expires_after: Optional[JSON] = None
    file_counts: Optional[VectorStoreFileCounts] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw: JSON = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "VectorStore":
        file_counts = d.get("file_counts")
        return cls(
            id=d["id"],
            created_at=int(d["created_at"]),
            name=d.get("name"),
            description=d.get("description"),
            status=d.get("status"),
            usage_bytes=d.get("usage_bytes"),
            bytes=d.get("bytes"),
            last_active_at=d.get("last_active_at"),
            last_used_at=d.get("last_used_at"),
            expires_at=d.get("expires_at"),
            expires_after=d.get("expires_after"),
            file_counts=(
                VectorStoreFileCounts.from_dict(file_counts)
                if isinstance(file_counts, Mapping)
                else None
            ),
            metadata=dict(d.get("metadata", {})),
            raw=dict(d),
        )


@dataclass(frozen=True)
class VectorStoreFileLastError:
    code: Optional[str]
    message: Optional[str]
    raw: JSON = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "VectorStoreFileLastError":
        return cls(
            code=d.get("code"),
            message=d.get("message"),
            raw=dict(d),
        )


@dataclass(frozen=True)
class VectorStoreFile:
    """
    A file attached to a vector store.
    """

    id: str
    vector_store_id: str
    created_at: int
    status: Optional[str] = None
    usage_bytes: Optional[int] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    chunking_strategy: Optional[JSON] = None
    last_error: Optional[VectorStoreFileLastError] = None
    raw: JSON = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "VectorStoreFile":
        last_error = d.get("last_error")
        return cls(
            id=d["id"],
            vector_store_id=d["vector_store_id"],
            created_at=int(d["created_at"]),
            status=d.get("status"),
            usage_bytes=d.get("usage_bytes"),
            attributes=dict(d.get("attributes", {})),
            chunking_strategy=d.get("chunking_strategy"),
            last_error=(
                VectorStoreFileLastError.from_dict(last_error)
                if isinstance(last_error, Mapping) and last_error
                else None
            ),
            raw=dict(d),
        )


@dataclass(frozen=True)
class VectorStoreFileBatch:
    """
    A batch operation to add multiple files to a vector store.
    """

    id: str
    vector_store_id: str
    created_at: int
    status: str
    file_counts: VectorStoreFileCounts
    raw: JSON = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "VectorStoreFileBatch":
        return cls(
            id=d["id"],
            vector_store_id=d["vector_store_id"],
            created_at=int(d["created_at"]),
            status=d["status"],
            file_counts=VectorStoreFileCounts.from_dict(d.get("file_counts", {})),
            raw=dict(d),
        )


# ───────────────────────────────────────────────────────────────
# Search result dataclasses
# ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class VectorStoreSearchContent:
    """
    A single content chunk in a search result.
    """

    type: str
    text: Optional[str] = None
    raw: JSON = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "VectorStoreSearchContent":
        return cls(
            type=d.get("type", "text"),
            text=d.get("text"),
            raw=dict(d),
        )


@dataclass(frozen=True)
class VectorStoreSearchResult:
    """
    One search hit from a vector store.
    """

    file_id: str
    filename: Optional[str]
    score: float
    attributes: Dict[str, Any]
    content: List[VectorStoreSearchContent]
    raw: JSON = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "VectorStoreSearchResult":
        return cls(
            file_id=d["file_id"],
            filename=d.get("filename"),
            score=float(d.get("score", 0.0)),
            attributes=dict(d.get("attributes", {})),
            content=[
                VectorStoreSearchContent.from_dict(c)
                for c in d.get("content", [])
            ],
            raw=dict(d),
        )


@dataclass(frozen=True)
class VectorStoreSearchResultsPage:
    """
    A single page of search results from a vector store.
    """

    object: str
    search_query: str
    data: List[VectorStoreSearchResult]
    has_more: bool
    next_page: Optional[JSON]
    raw: JSON = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "VectorStoreSearchResultsPage":
        return cls(
            object=d["object"],
            search_query=d.get("search_query", ""),
            data=[
                VectorStoreSearchResult.from_dict(item)
                for item in d.get("data", [])
            ],
            has_more=bool(d.get("has_more", False)),
            next_page=d.get("next_page"),
            raw=dict(d),
        )


# ───────────────────────────────────────────────────────────────
# List wrapper
# ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class VectorStoreList:
    """
    Generic list wrapper for vector store listings.
    """

    data: List[VectorStore]
    first_id: Optional[str]
    last_id: Optional[str]
    has_more: bool
    raw: JSON = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "VectorStoreList":
        return cls(
            data=[VectorStore.from_dict(item) for item in d.get("data", [])],
            first_id=d.get("first_id"),
            last_id=d.get("last_id"),
            has_more=bool(d.get("has_more", False)),
            raw=dict(d),
        )


@dataclass(frozen=True)
class VectorStoreFileList:
    data: List[VectorStoreFile]
    first_id: Optional[str]
    last_id: Optional[str]
    has_more: bool
    raw: JSON = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "VectorStoreFileList":
        return cls(
            data=[VectorStoreFile.from_dict(item) for item in d.get("data", [])],
            first_id=d.get("first_id"),
            last_id=d.get("last_id"),
            has_more=bool(d.get("has_more", False)),
            raw=dict(d),
        )


# ───────────────────────────────────────────────────────────────
# Mixin
# ───────────────────────────────────────────────────────────────

class VectorStoresMixin:
    """
    High-level wrapper for Vector Store APIs.

    Typical workflow:

        # 1. Create a vector store
        vs = client.create_vector_store(
            name="Support FAQ",
            description="FAQ docs",
        )

        # 2. Attach existing uploaded files
        client.add_file_to_vector_store(vs.id, file_id="file-abc123")

        # 3. Or add multiple via a batch
        batch = client.create_vector_store_file_batch(
            vs.id,
            file_ids=["file-abc123", "file-abc456"],
        )

        # 4. Run a semantic search
        results = client.search_vector_store(vs.id, query="What is the return policy?")
        for hit in results.data:
            print(hit.score, hit.filename, hit.content[0].text)
    """

    _http: MerlinHTTPClient

    # ───────────── Vector stores ─────────────

    def create_vector_store(
        self,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        file_ids: Optional[Sequence[str]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        chunking_strategy: Optional[JSON] = None,
        expires_after: Optional[JSON] = None,
    ) -> VectorStore:
        """
        Create a vector store.

        Args:
            name: Optional human-friendly name.
            description: Optional description of the store's purpose.
            file_ids: Optional list of existing File IDs to ingest immediately.
            metadata: Optional metadata map (<=16 key/value pairs).
            chunking_strategy: Optional chunking strategy spec for auto/static.
            expires_after: Optional expiration policy object.
        """
        payload: JSON = {}
        if name is not None:
            payload["name"] = name
        if description is not None:
            payload["description"] = description
        if file_ids:
            payload["file_ids"] = list(file_ids)
        if metadata:
            payload["metadata"] = dict(metadata)
        if chunking_strategy is not None:
            payload["chunking_strategy"] = chunking_strategy
        if expires_after is not None:
            payload["expires_after"] = expires_after

        resp = self._http.post("/v1/vector_stores", json=payload, expect_json=True)
        return VectorStore.from_dict(resp)

    def list_vector_stores(
        self,
        *,
        limit: int = 20,
        after: Optional[str] = None,
        before: Optional[str] = None,
        order: Optional[str] = None,
    ) -> VectorStoreList:
        """
        List vector stores (paged).
        """
        params: JSON = {"limit": limit}
        if after is not None:
            params["after"] = after
        if before is not None:
            params["before"] = before
        if order is not None:
            params["order"] = order

        resp = self._http.get("/v1/vector_stores", params=params, expect_json=True)
        return VectorStoreList.from_dict(resp)

    def get_vector_store(self, vector_store_id: str) -> VectorStore:
        """
        Retrieve a single vector store by ID.
        """
        resp = self._http.get(
            f"/v1/vector_stores/{vector_store_id}", expect_json=True
        )
        return VectorStore.from_dict(resp)

    def update_vector_store(
        self,
        vector_store_id: str,
        *,
        name: Optional[Optional[str]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        expires_after: Optional[JSON] = None,
    ) -> VectorStore:
        """
        Modify a vector store.

        Args:
            name:
                - string: set name
                - None explicitly: clear the name
                - omit argument: leave unchanged
            metadata: New metadata map (replaces previous).
            expires_after: New expiration policy or None to clear.
        """
        payload: JSON = {}
        if name is not None or name is None:
            # We treat explicit None as "clear name".
            payload["name"] = name
        if metadata is not None:
            payload["metadata"] = dict(metadata)
        if expires_after is not None or expires_after is None:
            payload["expires_after"] = expires_after

        resp = self._http.post(
            f"/v1/vector_stores/{vector_store_id}", json=payload, expect_json=True
        )
        return VectorStore.from_dict(resp)

    def delete_vector_store(self, vector_store_id: str) -> bool:
        """
        Delete a vector store. Returns True if deletion was acknowledged.
        """
        resp = self._http.delete(
            f"/v1/vector_stores/{vector_store_id}", expect_json=True
        )
        # Spec: { id, object: "vector_store.deleted", deleted: true }
        return bool(resp.get("deleted"))

    def search_vector_store(
        self,
        vector_store_id: str,
        *,
        query: str,
        filters: Optional[JSON] = None,
        max_num_results: int = 10,
        rewrite_query: bool = False,
        ranking_options: Optional[JSON] = None,
    ) -> VectorStoreSearchResultsPage:
        """
        Run a semantic search over a vector store.

        Args:
            query: Natural language query string.
            filters: Optional file-attribute filter object.
            max_num_results: Max results to return (1–50).
            rewrite_query: Whether to let the API rewrite the query.
            ranking_options: Optional ranking options object.
        """
        payload: JSON = {
            "query": query,
            "max_num_results": max_num_results,
            "rewrite_query": rewrite_query,
        }
        if filters is not None:
            payload["filters"] = filters
        if ranking_options is not None:
            payload["ranking_options"] = ranking_options

        resp = self._http.post(
            f"/v1/vector_stores/{vector_store_id}/search",
            json=payload,
            expect_json=True,
        )
        return VectorStoreSearchResultsPage.from_dict(resp)

    # ───────────── Vector store files ─────────────

    def add_file_to_vector_store(
        self,
        vector_store_id: str,
        file_id: str,
        *,
        attributes: Optional[Mapping[str, Any]] = None,
        chunking_strategy: Optional[JSON] = None,
    ) -> VectorStoreFile:
        """
        Attach a File to a vector store.

        Args:
            attributes: Optional file-level attributes.
            chunking_strategy: Optional per-file chunking strategy.
        """
        payload: JSON = {"file_id": file_id}
        if attributes is not None:
            payload["attributes"] = dict(attributes)
        if chunking_strategy is not None:
            payload["chunking_strategy"] = chunking_strategy

        resp = self._http.post(
            f"/v1/vector_stores/{vector_store_id}/files",
            json=payload,
            expect_json=True,
        )
        return VectorStoreFile.from_dict(resp)

    def list_vector_store_files(
        self,
        vector_store_id: str,
        *,
        limit: int = 20,
        after: Optional[str] = None,
        before: Optional[str] = None,
        order: Optional[str] = None,
        status_filter: Optional[str] = None,
    ) -> VectorStoreFileList:
        """
        List files attached to a vector store.
        """
        params: JSON = {"limit": limit}
        if after is not None:
            params["after"] = after
        if before is not None:
            params["before"] = before
        if order is not None:
            params["order"] = order
        if status_filter is not None:
            params["filter"] = status_filter

        resp = self._http.get(
            f"/v1/vector_stores/{vector_store_id}/files",
            params=params,
            expect_json=True,
        )
        return VectorStoreFileList.from_dict(resp)

    def get_vector_store_file(
        self,
        vector_store_id: str,
        file_id: str,
    ) -> VectorStoreFile:
        """
        Retrieve a vector store file by ID.
        """
        resp = self._http.get(
            f"/v1/vector_stores/{vector_store_id}/files/{file_id}",
            expect_json=True,
        )
        return VectorStoreFile.from_dict(resp)

    def get_vector_store_file_content(
        self,
        vector_store_id: str,
        file_id: str,
    ) -> JSON:
        """
        Retrieve the parsed contents of a vector store file.

        Returns:
            A dict containing:
              - file_id
              - filename
              - attributes
              - content: list of {type, text, ...}
        """
        return self._http.get(
            f"/v1/vector_stores/{vector_store_id}/files/{file_id}/content",
            expect_json=True,
        )

    def update_vector_store_file_attributes(
        self,
        vector_store_id: str,
        file_id: str,
        *,
        attributes: Mapping[str, Any],
    ) -> VectorStoreFile:
        """
        Update attributes on a vector store file.
        """
        payload: JSON = {"attributes": dict(attributes)}
        resp = self._http.post(
            f"/v1/vector_stores/{vector_store_id}/files/{file_id}",
            json=payload,
            expect_json=True,
        )
        return VectorStoreFile.from_dict(resp)

    def delete_vector_store_file(
        self,
        vector_store_id: str,
        file_id: str,
    ) -> bool:
        """
        Remove a file from a vector store (does not delete the underlying File).
        """
        resp = self._http.delete(
            f"/v1/vector_stores/{vector_store_id}/files/{file_id}",
            expect_json=True,
        )
        return bool(resp.get("deleted"))

    # ───────────── Vector store file batches ─────────────

    def create_vector_store_file_batch(
        self,
        vector_store_id: str,
        *,
        file_ids: Optional[Sequence[str]] = None,
        files: Optional[Sequence[JSON]] = None,
        attributes: Optional[Mapping[str, Any]] = None,
        chunking_strategy: Optional[JSON] = None,
    ) -> VectorStoreFileBatch:
        """
        Create a vector store file batch.

        Args:
            file_ids:
                Simple list of File IDs; optional global attributes/chunking_strategy
                will be applied to all.
            files:
                List of objects with per-file overrides:
                    {"file_id": "...", "attributes": {...}, "chunking_strategy": {...}}
                Mutually exclusive with file_ids.
            attributes:
                Global attributes (only used if file_ids is provided).
            chunking_strategy:
                Global chunking strategy (only used if file_ids is provided).
        """
        payload: JSON = {}

        if files is not None and file_ids is not None:
            raise ValueError("Provide either file_ids or files, not both.")

        if files is not None:
            payload["files"] = list(files)
        elif file_ids is not None:
            payload["file_ids"] = list(file_ids)
            if attributes is not None:
                payload["attributes"] = dict(attributes)
            if chunking_strategy is not None:
                payload["chunking_strategy"] = chunking_strategy

        resp = self._http.post(
            f"/v1/vector_stores/{vector_store_id}/file_batches",
            json=payload,
            expect_json=True,
        )
        return VectorStoreFileBatch.from_dict(resp)

    def get_vector_store_file_batch(
        self,
        vector_store_id: str,
        batch_id: str,
    ) -> VectorStoreFileBatch:
        """
        Retrieve a vector store file batch.
        """
        resp = self._http.get(
            f"/v1/vector_stores/{vector_store_id}/file_batches/{batch_id}",
            expect_json=True,
        )
        return VectorStoreFileBatch.from_dict(resp)

    def cancel_vector_store_file_batch(
        self,
        vector_store_id: str,
        batch_id: str,
    ) -> VectorStoreFileBatch:
        """
        Cancel a vector store file batch (best-effort).
        """
        resp = self._http.post(
            f"/v1/vector_stores/{vector_store_id}/file_batches/{batch_id}/cancel",
            json={},
            expect_json=True,
        )
        return VectorStoreFileBatch.from_dict(resp)

    def list_vector_store_files_in_batch(
        self,
        vector_store_id: str,
        batch_id: str,
        *,
        limit: int = 20,
        after: Optional[str] = None,
        before: Optional[str] = None,
        order: Optional[str] = None,
        status_filter: Optional[str] = None,
    ) -> VectorStoreFileList:
        """
        List vector store files associated with a given batch.
        """
        params: JSON = {"limit": limit}
        if after is not None:
            params["after"] = after
        if before is not None:
            params["before"] = before
        if order is not None:
            params["order"] = order
        if status_filter is not None:
            params["filter"] = status_filter

        resp = self._http.get(
            f"/v1/vector_stores/{vector_store_id}/file_batches/{batch_id}/files",
            params=params,
            expect_json=True,
        )
        return VectorStoreFileList.from_dict(resp)


__all__ = [
    "VectorStore",
    "VectorStoreFile",
    "VectorStoreFileBatch",
    "VectorStoreFileCounts",
    "VectorStoreSearchContent",
    "VectorStoreSearchResult",
    "VectorStoreSearchResultsPage",
    "VectorStoreList",
    "VectorStoreFileList",
    "VectorStoresMixin",
]
