"""
Fine-tuning API
===============

High-level client for the `/v1/fine_tuning` family of endpoints.

Docs section: "Fine-tuning"

Core surfaces:

Jobs
----
- POST   /v1/fine_tuning/jobs                      → create job
- GET    /v1/fine_tuning/jobs                      → list jobs
- GET    /v1/fine_tuning/jobs/{job_id}             → retrieve job
- POST   /v1/fine_tuning/jobs/{job_id}/cancel      → cancel job
- POST   /v1/fine_tuning/jobs/{job_id}/pause       → pause job
- POST   /v1/fine_tuning/jobs/{job_id}/resume      → resume job

Events & checkpoints
--------------------
- GET    /v1/fine_tuning/jobs/{job_id}/events      → list job events
- GET    /v1/fine_tuning/jobs/{job_id}/checkpoints → list checkpoints

Checkpoint permissions (admin key required)
-------------------------------------------
- GET    /v1/fine_tuning/checkpoints/{ckpt}/permissions
                                                  → list permissions
- POST   /v1/fine_tuning/checkpoints/{ckpt}/permissions
                                                  → create permissions
- DELETE /v1/fine_tuning/checkpoints/{ckpt}/permissions/{perm_id}
                                                  → delete permission
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Mapping,
    Optional,
    Sequence,
    Type,
    TypeVar,
)

from merlin.http_client import MerlinHTTPClient


JSON = Dict[str, Any]
T = TypeVar("T")


# ───────────────────────────────────────────────────────────────
# Generic list wrapper (same shape as other list endpoints)
# ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ListPage(Generic[T]):
    """
    Wrapper for OpenAI-style list responses:

        {
          "object": "list",
          "data": [ ... ],
          "first_id": "...",   # sometimes present
          "last_id":  "...",   # sometimes present
          "has_more": true
        }
    """

    data: List[T]
    first_id: Optional[str]
    last_id: Optional[str]
    has_more: bool
    raw: JSON = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        item_type: Type[T],
        *,
        item_key: str = "data",
    ) -> "ListPage[T]":
        items_raw = data.get(item_key) or []
        items: List[T] = []
        for x in items_raw:
            if isinstance(x, Mapping) and hasattr(item_type, "from_dict"):
                items.append(item_type.from_dict(x))  # type: ignore[arg-type]
        return cls(
            data=items,
            first_id=data.get("first_id"),
            last_id=data.get("last_id"),
            has_more=bool(data.get("has_more", False)),
            raw=dict(data),
        )


# ───────────────────────────────────────────────────────────────
# Fine-tuning job object
# ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FineTuningJobMethod:
    """
    The `method` field in a fine-tuning job, e.g.:

        {
          "type": "supervised",
          "supervised": {
            "hyperparameters": {
              "n_epochs": 4,
              "batch_size": 1,
              "learning_rate_multiplier": 1.0
            }
          }
        }
    """

    type: str
    config: JSON
    raw: JSON = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FineTuningJobMethod":
        type_ = str(data.get("type", ""))
        config = dict(data)
        # keep `type` but also keep the entire raw for consumers to inspect
        return cls(type=type_, config=config, raw=dict(data))


@dataclass(frozen=True)
class FineTuningJob:
    """
    Fine-tuning job object (`fine_tuning.job`).

    Docs shape (simplified):

        {
          "object": "fine_tuning.job",
          "id": "ftjob-abc123",
          "model": "gpt-4o-mini-2024-07-18",
          "created_at": 1721764800,
          "finished_at": 1721765000,
          "fine_tuned_model": "ft:...",
          "organization_id": "org-123",
          "result_files": ["file-abc123"],
          "status": "succeeded",
          "validation_file": "file-...",
          "training_file": "file-...",
          "hyperparameters": { ... },
          "trained_tokens": 5768,
          "integrations": [ ... ],
          "seed": 0,
          "estimated_finish": 0,
          "method": { ... },
          "metadata": { ... },
          "error": { ... } | null
        }
    """

    id: str
    model: str
    object: str
    status: str
    created_at: int
    finished_at: Optional[int]
    fine_tuned_model: Optional[str]
    organization_id: Optional[str]
    result_files: List[str]
    validation_file: Optional[str]
    training_file: Optional[str]
    hyperparameters: JSON
    trained_tokens: Optional[int]
    integrations: List[JSON]
    seed: Optional[int]
    estimated_finish: Optional[int]
    method: Optional[FineTuningJobMethod]
    metadata: Dict[str, Any]
    error: Optional[JSON]
    raw: JSON = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FineTuningJob":
        method_raw = data.get("method")
        method = (
            FineTuningJobMethod.from_dict(method_raw)
            if isinstance(method_raw, Mapping)
            else None
        )

        def _opt_int(key: str) -> Optional[int]:
            v = data.get(key)
            if v is None:
                return None
            try:
                return int(v)
            except (TypeError, ValueError):
                return None

        return cls(
            id=str(data.get("id", "")),
            model=str(data.get("model", "")),
            object=str(data.get("object", "")),
            status=str(data.get("status", "")),
            created_at=int(data.get("created_at", 0)),
            finished_at=_opt_int("finished_at"),
            fine_tuned_model=data.get("fine_tuned_model"),
            organization_id=data.get("organization_id"),
            result_files=[str(f) for f in (data.get("result_files") or [])],
            validation_file=data.get("validation_file"),
            training_file=data.get("training_file"),
            hyperparameters=dict(data.get("hyperparameters") or {}),
            trained_tokens=_opt_int("trained_tokens"),
            integrations=[
                dict(x)
                for x in (data.get("integrations") or [])
                if isinstance(x, Mapping)
            ],
            seed=_opt_int("seed"),
            estimated_finish=_opt_int("estimated_finish"),
            method=method,
            metadata=dict(data.get("metadata") or {}),
            error=dict(data.get("error") or {})
            if isinstance(data.get("error"), Mapping)
            else None,
            raw=dict(data),
        )


# ───────────────────────────────────────────────────────────────
# Fine-tuning events
# ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FineTuningJobEvent:
    """
    Fine-tuning job event object (`fine_tuning.job.event`).

        {
          "object": "fine_tuning.job.event",
          "id": "ft-event-...",
          "created_at": 1721764800,
          "level": "info",
          "message": "Created fine-tuning job",
          "data": {},
          "type": "message"
        }
    """

    id: str
    object: str
    created_at: int
    level: str
    message: str
    type: str
    data: JSON
    raw: JSON = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FineTuningJobEvent":
        return cls(
            id=str(data.get("id", "")),
            object=str(data.get("object", "")),
            created_at=int(data.get("created_at", 0)),
            level=str(data.get("level", "")),
            message=str(data.get("message", "")),
            type=str(data.get("type", "")),
            data=dict(data.get("data") or {}),
            raw=dict(data),
        )


# ───────────────────────────────────────────────────────────────
# Checkpoints
# ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FineTuningCheckpoint:
    """
    fine_tuning.job.checkpoint object.

        {
          "object": "fine_tuning.job.checkpoint",
          "id": "ftckpt_...",
          "created_at": 1712211699,
          "fine_tuned_model_checkpoint": "ft:gpt-4o-mini-...:ckpt-step-88",
          "fine_tuning_job_id": "ftjob-abc123",
          "metrics": { ... },
          "step_number": 88
        }
    """

    id: str
    object: str
    created_at: int
    fine_tuned_model_checkpoint: str
    fine_tuning_job_id: str
    metrics: JSON
    step_number: int
    raw: JSON = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FineTuningCheckpoint":
        return cls(
            id=str(data.get("id", "")),
            object=str(data.get("object", "")),
            created_at=int(data.get("created_at", 0)),
            fine_tuned_model_checkpoint=str(
                data.get("fine_tuned_model_checkpoint", "")
            ),
            fine_tuning_job_id=str(data.get("fine_tuning_job_id", "")),
            metrics=dict(data.get("metrics") or {}),
            step_number=int(data.get("step_number", 0)),
            raw=dict(data),
        )


# ───────────────────────────────────────────────────────────────
# Checkpoint permissions
# ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CheckpointPermission:
    """
    checkpoint.permission object.

        {
          "object": "checkpoint.permission",
          "id": "cp_zc4Q7M...",
          "created_at": 1712211699,
          "project_id": "proj_...",
          "deleted": true?  # on delete response
        }
    """

    id: str
    object: str
    created_at: int
    project_id: Optional[str]
    deleted: Optional[bool]
    raw: JSON = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CheckpointPermission":
        return cls(
            id=str(data.get("id", "")),
            object=str(data.get("object", "")),
            created_at=int(data.get("created_at", 0)),
            project_id=data.get("project_id"),
            deleted=data.get("deleted"),
            raw=dict(data),
        )


# ───────────────────────────────────────────────────────────────
# Client mixin
# ───────────────────────────────────────────────────────────────


class FineTuningMixin:
    """
    High-level methods for the Fine-tuning API.

    Assumes the consuming client defines:

        self._http: MerlinHTTPClient
    """

    _http: MerlinHTTPClient  # for type-checkers

    # ── Jobs ───────────────────────────────────────────────────

    def create_fine_tuning_job(
        self,
        *,
        model: str,
        training_file: str,
        validation_file: Optional[str] = None,
        method: Optional[JSON] = None,
        seed: Optional[int] = None,
        suffix: Optional[str] = None,
        integrations: Optional[Sequence[JSON]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        hyperparameters: Optional[JSON] = None,  # deprecated but supported
        **extra: Any,
    ) -> FineTuningJob:
        """
        POST /v1/fine_tuning/jobs

        Minimum required: `model`, `training_file`.

        Args:
            model:
                Base model name (e.g. "gpt-4o-mini").
            training_file:
                File ID (purpose="fine-tune", JSONL).
            validation_file:
                Optional validation file ID.
            method:
                Fine-tuning method block (supervised / dpo / reinforcement).
            seed:
                Optional seed for reproducibility.
            suffix:
                Optional suffix for the resulting ft model.
            integrations:
                Optional list of integration configs (e.g., W&B).
            metadata:
                Optional metadata dictionary.
            hyperparameters:
                Deprecated top-level hyperparameters; use `method` instead.
                This is passed through if supplied for backwards compat.
            extra:
                Any future/extra fields you want to send verbatim.
        """
        payload: JSON = {
            "model": model,
            "training_file": training_file,
        }

        if validation_file is not None:
            payload["validation_file"] = validation_file
        if method is not None:
            payload["method"] = method
        if seed is not None:
            payload["seed"] = seed
        if suffix is not None:
            payload["suffix"] = suffix
        if integrations is not None:
            payload["integrations"] = list(integrations)
        if metadata is not None:
            payload["metadata"] = dict(metadata)
        if hyperparameters is not None:
            payload["hyperparameters"] = dict(hyperparameters)

        payload.update(extra)

        resp = self._http.post(
            "/v1/fine_tuning/jobs",
            json=payload,
            expect_json=True,
        )
        return FineTuningJob.from_dict(resp)

    def list_fine_tuning_jobs(
        self,
        *,
        limit: Optional[int] = None,
        after: Optional[str] = None,
        metadata: Optional[Mapping[str, str]] = None,
    ) -> ListPage[FineTuningJob]:
        """
        GET /v1/fine_tuning/jobs

        Args:
            limit:
                Max number of jobs (default 20).
            after:
                Pagination cursor (last job ID from previous page).
            metadata:
                Optional metadata filter. Equivalent to query params
                like `metadata[k]=v`.
        """
        params: Dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if after is not None:
            params["after"] = after
        if metadata is not None:
            # Flatten `metadata[k]=v` into query params
            for k, v in metadata.items():
                params[f"metadata[{k}]"] = v

        resp = self._http.get(
            "/v1/fine_tuning/jobs",
            params=params,
            expect_json=True,
        )
        return ListPage.from_dict(resp, FineTuningJob)

    def get_fine_tuning_job(self, job_id: str) -> FineTuningJob:
        """
        GET /v1/fine_tuning/jobs/{job_id}
        """
        resp = self._http.get(
            f"/v1/fine_tuning/jobs/{job_id}",
            expect_json=True,
        )
        return FineTuningJob.from_dict(resp)

    def cancel_fine_tuning_job(self, job_id: str) -> FineTuningJob:
        """
        POST /v1/fine_tuning/jobs/{job_id}/cancel
        """
        resp = self._http.post(
            f"/v1/fine_tuning/jobs/{job_id}/cancel",
            json={},
            expect_json=True,
        )
        return FineTuningJob.from_dict(resp)

    def pause_fine_tuning_job(self, job_id: str) -> FineTuningJob:
        """
        POST /v1/fine_tuning/jobs/{job_id}/pause
        """
        resp = self._http.post(
            f"/v1/fine_tuning/jobs/{job_id}/pause",
            json={},
            expect_json=True,
        )
        return FineTuningJob.from_dict(resp)

    def resume_fine_tuning_job(self, job_id: str) -> FineTuningJob:
        """
        POST /v1/fine_tuning/jobs/{job_id}/resume
        """
        resp = self._http.post(
            f"/v1/fine_tuning/jobs/{job_id}/resume",
            json={},
            expect_json=True,
        )
        return FineTuningJob.from_dict(resp)

    # ── Events ──────────────────────────────────────────────────

    def list_fine_tuning_events(
        self,
        job_id: str,
        *,
        limit: Optional[int] = None,
        after: Optional[str] = None,
    ) -> ListPage[FineTuningJobEvent]:
        """
        GET /v1/fine_tuning/jobs/{job_id}/events
        """
        params: Dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if after is not None:
            params["after"] = after

        resp = self._http.get(
            f"/v1/fine_tuning/jobs/{job_id}/events",
            params=params,
            expect_json=True,
        )
        return ListPage.from_dict(resp, FineTuningJobEvent)

    # ── Checkpoints ─────────────────────────────────────────────

    def list_fine_tuning_checkpoints(
        self,
        job_id: str,
        *,
        limit: Optional[int] = None,
        after: Optional[str] = None,
    ) -> ListPage[FineTuningCheckpoint]:
        """
        GET /v1/fine_tuning/jobs/{job_id}/checkpoints
        """
        params: Dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if after is not None:
            params["after"] = after

        resp = self._http.get(
            f"/v1/fine_tuning/jobs/{job_id}/checkpoints",
            params=params,
            expect_json=True,
        )
        return ListPage.from_dict(resp, FineTuningCheckpoint)

    # ── Checkpoint permissions (admin key required) ─────────────

    def list_checkpoint_permissions(
        self,
        fine_tuned_model_checkpoint: str,
        *,
        limit: Optional[int] = None,
        after: Optional[str] = None,
        order: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> ListPage[CheckpointPermission]:
        """
        GET /v1/fine_tuning/checkpoints/{ckpt}/permissions

        NOTE: Requires an admin API key.
        """
        params: Dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if after is not None:
            params["after"] = after
        if order is not None:
            params["order"] = order
        if project_id is not None:
            params["project_id"] = project_id

        resp = self._http.get(
            f"/v1/fine_tuning/checkpoints/{fine_tuned_model_checkpoint}/permissions",
            params=params,
            expect_json=True,
        )
        return ListPage.from_dict(resp, CheckpointPermission)

    def create_checkpoint_permissions(
        self,
        fine_tuned_model_checkpoint: str,
        *,
        project_ids: Sequence[str],
    ) -> ListPage[CheckpointPermission]:
        """
        POST /v1/fine_tuning/checkpoints/{ckpt}/permissions

        NOTE: Requires an admin API key.

        Args:
            fine_tuned_model_checkpoint:
                Checkpoint identifier (e.g. "ft:gpt-4o-mini-...:ckpt-step-1000").
            project_ids:
                List of project IDs to grant access to.
        """
        payload: JSON = {
            "project_ids": list(project_ids),
        }
        resp = self._http.post(
            f"/v1/fine_tuning/checkpoints/{fine_tuned_model_checkpoint}/permissions",
            json=payload,
            expect_json=True,
        )
        return ListPage.from_dict(resp, CheckpointPermission)

    def delete_checkpoint_permission(
        self,
        fine_tuned_model_checkpoint: str,
        permission_id: str,
    ) -> CheckpointPermission:
        """
        DELETE /v1/fine_tuning/checkpoints/{ckpt}/permissions/{permission_id}

        NOTE: Requires an admin API key.

        Returns the deleted permission object with `deleted: true`.
        """
        resp = self._http.delete(
            f"/v1/fine_tuning/checkpoints/{fine_tuned_model_checkpoint}/permissions/{permission_id}",
            expect_json=True,
        )
        return CheckpointPermission.from_dict(resp)


__all__ = [
    "FineTuningJob",
    "FineTuningJobEvent",
    "FineTuningCheckpoint",
    "CheckpointPermission",
    "FineTuningJobMethod",
    "ListPage",
    "FineTuningMixin",
]
