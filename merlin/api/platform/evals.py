"""
Evals API
=========

High-level client for the `/v1/evals` family of endpoints.

Docs section: "Evals"

Core surfaces:

- POST   /v1/evals                             → create eval
- GET    /v1/evals/{eval_id}                   → get eval
- POST   /v1/evals/{eval_id}                   → update eval
- DELETE /v1/evals/{eval_id}                   → delete eval
- GET    /v1/evals                             → list evals

- GET    /v1/evals/{eval_id}/runs              → list runs
- POST   /v1/evals/{eval_id}/runs              → create run
- GET    /v1/evals/{eval_id}/runs/{run_id}     → get run
- POST   /v1/evals/{eval_id}/runs/{run_id}/cancel
                                              → cancel run
- DELETE /v1/evals/{eval_id}/runs/{run_id}     → delete run

- GET    /v1/evals/{eval_id}/runs/{run_id}/output_items
                                              → list output items
- GET    /v1/evals/{eval_id}/runs/{run_id}/output_items/{output_item_id}
                                              → get one output item
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Type, TypeVar, Generic

from merlin.http_client import MerlinHTTPClient


JSON = Dict[str, Any]
T = TypeVar("T")


# ───────────────────────────────────────────────────────────────
# Small generic list wrapper
# ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ListPage(Generic[T]):  # type: ignore[name-defined]
    """
    Generic wrapper for list endpoints:

        {
          "object": "list",
          "data": [ ... ],
          "first_id": "...",
          "last_id": "...",
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
# Eval object
# ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EvalObject:
    """
    Represents an Eval configuration.
    """
    id: str
    name: Optional[str]
    created_at: Optional[int]
    data_source_config: JSON
    testing_criteria: List[JSON]
    metadata: Dict[str, Any]
    raw: JSON = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvalObject":
        return cls(
            id=str(data.get("id", "")),
            name=data.get("name"),
            created_at=data.get("created_at"),
            data_source_config=dict(data.get("data_source_config") or {}),
            testing_criteria=[
                dict(x) for x in (data.get("testing_criteria") or []) if isinstance(x, Mapping)
            ],
            metadata=dict(data.get("metadata") or {}),
            raw=dict(data),
        )


# ───────────────────────────────────────────────────────────────
# Eval run & usage / stats
# ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EvalRunResultCounts:
    total: int
    errored: int
    failed: int
    passed: int
    raw: JSON = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvalRunResultCounts":
        def _i(key: str) -> int:
            try:
                return int(data.get(key, 0))
            except (TypeError, ValueError):
                return 0

        return cls(
            total=_i("total"),
            errored=_i("errored"),
            failed=_i("failed"),
            passed=_i("passed"),
            raw=dict(data),
        )


@dataclass(frozen=True)
class EvalRunModelUsage:
    """
    Per-model usage entry in `per_model_usage`.
    """

    model_name: str
    invocation_count: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached_tokens: int
    raw: JSON = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvalRunModelUsage":
        def _i(key: str) -> int:
            try:
                return int(data.get(key, 0))
            except (TypeError, ValueError):
                return 0

        return cls(
            model_name=str(data.get("model_name", "")),
            invocation_count=_i("invocation_count"),
            prompt_tokens=_i("prompt_tokens"),
            completion_tokens=_i("completion_tokens"),
            total_tokens=_i("total_tokens"),
            cached_tokens=_i("cached_tokens"),
            raw=dict(data),
        )


@dataclass(frozen=True)
class EvalRunTestingCriteriaResult:
    """
    Per-testing-criteria summary.
    """
    testing_criteria: str
    passed: int
    failed: int
    raw: JSON = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvalRunTestingCriteriaResult":
        def _i(key: str) -> int:
            try:
                return int(data.get(key, 0))
            except (TypeError, ValueError):
                return 0

        return cls(
            testing_criteria=str(data.get("testing_criteria", "")),
            passed=_i("passed"),
            failed=_i("failed"),
            raw=dict(data),
        )


@dataclass(frozen=True)
class EvalRun:
    """
    Eval run object.
    """
    id: str
    eval_id: str
    status: str
    model: Optional[str]
    name: Optional[str]
    created_at: Optional[int]
    report_url: Optional[str]
    result_counts: Optional[EvalRunResultCounts]
    per_model_usage: List[EvalRunModelUsage]
    per_testing_criteria_results: List[EvalRunTestingCriteriaResult]
    data_source: JSON
    error: Optional[JSON]
    metadata: Dict[str, Any]
    raw: JSON = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvalRun":
        rc_raw = data.get("result_counts")
        rc = EvalRunResultCounts.from_dict(rc_raw) if isinstance(rc_raw, Mapping) else None

        pmu_raw = data.get("per_model_usage") or []
        pmu = [
            EvalRunModelUsage.from_dict(x)
            for x in pmu_raw
            if isinstance(x, Mapping)
        ]

        tcr_raw = data.get("per_testing_criteria_results") or []
        tcr = [
            EvalRunTestingCriteriaResult.from_dict(x)
            for x in tcr_raw
            if isinstance(x, Mapping)
        ]

        err_raw = data.get("error")
        err = dict(err_raw) if isinstance(err_raw, Mapping) else None

        return cls(
            id=str(data.get("id", "")),
            eval_id=str(data.get("eval_id", "")),
            status=str(data.get("status", "")),
            model=data.get("model"),
            name=data.get("name"),
            created_at=data.get("created_at"),
            report_url=data.get("report_url"),
            result_counts=rc,
            per_model_usage=pmu,
            per_testing_criteria_results=tcr,
            data_source=dict(data.get("data_source") or {}),
            error=err,
            metadata=dict(data.get("metadata") or {}),
            raw=dict(data),
        )


# ───────────────────────────────────────────────────────────────
# Eval run output item
# ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EvalRunOutputItemResult:
    """
    One grader's result for an output item.
    """
    name: str
    passed: bool
    score: Optional[float]
    sample: Optional[JSON]
    raw: JSON = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvalRunOutputItemResult":
        score = data.get("score")
        try:
            score_val: Optional[float] = float(score) if score is not None else None
        except (TypeError, ValueError):
            score_val = None

        sample_raw = data.get("sample")
        sample = dict(sample_raw) if isinstance(sample_raw, Mapping) else None

        return cls(
            name=str(data.get("name", "")),
            passed=bool(data.get("passed", False)),
            score=score_val,
            sample=sample,
            raw=dict(data),
        )


@dataclass(frozen=True)
class EvalRunOutputItem:
    """
    Output item object.
    """
    id: str
    run_id: str
    eval_id: str
    created_at: Optional[int]
    status: str
    datasource_item_id: Optional[int]
    datasource_item: JSON
    results: List[EvalRunOutputItemResult]
    sample: JSON
    raw: JSON = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvalRunOutputItem":
        results_raw = data.get("results") or []
        results = [
            EvalRunOutputItemResult.from_dict(x)
            for x in results_raw
            if isinstance(x, Mapping)
        ]

        ds_item = dict(data.get("datasource_item") or {})
        sample = dict(data.get("sample") or {})

        ds_item_id = data.get("datasource_item_id")
        try:
            ds_item_id_val: Optional[int] = int(ds_item_id) if ds_item_id is not None else None
        except (TypeError, ValueError):
            ds_item_id_val = None

        return cls(
            id=str(data.get("id", "")),
            run_id=str(data.get("run_id", "")),
            eval_id=str(data.get("eval_id", "")),
            created_at=data.get("created_at"),
            status=str(data.get("status", "")),
            datasource_item_id=ds_item_id_val,
            datasource_item=ds_item,
            results=results,
            sample=sample,
            raw=dict(data),
        )


# ───────────────────────────────────────────────────────────────
# Deletion result objects
# ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EvalDeletion:
    object: str
    deleted: bool
    eval_id: Optional[str]
    raw: JSON = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvalDeletion":
        return cls(
            object=str(data.get("object", "")),
            deleted=bool(data.get("deleted", False)),
            eval_id=data.get("eval_id") or data.get("id"),
            raw=dict(data),
        )


@dataclass(frozen=True)
class EvalRunDeletion:
    object: str
    deleted: bool
    run_id: Optional[str]
    raw: JSON = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvalRunDeletion":
        return cls(
            object=str(data.get("object", "")),
            deleted=bool(data.get("deleted", False)),
            run_id=data.get("run_id") or data.get("id"),
            raw=dict(data),
        )


# ───────────────────────────────────────────────────────────────
# Client mixin
# ───────────────────────────────────────────────────────────────


class EvalsMixin:
    """
    High-level methods for the Evals API.

    Assumptions:
        - The consuming client defines `self._http` as a MerlinHTTPClient.
    """

    _http: MerlinHTTPClient  # for type checkers

    # ── Eval objects ────────────────────────────────────────────

    def create_eval(
        self,
        *,
        data_source_config: JSON,
        testing_criteria: Sequence[JSON],
        name: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        **extra: Any,
    ) -> EvalObject:
        """
        POST /v1/evals

        Create the structure of an evaluation that can be used for runs.
        """
        payload: JSON = {
            "data_source_config": data_source_config,
            "testing_criteria": list(testing_criteria),
        }
        if name is not None:
            payload["name"] = name
        if metadata is not None:
            payload["metadata"] = dict(metadata)

        payload.update(extra)

        resp = self._http.post(
            "/v1/evals",
            json=payload,
            expect_json=True,
        )
        return EvalObject.from_dict(resp)

    def get_eval(self, eval_id: str) -> EvalObject:
        """
        GET /v1/evals/{eval_id}
        """
        resp = self._http.get(
            f"/v1/evals/{eval_id}",
            expect_json=True,
        )
        return EvalObject.from_dict(resp)

    def update_eval(
        self,
        eval_id: str,
        *,
        name: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        **extra: Any,
    ) -> EvalObject:
        """
        POST /v1/evals/{eval_id}

        Update name / metadata for an eval.
        """
        payload: JSON = {}
        if name is not None:
            payload["name"] = name
        if metadata is not None:
            payload["metadata"] = dict(metadata)
        payload.update(extra)

        resp = self._http.post(
            f"/v1/evals/{eval_id}",
            json=payload,
            expect_json=True,
        )
        return EvalObject.from_dict(resp)

    def delete_eval(self, eval_id: str) -> EvalDeletion:
        """
        DELETE /v1/evals/{eval_id}
        """
        resp = self._http.delete(
            f"/v1/evals/{eval_id}",
            expect_json=True,
        )
        return EvalDeletion.from_dict(resp)

    def list_evals(
        self,
        *,
        limit: Optional[int] = None,
        after: Optional[str] = None,
        order: Optional[str] = None,
        order_by: Optional[str] = None,
    ) -> ListPage[EvalObject]:
        """
        GET /v1/evals

        Pagination helpers (`after`, `limit`, `order`, `order_by`) are passed
        straight through.
        """
        params: Dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if after is not None:
            params["after"] = after
        if order is not None:
            params["order"] = order
        if order_by is not None:
            params["order_by"] = order_by

        resp = self._http.get(
            "/v1/evals",
            params=params,
            expect_json=True,
        )
        return ListPage.from_dict(resp, EvalObject)

    # ── Eval runs ───────────────────────────────────────────────

    def list_eval_runs(
        self,
        eval_id: str,
        *,
        limit: Optional[int] = None,
        after: Optional[str] = None,
        order: Optional[str] = None,
        status: Optional[str] = None,
    ) -> ListPage[EvalRun]:
        """
        GET /v1/evals/{eval_id}/runs
        """
        params: Dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if after is not None:
            params["after"] = after
        if order is not None:
            params["order"] = order
        if status is not None:
            params["status"] = status

        resp = self._http.get(
            f"/v1/evals/{eval_id}/runs",
            params=params,
            expect_json=True,
        )
        return ListPage.from_dict(resp, EvalRun)

    def create_eval_run(
        self,
        eval_id: str,
        *,
        data_source: JSON,
        name: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> EvalRun:
        """
        POST /v1/evals/{eval_id}/runs
        """
        payload: JSON = {
            "data_source": data_source,
        }
        if name is not None:
            payload["name"] = name
        if metadata is not None:
            payload["metadata"] = dict(metadata)

        resp = self._http.post(
            f"/v1/evals/{eval_id}/runs",
            json=payload,
            expect_json=True,
        )
        return EvalRun.from_dict(resp)

    def get_eval_run(self, eval_id: str, run_id: str) -> EvalRun:
        """
        GET /v1/evals/{eval_id}/runs/{run_id}
        """
        resp = self._http.get(
            f"/v1/evals/{eval_id}/runs/{run_id}",
            expect_json=True,
        )
        return EvalRun.from_dict(resp)

    def cancel_eval_run(self, eval_id: str, run_id: str) -> EvalRun:
        """
        POST /v1/evals/{eval_id}/runs/{run_id}/cancel
        """
        resp = self._http.post(
            f"/v1/evals/{eval_id}/runs/{run_id}/cancel",
            json={},
            expect_json=True,
        )
        return EvalRun.from_dict(resp)

    def delete_eval_run(self, eval_id: str, run_id: str) -> EvalRunDeletion:
        """
        DELETE /v1/evals/{eval_id}/runs/{run_id}
        """
        resp = self._http.delete(
            f"/v1/evals/{eval_id}/runs/{run_id}",
            expect_json=True,
        )
        return EvalRunDeletion.from_dict(resp)

    # ── Output items ────────────────────────────────────────────

    def list_eval_run_output_items(
        self,
        eval_id: str,
        run_id: str,
        *,
        limit: Optional[int] = None,
        after: Optional[str] = None,
        order: Optional[str] = None,
        status: Optional[str] = None,
    ) -> ListPage[EvalRunOutputItem]:
        """
        GET /v1/evals/{eval_id}/runs/{run_id}/output_items
        """
        params: Dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if after is not None:
            params["after"] = after
        if order is not None:
            params["order"] = order
        if status is not None:
            params["status"] = status

        resp = self._http.get(
            f"/v1/evals/{eval_id}/runs/{run_id}/output_items",
            params=params,
            expect_json=True,
        )
        return ListPage.from_dict(resp, EvalRunOutputItem)

    def get_eval_run_output_item(
        self,
        eval_id: str,
        run_id: str,
        output_item_id: str,
    ) -> EvalRunOutputItem:
        """
        GET /v1/evals/{eval_id}/runs/{run_id}/output_items/{output_item_id}
        """
        resp = self._http.get(
            f"/v1/evals/{eval_id}/runs/{run_id}/output_items/{output_item_id}",
            expect_json=True,
        )
        return EvalRunOutputItem.from_dict(resp)


__all__ = [
    "EvalObject",
    "EvalRun",
    "EvalRunOutputItem",
    "EvalRunResultCounts",
    "EvalRunModelUsage",
    "EvalRunTestingCriteriaResult",
    "EvalDeletion",
    "EvalRunDeletion",
    "ListPage",
    "EvalsMixin",
]
