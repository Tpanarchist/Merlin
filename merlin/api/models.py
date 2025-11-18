"""
Models API
==========

List and inspect available models, and delete fine-tuned models.

Endpoints
---------
GET    /v1/models           → list models
GET    /v1/models/{model}   → retrieve a model
DELETE /v1/models/{model}   → delete a fine-tuned model
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping

from merlin.http_client import MerlinHTTPClient

JSON = Dict[str, Any]


# ───────────────────────────────────────────────────────────────
# Dataclasses
# ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Model:
    """
    Describes an OpenAI model offering that can be used with the API.
    """

    id: str
    object: str
    created: int
    owned_by: str

    raw: JSON = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Model":
        return cls(
            id=d["id"],
            object=d.get("object", "model"),
            created=int(d["created"]),
            owned_by=d["owned_by"],
            raw=dict(d),
        )


@dataclass(frozen=True)
class ModelDeletion:
    """
    Response when deleting a fine-tuned model.
    """

    id: str
    object: str
    deleted: bool
    raw: JSON = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ModelDeletion":
        return cls(
            id=d["id"],
            object=d.get("object", "model"),
            deleted=bool(d.get("deleted", False)),
            raw=dict(d),
        )


# ───────────────────────────────────────────────────────────────
# ModelsMixin
# ───────────────────────────────────────────────────────────────

class ModelsMixin:
    """
    High-level wrapper for the Models API.

        client.list_models()
        client.get_model("gpt-5.1")
        client.delete_model("ft:gpt-4o-mini:org:suffix:abc123")
    """

    _http: MerlinHTTPClient

    # Core endpoints ---------------------------------------------------------

    def list_models(self) -> List[Model]:
        """
        List all available models for the current project/org.

        Returns:
            List[Model]: Parsed list of model objects.
        """
        resp = self._http.get("/v1/models")
        data = resp.get("data", [])
        return [Model.from_dict(m) for m in data]

    def get_model(self, model_id: str) -> Model:
        """
        Retrieve a single model by ID.
        """
        resp = self._http.get(f"/v1/models/{model_id}")
        return Model.from_dict(resp)

    def delete_model(self, model_id: str) -> ModelDeletion:
        """
        Delete a fine-tuned model by ID.

        You must have the Owner role in your organization to delete a model.
        """
        resp = self._http.delete(f"/v1/models/{model_id}")
        return ModelDeletion.from_dict(resp)

    # Convenience helpers ----------------------------------------------------

    def list_model_ids(self) -> List[str]:
        """
        Convenience: return just the list of model IDs.
        """
        return [m.id for m in self.list_models()]


__all__ = [
    "Model",
    "ModelDeletion",
    "ModelsMixin",
]
