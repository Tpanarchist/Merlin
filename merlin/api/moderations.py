"""
Moderations API
===============

Given text and/or image inputs, classifies if those inputs are potentially harmful.

Endpoint
--------
POST /v1/moderations  → create a moderation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from merlin.http_client import MerlinHTTPClient

JSON = Dict[str, Any]
ModerationInput = Union[str, Mapping[str, Any]]


# ───────────────────────────────────────────────────────────────
# Dataclasses
# ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModerationResult:
    """
    One moderation result corresponding to an input item.
    """

    flagged: bool
    categories: Dict[str, bool]
    category_scores: Dict[str, float]
    category_applied_input_types: Optional[Dict[str, List[str]]] = None

    raw: JSON = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ModerationResult":
        return cls(
            flagged=bool(d.get("flagged", False)),
            categories=dict(d.get("categories", {})),
            category_scores=dict(d.get("category_scores", {})),
            category_applied_input_types=(
                dict(d["category_applied_input_types"])
                if "category_applied_input_types" in d
                else None
            ),
            raw=dict(d),
        )


@dataclass(frozen=True)
class Moderation:
    """
    Top-level moderation response object.
    """

    id: str
    model: str
    results: List[ModerationResult]
    raw: JSON = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Moderation":
        return cls(
            id=d["id"],
            model=d["model"],
            results=[ModerationResult.from_dict(r) for r in d.get("results", [])],
            raw=dict(d),
        )

    @property
    def flagged_any(self) -> bool:
        """
        True if any result in this moderation is flagged.
        """
        return any(r.flagged for r in self.results)

    @property
    def first(self) -> Optional[ModerationResult]:
        """
        Convenience: return the first result, if any.
        """
        return self.results[0] if self.results else None


# ───────────────────────────────────────────────────────────────
# ModerationsMixin
# ───────────────────────────────────────────────────────────────

class ModerationsMixin:
    """
    High-level wrapper for the Moderations API.

        mod = client.create_moderation("I want to kill them.")
        mod.flagged_any  # → True / False

        # For quick checks:
        client.is_flagged("some text")
    """

    _http: MerlinHTTPClient

    def create_moderation(
        self,
        input: Union[ModerationInput, Sequence[ModerationInput]],
        model: Optional[str] = None,
    ) -> Moderation:
        """
        Create a moderation request.

        Args:
            input:
                - A single string
                - A list of strings
                - A list of multimodal input objects (dicts), like other models
            model:
                Optional moderation model ID (defaults to 'omni-moderation-latest').
        """
        payload: JSON = {
            "input": input,
        }
        if model is not None:
            payload["model"] = model

        resp = self._http.post("/v1/moderations", json=payload)
        return Moderation.from_dict(resp)

    # Convenience helpers ----------------------------------------------------

    def is_flagged(
        self,
        input: Union[ModerationInput, Sequence[ModerationInput]],
        model: Optional[str] = None,
    ) -> bool:
        """
        Convenience: return whether any result is flagged.
        """
        return self.create_moderation(input=input, model=model).flagged_any


__all__ = [
    "Moderation",
    "ModerationResult",
    "ModerationsMixin",
]
