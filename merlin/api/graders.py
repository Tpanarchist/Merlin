"""
Graders API
===========

High-level client for "Graders" (beta) and helper configs.

Docs section: "Graders"

Supported grader types
----------------------
- String check grader       → type: "string_check"
- Text similarity grader    → type: "text_similarity"
- Score model grader        → type: "score_model"
- Label model grader        → type: "label_model"
- Python grader             → type: "python"
- Multi grader              → type: "multi"

Endpoints
---------
Run grader (beta)
  POST /v1/fine_tuning/alpha/graders/run

Validate grader (beta)
  POST /v1/fine_tuning/alpha/graders/validate
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Union,
)

from merlin.http_client import MerlinHTTPClient


JSON = Dict[str, Any]


# ───────────────────────────────────────────────────────────────
# Base config + concrete grader config types
# ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GraderConfig:
    """
    Base class for grader configs.

    Concrete subclasses MUST set the `type` field to match the API's
    expected discriminator, e.g. "string_check", "text_similarity", etc.
    """

    type: str
    name: str

    def to_payload(self) -> JSON:
        """
        Convert this config into a JSON-serializable payload suitable
        for the API. Subclasses can override to customize behavior.
        """
        return asdict(self)


@dataclass(frozen=True)
class StringCheckGraderConfig(GraderConfig):
    """
    String Check Grader

        {
          "type": "string_check",
          "name": "Example string check grader",
          "input": "{{sample.output_text}}",
          "reference": "{{item.label}}",
          "operation": "eq"
        }
    """

    input: str
    reference: str
    operation: str  # "eq" | "ne" | "like" | "ilike"
    type: str = field(default="string_check", init=False)


@dataclass(frozen=True)
class TextSimilarityGraderConfig(GraderConfig):
    """
    Text Similarity Grader

        {
          "type": "text_similarity",
          "name": "Example text similarity grader",
          "input": "{{sample.output_text}}",
          "reference": "{{item.label}}",
          "evaluation_metric": "fuzzy_match"
        }
    """

    input: str
    reference: str
    evaluation_metric: str  # e.g. "cosine", "fuzzy_match", "bleu", ...
    type: str = field(default="text_similarity", init=False)


@dataclass(frozen=True)
class ScoreModelGraderConfig(GraderConfig):
    """
    Score Model Grader

        {
          "type": "score_model",
          "name": "Example score model grader",
          "input": [ { "role": "user", "content": "..." } ],
          "model": "o4-mini-2025-04-16",
          "range": [0, 1],
          "sampling_params": { ... }
        }
    """

    input: Sequence[JSON]
    model: str
    range: Optional[Sequence[float]] = None  # defaults to [0, 1] server-side
    sampling_params: Optional[JSON] = None
    type: str = field(default="score_model", init=False)

    def to_payload(self) -> JSON:
        data = asdict(self)
        # Strip None values so the API can apply its defaults cleanly.
        return {k: v for k, v in data.items() if v is not None}


@dataclass(frozen=True)
class LabelModelGraderConfig(GraderConfig):
    """
    Label Model Grader

        {
          "type": "label_model",
          "name": "First label grader",
          "model": "gpt-4o-2024-08-06",
          "input": [ ... ],
          "passing_labels": ["positive"],
          "labels": ["positive", "neutral", "negative"]
        }
    """

    model: str
    input: Sequence[JSON]
    labels: Sequence[str]
    passing_labels: Sequence[str]
    type: str = field(default="label_model", init=False)


@dataclass(frozen=True)
class PythonGraderConfig(GraderConfig):
    """
    Python Grader

        {
          "type": "python",
          "name": "Example python grader",
          "image_tag": "2025-05-08",
          "source": """def grade(sample: dict, item: dict) -> float: ..."""
        }
    """

    image_tag: str
    source: str
    type: str = field(default="python", init=False)


@dataclass(frozen=True)
class MultiGraderConfig(GraderConfig):
    """
    Multi Grader

        {
          "type": "multi",
          "name": "example multi grader",
          "graders": [ ...grader configs... ],
          "calculate_output": "0.5 * text_similarity_score + 0.5 * string_check_score"
        }

    `graders` can be either dicts (raw JSON) or GraderConfig instances.
    """

    graders: Sequence[Union[GraderConfig, Mapping[str, Any]]]
    calculate_output: str
    type: str = field(default="multi", init=False)

    def to_payload(self) -> JSON:
        # Normalize nested grader configs into dicts
        normalized_graders: List[JSON] = []
        for g in self.graders:
            if isinstance(g, GraderConfig):
                normalized_graders.append(g.to_payload())
            else:
                normalized_graders.append(dict(g))
        return {
            "type": self.type,
            "name": self.name,
            "graders": normalized_graders,
            "calculate_output": self.calculate_output,
        }


# Union type for any of the supported grader configs
AnyGraderConfig = Union[
    StringCheckGraderConfig,
    TextSimilarityGraderConfig,
    ScoreModelGraderConfig,
    LabelModelGraderConfig,
    PythonGraderConfig,
    MultiGraderConfig,
]


# ───────────────────────────────────────────────────────────────
# Run / validate responses
# ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GraderRunResult:
    """
    Response from `POST /v1/fine_tuning/alpha/graders/run`.

        {
          "reward": 1.0,
          "metadata": { ... },
          "sub_rewards": { ... },
          "model_grader_token_usage_per_model": { ... }
        }
    """

    reward: float
    metadata: JSON
    sub_rewards: JSON
    model_grader_token_usage_per_model: JSON
    raw: JSON = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GraderRunResult":
        return cls(
            reward=float(data.get("reward", 0.0)),
            metadata=dict(data.get("metadata") or {}),
            sub_rewards=dict(data.get("sub_rewards") or {}),
            model_grader_token_usage_per_model=dict(
                data.get("model_grader_token_usage_per_model") or {}
            ),
            raw=dict(data),
        )


@dataclass(frozen=True)
class ValidatedGrader:
    """
    Response from `POST /v1/fine_tuning/alpha/graders/validate`.

        {
          "grader": { ... }
        }
    """

    grader: JSON
    raw: JSON = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ValidatedGrader":
        grader = dict(data.get("grader") or {})
        return cls(grader=grader, raw=dict(data))


# ───────────────────────────────────────────────────────────────
# Client mixin
# ───────────────────────────────────────────────────────────────


class GradersMixin:
    """
    High-level helpers for the Graders API (beta).

    Assumes the consuming client defines:

        self._http: MerlinHTTPClient
    """

    _http: MerlinHTTPClient  # for type-checkers

    # Utility to normalize config → dict

    def _grader_payload(
        self,
        grader: Union[AnyGraderConfig, Mapping[str, Any]],
    ) -> JSON:
        if isinstance(grader, GraderConfig):
            return grader.to_payload()
        # already a dict-like grader spec (e.g. straight from docs)
        return dict(grader)

    # ── Run grader ─────────────────────────────────────────────

    def run_grader(
        self,
        *,
        grader: Union[AnyGraderConfig, Mapping[str, Any]],
        model_sample: str,
        item: Optional[Mapping[str, Any]] = None,
    ) -> GraderRunResult:
        """
        POST /v1/fine_tuning/alpha/graders/run

        Args:
            grader:
                Grader config – either one of the GraderConfig dataclasses
                or a raw JSON mapping that matches the API schema.
            model_sample:
                The model sample to evaluate. This populates the `sample`
                namespace, and `output_json` if it parses as JSON.
            item:
                Optional dataset item; populates the `item` namespace.

        Returns:
            GraderRunResult with the top-level reward, metadata, etc.
        """
        payload: MutableMapping[str, Any] = {
            "grader": self._grader_payload(grader),
            "model_sample": model_sample,
        }
        if item is not None:
            payload["item"] = dict(item)

        resp = self._http.post(
            "/v1/fine_tuning/alpha/graders/run",
            json=payload,
            expect_json=True,
        )
        return GraderRunResult.from_dict(resp)

    # ── Validate grader ────────────────────────────────────────

    def validate_grader(
        self,
        grader: Union[AnyGraderConfig, Mapping[str, Any]],
    ) -> ValidatedGrader:
        """
        POST /v1/fine_tuning/alpha/graders/validate

        Args:
            grader:
                Grader config to validate – either one of the GraderConfig
                dataclasses or a raw JSON mapping.

        Returns:
            ValidatedGrader, which exposes the server-normalized `grader`
            object (may be identical to the input, or may include
            additional defaults / normalization).
        """
        payload = {"grader": self._grader_payload(grader)}
        resp = self._http.post(
            "/v1/fine_tuning/alpha/graders/validate",
            json=payload,
            expect_json=True,
        )
        return ValidatedGrader.from_dict(resp)


__all__ = [
    # Configs
    "GraderConfig",
    "StringCheckGraderConfig",
    "TextSimilarityGraderConfig",
    "ScoreModelGraderConfig",
    "LabelModelGraderConfig",
    "PythonGraderConfig",
    "MultiGraderConfig",
    "AnyGraderConfig",
    # Results
    "GraderRunResult",
    "ValidatedGrader",
    # Mixin
    "GradersMixin",
]
