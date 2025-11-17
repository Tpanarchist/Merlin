"""
Backward Compatibility
======================

This module encodes Merlin's stance on API stability, based on the
OpenAI documentation.

Key points:

- The REST API is versioned (currently v1) and aims to avoid breaking
  changes within a major version.
- First-party SDKs follow semantic versioning.
- Model families (e.g., gpt-4o, o4-mini) are stable, but *snapshots*
  (dated variants) may change behavior.
- Recommended practice:
    - Pin model versions (e.g. "gpt-4o-2024-08-06")
    - Maintain evals to catch behavior drift between snapshots.

This module provides tiny utilities and constants around those ideas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

REST_API_VERSION: str = "2020-10-01"
"""
The REST API version reported by OpenAI in the `openai-version` header.

Note: This constant is informational; the actual behavior is determined by
the server. Do not attempt to override it client-side.
"""

CHANGELOG_URL: str = "https://platform.openai.com/docs/changelog"
"""
Canonical URL for backwards-compatible changes and rare breaking changes.
"""

@dataclass(frozen=True)
class ModelVersion:
    """
    A simple representation of a "pinned" model version.

    Example:
        family = "gpt-4o"
        snapshot = "2024-08-06"
        full_id = "gpt-4o-2024-08-06"
    """

    family: str
    snapshot: Optional[str] = None

    @property
    def full_id(self) -> str:
        """Return the full model identifier."""
        if self.snapshot:
            return f"{self.family}-{self.snapshot}"
        return self.family

def parse_model_version(model_id: str) -> ModelVersion:
    """
    Parse a model identifier into (family, snapshot) components.

    This is heuristic: it assumes that a trailing `-YYYY-MM-DD` component
    indicates a snapshot; otherwise the entire string is treated as the
    family name.

    Examples:
        "gpt-4o" -> ModelVersion(family="gpt-4o", snapshot=None)
        "gpt-4o-2024-08-06" -> ModelVersion(family="gpt-4o",
                                            snapshot="2024-08-06")
    """
    parts = model_id.split("-")
    if len(parts) >= 4:
        # Example: ["gpt", "4o", "2024", "08", "06"] or similar
        family = "-".join(parts[:-3])
        snapshot = "-".join(parts[-3:])
        return ModelVersion(family=family, snapshot=snapshot)
    return ModelVersion(family=model_id, snapshot=None)

__all__ = [
    "REST_API_VERSION",
    "CHANGELOG_URL",
    "ModelVersion",
    "parse_model_version",
]
