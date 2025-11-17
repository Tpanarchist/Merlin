"""
Authentication
==============

Helpers for configuring and applying authentication to Merlin HTTP requests.

This module encodes the rules from the OpenAI docs:

- API keys are required and must be kept secret.
- Authentication is via HTTP Bearer token:
      Authorization: Bearer OPENAI_API_KEY
- Optional headers:
      OpenAI-Organization: <organization_id>
      OpenAI-Project: <project_id>

In Merlin, we treat these as *context* attached to the client, not as
global state. This module provides:

- AuthConfig: typed configuration for auth-related values
- load_auth_from_env(): convenience loader for server-side usage
- build_auth_headers(): construct headers for a single request
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import os

# Environment variable names (convention; you can change if desired).
ENV_API_KEY = "OPENAI_API_KEY"
ENV_ORG_ID = "OPENAI_ORGANIZATION"
ENV_PROJECT_ID = "OPENAI_PROJECT"

class MissingAPIKeyError(RuntimeError):
    """Raised when no API key can be found for authentication."""

@dataclass(frozen=True)
class AuthConfig:
    """
    Authentication context for Merlin.

    Attributes:
        api_key:
            Secret API key. MUST be kept server-side only.
        organization_id:
            Optional OpenAI organization ID. If provided, requests will
            count usage against this organization.
        project_id:
            Optional OpenAI project ID. If provided, requests will
            count usage against this project within the organization.
    """

    api_key: str
    organization_id: Optional[str] = None
    project_id: Optional[str] = None

    @classmethod
    def from_env(cls) -> "AuthConfig":
        """
        Load authentication info from environment variables.

        Required:
            - OPENAI_API_KEY

        Optional:
            - OPENAI_ORGANIZATION
            - OPENAI_PROJECT

        Raises:
            MissingAPIKeyError: if OPENAI_API_KEY is not set.
        """
        api_key = os.getenv(ENV_API_KEY)
        if not api_key:
            raise MissingAPIKeyError(
                f"Missing API key: set {ENV_API_KEY} in your environment."
            )

        org_id = os.getenv(ENV_ORG_ID) or None
        project_id = os.getenv(ENV_PROJECT_ID) or None

        return cls(api_key=api_key, organization_id=org_id, project_id=project_id)

def build_auth_headers(auth: AuthConfig) -> Dict[str, str]:
    """
    Build the authentication-related HTTP headers for a request.

    Returns a dict containing:
        - Authorization: Bearer <api_key>
        - OpenAI-Organization: <organization_id>   (if provided)
        - OpenAI-Project: <project_id>             (if provided)
    """
    headers: Dict[str, str] = {
        "Authorization": f"Bearer {auth.api_key}",
    }

    if auth.organization_id:
        headers["OpenAI-Organization"] = auth.organization_id

    if auth.project_id:
        headers["OpenAI-Project"] = auth.project_id

    return headers

__all__ = [
    "AuthConfig",
    "MissingAPIKeyError",
    "build_auth_headers",
    "ENV_API_KEY",
    "ENV_ORG_ID",
    "ENV_PROJECT_ID",
]
