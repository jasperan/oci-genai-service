"""Authentication methods for OCI GenAI Service.

Supports: User Principal, Instance Principal, Resource Principal, Session Token, API Keys.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Optional

from oci_openai import OciUserPrincipalAuth

OCI_GENAI_URL_TEMPLATE = "https://inference.generativeai.{region}.oci.oraclecloud.com/20231130/actions/v1"

AuthType = Literal["user_principal", "instance_principal", "resource_principal", "session", "api_key"]


def get_base_url(region: str) -> str:
    """Build the OCI GenAI inference base URL for a region."""
    return OCI_GENAI_URL_TEMPLATE.format(region=region)


@dataclass
class AuthConfig:
    """Configuration for OCI authentication."""

    auth_type: AuthType = "user_principal"
    profile_name: str = "DEFAULT"
    region: str = "us-chicago-1"
    config_file: str = "~/.oci/config"
    compartment_id: Optional[str] = None
    api_key: Optional[str] = None

    @classmethod
    def from_env(cls) -> AuthConfig:
        """Create AuthConfig from environment variables."""
        return cls(
            profile_name=os.environ.get("OCI_PROFILE", "DEFAULT"),
            region=os.environ.get("OCI_REGION", "us-chicago-1"),
            compartment_id=os.environ.get("OCI_COMPARTMENT_ID"),
            config_file=os.environ.get("OCI_CONFIG_FILE", "~/.oci/config"),
            api_key=os.environ.get("OCI_GENAI_API_KEY"),
            auth_type="api_key" if os.environ.get("OCI_GENAI_API_KEY") else "user_principal",
        )


def create_auth(config: AuthConfig):
    """Create an auth object based on the config type.

    Returns None for api_key auth (uses bearer token via openai client).
    """
    if config.auth_type == "user_principal":
        return OciUserPrincipalAuth(
            profile_name=config.profile_name,
            config_file=config.config_file,
        )
    elif config.auth_type == "instance_principal":
        from oci_openai import OciInstancePrincipalAuth
        return OciInstancePrincipalAuth()
    elif config.auth_type == "resource_principal":
        from oci_openai import OciResourcePrincipalAuth
        return OciResourcePrincipalAuth()
    elif config.auth_type == "session":
        from oci_openai import OciSessionAuth
        return OciSessionAuth(
            profile_name=config.profile_name,
            config_file=config.config_file,
        )
    elif config.auth_type == "api_key":
        return None
    else:
        raise ValueError(f"Unsupported auth type: {config.auth_type}")
