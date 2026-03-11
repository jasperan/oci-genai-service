"""Tests for authentication module."""

import pytest
from unittest.mock import patch, MagicMock
from oci_genai_service.auth import AuthConfig, create_auth, get_base_url


class TestGetBaseUrl:
    def test_default_region(self):
        url = get_base_url("us-chicago-1")
        assert url == "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com/20231130/actions/v1"

    def test_frankfurt_region(self):
        url = get_base_url("eu-frankfurt-1")
        assert url == "https://inference.generativeai.eu-frankfurt-1.oci.oraclecloud.com/20231130/actions/v1"


class TestAuthConfig:
    def test_default_values(self):
        config = AuthConfig()
        assert config.profile_name == "DEFAULT"
        assert config.region == "us-chicago-1"
        assert config.config_file == "~/.oci/config"

    def test_custom_values(self):
        config = AuthConfig(profile_name="foosball", region="eu-frankfurt-1")
        assert config.profile_name == "foosball"
        assert config.region == "eu-frankfurt-1"

    def test_from_env(self):
        with patch.dict("os.environ", {
            "OCI_PROFILE": "myprofile",
            "OCI_REGION": "ap-osaka-1",
            "OCI_COMPARTMENT_ID": "ocid1.compartment.oc1..abc",
        }):
            config = AuthConfig.from_env()
            assert config.profile_name == "myprofile"
            assert config.region == "ap-osaka-1"
            assert config.compartment_id == "ocid1.compartment.oc1..abc"


class TestCreateAuth:
    def test_user_principal_type(self):
        config = AuthConfig(auth_type="user_principal")
        with patch("oci_genai_service.auth.OciUserPrincipalAuth") as mock_auth:
            mock_auth.return_value = MagicMock()
            auth = create_auth(config)
            mock_auth.assert_called_once_with(
                profile_name="DEFAULT",
                config_file="~/.oci/config",
            )

    def test_api_key_type(self):
        config = AuthConfig(auth_type="api_key", api_key="sk-test123")
        auth = create_auth(config)
        assert auth is None

    def test_invalid_type_raises(self):
        config = AuthConfig(auth_type="invalid")
        with pytest.raises(ValueError, match="Unsupported auth type"):
            create_auth(config)
