"""Shared test fixtures for oci-genai-service."""

import pytest


@pytest.fixture
def compartment_id():
    return "ocid1.compartment.oc1..test"


@pytest.fixture
def region():
    return "us-chicago-1"
