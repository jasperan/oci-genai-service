"""Tests for the CLI."""

import pytest
from click.testing import CliRunner
from oci_genai_service.cli.main import cli


class TestCLI:
    def test_version(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "2.0.0" in result.output

    def test_models_list(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["models", "list"])
        assert result.exit_code == 0
        assert "meta.llama-4-maverick" in result.output

    def test_models_list_filter_vendor(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["models", "list", "--vendor", "cohere"])
        assert result.exit_code == 0
        assert "cohere.embed" in result.output
        assert "meta.llama" not in result.output

    def test_models_info(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["models", "info", "meta.llama-4-maverick"])
        assert result.exit_code == 0
        assert "Llama 4 Maverick" in result.output
        assert "131072" in result.output
