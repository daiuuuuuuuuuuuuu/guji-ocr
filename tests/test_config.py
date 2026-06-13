"""Tests for shared config module (no LLM calls)."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from shared.config import create_llm_client


class TestCreateLlmClient:
    """Tests for create_llm_client config resolution."""

    def test_missing_key_raises(self, monkeypatch):
        """Should raise RuntimeError when no key is configured anywhere."""
        # Clear env vars and ensure no config.yaml with keys
        monkeypatch.delenv("GUJI_LLM_API_KEY", raising=False)
        monkeypatch.delenv("GUJI_VERIFY_API_KEY", raising=False)
        monkeypatch.delenv("GUJI_LLM_BASE_URL", raising=False)
        monkeypatch.delenv("GUJI_LLM_MODEL", raising=False)
        monkeypatch.setenv("GUJI_CONFIG", "/nonexistent/path/config.yaml")

        with pytest.raises(RuntimeError, match="No API key configured"):
            create_llm_client(section="agent_verify")

    def test_explicit_args_highest_priority(self, monkeypatch):
        monkeypatch.setenv("GUJI_CONFIG", "/nonexistent/path/config.yaml")
        client, model = create_llm_client(
            section="test",
            api_key="explicit-key",
            base_url="https://explicit.example.com/v1",
            model="explicit-model",
        )
        assert client.api_key == "explicit-key"
        assert str(client.base_url).rstrip("/") == "https://explicit.example.com/v1"
        assert model == "explicit-model"

    def test_env_var_override(self, monkeypatch, tmp_path):
        """GUJI_LLM_API_KEY env var should work."""
        monkeypatch.setenv("GUJI_LLM_API_KEY", "env-key")
        monkeypatch.setenv("GUJI_LLM_BASE_URL", "https://env.example.com/v1")
        monkeypatch.setenv("GUJI_LLM_MODEL", "env-model")
        monkeypatch.setenv("GUJI_CONFIG", "/nonexistent/path/config.yaml")

        client, model = create_llm_client(section="test")
        assert client.api_key == "env-key"
        assert model == "env-model"

    def test_section_specific_env_var(self, monkeypatch):
        """GUJI_TEST_API_KEY should work for section='test'."""
        monkeypatch.setenv("GUJI_CONFIG", "/nonexistent/path/config.yaml")
        monkeypatch.setenv("GUJI_TEST_API_KEY", "section-key")

        client, model = create_llm_client(section="test")
        assert client.api_key == "section-key"

    def test_config_yaml_loading(self, monkeypatch, tmp_path):
        """config.yaml section should be read correctly."""
        config = tmp_path / "config.yaml"
        config.write_text(
            yaml.dump({
                "test": {
                    "api_key": "yaml-key",
                    "base_url": "https://yaml.example.com/v1",
                    "model": "yaml-model",
                }
            }),
            encoding="utf-8",
        )
        monkeypatch.setenv("GUJI_CONFIG", str(config))

        client, model = create_llm_client(section="test")
        assert client.api_key == "yaml-key"
        assert model == "yaml-model"

    def test_fallback_to_llm_section(self, monkeypatch, tmp_path):
        """When section doesn't exist, fall back to llm section."""
        config = tmp_path / "config.yaml"
        config.write_text(
            yaml.dump({
                "llm": {
                    "api_key": "fallback-key",
                    "model": "fallback-model",
                }
            }),
            encoding="utf-8",
        )
        monkeypatch.setenv("GUJI_CONFIG", str(config))

        client, model = create_llm_client(section="nonexistent")
        assert client.api_key == "fallback-key"
        assert model == "fallback-model"

    def test_default_base_url(self, monkeypatch):
        monkeypatch.setenv("GUJI_CONFIG", "/nonexistent/path/config.yaml")
        client, _ = create_llm_client(
            section="test", api_key="key", model="m"
        )
        assert str(client.base_url).rstrip("/") == "https://api.openai.com/v1"
