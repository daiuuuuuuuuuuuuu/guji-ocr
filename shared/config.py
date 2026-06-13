"""Unified LLM client creation with .env support.

Priority chain (highest to lowest):
  1. Explicit keyword arguments
  2. Environment variables (GUJI_*)
  3. .env file
  4. config.yaml (agent-specific section → llm section fallback)
  5. Hard-coded defaults
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import openai
import yaml

logger = logging.getLogger(__name__)

# ── Defaults ────────────────────────────────────────────────
_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_DEFAULT_MODEL = "gpt-4o"
_DEFAULT_TIMEOUT = 300.0

# ── .env loading (safe, no error if python-dotenv is missing) ─
try:
    from dotenv import load_dotenv

    _env_file = Path(os.environ.get("GUJI_ENV", ".env"))
    if _env_file.exists():
        load_dotenv(_env_file)
except ImportError:
    pass  # python-dotenv not installed — that's fine


def create_llm_client(
    section: str = "llm",
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> tuple[openai.OpenAI, str]:
    """Create an OpenAI-compatible LLM client with unified config resolution.

    Args:
        section: Config section name in config.yaml (e.g. ``"agent_verify"``).
        model: Override model name (highest priority).
        base_url: Override API base URL (highest priority).
        api_key: Override API key (highest priority).
        timeout: HTTP request timeout in seconds.

    Returns:
        (client, resolved_model) tuple.

    Raises:
        RuntimeError: If no API key can be resolved.
    """
    # Start with defaults
    resolved_base_url = _DEFAULT_BASE_URL
    resolved_api_key = ""
    resolved_model = _DEFAULT_MODEL

    # Layer 4: config.yaml (agent section → llm fallback)
    config_path = Path(os.environ.get("GUJI_CONFIG", "config.yaml"))
    if config_path.exists():
        try:
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            section_cfg = raw.get(section) or {}
            fallback_cfg = raw.get("llm") or {}
            llm_cfg = {**fallback_cfg, **section_cfg}  # section overrides llm
            if llm_cfg.get("base_url"):
                resolved_base_url = llm_cfg["base_url"]
            if llm_cfg.get("api_key"):
                resolved_api_key = llm_cfg["api_key"]
            if llm_cfg.get("model"):
                resolved_model = llm_cfg["model"]
        except Exception:
            logger.debug("Failed to read config.yaml, using defaults")

    # Layer 3: environment variables
    if os.environ.get("GUJI_LLM_BASE_URL"):
        resolved_base_url = os.environ["GUJI_LLM_BASE_URL"]
    if os.environ.get("GUJI_LLM_API_KEY"):
        resolved_api_key = os.environ["GUJI_LLM_API_KEY"]
    if os.environ.get("GUJI_LLM_MODEL"):
        resolved_model = os.environ["GUJI_LLM_MODEL"]

    # Layer 2: section-specific env vars (e.g. GUJI_VERIFY_API_KEY)
    section_env = section.upper()
    if os.environ.get(f"GUJI_{section_env}_BASE_URL"):
        resolved_base_url = os.environ[f"GUJI_{section_env}_BASE_URL"]
    if os.environ.get(f"GUJI_{section_env}_API_KEY"):
        resolved_api_key = os.environ[f"GUJI_{section_env}_API_KEY"]
    if os.environ.get(f"GUJI_{section_env}_MODEL"):
        resolved_model = os.environ[f"GUJI_{section_env}_MODEL"]

    # Layer 1: explicit arguments (highest priority)
    if base_url:
        resolved_base_url = base_url
    if api_key:
        resolved_api_key = api_key
    if model:
        resolved_model = model

    if not resolved_api_key:
        raise RuntimeError(
            f"No API key configured for '{section}'. "
            f"Set GUJI_{section_env}_API_KEY or "
            f"GUJI_LLM_API_KEY environment variable, "
            f"or configure {section}.api_key in config.yaml."
        )

    logger.info(
        "LLM client [%s]: model=%s base_url=%s",
        section, resolved_model, resolved_base_url,
    )
    client = openai.OpenAI(
        base_url=resolved_base_url,
        api_key=resolved_api_key,
        timeout=timeout,
    )
    return client, resolved_model
