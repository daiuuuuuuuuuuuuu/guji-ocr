"""OpenAI-compatible LLM provider.

Works with any service that implements the OpenAI chat completions API:
OpenAI, Anthropic (via API gateway), local Ollama, vLLM, etc.
"""

import logging
import time
from typing import Any, Dict, List, Optional

import openai

from .base import AbstractLLM

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0  # seconds


class OpenAICompatLLM(AbstractLLM):
    """LLM provider backed by any OpenAI-compatible API endpoint.

    Configure via:
        base_url: API endpoint (default: https://api.openai.com/v1)
        api_key: API key
        model: Model name (default: gpt-4o)
    """

    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "",
        model: str = "gpt-4o",
        temperature: float = 0.1,
        max_tokens: int = 4096,
        timeout: float = 300.0,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        self._client = None
        if api_key:
            self._client = openai.OpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
            )

    @property
    def is_available(self) -> bool:
        return self._client is not None

    def chat(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Send a chat completion request.

        Args:
            messages: List of message dicts.
            response_format: Optional format specification.

        Returns:
            Response text from the assistant.

        Raises:
            RuntimeError: If the client is not configured or the API call fails.
        """
        if not self._client:
            raise RuntimeError(
                "LLM client not configured. Set GUJI_LLM_API_KEY environment variable."
            )

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        if response_format:
            kwargs["response_format"] = response_format

        try:
            response = self._client.chat.completions.create(**kwargs)
            return response.choices[0].message.content or ""
        except openai.APIError as e:
            # Retry on rate-limit (429) and server errors (5xx)
            should_retry = (
                getattr(e, "status_code", None) == 429
                or (getattr(e, "status_code", None) or 500) >= 500
            )
            if should_retry:
                for attempt in range(1, MAX_RETRIES):
                    delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    logger.debug(
                        "LLM API error (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1, MAX_RETRIES, delay, e,
                    )
                    time.sleep(delay)
                    try:
                        response = self._client.chat.completions.create(**kwargs)
                        return response.choices[0].message.content or ""
                    except openai.APIError:
                        continue
            logger.error("LLM API error: %s", e)
            raise RuntimeError(f"LLM API call failed: {e}") from e
        except Exception as e:
            logger.error("LLM call failed: %s", e)
            raise RuntimeError(f"LLM call failed: {e}") from e
