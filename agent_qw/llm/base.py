"""LLM abstract base class."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class AbstractLLM(ABC):
    """Pluggable LLM provider interface for text collation/review."""

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Send a chat completion request and return the response text.

        Args:
            messages: List of {"role": "user"|"system"|"assistant", "content": str}.
            response_format: Optional format specification (e.g. {"type": "json_object"}).

        Returns:
            The assistant's response text.
        """
        ...

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the LLM provider is configured and reachable."""
        ...
