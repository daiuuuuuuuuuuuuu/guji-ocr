"""Abstract base class for OCR engines."""

from abc import ABC, abstractmethod
from typing import List

from ..schemas import OCRLine


class AbstractOCR(ABC):
    """OCR engine interface. All OCR backends must implement this."""

    @abstractmethod
    def recognize(self, image) -> List[OCRLine]:
        """Run OCR on a single image and return recognized text lines.

        Args:
            image: PIL Image or numpy array (BGR).

        Returns:
            List of OCRLine objects with text, confidence, and bounding boxes.
        """
        ...

    @abstractmethod
    def recognize_text_only(self, image) -> str:
        """Run OCR and return concatenated text only (no bbox data).

        Useful as a fallback when layout analysis fails.
        """
        ...
