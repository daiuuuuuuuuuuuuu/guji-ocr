"""Multimodal LLM-based OCR engine.

Uses a vision-capable large language model (e.g. Qwen-VL) to read text
directly from images — ideal for vertical classical Chinese and other
scripts that are challenging for traditional OCR engines.
"""

import base64
import io
import logging
from typing import List

from PIL import Image

from .base import AbstractOCR
from ..schemas import OCRLine, BBox

logger = logging.getLogger(__name__)

# Imported here to avoid circular imports; resolved at call time.
# The prompt template is defined in agent_qw.llm.prompts.


class LLMVisionOCR(AbstractOCR):
    """OCR via a multimodal LLM (e.g. Qwen-VL).

    Each column-crop image is encoded as a JPEG base64 data URI and sent
    to the vision model along with a system prompt that instructs it to
    act as a classical-text OCR expert.

    Because the model sees the original image, no secondary LLM collation
    pass is needed — the output is already context-aware.
    """

    def __init__(self, llm_client, model, temperature, max_tokens):
        """Initialize with a shared LLM client for vision OCR.

        Args:
            llm_client: An OpenAICompatLLM instance (used for its base_url
                        and api_key). A separate internal client is created
                        with the vision-specific model / temperature / max_tokens.
            model: Vision model name (e.g. "qwen3.7-plus").
            temperature: Sampling temperature for OCR (0.0–0.3 recommended).
            max_tokens: Maximum tokens in the response.
        """
        # Create a dedicated LLM client for vision calls so that model,
        # temperature, and max_tokens can differ from the text-only LLM config.
        from ..llm.openai_compat import OpenAICompatLLM

        self._llm = OpenAICompatLLM(
            base_url=llm_client.base_url,
            api_key=llm_client.api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def recognize(self, image) -> List[OCRLine]:
        """Run vision OCR on a column-crop image.

        Returns a single OCRLine covering the full crop area, since the
        multimodal LLM reads the entire column in one pass and does not
        provide per-character bounding boxes.
        """
        text = self._call_vision(image)

        if not text.strip():
            return []

        w, h = image.size

        return [
            OCRLine(
                text=text,
                confidence=1.0,
                bbox=BBox(x=0, y=0, w=w, h=h),
                uncertain_spans=[],
            )
        ]

    def recognize_text_only(self, image) -> str:
        """Run vision OCR and return concatenated text."""
        return self._call_vision(image)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_vision(self, image: Image.Image) -> str:
        """Encode image as base64, send to multimodal LLM, return text."""
        from ..llm.prompts import OCR_VISION_PROMPT

        data_uri = self._encode_image(image)

        messages = [
            {"role": "system", "content": OCR_VISION_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": data_uri},
                    },
                    {
                        "type": "text",
                        "text": "请识别这张古籍图片中的文字，先输出 [[PAGE:数字]] 标记（如有页码），然后输出识别到的正文。",
                    },
                ],
            },
        ]

        # The multimodal content format uses list[dict] for `content`,
        # which is valid OpenAI API but doesn't match the type annotation
        # Dict[str, str] on the abstract chat() interface. The actual
        # openai client accepts it at runtime.
        return self._llm.chat(messages).strip()

    @staticmethod
    def _encode_image(image: Image.Image) -> str:
        """Encode a PIL Image as a JPEG base64 data URI."""
        # Convert to RGB in case the image has an alpha channel or is
        # in a mode that JPEG doesn't support (e.g. RGBA, P).
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")

        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=92)
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"
