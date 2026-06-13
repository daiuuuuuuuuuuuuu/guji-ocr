"""LLM-based text correction for classical Chinese OCR output.

Sends the full paragraph text to the LLM and receives corrected text
directly.  Designed for ``llm_vision`` engine output which has no
per-character confidence metadata.
"""

from __future__ import annotations

import logging
import time

import openai

from shared.config import create_llm_client  # noqa: F401 — re-exported

logger = logging.getLogger(__name__)

CORRECT_SYSTEM_PROMPT = """你是一位古籍校勘专家。以下文本来自 OCR 识别古籍图片的结果，可能包含识别错误。

## 你的任务
仔细阅读文本，修正 OCR 识别错误，然后输出完整的纠正后文本。

## 重点关注的错误类型
1. 形近字误识：己/已/巳、日/曰、未/末、干/千、人/入、大/太、王/玉、土/士、
   戍/戌/戊、母/毋、本/木、天/夫、午/牛、尤/龙、侯/候、爪/瓜 等
2. 上下文不通：根据古籍用词习惯和上下文判断该字是否合理
3. 缺字、衍字、明显乱码

## 核心原则
1. **保持原文**：不转换繁简体，不添加标点，不改变原有字形
2. **保护异体字和避讳字**：异体字（如「羣」「峯」）不改；避讳缺笔字保持原样
3. **保守修改**：有充分把握才改，拿不准就保留原字
4. **只输出纠正后的纯文本**：不要任何解释、标记、或格式，直接输出纠正后的完整文本
5. **保持原有换行和空白**：维持原文的段落结构和换行方式
6. **不添加任何内容**：不要添加页码、标题、注释等原文中没有的内容

## 示例
输入：
學而時習之不変楽乎

输出：
學而時習之不亦楽乎
（"変"是"亦"的形近误识，改之；"楽"是异体字，保留）"""

MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0  # seconds


def correct_page(
    text: str,
    client: openai.OpenAI,
    model: str = "gpt-4o",
    temperature: float = 0.1,
    max_tokens: int = 4096,
) -> str:
    """Send a page of OCR text to the LLM for correction.

    Args:
        text: Raw OCR output text for one page.
        client: Configured openai.OpenAI client.
        model: Model name to use.
        temperature: LLM temperature (low = more deterministic).
        max_tokens: Maximum tokens in the response.

    Returns:
        Corrected text.  Returns the original text unchanged on any error.
    """
    if not text.strip():
        return text

    messages = [
        {"role": "system", "content": CORRECT_SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]

    try:
        # Retry on transient errors (429 rate-limit, 5xx server errors)
        last_exc = None
        for attempt in range(MAX_RETRIES):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                corrected = response.choices[0].message.content or ""

                if not corrected.strip():
                    logger.warning("LLM returned empty response, keeping original")
                    return text

                if corrected != text:
                    logger.debug("LLM made corrections to the page")
                return corrected

            except openai.APIError as e:
                last_exc = e
                should_retry = (
                    getattr(e, "status_code", None) == 429
                    or (getattr(e, "status_code", None) or 500) >= 500
                )
                if not should_retry or attempt == MAX_RETRIES - 1:
                    raise
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                logger.debug(
                    "LLM API error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1, MAX_RETRIES, delay, e,
                )
                time.sleep(delay)

        raise last_exc  # type: ignore[misc]

    except openai.APIError as e:
        logger.warning("LLM API error, keeping original: %s", e)
        return text
    except Exception as e:
        logger.warning("LLM correction failed, keeping original: %s", e)
        return text
