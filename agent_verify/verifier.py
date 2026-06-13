"""LLM-based page-order verification via text-continuity analysis.

Unlike the old ``detector.py`` (regex + Chinese numeral parsing), this module
sends each page's head/tail text to an LLM and asks it to determine the correct
reading order based on textual coherence — the end of page N should flow
naturally into the beginning of page N+1.
"""

from __future__ import annotations

import json
import logging
import re
import time

import openai

from shared.config import create_llm_client  # noqa: F401 — re-exported

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HEAD_CHARS = 80   # characters extracted from the start of each page
TAIL_CHARS = 80   # characters extracted from the end of each page
BATCH_SIZE = 100  # maximum pages per LLM call

# ---------------------------------------------------------------------------
# System prompt — classical Chinese text continuity expert
# ---------------------------------------------------------------------------

VERIFY_SYSTEM_PROMPT = """你是一位中文古籍文献专家。你的任务是根据文字连贯性判断古籍页面的正确阅读顺序。

## 背景
古籍每页文字是连续的——第 N 页结尾的文字应当与第 N+1 页开头的文字连成一句通顺的话。你的任务是比较各页的「首部文字」和「尾部文字」，找出文字最连贯的页面排列顺序。

## 判断方法
- 如果页 A 的尾部文字与页 B 的首部文字能连成通顺的语句，则 A 之后应该是 B
- 注意古文用词习惯、句式结构、上下文逻辑
- 如果某页的首部是新段落开头（语义独立），它可能是一卷/一章的起始页
- 同一卷/章内的页面文字是连续流动的；卷与卷之间的文字通常不连续
- 如果多页首部句式相同（如都以"某年某月"开头），根据尾部与下页首部的衔接来判断顺序

## 输入格式
每页提供首部和尾部文字，格式为：
[页N] 首：「前 80 字内容」 尾：「后 80 字内容」

## 输出格式
严格按以下 JSON 格式输出，不要输出任何其他内容：
{"order": [3, 1, 2, 4], "notes": "简要说明排序依据"}

- order 数组：重新排列后的页码顺序（必须包含所有页号，每个页号恰好出现一次）
- notes：简要说明排序逻辑（如哪页接哪页最连贯，或发现了章节分界）
- 如果无法确定正确顺序（所有排列都同样合理），order 保持输入顺序即可，notes 说明原因"""


# ---------------------------------------------------------------------------
# Config / client helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def extract_head_tail(text: str) -> tuple[str, str]:
    """Extract the first HEAD_CHARS and last TAIL_CHARS characters from text.

    Whitespace-only lines are collapsed so the extracted snippets are
    meaningful contiguous text rather than empty newlines.

    Returns:
        (head, tail) — each a string of up to HEAD_CHARS/TAIL_CHARS characters.
    """
    # Collapse runs of whitespace to single spaces for denser snippets
    collapsed = re.sub(r"\s+", "", text)
    if not collapsed:
        return ("", "")

    head = collapsed[:HEAD_CHARS]
    tail = collapsed[-TAIL_CHARS:] if len(collapsed) > TAIL_CHARS else collapsed
    return (head, tail)


# ---------------------------------------------------------------------------
# LLM batch verification
# ---------------------------------------------------------------------------

def _build_batch_prompt(
    pages: list[tuple[str, str]],
    prev_tail: str | None = None,
) -> str:
    """Build the user message for a batch of pages.

    Args:
        pages: list of (head, tail) for each page in physical order.
        prev_tail: tail text from the last page of the *previous* batch,
                   used to validate cross-batch continuity.

    Returns:
        Formatted prompt string ready to send to the LLM.
    """
    lines = []

    if prev_tail:
        lines.append(f"【上一批最后一页的尾部文字】「{prev_tail}」")
        lines.append("（请确保本批第一页的首部文字能与上面的尾部文字连贯）")
        lines.append("")

    lines.append(f"本批共 {len(pages)} 页，请判断正确的阅读顺序：")
    lines.append("")

    for i, (head, tail) in enumerate(pages, 1):
        lines.append(f"[页{i}] 首：「{head}」 尾：「{tail}」")

    lines.append("")
    lines.append("请输出 JSON，不要输出其他内容。")

    return "\n".join(lines)


def _extract_json(text: str) -> dict | None:
    """Try to extract a JSON object from an LLM response.

    Handles responses wrapped in ```json ... ``` fences or raw JSON.
    Uses a balanced-brace approach to correctly handle nested braces.
    """
    # Try fenced code block first
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Try raw JSON — find the first balanced JSON object
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None

    return None


MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0  # seconds


def _call_llm_with_retry(
    client: openai.OpenAI,
    model: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
) -> str:
    """Call the LLM chat API with retry on transient errors.

    Retries on 429 (rate limit) and 5xx (server) errors with exponential
    backoff.  Raises the last exception if all retries are exhausted.
    """
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        except openai.APIError as e:
            last_exc = e
            # Retry on rate-limit (429) and server errors (5xx)
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


def verify_batch(
    pages: list[tuple[str, str]],
    client: openai.OpenAI,
    model: str,
    prev_tail: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 4096,
) -> list[int]:
    """Send a batch of pages to the LLM and return the reordered indices.

    Args:
        pages: list of (head, tail) tuples, one per page, in *physical* order.
        client: Configured openai.OpenAI client.
        model: Model name to use.
        prev_tail: Tail text from the previous batch's last page (cross-batch).
        temperature: LLM temperature.
        max_tokens: Maximum tokens in the response.

    Returns:
        A list of 0-based indices into ``pages`` representing the correct
        reading order.  Falls back to the original physical order on any error.
    """
    n = len(pages)

    # Trivial case
    if n <= 1:
        return list(range(n))

    user_message = _build_batch_prompt(pages, prev_tail)

    messages = [
        {"role": "system", "content": VERIFY_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    try:
        content = _call_llm_with_retry(
            client, model, messages, temperature, max_tokens,
        )
        logger.debug("LLM response: %s", content[:500])

        result = _extract_json(content)
        if result is None:
            logger.warning("Could not parse JSON from LLM response, keeping physical order")
            return list(range(n))

        # Parse order (1-based in prompt → 0-based)
        raw_order: list[int] = result.get("order", [])
        notes = result.get("notes", "")
        if notes:
            logger.info("LLM notes: %s", notes)

        # Validate: must contain exactly all pages 1..n
        expected = set(range(1, n + 1))
        actual = set(raw_order)
        if expected != actual:
            logger.warning(
                "LLM returned invalid order %s (expected permutation of 1..%d), "
                "keeping physical order",
                raw_order, n,
            )
            return list(range(n))

        order = [i - 1 for i in raw_order]
        logger.debug("Reordered: %s", order)
        return order

    except openai.APIError as e:
        logger.warning("LLM API error after retries, keeping physical order: %s", e)
        return list(range(n))
    except Exception as e:
        logger.warning("LLM call failed, keeping physical order: %s", e)
        return list(range(n))
