"""Local page-number detection from classical Chinese text content.

No LLM — pure regex + Chinese numeral parsing.  Detects page numbers
that appear as standalone characters at the top or bottom of a page
(e.g. 版心 "五", "十三", or marginal "第廿一頁").
"""

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Chinese numeral → integer conversion
# ---------------------------------------------------------------------------

_CN_NUMERALS = {
    "零": 0, "〇": 0,
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9,
    "十": 10, "廿": 20, "卅": 30, "卌": 40,
    "百": 100,
}

# Patterns for page-number-like strings
# Standalone Chinese numeral at the end of a page (most common in 古籍)
_CN_PAGE_RE = re.compile(
    r"(?:第\s*)?([零〇一二三四五六七八九十廿卅卌百]+)(?:\s*(?:頁|页|章|卷|回))?",
)

# Arabic numerals at page edge
_ARABIC_PAGE_RE = re.compile(r"\b(\d{1,4})\b")


def _cn_to_int(cn: str) -> Optional[int]:
    """Convert a Chinese numeral string to an integer.

    Supports: 一 → 1, 十二 → 12, 二十五 → 25, 一百 → 100,
              一百二十三 → 123, 廿一 → 21, etc.

    Returns None if the string cannot be parsed.
    """
    if not cn:
        return None

    total = 0
    current = 0  # accumulator before hitting 十/百

    for ch in cn:
        if ch not in _CN_NUMERALS:
            return None
        val = _CN_NUMERALS[ch]

        if val >= 100:  # 百
            if current == 0:
                current = 1
            total += current * val
            current = 0
        elif val >= 10:  # 十, 廿, 卅, 卌
            if current == 0:
                current = 1
            total += current * val
            current = 0
        else:  # 0-9
            current = val

    total += current
    return total if total > 0 else None


# ---------------------------------------------------------------------------
# Page-number extraction
# ---------------------------------------------------------------------------

def detect_page_number(text: str) -> Optional[int]:
    """Detect a page number from the text content of a single page.

    Strategy (ordered by priority):
      1. Look for "第X頁/页" patterns anywhere in the text.
      2. Check the last 1-3 non-empty lines for standalone numerals.
      3. Check the first line for a standalone numeral (less common).

    Returns the detected page number as an int, or None if not found.
    """
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if not lines:
        return None

    # 1) Explicit "第N頁" pattern — highest confidence
    explicit = re.findall(r"第\s*([零〇一二三四五六七八九十廿卅卌百\d]+)\s*(?:頁|页)", text)
    if explicit:
        num = _parse_numeral(explicit[0])
        if num is not None:
            return num

    # 2) Check last 1-3 lines for a short numeral-only line (版心 页码)
    for ln in reversed(lines[-3:]):
        num = _parse_numeral(ln)
        if num is not None:
            return num

    # 3) Check first line for a short numeral
    num = _parse_numeral(lines[0])
    if num is not None:
        return num

    return None


def _parse_numeral(s: str) -> Optional[int]:
    """Try to parse a string as a page number (Chinese or Arabic)."""
    s = s.strip()
    if not s:
        return None

    # Try Arabic first
    m = _ARABIC_PAGE_RE.fullmatch(s)
    if m:
        return int(m.group(1))

    # Try Chinese numeral (allow extra trailing characters like page markers)
    m = _CN_PAGE_RE.fullmatch(s)
    if m:
        return _cn_to_int(m.group(1))

    # Loose match: if the string starts with a Chinese numeral
    m = re.match(r"^([零〇一二三四五六七八九十廿卅卌百]+)$", s)
    if m:
        return _cn_to_int(m.group(1))

    return None
