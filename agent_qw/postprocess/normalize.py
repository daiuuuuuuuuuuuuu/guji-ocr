"""Post-processing: conservative rule-based text normalization."""

import re
import logging

logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """Apply conservative normalization rules to recognized text.

    Rules are deliberately minimal to preserve the original text:
    1. Normalize whitespace (collapse multiple spaces/newlines)
    2. Remove isolated control characters
    3. Strip extra blank lines

    Does NOT:
    - Convert traditional to simplified Chinese
    - Add or modify punctuation
    - Change character forms

    Args:
        text: Raw OCR output text.

    Returns:
        Normalized text.
    """
    if not text:
        return ""

    # Remove control characters except newlines
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # Collapse multiple spaces into one
    text = re.sub(r'[ \t]+', ' ', text)

    # Collapse 3+ newlines into at most 2
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Strip trailing whitespace from each line
    text = '\n'.join(line.rstrip() for line in text.split('\n'))

    # Strip leading/trailing blank lines
    text = text.strip('\n')

    # Normalize full-width characters that are sometimes garbled
    text = text.replace('\u3000', ' ')  # full-width space → regular space

    return text


def normalize_page_text(text: str) -> str:
    """Additional page-level text normalization for classical Chinese.

    - Removes empty lines that would break reading flow
    - Ensures blank line between distinct text regions
    """
    text = normalize_text(text)

    # Remove truly empty lines but keep single blank between paragraphs
    lines = text.split('\n')
    cleaned = []
    prev_blank = False
    for line in lines:
        if not line.strip():
            if not prev_blank and cleaned:
                cleaned.append('')
            prev_blank = True
        else:
            cleaned.append(line)
            prev_blank = False

    return '\n'.join(cleaned)
