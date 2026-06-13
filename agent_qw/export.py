"""Export pipeline results to TXT and JSON formats."""

import json
import logging
from pathlib import Path
from typing import List

from .schemas import Page

logger = logging.getLogger(__name__)


def export_txt(pages: List[Page], output_path: str) -> str:
    """Export recognized text to a plain text file.

    Pages are separated by a header line.
    Blocks within a page follow reading order (right-to-left for vertical).

    Args:
        pages: List of Page results.
        output_path: Path to write the .txt file.

    Returns:
        Absolute path to the written file.
    """
    path = Path(output_path)
    lines = []

    for page in pages:
        if len(pages) > 1:
            lines.append(f"===== 第 {page.page_index + 1} 页 =====")
            lines.append("")

        for block in page.ordered_blocks:
            if block.text.strip():
                lines.append(block.text)
            if block.note:
                lines.append(f"[注: {block.note}]")
            lines.append("")

        lines.append("")

    text = "\n".join(lines).strip()
    path.write_text(text, encoding="utf-8")
    logger.info("TXT exported to %s", path)
    return str(path.resolve())


def export_json(pages: List[Page], output_path: str) -> str:
    """Export full recognition results to a structured JSON file.

    Includes all metadata: bounding boxes, confidence scores,
    uncertain spans, and layout information.

    Args:
        pages: List of Page results.
        output_path: Path to write the .json file.

    Returns:
        Absolute path to the written file.
    """
    path = Path(output_path)

    data = {
        "version": "0.1.0",
        "page_count": len(pages),
        "pages": [page.to_dict() for page in pages],
    }

    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("JSON exported to %s", path)
    return str(path.resolve())
