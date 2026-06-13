"""Core data structures for the agent_qw recognition pipeline."""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional


class TextDirection(Enum):
    VERTICAL = "vertical"        # 竖排 (top-to-bottom columns)
    HORIZONTAL = "horizontal"    # 横排 (left-to-right rows)


class ReadingOrder(Enum):
    RIGHT_TO_LEFT = "right-to-left"  # columns ordered right to left
    LEFT_TO_RIGHT = "left-to-right"  # columns ordered left to right
    TOP_TO_BOTTOM = "top-to-bottom"  # rows ordered top to bottom


@dataclass
class BBox:
    """Bounding box in pixel coordinates."""
    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    def to_list(self) -> list:
        return [self.x, self.y, self.w, self.h]

    def to_xyxy(self) -> list:
        return [self.x, self.y, self.x2, self.y2]

    @classmethod
    def from_xyxy(cls, x1: int, y1: int, x2: int, y2: int) -> "BBox":
        return cls(x=x1, y=y1, w=x2 - x1, h=y2 - y1)


@dataclass
class UncertainSpan:
    """A span with low OCR confidence that may need review or substitution."""
    text: str
    start_char: int
    end_char: int
    confidence: float
    substituted: bool = False  # True if replaced with □


@dataclass
class OCRLine:
    """A single line/column of recognized text."""
    text: str
    confidence: float
    bbox: BBox
    uncertain_spans: list = field(default_factory=list)


@dataclass
class Block:
    """A layout block (column of text in vertical layout, row in horizontal)."""

    bbox: BBox
    text: str = ""                     # full text for this block
    id: str = ""
    lines: list = field(default_factory=list)   # OCRLine objects
    direction: TextDirection = TextDirection.VERTICAL
    order_index: int = 0               # reading position (0-based)
    note: str = ""                     # e.g. "layout_failed"


@dataclass
class Page:
    """Full page result."""

    page_index: int
    image_path: str
    width: int
    height: int
    blocks: list = field(default_factory=list)
    full_text: str = ""
    direction: TextDirection = TextDirection.VERTICAL
    reading_order: ReadingOrder = ReadingOrder.RIGHT_TO_LEFT
    note: str = ""
    detected_page_number: Optional[int] = None  # extracted by LLM from page content
    raw_text: str = ""  # full OCR text before header/footer stripping (for JSON export)

    @property
    def ordered_blocks(self) -> list:
        """Blocks sorted in correct reading order (right-to-left for vertical)."""
        if self.reading_order == ReadingOrder.RIGHT_TO_LEFT:
            return sorted(self.blocks, key=lambda b: (-b.bbox.x, b.bbox.y))
        elif self.reading_order == ReadingOrder.LEFT_TO_RIGHT:
            return sorted(self.blocks, key=lambda b: (b.bbox.x, b.bbox.y))
        else:
            return sorted(self.blocks, key=lambda b: (b.bbox.y, b.bbox.x))

    def to_dict(self) -> dict:
        return {
            "page_index": self.page_index,
            "image_path": self.image_path,
            "width": self.width,
            "height": self.height,
            "direction": self.direction.value,
            "reading_order": self.reading_order.value,
            "note": self.note,
            "detected_page_number": self.detected_page_number,
            "full_text": self.full_text,
            "raw_text": self.raw_text,
            "blocks": [
                {
                    "id": b.id,
                    "bbox": b.bbox.to_list(),
                    "text": b.text,
                    "direction": b.direction.value,
                    "order_index": b.order_index,
                    "note": b.note,
                    "lines": [
                        {
                            "text": line.text,
                            "confidence": line.confidence,
                            "bbox": line.bbox.to_list(),
                            "uncertain_spans": [
                                {
                                    "text": us.text,
                                    "start_char": us.start_char,
                                    "end_char": us.end_char,
                                    "confidence": us.confidence,
                                    "substituted": us.substituted,
                                }
                                for us in line.uncertain_spans
                            ],
                        }
                        for line in b.lines
                    ],
                }
                for b in self.blocks
            ],
        }
