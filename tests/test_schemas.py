"""Tests for data structures (schemas)."""

import json

import pytest
from agent_qw.schemas import (
    BBox, OCRLine, Block, Page, UncertainSpan,
    TextDirection, ReadingOrder,
)


class TestBBox:
    def test_coordinate_conversions(self):
        b = BBox(x=10, y=20, w=100, h=50)
        assert b.x2 == 110
        assert b.y2 == 70
        assert b.to_list() == [10, 20, 100, 50]
        assert b.to_xyxy() == [10, 20, 110, 70]

    def test_from_xyxy(self):
        b = BBox.from_xyxy(5, 5, 55, 35)
        assert b.to_list() == [5, 5, 50, 30]


class TestPage:
    def test_to_dict_produces_valid_json(self):
        page = Page(
            page_index=0,
            image_path="test.pdf",
            width=1000,
            height=2000,
            direction=TextDirection.VERTICAL,
            reading_order=ReadingOrder.RIGHT_TO_LEFT,
            blocks=[
                Block(
                    id=0,
                    bbox=BBox(700, 0, 100, 2000),
                    text="子曰學而時習之",
                    direction=TextDirection.VERTICAL,
                    order_index=0,
                    lines=[
                        OCRLine(
                            text="子",
                            confidence=0.95,
                            bbox=BBox(700, 0, 100, 30),
                        ),
                        OCRLine(
                            text="曰",
                            confidence=0.60,
                            bbox=BBox(700, 35, 100, 30),
                            uncertain_spans=[
                                UncertainSpan(
                                    text="曰",
                                    start_char=0,
                                    end_char=1,
                                    confidence=0.60,
                                    substituted=True,
                                )
                            ],
                        ),
                    ],
                ),
                Block(
                    id=1,
                    bbox=BBox(550, 0, 100, 2000),
                    text="不亦樂乎",
                    direction=TextDirection.VERTICAL,
                    order_index=1,
                ),
            ],
            full_text="子曰學而時習之\n不亦樂乎",
        )
        d = page.to_dict()
        assert d["page_index"] == 0
        assert d["direction"] == "vertical"
        assert d["reading_order"] == "right-to-left"
        assert len(d["blocks"]) == 2
        assert d["blocks"][0]["text"] == "子曰學而時習之"
        assert d["blocks"][0]["bbox"] == [700, 0, 100, 2000]
        assert len(d["blocks"][0]["lines"]) == 2
        assert d["blocks"][0]["lines"][1]["confidence"] == 0.60
        assert d["blocks"][0]["lines"][1]["uncertain_spans"][0]["substituted"] is True
        assert d["full_text"] == "子曰學而時習之\n不亦樂乎"
        # JSON roundtrip
        json_str = json.dumps(d, ensure_ascii=False)
        parsed = json.loads(json_str)
        assert parsed["page_index"] == 0


class TestReadingOrder:
    def test_right_to_left_orders_blocks_correctly(self):
        page = Page(
            page_index=0,
            image_path="test.pdf",
            width=1000,
            height=1000,
            reading_order=ReadingOrder.RIGHT_TO_LEFT,
        )
        block_right = Block(
            id=1, bbox=BBox(x=800, y=0, w=100, h=1000),
            text="右", order_index=0,
        )
        block_left = Block(
            id=0, bbox=BBox(x=100, y=0, w=100, h=1000),
            text="左", order_index=1,
        )
        page.blocks = [block_left, block_right]
        ordered = page.ordered_blocks
        assert ordered[0].text == "右"
        assert ordered[1].text == "左"
