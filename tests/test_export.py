"""Tests for export functionality."""

import json
import os
import tempfile

import pytest
from agent_qw.schemas import (
    BBox, OCRLine, Block, Page, ReadingOrder,
)
from agent_qw.export import export_txt, export_json


class TestExportTxt:
    def test_single_page(self):
        pages = [
            Page(
                page_index=0,
                image_path="test.pdf",
                width=1000,
                height=2000,
                reading_order=ReadingOrder.RIGHT_TO_LEFT,
                blocks=[
                    Block(
                        id=0, bbox=BBox(800, 0, 100, 2000),
                        text="子曰學而時習之",
                        order_index=0,
                    ),
                    Block(
                        id=1, bbox=BBox(600, 0, 100, 2000),
                        text="不亦樂乎",
                        order_index=1,
                    ),
                ],
                full_text="子曰學而時習之\n不亦樂乎",
            ),
        ]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            tmp_path = f.name

        try:
            result_path = export_txt(pages, tmp_path)
            assert os.path.exists(result_path)
            with open(result_path, "r", encoding="utf-8") as f:
                content = f.read()
            assert "子曰學而時習之" in content
            assert "不亦樂乎" in content
        finally:
            os.unlink(tmp_path)


class TestExportJson:
    def test_single_page(self):
        pages = [
            Page(
                page_index=0,
                image_path="test.pdf",
                width=1000,
                height=1000,
                reading_order=ReadingOrder.RIGHT_TO_LEFT,
                blocks=[
                    Block(
                        id=0, bbox=BBox(800, 0, 100, 1000),
                        text="天地玄黃",
                        order_index=0,
                        lines=[
                            OCRLine(
                                text="天",
                                confidence=0.95,
                                bbox=BBox(800, 0, 100, 20),
                            ),
                        ],
                    ),
                ],
                full_text="天地玄黃",
            ),
        ]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            tmp_path = f.name

        try:
            result_path = export_json(pages, tmp_path)
            assert os.path.exists(result_path)
            with open(result_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert data["version"] == "0.1.0"
            assert data["page_count"] == 1
            assert len(data["pages"]) == 1
            assert data["pages"][0]["full_text"] == "天地玄黃"
            assert data["pages"][0]["blocks"][0]["lines"][0]["confidence"] == 0.95
        finally:
            os.unlink(tmp_path)


class TestMultiPageExport:
    def test_three_pages(self):
        pages = [
            Page(
                page_index=i,
                image_path=f"page_{i}.png",
                width=500, height=800,
                blocks=[
                    Block(
                        id=0, bbox=BBox(100, 0, 100, 800),
                        text=f"第{i + 1}頁內容",
                        order_index=0,
                    ),
                ],
                full_text=f"第{i + 1}頁",
            )
            for i in range(3)
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            txt_path = export_txt(pages, os.path.join(tmpdir, "result.txt"))
            json_path = export_json(pages, os.path.join(tmpdir, "result.json"))
            with open(txt_path, "r", encoding="utf-8") as f:
                txt = f.read()
            assert "第1頁" in txt
            assert "第3頁" in txt
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert data["page_count"] == 3
