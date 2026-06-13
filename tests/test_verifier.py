"""Tests for agent_verify core functions (no LLM calls)."""

import pytest
from agent_verify.verifier import extract_head_tail, _build_batch_prompt, _extract_json


class TestExtractHeadTail:
    def test_normal_page(self):
        text = "A" * 200 + "MIDDLE" + "B" * 200
        head, tail = extract_head_tail(text)
        assert len(head) == 80
        assert len(tail) == 80
        assert head == "A" * 80
        assert tail == "B" * 80

    def test_short_page_returns_all(self):
        short = "Hello World"
        head, tail = extract_head_tail(short)
        assert head == "HelloWorld"
        assert tail == "HelloWorld"

    def test_empty_string(self):
        head, tail = extract_head_tail("")
        assert head == ""
        assert tail == ""

    def test_collapses_whitespace(self):
        text = "A  B\tC\nD\r\nE" * 30
        head, tail = extract_head_tail(text)
        assert "\t" not in head
        assert "\r" not in head
        assert "  " not in head


class TestBuildBatchPrompt:
    def test_basic_batch(self):
        pages = [("head1", "tail1"), ("head2", "tail2")]
        prompt = _build_batch_prompt(pages)
        assert "本批共 2 页" in prompt
        assert "[页1]" in prompt
        assert "[页2]" in prompt
        assert "head1" in prompt
        assert "tail2" in prompt
        assert "JSON" in prompt

    def test_with_prev_tail(self):
        pages = [("head1", "tail1")]
        prompt = _build_batch_prompt(pages, prev_tail="prev_tail_text")
        assert "上一批最后一页的尾部文字" in prompt
        assert "prev_tail_text" in prompt
        assert "连贯" in prompt

    def test_empty_batch(self):
        prompt = _build_batch_prompt([])
        assert "本批共 0 页" in prompt


class TestExtractJson:
    def test_plain_json(self):
        result = _extract_json('{"order": [1, 2, 3]}')
        assert result == {"order": [1, 2, 3]}

    def test_fenced_json(self):
        result = _extract_json('```json\n{"order": [3, 1, 2]}\n```')
        assert result == {"order": [3, 1, 2]}

    def test_nested_braces(self):
        result = _extract_json('{"order": [1, 2], "notes": "包含{特殊}字符"}')
        assert result == {"order": [1, 2], "notes": "包含{特殊}字符"}

    def test_escaped_quotes_in_json(self):
        result = _extract_json('{"key": "value with \\"escaped\\" quotes"}')
        assert result is not None
        assert result["key"] == 'value with "escaped" quotes'

    def test_invalid_json_returns_none(self):
        result = _extract_json("not json at all")
        assert result is None

    def test_json_with_surrounding_text(self):
        result = _extract_json('Some text {"order": [1, 2]} more text')
        assert result == {"order": [1, 2]}

    def test_no_braces(self):
        result = _extract_json("no braces here")
        assert result is None
