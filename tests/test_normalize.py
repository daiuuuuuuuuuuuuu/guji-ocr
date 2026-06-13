"""Tests for text normalization functions."""

import pytest
from agent_qw.postprocess.normalize import normalize_text, normalize_page_text


class TestNormalizeText:
    def test_empty_string(self):
        assert normalize_text("") == ""

    def test_collapse_multiple_spaces(self):
        result = normalize_text("hello    world")
        assert result == "hello world"

    def test_collapse_tabs_and_spaces(self):
        result = normalize_text("hello  \t  world")
        assert result == "hello world"

    def test_remove_control_characters(self):
        result = normalize_text("text\x00\x1f\x7fend")
        assert result == "textend"

    def test_preserve_newlines(self):
        result = normalize_text("line1\n\nline2")
        assert result == "line1\n\nline2"

    def test_collapse_excessive_newlines(self):
        result = normalize_text("a\n\n\n\n\nb")
        assert result == "a\n\nb"

    def test_strip_trailing_whitespace_per_line(self):
        result = normalize_text("hello   \nworld   ")
        assert result == "hello\nworld"

    def test_strip_leading_trailing_blank_lines(self):
        result = normalize_text("\n\nhello\n\n")
        assert result == "hello"

    def test_fullwidth_space_to_regular(self):
        result = normalize_text("hello\u3000world")
        assert result == "hello world"

    def test_preserve_chinese_characters(self):
        result = normalize_text("學而時習之　不亦楽乎")
        assert "學" in result
        assert "楽" in result


class TestNormalizePageText:
    def test_empty_string(self):
        assert normalize_page_text("") == ""

    def test_removes_empty_lines_keeps_content(self):
        result = normalize_page_text("line1\n\n\nline2\n\n\n\nline3")
        assert result == "line1\n\nline2\n\nline3"

    def test_no_consecutive_blank_lines(self):
        result = normalize_page_text("a\n\n\n\n\nb")
        lines = result.split("\n")
        blank_count = sum(1 for l in lines if l == "")
        assert blank_count <= 1
