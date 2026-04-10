"""Tests for character-based sliding window generation."""

import pytest

from cleanslate.evaluator import _generate_windows


def test_basic_windowing():
    text = "abcdefghij" * 50  # 500 chars
    windows = _generate_windows(text, prefix_len=100, suffix_len=100, stride=10)

    assert len(windows) == 31
    assert windows[0].start_char == 0
    assert windows[0].end_char == 200
    assert windows[-1].end_char == 500


def test_stride():
    text = "x" * 300
    windows = _generate_windows(text, prefix_len=50, suffix_len=50, stride=25)

    for i in range(1, len(windows)):
        assert windows[i].start_char - windows[i - 1].start_char == 25


def test_prefix_suffix_lengths():
    text = "a" * 400
    windows = _generate_windows(text, prefix_len=80, suffix_len=120, stride=10)

    for w in windows:
        assert len(w.prefix) == 80
        assert len(w.suffix) == 120


def test_reconstruction_matches_original():
    text = "The quick brown fox jumps over the lazy dog. " * 20
    windows = _generate_windows(text, prefix_len=100, suffix_len=100, stride=10)

    for w in windows:
        assert w.prefix + w.suffix == text[w.start_char : w.end_char]


def test_text_too_short():
    assert _generate_windows("short", prefix_len=100, suffix_len=100, stride=10) == []


def test_exact_fit():
    text = "a" * 200
    windows = _generate_windows(text, prefix_len=100, suffix_len=100, stride=10)

    assert len(windows) == 1
    assert windows[0].start_char == 0
    assert windows[0].end_char == 200


def test_chinese():
    text = "这是一个测试文本用于验证滑动窗口在中文环境下的正确性" * 3
    windows = _generate_windows(text, prefix_len=20, suffix_len=20, stride=5)

    assert len(windows) > 0
    for w in windows:
        assert len(w.prefix) == 20
        assert len(w.suffix) == 20
        assert w.prefix + w.suffix == text[w.start_char : w.end_char]


def test_emoji():
    text = "🎵🎶🎤🎸🎹🎺🎻🥁🪘🎷" * 10
    windows = _generate_windows(text, prefix_len=10, suffix_len=10, stride=5)

    assert len(windows) > 0
    for w in windows:
        assert len(w.prefix) == 10
        assert len(w.suffix) == 10
        assert w.prefix + w.suffix == text[w.start_char : w.end_char]


def test_mixed_scripts():
    text = "안녕하세요 Hello こんにちは World 你好世界 " * 10
    windows = _generate_windows(text, prefix_len=15, suffix_len=15, stride=5)

    assert len(windows) > 0
    for w in windows:
        assert len(w.prefix) == 15
        assert len(w.suffix) == 15
        assert w.prefix + w.suffix == text[w.start_char : w.end_char]


def test_arabic():
    text = "مرحبا بالعالم هذا نص اختبار للتحقق من صحة النوافذ المنزلقة " * 5
    windows = _generate_windows(text, prefix_len=20, suffix_len=20, stride=10)

    assert len(windows) > 0
    for w in windows:
        assert len(w.prefix) == 20
        assert len(w.suffix) == 20
        assert w.prefix + w.suffix == text[w.start_char : w.end_char]
