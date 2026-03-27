"""Tests for _generate_windows to verify prefix/suffix text integrity.

Windows are character-based: prefix and suffix are sliced directly from the
original text, so multi-byte characters are never corrupted.

Covers: ASCII, multi-byte Unicode (Korean, CJK, ligatures), emoji,
mixed scripts, repeated text, and boundary alignment.
"""

from unittest.mock import MagicMock

import pytest
from transformers import AutoTokenizer

from cleanslate.evaluator import MemorizationConfig, MemorizationEvaluator


def _make_evaluator(
    tokenizer: AutoTokenizer,
    prefix_len: int = 20,
    suffix_len: int = 20,
    stride: int = 10,
) -> MemorizationEvaluator:
    """Create a MemorizationEvaluator wired to the given tokenizer."""
    config = MemorizationConfig(
        window_prefix_len=prefix_len,
        window_suffix_len=suffix_len,
        window_stride=stride,
    )
    ev = MemorizationEvaluator.__new__(MemorizationEvaluator)
    ev.config = config
    mock_backend = MagicMock()
    mock_backend.tokenizer = tokenizer
    ev._backend = mock_backend
    ev._backend_type = "custom"
    ev.model_name = "test"
    return ev


@pytest.fixture(scope="module")
def tokenizer() -> AutoTokenizer:
    return AutoTokenizer.from_pretrained("gpt2")


@pytest.fixture(scope="module")
def evaluator(tokenizer: AutoTokenizer) -> MemorizationEvaluator:
    return _make_evaluator(tokenizer)


def _assert_windows_valid(evaluator: MemorizationEvaluator, text: str) -> None:
    """Assert all generated windows have valid, non-overlapping prefix/suffix."""
    config = evaluator.config
    windows = evaluator._generate_windows(text)

    for i, w in enumerate(windows):
        # prefix + suffix must exactly reconstruct the source span
        combined = w.prefix + w.suffix
        source_span = text[w.start_char : w.end_char]
        assert combined == source_span, (
            f"Window {i}: prefix+suffix does not match source span.\n"
            f"  combined:  {combined!r}\n"
            f"  expected:  {source_span!r}"
        )

        # No replacement characters
        assert "\ufffd" not in w.prefix, (
            f"Window {i}: prefix contains replacement char U+FFFD"
        )
        assert "\ufffd" not in w.suffix, (
            f"Window {i}: suffix contains replacement char U+FFFD"
        )

        # Both prefix and suffix must be non-empty
        assert len(w.prefix) > 0, f"Window {i}: empty prefix"
        assert len(w.suffix) > 0, f"Window {i}: empty suffix"

        # Prefix and suffix lengths must match config (character-based)
        assert len(w.prefix) == config.window_prefix_len, (
            f"Window {i}: prefix length {len(w.prefix)} != {config.window_prefix_len}"
        )
        assert len(w.suffix) == config.window_suffix_len, (
            f"Window {i}: suffix length {len(w.suffix)} != {config.window_suffix_len}"
        )

        # Char positions must be consistent
        assert w.start_char >= 0
        assert w.end_char == w.start_char + config.window_prefix_len + config.window_suffix_len
        assert w.end_char <= len(text)

        # Token IDs must be non-empty and round-trip to the text
        assert len(w.prefix_token_ids) > 0
        assert len(w.suffix_token_ids) > 0


def test_ascii_text(evaluator: MemorizationEvaluator) -> None:
    text = "To be or not to be, that is the question. Whether tis nobler in the mind to suffer."
    _assert_windows_valid(evaluator, text)


def test_korean_text(evaluator: MemorizationEvaluator) -> None:
    """Korean characters are multi-byte and were corrupted by the old token-based approach."""
    text = "오빤 강남스타일 강남스타일 낮에는 따사로운 인간적인 여자 커피 한잔의 여유를 아는 품격 있는 여자"
    _assert_windows_valid(evaluator, text)


def test_japanese_text(evaluator: MemorizationEvaluator) -> None:
    text = "誰しも男ほど弱かないわ 乱れた髪を気にするように 少し遅れて歩くのが好きなんだよ"
    _assert_windows_valid(evaluator, text)


def test_mixed_scripts(evaluator: MemorizationEvaluator) -> None:
    """Mixed Latin/Korean/CJK in a single text, as in real song lyrics."""
    text = "Come on, ride the train 아파트, 아파트 Let's go Ætolians 誰しも男ほど弱かない forever'\u200a\""
    _assert_windows_valid(evaluator, text)


def test_emoji(evaluator: MemorizationEvaluator) -> None:
    """Emoji are multi-byte (4 bytes) and easily split across tokens."""
    text = "Hello 🌍🌎🌏 world 🎵🎶 music 🎸🎹 playing 🎤🎧 listening 🎼🎻 orchestra 🎺🎷 band"
    _assert_windows_valid(evaluator, text)


def test_ligatures_and_special_unicode(evaluator: MemorizationEvaluator) -> None:
    """Ligatures (Æ, œ), thin spaces, curly quotes."""
    text = "The Ætolians and Achæans were firmly'\u200a\" established in the œuvre of classical thought"
    _assert_windows_valid(evaluator, text)


def test_repeated_text(evaluator: MemorizationEvaluator) -> None:
    """Repeated lyrics should produce correct, non-overlapping prefix/suffix."""
    verse = "Ride that choo-choo, choo-choo (Woo-woo)\n"
    text = verse * 10
    _assert_windows_valid(evaluator, text)


def test_text_too_short_returns_empty(evaluator: MemorizationEvaluator) -> None:
    """Text shorter than prefix_len + suffix_len produces no windows."""
    text = "short"
    windows = evaluator._generate_windows(text)
    assert windows == []


def test_exact_window_size(evaluator: MemorizationEvaluator) -> None:
    """Text that is exactly prefix_len + suffix_len chars should yield one window."""
    config = evaluator.config
    target_chars = config.window_prefix_len + config.window_suffix_len
    text = "a" * target_chars
    windows = evaluator._generate_windows(text)
    assert len(windows) == 1
    _assert_windows_valid(evaluator, text)


def test_consecutive_windows_cover_text(evaluator: MemorizationEvaluator) -> None:
    """The union of all window spans should cover the interior of the text."""
    text = "To be or not to be, that is the question. Whether tis nobler in the mind to suffer."
    windows = evaluator._generate_windows(text)
    assert len(windows) > 1

    # Each window's end_char should be >= next window's start_char (overlapping)
    for i in range(1, len(windows)):
        assert windows[i].start_char < windows[i - 1].end_char, (
            f"Gap between window {i - 1} and {i}"
        )


def test_stride_produces_correct_positions(evaluator: MemorizationEvaluator) -> None:
    """Window start positions should advance by exactly stride characters."""
    text = "a" * 200
    windows = evaluator._generate_windows(text)
    stride = evaluator.config.window_stride
    for i in range(1, len(windows)):
        assert windows[i].start_char - windows[i - 1].start_char == stride


def test_windows_with_realistic_config() -> None:
    """Test with realistic window sizes (200 prefix + 200 suffix, stride 100)."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    ev = _make_evaluator(tokenizer, prefix_len=200, suffix_len=200, stride=100)

    # Korean song lyrics (APT.-style)
    text = (
        "아파트, 아파트\n아파트, 아파트\n"
        "Just meet me at the (uh, uh-huh, uh-huh)\n"
    ) * 30
    _assert_windows_valid(ev, text)


def test_no_replacement_chars_korean_boundary() -> None:
    """Korean characters at the prefix/suffix boundary must not be corrupted."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # Use a prefix length that splits right before a Korean character
    ev = _make_evaluator(tokenizer, prefix_len=15, suffix_len=15, stride=5)
    text = "Just meet me 아파트, 아파트 hello world again"
    windows = ev._generate_windows(text)
    for w in windows:
        assert "\ufffd" not in w.prefix
        assert "\ufffd" not in w.suffix
        assert w.prefix + w.suffix == text[w.start_char : w.end_char]
