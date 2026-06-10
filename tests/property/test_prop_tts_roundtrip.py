"""
Property-based test for TTS text round-trip.

# Feature: sign-language-interpreter, Property 8: TTS text round-trip
**Validates: Requirements 5.5**

Property 8 — TTS text round-trip:
    For any string s (including Unicode, punctuation, and whitespace),
    serializing s to the TTS input format and deserializing it back SHALL
    produce a string equal to s.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from utils.tts_server import deserialize_tts_text, serialize_tts_text

# ---------------------------------------------------------------------------
# Property 8 — TTS text round-trip
# **Feature: sign-language-interpreter, Property 8: TTS text round-trip**
# **Validates: Requirements 5.5**
# ---------------------------------------------------------------------------


@given(s=st.text())
@settings(max_examples=100)
def test_tts_roundtrip_arbitrary_text(s: str) -> None:
    """
    # Feature: sign-language-interpreter, Property 8: TTS text round-trip
    **Validates: Requirements 5.5**

    For any string s (covering Unicode, punctuation, whitespace, and control
    characters), deserialize_tts_text(serialize_tts_text(s)) SHALL equal s.
    """
    serialized = serialize_tts_text(s)
    deserialized = deserialize_tts_text(serialized)
    assert deserialized == s, (
        f"TTS round-trip failed:\n"
        f"  original    : {s!r}\n"
        f"  round-tripped: {deserialized!r}"
    )


@given(s=st.text(alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "P", "S", "Z"))))
@settings(max_examples=100)
def test_tts_roundtrip_letters_digits_punctuation(s: str) -> None:
    """
    # Feature: sign-language-interpreter, Property 8: TTS text round-trip
    **Validates: Requirements 5.5**

    Round-trip holds for strings composed of letters, digits, punctuation,
    symbols, and separators (Unicode categories Lu/Ll/Nd/P/S/Z).
    """
    serialized = serialize_tts_text(s)
    deserialized = deserialize_tts_text(serialized)
    assert deserialized == s, (
        f"TTS round-trip failed for letters/punctuation:\n"
        f"  original    : {s!r}\n"
        f"  round-tripped: {deserialized!r}"
    )


# ---------------------------------------------------------------------------
# Unit tests — specific edge cases (Requirement 5.5)
# ---------------------------------------------------------------------------


def test_tts_roundtrip_empty_string() -> None:
    """Round-trip preserves the empty string."""
    assert deserialize_tts_text(serialize_tts_text("")) == ""


def test_tts_roundtrip_ascii_sentence() -> None:
    """Round-trip preserves a plain ASCII sentence."""
    s = "Hello, world!"
    assert deserialize_tts_text(serialize_tts_text(s)) == s


def test_tts_roundtrip_newline_and_tab() -> None:
    """Round-trip preserves newline and tab characters."""
    s = "line1\nline2\ttabbed"
    assert deserialize_tts_text(serialize_tts_text(s)) == s


def test_tts_roundtrip_unicode_cjk() -> None:
    """Round-trip preserves CJK Unicode characters."""
    s = "手話インタープリター"
    assert deserialize_tts_text(serialize_tts_text(s)) == s


def test_tts_roundtrip_emoji() -> None:
    """Round-trip preserves emoji (supplementary plane code points)."""
    s = "👋🤟🖐️"
    assert deserialize_tts_text(serialize_tts_text(s)) == s


def test_tts_roundtrip_quotes_and_backslash() -> None:
    """Round-trip preserves double quotes and backslashes."""
    s = '"Hello", she said. \\ backslash'
    assert deserialize_tts_text(serialize_tts_text(s)) == s


def test_tts_roundtrip_control_characters() -> None:
    """Round-trip preserves control characters (NUL, DEL, etc.)."""
    s = "\x00\x1f\x7f"
    assert deserialize_tts_text(serialize_tts_text(s)) == s


def test_serialize_tts_text_produces_nonempty_string() -> None:
    """serialize_tts_text always returns a non-empty string."""
    assert len(serialize_tts_text("")) > 0
    assert len(serialize_tts_text("abc")) > 0
