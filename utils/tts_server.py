"""
Server-side TTS fallback using pyttsx3.

Implements the same serialize/deserialize fidelity check as the browser TTS
module (browser/tts.ts) — confirming that the text survives a JSON round-trip
before speech synthesis begins (Requirement 5.5).

Serialization strategy: json.dumps / json.loads — preserves all Unicode,
punctuation, and whitespace exactly, giving a true round-trip for any Python
string.

Requirements: 5.2, 5.5
"""

import json


# ---------------------------------------------------------------------------
# Serialize / Deserialize
# ---------------------------------------------------------------------------


def serialize_tts_text(text: str) -> str:
    """Serialize *text* to the TTS input format.

    Uses ``json.dumps`` so that all Unicode, punctuation, and whitespace are
    preserved exactly in a self-describing, round-trippable representation.

    Requirements: 5.5
    """
    return json.dumps(text)


def deserialize_tts_text(serialized: str) -> str:
    """Deserialize a previously serialized TTS input string back to the
    original text.

    Requirements: 5.5
    """
    return json.loads(serialized)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def speak_server(text: str) -> None:
    """Speak *text* via pyttsx3 (server-side TTS fallback).

    Steps:
      1. Serialize *text* with :func:`serialize_tts_text`.
      2. Deserialize with :func:`deserialize_tts_text`.
      3. If the round-trip is not equal to *text*, raise ``ValueError``
         (Requirement 5.5).
      4. If the check passes, initialize pyttsx3, queue the text, and block
         until synthesis is complete.

    Requirements: 5.2, 5.5

    Raises:
        ValueError: When ``deserialize(serialize(text)) != text``.
        ImportError: When pyttsx3 is not installed in the current environment.
    """
    serialized = serialize_tts_text(text)
    deserialized = deserialize_tts_text(serialized)

    if deserialized != text:
        raise ValueError(
            f"TTS fidelity check failed: {text!r} != {deserialized!r}"
        )

    import pyttsx3  # imported here so the module is usable without pyttsx3 installed

    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
