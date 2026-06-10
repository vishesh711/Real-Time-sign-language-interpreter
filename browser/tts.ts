/**
 * Browser TTS integration using the Web Speech API (SpeechSynthesisUtterance).
 *
 * Implements a text serialize/deserialize fidelity check that runs before any
 * speech synthesis begins — blocking synthesis if the round-trip produces a
 * different string than the original (Requirement 5.5).
 *
 * Serialization strategy: JSON.stringify / JSON.parse — this preserves all
 * Unicode code points, whitespace, punctuation, and control characters
 * exactly, giving a true round-trip for any JavaScript string.
 *
 * Requirements: 5.2, 5.5
 */

// ---------------------------------------------------------------------------
// Error class
// ---------------------------------------------------------------------------

/**
 * Thrown when the TTS fidelity check fails, i.e. when
 * `deserializeTtsText(serializeTtsText(text)) !== text`.
 *
 * Requirements: 5.5
 */
export class TtsFidelityError extends Error {
  constructor(original: string, roundTripped: string) {
    super(
      `TTS fidelity check failed: original and round-tripped strings differ.\n` +
        `  original    : ${JSON.stringify(original)}\n` +
        `  round-tripped: ${JSON.stringify(roundTripped)}`,
    );
    this.name = 'TtsFidelityError';
  }
}

// ---------------------------------------------------------------------------
// Serialize / Deserialize
// ---------------------------------------------------------------------------

/**
 * Serialize `text` to the TTS input format.
 *
 * Uses JSON.stringify so that all Unicode, punctuation, and whitespace are
 * preserved exactly and the result is a self-describing, round-trippable
 * string.
 *
 * Requirements: 5.5
 */
export function serializeTtsText(text: string): string {
  return JSON.stringify(text);
}

/**
 * Deserialize a previously serialized TTS input string back to the original
 * text.
 *
 * Requirements: 5.5
 */
export function deserializeTtsText(serialized: string): string {
  return JSON.parse(serialized) as string;
}

// ---------------------------------------------------------------------------
// Fidelity check
// ---------------------------------------------------------------------------

/**
 * Run the serialize/deserialize fidelity check.
 * Returns the deserialized string when the round-trip succeeds.
 * Throws `TtsFidelityError` if `deserialize(serialize(text)) !== text`.
 *
 * Requirements: 5.5
 */
function checkFidelity(text: string): string {
  const serialized = serializeTtsText(text);
  const deserialized = deserializeTtsText(serialized);
  if (deserialized !== text) {
    throw new TtsFidelityError(text, deserialized);
  }
  return deserialized;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Speak `text` via the Web Speech API.
 *
 * Steps:
 *  1. Run the serialize/deserialize fidelity check (Requirement 5.5).
 *  2. If the check fails, throw `TtsFidelityError` and do not call
 *     `speechSynthesis.speak`.
 *  3. If the check passes, create a `SpeechSynthesisUtterance` and speak it.
 *
 * Requirements: 5.2, 5.5
 */
export function speak(text: string): void {
  // Fidelity check — throws TtsFidelityError on failure
  checkFidelity(text);

  const utterance = new SpeechSynthesisUtterance(text);
  window.speechSynthesis.speak(utterance);
}

/**
 * Cancel any ongoing or pending speech synthesis.
 *
 * Requirements: 5.2
 */
export function cancelSpeech(): void {
  window.speechSynthesis.cancel();
}
