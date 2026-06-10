/**
 * Property-based tests for TTS text round-trip.
 *
 * Feature: sign-language-interpreter, Property 8: TTS text round-trip
 * Validates: Requirements 5.5
 *
 * Property 8 — TTS text round-trip:
 *   For any string s (including Unicode, punctuation, and whitespace),
 *   serializing s to the TTS input format and deserializing it back SHALL
 *   produce a string equal to s.
 */

import * as fc from 'fast-check';
import { describe, it, expect } from 'vitest';
import { serializeTtsText, deserializeTtsText } from '../../browser/tts';

// ---------------------------------------------------------------------------
// Property 8 — TTS text round-trip
// ---------------------------------------------------------------------------

describe('TTS — Property 8: TTS text round-trip', () => {
  it(
    /** Feature: sign-language-interpreter, Property 8: TTS text round-trip */
    'Property 8: deserializeTtsText(serializeTtsText(s)) === s for arbitrary ASCII strings',
    () => {
      fc.assert(
        fc.property(fc.string(), (s) => {
          const serialized = serializeTtsText(s);
          const deserialized = deserializeTtsText(serialized);
          expect(deserialized).toBe(s);
        }),
        { numRuns: 100 },
      );
    },
  );

  it(
    /** Feature: sign-language-interpreter, Property 8: TTS text round-trip */
    'Property 8: deserializeTtsText(serializeTtsText(s)) === s for arbitrary Unicode strings',
    () => {
      fc.assert(
        fc.property(fc.unicodeString(), (s) => {
          const serialized = serializeTtsText(s);
          const deserialized = deserializeTtsText(serialized);
          expect(deserialized).toBe(s);
        }),
        { numRuns: 100 },
      );
    },
  );

  it(
    /** Feature: sign-language-interpreter, Property 8: TTS text round-trip */
    'Property 8: round-trip preserves empty string',
    () => {
      const s = '';
      expect(deserializeTtsText(serializeTtsText(s))).toBe(s);
    },
  );

  it(
    /** Feature: sign-language-interpreter, Property 8: TTS text round-trip */
    'Property 8: round-trip preserves strings with only whitespace and punctuation',
    () => {
      fc.assert(
        fc.property(
          fc.stringOf(fc.oneof(fc.constant(' '), fc.constant('\t'), fc.constant('\n'), fc.constant('!'), fc.constant('"'), fc.constant("'"), fc.constant('.'), fc.constant(','), fc.constant('?'))),
          (s) => {
            const serialized = serializeTtsText(s);
            const deserialized = deserializeTtsText(serialized);
            expect(deserialized).toBe(s);
          },
        ),
        { numRuns: 100 },
      );
    },
  );
});

// ---------------------------------------------------------------------------
// Unit tests — specific edge cases (Requirement 5.5)
// ---------------------------------------------------------------------------

describe('TTS — unit tests: serializeTtsText / deserializeTtsText', () => {
  it('round-trip: plain ASCII word', () => {
    const s = 'hello world';
    expect(deserializeTtsText(serializeTtsText(s))).toBe(s);
  });

  it('round-trip: string with newline and tab', () => {
    const s = 'line1\nline2\ttabbed';
    expect(deserializeTtsText(serializeTtsText(s))).toBe(s);
  });

  it('round-trip: string with emoji', () => {
    const s = '👋🤟🖐️';
    expect(deserializeTtsText(serializeTtsText(s))).toBe(s);
  });

  it('round-trip: string with CJK characters', () => {
    const s = '手話インタープリター';
    expect(deserializeTtsText(serializeTtsText(s))).toBe(s);
  });

  it('round-trip: string with control characters', () => {
    const s = '\u0000\u001f\u007f';
    expect(deserializeTtsText(serializeTtsText(s))).toBe(s);
  });

  it('round-trip: string with quotes and backslashes', () => {
    const s = '"Hello", she said. \\ backslash';
    expect(deserializeTtsText(serializeTtsText(s))).toBe(s);
  });

  it('serializeTtsText produces a non-empty string for any input', () => {
    expect(serializeTtsText('').length).toBeGreaterThan(0);
    expect(serializeTtsText('abc').length).toBeGreaterThan(0);
  });
});
