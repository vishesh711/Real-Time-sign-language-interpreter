/**
 * Property-based tests for browser landmark normalization utilities.
 *
 * Feature: sign-language-interpreter, Property 4: Word-level feature vector has correct shape
 * Validates: Requirements 3.1
 */

import * as fc from 'fast-check';
import { describe, it, expect } from 'vitest';
import {
  normalizeLandmarks,
  buildTwoHandVector,
  type HandLandmarks,
} from '../../browser/landmarks';

// ---------------------------------------------------------------------------
// Generators
// ---------------------------------------------------------------------------

/** Generate a valid 21-point hand landmark array with a non-degenerate scale. */
const validHandLandmarks = (): fc.Arbitrary<HandLandmarks> =>
  fc
    .array(
      fc.tuple(
        fc.float({ min: Math.fround(0), max: Math.fround(1), noNaN: true }),
        fc.float({ min: Math.fround(0), max: Math.fround(1), noNaN: true }),
        fc.float({ min: Math.fround(-0.1), max: Math.fround(0.1), noNaN: true }),
      ),
      { minLength: 21, maxLength: 21 },
    )
    .map((pts) => {
      // Ensure wrist→index-MCP distance is non-degenerate
      const out = pts as HandLandmarks;
      out[5] = [out[0][0] + 0.1, out[0][1], out[0][2]];
      return out;
    });

// ---------------------------------------------------------------------------
// Property 4 — normalizeLandmarks output shape
// ---------------------------------------------------------------------------

describe('normalizeLandmarks', () => {
  it(
    /**
     * Feature: sign-language-interpreter, Property 4: Word-level feature vector has correct shape
     * Validates: Requirements 3.1
     *
     * For any valid (21, 3) hand landmark input, normalizeLandmarks() must return
     * either null (degenerate) or a Float32Array of exactly 63 floats.
     */
    'Property 4: returns 63-float vector for valid hand landmarks',
    () => {
      fc.assert(
        fc.property(validHandLandmarks(), (lm) => {
          const result = normalizeLandmarks(lm, false);
          if (result !== null) {
            expect(result).toBeInstanceOf(Float32Array);
            expect(result.length).toBe(63);
          }
        }),
        { numRuns: 100 },
      );
    },
  );

  it(
    /**
     * Feature: sign-language-interpreter, Property 4: Word-level feature vector has correct shape
     * Validates: Requirements 3.1
     *
     * Mirrored (left-hand) path must produce the same output length.
     */
    'Property 4: mirrored path returns same length',
    () => {
      fc.assert(
        fc.property(validHandLandmarks(), (lm) => {
          const r = normalizeLandmarks(lm, false);
          const l = normalizeLandmarks(lm, true);
          if (r !== null && l !== null) {
            expect(l.length).toBe(r.length);
          }
        }),
        { numRuns: 100 },
      );
    },
  );
});

// ---------------------------------------------------------------------------
// Property 4 — buildTwoHandVector output shape
// ---------------------------------------------------------------------------

describe('buildTwoHandVector', () => {
  it(
    /**
     * Feature: sign-language-interpreter, Property 4: Word-level feature vector has correct shape
     * Validates: Requirements 3.1
     *
     * buildTwoHandVector() must always return a Float32Array of exactly 126
     * floats regardless of which combination of hands is provided.
     */
    'Property 4: returns 126-float vector for any hand combination',
    () => {
      fc.assert(
        fc.property(
          fc.oneof(validHandLandmarks(), fc.constant(null)),
          fc.oneof(validHandLandmarks(), fc.constant(null)),
          (right, left) => {
            const vec = buildTwoHandVector(right, left);
            expect(vec).toBeInstanceOf(Float32Array);
            expect(vec.length).toBe(126);
          },
        ),
        { numRuns: 100 },
      );
    },
  );

  it(
    /**
     * Feature: sign-language-interpreter, Property 4: Word-level feature vector has correct shape
     * Validates: Requirements 3.1
     *
     * When both hands are null, the output must be all zeros.
     */
    'Property 4: null inputs produce all-zero 126-float vector',
    () => {
      const vec = buildTwoHandVector(null, null);
      expect(vec.length).toBe(126);
      expect(Array.from(vec).every((v) => v === 0)).toBe(true);
    },
  );
});

// ---------------------------------------------------------------------------
// Property 9 — collapseAndSegment (fingerspelling collapse correctness)
// ---------------------------------------------------------------------------

import { collapseAndSegment } from '../../browser/postprocess';

const MIN_LETTER_HOLD = 4;
const PAUSE_THRESHOLD = 8;

/**
 * Build a stream with one letter repeated `runLen` times followed by
 * `gapLen` null frames.
 */
function makeStream(
  letter: string,
  runLen: number,
  gapLen: number,
): (string | null)[] {
  return [
    ...Array(runLen).fill(letter),
    ...Array(gapLen).fill(null),
  ];
}

describe('collapseAndSegment — Property 9', () => {
  /**
   * Feature: sign-language-interpreter, Property 9: Fingerspelling collapse correctness
   * Validates: Requirements 5.7
   *
   * Property 9a: For any single-letter run of length ≥ minLetterHold followed
   * by ≥ pauseThreshold nulls, the function produces exactly one word
   * containing exactly one letter.
   */
  it(
    'Property 9a: run ≥ minLetterHold + gap ≥ pauseThreshold → exactly one word with one letter',
    () => {
      fc.assert(
        fc.property(
          // A single uppercase letter
          fc.constantFrom(...'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('')),
          // Run length ≥ minLetterHold
          fc.integer({ min: MIN_LETTER_HOLD, max: 20 }),
          // Gap length ≥ pauseThreshold
          fc.integer({ min: PAUSE_THRESHOLD, max: 20 }),
          (letter, runLen, gapLen) => {
            const stream = makeStream(letter, runLen, gapLen);
            const words = collapseAndSegment(stream, PAUSE_THRESHOLD, MIN_LETTER_HOLD);

            // Exactly one word
            expect(words).toHaveLength(1);
            // That word contains exactly one letter (the collapsed run)
            expect(words[0]).toBe(letter);
          },
        ),
        { numRuns: 200 },
      );
    },
  );

  /**
   * Feature: sign-language-interpreter, Property 9: Fingerspelling collapse correctness
   * Validates: Requirements 5.7
   *
   * Property 9b: For any run of length < minLetterHold, the letter is NOT
   * included in any output word.
   */
  it(
    'Property 9b: run < minLetterHold → letter absent from all output words',
    () => {
      fc.assert(
        fc.property(
          // A single uppercase letter
          fc.constantFrom(...'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('')),
          // Run length strictly less than minLetterHold (1..3)
          fc.integer({ min: 1, max: MIN_LETTER_HOLD - 1 }),
          // Trailing gap of any length (0 or more)
          fc.integer({ min: 0, max: 20 }),
          (letter, runLen, trailingGap) => {
            const stream: (string | null)[] = [
              ...Array(runLen).fill(letter),
              ...Array(trailingGap).fill(null),
            ];
            const words = collapseAndSegment(stream, PAUSE_THRESHOLD, MIN_LETTER_HOLD);

            // The letter should NOT appear in any reconstructed word
            for (const word of words) {
              expect(word).not.toContain(letter);
            }
          },
        ),
        { numRuns: 200 },
      );
    },
  );

  /**
   * Feature: sign-language-interpreter, Property 9: Fingerspelling collapse correctness
   * Validates: Requirements 5.7
   *
   * Property 9c: For any gap of ≥ pauseThreshold nulls between two letter
   * groups (each with runLen ≥ minLetterHold), the output has at least two
   * words — one per group.
   */
  it(
    'Property 9c: gap ≥ pauseThreshold between two valid groups → at least two words',
    () => {
      fc.assert(
        fc.property(
          // First letter
          fc.constantFrom(...'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('')),
          // Second letter (may equal the first — still two separate words)
          fc.constantFrom(...'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('')),
          // Run lengths ≥ minLetterHold for both groups
          fc.integer({ min: MIN_LETTER_HOLD, max: 20 }),
          fc.integer({ min: MIN_LETTER_HOLD, max: 20 }),
          // Gap between the groups ≥ pauseThreshold
          fc.integer({ min: PAUSE_THRESHOLD, max: 20 }),
          (letterA, letterB, runA, runB, gap) => {
            const stream: (string | null)[] = [
              ...Array(runA).fill(letterA),
              ...Array(gap).fill(null),
              ...Array(runB).fill(letterB),
            ];
            const words = collapseAndSegment(stream, PAUSE_THRESHOLD, MIN_LETTER_HOLD);

            // At least two words must be produced
            expect(words.length).toBeGreaterThanOrEqual(2);
          },
        ),
        { numRuns: 200 },
      );
    },
  );

  /**
   * Feature: sign-language-interpreter, Property 9: Fingerspelling collapse correctness
   * Validates: Requirements 5.7
   *
   * Unit test: Spell "HELLO" using the stream:
   *   H×4, null×8, E×4, null×8, L×4, null×4, L×4, null×8, O×4
   *
   * Each null×8 gap equals pauseThreshold and therefore creates a word
   * boundary, so the algorithm produces four single-letter words for H, E,
   * and O plus one two-letter word for the double-L.
   *
   * The null×4 gap between the two L runs is below pauseThreshold so both L
   * runs belong to the same word segment. Because the second L run re-starts
   * a fresh identical run (runLetter reset to null during the gap), both runs
   * are flushed separately and each meets minLetterHold, giving "LL" — i.e.
   * the double-L is preserved (not over-collapsed to a single L).
   *
   * Expected: ["H", "E", "LL", "O"]
   */
  it('Unit: HELLO stream produces ["H", "E", "LL", "O"] — double-L preserved', () => {
    const stream: (string | null)[] = [
      // H  (run length 4 = minLetterHold)
      'H', 'H', 'H', 'H',
      // word boundary (8 nulls = pauseThreshold)
      null, null, null, null, null, null, null, null,
      // E  (run length 4)
      'E', 'E', 'E', 'E',
      // word boundary (8 nulls)
      null, null, null, null, null, null, null, null,
      // first L  (run length 4)
      'L', 'L', 'L', 'L',
      // short gap (4 nulls < pauseThreshold) — same word segment
      null, null, null, null,
      // second L  (run length 4 — new run after gap, also meets minLetterHold)
      'L', 'L', 'L', 'L',
      // word boundary (8 nulls)
      null, null, null, null, null, null, null, null,
      // O  (run length 4)
      'O', 'O', 'O', 'O',
    ];

    const words = collapseAndSegment(stream, PAUSE_THRESHOLD, MIN_LETTER_HOLD);
    // Each null×8 gap is a word boundary → 4 words: H, E, LL, O
    expect(words).toEqual(['H', 'E', 'LL', 'O']);
  });
});
