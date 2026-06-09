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
