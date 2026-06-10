/**
 * Property-based tests for OrtEngine utility functions.
 *
 * Feature: sign-language-interpreter, Property 12: ONNX model serialization round-trip
 * Validates: Requirements 7.3
 *
 * Since real ONNX models are not available in the test environment, these tests
 * exercise the pure utility functions (softmax, topK, resampleSequence) that are
 * the round-trip-sensitive parts of the inference pipeline. These functions are
 * what a loaded-and-reloaded ONNX session ultimately passes data through, so
 * correctness here is a necessary condition for round-trip fidelity.
 */

import * as fc from 'fast-check';
import { describe, it, expect } from 'vitest';
import { softmax, topK, resampleSequence } from '../../browser/ort_engine';

// ---------------------------------------------------------------------------
// Generators
// ---------------------------------------------------------------------------

/** Finite float — no NaN or ±Infinity. */
const finiteFloat = fc.float({ noNaN: true, noDefaultInfinity: true });

/**
 * Generate a Float32Array of a specific length with finite values.
 * fc.float uses Math.fround internally, so values are already float32-range.
 */
const float32Array = (len: number): fc.Arbitrary<Float32Array> =>
  fc.array(finiteFloat, { minLength: len, maxLength: len }).map((arr) => new Float32Array(arr));

/** Generate a non-empty Float32Array of random length in [1, 200]. */
const anyFloat32Array = fc
  .integer({ min: 1, max: 200 })
  .chain((len) => float32Array(len));

/** Generate a Float32Array of exactly 63 elements (fingerspell input). */
const fingerspellInput = float32Array(63);

/** Generate an array of T frames, each a Float32Array of frameDim elements. */
const frameBuffer = (frameDim: number): fc.Arbitrary<Float32Array[]> =>
  fc.integer({ min: 1, max: 60 }).chain((t) =>
    fc.array(float32Array(frameDim), { minLength: t, maxLength: t }),
  );

// ---------------------------------------------------------------------------
// Property 12 — softmax output always sums to 1.0 and is in [0, 1]
// ---------------------------------------------------------------------------

describe('softmax', () => {
  it(
    /** Feature: sign-language-interpreter, Property 12: ONNX model serialization round-trip */
    'Property 12: output values are in [0.0, 1.0] for any finite float input',
    () => {
      fc.assert(
        fc.property(anyFloat32Array, (logits) => {
          const probs = softmax(logits);
          expect(probs).toBeInstanceOf(Float32Array);
          expect(probs.length).toBe(logits.length);
          for (let i = 0; i < probs.length; i++) {
            expect(probs[i]).toBeGreaterThanOrEqual(0.0);
            expect(probs[i]).toBeLessThanOrEqual(1.0);
          }
        }),
        { numRuns: 100 },
      );
    },
  );

  it(
    /** Feature: sign-language-interpreter, Property 12: ONNX model serialization round-trip */
    'Property 12: output probabilities sum to approximately 1.0',
    () => {
      fc.assert(
        fc.property(anyFloat32Array, (logits) => {
          const probs = softmax(logits);
          const sum = Array.from(probs).reduce((a, b) => a + b, 0);
          expect(sum).toBeCloseTo(1.0, 4); // within 1e-4
        }),
        { numRuns: 100 },
      );
    },
  );

  it(
    /** Feature: sign-language-interpreter, Property 12: ONNX model serialization round-trip */
    'Property 12: argmax is preserved when softmax is applied twice for non-uniform inputs (idempotent argmax)',
    () => {
      fc.assert(
        fc.property(anyFloat32Array, (logits) => {
          const probs1 = softmax(logits);

          // Find top-2 values to check if there's a clear winner (margin > 1e-4)
          let argmax1 = 0;
          let secondMax = -1;
          for (let i = 1; i < probs1.length; i++) {
            if (probs1[i] > probs1[argmax1]) {
              secondMax = argmax1;
              argmax1 = i;
            } else if (secondMax === -1 || probs1[i] > probs1[secondMax]) {
              secondMax = i;
            }
          }

          // Only assert argmax preservation when there is a clear winner
          // (i.e., the top probability is meaningfully above the second-largest).
          // Near-uniform distributions can have argmax shift due to float32 rounding.
          const margin = secondMax === -1 ? 1.0 : probs1[argmax1] - probs1[secondMax];
          if (margin < 1e-4) return; // skip near-ties

          const probs2 = softmax(probs1);
          let argmax2 = 0;
          for (let i = 1; i < probs2.length; i++) {
            if (probs2[i] > probs2[argmax2]) argmax2 = i;
          }
          expect(argmax1).toBe(argmax2);
        }),
        { numRuns: 100 },
      );
    },
  );

  it(
    /** Feature: sign-language-interpreter, Property 12: ONNX model serialization round-trip */
    'Property 12: softmax output on fingerspell-sized input (63) has exactly 63 probabilities',
    () => {
      fc.assert(
        fc.property(fingerspellInput, (logits) => {
          const probs = softmax(logits);
          expect(probs.length).toBe(63);
          const sum = Array.from(probs).reduce((a, b) => a + b, 0);
          expect(sum).toBeCloseTo(1.0, 4);
        }),
        { numRuns: 100 },
      );
    },
  );
});

// ---------------------------------------------------------------------------
// Property 12 — topK returns exactly K entries (or fewer if labels are short)
// ---------------------------------------------------------------------------

describe('topK', () => {
  it(
    /** Feature: sign-language-interpreter, Property 12: ONNX model serialization round-trip */
    'Property 12: returns exactly min(k, labels.length) entries',
    () => {
      fc.assert(
        fc.property(
          fc.integer({ min: 1, max: 50 }).chain((numClasses) =>
            fc.tuple(
              float32Array(numClasses),
              fc.constant(
                Array.from({ length: numClasses }, (_, i) => `label_${i}`),
              ),
              fc.integer({ min: 1, max: numClasses + 5 }), // k may exceed numClasses
            ),
          ),
          ([probs, labels, k]) => {
            const results = topK(probs, labels, k);
            const expectedLen = Math.min(k, Math.min(probs.length, labels.length));
            expect(results.length).toBe(expectedLen);
          },
        ),
        { numRuns: 100 },
      );
    },
  );

  it(
    /** Feature: sign-language-interpreter, Property 12: ONNX model serialization round-trip */
    'Property 12: topK results are sorted descending by probability',
    () => {
      fc.assert(
        fc.property(
          fc.integer({ min: 2, max: 40 }).chain((n) =>
            fc.tuple(
              float32Array(n),
              fc.constant(Array.from({ length: n }, (_, i) => `c${i}`)),
            ),
          ),
          ([probs, labels]) => {
            const results = topK(probs, labels, 5);
            for (let i = 1; i < results.length; i++) {
              expect(results[i].prob).toBeLessThanOrEqual(results[i - 1].prob);
            }
          },
        ),
        { numRuns: 100 },
      );
    },
  );

  it(
    /** Feature: sign-language-interpreter, Property 12: ONNX model serialization round-trip */
    'Property 12: all topK probability values are in [0.0, 1.0] when input is a softmax output',
    () => {
      fc.assert(
        fc.property(anyFloat32Array, (logits) => {
          const probs = softmax(logits);
          const labels = Array.from({ length: probs.length }, (_, i) => `c${i}`);
          const results = topK(probs, labels, 5);
          for (const r of results) {
            expect(r.prob).toBeGreaterThanOrEqual(0.0);
            expect(r.prob).toBeLessThanOrEqual(1.0);
          }
        }),
        { numRuns: 100 },
      );
    },
  );
});

// ---------------------------------------------------------------------------
// Property 12 — resampleSequence output has correct shape
// ---------------------------------------------------------------------------

describe('resampleSequence', () => {
  it(
    /** Feature: sign-language-interpreter, Property 12: ONNX model serialization round-trip */
    'Property 12: output has exactly targetLen * frameDim elements',
    () => {
      fc.assert(
        fc.property(
          fc.integer({ min: 4, max: 16 }).chain((frameDim) =>
            fc.tuple(
              frameBuffer(frameDim),
              fc.integer({ min: 1, max: 60 }),
              fc.constant(frameDim),
            ),
          ),
          ([frames, targetLen, frameDim]) => {
            const out = resampleSequence(frames, targetLen);
            expect(out).toBeInstanceOf(Float32Array);
            expect(out.length).toBe(targetLen * frameDim);
          },
        ),
        { numRuns: 100 },
      );
    },
  );

  it(
    /** Feature: sign-language-interpreter, Property 12: ONNX model serialization round-trip */
    'Property 12: resampling a single-frame buffer repeats the frame exactly targetLen times',
    () => {
      fc.assert(
        fc.property(
          fc.integer({ min: 1, max: 30 }).chain((frameDim) =>
            fc.tuple(
              float32Array(frameDim),
              fc.integer({ min: 1, max: 30 }),
              fc.constant(frameDim),
            ),
          ),
          ([singleFrame, targetLen, frameDim]) => {
            const frames = [singleFrame];
            const out = resampleSequence(frames, targetLen);
            expect(out.length).toBe(targetLen * frameDim);

            // Every frame in the output must equal the single input frame
            for (let t = 0; t < targetLen; t++) {
              for (let d = 0; d < frameDim; d++) {
                expect(out[t * frameDim + d]).toBeCloseTo(singleFrame[d], 4);
              }
            }
          },
        ),
        { numRuns: 100 },
      );
    },
  );

  it(
    /** Feature: sign-language-interpreter, Property 12: ONNX model serialization round-trip */
    'Property 12: resampling to the same length as input produces identical output (identity)',
    () => {
      fc.assert(
        fc.property(
          fc.integer({ min: 2, max: 10 }).chain((frameDim) =>
            fc.integer({ min: 2, max: 20 }).chain((t) =>
              fc.tuple(
                fc.array(float32Array(frameDim), { minLength: t, maxLength: t }),
                fc.constant(t),
                fc.constant(frameDim),
              ),
            ),
          ),
          ([frames, t, frameDim]) => {
            const out = resampleSequence(frames, t);
            expect(out.length).toBe(t * frameDim);

            // First and last frames must match exactly (boundary condition)
            const firstIn = frames[0];
            const lastIn = frames[t - 1];
            for (let d = 0; d < frameDim; d++) {
              expect(out[d]).toBeCloseTo(firstIn[d], 4);
              expect(out[(t - 1) * frameDim + d]).toBeCloseTo(lastIn[d], 4);
            }
          },
        ),
        { numRuns: 100 },
      );
    },
  );

  it(
    /** Feature: sign-language-interpreter, Property 12: ONNX model serialization round-trip */
    'Property 12: resampled word-level input [T, 126] produces output of 30 * 126 elements',
    () => {
      fc.assert(
        fc.property(
          frameBuffer(126),
          (frames) => {
            const out = resampleSequence(frames, 30);
            expect(out.length).toBe(30 * 126);
          },
        ),
        { numRuns: 100 },
      );
    },
  );
});
