/**
 * Property-based tests for browser PredictionGate.
 *
 * Feature: sign-language-interpreter, Property 3: PredictionGate commits exactly once per stable run
 * Validates: Requirements 2.4, 9.5, 9.6
 *
 * Property 3 — PredictionGate commits exactly once per stable run:
 *   For any sequence of per-frame predictions fed into PredictionGate, if a
 *   single label occupies the majority of the vote window AND appears in every
 *   slot of the hold queue AND exceeds the confidence threshold AND the cooldown
 *   has expired, then the gate SHALL fire exactly once — emitting the label —
 *   and immediately enter cooldown, preventing a second emission from the same run.
 */

import * as fc from 'fast-check';
import { describe, it, expect } from 'vitest';
import { PredictionGate } from '../../browser/gate';

// ---------------------------------------------------------------------------
// Generators
// ---------------------------------------------------------------------------

const MAX_WINDOW = 10;
const MAX_HOLD = 10;

/** Label alphabet: uppercase ASCII letters A–Z (1–3 chars as per task spec). */
const labelArb = fc.stringOf(
  fc.mapToConstant(
    { num: 26, build: (i) => String.fromCharCode(65 + i) }, // A–Z
  ),
  { minLength: 1, maxLength: 3 },
);

/**
 * Gate parameters where cooldown >= voteWindow + holdFrames.
 * This guarantees a stable run of (voteWindow + holdFrames) frames can produce
 * at most one commit before cooldown blocks any second commit.
 */
const safeGateParamsArb = fc
  .tuple(
    fc.integer({ min: 1, max: MAX_WINDOW }),   // voteWindow
    fc.integer({ min: 1, max: MAX_HOLD }),     // holdFrames
    fc.float({ min: Math.fround(0.0), max: Math.fround(0.95), noNaN: true }),  // confidenceThreshold
  )
  .chain(([voteWindow, holdFrames, confidenceThreshold]) => {
    const stableRunLength = voteWindow + holdFrames;
    return fc
      .integer({ min: stableRunLength, max: stableRunLength + 20 })
      .map((cooldownFrames) => ({
        voteWindow,
        holdFrames,
        confidenceThreshold,
        cooldownFrames,
      }));
  });

/** Any valid gate parameters (unconstrained cooldown). */
const anyGateParamsArb = fc.record({
  voteWindow: fc.integer({ min: 1, max: MAX_WINDOW }),
  holdFrames: fc.integer({ min: 1, max: MAX_HOLD }),
  confidenceThreshold: fc.float({ min: Math.fround(0.0), max: Math.fround(0.95), noNaN: true }),
  cooldownFrames: fc.integer({ min: 0, max: 30 }),
});

// ---------------------------------------------------------------------------
// Helper: confidence just above the threshold (capped at 1.0)
// ---------------------------------------------------------------------------
function confAbove(threshold: number): number {
  return Math.min(threshold + 0.05, 1.0);
}

// ---------------------------------------------------------------------------
// Property 3 — commits exactly once per stable run
// ---------------------------------------------------------------------------

describe('PredictionGate', () => {
  it(
    /** Feature: sign-language-interpreter, Property 3: PredictionGate commits exactly once per stable run */
    'Property 3: commits exactly once given a stable run of voteWindow + holdFrames frames',
    () => {
      fc.assert(
        fc.property(safeGateParamsArb, labelArb, (params, label) => {
          const gate = new PredictionGate(
            params.voteWindow,
            params.holdFrames,
            params.cooldownFrames,
            params.confidenceThreshold,
          );
          const confidence = confAbove(params.confidenceThreshold);
          const stableRunLength = params.voteWindow + params.holdFrames;

          const accepted: string[] = [];
          for (let i = 0; i < stableRunLength; i++) {
            const result = gate.update(label, confidence);
            if (result !== null) {
              accepted.push(result);
            }
          }

          expect(accepted.length).toBe(1);
          expect(accepted[0]).toBe(label);
        }),
        { numRuns: 200 },
      );
    },
  );

  it(
    /** Feature: sign-language-interpreter, Property 3: PredictionGate commits exactly once per stable run */
    'Property 3: does not re-commit during cooldown after a stable run',
    () => {
      fc.assert(
        fc.property(safeGateParamsArb, labelArb, (params, label) => {
          const gate = new PredictionGate(
            params.voteWindow,
            params.holdFrames,
            params.cooldownFrames,
            params.confidenceThreshold,
          );
          const confidence = confAbove(params.confidenceThreshold);
          const stableRunLength = params.voteWindow + params.holdFrames;

          // Drive gate to first commit
          let commitFrameIdx: number | null = null;
          for (let i = 0; i < stableRunLength; i++) {
            const result = gate.update(label, confidence);
            if (result !== null && commitFrameIdx === null) {
              commitFrameIdx = i;
            }
          }

          if (commitFrameIdx === null) {
            // Gate never fired — skip (safe params guarantee this won't happen)
            return;
          }

          // Frames processed after commit within the stable-run loop already
          // decremented the cooldown by (stableRunLength - 1 - commitFrameIdx)
          const framesAfterCommit = stableRunLength - 1 - commitFrameIdx;
          const remainingCooldown = params.cooldownFrames - framesAfterCommit;

          if (remainingCooldown <= 0) {
            return; // All cooldown already drained
          }

          // Feed exactly remainingCooldown more same-label frames — none should commit
          const duringCooldown: string[] = [];
          for (let i = 0; i < remainingCooldown; i++) {
            const result = gate.update(label, confidence);
            if (result !== null) {
              duringCooldown.push(result);
            }
          }

          expect(duringCooldown.length).toBe(0);
        }),
        { numRuns: 200 },
      );
    },
  );

  it(
    /** Feature: sign-language-interpreter, Property 3: PredictionGate commits exactly once per stable run */
    'Property 3: every committed label equals the dominant input label',
    () => {
      fc.assert(
        fc.property(anyGateParamsArb, labelArb, (params, label) => {
          const gate = new PredictionGate(
            params.voteWindow,
            params.holdFrames,
            params.cooldownFrames,
            params.confidenceThreshold,
          );
          const confidence = confAbove(params.confidenceThreshold);

          // Run for enough frames to trigger multiple potential commits
          const totalFrames =
            (params.voteWindow + params.holdFrames + params.cooldownFrames + 1) * 3;

          for (let i = 0; i < totalFrames; i++) {
            const result = gate.update(label, confidence);
            if (result !== null) {
              expect(result).toBe(label);
            }
          }
        }),
        { numRuns: 200 },
      );
    },
  );

  // -------------------------------------------------------------------------
  // Scenario: mixed labels → gate never fires
  // -------------------------------------------------------------------------

  it(
    /** Feature: sign-language-interpreter, Property 3: PredictionGate commits exactly once per stable run */
    'Property 3: mixed labels with no majority never trigger a commit',
    () => {
      fc.assert(
        fc.property(
          // Use even voteWindow >= 2 so the window can be filled with an exact
          // 50/50 split of two labels — neither can form a strict >50% majority.
          // Require holdFrames >= 2 and cooldownFrames >= holdFrames to prevent
          // early partial-window commits from interfering.
          fc.record({
            voteWindow: fc.integer({ min: 2, max: MAX_WINDOW }).map((v) => v + (v % 2)), // always even
            holdFrames: fc.integer({ min: 2, max: MAX_HOLD }),
            confidenceThreshold: fc.float({ min: Math.fround(0.0), max: Math.fround(0.95), noNaN: true }),
            cooldownFrames: fc.integer({ min: MAX_HOLD, max: 30 }),
          }),
          // Two distinct labels to interleave
          fc.tuple(labelArb, labelArb).filter(([a, b]) => a !== b),
          (params, [labelA, labelB]) => {
            const gate = new PredictionGate(
              params.voteWindow,
              params.holdFrames,
              params.cooldownFrames,
              params.confidenceThreshold,
            );
            const confidence = confAbove(params.confidenceThreshold);

            // Pre-fill the window with equal counts of A and B (no net majority).
            // We feed exactly voteWindow frames with strict alternation.
            for (let i = 0; i < params.voteWindow; i++) {
              gate.update(i % 2 === 0 ? labelA : labelB, confidence);
            }

            // Now feed more alternating frames; the window always stays 50/50 → no majority.
            const extraFrames = params.holdFrames * 2;
            const accepted: string[] = [];
            for (let i = 0; i < extraFrames; i++) {
              const lbl = i % 2 === 0 ? labelA : labelB;
              const result = gate.update(lbl, confidence);
              if (result !== null) {
                accepted.push(result);
              }
            }

            expect(accepted.length).toBe(0);
          },
        ),
        { numRuns: 200 },
      );
    },
  );

  // -------------------------------------------------------------------------
  // Scenario: low-confidence frames → treated as blank votes, no commit
  // -------------------------------------------------------------------------

  it(
    /** Feature: sign-language-interpreter, Property 3: PredictionGate commits exactly once per stable run */
    'Property 3: low-confidence frames (below threshold) are blank votes and do not trigger commits',
    () => {
      fc.assert(
        fc.property(
          anyGateParamsArb.filter((p) => p.confidenceThreshold > 0.01),
          labelArb,
          (params, label) => {
            const gate = new PredictionGate(
              params.voteWindow,
              params.holdFrames,
              params.cooldownFrames,
              params.confidenceThreshold,
            );
            // Confidence strictly below threshold — all votes are blank
            const lowConf = Math.max(0, params.confidenceThreshold - 0.01);

            const runLength = (params.voteWindow + params.holdFrames) * 2;
            const accepted: string[] = [];
            for (let i = 0; i < runLength; i++) {
              const result = gate.update(label, lowConf);
              if (result !== null) {
                accepted.push(result);
              }
            }

            expect(accepted.length).toBe(0);
          },
        ),
        { numRuns: 200 },
      );
    },
  );

  // -------------------------------------------------------------------------
  // Scenario: reset() clears state so gate can fire again immediately
  // -------------------------------------------------------------------------

  it(
    /** Feature: sign-language-interpreter, Property 3: PredictionGate commits exactly once per stable run */
    'Property 3: reset() clears state — gate can fire again immediately after reset',
    () => {
      fc.assert(
        fc.property(safeGateParamsArb, labelArb, (params, label) => {
          const gate = new PredictionGate(
            params.voteWindow,
            params.holdFrames,
            params.cooldownFrames,
            params.confidenceThreshold,
          );
          const confidence = confAbove(params.confidenceThreshold);
          const stableRunLength = params.voteWindow + params.holdFrames;

          // First stable run — should fire once
          const firstRun: string[] = [];
          for (let i = 0; i < stableRunLength; i++) {
            const result = gate.update(label, confidence);
            if (result !== null) firstRun.push(result);
          }
          expect(firstRun.length).toBe(1);

          // Reset clears cooldown and all internal state
          gate.reset();

          // Second stable run — must also fire exactly once
          const secondRun: string[] = [];
          for (let i = 0; i < stableRunLength; i++) {
            const result = gate.update(label, confidence);
            if (result !== null) secondRun.push(result);
          }
          expect(secondRun.length).toBe(1);
          expect(secondRun[0]).toBe(label);
        }),
        { numRuns: 200 },
      );
    },
  );
});
