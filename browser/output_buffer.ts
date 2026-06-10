/**
 * OutputBuffer — manages committed text, the in-progress candidate, and the
 * raw gloss accumulation stream used for gloss-to-English post-processing.
 *
 * Requirements: 5.3, 5.4, 5.10
 */

import type { OutputBuffer as IOutputBuffer } from './types';

/**
 * The snapshot returned by `render()`.
 */
export interface RenderResult {
  /** Sequence of accepted words/letters committed to the transcript. */
  committed: string[];
  /** In-progress prediction in the hold window, not yet committed. */
  candidate: string | null;
}

/**
 * Manages the text output state for the Sign Language Interpreter.
 *
 * - `accept(word)`:       Append a word to `committed` and `glossBuffer`,
 *                         and clear the current candidate.
 * - `backspace()`:        Remove the last committed word (no-op on empty buffer).
 * - `clear()`:            Reset `committed`, `candidate`, and `glossBuffer`.
 * - `setCandidate(word)`: Set the in-progress prediction shown as a preview.
 * - `render()`:           Return a snapshot `{ committed, candidate }`.
 *
 * The `glossBuffer` mirrors every accepted word so that downstream
 * `GlossPostProcessor` can perform gloss-to-English translation over the full
 * session without re-scanning `committed`.
 */
export class OutputBuffer implements IOutputBuffer {
  /** Accepted words/letters committed to the session transcript. */
  committed: string[];

  /** In-progress prediction in the hold window, not yet committed. */
  candidate: string | null;

  /**
   * Raw gloss stream for translation — every accepted token is appended here.
   * Unlike `committed`, entries are never removed (even after `backspace`),
   * so the translator always has the full accumulated context.
   * `clear()` resets this along with the rest of the buffer (Req 5.4).
   */
  glossBuffer: string[];

  constructor() {
    this.committed = [];
    this.candidate = null;
    this.glossBuffer = [];
  }

  // ---------------------------------------------------------------------------
  // Mutation methods
  // ---------------------------------------------------------------------------

  /**
   * Commit a word to the output buffer.
   *
   * Appends `word` to both `committed` and `glossBuffer`, and clears the
   * current candidate so the UI transitions from "preview" to "committed".
   *
   * Requirements: 5.3
   *
   * @param word The word or letter to commit.
   */
  accept(word: string): void {
    this.committed.push(word);
    this.glossBuffer.push(word);
    this.candidate = null;
  }

  /**
   * Delete the last committed word from the output buffer.
   *
   * No-op when the buffer is already empty.
   *
   * Requirements: 5.10
   */
  backspace(): void {
    if (this.committed.length > 0) {
      this.committed.pop();
    }
  }

  /**
   * Reset the display and all internal accumulation state simultaneously.
   *
   * Clears `committed`, `candidate`, and `glossBuffer` so the output panel
   * and the gloss translator both start fresh.
   *
   * Requirements: 5.4
   */
  clear(): void {
    this.committed = [];
    this.candidate = null;
    this.glossBuffer = [];
  }

  /**
   * Update the in-progress candidate preview.
   *
   * Pass `null` to remove the candidate (e.g., after a commit or on hand loss).
   *
   * Requirements: 5.6 (candidate is displayed distinctly from committed text)
   *
   * @param word The candidate label, or null to clear.
   */
  setCandidate(word: string | null): void {
    this.candidate = word;
  }

  // ---------------------------------------------------------------------------
  // Read methods
  // ---------------------------------------------------------------------------

  /**
   * Return a snapshot of the current render state.
   *
   * The returned arrays are copies so callers cannot mutate internal state
   * through the snapshot.
   *
   * Requirements: 5.1, 5.3
   *
   * @returns `{ committed: string[], candidate: string | null }`
   */
  render(): RenderResult {
    return {
      committed: [...this.committed],
      candidate: this.candidate,
    };
  }
}
