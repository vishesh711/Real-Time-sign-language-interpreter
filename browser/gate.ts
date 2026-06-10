/**
 * PredictionGate — three-layer temporal filter applied after every inference call.
 *
 * Layer 1: Majority vote over a sliding window of the last N frames.
 * Layer 2: Hold queue — the voted label must be stable for M consecutive frames.
 * Layer 3: Cooldown — a minimum gap of K frames between two accepted predictions.
 *
 * This is a direct TypeScript port of the canonical Python implementation in
 * ``utils/gate.py`` — the logic is identical.
 *
 * Requirements: 2.4, 9.5, 9.6
 */

/**
 * Three-layer temporal filter for stable sign commitment.
 *
 * @param voteWindow      Number of recent frames used for majority voting (default 7).
 * @param holdFrames      Number of consecutive frames the voted label must be
 *                        stable before it is accepted (default 12).
 * @param cooldownFrames  Number of frames to suppress new predictions after an
 *                        acceptance event (default 20).
 * @param confidenceThreshold  Minimum per-frame confidence required for a frame
 *                        to contribute to the vote window (default 0.8).
 */
export class PredictionGate {
  readonly voteWindow: number;
  readonly holdFrames: number;
  readonly cooldownFrames: number;
  readonly confidenceThreshold: number;

  // Internal state — mirroring Python's deque / instance variables
  private _window: string[];          // circular-ish buffer capped at voteWindow
  private _holdLabel: string | null;
  private _holdCount: number;
  private _cooldownRemaining: number;

  constructor(
    voteWindow = 7,
    holdFrames = 12,
    cooldownFrames = 20,
    confidenceThreshold = 0.8,
  ) {
    if (voteWindow < 1) {
      throw new RangeError("voteWindow must be >= 1");
    }
    if (holdFrames < 1) {
      throw new RangeError("holdFrames must be >= 1");
    }
    if (confidenceThreshold < 0.0 || confidenceThreshold > 1.0) {
      throw new RangeError("confidenceThreshold must be in [0, 1]");
    }
    if (cooldownFrames < 0) {
      throw new RangeError("cooldownFrames must be >= 0");
    }

    this.voteWindow = voteWindow;
    this.holdFrames = holdFrames;
    this.cooldownFrames = cooldownFrames;
    this.confidenceThreshold = confidenceThreshold;

    this._window = [];
    this._holdLabel = null;
    this._holdCount = 0;
    this._cooldownRemaining = 0;
  }

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  /**
   * Feed one frame's prediction into the gate.
   *
   * @param label       Predicted class label for this frame.
   * @param confidence  Softmax probability in [0.0, 1.0] for the predicted label.
   * @returns The accepted label string if the gate fires, or null if no
   *          commitment is made this frame.
   */
  update(label: string, confidence: number): string | null {
    // Decrement cooldown counter
    if (this._cooldownRemaining > 0) {
      this._cooldownRemaining -= 1;
      // Still suppress: push to window but do not accept
      this._windowAppend(confidence >= this.confidenceThreshold ? label : "");
      // Reset hold state during cooldown
      this._holdLabel = null;
      this._holdCount = 0;
      return null;
    }

    // Layer 1 — majority vote
    this._windowAppend(confidence >= this.confidenceThreshold ? label : "");

    const votedLabel = this._majorityVote();

    if (votedLabel === null) {
      // No majority — reset hold
      this._holdLabel = null;
      this._holdCount = 0;
      return null;
    }

    // Layer 2 — hold queue
    if (votedLabel === this._holdLabel) {
      this._holdCount += 1;
    } else {
      this._holdLabel = votedLabel;
      this._holdCount = 1;
    }

    if (this._holdCount >= this.holdFrames) {
      // Layer 3 — commit and enter cooldown
      const accepted = this._holdLabel as string;
      this._holdLabel = null;
      this._holdCount = 0;
      this._window = [];
      this._cooldownRemaining = this.cooldownFrames;
      return accepted;
    }

    return null;
  }

  /**
   * Clear all internal state (call when the user clears the buffer or
   * the hand is lost).
   */
  reset(): void {
    this._window = [];
    this._holdLabel = null;
    this._holdCount = 0;
    this._cooldownRemaining = 0;
  }

  // ---------------------------------------------------------------------------
  // Internal helpers
  // ---------------------------------------------------------------------------

  /**
   * Append a label to the window, evicting the oldest entry when the window
   * is full — mirroring Python's `deque(maxlen=voteWindow)`.
   */
  private _windowAppend(label: string): void {
    this._window.push(label);
    if (this._window.length > this.voteWindow) {
      this._window.shift();
    }
  }

  /**
   * Return the label with a strict majority (> 50 %) in the window,
   * or null if no such label exists.  Blank votes ("") never win.
   */
  private _majorityVote(): string | null {
    if (this._window.length === 0) {
      return null;
    }

    const counts: Record<string, number> = {};
    for (const lbl of this._window) {
      if (lbl !== "") {   // ignore blank votes
        counts[lbl] = (counts[lbl] ?? 0) + 1;
      }
    }

    const labels = Object.keys(counts);
    if (labels.length === 0) {
      return null;
    }

    let bestLabel = labels[0];
    for (const lbl of labels) {
      if (counts[lbl] > counts[bestLabel]) {
        bestLabel = lbl;
      }
    }

    // Strict majority: count must exceed half the window size
    if (counts[bestLabel] > this._window.length / 2) {
      return bestLabel;
    }
    return null;
  }
}
