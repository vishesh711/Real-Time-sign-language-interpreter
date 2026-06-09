/**
 * Core TypeScript interfaces for the Sign Language Interpreter browser components.
 * Requirements: 1.1, 2.1, 3.1, 4.1, 8.1
 */

/**
 * Result from MediaPipe landmark extraction for a single frame.
 * Requirements: 1.1, 1.3
 */
export interface LandmarkResult {
  /** Hand landmarks: shape [2, 21, 3] — index 0 = right, index 1 = left */
  handLandmarks: number[][][];
  /** Body pose landmarks: shape [33, 3] */
  poseLandmarks: number[][];
  /** Face landmarks: shape [468, 3] */
  faceLandmarks: number[][];
  /** Which hands are detected, e.g. ['Right', 'Left'] */
  handedness: Array<'Left' | 'Right'>;
  /** MediaPipe per-hand detection confidence [0.0, 1.0] */
  detectionConfidence: number;
  /** Number of landmarks with visibility score >= 0.5 */
  visibleLandmarkCount: number;
}

/**
 * Result returned from the classifier (fingerspell or word-level).
 * Requirements: 2.1, 2.2, 3.2, 8.3
 */
export interface PredictionResult {
  /** 'fingerspell' for letter prediction, 'word' for gloss prediction */
  type: 'fingerspell' | 'word';
  /** Top predicted class label (letter A–Z or gloss string) */
  prediction: string;
  /** Confidence of top prediction, in range [0.0, 1.0] */
  confidence: number;
  /** Top-5 predictions with probabilities */
  top5: Array<{ label: string; prob: number }>;
  /** Whether the prediction was accepted (committed) by PredictionGate */
  accepted: boolean;
  /** The label that was accepted, or null if not yet committed */
  acceptedLabel: string | null;
  /** Round-trip latency from frame capture to result, in milliseconds */
  latencyMs: number;
}

/**
 * The text output buffer holding committed and candidate text.
 * Requirements: 5.3, 5.4, 5.10
 */
export interface OutputBuffer {
  /** Sequence of accepted words/letters committed to the transcript */
  committed: string[];
  /** In-progress prediction in the hold window, not yet committed */
  candidate: string | null;
  /** Accumulation of raw glosses for gloss-to-English post-processing */
  glossBuffer: string[];
}

/**
 * Real-time status of webcam input quality.
 * Requirements: 9.1, 9.2, 9.3, 9.4
 */
export interface QualityStatus {
  /** MediaPipe per-hand detection confidence [0.0, 1.0] */
  detectionConf: number;
  /** Estimated luminance of the hand bounding-box region (lux-equivalent) */
  luminance: number;
  /** Delivered frame rate in frames per second */
  fps: number;
  /** Number of hands currently visible in frame (0, 1, or 2) */
  handsVisible: 0 | 1 | 2;
  /** Aggregated quality level for the UI status indicator */
  status: 'good' | 'warn' | 'poor';
}

/**
 * WebSocket payload sent from browser to server for fingerspelling inference.
 * Requirements: 8.1, 8.2, 4.4
 */
export interface FingerspellPayload {
  /** Normalized landmark array: 21 keypoints × 3 coordinates = 63 floats */
  landmarks: number[];
  handedness: 'Left' | 'Right';
}

/**
 * WebSocket payload sent from browser to server for word-level inference.
 * Requirements: 8.1, 8.2, 4.4
 */
export interface WordPayload {
  /** Two-hand landmark vector: 42 keypoints × 3 coordinates = 126 floats */
  landmarks: number[];
  /** Monotonically increasing frame index */
  frameIdx: number;
}
