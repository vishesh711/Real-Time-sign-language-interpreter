/**
 * InputQualityMonitor — samples webcam input quality every 10 frames and
 * exposes a QualityStatus to the UI.
 *
 * Metrics collected (per Requirements 9.1–9.4):
 *   - detectionConf: MediaPipe per-hand detection confidence [0.0, 1.0]
 *   - luminance:     estimated lux-equivalent of the hand bounding-box region
 *   - fps:           delivered frame rate in frames per second
 *   - handsVisible:  number of hands currently detected (0, 1, or 2)
 *
 * Status thresholds:
 *   'good' : conf >= 0.8 AND luminance >= 80 AND fps >= 25 AND handsVisible > 0
 *   'poor' : conf < 0.5 OR luminance < 40 OR fps < 15
 *   'warn' : any single metric below its warn threshold (everything else)
 *
 * Requirements: 9.1, 9.2, 9.3, 9.4
 */

import type { QualityStatus } from './types';

// ---------------------------------------------------------------------------
// Threshold constants
// ---------------------------------------------------------------------------

/** Minimum "good" detection confidence (Req 9.1). */
const CONF_GOOD = 0.8;
/** Detection confidence below which quality is 'poor'. */
const CONF_POOR = 0.5;

/** Luminance warn threshold — lux-equivalent (Req 9.2). */
const LUM_WARN = 80;
/** Luminance below which quality is 'poor'. */
const LUM_POOR = 40;

/** FPS warn threshold (Req 9.3). */
const FPS_WARN = 25;
/** FPS below which quality is 'poor'. */
const FPS_POOR = 15;

/** How long (ms) FPS must be below FPS_WARN before reporting warn/poor (Req 9.3). */
const FPS_WARN_DEBOUNCE_MS = 500;

/** How many frames to skip between full quality samples. */
const SAMPLE_INTERVAL = 10;

// ---------------------------------------------------------------------------
// Luminance helpers
// ---------------------------------------------------------------------------

/**
 * Bounding box for a hand expressed in normalised [0, 1] image coordinates.
 * If null, luminance sampling is skipped for that frame.
 */
export interface HandBoundingBox {
  xMin: number;
  yMin: number;
  xMax: number;
  yMax: number;
}

/**
 * Estimate the average luminance of a hand region from RGBA pixel data.
 *
 * Uses the standard ITU-R BT.709 luminance formula:
 *   L = 0.2126 * R + 0.7152 * G + 0.0722 * B
 * where R, G, B are in [0, 255]. The result is in the same [0, 255] range
 * (lux-equivalent per the spec, with 80 as the warn threshold).
 *
 * @param imageData  Raw RGBA pixel data from CanvasRenderingContext2D.getImageData.
 * @param bbox       Hand bounding box in normalised [0, 1] coordinates.
 * @param canvasW    Canvas width in pixels.
 * @param canvasH    Canvas height in pixels.
 * @returns Average luminance in [0, 255], or -1 if the region has no pixels.
 */
export function estimateLuminance(
  imageData: ImageData,
  bbox: HandBoundingBox,
  canvasW: number,
  canvasH: number,
): number {
  // Convert normalised coords to integer pixel bounds (clamped to canvas).
  const x0 = Math.max(0, Math.floor(bbox.xMin * canvasW));
  const y0 = Math.max(0, Math.floor(bbox.yMin * canvasH));
  const x1 = Math.min(canvasW - 1, Math.ceil(bbox.xMax * canvasW));
  const y1 = Math.min(canvasH - 1, Math.ceil(bbox.yMax * canvasH));

  const { data, width } = imageData;
  let totalLum = 0;
  let count = 0;

  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const idx = (y * width + x) * 4;
      const r = data[idx];
      const g = data[idx + 1];
      const b = data[idx + 2];
      totalLum += 0.2126 * r + 0.7152 * g + 0.0722 * b;
      count++;
    }
  }

  return count > 0 ? totalLum / count : -1;
}

/**
 * Derive a bounding box from an array of normalised MediaPipe hand landmarks.
 *
 * @param landmarks Array of [x, y, z] triplets (21 points for one hand).
 * @param padding   Fraction of bounding-box size to add on each side (default 0.1).
 * @returns         Clamped bounding box in [0, 1] space.
 */
export function landmarksToBoundingBox(
  landmarks: number[][],
  padding = 0.1,
): HandBoundingBox {
  let xMin = Infinity;
  let yMin = Infinity;
  let xMax = -Infinity;
  let yMax = -Infinity;

  for (const [x, y] of landmarks) {
    if (x < xMin) xMin = x;
    if (x > xMax) xMax = x;
    if (y < yMin) yMin = y;
    if (y > yMax) yMax = y;
  }

  const pw = (xMax - xMin) * padding;
  const ph = (yMax - yMin) * padding;

  return {
    xMin: Math.max(0, xMin - pw),
    yMin: Math.max(0, yMin - ph),
    xMax: Math.min(1, xMax + pw),
    yMax: Math.min(1, yMax + ph),
  };
}

// ---------------------------------------------------------------------------
// Status computation
// ---------------------------------------------------------------------------

/**
 * Determine the aggregated quality status from the four individual metrics.
 *
 * Rules (checked in order):
 *   1. 'poor' if conf < 0.5 OR luminance < 40 OR fps < 15
 *   2. 'good' if conf >= 0.8 AND luminance >= 80 AND fps >= 25 AND handsVisible > 0
 *   3. 'warn' for everything else
 */
export function computeStatus(
  conf: number,
  luminance: number,
  fps: number,
  handsVisible: number,
): 'good' | 'warn' | 'poor' {
  if (conf < CONF_POOR || luminance < LUM_POOR || fps < FPS_POOR) {
    return 'poor';
  }
  if (conf >= CONF_GOOD && luminance >= LUM_WARN && fps >= FPS_WARN && handsVisible > 0) {
    return 'good';
  }
  return 'warn';
}

// ---------------------------------------------------------------------------
// Frame input type
// ---------------------------------------------------------------------------

/**
 * Everything the monitor needs per-frame from the MediaPipe result and canvas.
 * Callers should pass this on every animation-loop tick.
 */
export interface FrameInput {
  /**
   * Timestamp of the frame in milliseconds (e.g. from requestAnimationFrame or
   * performance.now()).
   */
  timestampMs: number;

  /**
   * MediaPipe per-hand detection confidence for this frame.
   * Pass 0 when no hand is detected.
   */
  detectionConf: number;

  /**
   * Raw MediaPipe hand landmarks for luminance sampling.
   * Each inner array is one hand's 21 [x, y, z] normalised landmarks.
   * Pass an empty array when no hands are detected.
   */
  handLandmarks: number[][][];

  /**
   * Canvas and its 2D rendering context, used to sample pixel luminance.
   * May be null if canvas access is unavailable (luminance will be -1).
   */
  canvas: HTMLCanvasElement | null;
  ctx: CanvasRenderingContext2D | null;
}

// ---------------------------------------------------------------------------
// InputQualityMonitor
// ---------------------------------------------------------------------------

/**
 * InputQualityMonitor samples every 10 frames and maintains a current
 * QualityStatus that the UI can read at any time via `.status`.
 *
 * Usage:
 *   const monitor = new InputQualityMonitor({ onStatus: s => updateUI(s) });
 *   // inside requestAnimationFrame loop:
 *   monitor.update({ timestampMs, detectionConf, handLandmarks, canvas, ctx });
 */
export class InputQualityMonitor {
  // -- configuration ---------------------------------------------------------
  /** Number of frames between full quality samples (default 10). */
  readonly sampleInterval: number;

  // -- state -----------------------------------------------------------------
  private _frameCount = 0;
  private _lastStatus: QualityStatus = {
    detectionConf: 0,
    luminance: 0,
    fps: 0,
    handsVisible: 0,
    status: 'warn',
  };

  // FPS tracking
  private _lastFrameTimestampMs: number | null = null;
  private _recentFpsSamples: number[] = [];   // rolling buffer of recent FPS measurements
  private _smoothedFps = 0;

  // Low-FPS debounce (Req 9.3: warn only after 500 ms below threshold)
  private _lowFpsSinceMs: number | null = null;
  private _fpsWarningActive = false;

  // Callback invoked with the new status after each sample
  private _onStatus?: (status: QualityStatus) => void;

  constructor(options?: {
    sampleInterval?: number;
    onStatus?: (status: QualityStatus) => void;
  }) {
    this.sampleInterval = options?.sampleInterval ?? SAMPLE_INTERVAL;
    this._onStatus = options?.onStatus;
  }

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  /**
   * Current quality status. Reflects the last sampled state; call `update()`
   * every animation frame to keep it current.
   */
  get status(): QualityStatus {
    return this._lastStatus;
  }

  /**
   * Process one animation frame.
   *
   * - Always tracks FPS from consecutive frame timestamps.
   * - Performs a full quality sample every `sampleInterval` frames.
   *
   * @param frame Per-frame data from MediaPipe and the canvas.
   */
  update(frame: FrameInput): void {
    this._frameCount++;

    // --- FPS tracking (every frame for accuracy) ---
    const instantFps = this._trackFps(frame.timestampMs);

    // --- Decide whether to emit a full quality sample ---
    if (this._frameCount % this.sampleInterval !== 0) {
      return;
    }

    // --- Count visible hands (Req 9.4) ---
    const handsVisible = Math.min(
      frame.handLandmarks.length,
      2,
    ) as 0 | 1 | 2;

    // --- Luminance (Req 9.2) ---
    let luminance = 0;
    if (
      frame.canvas !== null &&
      frame.ctx !== null &&
      frame.handLandmarks.length > 0
    ) {
      luminance = this._sampleLuminance(
        frame.ctx,
        frame.canvas,
        frame.handLandmarks,
      );
    }

    // --- FPS with debounce (Req 9.3) ---
    const reportedFps = this._debouncedFps(instantFps, frame.timestampMs);

    // --- Build status ---
    const newStatus: QualityStatus = {
      detectionConf: frame.detectionConf,
      luminance,
      fps: reportedFps,
      handsVisible,
      status: computeStatus(
        frame.detectionConf,
        luminance,
        reportedFps,
        handsVisible,
      ),
    };

    this._lastStatus = newStatus;
    this._onStatus?.(newStatus);
  }

  /**
   * Reset all internal state. Call when the stream is restarted.
   */
  reset(): void {
    this._frameCount = 0;
    this._lastFrameTimestampMs = null;
    this._recentFpsSamples = [];
    this._smoothedFps = 0;
    this._lowFpsSinceMs = null;
    this._fpsWarningActive = false;
    this._lastStatus = {
      detectionConf: 0,
      luminance: 0,
      fps: 0,
      handsVisible: 0,
      status: 'warn',
    };
  }

  // ---------------------------------------------------------------------------
  // Internal helpers
  // ---------------------------------------------------------------------------

  /**
   * Measure instantaneous FPS from frame timestamps and maintain a
   * small smoothing buffer (last 10 samples).
   *
   * @returns Smoothed FPS value.
   */
  private _trackFps(timestampMs: number): number {
    if (this._lastFrameTimestampMs !== null) {
      const dtMs = timestampMs - this._lastFrameTimestampMs;
      if (dtMs > 0) {
        const instantFps = 1000 / dtMs;
        this._recentFpsSamples.push(instantFps);
        if (this._recentFpsSamples.length > 10) {
          this._recentFpsSamples.shift();
        }
        const sum = this._recentFpsSamples.reduce((a, b) => a + b, 0);
        this._smoothedFps = sum / this._recentFpsSamples.length;
      }
    }
    this._lastFrameTimestampMs = timestampMs;
    return this._smoothedFps;
  }

  /**
   * Apply the 500 ms debounce rule for low-FPS warnings (Req 9.3).
   *
   * - If FPS >= FPS_WARN: reset the low-FPS timer and return true FPS.
   * - If FPS < FPS_WARN but debounce not yet elapsed: return FPS_WARN (no warn yet).
   * - If FPS < FPS_WARN and debounce elapsed: return actual FPS (triggers warn/poor).
   */
  private _debouncedFps(fps: number, nowMs: number): number {
    if (fps >= FPS_WARN) {
      this._lowFpsSinceMs = null;
      this._fpsWarningActive = false;
      return fps;
    }

    // FPS is below threshold
    if (this._lowFpsSinceMs === null) {
      this._lowFpsSinceMs = nowMs;
    }

    const elapsedMs = nowMs - this._lowFpsSinceMs;
    if (elapsedMs >= FPS_WARN_DEBOUNCE_MS) {
      this._fpsWarningActive = true;
    }

    // Return actual FPS only when the warning period has elapsed;
    // otherwise clamp to FPS_WARN so `computeStatus` still sees 'good'.
    return this._fpsWarningActive ? fps : FPS_WARN;
  }

  /**
   * Sample the average luminance of all visible hand regions on the canvas.
   *
   * @param ctx     2D rendering context for pixel reads.
   * @param canvas  Canvas element (needed for width/height).
   * @param handLandmarks Raw normalised hand landmarks from MediaPipe.
   * @returns Average luminance across all detected hand regions, or 0 on error.
   */
  private _sampleLuminance(
    ctx: CanvasRenderingContext2D,
    canvas: HTMLCanvasElement,
    handLandmarks: number[][][],
  ): number {
    const w = canvas.width;
    const h = canvas.height;
    if (w === 0 || h === 0) return 0;

    let imageData: ImageData;
    try {
      imageData = ctx.getImageData(0, 0, w, h);
    } catch {
      // Canvas may be tainted (cross-origin) — skip luminance sampling.
      return 0;
    }

    let totalLum = 0;
    let sampledHands = 0;

    for (const lm of handLandmarks) {
      if (!lm || lm.length < 2) continue;
      const bbox = landmarksToBoundingBox(lm);
      const lum = estimateLuminance(imageData, bbox, w, h);
      if (lum >= 0) {
        totalLum += lum;
        sampledHands++;
      }
    }

    return sampledHands > 0 ? totalLum / sampledHands : 0;
  }
}
