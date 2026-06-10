/**
 * WebRTCAdapter — intercepts WebRTC media streams and overlays a real-time
 * caption panel on the video call UI.
 *
 * Responsibilities:
 *  - Patch `navigator.mediaDevices.getUserMedia` to capture every new stream
 *    automatically (Req 6.1).
 *  - Draw each video frame to a hidden canvas and run MediaPipe landmark
 *    extraction within 33 ms (Req 6.3).
 *  - Render a translucent caption panel over the video element (Req 6.2).
 *  - Pause inference and show a stream-interrupted indicator when the track
 *    ends; resume automatically when a new track is added (Req 6.4).
 *
 * The MediaPipe inference call inside `_processFrame()` is marked as a
 * documented stub — replace it with the real MediaPipe Web integration
 * (e.g. `@mediapipe/hands` or the Tasks Vision API) when connecting to the
 * full pipeline.
 *
 * Requirements: 6.1, 6.2, 6.3, 6.4
 */

import type { PredictionResult } from './types';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/**
 * Frame processing budget in milliseconds (Req 6.3).
 * If a frame takes longer we skip the next rAF tick to avoid building a
 * backlog and log a warning.
 */
const FRAME_BUDGET_MS = 33;

/** Z-index applied to the caption overlay so it floats above call UI chrome. */
const OVERLAY_Z_INDEX = 2147483647; // max safe CSS z-index

/** Text shown when the stream is interrupted (Req 6.4). */
const INTERRUPTED_TEXT = 'Stream interrupted — waiting to reconnect…';

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

/** Constructor options for WebRTCAdapter. */
export interface WebRTCAdapterOptions {
  /**
   * Called whenever the MediaPipe stub produces a prediction result.
   * Replace the stub body in `_processFrame()` with real inference to
   * populate this callback with meaningful data.
   */
  onPrediction?: (result: PredictionResult) => void;

  /**
   * Called whenever the caption text changes (either a new prediction string
   * or the stream-interrupted message).  Useful for propagating captions to
   * an external transcript panel.
   */
  onCaption?: (text: string) => void;
}

// ---------------------------------------------------------------------------
// WebRTCAdapter
// ---------------------------------------------------------------------------

/**
 * Browser-side WebRTC stream interceptor.
 *
 * Usage:
 *
 * ```typescript
 * const adapter = new WebRTCAdapter({
 *   onPrediction: (r) => console.log(r),
 *   onCaption:    (t) => transcriptPanel.textContent = t,
 * });
 *
 * // Patch getUserMedia before the video call SDK initialises its stream:
 * adapter.install();
 *
 * // OR attach to an existing <video> + stream pair directly:
 * adapter.attach(videoEl, stream);
 * ```
 */
export class WebRTCAdapter {
  // -- configuration ---------------------------------------------------------
  private readonly _onPrediction?: (result: PredictionResult) => void;
  private readonly _onCaption?: (text: string) => void;

  // -- DOM / capture state ---------------------------------------------------

  /** The video element the adapter is currently attached to. */
  private _videoElement: HTMLVideoElement | null = null;

  /** Hidden canvas used to grab individual video frames. */
  private _canvas: HTMLCanvasElement | null = null;

  /** 2D context of the hidden canvas. */
  private _ctx: CanvasRenderingContext2D | null = null;

  /** Caption <div> layered over the video element. */
  private _overlay: HTMLDivElement | null = null;

  // -- rAF loop state --------------------------------------------------------

  /** requestAnimationFrame handle; non-null while the loop is running. */
  private _rafHandle: number | null = null;

  /** Whether the frame-processing loop is intentionally paused. */
  private _paused = false;

  /**
   * Timestamp (ms) of the last call to `_processFrame`.
   * Used to enforce the 33 ms per-frame budget warning.
   */
  private _lastFrameTs = 0;

  // -- getUserMedia patching -------------------------------------------------

  /**
   * Reference to the original `navigator.mediaDevices.getUserMedia` so we
   * can restore it in `uninstall()` and call it from the interceptor.
   */
  private _originalGetUserMedia:
    | typeof navigator.mediaDevices.getUserMedia
    | null = null;

  // -- stream interruption ---------------------------------------------------

  /** Whether the stream is currently interrupted (Req 6.4). */
  private _interrupted = false;

  constructor(options?: WebRTCAdapterOptions) {
    this._onPrediction = options?.onPrediction;
    this._onCaption = options?.onCaption;
  }

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  /**
   * Patch `navigator.mediaDevices.getUserMedia` so every stream obtained
   * by the page is automatically intercepted.
   *
   * The interceptor:
   *  1. Calls the original `getUserMedia` and awaits the real stream.
   *  2. Attaches track-ended listeners to detect stream interruptions.
   *  3. Returns the stream unchanged (transparent pass-through).
   *
   * Requirements: 6.1, 6.4
   */
  install(): void {
    if (!navigator.mediaDevices) {
      console.warn('WebRTCAdapter.install(): navigator.mediaDevices is not available.');
      return;
    }

    if (this._originalGetUserMedia !== null) {
      // Already installed — idempotent.
      return;
    }

    this._originalGetUserMedia = navigator.mediaDevices.getUserMedia.bind(
      navigator.mediaDevices,
    );

    // Arrow function so `this` refers to the WebRTCAdapter instance.
    const self = this;

    navigator.mediaDevices.getUserMedia = async function (
      constraints?: MediaStreamConstraints,
    ): Promise<MediaStream> {
      const stream = await self._originalGetUserMedia!(constraints);
      self._attachStreamListeners(stream);
      return stream;
    };
  }

  /**
   * Restore `navigator.mediaDevices.getUserMedia` to its original implementation.
   *
   * Call this when the adapter is no longer needed to avoid interfering with
   * other code that calls `getUserMedia` after the call ends.
   */
  uninstall(): void {
    if (this._originalGetUserMedia !== null && navigator.mediaDevices) {
      navigator.mediaDevices.getUserMedia = this._originalGetUserMedia;
      this._originalGetUserMedia = null;
    }
  }

  /**
   * Attach the adapter to an existing video element and media stream.
   *
   * - Creates a hidden canvas matching the video's intrinsic dimensions.
   * - Creates and positions the caption overlay.
   * - Starts the requestAnimationFrame processing loop.
   * - Attaches stream interruption listeners.
   *
   * @param videoElement The <video> element showing the remote/local feed.
   * @param stream       The MediaStream assigned to videoElement.srcObject.
   *
   * Requirements: 6.1, 6.2, 6.3
   */
  attach(videoElement: HTMLVideoElement, stream: MediaStream): void {
    this.detach(); // Clean up any existing attachment first.

    this._videoElement = videoElement;

    // Create hidden canvas for frame capture.
    this._canvas = document.createElement('canvas');
    this._canvas.style.display = 'none';
    document.body.appendChild(this._canvas);
    this._ctx = this._canvas.getContext('2d');

    // Create and position the caption overlay (Req 6.2).
    this._overlay = this.createCaptionOverlay(videoElement);

    // Listen for stream interruption / reconnection (Req 6.4).
    this._attachStreamListeners(stream);

    // Start the capture loop once the video metadata is ready so we have
    // valid intrinsic dimensions.
    if (videoElement.readyState >= HTMLMediaElement.HAVE_METADATA) {
      this._syncCanvasDimensions();
      this._startLoop();
    } else {
      const onLoadedMetadata = () => {
        videoElement.removeEventListener('loadedmetadata', onLoadedMetadata);
        this._syncCanvasDimensions();
        this._startLoop();
      };
      videoElement.addEventListener('loadedmetadata', onLoadedMetadata);
    }
  }

  /**
   * Create and position a caption panel over a video element.
   *
   * The overlay is:
   *  - Absolutely positioned at the bottom of the video's bounding box.
   *  - Full-width, dark translucent background, white text.
   *  - Appended to `document.body` (avoids iframe containment issues in
   *    video-call platforms) and updated on every `resize` event.
   *
   * @param videoElement The <video> element to overlay.
   * @returns            The created <div> overlay element.
   *
   * Requirements: 6.2
   */
  createCaptionOverlay(videoElement: HTMLVideoElement): HTMLDivElement {
    const overlay = document.createElement('div');

    // Base styles — positioned to match the video element.
    overlay.style.cssText = [
      'position: fixed',
      'left: 0',
      'right: 0',
      'bottom: 0',
      'width: 100%',
      'padding: 8px 12px',
      'background: rgba(0, 0, 0, 0.65)',
      'color: #ffffff',
      'font-family: system-ui, sans-serif',
      'font-size: 16px',
      'line-height: 1.4',
      'text-align: center',
      'word-break: break-word',
      `z-index: ${OVERLAY_Z_INDEX}`,
      'pointer-events: none',          // allow clicks to pass through to UI
      'box-sizing: border-box',
      'transition: background 0.2s ease',
    ].join('; ');

    overlay.setAttribute('role', 'status');
    overlay.setAttribute('aria-live', 'polite');
    overlay.setAttribute('aria-label', 'Sign language caption');

    // Position the overlay relative to the video element's current layout.
    this._repositionOverlay(overlay, videoElement);

    // Re-position whenever the window is resized (video element moves).
    const onResize = () => this._repositionOverlay(overlay, videoElement);
    window.addEventListener('resize', onResize);

    // Store the resize handler on the element so detach() can remove it.
    (overlay as HTMLDivElement & { _resizeHandler?: () => void })._resizeHandler =
      onResize;

    document.body.appendChild(overlay);
    return overlay;
  }

  /**
   * Update the caption panel with new text.
   *
   * @param text          The caption to display (predicted letter / gloss string).
   * @param isInterrupted When true, overrides `text` with the stream-interrupted
   *                      message and tints the overlay red (Req 6.4).
   *
   * Requirements: 6.2, 6.4
   */
  updateCaption(text: string, isInterrupted = false): void {
    if (this._overlay === null) return;

    const displayText = isInterrupted ? INTERRUPTED_TEXT : text;
    this._overlay.textContent = displayText;

    // Visual distinction for the interrupted state.
    if (isInterrupted) {
      this._overlay.style.background = 'rgba(180, 0, 0, 0.75)';
    } else {
      this._overlay.style.background = 'rgba(0, 0, 0, 0.65)';
    }

    this._onCaption?.(displayText);
  }

  /**
   * Stop the frame-capture loop, remove the overlay, and release the canvas.
   *
   * After `detach()` the adapter instance can be reused by calling `attach()`
   * with a new video element and stream.
   */
  detach(): void {
    this._stopLoop();

    // Remove overlay and its resize listener.
    if (this._overlay !== null) {
      const handler = (
        this._overlay as HTMLDivElement & { _resizeHandler?: () => void }
      )._resizeHandler;
      if (handler) {
        window.removeEventListener('resize', handler);
      }
      this._overlay.remove();
      this._overlay = null;
    }

    // Remove hidden canvas.
    if (this._canvas !== null) {
      this._canvas.remove();
      this._canvas = null;
      this._ctx = null;
    }

    this._videoElement = null;
    this._paused = false;
    this._interrupted = false;
  }

  // ---------------------------------------------------------------------------
  // Frame processing loop
  // ---------------------------------------------------------------------------

  /**
   * Start the requestAnimationFrame loop.
   */
  private _startLoop(): void {
    if (this._rafHandle !== null) return; // Already running.
    this._paused = false;
    this._scheduleNextFrame();
  }

  /**
   * Stop the rAF loop without fully detaching.
   * Called when the stream is interrupted; `_startLoop()` restarts it.
   */
  private _stopLoop(): void {
    if (this._rafHandle !== null) {
      cancelAnimationFrame(this._rafHandle);
      this._rafHandle = null;
    }
  }

  /** Schedule the next animation frame tick. */
  private _scheduleNextFrame(): void {
    this._rafHandle = requestAnimationFrame((ts) => this._onAnimationFrame(ts));
  }

  /**
   * Called by the browser on each display refresh (≈60 Hz, or as fast as
   * the display allows).  We enforce the 33 ms/frame budget by measuring
   * wall-clock time and logging a warning when exceeded.
   */
  private _onAnimationFrame(timestamp: number): void {
    this._rafHandle = null;

    if (this._paused) return; // Loop suspended during interruption.

    const frameStart = performance.now();

    // Only process if the video is playing and has valid dimensions.
    const video = this._videoElement;
    if (
      video !== null &&
      video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
      !video.paused &&
      video.videoWidth > 0
    ) {
      this._syncCanvasDimensions();
      this._processFrame();
    }

    const elapsed = performance.now() - frameStart;
    if (elapsed > FRAME_BUDGET_MS) {
      console.warn(
        `WebRTCAdapter: frame processing took ${elapsed.toFixed(1)} ms ` +
          `(budget: ${FRAME_BUDGET_MS} ms). Consider reducing MediaPipe complexity.`,
      );
    }

    // Schedule next frame (only if loop was not stopped during processing).
    if (!this._paused) {
      this._scheduleNextFrame();
    }
  }

  /**
   * Capture the current video frame, run landmark extraction, and update the
   * caption overlay.
   *
   * **MediaPipe stub** — replace the stub section below with actual MediaPipe
   * Web integration (e.g. `@mediapipe/tasks-vision` HandLandmarker) to wire
   * real landmark extraction.
   *
   * Requirements: 6.1, 6.3
   */
  private _processFrame(): void {
    const video = this._videoElement;
    const ctx = this._ctx;
    const canvas = this._canvas;

    if (video === null || ctx === null || canvas === null) return;

    // Draw the current video frame to the hidden canvas.
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // -------------------------------------------------------------------
    // MediaPipe stub — replace this block with real landmark extraction.
    //
    // Example integration with @mediapipe/tasks-vision:
    //
    //   const result = handLandmarker.detectForVideo(video, performance.now());
    //   if (result.landmarks.length > 0) {
    //     const normalized = normalizeLandmarks(result.landmarks[0]);
    //     const prediction = await ortEngine.runFingerspell(normalized);
    //     this.updateCaption(prediction[0].label);
    //     this._onPrediction?.({ ... });
    //   }
    //
    // The canvas pixel data is available via:
    //   const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    // -------------------------------------------------------------------

    // Stub: emit a placeholder prediction result for wiring / testing.
    const stubResult: PredictionResult = {
      type: 'fingerspell',
      prediction: '',
      confidence: 0,
      top5: [],
      accepted: false,
      acceptedLabel: null,
      latencyMs: 0,
    };
    this._onPrediction?.(stubResult);
  }

  // ---------------------------------------------------------------------------
  // Canvas helpers
  // ---------------------------------------------------------------------------

  /**
   * Keep the hidden canvas dimensions in sync with the video's intrinsic size.
   * Called before every draw so resize events are handled automatically.
   */
  private _syncCanvasDimensions(): void {
    const video = this._videoElement;
    const canvas = this._canvas;
    if (video === null || canvas === null) return;

    const w = video.videoWidth || video.clientWidth;
    const h = video.videoHeight || video.clientHeight;

    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
  }

  // ---------------------------------------------------------------------------
  // Overlay positioning
  // ---------------------------------------------------------------------------

  /**
   * Position the overlay to sit at the bottom of the video element in
   * viewport coordinates.  Called on mount and on every resize event.
   */
  private _repositionOverlay(
    overlay: HTMLDivElement,
    videoElement: HTMLVideoElement,
  ): void {
    const rect = videoElement.getBoundingClientRect();

    overlay.style.position = 'fixed';
    overlay.style.left = `${rect.left}px`;
    overlay.style.width = `${rect.width}px`;
    overlay.style.bottom = `${window.innerHeight - rect.bottom}px`;
    // Reset the right / general width override we set in createCaptionOverlay:
    overlay.style.right = 'auto';
  }

  // ---------------------------------------------------------------------------
  // Stream interruption handling (Req 6.4)
  // ---------------------------------------------------------------------------

  /**
   * Attach `ended` event listeners to all video tracks in a stream, and an
   * `addtrack` listener on the stream itself for auto-resume.
   *
   * Requirements: 6.4
   */
  private _attachStreamListeners(stream: MediaStream): void {
    // Listen for each video track ending.
    for (const track of stream.getVideoTracks()) {
      track.addEventListener('ended', () => this._handleTrackEnded(stream));
    }

    // Also listen for the legacy onended property used in some browsers.
    for (const track of stream.getVideoTracks()) {
      track.onended = () => this._handleTrackEnded(stream);
    }

    // Resume when a new video track is added (stream reconnected).
    stream.addEventListener('addtrack', (event: MediaStreamTrackEvent) => {
      if (event.track.kind === 'video') {
        this._handleTrackAdded(stream);
      }
    });
  }

  /**
   * Handle a video track ending — pause inference and show interrupted indicator.
   *
   * Requirements: 6.4
   */
  private _handleTrackEnded(stream: MediaStream): void {
    // Check if any video tracks are still active before declaring interruption.
    const anyActive = stream.getVideoTracks().some((t) => t.readyState === 'live');
    if (anyActive) return;

    if (!this._interrupted) {
      this._interrupted = true;
      this._paused = true;
      this._stopLoop();
      this.updateCaption('', /* isInterrupted */ true);
    }
  }

  /**
   * Handle a new video track being added to the stream — resume inference
   * automatically.
   *
   * Requirements: 6.4
   */
  private _handleTrackAdded(stream: MediaStream): void {
    if (!this._interrupted) return;

    // Attach ended listeners to any new tracks.
    for (const track of stream.getVideoTracks()) {
      track.addEventListener('ended', () => this._handleTrackEnded(stream));
      track.onended = () => this._handleTrackEnded(stream);
    }

    this._interrupted = false;
    this.updateCaption('', /* isInterrupted */ false);
    this._startLoop();
  }
}
