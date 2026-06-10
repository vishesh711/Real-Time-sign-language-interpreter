/**
 * ASLClient — WebSocket client that connects MediaPipe landmark extraction to
 * the FastAPI server-side inference path.
 *
 * Pipeline:
 *   Webcam → MediaPipe (browser) → normalizeLandmarks() → WebSocket → FastAPI
 *                                                         → prediction → callbacks
 *
 * Features:
 *  - Exponential backoff reconnect: 500 ms → 8 s (factor 2×)
 *  - 10 s ping/pong keepalive (sends { "type": "ping" } every 10 s)
 *  - onPrediction / onAccepted / onStatus / onError callbacks
 *  - sendLandmarks() serializes and sends the landmark payload to the server
 *
 * Requirements: 4.4, 4.5, 6.1
 */

import type { PredictionResult } from './types';

// ---------------------------------------------------------------------------
// Reconnect / keepalive constants
// ---------------------------------------------------------------------------

/** Initial backoff delay in milliseconds. */
const BACKOFF_INITIAL_MS = 500;

/** Maximum backoff delay in milliseconds. */
const BACKOFF_MAX_MS = 8000;

/** Backoff growth factor applied after each failed attempt. */
const BACKOFF_FACTOR = 2;

/** Interval in milliseconds between ping messages for keepalive. */
const PING_INTERVAL_MS = 10_000;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** Inference mode — determines the WebSocket endpoint and payload shape. */
export type ASLMode = 'fingerspell' | 'word';

/** Connection lifecycle status forwarded via the onStatus callback. */
export type ConnectionStatus =
  | 'connecting'
  | 'connected'
  | 'disconnected'
  | 'reconnecting';

/** Options passed to the ASLClient constructor. */
export interface ASLClientOptions {
  /** WebSocket URL, e.g. `ws://localhost:8000/ws/fingerspell`. */
  wsUrl: string;

  /** Inference mode: 'fingerspell' (single-frame, 63 floats) or 'word' (two-hand, 126 floats). */
  mode: ASLMode;

  /** Called when a prediction response is received from the server. */
  onPrediction?: (result: PredictionResult) => void;

  /** Called when the server commits a label (result.accepted === true). */
  onAccepted?: (label: string) => void;

  /** Called on connection lifecycle events. */
  onStatus?: (status: ConnectionStatus) => void;

  /** Called on connection errors or message parse errors. */
  onError?: (err: Error) => void;
}

// ---------------------------------------------------------------------------
// ASLClient
// ---------------------------------------------------------------------------

/**
 * Browser WebSocket client for the sign-language-interpreter inference server.
 *
 * Usage:
 *   const client = new ASLClient({ wsUrl, mode: 'fingerspell', onPrediction: ... });
 *   client.connect();
 *   // inside animation loop:
 *   client.sendLandmarks(landmarks63, 'Right');
 *   // when done:
 *   client.disconnect();
 */
export class ASLClient {
  // -- configuration ---------------------------------------------------------
  private readonly _wsUrl: string;
  private readonly _mode: ASLMode;
  private readonly _onPrediction?: (result: PredictionResult) => void;
  private readonly _onAccepted?: (label: string) => void;
  private readonly _onStatus?: (status: ConnectionStatus) => void;
  private readonly _onError?: (err: Error) => void;

  // -- state -----------------------------------------------------------------
  private _ws: WebSocket | null = null;

  /** Whether the client has been explicitly disconnected by the caller. */
  private _intentionalClose = false;

  /** Current exponential backoff delay. */
  private _backoffMs = BACKOFF_INITIAL_MS;

  /** Timer handle for the pending reconnect attempt. */
  private _reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  /** Timer handle for the ping keepalive interval. */
  private _pingTimer: ReturnType<typeof setInterval> | null = null;

  constructor(options: ASLClientOptions) {
    this._wsUrl = options.wsUrl;
    this._mode = options.mode;
    this._onPrediction = options.onPrediction;
    this._onAccepted = options.onAccepted;
    this._onStatus = options.onStatus;
    this._onError = options.onError;
  }

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  /**
   * Open the WebSocket connection.
   *
   * If the connection fails or drops, ASLClient automatically retries with
   * exponential backoff (500 ms → 8 s cap, factor 2×).
   */
  connect(): void {
    this._intentionalClose = false;
    this._openSocket();
  }

  /**
   * Close the WebSocket connection and cancel all pending timers.
   *
   * After calling disconnect() no reconnect will be attempted.
   */
  disconnect(): void {
    this._intentionalClose = true;
    this._clearPingTimer();
    this._clearReconnectTimer();

    if (this._ws !== null) {
      // Remove listeners before closing to avoid triggering the reconnect path.
      this._removeSocketListeners(this._ws);
      this._ws.close();
      this._ws = null;
    }

    this._onStatus?.('disconnected');
  }

  /**
   * Serialize the landmark array and send it to the server.
   *
   * For fingerspell mode: sends `{ landmarks: [63 floats], handedness: "Right" | "Left" }`.
   * For word mode: sends `{ landmarks: [126 floats], frame_idx: <number> }`.
   *
   * @param landmarks   Normalized landmark vector (63 or 126 floats).
   * @param handedness  Hand side — required for fingerspell mode (default "Right").
   * @param frameIdx    Monotonically increasing frame index — used in word mode.
   */
  sendLandmarks(
    landmarks: number[],
    handedness: 'Left' | 'Right' = 'Right',
    frameIdx?: number,
  ): void {
    if (this._ws === null || this._ws.readyState !== WebSocket.OPEN) {
      // Drop silently when not connected — caller should check isConnected
      // or rely on onStatus callbacks.
      return;
    }

    let payload: string;
    if (this._mode === 'fingerspell') {
      // Requirement 4.4: transmit only landmark arrays (no raw video)
      payload = JSON.stringify({ landmarks, handedness });
    } else {
      payload = JSON.stringify({
        landmarks,
        frame_idx: frameIdx ?? 0,
      });
    }

    this._ws.send(payload);
  }

  /**
   * Whether the WebSocket is currently in the OPEN state.
   */
  get isConnected(): boolean {
    return this._ws !== null && this._ws.readyState === WebSocket.OPEN;
  }

  // ---------------------------------------------------------------------------
  // Socket lifecycle
  // ---------------------------------------------------------------------------

  /**
   * Create a new WebSocket and attach event handlers.
   */
  private _openSocket(): void {
    this._onStatus?.('connecting');

    let ws: WebSocket;
    try {
      ws = new WebSocket(this._wsUrl);
    } catch (err) {
      this._onError?.(err instanceof Error ? err : new Error(String(err)));
      this._scheduleReconnect();
      return;
    }

    this._ws = ws;

    ws.addEventListener('open', this._handleOpen);
    ws.addEventListener('message', this._handleMessage);
    ws.addEventListener('error', this._handleError);
    ws.addEventListener('close', this._handleClose);
  }

  /**
   * Remove all event listeners from a WebSocket instance without closing it.
   * Used before calling ws.close() to avoid double-handling.
   */
  private _removeSocketListeners(ws: WebSocket): void {
    ws.removeEventListener('open', this._handleOpen);
    ws.removeEventListener('message', this._handleMessage);
    ws.removeEventListener('error', this._handleError);
    ws.removeEventListener('close', this._handleClose);
  }

  // ---------------------------------------------------------------------------
  // Event handlers (arrow functions so `this` is bound correctly)
  // ---------------------------------------------------------------------------

  private readonly _handleOpen = (): void => {
    // Reset backoff on successful connection.
    this._backoffMs = BACKOFF_INITIAL_MS;
    this._onStatus?.('connected');
    this._startPingTimer();
  };

  private readonly _handleMessage = (event: MessageEvent): void => {
    const raw = typeof event.data === 'string' ? event.data : null;
    if (raw === null) return;

    let parsed: unknown;
    try {
      parsed = JSON.parse(raw);
    } catch {
      this._onError?.(new Error(`ASLClient: failed to parse server message: ${raw}`));
      return;
    }

    // Handle pong responses — no forwarding needed.
    if (
      parsed !== null &&
      typeof parsed === 'object' &&
      (parsed as Record<string, unknown>)['type'] === 'pong'
    ) {
      return;
    }

    // Handle error responses from the server.
    if (
      parsed !== null &&
      typeof parsed === 'object' &&
      (parsed as Record<string, unknown>)['type'] === 'error'
    ) {
      const msg = (parsed as Record<string, unknown>)['message'];
      this._onError?.(new Error(`ASLClient: server error: ${String(msg ?? 'unknown')}`));
      return;
    }

    // Attempt to cast to PredictionResult.
    const result = parsed as PredictionResult;

    // Basic structural validation before forwarding.
    if (
      typeof result.prediction !== 'string' ||
      typeof result.confidence !== 'number'
    ) {
      this._onError?.(
        new Error(`ASLClient: unexpected message shape: ${raw}`),
      );
      return;
    }

    this._onPrediction?.(result);

    if (result.accepted && result.acceptedLabel !== null) {
      this._onAccepted?.(result.acceptedLabel);
    }
  };

  private readonly _handleError = (_event: Event): void => {
    // The browser WebSocket API does not expose meaningful error details via
    // the 'error' event; the 'close' event follows immediately and contains
    // the actual reason.  We emit a generic error here for visibility.
    this._onError?.(new Error('ASLClient: WebSocket connection error'));
  };

  private readonly _handleClose = (_event: CloseEvent): void => {
    this._clearPingTimer();

    if (this._intentionalClose) {
      // Caller explicitly called disconnect() — do not reconnect.
      this._onStatus?.('disconnected');
      return;
    }

    this._onStatus?.('reconnecting');
    this._scheduleReconnect();
  };

  // ---------------------------------------------------------------------------
  // Reconnect logic
  // ---------------------------------------------------------------------------

  /**
   * Schedule the next reconnect attempt after the current backoff delay,
   * then double the delay (capped at BACKOFF_MAX_MS) for the next attempt.
   */
  private _scheduleReconnect(): void {
    this._clearReconnectTimer();

    const delay = this._backoffMs;
    this._reconnectTimer = setTimeout(() => {
      this._reconnectTimer = null;
      if (!this._intentionalClose) {
        this._openSocket();
      }
    }, delay);

    // Advance backoff for the next potential failure.
    this._backoffMs = Math.min(this._backoffMs * BACKOFF_FACTOR, BACKOFF_MAX_MS);
  }

  private _clearReconnectTimer(): void {
    if (this._reconnectTimer !== null) {
      clearTimeout(this._reconnectTimer);
      this._reconnectTimer = null;
    }
  }

  // ---------------------------------------------------------------------------
  // Ping / pong keepalive
  // ---------------------------------------------------------------------------

  /**
   * Start sending a `{"type":"ping"}` message every 10 seconds to keep the
   * WebSocket connection alive through proxies and load balancers.
   */
  private _startPingTimer(): void {
    this._clearPingTimer();
    this._pingTimer = setInterval(() => {
      if (this._ws !== null && this._ws.readyState === WebSocket.OPEN) {
        this._ws.send(JSON.stringify({ type: 'ping' }));
      }
    }, PING_INTERVAL_MS);
  }

  private _clearPingTimer(): void {
    if (this._pingTimer !== null) {
      clearInterval(this._pingTimer);
      this._pingTimer = null;
    }
  }
}
