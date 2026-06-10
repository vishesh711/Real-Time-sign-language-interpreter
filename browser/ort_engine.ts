/**
 * OrtEngine — in-browser ONNX Runtime inference engine.
 *
 * Responsibilities:
 *  - Download ONNX model files with progress reporting.
 *  - Select the best available backend: WebGPU → WASM fallback.
 *  - Run a warm-up inference pass after loading to JIT-compile shaders.
 *  - Expose runFingerspell() and runWord() for the two model heads.
 *  - Provide softmax(), topK(), and resampleSequence() as utilities.
 *
 * Requirements: 4.1, 4.2, 2.5
 */

import * as ort from 'onnxruntime-web';

/** Top-K prediction entry returned by runFingerspell / runWord. */
export interface TopKResult {
  label: string;
  prob: number;
}

/** Options passed to OrtEngine constructor. */
export interface OrtEngineOptions {
  /** URL or path to the INT8-quantized fingerspelling ONNX model. */
  fingerspellModelUrl: string;
  /** URL or path to the INT8-quantized word-level ONNX model. */
  wordModelUrl: string;
  /** Label array for fingerspell model (index → letter). */
  fingerspellLabels: string[];
  /** Label array for word model (index → gloss). */
  wordLabels: string[];
  /**
   * Called during model download with a progress value in [0, 1].
   * Two phases are reported: fingerspell download and word download,
   * each contributing 0.5 of the total progress.
   */
  onProgress?: (progress: number) => void;
}

/** Sequence length the word-level model expects (fixed at 30 frames). */
const WORD_SEQ_LEN = 30;

/** Feature vector dimension per frame for the word model (2 hands × 21 × 3). */
const WORD_FRAME_DIM = 126;

/** Feature vector dimension for the fingerspell model (1 hand × 21 × 3). */
const FINGERSPELL_DIM = 63;

/**
 * Download a model from a URL, reporting byte-level progress.
 * Falls back to a simple fetch if the Response body is not readable.
 *
 * @param url        Model URL.
 * @param onProgress Progress callback receiving a value in [0, 1].
 * @returns          ArrayBuffer containing the raw model bytes.
 */
async function downloadWithProgress(
  url: string,
  onProgress?: (p: number) => void,
): Promise<ArrayBuffer> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to download model from ${url}: HTTP ${response.status}`);
  }

  // Try streaming download with progress.
  const contentLength = response.headers.get('content-length');
  if (response.body && contentLength) {
    const total = parseInt(contentLength, 10);
    const reader = response.body.getReader();
    const chunks: Uint8Array[] = [];
    let received = 0;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      onProgress?.(received / total);
    }

    // Concatenate all chunks into a single ArrayBuffer.
    const buffer = new Uint8Array(received);
    let offset = 0;
    for (const chunk of chunks) {
      buffer.set(chunk, offset);
      offset += chunk.length;
    }
    return buffer.buffer;
  }

  // Fallback: no content-length or no streaming body.
  onProgress?.(1);
  return response.arrayBuffer();
}

/**
 * Attempt to create an ONNX Runtime inference session with WebGPU backend.
 * Returns the session on success, or null if WebGPU is unavailable.
 */
async function tryCreateWebGpuSession(
  modelBuffer: ArrayBuffer,
): Promise<ort.InferenceSession | null> {
  try {
    const session = await ort.InferenceSession.create(modelBuffer, {
      executionProviders: ['webgpu'],
    });
    return session;
  } catch {
    return null;
  }
}

/**
 * Create an ONNX Runtime inference session using the WASM backend.
 * This is the guaranteed-available fallback.
 */
async function createWasmSession(
  modelBuffer: ArrayBuffer,
): Promise<ort.InferenceSession> {
  return ort.InferenceSession.create(modelBuffer, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  });
}

/**
 * Compute the softmax of a logits array, returning a probability vector.
 *
 * @param logits Raw model output logits.
 * @returns      Float32Array of probabilities that sum to 1.
 */
export function softmax(logits: Float32Array | number[]): Float32Array {
  const len = logits.length;
  const probs = new Float32Array(len);

  // Numerical stability: subtract the max before exponentiation.
  let max = -Infinity;
  for (let i = 0; i < len; i++) {
    if (logits[i] > max) max = logits[i];
  }

  let sum = 0;
  for (let i = 0; i < len; i++) {
    probs[i] = Math.exp(logits[i] - max);
    sum += probs[i];
  }
  for (let i = 0; i < len; i++) {
    probs[i] /= sum;
  }
  return probs;
}

/**
 * Return the top-K entries from a probability array, paired with labels.
 *
 * @param probs  Probability array (output of softmax).
 * @param labels Label array aligned with probs indices.
 * @param k      Number of top entries to return (default 5).
 * @returns      Array of { label, prob } sorted descending by probability.
 */
export function topK(
  probs: Float32Array | number[],
  labels: string[],
  k = 5,
): TopKResult[] {
  const len = Math.min(probs.length, labels.length);
  // Build index array and partial-sort to find top-k.
  const indices = Array.from({ length: len }, (_, i) => i);
  indices.sort((a, b) => probs[b] - probs[a]);

  const results: TopKResult[] = [];
  const limit = Math.min(k, len);
  for (let i = 0; i < limit; i++) {
    const idx = indices[i];
    results.push({ label: labels[idx], prob: probs[idx] });
  }
  return results;
}

/**
 * Resample a variable-length sequence of frames to a fixed target length
 * using linear interpolation.
 *
 * @param frames    Input sequence: array of Float32Arrays, each length frameDim.
 * @param targetLen Desired output length (default: WORD_SEQ_LEN = 30).
 * @returns         Float32Array of shape [targetLen × frameDim].
 */
export function resampleSequence(
  frames: Float32Array[],
  targetLen = WORD_SEQ_LEN,
): Float32Array {
  const srcLen = frames.length;
  if (srcLen === 0) {
    throw new Error('resampleSequence: frames array must not be empty');
  }

  const frameDim = frames[0].length;
  const out = new Float32Array(targetLen * frameDim);

  if (srcLen === 1 || targetLen === 1) {
    // Repeat the single source frame (or first frame) for every target position.
    for (let t = 0; t < targetLen; t++) {
      out.set(frames[0], t * frameDim);
    }
    return out;
  }

  for (let t = 0; t < targetLen; t++) {
    // Map target index to a fractional source index.
    const srcPos = (t / (targetLen - 1)) * (srcLen - 1);
    const lo = Math.floor(srcPos);
    const hi = Math.min(lo + 1, srcLen - 1);
    const alpha = srcPos - lo; // interpolation weight

    const outOffset = t * frameDim;
    const loFrame = frames[lo];
    const hiFrame = frames[hi];
    for (let d = 0; d < frameDim; d++) {
      out[outOffset + d] = loFrame[d] * (1 - alpha) + hiFrame[d] * alpha;
    }
  }
  return out;
}

/**
 * OrtEngine manages ONNX Runtime inference sessions for the two model heads
 * (fingerspelling and word-level) and exposes a clean async API.
 *
 * Usage:
 *   const engine = new OrtEngine(options);
 *   await engine.load();                     // download + warm-up
 *   const result = engine.runFingerspell(landmarks63);
 *   const result = engine.runWord(frameBuffer);
 */
export class OrtEngine {
  private readonly options: OrtEngineOptions;

  private fingerspellSession: ort.InferenceSession | null = null;
  private wordSession: ort.InferenceSession | null = null;

  /** Name of the backend that was selected ('webgpu' or 'wasm'). */
  public backendName: 'webgpu' | 'wasm' | null = null;

  constructor(options: OrtEngineOptions) {
    this.options = options;
  }

  /**
   * Download both models, create inference sessions (WebGPU → WASM fallback),
   * and run warm-up inference passes.
   *
   * Progress events are reported via options.onProgress:
   *   0.0 → 0.5: fingerspell model download
   *   0.5 → 1.0: word model download
   * An additional event at exactly 1.0 is emitted when loading is complete.
   */
  async load(): Promise<void> {
    const { fingerspellModelUrl, wordModelUrl, onProgress } = this.options;

    // --- Download fingerspell model (0% → 50%) ---
    const fingerspellBuffer = await downloadWithProgress(
      fingerspellModelUrl,
      (p) => onProgress?.(p * 0.5),
    );

    // --- Download word model (50% → 100%) ---
    const wordBuffer = await downloadWithProgress(
      wordModelUrl,
      (p) => onProgress?.(0.5 + p * 0.5),
    );

    // --- Create inference sessions with backend selection ---
    // Try WebGPU for fingerspell first; if successful, use the same backend for word.
    const webGpuFingerspell = await tryCreateWebGpuSession(fingerspellBuffer);

    if (webGpuFingerspell !== null) {
      // WebGPU available — also create word session with WebGPU.
      const webGpuWord = await tryCreateWebGpuSession(wordBuffer);
      if (webGpuWord !== null) {
        this.fingerspellSession = webGpuFingerspell;
        this.wordSession = webGpuWord;
        this.backendName = 'webgpu';
      } else {
        // WebGPU worked for fingerspell but not word — fall back both to WASM.
        await webGpuFingerspell.release?.();
        this.fingerspellSession = await createWasmSession(fingerspellBuffer);
        this.wordSession = await createWasmSession(wordBuffer);
        this.backendName = 'wasm';
      }
    } else {
      // WebGPU unavailable — use WASM for both.
      this.fingerspellSession = await createWasmSession(fingerspellBuffer);
      this.wordSession = await createWasmSession(wordBuffer);
      this.backendName = 'wasm';
    }

    // --- Warm-up inference passes ---
    // Running dummy data primes any JIT shader compilation or graph optimization,
    // so the first real inference call doesn't incur the setup cost.
    await this._warmUpFingerspell();
    await this._warmUpWord();

    onProgress?.(1);
  }

  /** Warm-up pass for the fingerspell model using a zero-filled input. */
  private async _warmUpFingerspell(): Promise<void> {
    const dummy = new Float32Array(FINGERSPELL_DIM);
    const tensor = new ort.Tensor('float32', dummy, [1, FINGERSPELL_DIM]);
    const inputName = this.fingerspellSession!.inputNames[0];
    await this.fingerspellSession!.run({ [inputName]: tensor });
  }

  /** Warm-up pass for the word model using a zero-filled input. */
  private async _warmUpWord(): Promise<void> {
    const dummy = new Float32Array(WORD_SEQ_LEN * WORD_FRAME_DIM);
    const tensor = new ort.Tensor('float32', dummy, [1, WORD_SEQ_LEN, WORD_FRAME_DIM]);
    const inputName = this.wordSession!.inputNames[0];
    await this.wordSession!.run({ [inputName]: tensor });
  }

  /**
   * Run fingerspelling inference on a single-frame landmark vector.
   *
   * @param landmarks63 Float32Array of length 63 (21 keypoints × 3 coords),
   *                    normalized to wrist origin.
   * @returns           Top-5 predictions sorted by probability.
   * @throws            If the engine has not been loaded, or input length is wrong.
   */
  async runFingerspell(landmarks63: Float32Array): Promise<TopKResult[]> {
    if (!this.fingerspellSession) {
      throw new Error('OrtEngine not loaded. Call load() first.');
    }
    if (landmarks63.length !== FINGERSPELL_DIM) {
      throw new Error(
        `runFingerspell: expected ${FINGERSPELL_DIM} floats, got ${landmarks63.length}`,
      );
    }

    const tensor = new ort.Tensor('float32', landmarks63, [1, FINGERSPELL_DIM]);
    const inputName = this.fingerspellSession.inputNames[0];
    const results = await this.fingerspellSession.run({ [inputName]: tensor });

    const outputName = this.fingerspellSession.outputNames[0];
    const logits = results[outputName].data as Float32Array;
    const probs = softmax(logits);
    return topK(probs, this.options.fingerspellLabels, 5);
  }

  /**
   * Run word-level inference on a variable-length frame buffer.
   *
   * The frame buffer is resampled to WORD_SEQ_LEN (30) frames via linear
   * interpolation before being fed to the model.
   *
   * @param frameBuffer Array of frames, each a Float32Array of length 126
   *                    (two-hand vector: right_63 | left_63).
   * @returns           Top-5 gloss predictions sorted by probability.
   * @throws            If the engine has not been loaded, or the buffer is empty.
   */
  async runWord(frameBuffer: Float32Array[]): Promise<TopKResult[]> {
    if (!this.wordSession) {
      throw new Error('OrtEngine not loaded. Call load() first.');
    }
    if (frameBuffer.length === 0) {
      throw new Error('runWord: frameBuffer must contain at least one frame');
    }

    // Resample to fixed 30-frame sequence.
    const resampled = resampleSequence(frameBuffer, WORD_SEQ_LEN);

    const tensor = new ort.Tensor('float32', resampled, [1, WORD_SEQ_LEN, WORD_FRAME_DIM]);
    const inputName = this.wordSession.inputNames[0];
    const results = await this.wordSession.run({ [inputName]: tensor });

    const outputName = this.wordSession.outputNames[0];
    const logits = results[outputName].data as Float32Array;
    const probs = softmax(logits);
    return topK(probs, this.options.wordLabels, 5);
  }

  /**
   * Release the ONNX Runtime sessions and free GPU/WASM memory.
   * After calling dispose(), load() must be called again before inference.
   */
  async dispose(): Promise<void> {
    if (this.fingerspellSession) {
      await this.fingerspellSession.release?.();
      this.fingerspellSession = null;
    }
    if (this.wordSession) {
      await this.wordSession.release?.();
      this.wordSession = null;
    }
    this.backendName = null;
  }
}
