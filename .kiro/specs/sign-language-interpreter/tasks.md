# Implementation Plan

- [x] 1. Set up project structure and core interfaces
  - Create directory structure: `models/`, `utils/`, `server/`, `browser/`, `tests/unit/`, `tests/property/`, `tests/js/`
  - Create `requirements.txt` with pinned versions: `torch`, `mediapipe`, `onnxruntime`, `fastapi`, `uvicorn`, `hypothesis`, `pydantic-settings`, `opencv-python`, `pandas`, `scikit-learn`
  - Create `package.json` with `fast-check`, `onnxruntime-web`, `@huggingface/transformers`, `vitest`
  - Define TypeScript interfaces in `browser/types.ts` for `LandmarkResult`, `PredictionResult`, `OutputBuffer`, `QualityStatus`
  - _Requirements: 1.1, 2.1, 3.1, 4.1, 8.1_

- [x] 2. Implement landmark normalization utilities
- [x] 2.1 Implement Python landmark normalizer
  - Write `utils/landmarks.py` with `normalize_landmarks()`, `landmarks_from_mediapipe()`, `get_handedness()`, `build_two_hand_vector()`, label maps (`LABEL_TO_IDX`, `IDX_TO_LABEL`)
  - Normalize to wrist origin, scale by wrist→index-MCP distance, mirror left-hand x-axis
  - _Requirements: 1.3, 1.7_

- [x] 2.2 Write property test for landmark normalization
  - **Property 4: Word-level feature vector has correct shape**
  - **Validates: Requirements 3.1**

- [x] 2.3 Implement velocity feature builder
  - Write `utils/landmarks.py::add_velocity_features(sequence)` that appends frame-over-frame deltas producing `[T, 252]` output
  - _Requirements: 3.1_

- [x] 2.4 Write property test for velocity features
  - **Property 4: Word-level feature vector has correct shape**
  - **Validates: Requirements 3.1**

- [x] 2.5 Implement JavaScript landmark normalizer
  - Write `browser/landmarks.ts` with `normalizeLandmarks()`, `buildTwoHandVector()` — must mirror Python logic exactly (same wrist-origin translation, same index-MCP scale)
  - _Requirements: 1.3, 4.1_

- [x] 2.6 Write property test for JavaScript landmark normalization
  - **Property 4: Word-level feature vector has correct shape** (fast-check)
  - **Validates: Requirements 3.1**

- [x] 3. Implement FingerspellMLP model and training pipeline
- [x] 3.1 Implement FingerspellMLP architecture
  - Write `models/mlp.py` with `FingerspellingMLP(input_dim=63, hidden_dims, num_classes, dropout)` and `FingerspellingMLPSmall`
  - Architecture: `Linear → BN → ReLU → Dropout` × 3 layers → output logits
  - _Requirements: 2.1, 2.2_

- [x] 3.2 Implement offline landmark dataset extraction
  - Write `utils/dataset.py` with `LandmarkDataset`, `build_landmark_csv_from_images()`, augmentation (Gaussian noise, z-axis rotation, scale jitter)
  - Support Sign Language MNIST CSV format and custom image folders
  - _Requirements: 2.1_

- [x] 3.3 Implement fingerspelling training script
  - Write `train.py` with stratified 80/20 split, cosine LR with warm restarts, label smoothing (0.1), early stopping, per-class accuracy breakdown
  - _Requirements: 2.1_

- [x] 3.4 Implement confidence gate
  - Write `utils/gate.py::PredictionGate` with vote window (7 frames), hold queue (12 frames), confidence threshold (0.8), cooldown (20 frames)
  - _Requirements: 2.4, 9.5, 9.6_

- [x] 3.5 Write property test for PredictionGate
  - **Property 3: PredictionGate commits exactly once per stable run**
  - **Validates: Requirements 2.4, 9.5, 9.6**

- [x] 4. Implement ONNX export and quantization pipeline
- [x] 4.1 Implement ONNX export script
  - Write `export_onnx.py` with PyTorch → ONNX export (opset 17, dynamic batch), INT8 dynamic quantization via `onnxruntime.quantization`, labels JSON sidecar
  - _Requirements: 7.1, 7.2, 7.3_

- [x] 4.2 Write property test for ONNX model serialization round-trip
  - **Property 12: ONNX model serialization round-trip**
  - **Validates: Requirements 7.3**

- [x] 4.3 Write property test for INT8 quantization agreement
  - **Property 13: INT8 quantization agreement with FP32**
  - **Validates: Requirements 7.2**

- [x] 5. Implement SignLSTM model and word-level training pipeline
- [x] 5.1 Implement SignLSTM architecture
  - Write `models/cnn3d_lstm.py` with `Conv3DBlock`, `TemporalAttention`, `SignLSTM`
  - Architecture: reshape to `[B, T, 1, 16, 16]` → Conv3D×3 → BiLSTM(hidden=256, layers=2) → TemporalAttention → LayerNorm → Dropout(0.4) → Linear(→300)
  - _Requirements: 3.1, 3.2_

- [x] 5.2 Implement sequence dataset for video-based training
  - Write `utils/sequence_dataset.py` with `SignSequenceDataset`, `_resample()` (linear interpolation to fixed seq_len), `extract_landmarks_from_videos()` offline preprocessing
  - Support WLASL folder structure and ASL Citizen metadata CSV with pre-defined signer-independent splits
  - _Requirements: 3.1, 10.1, 10.2, 10.3_

- [x] 5.3 Write property test for dataset split signer integrity
  - **Property 17: ASL Citizen split signer integrity**
  - **Validates: Requirements 10.3**

- [x] 5.4 Implement word-level training script
  - Write `train_word.py` with `WeightedRandomSampler` for WLASL class imbalance, `OneCycleLR`, mixed-precision (`--amp`), top-1/top-5 evaluation on ASL Citizen signer-independent test split, per-class breakdown
  - _Requirements: 10.1, 10.2, 10.4, 10.5_

- [x] 6. Checkpoint — ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 7. Implement FastAPI WebSocket server
- [ ] 7.1 Implement server configuration and ONNX session registry
  - Write `server/config.py` with `Settings` (pydantic-settings, env/dotenv): model paths, confidence thresholds, gate params, CORS origins
  - Write `server/main.py` lifespan: load ONNX sessions at startup, pick execution providers (CUDA → CoreML → DirectML → CPU), apply `SessionOptions` (ORT_SEQUENTIAL, `intra_op_num_threads=1`, `ORT_ENABLE_ALL`)
  - _Requirements: 8.1_

- [ ] 7.2 Implement WebSocket fingerspelling endpoint
  - Write `/ws/fingerspell` endpoint in `server/main.py`: validate 63-float payload, run ONNX inference, apply PredictionGate per-connection, respond with full JSON schema (`prediction`, `confidence`, `top5`, `accepted`, `accepted_label`, `latency_ms`)
  - _Requirements: 8.1, 8.2, 8.3, 8.4_

- [ ]* 7.3 Write property test for WebSocket payload validation
  - **Property 14: WebSocket payload validation rejects wrong-length arrays**
  - **Validates: Requirements 8.2**

- [ ]* 7.4 Write property test for WebSocket response schema
  - **Property 15: WebSocket response contains all required fields**
  - **Validates: Requirements 8.3**

- [ ] 7.5 Implement WebSocket word-level endpoint
  - Write `/ws/word` endpoint: validate 126-float payloads, maintain rolling frame buffer (capped at 2× seq_len), classify every `word_stride` frames via thread pool executor, apply word PredictionGate, respond with full JSON schema
  - _Requirements: 3.3, 8.1, 8.2, 8.3, 8.4_

- [ ]* 7.6 Write property test for landmark JSON round-trip
  - **Property 16: Landmark JSON serialization round-trip**
  - **Validates: Requirements 8.5**

- [ ] 8. Implement browser inference engine
- [ ] 8.1 Implement OrtEngine class
  - Write `browser/ort_engine.ts` with `OrtEngine`: model download with progress, WebGPU → WASM fallback, warm-up inference pass, `runFingerspell(landmarks63)`, `runWord(frameBuffer)`, `softmax()`, `topK()`, `resampleSequence()`
  - _Requirements: 4.1, 4.2, 2.5_

- [ ]* 8.2 Write property test for OrtEngine round-trip (fast-check)
  - **Property 12: ONNX model serialization round-trip** (browser path)
  - **Validates: Requirements 7.3**

- [ ] 8.3 Implement browser PredictionGate
  - Write `browser/gate.ts` with `PredictionGate` — identical three-layer logic as Python `utils/gate.py` (vote window, hold queue, cooldown)
  - _Requirements: 2.4, 9.5, 9.6_

- [ ]* 8.4 Write property test for browser PredictionGate (fast-check)
  - **Property 3: PredictionGate commits exactly once per stable run**
  - **Validates: Requirements 2.4, 9.5, 9.6**

- [ ] 9. Implement post-processing and output buffer
- [ ] 9.1 Implement OutputBuffer
  - Write `browser/output_buffer.ts` with `OutputBuffer`: `accept(word)`, `backspace()`, `clear()`, `setCandidate()`, `render()` returning `{ committed, candidate }`
  - _Requirements: 5.3, 5.4, 5.10_

- [ ]* 9.2 Write property test for OutputBuffer append invariant (fast-check)
  - **Property 7: Transcript append invariant**
  - **Validates: Requirements 5.3**

- [ ]* 9.3 Write property test for OutputBuffer backspace invariant (fast-check)
  - **Property 11: Backspace decrements committed length**
  - **Validates: Requirements 5.10**

- [ ] 9.4 Implement fingerspelling collapse and spell-check
  - Write `browser/postprocess.ts` with `collapseAndSegment(letterStream, pauseThreshold=8, minLetterHold=4)` and `spellCheck(word, dictionary)` using edit-distance
  - _Requirements: 5.7, 5.8_

- [ ]* 9.5 Write property test for fingerspelling collapse (fast-check)
  - **Property 9: Fingerspelling collapse correctness**
  - **Validates: Requirements 5.7**

- [ ]* 9.6 Write property test for spell-check idempotence (fast-check)
  - **Property 10: Spell-check idempotence on dictionary words**
  - **Validates: Requirements 5.8**

- [ ] 9.7 Implement gloss-to-English post-processor
  - Write `browser/gloss_processor.ts` loading T5-small via Transformers.js (`dtype: 'q8'`), staged loading (after ONNX classifier), `glossToEnglish(glosses)` returning non-empty string or raw gloss fallback
  - _Requirements: 5.9, 3.4_

- [ ]* 9.8 Write property test for gloss-to-English output (fast-check)
  - **Property 5: Gloss-to-English returns non-empty string**
  - **Validates: Requirements 3.4, 5.9**

- [ ] 9.9 Implement TTS integration
  - Wire Web Speech API (`SpeechSynthesisUtterance`) for browser TTS and pyttsx3 for server fallback; implement text serialize/deserialize fidelity check before synthesis
  - _Requirements: 5.2, 5.5_

- [ ]* 9.10 Write property test for TTS text round-trip (fast-check)
  - **Property 8: TTS text round-trip**
  - **Validates: Requirements 5.5**

- [ ] 10. Implement confidence gate and input quality monitor
- [ ] 10.1 Implement MediaPipe confidence gate in Python
  - Write `utils/extractor.py::ConfidenceGate.should_classify(detection_conf, visible_lm_count)` — returns True only when `detection_conf ≥ 0.8` AND `visible_lm_count ≥ 18`
  - _Requirements: 1.7_

- [ ]* 10.2 Write property test for confidence gate (Hypothesis)
  - **Property 1: Confidence gate rejects low-quality frames**
  - **Validates: Requirements 1.7**

- [ ] 10.3 Implement InputQualityMonitor
  - Write `browser/quality_monitor.ts` sampling every 10 frames: read MediaPipe detection confidence, estimate hand-region luminance, track delivered FPS, count visible hands; expose `QualityStatus` to UI
  - _Requirements: 9.1, 9.2, 9.3, 9.4_

- [ ]* 10.4 Write property test for classifier confidence range (Hypothesis)
  - **Property 2: Classifier confidence is always a valid probability**
  - **Validates: Requirements 2.2, 3.2**

- [ ]* 10.5 Write property test for landmark payload schema (Hypothesis)
  - **Property 6: Landmark payload contains no raw image data**
  - **Validates: Requirements 3.5, 4.4**

- [ ] 11. Implement browser ASLClient and main UI
- [ ] 11.1 Implement ASLClient WebSocket client
  - Write `browser/asl_client.ts` with `ASLClient`: MediaPipe initialization, landmark normalization, WS connection with exponential backoff reconnect (500ms → 8s), 10s ping/pong keepalive, `onPrediction` / `onAccepted` / `onStatus` / `onError` callbacks
  - _Requirements: 4.4, 4.5, 6.1_

- [ ] 11.2 Build main demo UI
  - Write `browser/index.html` with: video preview + canvas skeleton overlay, big prediction display (green if confident, gray if uncertain), confidence bar, top-5 chips, word buffer panel with cursor, camera status indicator (detection conf / luminance / FPS / hands), controls (Start / Clear / Space / Speak), TTS button, backend badge (WebGPU / WASM)
  - _Requirements: 5.1, 5.3, 5.6, 9.1, 9.2, 9.3_

- [ ] 12. Checkpoint — ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 13. WebRTC embedding
- [ ] 13.1 Implement WebRTC stream interceptor
  - Write `browser/webrtc_adapter.ts` intercepting `getUserMedia` media stream, running MediaPipe on each captured frame, overlaying caption panel on the video element
  - _Requirements: 6.1, 6.2, 6.3, 6.4_

- [ ] 14. Final Checkpoint — ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.
