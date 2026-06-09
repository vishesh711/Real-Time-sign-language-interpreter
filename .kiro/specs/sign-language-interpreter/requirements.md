# Requirements Document

## Introduction

A real-time computer vision pipeline that translates ASL (American Sign Language) and BSL (British Sign Language) gestures from a live webcam or uploaded video into text and synthesized speech. The system targets deaf and hard-of-hearing individuals who need to communicate in settings without human interpreters. It operates via a web interface with two deployment paths: server-side inference for high-accuracy word-level recognition, and in-browser inference for privacy-first fingerspelling. The system is designed to be embeddable in video call platforms via WebRTC.

## Glossary

- **System**: The Sign Language Interpreter application (frontend + backend together)
- **Pipeline**: The ordered sequence of stages — capture → landmark extraction → classification → output
- **Landmark**: A 3D coordinate (x, y, z) representing a joint or keypoint on the hand or body skeleton
- **MediaPipe**: Google's open-source framework for real-time hand and body landmark detection, offering both a Hands-only model (21 keypoints per hand) and a Holistic model (21 hand + 33 body pose + 468 face landmarks)
- **MediaPipe Holistic**: The full-body MediaPipe model that captures hand landmarks, body pose, and facial landmarks — used to encode non-manual markers (eyebrow raises, head tilts) that are grammatically meaningful in ASL
- **Non-manual markers**: Facial expressions, eyebrow positions, and head movements that carry grammatical meaning in ASL (e.g., raised eyebrows signal yes/no questions)
- **MLP**: Multi-Layer Perceptron — a feedforward neural network used for static gesture classification
- **3D CNN + LSTM**: A hybrid deep learning architecture combining 3D convolutional layers (spatial-temporal features) with Long Short-Term Memory layers (sequential modeling) for word-level sign recognition
- **YOLOv11**: A real-time object/gesture detection model used in the lightweight classification path
- **ONNX**: Open Neural Network Exchange — a portable model format enabling cross-platform inference
- **ONNX Runtime**: An inference engine that executes ONNX models, supporting CPU, GPU (CUDA, DirectML, CoreML), and WebAssembly backends
- **INT8 Quantization**: A model compression technique that reduces weight precision from 32-bit float to 8-bit integer, cutting memory usage and speeding inference
- **WebRTC**: Web Real-Time Communication — a browser API for capturing and streaming audio/video
- **WebSocket**: A persistent, bidirectional network protocol used to stream landmark arrays from browser to server
- **FastAPI**: A Python web framework used for the server-side inference backend
- **TTS**: Text-to-Speech — audio synthesis from recognized text output
- **Fingerspelling**: Spelling out words letter-by-letter using static hand shapes corresponding to A–Z
- **Gloss**: A written representation of a sign language word or phrase
- **ASL**: American Sign Language
- **BOBSL**: The landmark BSL dataset — 1,400 hours of BSL-interpreted BBC broadcast footage with 37 signers and signer-independent evaluation splits; used for BSL word-level model training
- **YouTube-SL-25**: A 3,207-hour multilingual corpus covering 25 sign languages with sentence-level annotations; suitable for multilingual pretraining
- **WLASL**: Word-Level American Sign Language dataset — 2,006 glosses across 21,083 video clips from 119 signers; WLASL-300 (300 most common words, ~5,000 videos) is the recommended training subset for this project; licensed under C-UDA for academic use only
- **Sign Language MNIST**: A dataset of static A–Z hand images (~87k images) used for fingerspelling training; suitable for prototyping but too simple for production
- **ASL Citizen**: A crowd-sourced, consent-collected dataset of 83,399 videos covering 2,731 distinct signs filmed by 52 signers across varied environments; uses a signer-independent test split (no test signers appear in training); achieves 74.16% top-1 on overlapping signs vs. WLASL-2000's 8.49% on the same split — the preferred primary training source for production generalization
- **ASLG-PC12**: A parallel corpus of 87,000 ASL gloss–English sentence pairs used for gloss-to-English translation model training
- **WebGPU**: A browser API for GPU-accelerated computation, used by ONNX Runtime Web for fast in-browser inference
- **Frame**: A single image extracted from a webcam stream or video at a point in time
- **Inference**: The process of running a trained model on input data to produce a prediction
- **Round-trip latency**: The total time from frame capture to displayed prediction result
- **Feature vector**: The normalized landmark array fed to the classifier — for fingerspelling, a single frame's 63 floats (21 × 3) after wrist-relative normalization; for word-level signs, a stacked temporal array of shape [T × 252] where T is the frame window (30 frames), 126 = 21 × 3 × 2 hands, plus 126 velocity (frame-over-frame delta) features appended per frame
- **Velocity features**: Per-frame differences of landmark positions (seq[t] − seq[t−1]) appended to the position features, doubling the feature dimension to 252 and explicitly encoding motion direction and speed
- **Temporal attention**: A learned scalar weight per time step in the LSTM output, allowing the model to up-weight informationally dense frames (peak handshapes) over transition frames
- **Confidence gate**: A pre-classification filter that suppresses inference on frames where MediaPipe detection confidence is below threshold or too many landmarks have low visibility
- **Visibility score**: MediaPipe's per-landmark estimate of whether the joint is visible or occluded; landmarks with low visibility should be treated as unreliable
- **Gloss-to-sentence**: Post-processing that converts a sequence of recognized glosses into grammatically natural text, implemented using a fine-tuned T5 model loaded via Transformers.js
- **Transformers.js**: A JavaScript library by Hugging Face that runs transformer models (T5, CLIP, etc.) directly in the browser using ONNX Runtime Web as its backend; used for the gloss-to-English translation layer

---

## Requirements

### Requirement 1

**User Story:** As a deaf or hard-of-hearing user, I want the system to capture my hand gestures from a webcam in real time, so that I can communicate without requiring a dedicated interpreter.

#### Acceptance Criteria

1. WHEN the user grants camera permission, THE System SHALL begin capturing webcam frames at a target rate of 30 frames per second at a minimum resolution of 640×480 pixels.
2. WHEN a video file is uploaded by the user, THE System SHALL process the file frame-by-frame through the same pipeline as live webcam input.
3. WHEN the webcam stream is active, THE System SHALL extract MediaPipe Holistic landmarks — 21 hand keypoints × 3D coordinates per hand, 33 body pose landmarks, and 468 face landmarks — from each frame within 35 milliseconds.
4. IF no hand is detected in a frame, THEN THE System SHALL skip classification for that frame and maintain the last valid prediction in the output display.
5. WHEN landmark extraction is complete for a frame, THE System SHALL pass the landmark array to the classification stage without storing raw video frames on the server.
6. WHEN the full pipeline processes a frame, THE System SHALL complete capture, landmark extraction, classification, and text render within a total end-to-end latency of 100 milliseconds.
7. WHEN landmark extraction produces results, THE System SHALL apply a confidence gate and skip classifier inference for any frame where the MediaPipe per-hand detection confidence is below 0.8 or more than 3 landmarks have a visibility score below 0.5.

---

### Requirement 2

**User Story:** As a user performing fingerspelling, I want the system to recognize individual ASL letters (A–Z) in real time, so that I can spell out words that have no established sign.

#### Acceptance Criteria

1. WHEN a static hand pose is detected, THE System SHALL classify it against the 26 ASL fingerspelling classes using the lightweight MLP model within 20 milliseconds of landmark extraction.
2. WHEN the MLP model produces a classification, THE System SHALL return the predicted letter along with a confidence score in the range [0.0, 1.0].
3. WHEN the confidence score is below 0.6, THE System SHALL display the prediction with a visual low-confidence indicator rather than suppressing it.
4. WHEN consecutive identical predictions are received for 500 milliseconds, THE System SHALL commit the letter to the output text buffer.
5. WHEN the fingerspelling model is loaded in-browser, THE System SHALL execute inference using ONNX Runtime Web with the INT8-quantized model within 20 milliseconds per frame on standard consumer hardware.

---

### Requirement 3

**User Story:** As a user communicating in full ASL vocabulary, I want the system to recognize word-level signs from video sequences, so that I can express complete thoughts beyond the fingerspelling alphabet.

#### Acceptance Criteria

1. WHEN a sequence of frames representing a sign is detected, THE System SHALL classify the sequence against the WLASL-300 vocabulary (300 most common ASL words) using the 3D CNN + LSTM model operating on a temporal window of 30 frames with a feature vector of shape [30 × 252] (21 keypoints × 3 coordinates × 2 hands with wrist-relative normalization, plus per-frame velocity features appended); body pose landmarks from MediaPipe Holistic SHALL be used to distinguish location-dependent sign pairs.
2. WHEN the 3D CNN + LSTM model produces a word-level prediction, THE System SHALL return the predicted gloss along with a confidence score in the range [0.0, 1.0].
3. WHEN operating in server-side mode, THE System SHALL complete word-level inference within 80 milliseconds of receiving the landmark sequence via WebSocket.
4. WHEN a sequence of glosses is accumulated, THE System SHALL apply gloss-to-sentence post-processing to produce grammatically natural output text.
5. WHEN the server-side path is active, THE System SHALL transmit only landmark arrays over the WebSocket and SHALL NOT transmit raw video frames to the server.

---

### Requirement 4

**User Story:** As a user who values privacy, I want the option to run the entire inference pipeline in-browser, so that my video and gesture data never leave my device.

#### Acceptance Criteria

1. WHERE the user selects in-browser mode, THE System SHALL load the INT8-quantized ONNX model directly into the browser and execute all inference locally without contacting the server.
2. WHERE the user selects in-browser mode, THE System SHALL complete the full pipeline (landmark extraction + classification) within 20 milliseconds per frame using ONNX Runtime Web with WebGPU acceleration.
3. WHERE in-browser mode is active, THE System SHALL NOT transmit any landmark data, video frames, or prediction results to any external server.
4. WHERE the user selects server-side mode, THE System SHALL transmit only landmark arrays (not raw video) over a WebSocket connection to the FastAPI backend.
5. WHEN the user switches between in-browser and server-side modes, THE System SHALL complete the transition within 2 seconds without interrupting the webcam stream.

---

### Requirement 5

**User Story:** As a user, I want recognized signs and letters to be displayed as text and optionally spoken aloud, so that people around me can understand what I am communicating.

#### Acceptance Criteria

1. WHEN a letter or word is committed to the output buffer, THE System SHALL display the accumulated text in a prominent, readable output panel within 100 milliseconds.
2. WHEN the user activates TTS output, THE System SHALL synthesize the committed text to speech using the Web Speech API (browser) or pyttsx3 (server fallback).
3. WHEN the output text panel is updated, THE System SHALL preserve the full session transcript and allow the user to copy or clear it.
4. WHEN the user clears the output buffer, THE System SHALL reset the display and the internal gloss accumulation state simultaneously.
5. WHEN the TTS engine receives text, THE System SHALL serialize the text and deserialize it back to confirm fidelity before speech synthesis begins.
6. WHEN a prediction is in the hold window but not yet committed, THE System SHALL display it as a visual candidate distinct from committed text, so the user can see the system is processing.
7. WHEN a fingerspelled letter stream is received, THE System SHALL collapse repeated letters and detect word boundaries using a pause-threshold of 8 silent frames before reconstructing the word.
8. WHEN a fingerspelled word is reconstructed, THE System SHALL apply edit-distance spell checking against a vocabulary to correct common classifier confusions (A↔E, M↔N, S↔A substitutions).
9. WHEN a sequence of ASL glosses is committed, THE System SHALL apply gloss-to-English post-processing using a quantized T5 model (loaded via Transformers.js, INT8) to produce grammatically natural output text.
10. WHEN the user performs a designated backspace gesture (closed fist held for 600 milliseconds), THE System SHALL delete the last committed word from the output buffer.

---

### Requirement 6

**User Story:** As a developer or meeting participant, I want the interpreter to be embeddable in a video call platform, so that sign language users can communicate during live video calls without switching applications.

#### Acceptance Criteria

1. WHERE WebRTC integration is enabled, THE System SHALL intercept the active media stream and run landmark extraction on each captured frame.
2. WHERE WebRTC integration is enabled, THE System SHALL overlay a real-time caption panel on the video call UI displaying the current prediction.
3. WHEN the WebRTC media stream provides frames, THE System SHALL process each frame through the landmark extraction stage within 33 milliseconds.
4. IF the WebRTC stream is interrupted, THEN THE System SHALL pause inference, display a stream-interrupted indicator, and resume automatically when the stream reconnects.

---

### Requirement 7

**User Story:** As a developer, I want the ML models to be exportable to ONNX format and quantizable to INT8, so that they can be deployed efficiently across server and browser environments.

#### Acceptance Criteria

1. WHEN a trained PyTorch model is exported, THE System SHALL produce a valid ONNX model file that passes ONNX model checker validation.
2. WHEN the ONNX model is quantized to INT8, THE System SHALL produce a quantized model whose output predictions match the full-precision model with a top-1 accuracy difference of no more than 2 percentage points on the WLASL-300 test set.
3. WHEN the ONNX model is serialized to disk, THE System SHALL deserialize it and produce an equivalent model that generates identical predictions on the same inputs.
4. WHEN the quantized ONNX model is loaded in ONNX Runtime, THE System SHALL run inference on a batch of landmark arrays and return results within the latency targets defined in Requirements 2.1 and 3.3.

---

### Requirement 8

**User Story:** As a developer, I want a well-structured FastAPI backend with a WebSocket endpoint, so that the server-side inference path is reliable, testable, and maintainable.

#### Acceptance Criteria

1. WHEN a WebSocket client connects to the inference endpoint, THE System SHALL accept the connection and begin receiving landmark arrays encoded as JSON.
2. WHEN a landmark array message is received, THE System SHALL validate the payload structure (63 floats for fingerspelling, 126 floats for word-level per frame) before passing it to the inference engine.
3. WHEN the inference engine returns a result, THE System SHALL respond over the WebSocket with a JSON payload containing the predicted class label, confidence score, top-5 predictions, accepted flag, and latency in milliseconds.
4. IF a malformed or out-of-range landmark payload is received, THEN THE System SHALL return a structured JSON error response and maintain the WebSocket connection.
5. WHEN the landmark payload is serialized to JSON for transmission, THE System SHALL deserialize the received JSON and confirm the landmark values are within the expected range [−1.0, 1.0] before inference.

---

### Requirement 9

**User Story:** As a user, I want the system to monitor webcam input quality and provide real-time feedback, so that I can adjust my position, lighting, or background to improve recognition accuracy.

#### Acceptance Criteria

1. WHEN landmark extraction runs on a frame, THE System SHALL read the MediaPipe per-hand detection confidence score and expose it to the UI status indicator.
2. WHEN the luminance of the hand bounding box region falls below a threshold of 80 lux equivalent, THE System SHALL display a low-lighting warning in the camera status panel.
3. WHEN the delivered frame rate drops below 25 frames per second for more than 500 milliseconds, THE System SHALL display a low frame-rate warning in the camera status panel.
4. WHEN a two-handed sign is in progress and only one hand is visible in the frame, THE System SHALL display a hands-not-in-frame indicator in the camera status panel.
5. WHEN raw per-frame classifier predictions are produced, THE System SHALL apply softmax temperature scaling to the logits, followed by a sliding-window majority vote over the last 7 frames, before committing a letter or word to the output buffer.
6. WHEN a predicted class appears for fewer than 400 milliseconds in the smoothing window, THE System SHALL treat the prediction as transient and suppress commitment to the output buffer.

---

### Requirement 10

**User Story:** As a developer, I want the word-level model to be trained using a principled dataset strategy, so that the system generalizes to real users whose signing styles were not seen during training.

#### Acceptance Criteria

1. WHEN training the word-level model, THE System SHALL use WLASL-300 for initial pre-training and ASL Citizen as the primary fine-tuning dataset.
2. WHEN evaluating model accuracy, THE System SHALL report top-1 and top-5 accuracy using ASL Citizen's signer-independent test split, where no test signer appears in the training or validation sets.
3. WHEN loading ASL Citizen training data, THE System SHALL use the official pre-defined train/val/test split from the dataset metadata rather than a randomly shuffled split, to prevent signer identity leakage between splits.
4. WHEN training on WLASL-300, THE System SHALL apply a weighted random sampler to compensate for class imbalance across the long-tailed gloss frequency distribution.
5. WHEN reporting model performance, THE System SHALL include a per-class accuracy breakdown to identify low-performing sign classes rather than reporting overall accuracy alone.
