# Real-Time Sign Language Interpreter

A real-time computer vision pipeline that translates **ASL** (American Sign Language) and **BSL** (British Sign Language) hand gestures from a live webcam or uploaded video into text and synthesized speech. The system is designed for deaf and hard-of-hearing users who need to communicate without a dedicated human interpreter, and can be embedded in video call platforms via WebRTC.

## How It Works

The pipeline is **landmark-first**: MediaPipe Holistic runs in the browser to extract skeletal coordinates from each frame. Raw video never leaves the device in privacy mode — only normalized landmark arrays are sent to the server when using the server-side path.

```
Webcam / Video (30 fps, 720p)
    ↓
MediaPipe Holistic (21 hand × 2 + 33 pose + 468 face landmarks)
    ↓ confidence gate (detection ≥ 0.8, ≥ 18/21 landmarks visible)
Classification
    ├── FingerspellMLP  — static pose → letter (A–Z, 0–9)
    └── SignLSTM        — 30-frame sequence → gloss (WLASL-300)
    ↓
Post-processing (temporal gate → spell-check → gloss-to-English)
    ↓
Output (text panel + TTS)
```

### Deployment Modes

| Mode | Inference | Network | Best for |
|------|-----------|---------|----------|
| **In-browser** | ONNX Runtime Web (WebGPU → WASM fallback) | No data sent | Fingerspelling, privacy-first use |
| **Server-side** | FastAPI + ONNX Runtime (GPU/CPU) | Landmark arrays only via WebSocket | Word-level sign recognition |

## Features

- **Fingerspelling** — Recognize individual ASL letters (A–Z) and digits (0–9) from static hand poses in ≤ 20 ms
- **Word-level signs** — Classify 300 common ASL words (WLASL-300) from 30-frame motion sequences in ≤ 80 ms
- **Privacy mode** — Run the full fingerspelling pipeline entirely in the browser with no server contact
- **Temporal stability** — Three-layer prediction gate (majority vote → hold queue → cooldown) prevents flickering output
- **Gloss-to-English** — Fine-tuned T5 model converts ASL gloss sequences into natural English (planned)
- **Input quality monitoring** — Real-time feedback on lighting, frame rate, and hand visibility (planned)
- **WebRTC embedding** — Overlay live captions on video call streams (planned)

## Project Structure

```
.
├── browser/                  # Browser-side TypeScript modules
│   ├── types.ts              # Core interfaces (LandmarkResult, PredictionResult, etc.)
│   └── landmarks.ts          # Landmark normalization (mirrors Python logic)
├── models/                   # PyTorch model architectures
│   ├── mlp.py                # FingerspellMLP — static pose classifier
│   └── cnn3d_lstm.py         # SignLSTM — 3D CNN + BiLSTM word classifier
├── utils/                    # Shared Python utilities
│   ├── landmarks.py          # Normalization, two-hand vectors, velocity features
│   ├── gate.py               # PredictionGate temporal filter
│   ├── dataset.py            # Fingerspelling dataset + augmentation
│   └── sequence_dataset.py   # Word-level sequence dataset (WLASL, ASL Citizen)
├── server/                   # FastAPI WebSocket backend (in progress)
├── tests/
│   ├── property/             # Property-based tests (Hypothesis)
│   └── js/                   # Browser tests (fast-check + Vitest)
├── train.py                  # Fingerspelling training script
├── train_word.py             # Word-level SignLSTM training script
├── export_onnx.py            # PyTorch → ONNX → INT8 quantization
├── requirements.txt          # Python dependencies
├── package.json              # Node.js dependencies
└── .kiro/specs/              # Design, requirements, and task specifications
```

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- A webcam (for live inference once the UI is built)

### Python Setup

Use a **project-local virtual environment** — do not reuse another project's venv (e.g. `agentapp/.venv`), as conflicting or corrupted packages can break installs.

```bash
# Create and activate a virtual environment (Python 3.10+)
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Node.js Setup

```bash
npm install
```

## Training

### Fingerspelling Model (FingerspellMLP)

Train on [Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist) or custom landmark CSVs:

```bash
python train.py --csv data/sign_mnist_train.csv --val-csv data/sign_mnist_test.csv
```

Options include model size (`--model full|small`), learning rate, dropout, and early stopping patience. Checkpoints are saved to `checkpoints/`.

### Word-Level Model (SignLSTM)

Pre-train on WLASL-300, then fine-tune on ASL Citizen with signer-independent splits:

```bash
python train_word.py \
  --wlasl-root data/wlasl_landmarks \
  --asl-root data/asl_citizen_landmarks \
  --asl-csv data/asl_citizen_metadata.csv \
  --amp
```

The training script uses a `WeightedRandomSampler` to handle WLASL class imbalance and reports top-1/top-5 accuracy with per-class breakdowns.

### Export to ONNX

Export a trained checkpoint to ONNX with optional INT8 quantization for browser deployment:

```bash
python export_onnx.py \
  --checkpoint checkpoints/best_fingerspell.pt \
  --output-dir exports/
```

This produces:
- `fingerspell.onnx` — full-precision model
- `fingerspell_int8.onnx` — INT8 quantized model (≤ 2% accuracy drop)
- `fingerspell.labels.json` — class label sidecar

## Testing

### Python (property-based tests with Hypothesis)

```bash
pip install pytest
pytest tests/
```

### JavaScript (property-based tests with fast-check + Vitest)

```bash
npm test
```

Tests validate 17 correctness properties defined in the design spec, including landmark normalization shapes, prediction gate behavior, ONNX round-trip fidelity, dataset signer split integrity, and INT8 quantization agreement.

## Architecture Details

### FingerspellMLP

- **Input**: 63 floats (21 hand landmarks × 3, wrist-normalized)
- **Architecture**: `Linear → BN → ReLU → Dropout` × 3 → `Linear(→36)`
- **Classes**: A–Z + 0–9
- **Latency target**: ≤ 20 ms per frame

### SignLSTM

- **Input**: `[30, 252]` — 30 frames × (126 position + 126 velocity features)
- **Architecture**: Conv3D × 3 → BiLSTM(256, 2 layers) → TemporalAttention → Linear(→300)
- **Vocabulary**: WLASL-300 (300 most common ASL words)
- **Latency target**: ≤ 80 ms (server-side)

### PredictionGate

Applied after every inference call to stabilize output:

1. **Majority vote** over the last 7 frames
2. **Hold queue** — voted label must be stable for 12 consecutive frames
3. **Cooldown** — 20-frame gap between accepted predictions

### Datasets

| Dataset | Purpose | Notes |
|---------|---------|-------|
| [Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist) | Fingerspelling prototyping | ~87k static hand images |
| [WLASL-300](https://dxli94.github.io/WLASL/) | Word-level pre-training | 300 glosses, C-UDA academic license |
| [ASL Citizen](https://aslcitizen.github.io/) | Production fine-tuning | 83k videos, signer-independent test split |
| [ASLG-PC12](https://github.com/sign-language-processing/aslg-pc12) | Gloss-to-English translation | 87k gloss–sentence pairs |

## Implementation Status

| Component | Status |
|-----------|--------|
| Landmark normalization (Python + TypeScript) | Done |
| FingerspellMLP model + training | Done |
| SignLSTM model + training | Done |
| ONNX export + INT8 quantization | Done |
| PredictionGate | Done |
| Property-based tests (partial) | Done |
| FastAPI WebSocket server | Planned |
| Browser inference engine (OrtEngine) | Planned |
| Demo UI + ASLClient | Planned |
| Post-processing (spell-check, gloss-to-English, TTS) | Planned |
| Input quality monitor | Planned |
| WebRTC embedding | Planned |

See [`.kiro/specs/sign-language-interpreter/tasks.md`](.kiro/specs/sign-language-interpreter/tasks.md) for the full implementation plan.

## Specifications

Detailed design and requirements are in the `.kiro/specs/` directory:

- [`design.md`](.kiro/specs/sign-language-interpreter/design.md) — Architecture, components, correctness properties, testing strategy
- [`requirements.md`](.kiro/specs/sign-language-interpreter/requirements.md) — User stories and acceptance criteria
- [`tasks.md`](.kiro/specs/sign-language-interpreter/tasks.md) — Sequential implementation plan

## Tech Stack

**Python**: PyTorch, MediaPipe, ONNX Runtime, FastAPI, Hypothesis, scikit-learn

**Browser**: ONNX Runtime Web, Transformers.js (T5), Web Speech API, Vitest, fast-check

**Models**: FingerspellMLP (MLP), SignLSTM (3D CNN + BiLSTM), T5-small (gloss-to-English)

## License

This project is for educational and research purposes. Note that WLASL is licensed under C-UDA for academic use only. Check individual dataset licenses before use in production.
