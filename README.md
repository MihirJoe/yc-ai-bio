# AlwaysOnPT

**An AI agent that reads raw biosignals and produces clinician-grade analysis — no ML model training required.**

AlwaysOnPT replaces traditional train-then-infer pipelines with a Recursive Language Model (RLM) architecture: an LLM writes and executes signal-processing code in a live Python REPL, inspects the results, and iterates until it reaches a clinical conclusion — complete with a reasoning trace and narrative report.

---

## Why This Matters

| Traditional ML Pipeline | AlwaysOnPT |
|------------------------|------------|
| Collect data → label → train model → deploy | Point the agent at any biosignal and ask a question |
| One model per task | One agent, many task types (EMG, ECG, gait, HRV) |
| Black-box prediction | Full reasoning trace with evidence at every step |
| No clinical context | Generates PT-relevant clinical narratives |
| Needs retraining for new domains | Generalizes out-of-distribution via tool use |

On the Zhang et al. 2017 knee EMG benchmark (4-class lower-limb motion classification), the agent competes with the published WT-SVD + SVM baseline (91.85%) — while also producing fatigue analysis, clinical narratives, and self-acquired skills that the SVM cannot.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        AlwaysOnPT Agent                         │
│                                                                 │
│   ┌──────────┐    ┌───────────────┐    ┌──────────────────┐    │
│   │ Root LLM │───▶│  Python REPL  │───▶│  Signal Tools    │    │
│   │ (Claude) │◀───│  (exec loop)  │    │  (time/freq/     │    │
│   │          │    │               │    │   wavelet/plots)  │    │
│   └──────────┘    └───────────────┘    └──────────────────┘    │
│        │                                        │               │
│        ▼                                        ▼               │
│   ┌──────────┐                         ┌──────────────────┐    │
│   │ Sub-LLM  │                         │  Skill Library   │    │
│   │ (vision, │                         │  (validate →     │    │
│   │  reason) │                         │   register →     │    │
│   └──────────┘                         │   reuse)         │    │
│                                        └──────────────────┘    │
│                                                                 │
│   Output: ReasoningTrace + Classification + Clinical Narrative  │
└─────────────────────────────────────────────────────────────────┘
```

### How a single analysis runs

1. **Task prompt selected** — the agent loop is generic; the system prompt specializes it (EMG classification, gait severity, ECG interpretation, HRV analysis, or open-ended).
2. **REPL initialized** — the record's signal arrays, metadata, and all tool functions are injected into a persistent `exec()` namespace.
3. **Turn loop** (up to 6 turns):
   - The LLM writes a `python` code block.
   - The code runs in the REPL; stdout + errors are captured.
   - Output is fed back as the next user message.
   - The LLM inspects results, writes more code or sets final output variables.
4. **Sub-LLM** — the agent can call a second model for focused reasoning or multimodal analysis (e.g., interpreting a generated plot image).
5. **Skill acquisition** — the agent can write, validate, and register new tool functions at runtime.
6. **Trace** — every step (code execution, observation, error, decision) is recorded in a `ReasoningTrace` and streamed to the frontend via SSE.

---

## Project Structure

```
alwaysonpt/
├── server.py              # FastAPI REST API + SSE streaming + demo UI
├── biosignal_agent.py     # BioSignalAgent — generalized RLM agent
├── rlm_agent.py           # RLMAgent — EMG-specific agent + ReasoningTrace
├── task_prompts.py        # Task registry (6 task types with clinical prompts)
├── data_loader.py         # EMG file parser, highpass filter, segmentation
├── emg_tools.py           # EMG-specific tools (time/freq/wavelet features, plots)
├── signal_tools.py        # Generic signal tools (PSD, variability, cross-corr)
├── eval_harness.py        # 4-class eval vs Zhang et al. 2017 SVM baseline
├── ood_eval.py            # Out-of-distribution eval across PhysioNet datasets
├── synthetic.py           # Synthetic EMG generator
├── demo.py                # CLI demo runner with LangSmith tracing
├── datasets/              # Loaders for PhysioNet datasets (gaitpdb, ptb-xl, etc.)
├── static/index.html      # Single-page demo UI (dark theme, signal viz, trace viewer)
└── requirements.txt       # Python dependencies

live_sensor_demo/
├── download_data.sh       # Download/generate all demo data
└── README.md              # Data setup instructions

ios/AlwaysOnPT/            # iOS companion app (Swift, BLE/IMU sensor streaming)
scripts/                   # PhysioNet dataset download scripts
data/                      # All signal data (gitignored)
├── knee_emg/              #   2-channel EMG dataset (Zhang 2017)
├── live_data_sample/      #   iOS sensor recordings (EMG + IMU)
└── multichannel_emg/      #   16-channel EMG array
```

---

## Quickstart

### Prerequisites

- Python 3.11+
- An [Anthropic API key](https://console.anthropic.com/)
- (Optional) A [LangSmith API key](https://smith.langchain.com/) for tracing

### 1. Clone and install

```bash
git clone <repo-url> && cd yc-ai-bio
python -m venv .venv && source .venv/bin/activate
pip install -r alwaysonpt/requirements.txt
```

### 2. Set environment variables

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=sk-ant-...
LANGSMITH_API_KEY=lsv2_...        # optional
LANGSMITH_PROJECT=alwaysonpt      # optional
```

### 3. Get demo data

The demo supports three data sources. Use the download script to check/generate what you need:

```bash
# Check what data is already present
./live_sensor_demo/download_data.sh --check

# Generate synthetic 16-channel data + verify other sources
./live_sensor_demo/download_data.sh --all
```

| Source | Description | Setup |
|--------|-------------|-------|
| **Knee EMG** | Zhang 2017, 14 subjects, 2ch (EMG + goniometer), 1kHz | Place in `data/knee_emg/S1File/Data/` |
| **Live Sensor** | iOS app recordings, EMG + IMU (pitch/roll/yaw) | Place in `data/live_data_sample/` |
| **16-Channel EMG** | 16-muscle EMG array | Auto-generated synthetic data, or place real data in `data/multichannel_emg/` |

For out-of-distribution evaluation, download PhysioNet datasets:

```bash
./scripts/download_datasets.sh gaitpdb gaitndd ptb-xl chfdb chf2db
```

### 4. Launch the demo

```bash
python -m alwaysonpt.server
```

Open **http://localhost:8000** — use the source tabs to switch between Knee EMG, Live Sensor, and 16-Channel EMG. Select a recording, visualize the signals, and hit **Analyze** to watch the agent reason in real-time.

---

## Supported Task Types

| Task Type | Domain | Output | Clinical Reference |
|-----------|--------|--------|-------------------|
| `emg_classification` | EMG | standing / sitting / stance / swing | Perry & Burnfield 2010, Winter 2009 |
| `gait_severity` | Gait (VGRF) | none / mild / moderate / severe | Hoehn & Yahr scale |
| `gait_diagnosis` | Gait (VGRF) | parkinsons / huntingtons / als / control | Differential diagnosis criteria |
| `ecg_interpretation` | 12-lead ECG | NORM / MI / STTC / CD / HYP / OTHER | Standard ECG interpretation |
| `hrv_analysis` | RR intervals | NYHA functional class (I–IV) | HRV clinical guidelines |
| `open_analysis` | Any signal | Free-form answer to a user question | — |

---

## API Reference

All endpoints are served by FastAPI at `http://localhost:8000`.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Demo UI |
| `GET` | `/api/sources` | List available data sources with channel info |
| `GET` | `/api/subjects` | List available knee EMG subjects and exercises |
| `GET` | `/api/load/{subject_id}/{exercise}` | Load recording, return segments + downsampled signals |
| `GET` | `/api/load/{subject_id}/{exercise}/segment/{idx}` | Load a single segment's raw data |
| `GET` | `/api/live/recordings` | List live sensor recordings |
| `GET` | `/api/live/load/{recording}` | Load a live sensor recording |
| `GET` | `/api/multichannel/recordings` | List 16-channel EMG recordings |
| `GET` | `/api/multichannel/load/{recording_id}` | Load a multichannel recording |
| `POST` | `/api/analyze/stream` | Run agent analysis with SSE streaming |
| `POST` | `/api/analyze` | Run agent analysis (blocking, returns full result) |
| `GET` | `/api/task-types` | List available task types |
| `GET` | `/health` | Health check |

### Analyze request body

```json
{
  "task_type": "emg_classification",
  "signals": { "emg": [0.01, -0.02, ...], "gonio": [5.1, 5.3, ...] },
  "fs": 1000,
  "domain": "emg",
  "record_id": "S1_standing_0",
  "metadata": { "subject_id": 1, "exercise": "standing" },
  "question": null
}
```

### Streamed response (SSE)

Each event is a JSON line with `event` field:

```json
{"event": "step", "step": {"type": "CODE_EXECUTION", "content": "...", "evidence": ["..."], "duration_ms": 45.2}}
{"event": "step", "step": {"type": "OBSERVATION", "content": "RMS = 0.032 mV, gonio range = 12.4 deg", ...}}
{"event": "step", "step": {"type": "DECISION", "content": "classification=standing (confidence: 0.92)", ...}}
{"event": "done", "result": {"classification": "standing", "confidence": 0.92, "clinical_narrative": "...", "steps": [...]}}
```

---

## Running Evaluations

### EMG benchmark (vs Zhang et al. 2017)

```bash
# All segments
python -m alwaysonpt.eval_harness

# Subset (8 segments, 2 per class)
python -m alwaysonpt.eval_harness --max-segments 8
```

Outputs a confusion matrix, per-class precision/recall/F1, and delta against the published 91.85% SVM baseline.

### Out-of-distribution generalization

```bash
python -m alwaysonpt.ood_eval --data-dir data --max-records 10
```

Tests the agent on gait (Parkinson's, Huntington's, ALS), ECG (PTB-XL), and HRV (CHF) datasets it was never prompted or tuned for. Scores: binary accuracy, categorical F1, ordinal rank correlation, and LLM-as-judge reasoning quality.

---

## Signal Tools Available to the Agent

The agent's REPL has access to these functions — it discovers them via `print(TOOLS)` and inspects signatures with `help_tool("name")`:

**EMG-specific:**
- `get_segment_overview(segment)` — summary stats, SNR, data quality
- `extract_time_features(segment)` — MAV, RMS, iEMG, ZC, SSC, WL, VAR
- `extract_freq_features(segment)` — MNF, MDF, peak frequency, band powers
- `extract_wavelet_features(segment)` — db4 wavelet decomposition coefficients
- `detect_fatigue_pattern(segment)` — MDF slope, RMS trend, fatigue index
- `generate_plot(segment, ...)` — multi-panel EMG/goniometer visualization

**Generic signal tools:**
- `compute_statistics(signal)` — descriptive stats
- `compute_psd(signal, fs)` — power spectral density via Welch's method
- `compute_wavelet(signal, ...)` — continuous wavelet transform
- `compute_variability(intervals)` — SDNN, RMSSD, pNN50 for interval data
- `compute_cross_correlation(sig1, sig2)` — cross-correlation analysis
- `detect_peaks(signal, ...)` — peak detection with configurable thresholds
- `segment_signal(signal, ...)` — automatic segmentation

**Meta-tools:**
- `sub_llm(prompt, images=[...])` — recursive sub-agent call (multimodal)
- `help_tool(name)` — inspect any tool's full docstring

---

## Observability

Every agent run is traced end-to-end via [LangSmith](https://smith.langchain.com/):

- `biosignal_analyze` — top-level chain
- `root_llm` — each LLM call with full prompt/response
- `repl_exec` — each code execution with input/output
- `sub_llm` — recursive sub-agent calls
- `search_literature`, `validate_tool`, `register_tool` — skill acquisition steps

Set `LANGSMITH_API_KEY` and `LANGSMITH_PROJECT` in `.env` to enable.

---

## iOS Companion App

The `ios/AlwaysOnPT/` directory contains a Swift app that streams BLE EMG sensor and IMU data to the server via WebSocket. It includes:

- `BLEManager` — Bluetooth Low Energy sensor discovery and connection
- `IMUManager` — device motion data capture
- `StreamingService` — WebSocket client for `ws://localhost:8000/ws/session/{id}`
- `SensorFrame` — unified sensor data model

> **Note:** The WebSocket endpoint is not yet implemented in the server. The iOS app is designed for future real-time sensor streaming integration.

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| LLM | Anthropic Claude (Sonnet 4 / Opus 4) |
| Backend | Python, FastAPI, Uvicorn |
| Signal Processing | NumPy, SciPy, PyWavelets, scikit-learn |
| Data Format | WFDB (PhysioNet), custom EMG text files |
| Observability | LangSmith |
| Frontend | Vanilla JS, Canvas API, Server-Sent Events |
| Mobile | Swift, Combine, CoreBluetooth |

---

## Key Design Decisions

- **No model training.** The agent uses an LLM to reason over extracted features rather than training a classifier. This enables zero-shot generalization to new signal types.
- **exec() REPL over function calling.** The agent writes arbitrary Python, giving it full flexibility to compose tools, compute custom features, and handle edge cases — more powerful than a fixed tool-calling schema.
- **Task-prompt architecture.** The agent loop is completely generic. Domain specialization lives entirely in the system prompt template, making it trivial to add new biosignal domains.
- **Recursive sub-agents.** The root LLM can spawn sub-LLM calls with images (e.g., analyzing its own generated plots), enabling multimodal reasoning chains.
- **Skill acquisition.** The agent can write new tool functions, validate them on synthetic data, and register them for use in subsequent analyses — a form of runtime self-improvement.

---

# Companion iOS App

This repository has a companion iOS app that records EMG + IMU data and provides export \& share functionality. Use the app to collect sessions and produce JSON logs for analysis or upload.

\- App repository: https://github.com/dkman94/yc-always-on-pt.git

\- Key features:
  \- Real-time EMG \& IMU recording  
  \- Manual `Export Now` and `Export & Share` for JSON logs  
  \- Simulator retrieval instructions in the app `readme/README.md`

See the companion app for usage details and export instructions.

---

## License

This project was built for a hackathon. See individual dataset licenses (PhysioNet requires credentialed access for some datasets).
