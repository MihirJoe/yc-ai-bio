"""
REST API for BioSignalAgent analysis + demo GUI.

Three tabs in one dashboard:
  - Knee EMG: RLM agent on Zhang 2017 dataset
  - Live Sensor: gait analysis pipeline (analyze_activation → Claude)
  - 16-Channel: adapter-routed analysis
"""

import os
import json
import asyncio
import queue
import threading
import numpy as np
from pathlib import Path

_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional

from alwaysonpt.task_prompts import list_task_types
from alwaysonpt.data_loader import (
    parse_emg_file, highpass_filter, segment_repetitions, _clean_goniometer,
)

STATIC_DIR = Path(__file__).parent / "static"
PLOTS_DIR = Path(__file__).parent / "static" / "plots"
DATA_DIR = Path(__file__).parent.parent / "data" / "knee_emg" / "S1File" / "Data"
LIVE_DIR = Path(__file__).parent.parent / "data" / "live_data_sample"
MULTI_DIR = Path(__file__).parent.parent / "data" / "multichannel_emg"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="AlwaysOnPT Analysis API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/plots", StaticFiles(directory=str(PLOTS_DIR)), name="plots")


# --- Data source overview ---

@app.get("/api/sources")
async def list_sources():
    """List available data sources for the demo."""
    sources = []

    if DATA_DIR.exists():
        count = len(list(DATA_DIR.glob("*.txt")))
        if count > 0:
            sources.append({
                "id": "knee_emg", "name": "Knee EMG (Zhang 2017)",
                "channels": 2, "channel_names": ["EMG (VM)", "Goniometer"],
                "fs": 1000, "n_files": count,
                "description": "14 subjects × 3 exercises, 2-channel (EMG + goniometer), 1kHz",
            })

    if LIVE_DIR.exists():
        json_files = list(LIVE_DIR.glob("*.json"))
        for d in LIVE_DIR.iterdir():
            if d.is_dir():
                json_files.extend(d.glob("*.json"))
        if json_files:
            sources.append({
                "id": "live_sensor", "name": "Live Sensor (iOS)",
                "channels": 1, "channel_names": ["EMG"],
                "fs": 10, "n_files": len(json_files),
                "description": "EMG recordings from iOS sensor app",
            })

    if MULTI_DIR.exists():
        json_files = list(MULTI_DIR.glob("*.json"))
        if json_files:
            first = json.loads(json_files[0].read_text())
            ch_names = first.get("channelNames", [])
            sources.append({
                "id": "multichannel", "name": "16-Channel EMG",
                "channels": len(ch_names), "channel_names": ch_names,
                "fs": first.get("samplingRate", 1000), "n_files": len(json_files),
                "description": f"{len(ch_names)}-channel EMG array, {first.get('samplingRate', 1000)}Hz",
            })

    return {"sources": sources}


# --- Knee EMG endpoints ---

@app.get("/api/subjects")
async def list_subjects():
    subjects = {}
    for f in sorted(DATA_DIR.glob("*.txt")):
        import re
        m = re.match(r'(\d+)(standing|sitting|gait)\.txt', f.name)
        if m:
            sid = int(m.group(1))
            exercise = m.group(2)
            subjects.setdefault(sid, []).append(exercise)
    return {"subjects": {str(k): sorted(v) for k, v in sorted(subjects.items())}}


@app.get("/api/load/{subject_id}/{exercise}")
async def load_recording(subject_id: int, exercise: str):
    filename = f"{subject_id}{exercise}.txt"
    filepath = DATA_DIR / filename
    if not filepath.exists():
        raise HTTPException(404, f"File not found: {filename}")

    rec = parse_emg_file(str(filepath))
    emg_filtered = highpass_filter(rec.emg_raw)
    gonio_clean = _clean_goniometer(rec.gonio)
    segments_raw = segment_repetitions(emg_filtered, gonio_clean, exercise)

    segments = []
    for i, seg in enumerate(segments_raw):
        segments.append({
            "index": i,
            "motion_class": seg["motion_class"],
            "start_idx": int(seg["start_idx"]),
            "end_idx": int(seg["end_idx"]),
            "n_samples": len(seg["emg"]),
            "duration_s": round(len(seg["emg"]) / 1000.0, 3),
        })

    step = max(1, len(emg_filtered) // 10000)
    emg_ds = emg_filtered[::step].tolist()
    gonio_ds = gonio_clean[::step].tolist()

    return {
        "subject_id": subject_id, "exercise": exercise, "fs": 1000,
        "n_samples": len(emg_filtered),
        "duration_s": round(len(emg_filtered) / 1000.0, 2),
        "emg_downsampled": emg_ds, "gonio_downsampled": gonio_ds,
        "downsample_step": step, "segments": segments,
        "n_segments": len(segments),
    }


@app.get("/api/load/{subject_id}/{exercise}/segment/{seg_idx}")
async def load_segment(subject_id: int, exercise: str, seg_idx: int):
    filename = f"{subject_id}{exercise}.txt"
    filepath = DATA_DIR / filename
    if not filepath.exists():
        raise HTTPException(404, f"File not found: {filename}")

    rec = parse_emg_file(str(filepath))
    emg_filtered = highpass_filter(rec.emg_raw)
    gonio_clean = _clean_goniometer(rec.gonio)
    segments_raw = segment_repetitions(emg_filtered, gonio_clean, exercise)

    if seg_idx < 0 or seg_idx >= len(segments_raw):
        raise HTTPException(404, f"Segment {seg_idx} not found (total: {len(segments_raw)})")

    seg = segments_raw[seg_idx]
    return {
        "subject_id": subject_id, "exercise": exercise,
        "segment_index": seg_idx, "motion_class": seg["motion_class"],
        "emg": seg["emg"].tolist(), "gonio": seg["gonio"].tolist(),
        "fs": 1000, "n_samples": len(seg["emg"]),
        "duration_s": round(len(seg["emg"]) / 1000.0, 3),
    }


# --- Live sensor endpoints ---

@app.get("/api/live/recordings")
async def list_live_recordings():
    if not LIVE_DIR.exists():
        return {"recordings": []}

    recordings = []

    def _add_json(jf: Path, session: str):
        try:
            data = json.loads(jf.read_text())
            recordings.append({
                "id": f"{session}/{jf.stem}" if session else jf.stem,
                "session": session or "default",
                "filename": jf.name,
                "duration": data.get("duration", 0),
                "sampling_rate": data.get("samplingRate", 10),
                "has_emg": "emgSignal" in data,
                "n_emg_samples": len(data.get("emgSignal", [])),
            })
        except Exception:
            pass

    for item in sorted(LIVE_DIR.iterdir()):
        if item.is_file() and item.suffix == ".json":
            _add_json(item, "")
        elif item.is_dir():
            for jf in sorted(item.glob("*.json")):
                _add_json(jf, item.name)

    return {"recordings": recordings}


def _build_live_response(filepath: Path, record_id: str, session: str):
    data = json.loads(filepath.read_text())

    emg_raw = data.get("emgSignal", [])
    signals = {"emg": [float(v) for v in emg_raw]} if emg_raw else {}

    fs = data.get("samplingRate", 10)

    return {
        "source": "live_sensor",
        "record_id": record_id,
        "session": session,
        "channels": list(signals.keys()),
        "signals": signals,
        "fs": fs,
        "n_samples": len(emg_raw),
        "duration_s": data.get("duration", len(emg_raw) / max(fs, 1)),
        "recorded_at": data.get("recordedAt"),
        "metadata": {
            "session": session,
            "sampling_rate": fs,
            "has_emg": bool(emg_raw),
        },
    }


@app.get("/api/live/load/{recording}")
async def load_live_recording(recording: str):
    filepath = LIVE_DIR / f"{recording}.json"
    if not filepath.exists():
        raise HTTPException(404, f"Recording not found: {recording}")
    return _build_live_response(filepath, recording, "default")


@app.get("/api/live/load/{session}/{recording}")
async def load_live_recording_in_session(session: str, recording: str):
    filepath = LIVE_DIR / session / f"{recording}.json"
    if not filepath.exists():
        raise HTTPException(404, f"Recording not found: {session}/{recording}")
    return _build_live_response(filepath, f"{session}/{recording}", session)


# --- Multichannel EMG endpoints ---

@app.get("/api/multichannel/recordings")
async def list_multichannel_recordings():
    if not MULTI_DIR.exists():
        return {"recordings": []}

    recordings = []
    for jf in sorted(MULTI_DIR.glob("*.json")):
        try:
            data = json.loads(jf.read_text())
            ch_names = data.get("channelNames", list(data.get("channels", {}).keys()))
            recordings.append({
                "id": jf.stem,
                "filename": jf.name,
                "scenario": data.get("scenario", jf.stem),
                "n_channels": len(ch_names),
                "channel_names": ch_names,
                "duration": data.get("duration", 0),
                "sampling_rate": data.get("samplingRate", 1000),
                "source": data.get("source", "unknown"),
            })
        except Exception:
            continue

    return {"recordings": recordings}


@app.get("/api/multichannel/load/{recording_id}")
async def load_multichannel_recording(recording_id: str):
    filepath = MULTI_DIR / f"{recording_id}.json"
    if not filepath.exists():
        raise HTTPException(404, f"Recording not found: {recording_id}")

    data = json.loads(filepath.read_text())
    channels = data.get("channels", {})
    ch_names = data.get("channelNames", list(channels.keys()))
    fs = data.get("samplingRate", 1000)

    signals = {}
    for name in ch_names:
        raw = channels.get(name, [])
        step = max(1, len(raw) // 2000)
        signals[name] = [float(raw[i]) for i in range(0, len(raw), step)]

    first_sig = channels.get(ch_names[0], []) if ch_names else []

    return {
        "source": "multichannel",
        "record_id": recording_id,
        "scenario": data.get("scenario", recording_id),
        "channels": ch_names,
        "signals": signals,
        "signals_full": {name: channels[name] for name in ch_names},
        "fs": fs,
        "n_samples": len(first_sig),
        "duration_s": data.get("duration", len(first_sig) / max(fs, 1)),
        "downsample_step": max(1, len(first_sig) // 2000),
        "metadata": data.get("metadata", {}),
    }


# --- Agent analysis (SSE streaming) ---

class AnalyzeRequest(BaseModel):
    task_type: str = "emg_classification"
    question: Optional[str] = None
    domain: str = "emg"
    fs: int = 1000
    signals: dict[str, list[float]] = {}
    metadata: dict = {}
    record_id: str = ""


def _step_to_dict(step) -> dict:
    return {
        'type': step.step_type.value,
        'content': step.content,
        'evidence': [str(e) for e in step.evidence],
        'timestamp': step.timestamp,
        'duration_ms': step.duration_ms,
    }


@app.post("/api/analyze/stream")
async def analyze_stream(req: AnalyzeRequest):
    """Run the RLM agent and stream steps via SSE as they happen."""
    from alwaysonpt.biosignal_agent import BioSignalAgent
    from alwaysonpt.datasets.base import BioSignalRecord

    if not req.signals:
        raise HTTPException(400, "No signals provided")

    np_signals = {k: np.array(v, dtype=np.float64) for k, v in req.signals.items()}
    first_sig = next(iter(np_signals.values()))

    record = BioSignalRecord(
        record_id=req.record_id or "demo_session",
        domain=req.domain,
        signals=np_signals,
        fs=req.fs,
        duration_s=len(first_sig) / req.fs,
        metadata=req.metadata,
    )

    step_queue: queue.Queue = queue.Queue()

    def on_step(step, trace):
        step_queue.put(('step', step))

    def run_agent():
        try:
            agent = BioSignalAgent(verbose=True)
            trace = agent.analyze(
                record,
                task_type=req.task_type,
                question=req.question,
                record_id=req.record_id or "demo_session",
                on_step=on_step,
            )
            report = trace.to_dict()
            plots = []
            for p in trace.plots_generated:
                pp = Path(p)
                if pp.exists():
                    plots.append(f"/plots/{pp.name}")
            report["plot_urls"] = plots
            step_queue.put(('done', report))
        except Exception as e:
            step_queue.put(('error', str(e)))

    thread = threading.Thread(target=run_agent, daemon=True)
    thread.start()

    async def event_stream():
        while True:
            try:
                msg = await asyncio.to_thread(step_queue.get, timeout=180)
            except Exception:
                yield f"data: {json.dumps({'event': 'error', 'message': 'timeout'})}\n\n"
                break

            event_type, payload = msg

            if event_type == 'step':
                yield f"data: {json.dumps({'event': 'step', 'step': _step_to_dict(payload)})}\n\n"
            elif event_type == 'done':
                yield f"data: {json.dumps({'event': 'done', 'result': payload})}\n\n"
                break
            elif event_type == 'error':
                yield f"data: {json.dumps({'event': 'error', 'message': payload})}\n\n"
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest):
    """Run the full RLM agent (non-streaming fallback)."""
    from alwaysonpt.biosignal_agent import BioSignalAgent
    from alwaysonpt.datasets.base import BioSignalRecord

    if not req.signals:
        raise HTTPException(400, "No signals provided")

    np_signals = {k: np.array(v, dtype=np.float64) for k, v in req.signals.items()}
    first_sig = next(iter(np_signals.values()))

    record = BioSignalRecord(
        record_id=req.record_id or "demo_session",
        domain=req.domain,
        signals=np_signals,
        fs=req.fs,
        duration_s=len(first_sig) / req.fs,
        metadata=req.metadata,
    )

    agent = BioSignalAgent(verbose=True)
    trace = await asyncio.to_thread(
        agent.analyze, record,
        task_type=req.task_type, question=req.question,
        record_id=req.record_id or "demo_session",
    )

    report = trace.to_dict()
    plots = []
    for p in trace.plots_generated:
        pp = Path(p)
        if pp.exists():
            plots.append(f"/plots/{pp.name}")
    report["plot_urls"] = plots
    return report


@app.get("/api/task-types")
async def get_task_types():
    return {"task_types": [
        {"id": tid, "name": name} for tid, name in list_task_types()
    ]}


@app.get("/api/config")
async def get_config():
    return {"status": "ok"}


# --- Routed adapter analysis (static mode) ---

class RoutedAnalyzeRequest(BaseModel):
    query: str
    recording_id: str = ""
    task_type: str = "adapter_analysis"


@app.post("/api/analyze/routed")
async def analyze_routed(req: RoutedAnalyzeRequest):
    """Route physician query to adapters, generate plots, then run agent analysis via SSE."""
    from alwaysonpt.biosignal_agent import BioSignalAgent
    from alwaysonpt.datasets.base import BioSignalRecord
    from alwaysonpt.adapter_router import run_routed_analysis
    from alwaysonpt.adapter_plots import generate_adapter_plots

    recording_id = req.recording_id
    filepath = MULTI_DIR / f"{recording_id}.json" if recording_id else None

    if filepath and filepath.exists():
        data = json.loads(filepath.read_text())
        channels = data.get("channels", {})
        ch_names = data.get("channelNames", list(channels.keys()))
        fs = data.get("samplingRate", 1000)
        emg_array = np.column_stack([np.array(channels[c], dtype=np.float64) for c in ch_names])
    else:
        raise HTTPException(400, "No multichannel recording specified or file not found")

    routed = run_routed_analysis(req.query, emg_array, fs, ch_names)
    adapter_results = routed["adapter_results"]

    duration_s = len(emg_array) / max(fs, 1)
    plot_paths = generate_adapter_plots(adapter_results, total_duration_s=duration_s)
    plot_urls = {k: f"/plots/{Path(v).name}" for k, v in plot_paths.items()}

    first_ch = np.array(channels[ch_names[0]], dtype=np.float64) if ch_names else np.array([])
    np_signals = {name: np.array(channels[name], dtype=np.float64) for name in ch_names}

    record = BioSignalRecord(
        record_id=recording_id or "multichannel_demo",
        domain="multi_emg",
        signals=np_signals,
        fs=fs,
        duration_s=duration_s,
        metadata={
            "adapter_results": adapter_results,
            "routed_adapters": routed["routed_adapters"],
            "channel_names": ch_names,
        },
    )

    step_queue: queue.Queue = queue.Queue()

    def on_step(step, trace):
        step_queue.put(("step", step))

    def run_agent():
        try:
            agent = BioSignalAgent(verbose=True)
            trace = agent.analyze(
                record,
                task_type=req.task_type,
                question=req.query,
                record_id=recording_id,
                on_step=on_step,
            )
            report = trace.to_dict()
            report["adapter_results"] = adapter_results
            report["routed_adapters"] = routed["routed_adapters"]
            report["plot_urls"] = plot_urls
            step_queue.put(("done", report))
        except Exception as e:
            step_queue.put(("error", str(e)))

    thread = threading.Thread(target=run_agent, daemon=True)
    thread.start()

    async def event_stream():
        adapter_event = {
            "event": "adapters_done",
            "routed_adapters": routed["routed_adapters"],
        }
        yield f"data: {json.dumps(adapter_event, default=str)}\n\n"

        while True:
            try:
                msg = await asyncio.to_thread(step_queue.get, timeout=180)
            except Exception:
                yield f"data: {json.dumps({'event': 'error', 'message': 'timeout'})}\n\n"
                break
            event_type, payload = msg
            if event_type == "step":
                yield f"data: {json.dumps({'event': 'step', 'step': _step_to_dict(payload)})}\n\n"
            elif event_type == "done":
                yield f"data: {json.dumps({'event': 'done', 'result': payload}, default=str)}\n\n"
                break
            elif event_type == "error":
                yield f"data: {json.dumps({'event': 'error', 'message': payload})}\n\n"
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# --- Live sensor analysis ---

class LiveAnalyzeRequest(BaseModel):
    recording_id: str
    task_type: str = "activity_classification"
    question: Optional[str] = None


@app.post("/api/analyze/live")
async def analyze_live(req: LiveAnalyzeRequest):
    """Load a live JSON recording and run activity classification via SSE."""
    from alwaysonpt.biosignal_agent import BioSignalAgent
    from alwaysonpt.data_loader_live import load_live_recording

    filepath = LIVE_DIR / f"{req.recording_id}.json"
    if not filepath.exists():
        for sub in LIVE_DIR.iterdir():
            if sub.is_dir():
                candidate = sub / f"{req.recording_id}.json"
                if candidate.exists():
                    filepath = candidate
                    break
        if not filepath.exists():
            raise HTTPException(404, f"Recording not found: {req.recording_id}")

    record = load_live_recording(filepath)

    step_queue: queue.Queue = queue.Queue()

    def on_step(step, trace):
        step_queue.put(("step", step))

    def run_agent():
        try:
            agent = BioSignalAgent(verbose=True)
            trace = agent.analyze(
                record,
                task_type=req.task_type,
                question=req.question,
                record_id=req.recording_id,
                on_step=on_step,
            )
            report = trace.to_dict()
            step_queue.put(("done", report))
        except Exception as e:
            step_queue.put(("error", str(e)))

    thread = threading.Thread(target=run_agent, daemon=True)
    thread.start()

    async def event_stream():
        while True:
            try:
                msg = await asyncio.to_thread(step_queue.get, timeout=180)
            except Exception:
                yield f"data: {json.dumps({'event': 'error', 'message': 'timeout'})}\n\n"
                break
            event_type, payload = msg
            if event_type == "step":
                yield f"data: {json.dumps({'event': 'step', 'step': _step_to_dict(payload)})}\n\n"
            elif event_type == "done":
                yield f"data: {json.dumps({'event': 'done', 'result': payload}, default=str)}\n\n"
                break
            elif event_type == "error":
                yield f"data: {json.dumps({'event': 'error', 'message': payload})}\n\n"
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# --- Live gait analysis (two-stage: deterministic + Claude) ---

class GaitSignalRequest(BaseModel):
    emg: list[float]
    fs: float = 1000.0
    duration_s: float = 0
    record_id: str = "static_gait"


@app.post("/api/analyze/gait")
async def analyze_gait_from_signals(req: GaitSignalRequest):
    """
    Gait pipeline for raw EMG signals (used by Knee EMG tab's gait task).
    Same two-stage flow as /api/live/analyze/gait but accepts signal arrays directly.
    """
    from alwaysonpt.gait_analyzer import (
        analyze_activation, generate_gait_plot, stream_gait_assessment,
    )

    if not req.emg or len(req.emg) < 10:
        raise HTTPException(400, "EMG signal too short")

    duration = req.duration_s if req.duration_s > 0 else len(req.emg) / max(req.fs, 1)

    activation = analyze_activation(req.emg, duration)
    if "error" in activation:
        raise HTTPException(400, activation["error"])

    result = {
        "recording_info": {
            "source_file": req.record_id,
            "duration_s": duration,
            "stated_sampling_rate": req.fs,
            "emg_samples": len(req.emg),
        },
        "muscle_activation": activation,
    }

    plot_url = generate_gait_plot(req.emg, result, req.record_id)

    text_queue: queue.Queue = queue.Queue()

    def run_assessment():
        try:
            for chunk in stream_gait_assessment(result):
                text_queue.put(("text", chunk))
            text_queue.put(("done", None))
        except Exception as e:
            text_queue.put(("error", str(e)))

    thread = threading.Thread(target=run_assessment, daemon=True)
    thread.start()

    async def event_stream():
        yield f"data: {json.dumps({'event': 'metrics', 'result': result, 'plot_url': plot_url}, default=str)}\n\n"

        while True:
            try:
                msg = await asyncio.to_thread(text_queue.get, timeout=120)
            except Exception:
                yield f"data: {json.dumps({'event': 'error', 'message': 'timeout'})}\n\n"
                break

            event_type, payload = msg
            if event_type == "text":
                yield f"data: {json.dumps({'event': 'text', 'chunk': payload})}\n\n"
            elif event_type == "done":
                yield f"data: {json.dumps({'event': 'done'})}\n\n"
                break
            elif event_type == "error":
                yield f"data: {json.dumps({'event': 'error', 'message': payload})}\n\n"
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream")


class LiveGaitRequest(BaseModel):
    recording_id: str


@app.post("/api/live/analyze/gait")
async def analyze_live_gait(req: LiveGaitRequest):
    """
    Two-stage live gait pipeline (--live mode):
      1. Run analyze_activation() for deterministic metrics
      2. Stream Claude gait assessment via SSE
    """
    from alwaysonpt.gait_analyzer import (
        process_recording, stream_gait_assessment, generate_gait_plot,
    )

    filepath = _resolve_live_path(req.recording_id)
    if filepath is None:
        raise HTTPException(404, f"Recording not found: {req.recording_id}")

    data = json.loads(filepath.read_text())
    emg_signal = data.get("emgSignal", [])
    result = process_recording(data)

    if "error" in result:
        raise HTTPException(400, result["error"])

    plot_url = generate_gait_plot(emg_signal, result, req.recording_id)

    text_queue: queue.Queue = queue.Queue()

    def run_assessment():
        try:
            for chunk in stream_gait_assessment(result):
                text_queue.put(("text", chunk))
            text_queue.put(("done", None))
        except Exception as e:
            text_queue.put(("error", str(e)))

    thread = threading.Thread(target=run_assessment, daemon=True)
    thread.start()

    async def event_stream():
        metrics_event = {
            "event": "metrics",
            "result": result,
            "plot_url": plot_url,
        }
        yield f"data: {json.dumps(metrics_event, default=str)}\n\n"

        while True:
            try:
                msg = await asyncio.to_thread(text_queue.get, timeout=120)
            except Exception:
                yield f"data: {json.dumps({'event': 'error', 'message': 'timeout'})}\n\n"
                break

            event_type, payload = msg
            if event_type == "text":
                yield f"data: {json.dumps({'event': 'text', 'chunk': payload})}\n\n"
            elif event_type == "done":
                yield f"data: {json.dumps({'event': 'done'})}\n\n"
                break
            elif event_type == "error":
                yield f"data: {json.dumps({'event': 'error', 'message': payload})}\n\n"
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _resolve_live_path(recording_id: str) -> Path | None:
    """Find a live recording JSON file by its ID."""
    filepath = LIVE_DIR / f"{recording_id}.json"
    if filepath.exists():
        return filepath
    parts = recording_id.split("/", 1)
    if len(parts) == 2:
        filepath = LIVE_DIR / parts[0] / f"{parts[1]}.json"
        if filepath.exists():
            return filepath
    for sub in LIVE_DIR.iterdir():
        if sub.is_dir():
            candidate = sub / f"{recording_id}.json"
            if candidate.exists():
                return candidate
    return None


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def root():
    return FileResponse(str(STATIC_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
