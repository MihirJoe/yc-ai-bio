"""
REST API for BioSignalAgent analysis + demo GUI.

Loads real EMG data from the knee_emg dataset (S1File),
segments it, runs the RLM agent, and compares against ground truth.
Supports SSE streaming of agent steps in real-time.
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
PLOTS_DIR = Path(__file__).parent / "output" / "plots"
DATA_DIR = Path(__file__).parent.parent / "data" / "knee_emg" / "S1File" / "Data"
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


# --- Data loading endpoints ---

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


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def root():
    return FileResponse(str(STATIC_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
