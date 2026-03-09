"""
Live data loader for EMG JSON recordings from iOS sensor app.

Parses the JSON format: {emgSignal: [...], samplingRate, duration, recordedAt}
into BioSignalRecord. IMU data is ignored.
"""

import json
from pathlib import Path

import numpy as np

from alwaysonpt.datasets.base import BioSignalRecord


def load_live_recording(path: str | Path) -> BioSignalRecord:
    """Load a live sensor JSON recording into a BioSignalRecord (EMG only)."""
    path = Path(path)
    data = json.loads(path.read_text())

    emg = np.array(data.get("emgSignal", []), dtype=np.float64)
    signals = {"emg": emg}

    fs = data.get("samplingRate", 10)
    duration = data.get("duration", len(emg) / max(fs, 1))

    return BioSignalRecord(
        record_id=path.stem,
        domain="emg",
        signals=signals,
        fs=fs,
        duration_s=duration,
        metadata={
            "recordedAt": data.get("recordedAt"),
            "source": "live_sensor",
            "n_emg_samples": len(emg),
        },
    )


def list_live_recordings(directory: str | Path) -> list[dict]:
    """List available live recordings in a directory."""
    directory = Path(directory)
    recordings = []

    if not directory.exists():
        return recordings

    for jf in sorted(directory.glob("*.json")):
        try:
            data = json.loads(jf.read_text())
            recordings.append({
                "id": jf.stem,
                "path": str(jf),
                "duration": data.get("duration", 0),
                "sampling_rate": data.get("samplingRate", 10),
                "n_emg": len(data.get("emgSignal", [])),
                "has_emg": bool(data.get("emgSignal")),
            })
        except Exception:
            continue

    return recordings
