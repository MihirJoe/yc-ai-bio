"""Lightweight FastAPI server for EMG Model Server."""

import base64
import logging
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from emg_model_server.api import (
    run_emg_experts,
    run_single_expert,
    list_available_experts,
    get_capability_map,
)
from emg_model_server.types import EMGInput

logger = logging.getLogger(__name__)

app = FastAPI(
    title="EMG Model Server",
    description="Model-serving and normalization layer for EMG expert models",
    version="0.1.0",
)


class RunRequest(BaseModel):
    """Request body for POST /run."""

    task: str = Field(..., description="Task identifier")
    input_path: str | None = Field(default=None, description="Path to EMG file")
    inline_data: list[float] | None = Field(default=None, description="Inline EMG samples")
    inline_data_b64: str | None = Field(default=None, description="Base64-encoded numpy array")
    sample_rate: int = Field(default=1000, ge=1)
    channel_names: list[str] | None = None
    mode: str = Field(default="auto", description="auto | benchmark | live_lite")
    preferred_experts: list[str] | None = None
    optional_modalities: dict[str, Any] | None = None


class RunExpertRequest(BaseModel):
    """Request body for POST /run_expert."""

    expert_name: str = Field(..., description="Expert to run")
    input_path: str | None = None
    inline_data: list[float] | None = None
    inline_data_b64: str | None = None
    sample_rate: int = 1000
    mode: str = "auto"
    optional_modalities: dict[str, Any] | None = None


def _build_emg_input(req: RunRequest | RunExpertRequest) -> EMGInput:
    """Build EMGInput from request."""
    if hasattr(req, "input_path") and req.input_path:
        return EMGInput(
            file_path=req.input_path,
            sample_rate=req.sample_rate,
            channel_names=getattr(req, "channel_names", None),
        )
    if hasattr(req, "inline_data") and req.inline_data:
        return EMGInput(
            data=np.array(req.inline_data, dtype=np.float64),
            sample_rate=req.sample_rate,
        )
    if hasattr(req, "inline_data_b64") and req.inline_data_b64:
        arr = np.frombuffer(base64.b64decode(req.inline_data_b64), dtype=np.float64)
        return EMGInput(data=arr, sample_rate=req.sample_rate)
    raise HTTPException(
        status_code=400,
        detail="Provide input_path, inline_data, or inline_data_b64",
    )


@app.get("/health")
def health() -> dict:
    """Health check."""
    return {"status": "ok", "service": "emg-model-server"}


@app.get("/experts")
def experts() -> dict:
    """List available experts and capabilities."""
    names = list_available_experts()
    caps = get_capability_map()
    return {"experts": names, "capabilities": caps}


@app.post("/run")
def run(req: RunRequest) -> dict:
    """
    Run experts for a task.
    Accepts input_path or inline EMG data.
    """
    try:
        emg_input = _build_emg_input(req)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    resp = run_emg_experts(
        task=req.task,
        emg_input=emg_input,
        optional_modalities=req.optional_modalities,
        preferred_experts=req.preferred_experts,
        mode=req.mode,
    )
    return resp.model_dump_json_serializable()


@app.post("/run_expert")
def run_expert(req: RunExpertRequest) -> dict:
    """Run a single expert by name."""
    try:
        emg_input = _build_emg_input(req)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    pred = run_single_expert(
        name=req.expert_name,
        emg_input=emg_input,
        mode=req.mode,
        optional_modalities=req.optional_modalities,
    )
    if pred is None:
        raise HTTPException(status_code=404, detail=f"Expert not found or failed: {req.expert_name}")
    return pred.model_dump(mode="json")


def main() -> None:
    """Run server."""
    import os
    import uvicorn

    host = os.environ.get("SERVER_HOST", "0.0.0.0")
    port = int(os.environ.get("SERVER_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
