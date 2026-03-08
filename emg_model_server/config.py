"""Configuration loading and management for EMG Model Server."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class PreprocessingConfig(BaseModel):
    """Preprocessing pipeline configuration."""

    target_sample_rate: int = 1000
    bandpass_low: int = 20
    bandpass_high: int = 500
    notch_freq: int = 50
    window_size_ms: int = 200
    overlap_ratio: float = 0.5
    normalize: bool = True


class ServerConfig(BaseModel):
    """Configuration for the EMG Model Server."""

    mode: str = Field(default="auto", description="auto | benchmark | live_lite")
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    task_expert_mapping: dict[str, list[str]] = Field(default_factory=dict)
    single_channel_compatible: list[str] = Field(default_factory=list)
    benchmark_only: list[str] = Field(default_factory=list)


def load_config(config_path: str | Path | None = None) -> ServerConfig:
    """
    Load configuration from YAML file or use defaults.

    Args:
        config_path: Path to YAML config. If None, uses default.yaml from configs/.

    Returns:
        ServerConfig instance
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "configs" / "default.yaml"

    config_path = Path(config_path)
    if not config_path.exists():
        return ServerConfig()

    with open(config_path) as f:
        data = yaml.safe_load(f) or {}

    return ServerConfig(**data)


def get_mode_from_env() -> str:
    """Get operating mode from environment variable."""
    import os
    return os.environ.get("EMG_MODE", "auto").lower()
