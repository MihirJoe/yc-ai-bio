# EMG Model Server Integration

All EMG expert code and models are **bundled in this repo**. Clone, run setup, and use.

## Quick Setup

```bash
# Clone with submodules (gesture model + emg2pose code)
git clone --recursive https://github.com/MihirJoe/yc-ai-bio.git
cd yc-ai-bio

# Install deps, init submodules, download checkpoints
./scripts/setup.sh

# Verify
python scripts/verify_models.py
```

No manual env vars needed for fatigue, effort, or gesture — paths are auto-detected.

## What's Included

| Component | In Repo | Works OOTB |
|-----------|---------|------------|
| **emg_model_server** | Yes (vendored) | Yes |
| **Fatigue adapter** | Yes | Yes (feature-based) |
| **Effort adapter** | Yes | Yes (feature-based) |
| **Gesture model** (intent) | Yes (submodule) | Yes (auto-detected) |
| **emg2pose** (pose) | Yes (submodule) | Needs emg2pose env + checkpoints |

## Usage

```python
from alwaysonpt.emg_model_bridge import (
    load_emg_input,
    get_fatigue,
    get_effort,
    get_pose,
    get_intent,
    get_all,
)

emg_input = load_emg_input(data=emg_array, sample_rate=1000)

# Always work (no model files)
fatigue_out = get_fatigue(emg_input)
effort_out = get_effort(emg_input)

# Works when submodule + pipeline.joblib present
intent_out = get_intent(emg_input)

# Needs emg2pose env + checkpoint
pose_out = get_pose(emg_input)  # status=unavailable until pose setup
```

## Pose (emg2pose) Setup

Pose estimation requires emg2pose's conda environment and the checkpoint (downloaded by setup.sh):

1. `conda env create -f emg2pose/environment.yml`
2. `conda activate emg2pose`
3. `pip install -e ./emg2pose`
4. `pip install -e ./emg2pose/emg2pose/UmeTrack`
5. Checkpoint is at `models/emg2pose_model_checkpoints/tracking_vemg2pose.ckpt` (after `./scripts/download_models.sh`)

Paths are auto-detected when running from this repo; no env vars needed if structure is unchanged.
