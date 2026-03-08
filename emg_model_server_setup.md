# EMG Model Server Integration

All EMG expert code is **bundled in this repo**. Clone, run setup, and use.

## Quick Setup

```bash
# Clone (includes gesture model via submodule)
git clone --recursive https://github.com/MihirJoe/yc-ai-bio.git
cd yc-ai-bio

# Install and init submodules
./scripts/setup.sh

# Optional: enable gesture (intent) model
export EMG_GESTURE_MODEL_DIR="$PWD/EMG-Gesture-Recognition-System/models-example/gesture_cls/1.0.0_20250821T094534Z"
```

## What's Included

| Component | In Repo | Works OOTB |
|-----------|---------|------------|
| **emg_model_server** | Yes (vendored) | Yes |
| **Fatigue adapter** | Yes | Yes (feature-based) |
| **Effort adapter** | Yes | Yes (feature-based) |
| **Gesture model** (intent) | Yes (submodule) | Yes after `EMG_GESTURE_MODEL_DIR` |
| **emg2pose** (pose) | No | Optional, manual setup |

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

# Work if EMG_GESTURE_MODEL_DIR set
intent_out = get_intent(emg_input)

# Optional: emg2pose (16-ch, 2000 Hz, checkpoint required)
pose_out = get_pose(emg_input)  # status=unavailable without emg2pose
```

## Optional: Pose (emg2pose)

Pose estimation requires Meta's emg2pose package and checkpoint:

1. Clone emg2pose: `git clone https://github.com/facebookresearch/emg2pose`
2. `pip install -e ./emg2pose`
3. Download checkpoint to `~/emg2pose_model_checkpoints/tracking_vemg2pose.ckpt` (see emg2pose README)
4. `export EMG2POSE_PROJECT_ROOT=/path/to/emg2pose`
