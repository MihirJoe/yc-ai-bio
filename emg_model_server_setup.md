# EMG Model Server integration

The `alwaysonpt.emg_model_bridge` module provides access to EMG expert models (fatigue, effort, pose, intent).

## Setup

1. **Install emg-model-server** (from sibling repo or clone):

   ```bash
   # Option A: Local path (sibling project)
   pip install -e /path/to/emg-model-server

   # Option B: Git submodule
   git submodule add <emg-model-server-repo-url> emg-model-server
   pip install -e ./emg-model-server
   ```

2. **Environment variables** (for pose + intent):

   ```bash
   export EMG2POSE_PROJECT_ROOT=/path/to/emg2pose/project
   export EMG_GESTURE_MODEL_DIR=/path/to/EMG-Gesture-Recognition-System/models-example/gesture_cls/...
   ```

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

# Individual adapters
fatigue_out = get_fatigue(emg_input)   # {"adapter": "fatigue", "fatigue_score": 0.63, ...}
effort_out = get_effort(emg_input)     # {"adapter": "effort", "effort_score": 0.17, ...}
pose_out = get_pose(emg_input)         # {"adapter": "pose", "pose_features": [...]}  # 16-ch
intent_out = get_intent(emg_input)     # {"adapter": "intent", "intent_labels": [...]} # 8+ ch

# Or run all at once
all_out = get_all(emg_input)
```
