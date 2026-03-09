# Live Sensor Demo — Data Setup

This folder contains the data download/generation script for the AlwaysOnPT static demo.

The demo supports three data sources:

| Source | Channels | Format | Status |
|--------|----------|--------|--------|
| **Knee EMG** (Zhang 2017) | 2 (EMG + goniometer) | `.txt`, 1kHz | Available |
| **Live Sensor** (iOS app) | EMG + IMU (pitch/roll/yaw) | `.json`, 10Hz EMG | Available |
| **16-Channel EMG** | 16 muscle channels | `.json`, 1kHz | Synthetic (real data TBD) |

## Setup

```bash
# Check what data you already have
./live_sensor_demo/download_data.sh --check

# Download/generate all demo data
./live_sensor_demo/download_data.sh --all

# Or selectively
./live_sensor_demo/download_data.sh --multichannel   # generates synthetic 16-ch data
./live_sensor_demo/download_data.sh --live-sensor    # checks for live recordings
```

## Run the demo

```bash
python -m alwaysonpt.server
open http://localhost:8000
```

The demo UI lets you switch between data sources, visualize signals, and run the AI agent for analysis.

## Data layout

```
data/
├── knee_emg/S1File/Data/       # 2-channel EMG (14 subjects × 3 exercises)
│   ├── 1standing.txt
│   ├── 1sitting.txt
│   └── ...
├── live_data_sample/           # iOS sensor recordings
│   └── Test 1/
│       ├── emg_recording_*.json
│       └── ...
└── multichannel_emg/           # 16-channel EMG
    ├── multichannel_walking.json
    ├── multichannel_squat.json
    └── multichannel_stair_climb.json
```

## Adding real 16-channel data

When real 16-channel recordings become available, place them in `data/multichannel_emg/` as JSON with this schema:

```json
{
  "channels": {
    "vastus_medialis": [0.01, -0.02, ...],
    "vastus_lateralis": [0.03, 0.01, ...],
    ...
  },
  "channelNames": ["vastus_medialis", "vastus_lateralis", ...],
  "samplingRate": 1000,
  "duration": 10.0,
  "scenario": "walking",
  "metadata": { "units": "mV" }
}
```

The demo will automatically detect and load them.
