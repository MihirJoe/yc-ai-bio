#!/usr/bin/env bash
# Setup yc-ai-bio for EMG model integration.
# Run from repo root: ./scripts/setup.sh

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== Initializing submodules ==="
git submodule update --init --recursive

echo "=== Installing Python package (editable) ==="
pip install -e .

GESTURE_MODEL_DIR="$REPO_ROOT/EMG-Gesture-Recognition-System/models-example/gesture_cls/1.0.0_20250821T094534Z"
if [ -d "$GESTURE_MODEL_DIR" ] && [ -f "$GESTURE_MODEL_DIR/pipeline.joblib" ]; then
  echo ""
  echo "=== Gesture model ready ==="
  echo "Set in your environment or .env:"
  echo "  export EMG_GESTURE_MODEL_DIR=$GESTURE_MODEL_DIR"
else
  echo ""
  echo "WARNING: Gesture model not found at $GESTURE_MODEL_DIR"
  echo "Run: git submodule update --init EMG-Gesture-Recognition-System"
fi

echo ""
echo "=== Setup complete ==="
echo "Fatigue & effort: work immediately (no model files)"
echo "Intent (gesture): set EMG_GESTURE_MODEL_DIR as above"
echo "Pose (emg2pose): optional, see emg_model_server_setup.md"
