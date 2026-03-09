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

echo "=== Downloading model checkpoints (emg2pose) ==="
if ./scripts/download_models.sh 2>/dev/null; then
  echo "emg2pose checkpoints ready"
else
  echo "Skipped or failed: emg2pose checkpoints (optional for pose)"
fi

echo ""
echo "=== Setup complete ==="
echo "Fatigue & effort: work immediately"
echo "Intent (gesture): auto-detected from submodule EMG-Gesture-Recognition-System"
echo "Pose (emg2pose): checkpoints downloaded; requires emg2pose env for inference"
echo "  See emg_model_server_setup.md for pose setup"
echo ""
echo "Verify: python scripts/verify_models.py"
