#!/usr/bin/env bash
# Download model checkpoints. Run from repo root: ./scripts/download_models.sh

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODELS_DIR="$REPO_ROOT/models"
EMG2POSE_DIR="$MODELS_DIR/emg2pose_model_checkpoints"
CHECKPOINT_URL="https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_model_checkpoints.tar.gz"

echo "=== Downloading emg2pose checkpoints ==="
mkdir -p "$MODELS_DIR"
if [ -f "$EMG2POSE_DIR/tracking_vemg2pose.ckpt" ]; then
  echo "emg2pose checkpoint already exists at $EMG2POSE_DIR"
  exit 0
fi

TARBALL="$MODELS_DIR/emg2pose_model_checkpoints.tar.gz"
if [ ! -f "$TARBALL" ]; then
  echo "Downloading from $CHECKPOINT_URL ..."
  curl -L -o "$TARBALL" "$CHECKPOINT_URL"
fi

echo "Extracting to $MODELS_DIR ..."
tar -xzf "$TARBALL" -C "$MODELS_DIR"
rm -f "$TARBALL"

if [ -f "$EMG2POSE_DIR/tracking_vemg2pose.ckpt" ]; then
  echo "emg2pose checkpoint ready: $EMG2POSE_DIR/tracking_vemg2pose.ckpt"
else
  echo "WARNING: tracking_vemg2pose.ckpt not found after extract"
  exit 1
fi
