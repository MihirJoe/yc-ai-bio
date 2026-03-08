#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATASETS_DIR="$PROJECT_ROOT/datasets"

DATASETS=(
    "gaitpdb|https://physionet.org/content/gaitpdb/1.0.0/|Gait in Parkinson's Disease|~288 MB"
    "gaitndd|https://physionet.org/content/gaitndd/1.0.0/|Gait in Neurodegenerative Disease|~18 MB"
    "ptb-xl|https://physionet.org/content/ptb-xl/1.0.3/|PTB-XL ECG Database|~3 GB"
    "chfdb|https://physionet.org/content/chfdb/1.0.0/|BIDMC Congestive Heart Failure ECG|~90 MB"
    "chf2db|https://physionet.org/content/chf2db/1.0.0/|CHF RR Interval Database (NYHA)|~600 MB"
)

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] [DATASET...]

Download PhysioNet datasets for OOD evaluation into datasets/ at project root.

DATASETS (specify by short name, or omit to download all):
  gaitpdb   Gait in Parkinson's Disease          (~288 MB)
  gaitndd   Gait in Neurodegenerative Disease    (~18 MB)
  ptb-xl    PTB-XL ECG Database                  (~3 GB)
  chfdb     BIDMC Congestive Heart Failure ECG    (~90 MB)
  chf2db    CHF RR Interval Database (NYHA)      (~600 MB)

OPTIONS:
  -h, --help    Show this help message and exit

Examples:
  $(basename "$0")                  # download all datasets
  $(basename "$0") gaitndd chfdb   # download only two datasets
EOF
    exit 0
}

download_dataset() {
    local name="$1" url="$2" desc="$3" size="$4"
    local dest="$DATASETS_DIR/$name"

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Dataset : $desc"
    echo "  Name    : $name"
    echo "  Size    : $size"
    echo "  URL     : $url"
    echo "  Dest    : $dest"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    mkdir -p "$dest"

    wget -r -N -c \
        --no-host-directories \
        --cut-dirs=3 \
        --directory-prefix="$dest" \
        --no-parent \
        "$url"

    echo ""
    echo "  ✓ $name complete"
    echo ""
}

# Parse args
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
fi

SELECTED=("$@")

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║       PhysioNet Dataset Downloader — OOD Evaluation        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Target directory: $DATASETS_DIR"
echo ""

if ! command -v wget &>/dev/null; then
    echo "Error: wget is required but not installed."
    echo "  macOS:  brew install wget"
    echo "  Linux:  sudo apt-get install wget"
    exit 1
fi

mkdir -p "$DATASETS_DIR"

downloaded=0
skipped=0

for entry in "${DATASETS[@]}"; do
    IFS='|' read -r name url desc size <<< "$entry"

    # If specific datasets were requested, skip those not listed
    if [[ ${#SELECTED[@]} -gt 0 ]]; then
        match=false
        for sel in "${SELECTED[@]}"; do
            if [[ "$sel" == "$name" ]]; then
                match=true
                break
            fi
        done
        if [[ "$match" == false ]]; then
            ((skipped++)) || true
            continue
        fi
    fi

    download_dataset "$name" "$url" "$desc" "$size"
    ((downloaded++)) || true
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Done. $downloaded dataset(s) downloaded, $skipped skipped."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
