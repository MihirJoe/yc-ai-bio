#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_ROOT/data"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          AlwaysOnPT — Demo Data Downloader                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Downloads sample biosignal data for the AlwaysOnPT static demo.

DATA SOURCES:
  knee_emg        Zhang et al. 2017 knee EMG dataset (2-channel, 14 subjects)
                  Already included if you cloned with data.
  live_sensor     Live EMG + IMU recordings from iOS sensor app
                  Downloaded from Google Drive.
  multichannel    16-channel EMG array recordings
                  Downloaded from Google Drive (when available).

OPTIONS:
  --all           Download all available datasets
  --knee-emg      Download/verify knee EMG dataset only
  --live-sensor   Download live sensor recordings only
  --multichannel  Download 16-channel EMG data only
  --check         Check which data is already present
  -h, --help      Show this help message and exit

Examples:
  $(basename "$0") --all
  $(basename "$0") --live-sensor --multichannel
  $(basename "$0") --check
EOF
    exit 0
}

check_data() {
    echo "  Checking data directory: $DATA_DIR"
    echo ""

    # Knee EMG
    local knee_count=0
    if [[ -d "$DATA_DIR/knee_emg/S1File/Data" ]]; then
        knee_count=$(find "$DATA_DIR/knee_emg/S1File/Data" -name "*.txt" 2>/dev/null | wc -l | tr -d ' ')
    fi
    if [[ $knee_count -gt 0 ]]; then
        echo "  ✓ knee_emg          $knee_count files (2-channel, 1kHz)"
    else
        echo "  ✗ knee_emg          not found"
    fi

    # Live sensor
    local live_count=0
    if [[ -d "$DATA_DIR/live_data_sample" ]]; then
        live_count=$(find "$DATA_DIR/live_data_sample" -name "*.json" 2>/dev/null | wc -l | tr -d ' ')
    fi
    if [[ $live_count -gt 0 ]]; then
        echo "  ✓ live_data_sample  $live_count recordings (EMG + IMU)"
    else
        echo "  ✗ live_data_sample  not found"
    fi

    # 16-channel
    local multi_count=0
    if [[ -d "$DATA_DIR/multichannel_emg" ]]; then
        multi_count=$(find "$DATA_DIR/multichannel_emg" -name "*.json" -o -name "*.csv" 2>/dev/null | wc -l | tr -d ' ')
    fi
    if [[ $multi_count -gt 0 ]]; then
        echo "  ✓ multichannel_emg  $multi_count files (16-channel)"
    else
        echo "  ✗ multichannel_emg  not found (will use synthetic data in demo)"
    fi

    echo ""
}

download_gdrive() {
    local file_id="$1"
    local dest="$2"
    local desc="$3"

    echo "  Downloading: $desc"
    echo "  Destination: $dest"

    if command -v gdown &>/dev/null; then
        gdown "$file_id" -O "$dest" --fuzzy
    elif command -v curl &>/dev/null; then
        echo "  (Using curl — for large files, install gdown: pip install gdown)"
        curl -L "https://drive.google.com/uc?export=download&id=$file_id" -o "$dest"
    else
        echo "  Error: need curl or gdown (pip install gdown) to download."
        return 1
    fi
    echo "  ✓ Downloaded $desc"
    echo ""
}

generate_synthetic_multichannel() {
    echo "  Generating synthetic 16-channel EMG data for demo..."
    python3 -c "
import json, os, math, random
random.seed(42)

out_dir = '$DATA_DIR/multichannel_emg'
os.makedirs(out_dir, exist_ok=True)

MUSCLES = [
    'vastus_medialis', 'vastus_lateralis', 'rectus_femoris', 'biceps_femoris',
    'semitendinosus', 'gastrocnemius_med', 'gastrocnemius_lat', 'tibialis_anterior',
    'soleus', 'peroneus_longus', 'gluteus_medius', 'gluteus_maximus',
    'tensor_fasciae', 'adductor_longus', 'sartorius', 'gracilis'
]

for scenario, dur in [('walking', 10.0), ('squat', 8.0), ('stair_climb', 12.0)]:
    fs = 1000
    n = int(dur * fs)
    channels = {}
    for i, m in enumerate(MUSCLES):
        base_freq = 15 + i * 3
        phase = i * 0.4
        t = [j / fs for j in range(n)]
        envelope = [0.02 + 0.08 * abs(math.sin(2 * math.pi * 1.2 * tj + phase)) for tj in t]
        signal = [env * (0.5 * math.sin(2 * math.pi * base_freq * tj) +
                         0.3 * math.sin(2 * math.pi * base_freq * 2.3 * tj) +
                         random.gauss(0, 0.3))
                  for tj, env in zip(t, envelope)]
        channels[m] = [round(v, 6) for v in signal]

    rec = {
        'channels': channels,
        'channelNames': MUSCLES,
        'samplingRate': fs,
        'duration': dur,
        'scenario': scenario,
        'source': 'synthetic',
        'metadata': {
            'description': f'Synthetic 16-channel EMG: {scenario}',
            'muscles': MUSCLES,
            'units': 'mV',
        }
    }
    path = os.path.join(out_dir, f'multichannel_{scenario}.json')
    with open(path, 'w') as f:
        json.dump(rec, f)
    print(f'    Created {path} ({dur}s, 16 channels)')

print('  ✓ Synthetic 16-channel data generated')
"
    echo ""
}

# --- Main ---

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
fi

DO_KNEE=false
DO_LIVE=false
DO_MULTI=false
DO_CHECK=false

if [[ $# -eq 0 ]]; then
    echo "  Run with --help to see options, or --check to see current data status."
    echo ""
    DO_CHECK=true
fi

for arg in "$@"; do
    case "$arg" in
        --all)         DO_KNEE=true; DO_LIVE=true; DO_MULTI=true ;;
        --knee-emg)    DO_KNEE=true ;;
        --live-sensor) DO_LIVE=true ;;
        --multichannel) DO_MULTI=true ;;
        --check)       DO_CHECK=true ;;
        *)             echo "  Unknown option: $arg"; usage ;;
    esac
done

if $DO_CHECK; then
    check_data
fi

mkdir -p "$DATA_DIR"

if $DO_KNEE; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Knee EMG Dataset (Zhang et al. 2017)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    if [[ -d "$DATA_DIR/knee_emg/S1File/Data" ]]; then
        count=$(find "$DATA_DIR/knee_emg/S1File/Data" -name "*.txt" | wc -l | tr -d ' ')
        echo "  Already present: $count files"
    else
        echo "  Not found. This dataset must be manually placed at:"
        echo "    $DATA_DIR/knee_emg/S1File/Data/"
        echo "  Download from the original source or ask the team for a copy."
    fi
    echo ""
fi

if $DO_LIVE; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Live Sensor Recordings (EMG + IMU)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    if [[ -d "$DATA_DIR/live_data_sample" ]]; then
        count=$(find "$DATA_DIR/live_data_sample" -name "*.json" | wc -l | tr -d ' ')
        echo "  Already present: $count recordings"
    else
        echo "  Not found. Place live sensor JSON files at:"
        echo "    $DATA_DIR/live_data_sample/"
        echo ""
        echo "  Or download from Google Drive (ask team for link):"
        echo "    gdown <GDRIVE_FILE_ID> -O /tmp/live_data.zip"
        echo "    unzip /tmp/live_data.zip -d $DATA_DIR/live_data_sample/"
    fi
    echo ""
fi

if $DO_MULTI; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  16-Channel EMG Data"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    if [[ -d "$DATA_DIR/multichannel_emg" ]]; then
        count=$(find "$DATA_DIR/multichannel_emg" -name "*.json" -o -name "*.csv" | wc -l | tr -d ' ')
        echo "  Found $count files"
    else
        echo "  Not found. Generating synthetic 16-channel data for demo..."
        echo ""
        generate_synthetic_multichannel
    fi
    echo ""
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Done. Run the demo:"
echo "    python -m alwaysonpt.server"
echo "    open http://localhost:8000"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
