"""
AlwaysOnPT — Real-Time Demo
Simulates continuous EMG signal intake from a wearable sensor,
processes segments through the RLM agent, and streams every step
to LangSmith for real-time observability.

Run:
  python3 -m alwaysonpt.demo                    # default: 2 per class
  python3 -m alwaysonpt.demo --n-per-class 5    # 5 per class = 20 total
  python3 -m alwaysonpt.demo --all              # all segments

Dashboard:
  Open https://smith.langchain.com → project "always-on-pt"
  Every segment classification appears as a trace with nested spans:
    classify_segment
    ├── root_llm (turn 1)
    ├── repl_exec (code block 1)
    ├── root_llm (turn 2)
    ├── repl_exec (code block 2)
    ├── sub_llm  (multimodal verification)
    ├── classify_emg (specialized model call)
    └── ...
"""

import os
import sys
import time
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from langsmith import traceable

from alwaysonpt.data_loader import load_dataset
from alwaysonpt.rlm_agent import RLMAgent, save_traces


OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ['standing', 'sitting', 'stance', 'swing']
PAPER_BASELINE = {'accuracy': 91.85, 'std': 0.88}


def select_segments(dataset, n_per_class=None, use_all=False):
    """Pick segments for evaluation — balanced across classes."""
    if use_all:
        return dataset.segments

    if n_per_class is None:
        n_per_class = 2

    np.random.seed(42)
    selected = []
    for cls in CLASS_NAMES:
        cls_segs = [s for s in dataset.segments if s.motion_class == cls]
        if len(cls_segs) > n_per_class:
            chosen = np.random.choice(len(cls_segs), n_per_class, replace=False)
            selected.extend(cls_segs[i] for i in chosen)
        else:
            selected.extend(cls_segs)

    return selected


@traceable(run_type="chain", name="realtime_pt_session",
           tags=["alwaysonpt", "demo", "realtime"])
def run_realtime_session(segments: list, verbose: bool = True) -> dict:
    """
    Simulate a real-time PT monitoring session.

    Each segment arrives as if from a wearable sensor — the agent
    analyzes it, classifies the motion, generates a clinical narrative,
    and accumulates session context. Every step streams to LangSmith.
    """
    agent = RLMAgent(verbose=verbose)

    y_true = []
    y_pred = []
    traces = []
    confidences = []
    narratives = []
    timings = []

    print(f"\n{'='*70}")
    print(f"  AlwaysOnPT — Real-Time PT Session")
    print(f"  Segments: {len(segments)}")
    print(f"  LangSmith project: {os.environ.get('LANGSMITH_PROJECT', 'default')}")
    print(f"  Dashboard: https://smith.langchain.com")
    print(f"{'='*70}\n")

    session_start = time.time()

    for i, seg in enumerate(segments):
        seg_id = f"S{seg.subject_id}_{seg.motion_class}_{i:03d}"

        print(f"\n  [{i+1}/{len(segments)}] Signal received: {seg_id}")
        print(f"    Subject {seg.subject_id} | {seg.duration_s:.2f}s | "
              f"{len(seg.emg)} samples @ {seg.fs}Hz")

        t0 = time.time()
        trace = agent.classify_segment(seg, seg_id)
        elapsed = time.time() - t0

        y_true.append(seg.motion_class)
        y_pred.append(trace.classification)
        traces.append(trace)
        confidences.append(trace.confidence)
        narratives.append(trace.clinical_narrative)
        timings.append(elapsed)

        correct = trace.classification == seg.motion_class
        status = "CORRECT" if correct else "WRONG"

        running_correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        running_acc = running_correct / len(y_true)

        print(f"    Classification: {trace.classification} "
              f"(true: {seg.motion_class}) — {status}")
        print(f"    Confidence: {trace.confidence:.2f} | "
              f"Time: {elapsed:.1f}s | "
              f"Running acc: {running_acc*100:.1f}%")
        if trace.clinical_narrative:
            print(f"    Narrative: {trace.clinical_narrative[:120]}...")

    total_time = time.time() - session_start

    n_correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = n_correct / len(y_true) if y_true else 0.0

    cm = np.zeros((4, 4), dtype=int)
    label_map = {name: i for i, name in enumerate(CLASS_NAMES)}
    for t, p in zip(y_true, y_pred):
        if p in label_map:
            cm[label_map[t]][label_map[p]] += 1

    per_class = {}
    for cls in CLASS_NAMES:
        idx = label_map[cls]
        tp = cm[idx, idx]
        fp = cm[:, idx].sum() - tp
        fn = cm[idx, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[cls] = {'P': precision, 'R': recall, 'F1': f1,
                          'n': int(cm[idx, :].sum())}

    result = {
        'accuracy': float(accuracy),
        'accuracy_pct': float(accuracy * 100),
        'n_correct': n_correct,
        'n_segments': len(segments),
        'mean_confidence': float(np.mean(confidences)),
        'mean_time_s': float(np.mean(timings)),
        'total_time_s': float(total_time),
        'confusion_matrix': cm.tolist(),
        'per_class': per_class,
        'paper_baseline': PAPER_BASELINE,
        'delta_vs_paper': float(accuracy * 100 - PAPER_BASELINE['accuracy']),
        'narratives_generated': len([n for n in narratives if n]),
        'session_context': agent.session_context,
    }

    _print_session_summary(result)

    trace_path = save_traces(traces)

    out_path = OUTPUT_DIR / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  Session saved to {out_path}")
    print(f"  Traces saved to {trace_path}")
    print(f"  View in LangSmith: https://smith.langchain.com")

    return result


def _print_session_summary(result: dict):
    """Print the final comparison table."""
    acc = result['accuracy_pct']
    paper = PAPER_BASELINE['accuracy']
    delta = result['delta_vs_paper']
    sign = '+' if delta >= 0 else ''

    print(f"\n{'='*70}")
    print(f"  SESSION COMPLETE — AlwaysOnPT vs SVM Baseline")
    print(f"{'='*70}")
    print()
    print(f"  {'Metric':<30} {'SVM (Zhang 2017)':<20} {'AlwaysOnPT':<20}")
    print(f"  {'-'*70}")
    print(f"  {'Classification Accuracy':<30} "
          f"{paper:.2f}% +/- {PAPER_BASELINE['std']:.2f}%{'':<3} "
          f"{acc:.2f}%")
    print(f"  {'Clinical Narrative':<30} {'None':<20} "
          f"{'Yes (' + str(result['narratives_generated']) + ')':<20}")
    print(f"  {'Mean Confidence':<30} {'N/A':<20} "
          f"{result['mean_confidence']:.2f}")
    print(f"  {'Time per Segment':<30} {'<1ms':<20} "
          f"{result['mean_time_s']:.1f}s")
    print(f"  {'Fatigue Detection':<30} {'No':<20} {'Yes':<20}")
    print(f"  {'Self-improving':<30} {'No':<20} {'Yes':<20}")
    print(f"  {'LangSmith Observability':<30} {'No':<20} {'Yes':<20}")
    print()
    print(f"  Delta vs paper: {sign}{delta:.2f}pp")
    print()
    print(f"  Confusion Matrix:")
    cm = result['confusion_matrix']
    print(f"  {'':>12} {'standing':>10} {'sitting':>10} {'stance':>10} {'swing':>10}")
    for i, cls in enumerate(CLASS_NAMES):
        print(f"  {cls:>12} {cm[i][0]:>10} {cm[i][1]:>10} {cm[i][2]:>10} {cm[i][3]:>10}")
    print()
    print(f"  Per-class:")
    for cls in CLASS_NAMES:
        r = result['per_class'][cls]
        print(f"    {cls:>10}: P={r['P']:.2f}  R={r['R']:.2f}  "
              f"F1={r['F1']:.2f}  (n={r['n']})")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AlwaysOnPT — Real-Time EMG Classification Demo"
    )
    parser.add_argument('--n-per-class', type=int, default=2,
                       help='Segments per class (default: 2)')
    parser.add_argument('--all', action='store_true',
                       help='Run on all segments')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress per-turn agent output')
    args = parser.parse_args()

    dataset = load_dataset()
    segments = select_segments(
        dataset,
        n_per_class=args.n_per_class,
        use_all=args.all,
    )

    result = run_realtime_session(segments, verbose=not args.quiet)
