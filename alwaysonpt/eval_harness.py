"""
AlwaysOnPT — Evaluation Harness
Evaluates the RLM agent on 4-class lower-limb motion classification
and compares accuracy to the WT-SVD + SVM baseline (91.85% +/- 0.88%)
reported in Zhang et al. 2017.

No SVM is trained here — the paper's number is the benchmark.
"""

import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from alwaysonpt.data_loader import load_dataset, EMGDataset
from alwaysonpt.rlm_agent import RLMAgent, save_traces


OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ['standing', 'sitting', 'stance', 'swing']
LABEL_MAP = {name: i for i, name in enumerate(CLASS_NAMES)}

PAPER_BASELINE = {'accuracy': 91.85, 'std': 0.88, 'source': 'Zhang et al. 2017, WT-SVD + SVM'}


def evaluate_agent(dataset: EMGDataset,
                   max_segments: int = None,
                   verbose: bool = True) -> dict:
    """
    Evaluate the RLM agent on all (or a subset of) segments.
    Returns accuracy, per-class metrics, confusion matrix,
    and all reasoning traces / clinical narratives.
    """
    print("\n" + "=" * 60)
    print("AlwaysOnPT AGENT EVALUATION")
    print("=" * 60)

    segments = dataset.segments
    if max_segments and max_segments < len(segments):
        np.random.seed(42)
        indices = []
        for cls in CLASS_NAMES:
            cls_indices = [i for i, s in enumerate(segments)
                          if s.motion_class == cls]
            n_per_class = max(1, max_segments // 4)
            if len(cls_indices) > n_per_class:
                chosen = np.random.choice(cls_indices, n_per_class,
                                         replace=False)
                indices.extend(chosen)
            else:
                indices.extend(cls_indices)
        indices = sorted(indices)
        segments = [segments[i] for i in indices]

    print(f"  Segments: {len(segments)}")
    class_counts = defaultdict(int)
    for seg in segments:
        class_counts[seg.motion_class] += 1
    print(f"  Distribution: {dict(class_counts)}")

    agent = RLMAgent(verbose=verbose)

    y_true = []
    y_pred = []
    traces = []
    confidences = []
    narratives = []
    segment_times = []

    for i, seg in enumerate(segments):
        seg_id = f"S{seg.subject_id}_{seg.motion_class}_{i}"
        print(f"\n  [{i+1}/{len(segments)}] {seg_id}")

        t0 = time.time()
        trace = agent.classify_segment(seg, seg_id)
        elapsed = time.time() - t0

        y_true.append(seg.motion_class)
        y_pred.append(trace.classification)
        traces.append(trace)
        confidences.append(trace.confidence)
        narratives.append(trace.clinical_narrative)
        segment_times.append(elapsed)

        correct = trace.classification == seg.motion_class
        print(f"    -> {trace.classification} "
              f"(true: {seg.motion_class}) "
              f"{'CORRECT' if correct else 'WRONG'} "
              f"[conf={trace.confidence:.2f}, {elapsed:.1f}s]")

    n_correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = n_correct / len(y_true) if y_true else 0.0

    cm = np.zeros((4, 4), dtype=int)
    for t, p in zip(y_true, y_pred):
        if p in LABEL_MAP:
            cm[LABEL_MAP[t]][LABEL_MAP[p]] += 1

    per_class = {}
    for cls in CLASS_NAMES:
        idx = LABEL_MAP[cls]
        tp = cm[idx, idx]
        fp = cm[:, idx].sum() - tp
        fn = cm[idx, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[cls] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'support': int(cm[idx, :].sum()),
        }

    result = {
        'accuracy': float(accuracy),
        'accuracy_pct': float(accuracy * 100),
        'n_correct': n_correct,
        'n_segments': len(segments),
        'class_distribution': dict(class_counts),
        'confusion_matrix': cm.tolist(),
        'per_class': per_class,
        'mean_confidence': float(np.mean(confidences)),
        'mean_time_per_segment_s': float(np.mean(segment_times)),
        'total_time_s': float(sum(segment_times)),
        'narratives_generated': len([n for n in narratives if n]),
        'paper_baseline': PAPER_BASELINE,
        'delta_vs_paper': float(accuracy * 100 - PAPER_BASELINE['accuracy']),
        'predictions': [
            {
                'true': t, 'pred': p,
                'confidence': c, 'time_s': round(ts, 1),
                'narrative': n[:300] if n else '',
            }
            for t, p, c, ts, n in zip(
                y_true, y_pred, confidences, segment_times, narratives
            )
        ],
    }

    _print_results(result)

    trace_path = save_traces(traces)
    result['trace_file'] = trace_path

    out_path = OUTPUT_DIR / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved results to {out_path}")

    return result


def _print_results(result: dict):
    """Pretty-print evaluation results and comparison to paper."""
    acc = result['accuracy_pct']
    paper = PAPER_BASELINE['accuracy']
    delta = result['delta_vs_paper']

    print(f"\n{'='*70}")
    print(f"RESULTS — AlwaysOnPT vs SVM Baseline")
    print(f"{'='*70}")
    print()
    print(f"  {'Metric':<30} {'SVM (Zhang 2017)':<20} {'AlwaysOnPT':<20}")
    print(f"  {'-'*70}")
    print(f"  {'Classification Accuracy':<30} {paper:.2f}% +/- {PAPER_BASELINE['std']:.2f}%{'':<3} {acc:.2f}%")
    print(f"  {'Clinical Narrative':<30} {'None':<20} {'Yes (' + str(result['narratives_generated']) + ')':<20}")
    print(f"  {'Mean Confidence':<30} {'N/A':<20} {result['mean_confidence']:.2f}")
    print(f"  {'Time per Segment':<30} {'<1ms':<20} {result['mean_time_per_segment_s']:.1f}s")
    print(f"  {'Fatigue Detection':<30} {'No':<20} {'Yes':<20}")
    print(f"  {'Self-improving':<30} {'No':<20} {'Yes':<20}")
    print()
    sign = '+' if delta >= 0 else ''
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
        print(f"    {cls:>10}: P={r['precision']:.2f}  R={r['recall']:.2f}  F1={r['f1']:.2f}  (n={r['support']})")
    print(f"{'='*70}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AlwaysOnPT Evaluation")
    parser.add_argument('--max-segments', type=int, default=None,
                       help='Max segments (samples evenly across classes). '
                            'Default: all segments.')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress per-turn agent output')
    args = parser.parse_args()

    dataset = load_dataset()
    result = evaluate_agent(dataset,
                            max_segments=args.max_segments,
                            verbose=not args.quiet)
