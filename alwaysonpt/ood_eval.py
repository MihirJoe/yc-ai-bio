"""
Out-of-Distribution Evaluation Harness.
Tests the BioSignalAgent's reasoning generalization across 5 PhysioNet datasets.
Scores: binary accuracy, categorical F1, ordinal rank correlation, LLM-as-judge reasoning.
"""

import json
import time
import re
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import anthropic

from alwaysonpt.biosignal_agent import BioSignalAgent
from alwaysonpt.datasets.base import BioSignalRecord
from alwaysonpt.datasets.questions import QUESTION_BANK, get_questions
from alwaysonpt.rlm_agent import save_traces

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

JUDGE_MODEL = "claude-sonnet-4-6"


def load_dataset_records(dataset_name: str, data_dir: str,
                          max_records: int = None) -> list:
    """Load records for a given dataset name."""
    if dataset_name == 'gaitpdb':
        from alwaysonpt.datasets.gaitpdb_loader import load_gaitpdb
        return load_gaitpdb(data_dir, max_records)
    elif dataset_name == 'gaitndd':
        from alwaysonpt.datasets.gaitndd_loader import load_gaitndd
        return load_gaitndd(data_dir, max_records)
    elif dataset_name == 'ptb_xl':
        from alwaysonpt.datasets.ptbxl_loader import load_ptbxl
        return load_ptbxl(data_dir, max_records)
    elif dataset_name == 'chfdb':
        from alwaysonpt.datasets.chfdb_loader import load_chfdb
        return load_chfdb(data_dir, max_records)
    elif dataset_name == 'chf2db':
        from alwaysonpt.datasets.chf2db_loader import load_chf2db
        return load_chf2db(data_dir, max_records)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def evaluate_dataset(dataset_name: str, records: list,
                      questions: list = None, verbose: bool = True) -> dict:
    """
    Evaluate the BioSignalAgent on one dataset.
    Returns per-question scores and reasoning traces.
    """
    if questions is None:
        questions = get_questions(dataset_name)

    if not questions:
        return {'dataset': dataset_name, 'error': 'no questions defined'}

    agent = BioSignalAgent(verbose=verbose)
    results = {
        'dataset': dataset_name,
        'n_records': len(records),
        'questions': {},
        'traces': [],
        'total_time_s': 0.0,
    }

    for q in questions:
        q_id = q['id']
        q_results = _evaluate_question(agent, records, q, verbose)
        results['questions'][q_id] = q_results
        results['total_time_s'] += q_results.get('total_time_s', 0)

    results['mean_time_per_record_s'] = (
        results['total_time_s'] / max(1, len(records) * len(questions))
    )

    return results


def _evaluate_question(agent: BioSignalAgent, records: list,
                        question: dict, verbose: bool) -> dict:
    """Evaluate a single question across all records."""
    q_id = question['id']
    task_type = question['task_type']
    q_text = question['question']
    eval_type = question['eval_type']
    extract_fn = question.get('extract_answer', lambda gt: None)

    predictions = []
    ground_truths = []
    traces = []
    times = []

    for record in records:
        gt_value = extract_fn(record.ground_truth)

        if verbose:
            print(f"\n  [{q_id}] Record: {record.record_id}")

        t0 = time.time()
        trace = agent.analyze(
            record, task_type=task_type,
            question=q_text, record_id=record.record_id,
        )
        elapsed = time.time() - t0
        times.append(elapsed)

        pred = trace.classification
        predictions.append(pred)
        ground_truths.append(gt_value)
        traces.append(trace)

        if verbose:
            print(f"    Prediction: {pred} | Ground truth: {gt_value} | "
                  f"Confidence: {trace.confidence:.2f} | Time: {elapsed:.1f}s")

    result = {
        'question_id': q_id,
        'eval_type': eval_type,
        'n_evaluated': len(predictions),
        'total_time_s': float(sum(times)),
        'mean_time_s': float(np.mean(times)) if times else 0,
        'predictions': [
            {'record_id': r.record_id, 'pred': p, 'truth': str(gt),
             'confidence': t.confidence, 'narrative': t.clinical_narrative[:300]}
            for r, p, gt, t in zip(records, predictions, ground_truths, traces)
        ],
    }

    if eval_type == 'binary':
        result.update(_score_binary(predictions, ground_truths, question))
    elif eval_type == 'categorical':
        result.update(_score_categorical(predictions, ground_truths, question))
    elif eval_type == 'ordinal':
        result.update(_score_ordinal(predictions, ground_truths, question))
    elif eval_type == 'reasoning_quality':
        result.update(_score_reasoning(traces, records, question))

    return result


def _score_binary(predictions, ground_truths, question) -> dict:
    """Score binary classification (accuracy, precision, recall, F1)."""
    positive = question.get('positive_label', 'patient')

    tp = fp = tn = fn = 0
    for pred, truth in zip(predictions, ground_truths):
        pred_pos = _fuzzy_match_binary(pred, positive)
        truth_pos = str(truth) == positive

        if pred_pos and truth_pos:
            tp += 1
        elif pred_pos and not truth_pos:
            fp += 1
        elif not pred_pos and truth_pos:
            fn += 1
        else:
            tn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
    }


def _fuzzy_match_binary(pred: str, positive_label: str) -> bool:
    """Fuzzy match a prediction to the positive label."""
    pred_lower = str(pred).lower().strip()
    pos_lower = positive_label.lower()

    if pos_lower in pred_lower:
        return True

    positive_synonyms = {
        'patient': ['patient', 'parkinsonian', 'abnormal', 'pathological', 'disease', 'pd'],
        'abnormal': ['abnormal', 'pathological', 'disease', 'mi', 'sttc', 'cd', 'hyp'],
    }
    for syn in positive_synonyms.get(pos_lower, []):
        if syn in pred_lower:
            return True

    return False


def _score_categorical(predictions, ground_truths, question) -> dict:
    """Score multi-class classification (accuracy, macro F1)."""
    categories = question.get('categories', [])
    if not categories:
        categories = list(set(str(gt) for gt in ground_truths if gt))

    correct = 0
    per_class = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})

    for pred, truth in zip(predictions, ground_truths):
        pred_cat = _fuzzy_match_category(pred, categories)
        truth_cat = str(truth).lower()

        if pred_cat == truth_cat:
            correct += 1
            per_class[truth_cat]['tp'] += 1
        else:
            per_class[truth_cat]['fn'] += 1
            if pred_cat:
                per_class[pred_cat]['fp'] += 1

    total = len(predictions)
    accuracy = correct / total if total > 0 else 0

    f1s = []
    for cls in categories:
        cls_lower = cls.lower()
        tp = per_class[cls_lower]['tp']
        fp = per_class[cls_lower]['fp']
        fn = per_class[cls_lower]['fn']
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        f1s.append(f1)

    return {
        'accuracy': float(accuracy),
        'macro_f1': float(np.mean(f1s)) if f1s else 0.0,
        'n_correct': correct,
    }


def _fuzzy_match_category(pred: str, categories: list) -> str:
    """Fuzzy match a prediction to the closest category."""
    pred_lower = str(pred).lower().strip()
    for cat in categories:
        if cat.lower() in pred_lower:
            return cat.lower()
    for cat in categories:
        if pred_lower in cat.lower():
            return cat.lower()
    return pred_lower


def _score_ordinal(predictions, ground_truths, question) -> dict:
    """Score ordinal predictions with Spearman rank correlation."""
    mapping = question.get('mapping', {})

    pred_values = []
    truth_values = []

    for pred, truth in zip(predictions, ground_truths):
        if truth is None:
            continue
        tv = float(truth) if isinstance(truth, (int, float)) else 0
        truth_values.append(tv)

        pv = _parse_ordinal(pred, mapping)
        pred_values.append(pv)

    if len(pred_values) < 3:
        return {'spearman_rho': float('nan'), 'error': 'too few valid predictions'}

    pred_arr = np.array(pred_values)
    truth_arr = np.array(truth_values)

    pred_ranks = _rank(pred_arr)
    truth_ranks = _rank(truth_arr)
    n = len(pred_ranks)
    d_sq = np.sum((pred_ranks - truth_ranks) ** 2)
    rho = 1 - (6 * d_sq) / (n * (n ** 2 - 1))

    errors = np.abs(pred_arr - truth_arr)

    return {
        'spearman_rho': float(rho),
        'mean_absolute_error': float(np.mean(errors)),
        'median_absolute_error': float(np.median(errors)),
    }


def _parse_ordinal(pred: str, mapping: dict) -> float:
    """Parse an ordinal prediction into a numeric value."""
    pred_lower = str(pred).lower().strip()

    for val, label in mapping.items():
        if str(label).lower() in pred_lower:
            return float(val)

    numbers = re.findall(r'[\d.]+', pred_lower)
    if numbers:
        try:
            return float(numbers[0])
        except ValueError:
            pass

    ordinal_words = {
        'none': 0, 'healthy': 0, 'normal': 0,
        'mild': 1, 'slight': 1,
        'moderate': 2, 'mild-moderate': 1.5,
        'severe': 3, 'marked': 3,
        'very severe': 4,
    }
    for word, val in ordinal_words.items():
        if word in pred_lower:
            return float(val)

    return 0.0


def _rank(arr: np.ndarray) -> np.ndarray:
    """Compute ranks for Spearman correlation."""
    order = arr.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(arr) + 1, dtype=float)
    return ranks


def _score_reasoning(traces, records, question) -> dict:
    """Score reasoning quality using LLM-as-judge."""
    try:
        client = anthropic.Anthropic()
    except Exception:
        return {'mean_reasoning_score': 0.0, 'error': 'no API key'}

    scores = []
    for trace, record in zip(traces, records):
        score = _judge_reasoning(client, trace, record, question)
        scores.append(score)

    return {
        'mean_reasoning_score': float(np.mean(scores)) if scores else 0.0,
        'std_reasoning_score': float(np.std(scores)) if scores else 0.0,
        'scores': scores,
    }


def _judge_reasoning(client, trace, record, question) -> float:
    """Use LLM-as-judge to score reasoning quality (1-5)."""
    rubric = (
        "Score this biomedical signal analysis on a scale of 1-5:\n"
        "1: No meaningful analysis, vague or incorrect statements\n"
        "2: Some analysis but major gaps, incorrect conclusions\n"
        "3: Reasonable analysis with some specific findings, minor errors\n"
        "4: Good analysis grounded in signal evidence, mostly correct\n"
        "5: Excellent analysis with specific metrics, physiologically sound reasoning\n\n"
        "Criteria:\n"
        "- Does the narrative reference specific computed signal features?\n"
        "- Are the cited values plausible for this signal domain?\n"
        "- Is the clinical reasoning physiologically coherent?\n"
        "- Does the analysis demonstrate systematic investigation?\n"
    )

    prompt = (
        f"## Question asked\n{question['question']}\n\n"
        f"## Signal domain\n{record.domain}\n\n"
        f"## Agent's narrative\n{trace.clinical_narrative}\n\n"
        f"## Agent's conclusion\n{trace.classification} "
        f"(confidence: {trace.confidence:.2f})\n\n"
        f"## Number of analysis steps\n{len(trace.steps)}\n\n"
        f"{rubric}\n\n"
        f"Respond with ONLY a single integer 1-5."
    )

    try:
        response = client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=10,
            messages=[{'role': 'user', 'content': prompt}],
        )
        text = response.content[0].text.strip()
        score = int(re.search(r'[1-5]', text).group())
        return float(score)
    except Exception:
        return 3.0


def run_ood_eval(data_base_dir: str, datasets: list = None,
                  max_records_per_dataset: int = 10,
                  verbose: bool = True) -> dict:
    """
    Run the full OOD evaluation across multiple datasets.
    """
    if datasets is None:
        datasets = list(QUESTION_BANK.keys())

    dataset_dirs = {
        'gaitpdb': 'gaitpdb',
        'gaitndd': 'gaitndd',
        'ptb_xl': 'ptb-xl',
        'chfdb': 'chfdb',
        'chf2db': 'chf2db',
    }

    print("=" * 70)
    print("OOD EVALUATION — BioSignalAgent Reasoning Generalization")
    print("=" * 70)

    all_results = {}
    for ds_name in datasets:
        ds_dir = Path(data_base_dir) / dataset_dirs.get(ds_name, ds_name)
        if not ds_dir.exists():
            print(f"\n  SKIP {ds_name}: data not found at {ds_dir}")
            continue

        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")

        try:
            records = load_dataset_records(
                ds_name, str(ds_dir), max_records=max_records_per_dataset
            )
        except Exception as e:
            print(f"  ERROR loading {ds_name}: {e}")
            continue

        if not records:
            print(f"  No records loaded for {ds_name}")
            continue

        print(f"  Loaded {len(records)} records")
        result = evaluate_dataset(ds_name, records, verbose=verbose)
        all_results[ds_name] = result

    summary = _build_summary(all_results)

    out_path = OUTPUT_DIR / f"ood_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, 'w') as f:
        json.dump({'results': all_results, 'summary': summary},
                  f, indent=2, default=str)
    print(f"\nSaved OOD eval results to {out_path}")

    _print_summary(summary, all_results)
    return {'results': all_results, 'summary': summary}


def _build_summary(all_results: dict) -> dict:
    """Build aggregate summary across all datasets."""
    total_records = 0
    total_time = 0
    all_scores = []

    for ds_name, result in all_results.items():
        total_records += result.get('n_records', 0)
        total_time += result.get('total_time_s', 0)

        for q_id, q_result in result.get('questions', {}).items():
            if 'accuracy' in q_result:
                all_scores.append(q_result['accuracy'])
            if 'spearman_rho' in q_result:
                rho = q_result['spearman_rho']
                if not np.isnan(rho):
                    all_scores.append(max(0, rho))
            if 'mean_reasoning_score' in q_result:
                all_scores.append(q_result['mean_reasoning_score'] / 5.0)

    return {
        'n_datasets_evaluated': len(all_results),
        'total_records': total_records,
        'total_time_s': total_time,
        'mean_generalization_score': float(np.mean(all_scores)) if all_scores else 0.0,
    }


def _print_summary(summary: dict, all_results: dict):
    """Pretty-print the evaluation summary."""
    print(f"\n{'='*70}")
    print("OOD EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"  Datasets evaluated: {summary['n_datasets_evaluated']}")
    print(f"  Total records: {summary['total_records']}")
    print(f"  Total time: {summary['total_time_s']:.0f}s")
    print(f"  Mean generalization score: {summary['mean_generalization_score']:.3f}")

    for ds_name, result in all_results.items():
        print(f"\n  {ds_name}:")
        for q_id, q_result in result.get('questions', {}).items():
            eval_type = q_result.get('eval_type', '')
            if eval_type == 'binary':
                print(f"    {q_id}: acc={q_result['accuracy']:.2f} "
                      f"f1={q_result['f1']:.2f}")
            elif eval_type == 'categorical':
                print(f"    {q_id}: acc={q_result['accuracy']:.2f} "
                      f"macro_f1={q_result['macro_f1']:.2f}")
            elif eval_type == 'ordinal':
                rho = q_result.get('spearman_rho', float('nan'))
                print(f"    {q_id}: spearman_rho={rho:.3f} "
                      f"mae={q_result.get('mean_absolute_error', 0):.2f}")
            elif eval_type == 'reasoning_quality':
                print(f"    {q_id}: reasoning_score="
                      f"{q_result.get('mean_reasoning_score', 0):.1f}/5")

    print(f"{'='*70}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OOD Evaluation")
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Base directory containing dataset subdirectories')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Specific datasets to evaluate')
    parser.add_argument('--max-records', type=int, default=10,
                        help='Max records per dataset')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    run_ood_eval(
        data_base_dir=args.data_dir,
        datasets=args.datasets,
        max_records_per_dataset=args.max_records,
        verbose=not args.quiet,
    )
