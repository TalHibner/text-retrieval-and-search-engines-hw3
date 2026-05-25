"""
Experiment Helper Script for RAG System Optimization

This script provides utilities for quick experimentation and analysis
without running the full notebook.
"""

import json
import re
import string
from collections import Counter
import pandas as pd


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in set(string.punctuation))

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    """Compute token-level F1 score between prediction and a ground truth."""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())

    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return int(pred_tokens == gt_tokens)
    if num_same == 0:
        return 0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """Calculate max metric over all ground truths."""
    return max(metric_fn(prediction, gt) for gt in ground_truths)


def analyze_predictions(predictions_file, gold_file):
    """
    Analyze prediction file against ground truth.

    Args:
        predictions_file: Path to CSV with predictions
        gold_file: Path to CSV with ground truth answers

    Returns:
        Dictionary with analysis results
    """
    # Load files
    # IMPORTANT: Use converters to properly parse JSON columns
    pred_df = pd.read_csv(predictions_file)
    gold_df = pd.read_csv(gold_file, converters={"answers": json.loads})

    # Parse predictions
    pred_df['prediction'] = pred_df['prediction'].apply(
        lambda x: json.loads(x)[0] if isinstance(x, str) else x
    )

    # Calculate metrics
    results = {
        'total_questions': len(gold_df),
        'predictions_made': len(pred_df),
        'f1_scores': [],
        'perfect_matches': 0,
        'zero_scores': 0,
        'examples': []
    }

    for idx, row in gold_df.iterrows():
        qid = row['id']

        # Find prediction
        pred_row = pred_df[pred_df['id'] == qid]
        if pred_row.empty:
            results['f1_scores'].append(0)
            continue

        prediction = pred_row.iloc[0]['prediction']
        ground_truths = row['answers']

        # Calculate F1
        f1 = metric_max_over_ground_truths(f1_score, prediction, ground_truths)
        results['f1_scores'].append(f1)

        if f1 == 1.0:
            results['perfect_matches'] += 1
        elif f1 == 0.0:
            results['zero_scores'] += 1

        # Store example
        if len(results['examples']) < 10:
            results['examples'].append({
                'question': row['question'],
                'prediction': prediction,
                'ground_truths': ground_truths,
                'f1': f1
            })

    # Calculate summary statistics
    results['mean_f1'] = sum(results['f1_scores']) / len(results['f1_scores']) * 100
    results['median_f1'] = sorted(results['f1_scores'])[len(results['f1_scores']) // 2] * 100
    results['perfect_match_rate'] = results['perfect_matches'] / len(gold_df) * 100
    results['zero_score_rate'] = results['zero_scores'] / len(gold_df) * 100

    return results


def print_analysis(results):
    """Pretty print analysis results."""
    print("=" * 60)
    print("PREDICTION ANALYSIS REPORT")
    print("=" * 60)
    print(f"\nTotal Questions: {results['total_questions']}")
    print(f"Predictions Made: {results['predictions_made']}")
    print(f"\nMean F1 Score: {results['mean_f1']:.2f}%")
    print(f"Median F1 Score: {results['median_f1']:.2f}%")
    print(f"Perfect Matches: {results['perfect_matches']} ({results['perfect_match_rate']:.1f}%)")
    print(f"Zero Scores: {results['zero_scores']} ({results['zero_score_rate']:.1f}%)")

    print("\n" + "=" * 60)
    print("EXAMPLE PREDICTIONS")
    print("=" * 60)

    for i, example in enumerate(results['examples'][:5], 1):
        print(f"\n--- Example {i} (F1: {example['f1']:.2f}) ---")
        print(f"Question: {example['question']}")
        print(f"Prediction: {example['prediction']}")
        print(f"Ground Truth: {example['ground_truths']}")


def compare_predictions(file1, file2, gold_file, name1="System 1", name2="System 2"):
    """
    Compare two prediction files side by side.

    Args:
        file1: Path to first predictions CSV
        file2: Path to second predictions CSV
        gold_file: Path to ground truth CSV
        name1: Name for first system
        name2: Name for second system
    """
    print(f"\nComparing {name1} vs {name2}\n")

    results1 = analyze_predictions(file1, gold_file)
    results2 = analyze_predictions(file2, gold_file)

    print("=" * 70)
    print(f"{'Metric':<30} {name1:<20} {name2:<20}")
    print("=" * 70)
    print(f"{'Mean F1':<30} {results1['mean_f1']:>18.2f}% {results2['mean_f1']:>18.2f}%")
    print(f"{'Perfect Matches':<30} {results1['perfect_matches']:>20} {results2['perfect_matches']:>20}")
    print(f"{'Zero Scores':<30} {results1['zero_scores']:>20} {results2['zero_scores']:>20}")

    improvement = results2['mean_f1'] - results1['mean_f1']
    print(f"\n{'Improvement':<30} {improvement:>+18.2f}%")
    print("=" * 70)


def analyze_errors(predictions_file, gold_file, top_n=20):
    """
    Find and analyze the worst predictions.

    Args:
        predictions_file: Path to predictions CSV
        gold_file: Path to ground truth CSV
        top_n: Number of worst cases to show
    """
    pred_df = pd.read_csv(predictions_file)
    gold_df = pd.read_csv(gold_file, converters={"answers": json.loads})

    pred_df['prediction'] = pred_df['prediction'].apply(
        lambda x: json.loads(x)[0] if isinstance(x, str) else x
    )

    errors = []

    for idx, row in gold_df.iterrows():
        qid = row['id']
        pred_row = pred_df[pred_df['id'] == qid]

        if pred_row.empty:
            continue

        prediction = pred_row.iloc[0]['prediction']
        ground_truths = row['answers']
        f1 = metric_max_over_ground_truths(f1_score, prediction, ground_truths)

        if f1 < 0.5:  # Focus on poor predictions
            errors.append({
                'question': row['question'],
                'prediction': prediction,
                'ground_truths': ground_truths,
                'f1': f1
            })

    # Sort by F1 score (worst first)
    errors.sort(key=lambda x: x['f1'])

    print("=" * 80)
    print(f"TOP {top_n} ERRORS (Lowest F1 Scores)")
    print("=" * 80)

    for i, error in enumerate(errors[:top_n], 1):
        print(f"\n{i}. F1 Score: {error['f1']:.3f}")
        print(f"   Question: {error['question']}")
        print(f"   Predicted: '{error['prediction']}'")
        print(f"   Expected: {error['ground_truths']}")


def suggest_improvements(predictions_file, gold_file):
    """
    Analyze predictions and suggest improvements.

    Args:
        predictions_file: Path to predictions CSV
        gold_file: Path to ground truth CSV
    """
    pred_df = pd.read_csv(predictions_file)
    gold_df = pd.read_csv(gold_file, converters={"answers": json.loads})

    pred_df['prediction'] = pred_df['prediction'].apply(
        lambda x: json.loads(x)[0] if isinstance(x, str) else x
    )

    # Analyze patterns
    issues = {
        'too_long': 0,
        'too_short': 0,
        'has_prefix': 0,
        'has_explanation': 0,
        'unknown': 0
    }

    for idx, row in pred_df.iterrows():
        pred = row['prediction']

        # Check for common issues
        if len(pred.split()) > 10:
            issues['too_long'] += 1
        if len(pred.split()) < 2 and pred.lower() not in ['unknown', 'paris', 'london']:
            issues['too_short'] += 1
        if any(pred.lower().startswith(p) for p in ['the answer is', 'according to']):
            issues['has_prefix'] += 1
        if '.' in pred and len(pred.split()) > 5:
            issues['has_explanation'] += 1
        if pred.lower() in ['unknown', 'i dont know']:
            issues['unknown'] += 1

    print("=" * 60)
    print("IMPROVEMENT SUGGESTIONS")
    print("=" * 60)

    total = len(pred_df)

    if issues['too_long'] / total > 0.1:
        print(f"\n⚠️  {issues['too_long']/total*100:.1f}% of answers are too long (>10 words)")
        print("   → Decrease max_tokens parameter")
        print("   → Improve post-processing to extract key entities")

    if issues['has_prefix'] / total > 0.05:
        print(f"\n⚠️  {issues['has_prefix']/total*100:.1f}% of answers have prefixes")
        print("   → Strengthen post-processing to remove common prefixes")
        print("   → Improve prompt to emphasize 'only the answer'")

    if issues['unknown'] / total > 0.2:
        print(f"\n⚠️  {issues['unknown']/total*100:.1f}% of answers are 'unknown'")
        print("   → Increase k (number of retrieved documents)")
        print("   → Improve query expansion")
        print("   → Try different BM25 parameters (mu)")

    if issues['too_short'] / total > 0.05:
        print(f"\n⚠️  {issues['too_short']/total*100:.1f}% of answers might be too short")
        print("   → Check if single-word answers are correct")
        print("   → May need to include more context")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage:")
        print("  python experiment_helper.py <predictions.csv> <gold.csv>")
        print("\nExample:")
        print("  python experiment_helper.py improved_predictions.csv train.csv")
        print("\nNote: Ensure predictions are formatted correctly:")
        print("  df['prediction'] = df['prediction'].apply(lambda x: json.dumps([x], ensure_ascii=False))")
        sys.exit(1)

    predictions_file = sys.argv[1]
    gold_file = sys.argv[2]

    # Run full analysis
    results = analyze_predictions(predictions_file, gold_file)
    print_analysis(results)

    print("\n" * 2)
    analyze_errors(predictions_file, gold_file, top_n=10)

    print("\n" * 2)
    suggest_improvements(predictions_file, gold_file)
