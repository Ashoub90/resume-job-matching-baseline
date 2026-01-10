"""
Evaluate job -> candidate ranking quality using human-labeled (gold) partial judgments.

Input (default): data/processed/gold_eval_with_scores.csv
Required columns: job_id, candidate_id, final_score, expected_label

This script computes nDCG@10 per job (with log2 discounting) using the relevance mapping:
  strong -> 3
  medium -> 2
  weak -> 1
  no_fit -> 0

Jobs with fewer than 2 labeled candidates are skipped. Jobs with IDCG == 0 are also skipped.
The final output is the number of evaluated jobs and the average nDCG@10.

Do NOT compute or modify labels or scores. This script is strictly for evaluation.
"""

from pathlib import Path
import sys
import math
import pandas as pd
from typing import List
import numpy as np


RELEVANCE_MAP = {
    "strong": 3,
    "medium": 2,
    "weak": 1,
    "no_fit": 0,
}


def dcg(relevances: List[int]) -> float:
    """Compute DCG using log2(rank + 1) discounting.

    relevances: list of integer relevance values in ranked order (rank 1 first)
    """
    total = 0.0
    for i, rel in enumerate(relevances, start=1):
        # numerator: (2^rel - 1) is a common graded relevance formulation
        gain = (2 ** rel - 1)
        denom = math.log2(i + 1)
        total += gain / denom
    return total


def idcg(relevances: List[int], k: int) -> float:
    """Compute ideal DCG (IDCG) for a list of relevance values up to cutoff k."""
    sorted_rels = sorted(relevances, reverse=True)
    return dcg(sorted_rels[:k])


def compute_ndcg_at_k(predicted_scores: List[float], relevances: List[int], k: int = 10) -> float:
    """Compute nDCG@k for a single job.

    predicted_scores and relevances must be aligned by candidate.
    """
    if len(relevances) == 0:
        return 0.0

    # Pair scores with relevances and sort by score descending
    paired = list(zip(predicted_scores, relevances))
    paired_sorted = sorted(paired, key=lambda x: x[0], reverse=True)
    ranked_rels = [int(r) for (_s, r) in paired_sorted][:k]

    actual_dcg = dcg(ranked_rels)
    ideal_dcg = idcg(relevances, k)

    if ideal_dcg <= 0:
        # No positive relevance in this job; cannot compute meaningful nDCG
        return None

    return actual_dcg / ideal_dcg


def validate_columns(df: pd.DataFrame) -> None:
    required = {"job_id", "candidate_id", "final_score", "expected_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def map_labels_to_relevance(series: pd.Series) -> pd.Series:
    """Map textual labels to integer relevance values. Unknown labels map to 0 with a warning."""
    def _map_label(x):
        if pd.isna(x):
            return 0
        x_str = str(x).strip()
        if x_str in RELEVANCE_MAP:
            return int(RELEVANCE_MAP[x_str])
        # unknown label
        return 0

    mapped = series.apply(_map_label).astype(int)
    return mapped


def evaluate(df: pd.DataFrame, k: int = 10):
    # Validate input
    validate_columns(df)

    # Ensure relevance values are integers based on mapping
    df = df.copy()
    df["relevance"] = map_labels_to_relevance(df["expected_label"])

    ndcg_scores = []
    skipped_few_labels = 0
    skipped_zero_idcg = 0

    groups = df.groupby("job_id")
    for job_id, group in groups:
        if len(group) < 2:
            skipped_few_labels += 1
            continue

        scores = group["final_score"].astype(float).tolist()
        relevances = group["relevance"].astype(int).tolist()

        ndcg = compute_ndcg_at_k(scores, relevances, k=k)
        if ndcg is None:
            skipped_zero_idcg += 1
            continue

        ndcg_scores.append(ndcg)

    num_evaluated = len(ndcg_scores)
    avg_ndcg = float(sum(ndcg_scores) / num_evaluated) if num_evaluated > 0 else 0.0

    return {
        "num_jobs_total": len(groups),
        "num_evaluated_jobs": num_evaluated,
        "skipped_few_labels": skipped_few_labels,
        "skipped_zero_idcg": skipped_zero_idcg,
        "average_ndcg_at_{}".format(k): avg_ndcg,
    }


def main(input_path: str = None):
    project_root = Path(__file__).parent.parent
    default_path = project_root / "data" / "processed" / "gold_eval_with_scores.csv"
    input_path = Path(input_path) if input_path else default_path

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        print("Run the script that generates `gold_eval_with_scores.csv` first.")
        sys.exit(1)

    print("Loading gold evaluation file:", input_path)
    df = pd.read_csv(input_path)
    print(f"  Loaded {len(df)} rows")

    results = evaluate(df, k=10)

    # Print clean evaluation report
    print("\n" + "=" * 40)
    print("Gold Evaluation (job -> candidate ranking)")
    print("=" * 40)
    print(f"Number of jobs in file: {results['num_jobs_total']}")
    print(f"Number of jobs evaluated: {results['num_evaluated_jobs']}")
    print(f"Jobs skipped (fewer than 2 labels): {results['skipped_few_labels']}")
    print(f"Jobs skipped (zero ideal DCG): {results['skipped_zero_idcg']}")
    avg_key = [k for k in results.keys() if k.startswith("average_ndcg")][0]
    print(f"Average nDCG@10: {results[avg_key]:.6f}")
    print("=" * 40)

    # Additionally report per-job nDCG@10 distribution (min/median/max)
    # Note: the list of per-job nDCG scores was collected during evaluation
    # and is available via the evaluate function return if needed. To avoid
    # changing the evaluate signature, recompute per-job scores here.
    # Load per-job nDCG scores again from the dataframe for reporting.
    ndcg_scores = []
    groups = df.groupby("job_id")
    for job_id, group in groups:
        if len(group) < 2:
            continue
        scores = group["final_score"].astype(float).tolist()
        relevances = map_labels_to_relevance(group["expected_label"]).astype(int).tolist()
        ndcg = compute_ndcg_at_k(scores, relevances, k=10)
        if ndcg is None:
            continue
        ndcg_scores.append(ndcg)

    # Safe handling when no jobs were evaluated
    if len(ndcg_scores) == 0:
        min_val = median_val = max_val = 0.0
    else:
        min_val = float(min(ndcg_scores))
        median_val = float(np.median(np.array(ndcg_scores)))
        max_val = float(max(ndcg_scores))

    print("\nnDCG@10 per job:")
    print(f"min:    {min_val:.4f}")
    print(f"median: {median_val:.4f}")
    print(f"max:    {max_val:.4f}")


if __name__ == "__main__":
    # Allow passing a custom path as the first argument
    arg_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(arg_path)
