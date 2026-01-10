"""
Evaluate job -> candidate ranking performance separately for rule-based,
semantic, and hybrid (final) scores using human-labeled gold judgments.

Run as:
  python -m scripts.evaluate_job_to_candidate_ranking_by_method

Input (default): data/processed/gold_eval_with_all_scores.csv

This script computes nDCG@10 per job (log2 discounting) and reports the
average nDCG@10 for each scoring method. Jobs with fewer than 2 labeled
candidates or zero ideal DCG are skipped.
"""

from pathlib import Path
import sys
import math
from typing import List, Optional

import numpy as np
import pandas as pd


RELEVANCE_MAP = {
    "strong": 3,
    "medium": 2,
    "weak": 1,
    "no_fit": 0,
}


def dcg(relevances: List[int]) -> float:
    total = 0.0
    for i, rel in enumerate(relevances, start=1):
        gain = (2 ** rel - 1)
        denom = math.log2(i + 1)
        total += gain / denom
    return total


def idcg(relevances: List[int], k: int) -> float:
    sorted_rels = sorted(relevances, reverse=True)
    return dcg(sorted_rels[:k])


def compute_ndcg_at_k(predicted_scores: List[float], relevances: List[int], k: int = 10) -> Optional[float]:
    if len(relevances) == 0:
        return None

    paired = list(zip(predicted_scores, relevances))
    paired_sorted = sorted(paired, key=lambda x: x[0], reverse=True)
    ranked_rels = [int(r) for (_s, r) in paired_sorted][:k]

    actual_dcg = dcg(ranked_rels)
    ideal = idcg(relevances, k)
    if ideal <= 0:
        return None
    return actual_dcg / ideal


def validate_columns(df: pd.DataFrame, required: List[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {name}: {missing}")


def map_labels_to_relevance(series: pd.Series) -> pd.Series:
    def _map(x):
        if pd.isna(x):
            return 0
        s = str(x).strip()
        return int(RELEVANCE_MAP.get(s, 0))

    return series.apply(_map).astype(int)


def evaluate_by_method(df: pd.DataFrame, k: int = 10):
    required = [
        "job_id",
        "candidate_id",
        "expected_label",
        "rule_based_score",
        "semantic_similarity",
        "final_score",
    ]
    validate_columns(df, required, "gold_eval_with_all_scores.csv")

    df = df.copy()
    df["relevance"] = map_labels_to_relevance(df["expected_label"])

    # Prepare lists for per-method scores
    ndcg_rule = []
    ndcg_sem = []
    ndcg_final = []

    groups = df.groupby("job_id")
    for job_id, group in groups:
        if len(group) < 2:
            continue

        relevances = group["relevance"].astype(int).tolist()
        # Skip jobs with zero ideal DCG
        if idcg(relevances, k) <= 0:
            continue

        # Rule-based
        scores_rule = group["rule_based_score"].astype(float).tolist()
        val_rule = compute_ndcg_at_k(scores_rule, relevances, k=k)
        if val_rule is not None:
            ndcg_rule.append(val_rule)

        # Semantic-only
        scores_sem = group["semantic_similarity"].astype(float).tolist()
        val_sem = compute_ndcg_at_k(scores_sem, relevances, k=k)
        if val_sem is not None:
            ndcg_sem.append(val_sem)

        # Hybrid / final
        scores_final = group["final_score"].astype(float).tolist()
        val_final = compute_ndcg_at_k(scores_final, relevances, k=k)
        if val_final is not None:
            ndcg_final.append(val_final)

    def avg(lst: List[float]) -> float:
        return float(np.mean(lst)) if len(lst) > 0 else 0.0

    # Number of jobs evaluated: use intersection where at least one method was computed?
    # Per requirements, report Jobs evaluated: number of jobs considered (those with >=2 labels and non-zero IDCG)
    num_jobs_total = len(groups)
    # Count jobs actually evaluated (based on filtering rules)
    num_evaluated_jobs = len(ndcg_final) if len(ndcg_final) >= len(ndcg_rule) and len(ndcg_final) >= len(ndcg_sem) else max(len(ndcg_rule), len(ndcg_sem), len(ndcg_final))

    return {
        "jobs_in_file": num_jobs_total,
        "jobs_evaluated": num_evaluated_jobs,
        "rule_avg": avg(ndcg_rule),
        "sem_avg": avg(ndcg_sem),
        "final_avg": avg(ndcg_final),
    }


def main():
    project_root = Path(__file__).parent.parent
    default_path = project_root / "data" / "processed" / "gold_eval_with_all_scores.csv"

    input_path = default_path
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        sys.exit(1)

    df = pd.read_csv(input_path)

    results = evaluate_by_method(df, k=10)

    # Print the requested report
    print("=" * 60)
    print("Job → Candidate Ranking (Baseline Comparison)")
    print("=" * 60)
    print(f"Jobs evaluated: {results['jobs_evaluated']}")
    print("")
    print("nDCG@10 by scoring method:")
    print(f"- Rule-based only:    {results['rule_avg']:.6f}")
    print(f"- Semantic-only:      {results['sem_avg']:.6f}")
    print(f"- Hybrid (final):     {results['final_avg']:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
