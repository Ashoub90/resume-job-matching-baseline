"""evaluate_job_to_candidate_ranking.py

Evaluates job → candidate ranking quality using ground-truth labels.

Input files:
    data/processed/job_top_20_candidates.csv
    data/processed/silver_train_cartesian.csv

Metrics computed per job:
    Recall@5, Recall@10, Recall@20, MRR

Relevant candidates: expected_label ∈ {"medium", "strong"} (relevance ≥ 2)

Output:
    Console summary (no files written)

Run:
    python scripts/evaluate_job_to_candidate_ranking.py
"""
from pathlib import Path
import sys
import math

try:
    import pandas as pd
except ImportError:
    print("pandas not available. Please install pandas and try again.")
    sys.exit(1)


def compute_dcg(relevances, k=10):
    """Compute DCG@k: sum of (2^rel - 1) / log2(rank + 1) for rank <= k."""
    dcg = 0.0
    for rank, rel in enumerate(relevances[:k], start=1):
        if rel > 0:
            dcg += (2.0 ** rel - 1.0) / math.log2(rank + 1)
    return dcg

repo_root = Path(__file__).resolve().parents[1]
ranked_csv = repo_root / 'data' / 'processed' / 'job_top_20_candidates.csv'
ground_truth_csv = repo_root / 'data' / 'processed' / 'silver_train_cartesian.csv'

if not ranked_csv.exists():
    print(f"Input file not found: {ranked_csv}")
    sys.exit(2)
if not ground_truth_csv.exists():
    print(f"Input file not found: {ground_truth_csv}")
    sys.exit(3)

# Load files
ranked_df = pd.read_csv(ranked_csv)
ground_truth_df = pd.read_csv(ground_truth_csv)

# Map expected_label to relevance
label_to_relevance = {
    'strong': 3,
    'medium': 2,
    'weak': 1,
    'no_fit': 0
}
ground_truth_df['relevance'] = ground_truth_df['expected_label'].map(label_to_relevance)

# Join on (job_id, candidate_id)
merged = ranked_df.merge(
    ground_truth_df[['candidate_id', 'job_id', 'relevance']],
    on=['candidate_id', 'job_id'],
    how='left'
)

# Initialize metrics
recall_5_list = []
recall_10_list = []
recall_20_list = []
mrr_list = []
ndcg_10_list = []

# Compute metrics per job
for job_id in merged['job_id'].unique():
    job_data = merged[merged['job_id'] == job_id].sort_values('rank')
    
    # Check if there is at least one relevant candidate (relevance >= 2)
    relevant_indices = job_data[job_data['relevance'] >= 2].index
    has_relevant = len(relevant_indices) > 0
    
    # Recall@K: 1 if at least one relevant candidate in top K
    recall_5 = 1 if has_relevant and job_data[job_data['rank'] <= 5]['relevance'].max() >= 2 else 0
    recall_10 = 1 if has_relevant and job_data[job_data['rank'] <= 10]['relevance'].max() >= 2 else 0
    recall_20 = 1 if has_relevant and job_data[job_data['rank'] <= 20]['relevance'].max() >= 2 else 0
    
    # MRR: 1 / rank of first relevant, or 0
    first_relevant = job_data[job_data['relevance'] >= 2]
    mrr = 1.0 / first_relevant['rank'].iloc[0] if len(first_relevant) > 0 else 0.0
    
    # nDCG@10
    # DCG@10: use existing ranking (rank column)
    ranked_relevances = job_data['relevance'].values
    dcg_10 = compute_dcg(ranked_relevances, k=10)
    
    # IDCG@10: sort by relevance (descending) and compute DCG
    sorted_relevances = sorted(job_data['relevance'].values, reverse=True)
    idcg_10 = compute_dcg(sorted_relevances, k=10)
    
    # nDCG@10
    ndcg_10 = dcg_10 / idcg_10 if idcg_10 > 0 else 0.0
    
    recall_5_list.append(recall_5)
    recall_10_list.append(recall_10)
    recall_20_list.append(recall_20)
    mrr_list.append(mrr)
    ndcg_10_list.append(ndcg_10)

# Compute averages
num_jobs = len(recall_5_list)
avg_recall_5 = sum(recall_5_list) / num_jobs if num_jobs > 0 else 0.0
avg_recall_10 = sum(recall_10_list) / num_jobs if num_jobs > 0 else 0.0
avg_recall_20 = sum(recall_20_list) / num_jobs if num_jobs > 0 else 0.0
avg_mrr = sum(mrr_list) / num_jobs if num_jobs > 0 else 0.0
avg_ndcg_10 = sum(ndcg_10_list) / num_jobs if num_jobs > 0 else 0.0

# Print summary
print("=" * 60)
print("Job → Candidate Ranking Evaluation")
print("=" * 60)
print(f"Number of jobs evaluated: {num_jobs}")
print(f"Recall@5:  {avg_recall_5:.4f}")
print(f"Recall@10: {avg_recall_10:.4f}")
print(f"Recall@20: {avg_recall_20:.4f}")
print(f"MRR:       {avg_mrr:.4f}")
print(f"nDCG@10:   {avg_ndcg_10:.4f}")
print("=" * 60)
