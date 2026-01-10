"""
Ranking-based evaluation for CV-to-job matching.

Metrics:
- Recall@1 and Recall@3: whether a relevant job (relevance >= 2) appears in top K
- Mean Reciprocal Rank (MRR): position of first relevant job
"""

import pandas as pd
import numpy as np

def relevance_mapping(label):
    """Map expected_label to numeric relevance."""
    mapping = {
        'strong': 3,
        'medium': 2,
        'weak': 1,
        'no_fit': 0
    }
    return mapping.get(label, 0)


def evaluate_ranking(predictions_file):
    """
    Evaluate ranking performance using Recall@K and MRR.
    
    Parameters:
    -----------
    predictions_file : str
        Path to predictions_with_semantic.csv
    """
    # Load data
    df = pd.read_csv(predictions_file)
    
    # Create numeric relevance from expected_label
    df['relevance'] = df['expected_label'].apply(relevance_mapping)
    
    print("=" * 70)
    print("RANKING-BASED EVALUATION")
    print("=" * 70)
    print(f"\nDataset: {len(df)} predictions")
    print(f"Unique candidates: {df['candidate_id'].nunique()}")
    print(f"Unique jobs: {df['job_id'].nunique()}")
    
    # Group by candidate and evaluate
    candidates = df['candidate_id'].unique()
    recall_at_1_scores = []
    recall_at_3_scores = []
    mrr_scores = []
    
    for candidate_id in candidates:
        candidate_df = df[df['candidate_id'] == candidate_id].copy()
        
        # Sort by final_score descending (ranking)
        candidate_df = candidate_df.sort_values('final_score', ascending=False).reset_index(drop=True)
        
        # Recall@1: is there a relevant job in top 1?
        top_1_relevances = candidate_df.head(1)['relevance'].values
        recall_at_1 = 1 if len(top_1_relevances) > 0 and top_1_relevances[0] >= 2 else 0
        recall_at_1_scores.append(recall_at_1)
        
        # Recall@3: is there a relevant job in top 3?
        top_3_relevances = candidate_df.head(3)['relevance'].values
        recall_at_3 = 1 if (top_3_relevances >= 2).any() else 0
        recall_at_3_scores.append(recall_at_3)
        
        # Mean Reciprocal Rank (MRR): rank of first relevant job
        relevant_jobs = candidate_df[candidate_df['relevance'] >= 2]
        if len(relevant_jobs) > 0:
            # Get position in ranking (0-indexed, convert to 1-indexed for rank)
            first_relevant_rank = relevant_jobs.index[0] + 1
            mrr_score = 1.0 / first_relevant_rank
        else:
            mrr_score = 0.0
        mrr_scores.append(mrr_score)
    
    # Compute averages
    avg_recall_at_1 = np.mean(recall_at_1_scores)
    avg_recall_at_3 = np.mean(recall_at_3_scores)
    avg_mrr = np.mean(mrr_scores)
    
    # Print results
    print("\n" + "-" * 70)
    print("RESULTS")
    print("-" * 70)
    print(f"\nRecall@1:     {avg_recall_at_1:.4f} ({int(sum(recall_at_1_scores))}/{len(recall_at_1_scores)} candidates)")
    print(f"Recall@3:     {avg_recall_at_3:.4f} ({int(sum(recall_at_3_scores))}/{len(recall_at_3_scores)} candidates)")
    print(f"MRR:          {avg_mrr:.4f}")
    print("\n" + "=" * 70)
    
    return {
        'recall_at_1': avg_recall_at_1,
        'recall_at_3': avg_recall_at_3,
        'mrr': avg_mrr,
        'num_candidates': len(candidates)
    }


if __name__ == '__main__':
    predictions_file = r'data\processed\predictions_with_semantic.csv'
    results = evaluate_ranking(predictions_file)
