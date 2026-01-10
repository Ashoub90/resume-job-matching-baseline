"""
Build a gold evaluation file by joining human-labeled gold pairs with model scores.

This script combines:
- Predictions from the cartesian matching approach (with final scores)
- Human-labeled gold evaluation pairs

The result is used for evaluating model performance against human labels.
"""

import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_data(predictions_path: str, gold_pairs_path: str) -> tuple:
    """Load the predictions and gold evaluation pairs."""
    print(f"Loading predictions from: {predictions_path}")
    predictions = pd.read_csv(predictions_path)
    print(f"  Loaded {len(predictions)} predictions")
    
    print(f"Loading gold pairs from: {gold_pairs_path}")
    gold_pairs = pd.read_csv(gold_pairs_path)
    print(f"  Loaded {len(gold_pairs)} gold pairs")
    
    return predictions, gold_pairs


def validate_data(predictions: pd.DataFrame, gold_pairs: pd.DataFrame) -> None:
    """Validate that required columns exist."""
    required_pred_cols = ['candidate_id', 'job_id', 'final_score']
    required_gold_cols = ['job_id', 'candidate_id', 'expected_label']
    
    missing_pred = [col for col in required_pred_cols if col not in predictions.columns]
    missing_gold = [col for col in required_gold_cols if col not in gold_pairs.columns]
    
    if missing_pred:
        raise ValueError(f"Missing columns in predictions: {missing_pred}")
    if missing_gold:
        raise ValueError(f"Missing columns in gold pairs: {missing_gold}")
    
    print("✓ All required columns are present")


def merge_data(predictions: pd.DataFrame, gold_pairs: pd.DataFrame, 
               gold_pairs_count: int) -> pd.DataFrame:
    """Merge predictions with gold pairs and validate row count."""
    print("\nMerging predictions with gold pairs on (job_id, candidate_id)...")
    
    merged = predictions.merge(
        gold_pairs,
        on=['job_id', 'candidate_id'],
        how='inner'
    )
    
    print(f"  Merged result: {len(merged)} rows")
    
    # Assert that the number of rows matches the original gold pairs
    assert len(merged) == gold_pairs_count, (
        f"Row count mismatch: merged has {len(merged)} rows but "
        f"gold_pairs has {gold_pairs_count} rows"
    )
    print(f"✓ Row count validation passed: {len(merged)} == {gold_pairs_count}")
    
    return merged


def select_columns(merged: pd.DataFrame) -> pd.DataFrame:
    """Select and reorder the required columns."""
    required_cols = ['job_id', 'candidate_id', 'final_score', 'expected_label']
    result = merged[required_cols].copy()
    print(f"\n✓ Selected columns: {required_cols}")
    return result


def save_result(result: pd.DataFrame, output_path: str) -> None:
    """Save the result to CSV."""
    print(f"\nSaving result to: {output_path}")
    result.to_csv(output_path, index=False)
    print(f"✓ Saved {len(result)} rows to {output_path}")


def main():
    """Main execution function."""
    # Define paths
    project_root = Path(__file__).parent.parent
    predictions_path = project_root / "data" / "processed" / "predictions_cartesian.csv"
    gold_pairs_path = project_root / "data" / "processed" / "gold_eval_pairs_reduced.csv"
    output_path = project_root / "data" / "processed" / "gold_eval_with_scores.csv"
    
    # Validate paths exist
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")
    if not gold_pairs_path.exists():
        raise FileNotFoundError(f"Gold pairs file not found: {gold_pairs_path}")
    
    print("=" * 70)
    print("Building Gold Evaluation Set with Scores")
    print("=" * 70)
    
    # Load data
    predictions, gold_pairs = load_data(str(predictions_path), str(gold_pairs_path))
    
    # Validate structure
    validate_data(predictions, gold_pairs)
    
    # Merge and validate
    gold_pairs_count = len(gold_pairs)
    merged = merge_data(predictions, gold_pairs, gold_pairs_count)
    
    # Select columns
    result = select_columns(merged)
    
    # Save result
    save_result(result, str(output_path))
    
    print("=" * 70)
    print("✓ Pipeline completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
