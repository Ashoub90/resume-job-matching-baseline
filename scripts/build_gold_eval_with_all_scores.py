"""
Build gold evaluation file by joining gold labels with all available model scores.

Input files:
- data/processed/gold_eval_with_scores.csv
- data/processed/predictions_cartesian.csv

Output:
- data/processed/gold_eval_with_all_scores.csv

Rules:
- Inner join on (job_id, candidate_id) -> drops non-matching rows
- Do NOT modify labels or recompute scores
- Print how many rows were merged
"""

from pathlib import Path
import sys
import pandas as pd


def validate_columns(df: pd.DataFrame, required: list, name: str):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {name}: {missing}")


def main(gold_path: str = None, preds_path: str = None):
    project_root = Path(__file__).parent.parent
    gold_path = Path(gold_path) if gold_path else project_root / "data" / "processed" / "gold_eval_with_scores.csv"
    preds_path = Path(preds_path) if preds_path else project_root / "data" / "processed" / "predictions_cartesian.csv"
    out_path = project_root / "data" / "processed" / "gold_eval_with_all_scores.csv"

    if not gold_path.exists():
        print(f"Gold file not found: {gold_path}")
        sys.exit(1)
    if not preds_path.exists():
        print(f"Predictions file not found: {preds_path}")
        sys.exit(1)

    print(f"Loading gold labels from: {gold_path}")
    gold = pd.read_csv(gold_path)
    print(f"  {len(gold)} rows loaded")

    print(f"Loading predictions from: {preds_path}")
    preds = pd.read_csv(preds_path)
    print(f"  {len(preds)} rows loaded")

    # Validate expected columns
    validate_columns(gold, ["job_id", "candidate_id", "expected_label"], "gold_eval_with_scores.csv")
    # Drop any existing final_score in gold to ensure we only use scores from predictions
    if "final_score" in gold.columns:
        print("Dropping 'final_score' column from gold to ensure scores come from predictions")
        gold = gold.drop(columns=["final_score"])
    validate_columns(preds, ["job_id", "candidate_id", "rule_based_score", "semantic_similarity", "final_score"], "predictions_cartesian.csv")

    # Inner join on (job_id, candidate_id)
    merged = gold.merge(preds, on=["job_id", "candidate_id"], how="inner")

    # Select and order columns as requested
    out_cols = ["job_id", "candidate_id", "expected_label", "rule_based_score", "semantic_similarity", "final_score"]
    # Some safety: ensure columns exist after merge
    missing_after = [c for c in out_cols if c not in merged.columns]
    if missing_after:
        raise ValueError(f"Missing expected output columns after merge: {missing_after}")

    result = merged[out_cols].copy()

    # Print merged row count
    print(f"Merged rows: {len(result)}")

    # Save result
    result.to_csv(out_path, index=False)
    print(f"Saved output to: {out_path}")


if __name__ == "__main__":
    # Accept optional command-line args for custom paths
    gold_arg = sys.argv[1] if len(sys.argv) > 1 else None
    preds_arg = sys.argv[2] if len(sys.argv) > 2 else None
    main(gold_arg, preds_arg)
