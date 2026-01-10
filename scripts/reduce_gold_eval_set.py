"""reduce_gold_eval_set.py

Reduces the gold evaluation set to the first 10 job_ids (alphabetical).

Input:
  data/processed/gold_eval_pairs.csv

Output:
  data/processed/gold_eval_pairs_reduced.csv

Behavior:
  - Select first 10 job_ids sorted alphabetically
  - Keep all rows belonging to those jobs
  - Preserve original columns and row order
  - Do not modify/generate labels
  - Print final row count and number of jobs selected

Run:
  python scripts/reduce_gold_eval_set.py
"""
from pathlib import Path
import sys

try:
    import pandas as pd
except ImportError:
    print("pandas not available. Please install pandas and try again.")
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT = REPO_ROOT / 'data' / 'processed' / 'gold_eval_pairs.csv'
OUTPUT = REPO_ROOT / 'data' / 'processed' / 'gold_eval_pairs_reduced.csv'
NUM_JOBS = 10

if not INPUT.exists():
    print(f"Input file not found: {INPUT}")
    sys.exit(2)

# Read preserving original order
df = pd.read_csv(INPUT)

# Determine first 10 job_ids alphabetically
unique_jobs = sorted(df['job_id'].dropna().unique())
selected_jobs = unique_jobs[:NUM_JOBS]

# Keep all rows for selected jobs, preserving original order
reduced = df[df['job_id'].isin(selected_jobs)].copy()

# Save with original columns preserved
reduced.to_csv(OUTPUT, index=False)

print(f"Selected jobs: {len(selected_jobs)}")
print(f"Output rows: {len(reduced)}")
