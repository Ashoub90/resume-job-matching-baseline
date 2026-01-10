"""retrieve_top_candidates_per_job.py

Retrieves the top 20 candidates per job based on final_score.

Input:
    data/processed/predictions_cartesian.csv

Output:
    data/processed/job_top_20_candidates.csv
    Columns: job_id, candidate_id, final_score, rank

Ranking resets per job, with deterministic tie-breaking by candidate_id.

Run:
    python scripts/retrieve_top_candidates_per_job.py
"""
from pathlib import Path
import sys

try:
    import pandas as pd
except ImportError:
    print("pandas not available. Please install pandas and try again.")
    sys.exit(1)

repo_root = Path(__file__).resolve().parents[1]
input_csv = repo_root / 'data' / 'processed' / 'predictions_cartesian.csv'
output_csv = repo_root / 'data' / 'processed' / 'job_top_20_candidates.csv'

if not input_csv.exists():
    print(f"Input file not found: {input_csv}")
    sys.exit(2)

# Load predictions
df = pd.read_csv(input_csv)

# Sort by job_id, final_score (descending), and candidate_id (ascending for deterministic tie-breaking)
df = df.sort_values(by=['job_id', 'final_score', 'candidate_id'], ascending=[True, False, True])

# Assign rank per job
df['rank'] = df.groupby('job_id').cumcount() + 1

# Filter to top 20 per job
df_top20 = df[df['rank'] <= 20].copy()

# Select and reorder columns
output_df = df_top20[['job_id', 'candidate_id', 'final_score', 'rank']].copy()

# Write output
output_df.to_csv(output_csv, index=False)

print(f"Wrote {len(output_df)} rows to {output_csv}")
print(f"Unique jobs: {output_df['job_id'].nunique()}")
