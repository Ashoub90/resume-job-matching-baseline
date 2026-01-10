"""build_gold_eval_pairs.py

Builds a deterministic gold evaluation candidate set per job.

Input:
  data/processed/predictions_cartesian.csv  (must contain job_id,candidate_id,final_score)

For each job_id:
  - sort candidates by final_score desc
  - select top 10
  - select bottom 5
  - select 5 random from the middle (excluding top10 and bottom5)

Output:
  data/processed/gold_eval_pairs.csv
  Columns (exact): job_id,candidate_id
  Rows are shuffled deterministically before saving.

Deterministic behavior:
  - random seed is fixed and per-job sampling uses a seed derived from job_id

Run:
  python scripts/build_gold_eval_pairs.py
"""
from pathlib import Path
import sys
import hashlib

try:
    import pandas as pd
except ImportError:
    print("pandas not available. Please install pandas and try again.")
    sys.exit(1)

# Config
REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = REPO_ROOT / 'data' / 'processed' / 'predictions_cartesian.csv'
OUTPUT_CSV = REPO_ROOT / 'data' / 'processed' / 'gold_eval_pairs.csv'
TOP_K = 10
BOTTOM_K = 5
MIDDLE_SAMPLE = 5
GLOBAL_SEED = 42

if not INPUT_CSV.exists():
    print(f"Input file not found: {INPUT_CSV}")
    sys.exit(2)

# Load required columns only
df = pd.read_csv(INPUT_CSV, usecols=['job_id', 'candidate_id', 'final_score'])

rows = []
# Process per job
for job_id, group in df.groupby('job_id'):
    # Deterministic sort: final_score desc, tie-break candidate_id asc
    grp = group.sort_values(by=['final_score', 'candidate_id'], ascending=[False, True]).reset_index(drop=True)

    # Top K
    top = grp.head(TOP_K)
    # Bottom K (from the end)
    bottom = grp.tail(BOTTOM_K)

    # Middle: exclude indices present in top or bottom
    excl_idx = set(top.index.tolist()) | set(bottom.index.tolist())
    middle = grp.drop(index=list(excl_idx))

    # Deterministic per-job sample from middle using MD5-derived seed
    # Create integer seed from job_id + global seed
    md5 = hashlib.md5(job_id.encode('utf-8')).hexdigest()
    job_seed = GLOBAL_SEED + (int(md5[:8], 16) & 0x7FFFFFFF)

    n_middle = min(MIDDLE_SAMPLE, len(middle))
    if n_middle > 0:
        sampled_middle = middle.sample(n=n_middle, random_state=job_seed)
    else:
        sampled_middle = middle.iloc[0:0]

    # Collect unique (job_id, candidate_id) pairs
    selected = pd.concat([top, sampled_middle, bottom], ignore_index=True)
    selected = selected[['job_id', 'candidate_id']].drop_duplicates()

    for _, r in selected.iterrows():
        rows.append((r['job_id'], r['candidate_id']))

# Build final DataFrame and shuffle deterministically
out_df = pd.DataFrame(rows, columns=['job_id', 'candidate_id'])
out_df = out_df.drop_duplicates()
if len(out_df) > 0:
    out_df = out_df.sample(frac=1, random_state=GLOBAL_SEED).reset_index(drop=True)

# Save with exact columns and no index
out_df.to_csv(OUTPUT_CSV, index=False)
print(f"Wrote {len(out_df)} rows to {OUTPUT_CSV}")
