from pathlib import Path
import csv
import sys

repo_root = Path(__file__).resolve().parents[1]
inp = repo_root / 'data' / 'processed' / 'silver_train_cartesian.csv'
out = repo_root / 'data' / 'processed' / 'predictions_cartesian.csv'

if not inp.exists():
    print('MISSING_INPUT')
    sys.exit(2)
if not out.exists():
    print('MISSING_OUTPUT')
    sys.exit(3)

with inp.open('r', encoding='utf-8', newline='') as f:
    r = csv.DictReader(f)
    in_pairs = [(row.get('candidate_id','').strip(), row.get('job_id','').strip()) for row in r]

with out.open('r', encoding='utf-8', newline='') as f:
    r = csv.DictReader(f)
    out_pairs = [(row.get('candidate_id','').strip(), row.get('job_id','').strip()) for row in r]

with out.open('r', encoding='utf-8') as f:
    header = f.readline()

bad_scores = []
with out.open('r', encoding='utf-8', newline='') as f:
    r = csv.DictReader(f)
    for row in r:
        try:
            v = float(row.get('final_score',''))
            if v < 0 or v > 1:
                bad_scores.append((row.get('candidate_id'), row.get('job_id'), v))
        except Exception:
            bad_scores.append((row.get('candidate_id'), row.get('job_id'), 'nan'))

print('input_count', len(in_pairs))
print('output_count', len(out_pairs))
print('pairs_equal', in_pairs == out_pairs)
print('expected_label_in_output', 'expected_label' in header)
print('bad_final_scores_count', len(bad_scores))
if bad_scores:
    print('bad_examples', bad_scores[:5])

ok = (len(in_pairs) == len(out_pairs) and in_pairs == out_pairs and ('expected_label' not in header) and len(bad_scores) == 0)
print('RESULT', 'PASS' if ok else 'FAIL')
