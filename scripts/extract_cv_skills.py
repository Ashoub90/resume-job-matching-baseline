"""extract_cv_skills.py

Simple preview script to inspect rule-based extraction on the candidates CVs.

This script is intentionally lightweight and not a CLI framework. It:
- Loads the skills dictionary via `src.skills.loader.load_skills_dictionary()`
- Reads `data/raw/candidates.csv`
- Calls `extract_skills(cv_text, skills_dict)` for each CV
- Prints candidate_id and sorted extracted skills for manual inspection

Run from the repository root, e.g.:
    python scripts/extract_cv_skills.py
"""
from pathlib import Path
import sys
import csv

# Make project root importable so we can import the package modules in src/
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from src.skills.loader import load_skills_dictionary
from src.skills.rule_based_extraction import extract_skills


def _read_candidates(csv_path: Path):
    with csv_path.open("r", encoding="utf-8", newline='') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            # Ensure expected keys exist
            if 'candidate_id' in row and 'cv_text' in row:
                yield {'candidate_id': row['candidate_id'], 'cv_text': row['cv_text']}


def main():
    skills = load_skills_dictionary()
    candidates_csv = repo_root / 'data' / 'raw' / 'candidates.csv'
    
    if not candidates_csv.exists():
        print(f"Candidates file not found: {candidates_csv}")
        return

    print("Previewing rule-based extraction over a few CVs:\n")

    for cand in _read_candidates(candidates_csv):
        cid = cand['candidate_id']
        text = cand['cv_text'] or ""
        extracted = extract_skills(text, skills)
        extracted_list = sorted(extracted)
        if extracted_list:
            skills_str = ", ".join(extracted_list)
        else:
            skills_str = "(no skills matched)"

        print(f"Candidate: {cid}")
        print(f"Extracted skills: {skills_str}")
        print("-" * 40)

if __name__ == '__main__':
    main()
