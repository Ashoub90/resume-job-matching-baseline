"""match_cv_to_job.py

Simple, explainable baseline matching system between CVs and jobs.

This script:
1. Loads the skills dictionary and extracts skills from all CVs and jobs
2. Reads expected matches (ground truth labels) from data/processed/expected_matches.csv
3. Computes a skill overlap score for each candidate-job pair
4. Maps the score to a predicted label (strong/medium/weak/no_fit)
5. Writes a CSV with candidate_id, job_id, score, predicted_label, expected_label

Matching logic:
    score = |cv_skills ∩ job_skills| / |job_skills|
    If job_skills is empty, score = 0.0

Label mapping:
    score >= 0.70 → strong
    score >= 0.40 → medium
    score >= 0.20 → weak
    score <  0.20 → no_fit

Run from the repository root, e.g.:
    python scripts/match_cv_to_job.py
"""
from pathlib import Path
import sys
import csv
from typing import Dict, Set, Tuple

# Make project root importable
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from src.skills.loader import load_skills_dictionary
from src.skills.rule_based_extraction import extract_skills


# Job family definitions with skill category weights
JOB_FAMILIES = {
    "data_analyst": {
        "Data & Databases": 3,
        "Analytics & BI": 3,
        "Programming": 2,
        "Machine Learning & AI": 1,
        "DevOps & Cloud": 1,
        "Tools & Platforms": 1,
        "Soft Skills": 1,
        "Security & Compliance": 1,
        "HR & Talent": 0,
    },
    "ml_engineer": {
        "Programming": 3,
        "Machine Learning & AI": 3,
        "Data & Databases": 2,
        "DevOps & Cloud": 2,
        "Analytics & BI": 1,
        "Tools & Platforms": 1,
        "Soft Skills": 1,
        "Security & Compliance": 1,
        "HR & Talent": 0,
    },
    "backend_engineer": {
        "Programming": 3,
        "DevOps & Cloud": 3,
        "Data & Databases": 2,
        "Tools & Platforms": 1,
        "Analytics & BI": 1,
        "Machine Learning & AI": 1,
        "Soft Skills": 1,
        "Security & Compliance": 1,
        "HR & Talent": 0,
    },
    "general_software": {
        "Programming": 3,
        "Data & Databases": 2,
        "DevOps & Cloud": 2,
        "Tools & Platforms": 1,
        "Analytics & BI": 1,
        "Machine Learning & AI": 1,
        "Soft Skills": 1,
        "Security & Compliance": 1,
        "HR & Talent": 0,
    },
    "hr_generalist": {
        "HR & Talent": 3,
        "Soft Skills": 3,
        "Tools & Platforms": 2,
        "Security & Compliance": 2,
        "Data & Databases": 1,
        "Analytics & BI": 1,
        "Programming": 0,
        "Machine Learning & AI": 0,
        "DevOps & Cloud": 0,
    },
}

'''

def _read_csv_dict(csv_path: Path) -> Dict[str, str]:
    """Read a CSV and return dict mapping first column to second column."""
    result = {}
    with csv_path.open("r", encoding="utf-8", newline='') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            # Assumes first column is the key, second is the value
            items = list(row.items())
            if len(items) >= 2:
                key = items[0][1]
                value = items[1][1]
                result[key] = value
    return result

'''
def _extract_all_cv_skills(candidates_csv: Path, skills_dict: Dict) -> Dict[str, Set[str]]:
    """Extract skills for all candidates.

    Returns: { candidate_id: set of skill names }
    """
    result = {}
    with candidates_csv.open("r", encoding="utf-8", newline='') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            candidate_id = row.get('candidate_id', '').strip()
            cv_text = row.get('cv_text', '')
            
            if candidate_id and cv_text:
                
                
                skills = extract_skills(cv_text, skills_dict)
                result[candidate_id] = skills
                
    return result



def _extract_all_job_skills(jobs_csv: Path, skills_dict: Dict) -> Dict[str, Set[str]]:
    """Extract skills for all jobs.

    Returns: { job_id: set of skill names }
    """
    result = {}
    with jobs_csv.open("r", encoding="utf-8", newline='') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            job_id = row.get('job_id', '').strip()
            job_description = row.get('job_description', '')
            # Also try job_title if job_description is missing
            if not job_description:
                job_description = row.get('job_title', '')
            if job_id and job_description:
                skills = extract_skills(job_description, skills_dict)
                result[job_id] = skills
    return result


def _read_expected_matches(expected_csv: Path) -> Dict[Tuple[str, str], str]:
    """Read expected matches and return { (candidate_id, job_id): expected_label }."""
    result = {}
    with expected_csv.open("r", encoding="utf-8", newline='') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            candidate_id = row.get('candidate_id', '').strip()
            job_id = row.get('job_id', '').strip()
            expected_fit = row.get('expected_label', '')
            if candidate_id and job_id:
                result[(candidate_id, job_id)] = expected_fit
    return result


def _build_skill_to_category_map(skills_dict: Dict) -> Dict[str, str]:
    """Build a reverse map from skill name to category.

    Returns: { skill_name: category_name }
    """
    result = {}
    for category, skills in (skills_dict or {}).items():
        if isinstance(skills, dict):
            for skill_name in skills.keys():
                result[skill_name] = category
    return result


def _detect_job_family(job_title: str) -> str:
    """Detect job family from job title using keyword matching.

    Returns one of: data_analyst, ml_engineer, backend_engineer, general_software, hr_generalist
    Defaults to general_software if no keywords match.
    """
    title_lower = job_title.lower()

    if any(kw in title_lower for kw in ["data analyst", "analytics", "bi"]):
        return "data_analyst"
    elif any(kw in title_lower for kw in ["machine learning", "ml", "ai"]):
        return "ml_engineer"
    elif any(kw in title_lower for kw in ["backend", "server", "api"]):
        return "backend_engineer"
    elif any(kw in title_lower for kw in ["hr", "human resources", "recruit", "talent", "people", "payroll"]):
        return "hr_generalist"
    elif any(kw in title_lower for kw in ["software", "developer", "engineer"]):
        return "general_software"
    else:
        # Default
        return "general_software"


def _compute_weighted_score(
    cv_skills: Set[str],
    job_skills: Set[str],
    job_family: str,
    skill_to_category: Dict[str, str]
) -> float:
    """Compute weighted skill overlap score based on job family.

    score = sum(category_weight for matched skills)
            /
            sum(category_weight for all job skills)

    If job_skills is empty, return 0.0.
    If a category is not in the weight table, default weight = 1.
    """
    if not job_skills:
        return 0.0

    weights = JOB_FAMILIES.get(job_family, JOB_FAMILIES["general_software"])

    numerator = 0.0
    denominator = 0.0

    # Sum weights for matched skills (in both CV and job)
    for skill in cv_skills & job_skills:
        category = skill_to_category.get(skill, "")
        weight = weights.get(category, 1)
        numerator += weight

    # Sum weights for all job skills
    for skill in job_skills:
        category = skill_to_category.get(skill, "")
        weight = weights.get(category, 1)
        denominator += weight

    if denominator == 0:
        return 0.0

    score = numerator / denominator
    return min(score, 1.0)  # Clamp to [0, 1]


def _score_to_label(score: float) -> str:
    """Map score to predicted label."""
    if score >= 0.40:
        return "strong"
    elif score >= 0.25:
        return "medium"
    elif score >= 0.10:
        return "weak"
    else:
        return "no_fit"


def main():
    skills_dict = load_skills_dictionary()
    skill_to_category = _build_skill_to_category_map(skills_dict)

    candidates_csv = repo_root / 'data' / 'raw' / 'candidates.csv'
    jobs_csv = repo_root / 'data' / 'raw' / 'jobs.csv'
    expected_csv = repo_root / 'data' / 'processed' / 'silver_train_cartesian.csv'
    output_csv = repo_root / 'data' / 'processed' / 'predictions.csv'

    # Validate inputs exist
    if not candidates_csv.exists():
        print(f"Candidates file not found: {candidates_csv}")
        return
    if not jobs_csv.exists():
        print(f"Jobs file not found: {jobs_csv}")
        return
    if not expected_csv.exists():
        print(f"Expected matches file not found: {expected_csv}")
        return

    print("Extracting CV skills...")
    cv_skills = _extract_all_cv_skills(candidates_csv, skills_dict)
    print(f"  Extracted skills for {len(cv_skills)} candidates")

    print("Extracting job skills...")
    job_skills = _extract_all_job_skills(jobs_csv, skills_dict)
    print(f"  Extracted skills for {len(job_skills)} jobs")

    # Load job titles for job family detection
    job_titles = {}
    with jobs_csv.open("r", encoding="utf-8", newline='') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            job_id = row.get('job_id', '').strip()
            job_title = row.get('job_title', '')
            if job_id:
                job_titles[job_id] = job_title

    print("Loading expected matches...")
    expected_matches = _read_expected_matches(expected_csv)
    print(f"  Loaded {len(expected_matches)} expected matches")

    print("Computing weighted scores and generating predictions...")

    # Prepare output rows
    output_rows = []
    for (candidate_id, job_id), expected_label in expected_matches.items():
        c_skills = cv_skills.get(candidate_id, set())
        j_skills = job_skills.get(job_id, set())
        title = job_titles.get(job_id, "")

        # Detect job family and compute weighted score
        job_family = _detect_job_family(title)
        score = _compute_weighted_score(c_skills, j_skills, job_family, skill_to_category)
        predicted_label = _score_to_label(score)

        output_rows.append({
            'candidate_id': candidate_id,
            'job_id': job_id,
            'score': round(score, 3),
            'predicted_label': predicted_label,
            'expected_label': expected_label
        })

    # Write output CSV
    print(f"Writing {len(output_rows)} predictions to {output_csv}...")
    with output_csv.open("w", encoding="utf-8", newline='') as fh:
        fieldnames = ['candidate_id', 'job_id', 'score', 'predicted_label', 'expected_label']
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print("Done. Results saved to data/processed/predictions.csv")


if __name__ == '__main__':
    main()
