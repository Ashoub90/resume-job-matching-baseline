"""match_cv_to_job_cartesian.py

Scoring pipeline for the full cartesian silver training file.

This reuses the exact scoring methodology from the semantic scoring script
(rule-based weighted score + semantic similarity + final score combination)
but reads pairs from `data/processed/silver_train_cartesian.csv` and writes
`data/processed/predictions_cartesian.csv` with the required columns.

Constraints respected:
- Do not use `expected_label` in any computation
- Do not change skill dictionaries, weights, thresholds, or scoring logic
- Output row count equals input row count; candidate/job pairs preserved

Run:
    python scripts/match_cv_to_job_cartesian.py
"""
from pathlib import Path
import sys
import csv
from typing import Dict, Set, Tuple

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
except Exception:
    SentenceTransformer = None

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from src.skills.loader import load_skills_dictionary
from src.skills.rule_based_extraction import extract_skills

# Keep same coefficients and helpers as original script
ALPHA = 0.6

# JOB_FAMILIES and helper functions copied unchanged
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


def _build_skill_to_category_map(skills_dict: Dict) -> Dict[str, str]:
    result = {}
    for category, skills in (skills_dict or {}).items():
        if isinstance(skills, dict):
            for skill_name in skills.keys():
                result[skill_name] = category
    return result


def _detect_job_family(job_title: str) -> str:
    title_lower = (job_title or "").lower()
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
        return "general_software"


def _compute_weighted_score(cv_skills: Set[str], job_skills: Set[str], job_family: str, skill_to_category: Dict[str, str]) -> float:
    if not job_skills:
        return 0.0
    weights = JOB_FAMILIES.get(job_family, JOB_FAMILIES["general_software"])
    numerator = 0.0
    denominator = 0.0
    for skill in cv_skills & job_skills:
        category = skill_to_category.get(skill, "")
        weight = weights.get(category, 1)
        numerator += weight
    for skill in job_skills:
        category = skill_to_category.get(skill, "")
        weight = weights.get(category, 1)
        denominator += weight
    if denominator == 0:
        return 0.0
    return min(numerator / denominator, 1.0)


def _score_to_label(score: float) -> str:
    if score >= 0.70:
        return "strong"
    elif score >= 0.40:
        return "medium"
    elif score >= 0.20:
        return "weak"
    else:
        return "no_fit"


def main():
    if SentenceTransformer is None:
        print("sentence-transformers not available. Please install requirements and try again.")
        return

    skills_dict = load_skills_dictionary()
    skill_to_category = _build_skill_to_category_map(skills_dict)

    candidates_csv = repo_root / 'data' / 'raw' / 'candidates.csv'
    jobs_csv = repo_root / 'data' / 'raw' / 'jobs.csv'
    cartesian_csv = repo_root / 'data' / 'processed' / 'silver_train_cartesian.csv'
    baseline_predictions = repo_root / 'data' / 'processed' / 'predictions.csv'
    output_csv = repo_root / 'data' / 'processed' / 'predictions_cartesian.csv'

    # Basic checks
    if not candidates_csv.exists() or not jobs_csv.exists() or not cartesian_csv.exists():
        print("Ensure candidates, jobs, and silver_train_cartesian CSV files exist in data/raw and data/processed respectively.")
        return

    # Load texts
    candidates_text = {}
    with candidates_csv.open("r", encoding="utf-8", newline='') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            cid = row.get('candidate_id', '').strip()
            txt = row.get('cv_text', '')
            if cid:
                candidates_text[cid] = txt or ""

    jobs_text = {}
    job_titles = {}
    with jobs_csv.open("r", encoding="utf-8", newline='') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            jid = row.get('job_id', '').strip()
            title = row.get('job_title', '')
            desc = row.get('job_description', '')
            if not desc:
                desc = title
            if jid:
                jobs_text[jid] = desc or ""
                job_titles[jid] = title or ""

    # Read cartesian pairs (do NOT use expected_label for computation)
    pairs = []  # list of (cid, jid)
    with cartesian_csv.open('r', encoding='utf-8', newline='') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            cid = row.get('candidate_id', '').strip()
            jid = row.get('job_id', '').strip()
            if cid and jid:
                pairs.append((cid, jid))

    # Load or compute baseline rule-based scores for the exact pairs
    baseline_scores: Dict[Tuple[str, str], float] = {}
    baseline_labels: Dict[Tuple[str, str], str] = {}
    if baseline_predictions.exists():
        with baseline_predictions.open('r', encoding='utf-8', newline='') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                key = (row.get('candidate_id','').strip(), row.get('job_id','').strip())
                try:
                    baseline_scores[key] = float(row.get('score', row.get('rule_based_score','0')) or 0.0)
                except Exception:
                    baseline_scores[key] = 0.0
                baseline_labels[key] = row.get('predicted_label', '')
        # Note: baseline may contain extra pairs; we'll only use keys for our pairs
    else:
        # Compute baseline on-the-fly for required pairs
        cv_skills = {}
        with candidates_csv.open('r', encoding='utf-8', newline='') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                cid = row.get('candidate_id','').strip()
                txt = row.get('cv_text','')
                if cid:
                    cv_skills[cid] = extract_skills(txt or "", skills_dict)
        job_skills = {}
        with jobs_csv.open('r', encoding='utf-8', newline='') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                jid = row.get('job_id','').strip()
                desc = row.get('job_description','')
                if not desc:
                    desc = row.get('job_title','')
                if jid:
                    job_skills[jid] = extract_skills(desc or "", skills_dict)

        for cid, jid in pairs:
            c_sk = cv_skills.get(cid, set())
            j_sk = job_skills.get(jid, set())
            family = _detect_job_family(job_titles.get(jid, ""))
            score = _compute_weighted_score(c_sk, j_sk, family, skill_to_category)
            baseline_scores[(cid, jid)] = score
            baseline_labels[(cid, jid)] = _score_to_label(score)

    # Unique ids to embed
    unique_cids = sorted({cid for cid, _ in pairs})
    unique_jids = sorted({jid for _, jid in pairs})

    # Build embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    cid_to_emb = {}
    batch_cands = [candidates_text.get(cid, "") for cid in unique_cids]
    if batch_cands:
        cand_embs = model.encode(batch_cands, convert_to_numpy=True, show_progress_bar=False)
        for i, cid in enumerate(unique_cids):
            cid_to_emb[cid] = cand_embs[i]
    jid_to_emb = {}
    batch_jobs = [jobs_text.get(jid, "") for jid in unique_jids]
    if batch_jobs:
        job_embs = model.encode(batch_jobs, convert_to_numpy=True, show_progress_bar=False)
        for i, jid in enumerate(unique_jids):
            jid_to_emb[jid] = job_embs[i]

    import numpy as _np
    def _cosine(a: 'np.ndarray', b: 'np.ndarray') -> float:
        if a is None or b is None:
            return 0.0
        na = _np.linalg.norm(a)
        nb = _np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        sim = float(_np.dot(a, b) / (na * nb))
        return max(min((sim + 1.0) / 2.0, 1.0), 0.0)

    # Compute outputs for all pairs
    output_rows = []
    for cid, jid in pairs:
        rule_score = baseline_scores.get((cid, jid), 0.0)
        emb_c = cid_to_emb.get(cid)
        emb_j = jid_to_emb.get(jid)
        sem_sim = _cosine(emb_c, emb_j)

        # Follow identical final_score logic
        if rule_score > 0:
            final_score = ALPHA * rule_score + (1 - ALPHA) * sem_sim
        
        else:
            final_score = 0.20 * sem_sim

        # Clamp final_score to [0,1]
        final_score = max(0.0, min(1.0, float(final_score)))

        output_rows.append({
            'candidate_id': cid,
            'job_id': jid,
            'rule_based_score': round(float(rule_score), 3),
            'semantic_similarity': round(float(sem_sim), 3),
            'final_score': round(float(final_score), 3),
        })

    # Validation: ensure same number of rows
    if len(output_rows) != len(pairs):
        print(f"ERROR: output row count {len(output_rows)} != input pairs {len(pairs)}")
        return

    # Write CSV
    with output_csv.open('w', encoding='utf-8', newline='') as fh:
        fieldnames = ['candidate_id', 'job_id', 'rule_based_score', 'semantic_similarity', 'final_score']
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Wrote {len(output_rows)} rows to {output_csv}")


if __name__ == '__main__':
    main()
