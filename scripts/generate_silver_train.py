"""
Generate silver training data: full Cartesian product of candidates × jobs.

Labels follow a human-like rubric:
- strong: clear role alignment, core skills match
- medium: transferable skills overlap, missing domain/seniority/1-2 requirements
- weak: some surface overlap, role expectations differ
- no_fit: different profession, no meaningful overlap, would never be considered

Target distribution: no_fit 25-30%, weak 30-35%, medium 20-25%, strong 15-20%
"""

import pandas as pd
import re
from collections import Counter

# Job definitions for matching
JOB_PROFILES = {
    'J001': {
        'title': 'ML Engineer',
        'core_skills': {'ml', 'pytorch', 'tensorflow', 'python', 'model', 'docker'},
        'keywords': ['ml engineer', 'machine learning', 'model development', 'pytorch', 'tensorflow', 'python', 'docker'],
        'domain': 'ML Engineering',
    },
    'J002': {
        'title': 'Data Analyst',
        'core_skills': {'sql', 'tableau', 'powerbi', 'dashboard', 'analytics', 'excel'},
        'keywords': ['data analyst', 'sql', 'tableau', 'power bi', 'dashboard', 'analytics', 'excel'],
        'domain': 'Analytics',
    },
    'J003': {
        'title': 'HR Recruiter',
        'core_skills': {'recruiting', 'ats', 'sourcing', 'screening', 'hr'},
        'keywords': ['recruiter', 'recruiting', 'ats', 'sourcing', 'screening', 'hr', 'greenhouse'],
        'domain': 'HR/Recruiting',
    },
    'J004': {
        'title': 'People Analytics Analyst',
        'core_skills': {'hr', 'analytics', 'sql', 'excel', 'dashboard', 'people'},
        'keywords': ['people analytics', 'hr analytics', 'people data', 'sql', 'dashboard', 'excel'],
        'domain': 'HR Analytics',
    },
    'J005': {
        'title': 'Data Analyst – Growth/Ops',
        'core_skills': {'sql', 'analytics', 'kpi', 'event', 'dashboard', 'python'},
        'keywords': ['data analyst', 'growth', 'kpi', 'event data', 'sql', 'analytics', 'dashboard'],
        'domain': 'Analytics',
    },
    'J006': {
        'title': 'Senior ML Engineer',
        'core_skills': {'ml', 'pytorch', 'tensorflow', 'docker', 'kubernetes', 'mlops', 'senior'},
        'keywords': ['senior ml', 'ml engineer', 'pytorch', 'tensorflow', 'docker', 'kubernetes', 'mlops'],
        'domain': 'ML Engineering',
    },
    'J007': {
        'title': 'Senior Data Engineer',
        'core_skills': {'spark', 'airflow', 'python', 'sql', 'data engineer', 'etl', 'pipeline'},
        'keywords': ['data engineer', 'spark', 'airflow', 'etl', 'pipeline', 'sql', 'bigquery'],
        'domain': 'Data Engineering',
    },
    'J008': {
        'title': 'BI Developer',
        'core_skills': {'tableau', 'powerbi', 'sql', 'dashboard', 'data modeling'},
        'keywords': ['bi developer', 'tableau', 'power bi', 'dashboard', 'sql', 'data modeling'],
        'domain': 'BI/Analytics',
    },
    'J009': {
        'title': 'ML Researcher',
        'core_skills': {'ml', 'pytorch', 'tensorflow', 'nlp', 'cv', 'research', 'transformer'},
        'keywords': ['ml researcher', 'machine learning research', 'nlp', 'cv', 'pytorch', 'tensorflow', 'transformer'],
        'domain': 'ML Research',
    },
    'J010': {
        'title': 'Backend Engineer',
        'core_skills': {'backend', 'api', 'java', 'go', 'node', 'database', 'python'},
        'keywords': ['backend', 'api', 'server', 'java', 'go', 'node', 'database', 'microservices'],
        'domain': 'Backend Engineering',
    },
    'J011': {
        'title': 'HRIS Analyst',
        'core_skills': {'hris', 'hr', 'data', 'sql', 'excel', 'system'},
        'keywords': ['hris', 'hr system', 'hr', 'data', 'excel', 'reporting'],
        'domain': 'HR Systems',
    },
    'J012': {
        'title': 'People Ops Manager',
        'core_skills': {'hr', 'ops', 'people', 'operations', 'management', 'leadership'},
        'keywords': ['people ops', 'hr ops', 'operations', 'management', 'hr', 'program'],
        'domain': 'HR Operations',
    },
    'J013': {
        'title': 'Data Scientist',
        'core_skills': {'data scientist', 'statistics', 'python', 'sql', 'modeling', 'ml'},
        'keywords': ['data scientist', 'statistical', 'python', 'modeling', 'scikit', 'ml'],
        'domain': 'Data Science',
    },
    'J014': {
        'title': 'DevOps Engineer',
        'core_skills': {'devops', 'kubernetes', 'terraform', 'ci/cd', 'docker', 'infrastructure'},
        'keywords': ['devops', 'kubernetes', 'terraform', 'ci/cd', 'docker', 'infrastructure'],
        'domain': 'DevOps/Infrastructure',
    },
    'J015': {
        'title': 'Talent Acquisition Lead',
        'core_skills': {'recruiting', 'ats', 'sourcing', 'leadership', 'talent', 'senior'},
        'keywords': ['talent acquisition', 'recruiter', 'sourcing', 'recruiting', 'ats', 'leadership'],
        'domain': 'HR/Recruiting',
    },
    'J016': {
        'title': 'Product Analytics Manager',
        'core_skills': {'analytics', 'product', 'sql', 'dashboard', 'a/b', 'metrics', 'leadership'},
        'keywords': ['product analytics', 'analytics', 'product', 'dashboard', 'a/b', 'metrics'],
        'domain': 'Product Analytics',
    },
    'J017': {
        'title': 'Junior Data Engineer',
        'core_skills': {'data engineer', 'python', 'sql', 'etl', 'pipeline', 'junior'},
        'keywords': ['junior data engineer', 'data engineer', 'python', 'sql', 'etl', 'pipeline'],
        'domain': 'Data Engineering',
    },
    'J018': {
        'title': 'Embedding Engineer',
        'core_skills': {'embedding', 'ml', 'retrieval', 'vector', 'faiss', 'neural'},
        'keywords': ['embedding', 'retrieval', 'vector', 'faiss', 'similarity', 'neural'],
        'domain': 'ML Engineering',
    },
    'J019': {
        'title': 'Recruiter',
        'core_skills': {'recruiting', 'ats', 'sourcing', 'screening', 'hr'},
        'keywords': ['recruiter', 'recruiting', 'sourcing', 'ats', 'screening', 'hr'],
        'domain': 'HR/Recruiting',
    },
    'J020': {
        'title': 'Senior Analytics Engineer',
        'core_skills': {'sql', 'analytics', 'dbt', 'modeling', 'data', 'python', 'senior'},
        'keywords': ['analytics engineer', 'sql', 'dbt', 'data modeling', 'analytics', 'python'],
        'domain': 'Data/Analytics',
    },
    'J021': {
        'title': 'People Analytics Lead',
        'core_skills': {'people analytics', 'hr', 'sql', 'analytics', 'leadership', 'dashboard'},
        'keywords': ['people analytics', 'hr analytics', 'sql', 'analytics', 'leadership', 'dashboard'],
        'domain': 'HR Analytics',
    },
}


def extract_skills(text):
    """Extract skills from candidate profile."""
    text_lower = text.lower()
    skills = set()
    
    skill_patterns = {
        'python': ['python', 'py'],
        'sql': ['sql', 'mysql', 'postgres', 'redshift', 'bigquery'],
        'tableau': ['tableau'],
        'powerbi': ['power bi', 'powerbi'],
        'ml': ['machine learning', 'ml ', 'deep learning'],
        'pytorch': ['pytorch'],
        'tensorflow': ['tensorflow', 'tf '],
        'docker': ['docker'],
        'kubernetes': ['kubernetes', 'k8s'],
        'spark': ['spark', 'pyspark'],
        'airflow': ['airflow', 'dags'],
        'pandas': ['pandas'],
        'sklearn': ['scikit', 'sklearn'],
        'nlp': ['nlp', 'natural language'],
        'cv': ['computer vision', ' cv '],
        'excel': ['excel'],
        'analytics': ['analytics', 'analytical'],
        'dashboard': ['dashboard'],
        'etl': ['etl', 'pipeline'],
        'recruiting': ['recruiting', 'recruiter', 'recruitment'],
        'ats': ['ats', 'greenhouse', 'workable'],
        'sourcing': ['sourcing', 'source'],
        'hr': ['hr ', 'human resources', ' hr'],
        'java': ['java'],
        'go': ['go ', ' golang'],
        'node': ['node.js', 'nodejs'],
        'api': ['api', 'rest', 'graphql'],
        'aws': ['aws', 'amazon'],
        'gcp': ['gcp', 'google cloud'],
        'dbt': ['dbt', 'data build tool'],
        'terraform': ['terraform'],
        'ci/cd': ['ci/cd', 'cicd', 'continuous'],
        'devops': ['devops'],
        'hris': ['hris'],
        'statistics': ['statistics', 'statistical'],
        'embedded': ['embedding', 'embeddings'],
    }
    
    for skill, patterns in skill_patterns.items():
        for pattern in patterns:
            if pattern in text_lower:
                skills.add(skill)
                break
    
    return skills


def calculate_match_score(candidate_skills, job_profile):
    """Calculate match score between candidate skills and job requirements."""
    core_skills = job_profile['core_skills']
    
    # Count overlaps
    overlap = candidate_skills & core_skills
    overlap_count = len(overlap)
    core_count = len(core_skills)
    
    if core_count == 0:
        return 0
    
    return overlap_count / core_count


def label_pair(candidate_text, job_id, job_profile):
    """
    Label a candidate-job pair using the rubric.
    
    Returns: (label, rationale)
    """
    candidate_skills = extract_skills(candidate_text)
    match_score = calculate_match_score(candidate_skills, job_profile)
    job_title = job_profile['title']
    job_domain = job_profile['domain']
    
    # Count core skill matches
    core_skills = job_profile['core_skills']
    overlap = candidate_skills & core_skills
    
    # Heuristic labeling based on match score and domain keywords
    text_lower = candidate_text.lower()
    
    # Strong: high overlap with core skills, clear domain match
    if match_score >= 0.6 and len(overlap) >= 3:
        return ('strong', f"Strong match with {len(overlap)} core skills directly relevant to {job_title}")
    
    # Strong for perfect/near-perfect overlaps
    if match_score >= 0.75:
        return ('strong', f"Direct experience in {job_title} with matching skill set")
    
    # Medium: good overlap but missing some requirements (seniority, domain context, 1-2 key skills)
    if match_score >= 0.4 and len(overlap) >= 2:
        # Check for seniority mismatch
        if 'senior' in job_title.lower() and not re.search(r'senior|lead|manager|\d+\s*y', text_lower):
            return ('medium', f"Has core {job_title} skills but may lack seniority/leadership experience")
        return ('medium', f"Meets most {job_title} requirements with some domain/skill gaps")
    
    # Weak: some overlap but different role expectations or missing key domain
    if match_score >= 0.2 and len(overlap) >= 1:
        # Check for adjacent domains
        if 'analytics' in job_domain and any(s in candidate_skills for s in ['python', 'sql', 'data']):
            return ('weak', f"Analytics background transferable to {job_title} but lacks specific focus")
        if 'ml' in job_domain.lower() and 'python' in candidate_skills:
            return ('weak', f"Python skills relevant to {job_title} but lacks ML production experience")
        if 'hr' in job_domain.lower() and 'analytics' in candidate_skills:
            return ('weak', f"Analytics experience could support {job_title} but lacks HR domain knowledge")
        return ('weak', f"Some {job_title} skills present but significant gaps in domain/tools")
    
    # No-fit: minimal/no overlap or completely different profession
    return ('no_fit', f"Different professional domain from {job_title} requirements")


def main():
    # Load candidates and jobs
    candidates_df = pd.read_csv(r'data\raw\candidates.csv')
    jobs_df = pd.read_csv(r'data\raw\jobs.csv')
    
    # Generate all pairs
    rows = []
    label_counts = Counter()
    
    for _, cand_row in candidates_df.iterrows():
        candidate_id = cand_row['candidate_id']
        cv_text = cand_row['cv_text']
        
        for _, job_row in jobs_df.iterrows():
            job_id = job_row['job_id']
            job_profile = JOB_PROFILES.get(job_id, {})
            
            if not job_profile:
                continue
            
            label, rationale = label_pair(cv_text, job_id, job_profile)
            label_counts[label] += 1
            
            rows.append({
                'candidate_id': candidate_id,
                'job_id': job_id,
                'expected_label': label,
                'rationale': rationale
            })
    
    # Create DataFrame and write
    output_df = pd.DataFrame(rows)
    output_path = r'data\processed\silver_train_cartesian.csv'
    output_df.to_csv(output_path, index=False, header=True)
    
    # Print distribution
    print("=" * 70)
    print("SILVER TRAINING DATA GENERATED")
    print("=" * 70)
    print(f"\nTotal pairs: {len(rows)}")
    print(f"Candidates: {candidates_df.shape[0]}")
    print(f"Jobs: {jobs_df.shape[0]}")
    print(f"\nLabel Distribution:")
    total = sum(label_counts.values())
    for label in ['strong', 'medium', 'weak', 'no_fit']:
        count = label_counts[label]
        pct = 100.0 * count / total
        print(f"  {label:10s}: {count:4d} ({pct:5.1f}%)")
    print(f"\nFile written: {output_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()
