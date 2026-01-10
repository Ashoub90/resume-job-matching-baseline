# HR AI CV–Job Matching & Ranking System

> This repository represents a **foundational baseline project**.  
> It intentionally focuses on ranking logic and evaluation rigor and serves as the basis for a larger end-to-end CV ingestion and ranking application.

---

## Overview

This project builds and evaluates a **job → candidate ranking system** for CV–job matching.  
Given a job description, the system ranks candidates based on a **hybrid scoring approach** that combines:

1. Rule-based skill matching  
2. Semantic similarity using text embeddings  

The goal is **not binary classification**, but **ranking quality** — surfacing the best candidates first, which mirrors real recruiter workflows.

This project emphasizes:

- Information Retrieval (IR)–grade evaluation  
- Robust handling of noisy labels  
- Clear separation between training (silver) and evaluation (gold) data  
- Practical, explainable system design  

---

## Problem Framing

Traditional ML CV-matching projects often frame the task as:

> “Does candidate X match job Y? (yes/no)”

This project instead frames the problem as:

> **“For a given job, how well does the system rank candidates by relevance?”**

This distinction matters because:

- Recruiters review ranked shortlists, not isolated predictions  
- Accuracy and F1 are misleading for ranking tasks  
- IR metrics (Recall@K, MRR, nDCG) better capture real-world usefulness  

---

## Data Sources

### Candidates
- Short free-text CV summaries (2–3 sentences each)  
- No structured sections (e.g., no explicit “Skills” or “Experience” headers)  

### Jobs
- 21 job descriptions across:
  - ML / Data  
  - Analytics  
  - HR / People Ops  

Some roles are intentionally similar to reflect real hiring ambiguity.

---

## Setup

```bash
pip install -r requirements.txt
Project Structure
graphql
Copy code
data/
  raw/                 # Original job and candidate descriptions
  processed/           # Model outputs, evaluation sets, predictions
  skills_dictionary.json

skills/                # Reusable skill extraction logic
  loader.py
  rule_based_extraction.py
  ml_based_extraction.py

scripts/               # Pipeline and evaluation scripts
  match_cv_to_job.py
  match_cv_to_job_with_semantic.py
  evaluate_job_to_candidate_ranking.py
  build_gold_eval_pairs.py
  ...
Scoring Architecture
Each job–candidate pair is scored using three signals:

1. Rule-Based Score
Regex-based skill extraction

Measures overlap between job-required skills and candidate-mentioned skills

Encodes hard constraints (e.g., SQL, ML frameworks, ATS experience)

2. Semantic Similarity
Sentence embeddings over full CV text and job description

Captures:

Transferable skills

Paraphrased experience

Related roles without exact keyword overlap

3. Hybrid Final Score
Weighted combination of rule-based score and semantic similarity

Designed to balance:

Precision (rules)

Recall & flexibility (semantics)

Labeling Strategy (Critical Design Choice)
Why full manual labeling was not feasible
Cartesian product: 70 candidates × 21 jobs = 1,470 pairs

Manual labeling at this scale is unrealistic

Silver Labels (Bootstrapping)
Initial labels were generated to:

Train

Debug

Stress-test the system

They are treated as noisy, non-authoritative, and are explicitly not used for final evaluation.

Silver labels exist to help the system learn — not to prove correctness.

Gold Evaluation Set (Human-Labeled)
To ensure credible evaluation, a gold dataset was created manually.

Selection strategy
From the full Cartesian predictions:

Top-ranked candidates per job

Bottom-ranked candidates per job

(Optional) mid-range candidates

This resulted in:

~100 manually labeled job–candidate pairs

Balanced across strong / medium / weak / no-fit cases

Why this works
Focuses labeling effort where ranking quality matters

Avoids label leakage

Mirrors real evaluation practices in search & recommender systems

Evaluation Methodology
Why classification metrics were rejected ❌
Accuracy, F1, and confusion matrices assume:

Mutually exclusive classes

No ranking order

They fail to capture how early good matches appear.

Metrics Used ✅ (IR-grade)
nDCG@10 (Primary Metric)
Measures ranking quality with graded relevance

Rewards:

Strong matches appearing early

Penalizes good matches ranked too low

Recall@K
Answers: “Does the system surface at least one good candidate early?”

Used during development for sanity checks

MRR
Measures how early the first good match appears

Final Evaluation Results (Gold Data)
Job → Candidate Ranking
Average nDCG@10 (gold evaluation):

≈ 0.95

Baseline Comparison
Scoring Method	nDCG@10
Rule-based only	0.94
Semantic-only	0.93
Hybrid (final)	0.96

Interpretation
Rules alone are strong for hard skill matching

Semantic similarity adds recall and flexibility

Hybrid approach consistently performs best

This confirms that:

The system is not just embeddings

Design choices added measurable value

Key Challenges & How They Were Addressed1. Noisy Labels
LLM-generated labels sometimes over-penalized missing domain keywords

Solution:

Treat silver labels as approximate

Use human-labeled gold set for evaluation

2. Similar Job Descriptions
Some roles (e.g., analyst variants) are naturally ambiguous

This reflects real hiring scenarios

Ranking metrics handle ambiguity better than classification.

3. Limited CV Structure
CVs lacked explicit sections

Regex + embeddings were sufficient

spaCy / NER deferred to future explainability work



Limitations & Future Work
Current Limitations
Small gold evaluation set (by design)

CVs are short and unstructured

Rule-based skill extraction is shallow

Future Improvements
Add spaCy-based explainability:

Missing skills

Skill categories

Expand to:

File uploads (PDF, DOCX)

Arbitrary new jobs & candidates

Learn score weights from gold data

Add recruiter-facing explanations:

“Candidate ranked #2 because of X, Y, missing Z”

Final Note
This project prioritizes correct problem framing and evaluation over flashy modeling.
It demonstrates an understanding of:

Ranking systems

IR metrics

Label noise

Evaluation leakage

Practical ML system design

This is intentionally not a one-click AI-generated app — and it cannot be replaced by one.