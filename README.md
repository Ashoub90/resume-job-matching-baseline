# HR AI CV–Job Matching & Ranking System

An AI-powered system that ranks candidates for a given job description using a hybrid scoring approach that combines rule-based skill matching and semantic similarity from text embeddings.

This project focuses on **ranking quality and evaluation rigor** rather than simple binary classification, reflecting real recruiter workflows where candidates are reviewed as a ranked shortlist.

---

## Overview

Traditional CV matching systems often frame the task as:

> “Does candidate X match job Y? (yes/no)”

This project instead frames the problem as:

> “For a given job, how well does the system rank candidates by relevance?”

The system combines:
- Rule-based skill extraction  
- Semantic similarity using sentence embeddings  
- Information Retrieval (IR) evaluation metrics  

to produce explainable and measurable candidate rankings.

---

## Key Features

- Hybrid scoring system (rule-based + semantic similarity)  
- Regex-based skill extraction  
- Sentence embeddings for semantic matching  
- Job → candidate ranking (not binary classification)  
- Gold vs silver label separation for robust evaluation  
- IR-grade evaluation metrics:
  - nDCG@10  
  - Recall@K  
  - MRR  
- Designed for explainability and recruiter-oriented use cases  

---

## Tech Stack

- Python  
- Sentence Embeddings  
- Regex-based Skill Extraction  
- Information Retrieval Metrics (nDCG, MRR, Recall@K)  

---

## System Design

Each job–candidate pair is scored using three signals:

### 1. Rule-Based Skill Matching
- Regex-based skill extraction  
- Measures overlap between job-required skills and candidate-mentioned skills  
- Encodes hard constraints (e.g., SQL, ML frameworks, ATS experience)  

### 2. Semantic Similarity
- Sentence embeddings over full CV text and job description  
- Captures transferable and paraphrased skills  
- Handles non-exact keyword matches  

### 3. Hybrid Final Score
- Weighted combination of rule-based score and semantic similarity  
- Balances precision (rules) and recall (semantics)  

---

## Evaluation Methodology

This project emphasizes correct evaluation practices used in search and recommender systems.

### Metrics Used
- **nDCG@10 (primary metric)** – measures ranking quality with graded relevance  
- **Recall@K** – checks whether good candidates appear early  
- **MRR** – measures how early the first good match appears  

### Labeling Strategy
- **Silver labels:** automatically generated for training and debugging (noisy)  
- **Gold labels:** manually curated for final evaluation  

This separation avoids label leakage and ensures credible performance measurement.

---

## Results (Gold Evaluation Set)

| Scoring Method     | nDCG@10 |
|-------------------|---------|
| Rule-based only   | 0.94    |
| Semantic only     | 0.93    |
| Hybrid (final)    | 0.96    |

**Interpretation:**
- Rule-based matching captures hard skill requirements  
- Semantic similarity improves recall and flexibility  
- Hybrid approach achieves the best overall ranking quality  

---

## Limitations

- Small gold evaluation set (by design)  
- CVs are short and unstructured  
- Rule-based skill extraction is shallow  
- No user interface or file upload support  

---

## Future Improvements

- Add spaCy-based explainability (missing skills, skill categories)  
- Support file uploads (PDF, DOCX)  
- Learn hybrid score weights from gold data  
- Add recruiter-facing explanations:
  > “Candidate ranked #2 because of X, Y, and missing Z”  

---

## Documentation

Full project documentation is available:

- [Overview](docs/overview.md)  
- [User Guide](docs/user-guide.md)  
- [System Architecture](docs/system-architecture.md)  
- [Pipeline Reference](docs/api-reference.md)  
- [Limitations & Risks](docs/limitations-and-risks.md)  

---

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
Run evaluation scripts:

python scripts/match_cv_to_job_with_semantic.py
python scripts/evaluate_job_to_candidate_ranking.py
```
## Disclaimer
This project is for educational and portfolio purposes only.
It does not replace professional recruitment systems or human hiring decisions.