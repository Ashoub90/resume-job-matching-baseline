# System Architecture

## Overview  
The HR AI CV–Job Matching & Ranking System is designed as a scoring and evaluation pipeline that ranks candidates for each job description using a hybrid of rule-based and semantic similarity signals.

---

## High-Level Flow  

Job Description → Candidate Scoring → Ranking → Evaluation  

---

## Components  

### 1. Data Preparation  
- Raw CV summaries and job descriptions  
- Preprocessing and normalization  
- Separation into silver (training) and gold (evaluation) datasets  

---

### 2. Rule-Based Skill Extraction  
- Regex-based skill matching  
- Measures overlap between required and candidate skills  
- Encodes hard constraints (e.g., SQL, ML frameworks)  

---

### 3. Semantic Similarity Engine  
- Sentence embeddings for CV and job text  
- Captures paraphrased and transferable skills  
- Handles non-exact keyword matches  

---

### 4. Hybrid Scoring Module  
- Weighted combination of rule-based score and semantic similarity  
- Balances precision and recall  
- Produces final relevance score per job–candidate pair  

---

### 5. Ranking Engine  
- Sorts candidates by hybrid score  
- Produces ranked shortlists  

---

### 6. Evaluation Pipeline  
- Gold dataset manually labeled by humans  
- Metrics used:
  - nDCG@10  
  - Recall@K  
  - MRR  

---

## Design Considerations  
- Ranking rather than classification  
- Explicit handling of noisy labels  
- Separation of training and evaluation data  
- Explainable scoring logic  

---

## Future Improvements  
- Add spaCy-based skill explainability  
- Support file uploads (PDF, DOCX)  
- Learn hybrid weights from gold data  
- Add recruiter-facing explanations  
