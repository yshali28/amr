# Project: Dependency-Aware AMR Prediction

## What this is
B.Tech semester project. Predicting antibiotic resistance from 
MALDI-TOF spectra using Classifier Chains + temporal validation.
Target: Scopus conference paper by Aug 15.

## Data
- Primary: DRIAMS-A (Dryad), ~145k clinical MALDI-TOF spectra
- Secondary: BV-BRC AMR TSV
- Species: E. coli, S. aureus, K. pneumoniae, P. aeruginosa
- Features: ~6000-dim binned spectra (2000-20000 Da)
- Labels: R/S per antibiotic, multi-label. Merge I into R.

## Approach
- Baseline: Binary Relevance (independent XGBoost per antibiotic)
- Main: Classifier Chains (XGBoost base) — models label dependencies
- Validation: temporal split (train on older, test on newer) 
  AND random split, to compare
- Interpretability: SHAP per antibiotic and per chain step

## Constraints
- CPU only, no GPU
- Python, scikit-learn, XGBoost, SHAP, maldi-learn

## Metrics
F1, Hamming loss, Jaccard, AUROC — not accuracy alone