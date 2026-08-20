# Dependency-Aware AMR Prediction

Predicting antibiotic resistance (AMR) from MALDI-TOF mass spectra using
Classifier Chains, with temporal validation against a random-split baseline.
B.Tech semester project targeting a Scopus conference paper.

## Overview

Antibiotic resistance predictions across multiple drugs are not independent —
resistance to one antibiotic is often correlated with resistance to others.
This project compares two multi-label strategies for predicting R/S status
per antibiotic from MALDI-TOF spectra:

- **Binary Relevance (baseline)** — an independent XGBoost classifier per antibiotic
- **Classifier Chains (main)** — XGBoost classifiers chained so each antibiotic's
  prediction can use the predicted labels of prior antibiotics in the chain

Both are evaluated under a **random split** and a **temporal split** (train on
older isolates, test on newer ones) to check whether performance holds up
under realistic, time-shifted deployment conditions. Model behavior is
inspected per antibiotic and per chain step using SHAP.

## Data

- **Primary:** [DRIAMS-A](https://datadryad.org/) — ~145k clinical MALDI-TOF spectra
- **Secondary:** BV-BRC AMR TSV
- **Species:** *E. coli*, *S. aureus*, *K. pneumoniae*, *P. aeruginosa*
- **Features:** ~6000-dim binned spectra (2000–20000 Da)
- **Labels:** R/S per antibiotic (multi-label); intermediate (I) is merged into R

Raw and processed data are not tracked in this repo (see `.gitignore`) —
place downloaded data under `data/raw/`.

## Project structure

```
src/
  config.py               # paths, constants
  data/
    load.py                # load DRIAMS / BV-BRC data
    features.py             # spectral binning / feature extraction
    labels.py                # R/S/I label handling
    split.py                  # random and temporal train/test splits
  models/
    baseline.py             # Binary Relevance (XGBoost)
    chains.py                # Classifier Chains (XGBoost)
    ensemble.py               # chain ensembling
  evaluation/
    metrics.py               # F1, Hamming loss, Jaccard, AUROC
  interpretability/
    shap_analysis.py         # SHAP per antibiotic / per chain step
experiments/
  run_baseline.py           # baseline training + evaluation entrypoint
  run_chains.py              # classifier chains entrypoint
  run_ensemble.py             # ensemble entrypoint
notebooks/                  # exploratory / pipeline-check notebooks
results/metrics/            # experiment outputs (CSV / SHAP arrays)
paper/                      # write-up
```

## Setup

```bash
python -m venv amr_env
amr_env\Scripts\activate        # Windows
pip install -r requirements.txt
```

CPU only — no GPU is required or used.

## Running experiments

```bash
python experiments/run_baseline.py
python experiments/run_chains.py
python experiments/run_ensemble.py
```

Results (metrics CSVs, SHAP values) are written to `results/metrics/`.

## Metrics

Evaluated with F1, Hamming loss, Jaccard index, and AUROC — not accuracy
alone, since AMR labels are imbalanced and multi-label.

## Status

Work in progress toward a Scopus conference paper submission.
