"""Shared helpers for Phase 0 analysis scripts."""

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRICS_DIR = PROJECT_ROOT / 'results' / 'metrics'


def load_raw_metadata(id_folder):
    """Load all *_clean.csv files under id_folder, tagged with year_folder (full columns)."""
    paths = glob.glob(os.path.join(id_folder, '*', '*_clean.csv'))
    frames = []
    for path in paths:
        year_folder = os.path.basename(os.path.dirname(path))
        df = pd.read_csv(path, low_memory=False)
        df['year_folder'] = year_folder
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def collapse_label(val, missing_token='-'):
    """Map a raw DRIAMS antibiotic result to 0 (S) / 1 (R/I), NaN if missing."""
    if pd.isna(val) or val == missing_token:
        return np.nan
    if 'R' in val:
        return 1
    if 'I' in val:
        return 1
    if 'S' in val:
        return 0
    return np.nan
