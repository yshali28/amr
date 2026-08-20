import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

SPECIES = 'Escherichia coli'
ANTIBIOTICS = [
    'Ciprofloxacin',
    'Cotrimoxazole',
    'Ceftriaxone',
    'Amoxicillin-Clavulanic acid',
    'Ampicillin-Amoxicillin',
]
MISSING_TOKEN = '-'
TRAIN_YEARS = [2015, 2016, 2017]
TEST_YEARS = [2018]
RANDOM_SEED = 42

DATA_ROOT = Path(os.environ['DATA_ROOT'])
ID_FOLDER = DATA_ROOT / 'id'
