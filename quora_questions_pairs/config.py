import pathlib
import pandas as pd
from sklearn.model_selection import KFold

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent
DATASETS_DIR = PACKAGE_ROOT / 'datasets'
TRAINED_MODEL_DIR = PACKAGE_ROOT / 'trained_models'
TRAINING_DATA_DIR = PACKAGE_ROOT / 'training_data'

VARIABLES = ['question1', 'question2']
RANDOM_STATE = 60
SPLITS = 5

HYPERPARAMETERS = {'C': 1,
                   'solver': 'sag',
                   'max_iter': 1000,
                   'random_state': RANDOM_STATE,
                   'n_jobs': -1}

