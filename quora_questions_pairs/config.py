import pathlib
import pandas as pd

PACKAGE_ROOT = pathlib.Path(quora_questions_pairs.__file__).resolve().parent
DATASETS_DIR = PACKAGE_ROOT / 'datasets'
TRAINED_MODEL_DIR = PACKAGE_ROOT / 'trained_models'

X = pd.read_csv(DATASETS_DIR / 'train.csv', usecols=['question1', 'question2']).dropna()
TARGET = pd.read_csv(DATASETS_DIR / 'train.csv', usecols=['is_duplicate'], squeeze=True).iloc[X.index]

VARIABLES = ['question1', 'question2']
RANDOM_STATE = 60
SPLITS = 5

HYPERPARAMETERS = {'C': 1,
                   'solver': 'sag',
                   'max_iter': 1000,
                   'random_state': RANDOM_STATE,
                   'n_jobs': -1}