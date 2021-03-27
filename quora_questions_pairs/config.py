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

X = pd.read_csv(DATASETS_DIR / 'train.csv', usecols=['question1', 'question2']).dropna()
TARGET = pd.read_csv(DATASETS_DIR / 'train.csv', usecols=['is_duplicate'], squeeze=True).iloc[X.index]

def data_split(X, y):
    """Get the last fold in the cross validation process due to reproducibility purposes"""
    folds = KFold(n_splits=SPLITS, shuffle=True, random_state=RANDOM_STATE)
    train_indices, validation_indices = list(folds.split(X))[-1][0], list(folds.split(X))[-1][1]

    X_train = X.iloc[train_indices]
    X_validation = X.iloc[validation_indices]

    y_train = y.iloc[train_indices]
    y_validation = y.iloc[validation_indices]

    return X_train, X_validation, y_train, y_validation

X_TRAIN, X_VALIDATION, Y_TRAIN, Y_VALIDATION = data_split(X, TARGET)

