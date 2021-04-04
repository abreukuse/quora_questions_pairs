import pandas as pd
from sklearn.model_selection import KFold
from quora_questions_pairs.config import X, TARGET, SPLITS, RANDOM_STATE

def data_split(X, y):
    """Get the last fold in the cross validation process due to reproducibility purposes"""
    folds = KFold(n_splits=SPLITS, shuffle=True, random_state=RANDOM_STATE)
    train_indices, validation_indices = list(folds.split(X))[-1][0], list(folds.split(X))[-1][1]

    X_train = X.iloc[train_indices]
    X_validation = X.iloc[validation_indices]

    y_train = y.iloc[train_indices]
    y_validation = y.iloc[validation_indices]

    return X_train, X_validation, y_train, y_validation