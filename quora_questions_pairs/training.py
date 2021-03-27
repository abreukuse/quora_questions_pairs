import pandas as pd
import joblib
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from config import (X_TRAIN, 
                    X_VALIDATION,
                    Y_TRAIN,
                    Y_VALIDATION, 
                    VARIABLES, 
                    RANDOM_STATE, 
                    SPLITS, 
                    TRAINED_MODEL_DIR)
from pipeline import training_pipeline
from quora_questions_pairs import __version__ as version


def save_pipeline(*, pipeline_to_persist) -> None:
    """Persist the pipeline"""
    save_file_name = f"pipeline_v{version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name
    joblib.dump(pipeline_to_persist, save_path)
    print(f'Pipeline version {version} saved.')


def run_training(*, X_train, X_validation, y_train, y_validation):
    """Training"""
    training_pipeline.fit(X_train, y_train)
    y_pred = training_pipeline.predict(X_validation)
    
    print(f'EVALUATION: {accuracy_score(y_validation, y_pred)}')
    save_pipeline(pipeline_to_persist=training_pipeline)

if __name__ == '__main__':
    run_training(
                 X_train=X_TRAIN, 
                 X_validation=X_VALIDATION,
                 y_train=Y_TRAIN, 
                 y_validation=Y_VALIDATION
                 )