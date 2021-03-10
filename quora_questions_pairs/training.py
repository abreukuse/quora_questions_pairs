import pandas as pd
import joblib
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from config import X, TARGET, VARIABLES, RANDOM_STATE, SPLITS, TRAINED_MODEL_DIR
from pipeline import pipeline
from quora_questions_pairs import __version__ as version


def save_pipeline(*, pipeline_to_persist) -> None:
    """Persist the pipeline"""
    save_file_name = f"pipeline_version_{version.replace('.','-')}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name
    remove_old_pipelines(file_to_keep=save_file_name)
    joblib.dump(pipeline_to_persist, save_path)
    print(f'Pipeline version {version} saved.')


def remove_old_pipelines(*, file_to_keep):
    """Delete old model pipelines from the package"""
    for model_file in config.TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in [file_to_keep, '__init__.py', '.gitkeep']:
            model_file.unlink()


def run_training():
    """Get the last fold in the cross validation process due to reproducibility purposes"""
    folds = KFold(n_splits=SPLITS, shuffle=True, random_state=RANDOM_STATE)
    train_indices, validation_indices = list(folds.split(X))[-1][0], list(folds.split(X))[-1][1]

    X_train = X.iloc[train_indices]
    X_validation = X.iloc[validation_indices]

    y_train = TARGET.iloc[train_indices]
    y_validation = TARGET.iloc[validation_indices]

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_validation)
    
    prtin(accuracy_score(y_validation, y_pred))
    save_pipeline(pipeline_to_persist=pipeline)

if __name__ == '__main__':
    run_training()