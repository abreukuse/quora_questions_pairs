import joblib
from quora_questions_pairs import __version__ as version
from quora_questions_pairs.config import TRAINED_MODEL_DIR

def load_pipeline(*, file_name):
    """Load a persisted pipeline."""

    file_path = TRAINED_MODEL_DIR / file_name
    saved_pipeline = joblib.load(file_path)
    return saved_pipeline

pipeline_file_name = f'pipeline_v{version}.pkl'
pipeline = load_pipeline(file_name=pipeline_file_name)


def make_prediction(*, input_data):
    """Make a prediction using the saved model pipeline."""

    classify = pipeline.predict(input_data)
    classify_proba = pipeline.predict_proba(input_data)
    print(classify, classify_proba)
    return classify, classify_proba


if __name__ == '__main__':
    # test pipeline with a input sample
    import pandas as pd
    from quora_questions_pairs.config import DATASETS_DIR
    
    test_input = pd.read_csv(DATASETS_DIR / 'input_test_make_prediction.csv')
    make_prediction(input_data=test_input)