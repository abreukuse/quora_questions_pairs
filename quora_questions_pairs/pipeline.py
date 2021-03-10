from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from functions import basic_feature_engineering, fuzzy_features, drop_original_columns
from config import VARIABLES, HYPERPARAMETERS

pipeline = Pipeline([
                     (
                         'basic_feature_engineering', 
                          FunctionTransformer(basic_feature_engineering)
                      ),
                     (
                         'fuzzy_features', 
                          FunctionTransformer(fuzzy_features)
                      ),
                     (
                         'drop_original_columns',
                         FunctionTransformer(drop_original_columns, kw_args={'variables': VARIABLES})
                     ),
                     (
                         'Scaling',
                         StandardScaler()
                     ),
                     (
                         'algorithm',
                         LogisticRegression(**HYPERPARAMETERS)
                     )
                     ]
                    )