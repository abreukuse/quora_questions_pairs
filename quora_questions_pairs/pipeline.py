from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from functions import *

pipeline = Pipeline([
                     (
                         'select_variables',
                         FunctionTransformer(select_variables, kw_args={'variables': VARIABLES})
                     ),
                     (
                         'dropna',
                         FunctionTransformer(dropna)
                     ),
                     (
                         'basic_feature_engineering', 
                          FunctionTransformer(basic_feature_engineering)
                      ),
                     (
                         'fuzzy_features', 
                          FunctionTransformer(fuzzy_features)
                      ),
                     (
                         'drop',
                         FunctionTransformer(drop, kw_args={'variables': VARIABLES})
                     )
                     ]
                    )