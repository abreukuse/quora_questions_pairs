from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression
from feature_engine.wrappers import SklearnTransformerWrapper
from quora_questions_pairs.config import VARIABLES, HYPERPARAMETERS
from quora_questions_pairs.functions import (basic_feature_engineering, 
					     fuzzy_features, 
					     drop_original_columns,
					     training_data_snapshot)


training_pipeline = Pipeline([
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
					 FunctionTransformer(drop_original_columns, 
										 kw_args={'variables': VARIABLES})
				 ),
				 (
					 'scaling',
					 SklearnTransformerWrapper(transformer = StandardScaler())
				 ),
				 (
					 'save_data_snapshot',
					 FunctionTransformer(training_data_snapshot)
				 ),
				 (
					 'algorithm',
					 LogisticRegression(**HYPERPARAMETERS)
				 )
			     ]
			   )
