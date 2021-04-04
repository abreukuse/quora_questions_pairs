import pandas as pd
import numpy as np
from rapidfuzz import fuzz
import os
from quora_questions_pairs.config import TRAINING_DATA_DIR, X, TARGET
from quora_questions_pairs.data_management import data_split
from quora_questions_pairs import __version__ as version

X_TRAIN = data_split(X, TARGET)[0]

def basic_feature_engineering(X):
	"""Extract some basic features"""
	X = X.copy()
	print('*Creating basic features.')
	for i, column in enumerate(X.columns):
		X[f'len_q{i+1}'] = X[f'question{i+1}'].apply(lambda a: len(a))
		X[f'len_char_q{i+1}'] = X[f'question{i+1}'].apply(lambda a: len(''.join(set(a.replace(' ', '')))))
		X[f'len_word_q{i+1}'] = X[f'question{i+1}'].apply(lambda a: len(a.split()))
	X['diff_len'] = X['len_q1'] - X['len_q2']
	X['common_words'] = [len(set(a.lower().split()).intersection(set(b.lower().split()))) 
			     for a, b in zip(X['question1'], X['question2'])]
	return X

def fuzzy_features(X):
	"""Generate fuzzy based features"""
	print('*Creating fuzzy features.')
	attributes = ['QRatio',
		      'WRatio',
		      'partial_ratio',
		      'partial_token_set_ratio',
		      'partial_token_sort_ratio',
		      'token_set_ratio',
		      'token_sort_ratio']
	for attribute in attributes:
		print(f'-{attribute}')
		X[f'fuzz_{attribute}'] = [getattr(fuzz, attribute)(a, b) for a, b in zip(X['question1'], X['question2'])]
	return X

def drop_original_columns(X, variables):
	"""Drop the original questions columns"""
	print('* Dropping original columns.')
	X = X.drop(columns=variables)
	return X

def training_data_snapshot(X):
	"""Save a sample of the training data right before it enters in the algorithm"""
	if not os.path.isdir(TRAINING_DATA_DIR):
		os.mkdir(TRAINING_DATA_DIR)
		
	file_path = TRAINING_DATA_DIR / f'training_data_v{version}.csv'

	if isinstance(X, pd.DataFrame) and all(X.index.isin(X_TRAIN.index)):
		print('* Saving training data.')
		X.head(10).to_csv(file_path, index=False, header=True)
		return X

	elif isinstance(X, np.ndarray) and all(X.index.isin(X_TRAIN.index)):
		print('* Saving training data.')
		np.savetxt(file_path, X[:10, :], delimiter = ",")
		return X
	else:
		return X
