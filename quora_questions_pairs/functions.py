import pandas as pd
import numpy as np
from rapidfuzz import fuzz
import os
from quora_questions_pairs.config import TRAINING_DATA_DIR
from quora_questions_pairs import __version__ as version


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

