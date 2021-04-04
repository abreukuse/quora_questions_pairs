import pandas as pd
from quora_questions_pairs.functions import basic_feature_engineering, fuzzy_features
from quora_questions_pairs.config import X, RANDOM_STATE

def test_basic_feature_engineering():
	# Given:
	test_data = X.sample(n=10, random_state=RANDOM_STATE)
	new_columns = ['len_q1','len_char_q1',
				   'len_word_q1','len_q2',
				   'len_char_q2','len_word_q2',
				   'diff_len','common_words']

	# When:
	subject = basic_feature_engineering(test_data)

	# Then:

	# Check number of columns:
	assert subject.shape[1] == 10
	# Check if the list new_columns is a subset of the transformed dataframe columns
	assert set(new_columns).issubset(list(subject.columns))


def test_fuzzy_features():
	# Given:
	test_data = X.sample(n=10, random_state=RANDOM_STATE)
	new_columns = ['fuzz_QRatio','fuzz_WRatio',
				   'fuzz_partial_ratio','fuzz_partial_token_set_ratio',
				   'fuzz_partial_token_sort_ratio','fuzz_token_set_ratio',
				   'fuzz_token_sort_ratio']

	
	# When:
	subject = fuzzy_features(test_data)  

	# Then:

	# Check number of columns:
	assert subject.shape[1] == 9
	# Check if the list new_columns is a subset of the transformed dataframe columns
	assert set(new_columns).issubset(list(subject.columns))