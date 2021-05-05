import pandas as pd
import numpy as np
import streamlit as st
from quora_questions_pairs.predict import make_prediction

first_question = st.text_input('Enter a question.')
second_question = st.text_input('Enter another question.')

dict_data = {'question1': [first_question],
             'question2': [second_question]}

input_data = pd.DataFrame(dict_data)

button = st.button('Are the questions duplicate?')

mapping = {0: 'No', 1: 'Yes'}

if button:
    classify, classify_proba = make_prediction(input_data=input_data)
    st.write(mapping[classify[0]])
    score = np.round(classify_proba[:,1][0], 3)
    st.write(f'Score from Logistic Regression: {score}')

button = False