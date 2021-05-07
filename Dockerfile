FROM python:3.7

WORKDIR /quora_questions_pairs

COPY ./api ./api

COPY ./requirements.txt ./requirements.txt

COPY ./setup.py ./setup.py

COPY ./quora_questions_pairs ./quora_questions_pairs

RUN pip install -r api/requirements.txt

ENV PYTHONPATH .

CMD streamlit run api/app.py