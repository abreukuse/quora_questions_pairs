FROM python:3.7

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY ./quora_questions_pairs ./quora_questions_pairs

ENV PYTHONPATH .

CMD streamlit run quora_questions_pairs/app.py