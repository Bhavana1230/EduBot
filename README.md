# EduBot — AI-Powered Student Query Chatbot

An NLP-based chatbot that answers student questions about admissions, exams,
attendance, timetables, assignments, fees, placements and resources —
in English, Hindi, Marathi or Hinglish.

## Features
- Custom NLP pipeline: tokenization, spelling correction, POS tagging,
  lemmatization, rule/gazetteer Named Entity Recognition (roll no, semester,
  subject, day, exam type), lexicon-based sentiment analysis.
- Intent classification: TF-IDF (word + char n-grams) + Logistic Regression,
  trained on 180+ labelled examples across 14 intents.
- Retrieval-based Question Answering over a 23-entry FAQ knowledge base
  (TF-IDF cosine similarity + intent boosting).
- Dynamic, per-student answers (attendance %, today's timetable, pending
  assignments) pulled from a mock student-records store, using entities
  extracted from the query (roll number, subject, day).
- Multilingual replies (English / Hindi / Marathi), auto-detected or
  user-selected.
- Sentiment-aware escalation: strongly negative queries are flagged for
  human follow-up.
- Staff dashboard: query volume by intent, sentiment distribution, and a
  form to add new knowledge-base answers without touching code.

## Run it
```bash
pip install -r requirements.txt
python app.py
```
Visit `http://localhost:5057` for the chat UI and `/admin` for the staff
dashboard. The intent model trains automatically on first run and is
cached at `model/intent.joblib`.

## Run tests
```bash
python -m pytest tests/ -q
```

## Project layout
```
app.py               Flask app & routes
nlp/pipeline.py       tokenisation, POS tagging, NER, sentiment
nlp/lang.py            language identification
nlp/intent.py          intent classifier (train/predict)
nlp/qa.py               FAQ retrieval engine
data/intents.json      intent training examples
data/faq.json            FAQ knowledge base
data/students.json    mock student records
templates/, static/  chat UI + admin dashboard
tests/                     pytest suite
```
