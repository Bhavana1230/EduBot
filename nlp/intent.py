"""Intent classifier: word + character TF-IDF features -> Logistic Regression."""
import json
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from nlp.pipeline import analyze

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE, "model", "intent.joblib")
TOKEN = r"[\w\u0900-\u097F']+"


def load_intents():
    with open(os.path.join(BASE, "data", "intents.json"), encoding="utf-8") as f:
        return json.load(f)


def build_pipeline():
    feats = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), token_pattern=TOKEN, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)),
    ])
    return Pipeline([("f", feats), ("clf", LogisticRegression(C=30, max_iter=2000))])


def training_data():
    intents = load_intents()
    X, y = [], []
    for name, spec in intents.items():
        for ex in spec["examples"]:
            X.append(analyze(ex)["clean_text"])
            y.append(name)
    return X, y


def train(save=True):
    X, y = training_data()
    pipe = build_pipeline().fit(X, y)
    vocab = sorted({w for t in X for w in t.split() if len(w) >= 4 and w.isascii()})
    if save:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump({"pipe": pipe, "vocab": vocab}, MODEL_PATH)
    return pipe, vocab


def load_or_train():
    if os.path.exists(MODEL_PATH):
        d = joblib.load(MODEL_PATH)
        return d["pipe"], d["vocab"]
    return train()


def predict(pipe, clean_text):
    proba = pipe.predict_proba([clean_text])[0]
    order = proba.argsort()[::-1]
    classes = pipe.classes_
    return classes[order[0]], float(proba[order[0]]), [(classes[i], round(float(proba[i]), 3)) for i in order[:3]]
