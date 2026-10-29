"""Retrieval-based question answering over the FAQ knowledge base
(TF-IDF cosine similarity, boosted when the FAQ shares the predicted intent)."""
import json
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from nlp.pipeline import normalize

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class FAQIndex:
    def __init__(self, extra=None):
        with open(os.path.join(BASE, "data", "faq.json"), encoding="utf-8") as f:
            self.entries = json.load(f)
        for e in (extra or []):  # staff-added answers from the admin panel
            self.entries.append({"intent": e.get("intent") or "custom", "q": [e["question"]],
                                 "a": {"en": e["answer"]}})
        self.questions, self.owner = [], []
        for i, e in enumerate(self.entries):
            for q in e["q"]:
                self.questions.append(normalize(q))
                self.owner.append(i)
        self.wv = TfidfVectorizer(ngram_range=(1, 2), token_pattern=r"[\w\u0900-\u097F']+", sublinear_tf=True)
        self.cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
        self.M = hstack([self.wv.fit_transform(self.questions), self.cv.fit_transform(self.questions)]).tocsr()

    def search(self, text, intent=None):
        q = normalize(text)
        v = hstack([self.wv.transform([q]), self.cv.transform([q])]).tocsr()
        sims = (self.M @ v.T).toarray().ravel() / 2.0  # two L2-normalised blocks -> cosine in [0, 1]
        for j, own in enumerate(self.owner):
            if intent and self.entries[own]["intent"] == intent:
                sims[j] += 0.15
        j = int(np.argmax(sims))
        return self.entries[self.owner[j]], float(sims[j])
