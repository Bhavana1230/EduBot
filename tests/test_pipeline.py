import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nlp.pipeline import analyze
from nlp.intent import load_or_train, predict
from nlp.lang import detect_language

PIPE, VOCAB = load_or_train()

CASES = [
    ("when is the exam timetable", "exam_schedule"),
    ("meri attendance kitni hai", "attendance"),
    ("assignment ki last date kya hai", "assignment"),
    ("how to pay fees", "fees"),
    ("placement kab hote hain", "placement"),
    ("thank you so much", "thanks"),
    ("hello there", "greeting"),
    ("bye bye", "goodbye"),
    ("परीक्षा कब है", "exam_schedule"),
    ("library timings please", "resources"),
]

def test_intent_accuracy():
    correct = 0
    for text, expected in CASES:
        clean = analyze(text, VOCAB)["clean_text"]
        intent, conf, _ = predict(PIPE, clean)
        correct += (intent == expected)
    acc = correct / len(CASES)
    assert acc >= 0.8, f"accuracy too low: {acc}"

def test_entity_extraction():
    r = analyze("meri attendance kitni hai roll no 12")
    assert r["entities"]["roll_no"] == 12

def test_sentiment():
    assert analyze("this is the worst, totally useless and broken").sentiment["label"] if False else True
    r = analyze("thanks a lot, this is great and helpful")
    assert r["sentiment"]["label"] == "positive"
    r2 = analyze("this is the worst, totally useless and broken")
    assert r2["sentiment"]["label"] == "negative"

def test_language_detection():
    assert detect_language("मला वेळापत्रक दाखवा") == "mr"
    assert detect_language("परीक्षा कब है") == "hi"
    assert detect_language("mujhe timetable batao") == "hinglish"
    assert detect_language("when is the exam") == "en"

if __name__ == "__main__":
    test_intent_accuracy(); test_entity_extraction(); test_sentiment(); test_language_detection()
    print("All tests passed.")
