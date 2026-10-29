"""EduBot - AI-Powered Student Query Chatbot
Flask backend wiring the NLP pipeline (tokenisation, POS tagging, NER,
intent classification, sentiment analysis) to a retrieval-based QA engine
and a small mock student-records store."""
import json
import os
import uuid
from datetime import datetime

from flask import Flask, jsonify, render_template, request, session

from nlp.intent import load_intents, load_or_train, predict
from nlp.lang import detect_language, reply_language
from nlp.pipeline import analyze, setup_nltk
from nlp.qa import FAQIndex

BASE = os.path.dirname(os.path.abspath(__file__))
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

setup_nltk()
app = Flask(__name__)
app.secret_key = "edubot-dev-secret"

INTENTS = load_intents()
PIPE, VOCAB = load_or_train()

CUSTOM_QA_PATH = os.path.join(BASE, "data", "custom_qa.json")


def load_custom_qa():
    if os.path.exists(CUSTOM_QA_PATH):
        with open(CUSTOM_QA_PATH, encoding="utf-8") as f:
            return json.load(f)
    return []


def save_custom_qa(items):
    with open(CUSTOM_QA_PATH, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


FAQ = FAQIndex(extra=load_custom_qa())

with open(os.path.join(BASE, "data", "students.json"), encoding="utf-8") as f:
    STUDENTS = json.load(f)

LOG_PATH = os.path.join(BASE, "data", "chat_log.jsonl")
LOW_CONF = 0.35
FALLBACK = {
    "en": "I'm not fully sure I understood that. Could you rephrase, or try asking about admissions, attendance, timetable, exams, assignments, fees, placements or resources?",
    "hi": "मुझे यह पूरी तरह समझ नहीं आया। कृपया दोबारा पूछें, या प्रवेश, उपस्थिति, टाइम-टेबल, परीक्षा, असाइनमेंट, फीस के बारे में पूछें।",
    "mr": "मला हे नीट समजले नाही. कृपया पुन्हा विचारा, किंवा प्रवेश, उपस्थिती, वेळापत्रक, परीक्षा, फी याबद्दल विचारा.",
}


def log_turn(row):
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        pass


def student_lookup(ents, session_roll):
    roll = str(ents.get("roll_no")) if ents.get("roll_no") is not None else session_roll
    return roll, STUDENTS.get(roll) if roll else None


def answer_attendance(stu, ents):
    if not stu:
        return "Tell me your roll number (e.g. 'roll no 12') and I'll pull up your attendance."
    subs = ents.get("subject")
    att = stu["attendance"]
    if subs:
        lines = [f"{s}: {att[s]}%" for s in subs if s in att]
        if lines:
            return f"{stu['name']}'s attendance — " + ", ".join(lines) + "."
    lines = ", ".join(f"{k}: {v}%" for k, v in att.items())
    low = [k for k, v in att.items() if v < 75]
    warn = f" Note: {', '.join(low)} is below the 75% minimum." if low else " All subjects meet the 75% requirement."
    return f"{stu['name']}'s attendance — {lines}.{warn}"


def answer_timetable(stu, ents):
    if not stu:
        return "Tell me your roll number and I'll show your timetable."
    tt = stu["timetable"]
    day_idx = ents.get("day")
    if day_idx is None:
        day_idx = datetime.now().weekday() if ents.get("date_ref") in (None, "today") else (datetime.now().weekday() + 1) % 7
    day_idx = day_idx % 7
    slots = tt.get(str(day_idx))
    if not slots:
        return f"No lectures scheduled for {stu['name']} on {DAY_NAMES[day_idx]} (holiday or off day)."
    return f"{stu['name']}'s {DAY_NAMES[day_idx]} timetable: " + "; ".join(slots) + "."


def answer_assignment(stu, ents):
    if not stu:
        return "Tell me your roll number to see your pending assignments."
    subs = ents.get("subject")
    items = [a for a in stu["assignments"] if a["status"] == "pending"]
    if subs:
        items = [a for a in items if a["subject"] in subs] or items
    if not items:
        return f"No pending assignments for {stu['name']} right now. Nice work staying on top of it!"
    lines = "; ".join(f"{a['subject']}: {a['title']} (due {a['due']})" for a in items)
    return f"{stu['name']}'s pending assignments — {lines}."


DYNAMIC_HANDLERS = {"attendance": answer_attendance, "timetable": answer_timetable, "assignment": answer_assignment}


def build_reply(text, session_roll):
    nlp_res = analyze(text, VOCAB)
    lang = reply_language(detect_language(text), session.get("lang_pref", "auto"))
    intent, conf, top3 = predict(PIPE, nlp_res["clean_text"])
    ents = nlp_res["entities"]

    if intent in ("greeting", "goodbye", "thanks") and conf >= LOW_CONF:
        opts = INTENTS[intent]["responses"].get(lang) or INTENTS[intent]["responses"]["en"]
        reply, source = opts[0], "intent-template"
    elif conf < LOW_CONF:
        reply, source = FALLBACK.get(lang, FALLBACK["en"]), "fallback"
    elif intent in DYNAMIC_HANDLERS:
        roll, stu = student_lookup(ents, session_roll)
        reply, source = DYNAMIC_HANDLERS[intent](stu, ents), "record-lookup"
        if stu:
            session_roll = roll
    else:
        faq, sim = FAQ.search(text, intent=intent)
        if sim < 0.12:
            reply, source = FALLBACK.get(lang, FALLBACK["en"]), "fallback"
        else:
            reply, source = faq["a"].get(lang) or faq["a"]["en"], "faq-retrieval"

    sentiment = nlp_res["sentiment"]
    escalate = sentiment["label"] == "negative" and sentiment["score"] < -0.4
    if escalate:
        esc_msg = {"en": " I can see this is frustrating — I've flagged this for the helpdesk to follow up.",
                   "hi": " मैं समझ सकता हूँ यह परेशान करने वाला है — मैंने इसे हेल्पडेस्क के लिए चिह्नित कर दिया है।",
                   "mr": " हे त्रासदायक आहे हे समजते — मी हे हेल्पडेस्कसाठी नोंदवले आहे."}
        reply += esc_msg.get(lang, esc_msg["en"])

    return {
        "reply": reply, "intent": intent, "confidence": round(conf, 3), "top3": top3,
        "entities": ents, "language": lang, "sentiment": sentiment, "source": source,
        "escalate": escalate, "tokens": nlp_res["tokens"], "pos": nlp_res["pos"],
        "session_roll": session_roll,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/admin")
def admin():
    return render_template("admin.html")


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(force=True) or {}
    text = (data.get("message") or "").strip()
    if not text:
        return jsonify({"error": "empty message"}), 400
    if data.get("lang_pref"):
        session["lang_pref"] = data["lang_pref"]
    session_id = session.get("sid") or str(uuid.uuid4())
    session["sid"] = session_id
    result = build_reply(text, session.get("roll"))
    session["roll"] = result.pop("session_roll")
    log_turn({"ts": datetime.now().isoformat(), "session": session_id, "message": text,
              "intent": result["intent"], "confidence": result["confidence"],
              "sentiment": result["sentiment"]["label"], "escalate": result["escalate"]})
    return jsonify(result)


@app.route("/api/reset", methods=["POST"])
def api_reset():
    session.clear()
    return jsonify({"ok": True})


@app.route("/api/stats")
def api_stats():
    if not os.path.exists(LOG_PATH):
        return jsonify({"total": 0, "intents": {}, "sentiment": {}, "escalations": 0, "avg_confidence": 0})
    rows = [json.loads(l) for l in open(LOG_PATH, encoding="utf-8") if l.strip()]
    intents, sentiment = {}, {}
    for r in rows:
        intents[r["intent"]] = intents.get(r["intent"], 0) + 1
        sentiment[r["sentiment"]] = sentiment.get(r["sentiment"], 0) + 1
    avg_conf = sum(r["confidence"] for r in rows) / len(rows) if rows else 0
    return jsonify({"total": len(rows), "intents": intents, "sentiment": sentiment,
                     "escalations": sum(1 for r in rows if r["escalate"]), "avg_confidence": round(avg_conf, 3)})


@app.route("/api/add_faq", methods=["POST"])
def api_add_faq():
    global FAQ
    data = request.get_json(force=True) or {}
    q, a, intent = data.get("question", "").strip(), data.get("answer", "").strip(), data.get("intent", "custom")
    if not q or not a:
        return jsonify({"error": "question and answer required"}), 400
    items = load_custom_qa()
    items.append({"question": q, "answer": a, "intent": intent})
    save_custom_qa(items)
    FAQ = FAQIndex(extra=items)
    return jsonify({"ok": True, "count": len(items)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5057, debug=False)
