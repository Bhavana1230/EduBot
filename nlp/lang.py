"""Lightweight language identification: English / Hinglish / Hindi / Marathi."""
import re

DEV = re.compile(r"[\u0900-\u097F]")
MR_MARKERS = {"आहे", "आहेत", "काय", "मला", "माझी", "माझा", "माझे", "कधी", "किती", "कुठे", "कशी", "कसे", "तुम्ही", "मी", "उद्या", "आजचे"}
HINGLISH = {"kya", "hai", "hain", "kab", "kaise", "kaha", "kahan", "mujhe", "meri", "mera", "batao", "kitni", "kitna", "nahi",
            "karna", "karni", "kare", "ka", "ki", "ke", "aaj", "kal", "mein", "ho", "hoga", "chahiye", "bharni", "aayega", "kaunsi"}


def detect_language(text):
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "en"
    if len(DEV.findall(text)) / len(letters) > 0.3:
        words = set(re.findall(r"[\u0900-\u097F]+", text))
        return "mr" if words & MR_MARKERS else "hi"
    words = set(re.findall(r"[a-z]+", text.lower()))
    return "hinglish" if len(words & HINGLISH) >= 1 and len(words) > 1 else "en"


def reply_language(detected, preference="auto"):
    if preference in ("en", "hi", "mr"):
        return preference
    return detected if detected in ("hi", "mr") else "en"
