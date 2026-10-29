"""NLP pipeline: normalisation, tokenisation, lexical analysis, POS tagging,
rule/gazetteer NER and lexicon-based sentiment analysis."""
import re
import difflib
import unicodedata

try:
    import nltk
    from nltk.stem import WordNetLemmatizer
    _LEM = WordNetLemmatizer()
    _LEM.lemmatize("tests")
except Exception:  # NLTK / wordnet not available -> graceful fallback
    nltk, _LEM = None, None


def setup_nltk():
    """Best-effort download of the small NLTK resources used (safe to skip)."""
    try:
        import nltk as _n
        for pkg in ("averaged_perceptron_tagger", "averaged_perceptron_tagger_eng", "wordnet", "stopwords"):
            _n.download(pkg, quiet=True)
    except Exception:
        pass


TOKEN_RE = re.compile(r"[\w\u0900-\u097F']+", re.UNICODE)
DEVANAGARI = re.compile(r"[\u0900-\u097F]")

SLANG = {"pls": "please", "plz": "please", "sem": "semester", "tmrw": "tomorrow", "tomorow": "tomorrow",
         "u": "you", "r": "are", "thx": "thanks", "exm": "exam", "assgn": "assignment", "asgn": "assignment",
         "tt": "timetable", "lec": "lecture", "lecs": "lectures", "atendance": "attendance", "attendence": "attendance"}

STOP_EN = {"a", "an", "the", "is", "are", "am", "was", "were", "be", "to", "of", "in", "on", "at", "for", "and", "or",
           "i", "me", "my", "you", "your", "it", "this", "that", "do", "does", "can", "will", "please", "there", "so"}
STOP_HI = {"है", "हैं", "का", "की", "के", "में", "से", "को", "और", "मैं", "मुझे", "क्या", "आहे", "काय", "मला", "ka", "ki",
           "ke", "hai", "hain", "mein", "se", "ko", "aur", "mujhe", "kya", "me", "hoga", "karna", "kare"}

SUBJECTS = {"nlp": "NLP", "natural language processing": "NLP", "dl": "DL", "deep learning": "DL",
            "bda": "BDA", "big data": "BDA", "cloud": "Cloud", "project": "Project", "dbms": "DBMS",
            "database": "DBMS", "os": "OS", "operating system": "OS", "cn": "CN", "computer network": "CN",
            "dsa": "DSA", "data structures": "DSA", "ml": "ML", "machine learning": "ML", "ai": "AI",
            "maths": "Maths", "mathematics": "Maths", "iot": "IoT"}
BRANCHES = {"cse": "CSE", "aiml": "AIML", "ai ml": "AIML", "ai&ml": "AIML", "it": "IT", "extc": "EXTC", "mech": "MECH"}
DAYS = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6,
        "mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5}
TODAY_W = {"today", "aaj", "आज", "todays", "today's"}
TOMORROW_W = {"tomorrow", "kal", "कल", "उद्या"}
EXAM_TYPES = [(r"mid[\s-]?term|midterm|internal|\bia\b", "Mid-term"), (r"end[\s-]?sem\w*|endsem", "End-sem"),
              (r"viva|oral", "Viva"), (r"practical|\bpr\b", "Practical"), (r"unit test", "Unit test"),
              (r"\bkt\b|atkt|backlog", "ATKT")]

POS_LEX = {"good": 1, "great": 1.5, "thanks": 1, "thank": 1, "awesome": 1.5, "helpful": 1, "nice": 1, "excellent": 1.5,
           "love": 1.5, "perfect": 1.5, "happy": 1, "best": 1.5, "working": 0.5, "achha": 1, "accha": 1, "badhiya": 1,
           "shukriya": 1, "dhanyavad": 1, "मस्त": 1, "धन्यवाद": 1}
NEG_LEX = {"bad": -1, "worst": -2, "terrible": -2, "angry": -1.5, "frustrated": -1.5, "annoyed": -1, "annoying": -1,
           "useless": -2, "waste": -1.5, "problem": -0.7, "issue": -0.5, "complaint": -1, "unfair": -1.5, "wrong": -1,
           "stressed": -1, "worried": -1, "tension": -1, "pareshan": -1.5, "bekar": -1.5, "bakwas": -1.5, "broken": -1.5,
           "error": -1, "crash": -1.5, "late": -0.5, "गलत": -1, "परेशान": -1.5, "नाराज": -1.5, "त्रास": -1.5, "fail": -1, "failed": -1}
NEGATORS = {"not", "no", "never", "dont", "don't", "cant", "can't", "nahi", "nahin", "नहीं", "नाही", "won't", "isn't"}
INTENSIFIERS = {"very", "so", "really", "extremely", "bahut", "too", "बहुत"}


def normalize(text):
    text = unicodedata.normalize("NFKC", text or "")
    return re.sub(r"\s+", " ", text).strip().lower()


def tokenize(text):
    return TOKEN_RE.findall(text)


def correct(token, vocab):
    """Spelling correction against the domain vocabulary (difflib)."""
    if not vocab or len(token) < 5 or token in vocab or token.isdigit() or DEVANAGARI.search(token):
        return token
    m = difflib.get_close_matches(token, vocab, n=1, cutoff=0.82)
    return m[0] if m else token


def pos_tag(tokens):
    latin = [t for t in tokens if not DEVANAGARI.search(t)]
    tagged = {}
    if nltk and latin:
        try:
            for (w, tag), orig in zip(nltk.pos_tag(latin), latin):
                tagged[orig] = tag
        except Exception:
            pass
    out = []
    for t in tokens:
        if DEVANAGARI.search(t):
            out.append((t, "X"))
        elif t.isdigit():
            out.append((t, "CD"))
        else:
            out.append((t, tagged.get(t, "NN")))
    return out


def lemma(tok):
    if _LEM and tok.isascii():
        try:
            return _LEM.lemmatize(_LEM.lemmatize(tok, "v"))
        except Exception:
            return tok
    return tok


def extract_entities(text):
    """Hybrid NER: regex patterns + domain gazetteers."""
    ents = {}
    m = re.search(r"(?:roll\s*(?:no\.?|number|num|#)?|रोल\s*(?:नंबर|नं\.?)?)\s*[:\-]?\s*(\d{1,4})\b", text)
    if m:
        ents["roll_no"] = int(m.group(1))
    m = re.search(r"\bsem(?:ester)?\s*[-:]?\s*(\d)\b|\b(\d)(?:st|nd|rd|th)?\s*sem(?:ester)?\b|सेमेस्टर\s*(\d)", text)
    if m:
        ents["semester"] = int(next(g for g in m.groups() if g))
    for k, v in BRANCHES.items():
        if re.search(r"(?<![\w])" + re.escape(k) + r"(?![\w])", text):
            ents["branch"] = v
            break
    subs = []
    for k, v in SUBJECTS.items():
        if re.search(r"(?<![\w])" + re.escape(k) + r"(?![\w])", text) and v not in subs:
            subs.append(v)
    if subs:
        ents["subject"] = subs
    toks = set(tokenize(text))
    for k, v in DAYS.items():
        if k in toks:
            ents["day"] = v
            break
    if toks & TODAY_W:
        ents["date_ref"] = "today"
    elif toks & TOMORROW_W:
        ents["date_ref"] = "tomorrow"
    for pat, name in EXAM_TYPES:
        if re.search(pat, text):
            ents["exam_type"] = name
            break
    return ents


def analyze_sentiment(tokens):
    score, i = 0.0, 0
    for i, t in enumerate(tokens):
        val = POS_LEX.get(t, 0) + NEG_LEX.get(t, 0)
        if not val:
            continue
        window = tokens[max(0, i - 2):i]
        if any(w in NEGATORS for w in window):
            val = -val * 0.8
        if any(w in INTENSIFIERS for w in window):
            val *= 1.5
        score += val
    norm = score / (abs(score) + 1.5) if score else 0.0
    label = "positive" if norm > 0.2 else "negative" if norm < -0.2 else "neutral"
    return {"score": round(norm, 2), "label": label}


def analyze(text, vocab=None):
    norm = normalize(text)
    toks = [SLANG.get(t, t) for t in tokenize(norm)]
    toks = [correct(t, vocab) for t in toks]
    pos = pos_tag(toks)
    lemmas = [lemma(t) for t in toks]
    stop = STOP_EN | STOP_HI
    keywords = [l for l in lemmas if l not in stop and len(l) > 1]
    return {"normalized": norm, "tokens": toks, "pos": pos, "lemmas": lemmas, "keywords": keywords,
            "clean_text": " ".join(toks), "entities": extract_entities(norm),
            "sentiment": analyze_sentiment(toks)}
