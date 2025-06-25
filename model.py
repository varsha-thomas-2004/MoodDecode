from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

emotion_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")


CRISIS_KEYWORDS = set([
    "hopeless", "kill myself", "hurt myself", "suicidal", "end it",
    "worthless", "give up", "no reason to live", "depressed", "sad",
    "cry", "overwhelmed", "anxious", "panic", "despair", "lost hope",
    "can't go on", "can't cope", "need help", "need support", "feel alone",
    "feel empty", "feel trapped", "feel hopeless", "feel worthless",
    "feel like a burden", "feel like giving up", "feel like crying",
    "feel like screaming", "feel like running away", "feel like no one cares",
    "feel like everything is pointless", "feel like i'm a failure",
    "feel like i'm a burden", "feel like i'm not enough",
    "feel like i'm not worthy", "feel like i'm a disappointment",
    "feel like i'm a mistake", "feel like i'm lost", "feel like i'm drowning",
    "feel like i'm in a dark place", "feel like i'm in a hole",
    "feel like i'm in a rut", "feel like i'm in a crisis",
    "feel like i'm in a downward spiral", "feel like i'm in a never-ending cycle",
    "i want to die", "why am i alive", "what's the point", "no one cares",
    "i can't handle this", "i'm done", "i'm tired of life", "self-harm",
    "i don't want to be here", "everything hurts", "life is pointless",
    "can't take this anymore", "ending it all", "make it stop"
])

@app.route('/analyze-mood', methods = ['POST'])
def analyze_mood():
    data = request.json
    text = data.get("text","")
    result = emotion_pipeline(text)[0]
    return {"emotion": result["label"].lower()}

@app.route('/detect-crisis', methods = ['POST'])
def detect_crisis():
    data = request.json
    text = data.get("text","").lower()
    crisis = any(kw in text for kw in CRISIS_KEYWORDS)
    return {"crisis_detected": crisis}

@app.route('/summarize', methods = ['POST'])
def summarize():
    data = request.json
    text = data.get("text", "")
    result = summarizer(text, max_length=60, min_length=15, do_sample=False)[0]["summary_text"]
    return {"summary": result}

# Test block
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)