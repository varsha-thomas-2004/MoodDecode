from transformers import pipeline


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

def analyze_mood(text: str) -> dict:
    result = emotion_pipeline(text)[0]
    return {"emotion": result["label"].lower()}

def detect_crisis(text: str) -> dict:
    text_lower = text.lower()
    crisis = any(kw in text_lower for kw in CRISIS_KEYWORDS)
    return {"crisis_detected": crisis}

def summarize(text: str) -> dict:
    result = summarizer(text, max_length=60, min_length=15, do_sample=False)[0]["summary_text"]
    return {"summary": result}

# Test block
if __name__ == "__main__":
    print("\n--- Emotion Tests ---")
    print(analyze_mood("I'm so excited for my trip tomorrow!"))
    print(analyze_mood("I just want to be alone and cry."))
    print(analyze_mood("That made me so angry!"))
    print(analyze_mood("I'm afraid things are going to go wrong."))

    print("\n--- Crisis Detection Tests ---")
    print(detect_crisis("I feel like no one cares and everything is pointless."))
    print(detect_crisis("I'm overwhelmed but I'll be okay."))
    print(detect_crisis("I can't take this anymore."))
    print(detect_crisis("Feeling a little down today, but hanging in there."))

    print("\n--- Summarization Test ---")
    long_text = """
Today started off as one of those days where everything just felt heavy. I woke up late because I couldn’t sleep last night — my mind kept spinning with thoughts about everything I had to do and everything I haven’t done. I skipped breakfast, which probably made me even more irritable. As I got to work, the number of emails and Slack messages waiting for me was overwhelming. I felt like no matter how fast I tried to move through my tasks, they just kept piling up.

During the morning standup, I didn’t feel like talking. I kept my camera off and just said I was ‘fine,’ even though I wasn’t. I know people care, but sometimes it’s easier to just keep it all inside. Around noon, I tried to take a break and went for a walk. The weather was warm, and there were kids playing in the park nearby. That gave me a moment of calm — like a small reminder that the world keeps moving even when you feel stuck.

In the afternoon, I finally had a long-overdue conversation with a teammate about something that had been bothering me. It went better than I expected, and I felt a little lighter afterward. I even managed to finish two big tasks that had been on my plate for over a week. That gave me a small sense of accomplishment, even if the day started off badly.

Now that I’m home, I feel emotionally drained but also proud that I didn’t give up on the day. I wrote this all down to remind myself that even hard days can have good moments — and maybe tomorrow will be easier.
"""

    print(summarize(long_text))
