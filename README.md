# MoodDecode — NLP Logic and Model Documentation

This project is part of the **MoodDecode** challenge. It contains the NLP logic and model integration for three API endpoints that analyze human text to understand emotion, detect potential crises, and summarize long-form content.

## Features

- **Emotion Analysis**: Detect emotional tone from text using state-of-the-art NLP models
- **Crisis Detection**: Identify potential mental health crisis indicators using rule-based matching
- **Text Summarization**: Generate concise summaries of long-form content and journal entries

## API Endpoints

### 1. `POST /analyze_mood`

**Description:** Detects the user's emotional tone from input text.

**Model Used:** [`j-hartmann/emotion-english-distilroberta-base`](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)

**Request:**
```json
{
  "text": "I'm feeling excited and full of energy!"
}
```

**Response:**
```json
{
  "emotion": "joy"
}
```

### 2. `POST /detect_crisis`

**Description:** Detects signs of a mental health crisis using a rule-based keyword detection system.

**Logic:** A curated set of phrases related to crisis situations is matched against the lowercase version of the input text.

**Request:**
```json
{
  "text": "I don't see any reason to live anymore."
}
```

**Response:**
```json
{
  "crisis_detected": true
}
```

### 3. `POST /summarize`

**Description:** Summarizes long paragraphs or journal entries into concise, meaningful summaries.

**Model Used:** [`sshleifer/distilbart-cnn-12-6`](https://huggingface.co/sshleifer/distilbart-cnn-12-6)

**Sample Request:**
```json
{
  "text": "Today started off as one of those days where everything just felt heavy. I woke up late because I couldn't sleep last night — my mind kept spinning with thoughts about everything I had to do and everything I haven't done. I skipped breakfast, which probably made me even more irritable. As I got to work, the number of emails and Slack messages waiting for me was overwhelming. I felt like no matter how fast I tried to move through my tasks, they just kept piling up. During the morning standup, I didn't feel like talking. I kept my camera off and just said I was 'fine,' even though I wasn't. I know people care, but sometimes it's easier to just keep it all inside. Around noon, I tried to take a break and went for a walk. The weather was warm, and there were kids playing in the park nearby. That gave me a moment of calm — like a small reminder that the world keeps moving even when you feel stuck. In the afternoon, I finally had a long-overdue conversation with a teammate about something that had been bothering me. It went better than I expected, and I felt a little lighter afterward. I even managed to finish two big tasks that had been on my plate for over a week. That gave me a small sense of accomplishment, even if the day started off badly. Now that I'm home, I feel emotionally drained but also proud that I didn't give up on the day. I wrote this all down to remind myself that even hard days can have good moments — and maybe tomorrow will be easier."
}
```

**Sample Response:**
```json
{
  "summary": "The day began emotionally heavy and overwhelming, but small moments of calm and a productive afternoon helped turn it around. Despite exhaustion, there was a sense of pride in not giving up."
}
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd mooddecode
   ```

2. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```


This will:
- Output the detected emotion
- Indicate whether the text suggests a crisis
- Print a summary of long-form input

## Dependencies

The project requires the following Python packages:
- `transformers` - For loading and using pre-trained NLP models
- `torch` - PyTorch framework for model inference

## Project Structure

```
mooddecode/
├── model.py              # Main script with NLP logic
├── requirements.txt      # Python dependencies
├── README.md            # This file

```

## Models Used

### Emotion Analysis
- **Model:** `j-hartmann/emotion-english-distilroberta-base`
- **Type:** Fine-tuned DistilRoBERTa
- **Purpose:** Multi-class emotion classification
- **Emotions:** Joy, sadness, anger, fear, surprise, disgust, etc.

### Text Summarization
- **Model:** `sshleifer/distilbart-cnn-12-6`
- **Type:** DistilBART fine-tuned on CNN/DailyMail
- **Purpose:** Abstractive text summarization
- **Strength:** Generating concise, coherent summaries

### Crisis Detection
- **Type:** Rule-based keyword matching
- **Approach:** Pattern matching against curated crisis-related phrases
- **Processing:** Case-insensitive text analysis

## Use Cases

- **Mental Health Monitoring:** Track emotional patterns over time
- **Crisis Intervention:** Early detection of mental health emergencies
- **Journal Analysis:** Summarize and understand personal reflections
- **Content Moderation:** Identify concerning content in user-generated text
- **Research Applications:** Analyze emotional trends in text data

## Important Notes

- **Crisis Detection:** This tool is designed to assist, not replace, professional mental health assessment
- **Privacy:** Ensure sensitive text data is handled according to privacy regulations
- **Model Limitations:** Results may vary based on text quality, length, and context





---

**MoodDecode** - Understanding emotions through intelligent text analysis 