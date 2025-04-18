from transformers import pipeline
from datetime import datetime

emotion_classifier = pipeline(
    "text-classification",
    model="nateraw/bert-base-uncased-emotion",
    top_k=1
)


opposite_emotions = {
    'joy': 'sadness',
    'sadness': 'joy',
    'anger': 'calm',
    'fear': 'confidence',
    'love': 'hate',
    'surprise': 'anticipation',
    'disgust': 'admiration'
}

critical_keywords = [
    "suicidal", "kill myself", "end it all", "can't go on", "no reason to live",
    "want to die", "i want to disappear", "ending my life", "life is meaningless",
    "i wish i were dead", "taking pills", "jump off", "overdose", "hang myself",
    "slit my wrist", "tired of living", "give up on life", "i just want to sleep forever",
    "i want to end it", "ending everything", "ending all of this", "can’t live like this"
]

high_risk_keywords = [
    "hurting myself", "cutting", "i give up", "unbearable pain", "numb all the time",
    "i'm broken", "can't stop crying", "never happy", "i hate myself", "i'm a failure",
    "worthless", "wish i wasn't born", "i'm exhausted emotionally", "feel like a burden",
    "nothing matters", "done with life"
]

concerning_keywords = [
    "what's the point", "i'm done", "i can't take it anymore", "sick of everything",
    "i'm lost", "feel like a burden", "no one would care", "nobody cares",
    "can’t feel anything", "i'm always anxious", "panic attacks", "i'm always tired"
]

risky_behavior_keywords = [
    "drinking to forget", "popped too many pills", "high all the time", "can’t stop using"
]

chat_history = []

def detect_emotion(text):
    text_lower = text.lower()
    urgency = None
    urgency_score = 0.0


    if any(kw in text_lower for kw in critical_keywords):
        urgency, urgency_score = "critical", 1.0
    elif any(kw in text_lower for kw in high_risk_keywords):
        urgency, urgency_score = "high_risk", 0.9
    elif any(kw in text_lower for kw in concerning_keywords):
        urgency, urgency_score = "concerning", 0.7
    elif any(kw in text_lower for kw in risky_behavior_keywords):
        urgency, urgency_score = "risky_behavior", 0.6


    results = emotion_classifier(text)
    best = results[0][0]
    emotion = best['label']
    score = best['score']


    if "not" in text_lower:
        emotion = opposite_emotions.get(emotion, emotion)


    final_label = urgency if urgency else emotion
    final_score = urgency_score if urgency else score


    chat_history.append({
        "timestamp": datetime.now().isoformat(),
        "text": text,
        "emotion": emotion,
        "emotion_score": score,
        "urgency": urgency,
        "urgency_score": urgency_score,
        "final_label": final_label,
        "final_score": final_score
    })

    return final_label, final_score