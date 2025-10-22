import requests
import json
from typing import Optional

EMOTION_URL = "https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
HEADERS = {
    "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock",
    "Content-Type": "application/json"
}

def emotion_detector(text_to_analyze: str) -> Optional[dict]:
    """
    Handles Watson EmotionPredict API call.
    If status_code == 400 (blank input), returns all keys as None.
    """
    payload = {"raw_document": {"text": text_to_analyze}}

    try:
        resp = requests.post(EMOTION_URL, headers=HEADERS, json=payload, timeout=10)
    except requests.RequestException as e:
        raise RuntimeError("Request failed: " + str(e))

    if getattr(resp, "status_code", None) == 400:
        return {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None
        }

    try:
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        raise RuntimeError("Invalid response from Watson: " + str(e))

    try:
        emotions = data["emotionPredictions"][0]["emotion"]
    except (KeyError, IndexError):
        raise ValueError("Unexpected response format")

    anger = emotions.get("anger", 0)
    disgust = emotions.get("disgust", 0)
    fear = emotions.get("fear", 0)
    joy = emotions.get("joy", 0)
    sadness = emotions.get("sadness", 0)

    emotion_dict = {
        "anger": anger,
        "disgust": disgust,
        "fear": fear,
        "joy": joy,
        "sadness": sadness
    }
    dominant_emotion = max(emotion_dict, key=emotion_dict.get)
    emotion_dict["dominant_emotion"] = dominant_emotion

    return emotion_dict
