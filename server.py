"""
Flask-based Emotion Detection Server.

Provides a single POST endpoint '/emotionDetector' which accepts a JSON payload
or form data with key 'text' and returns emotion scores and dominant emotion.
"""

from typing import Dict, Any
from flask import Flask, request, jsonify
from EmotionDetection import emotion_detector

APP = Flask(__name__)


def build_sentence(result: Dict[str, Any]) -> str:
    """
    Build a user-friendly sentence from the emotion result dict.

    The sentence format:
    For the given statement, the system response is 'anger': X, 'disgust': Y, 'fear': Z,
    'joy': A and 'sadness': B. The dominant emotion is <dominant>.
    """
    anger = result.get("anger", 0)
    disgust = result.get("disgust", 0)
    fear = result.get("fear", 0)
    joy = result.get("joy", 0)
    sadness = result.get("sadness", 0)
    dominant = result.get("dominant_emotion", "")
    sentence = (
        f"For the given statement, the system response is "
        f"'anger': {anger}, 'disgust': {disgust}, 'fear': {fear}, "
        f"'joy': {joy} and 'sadness': {sadness}. The dominant emotion is {dominant}."
    )
    return sentence


@APP.route("/emotionDetector", methods=("POST",))
def emotion_detector_endpoint():
    """
    Flask endpoint that receives text, calls the emotion detector, and returns JSON.

    Accepts:
        - JSON body: {"text": "..."}
        - form data: text=...

    Returns:
        JSON object with keys:
            - "sentence": formatted string
            - "result": the dictionary of emotion scores + dominant_emotion
        Or in case of invalid text: {"message": "Invalid text! Please try again!"}
    """
    text_input = None
    if request.is_json:
        payload = request.get_json(silent=True) or {}
        text_input = payload.get("text")
    if text_input is None:
        text_input = request.form.get("text", "")

    try:
        result = emotion_detector(text_input)
    except (RuntimeError, ValueError) as error:  # handle known detector errors
        return jsonify({"error": "Analysis failed", "details": str(error)}), 500

    if not result or result.get("dominant_emotion") is None:
        return jsonify({"message": "Invalid text! Please try again!"}), 200

    sentence = build_sentence(result)
    return jsonify({"sentence": sentence, "result": result})


def main() -> None:
    """
    Entry point to run the Flask application.
    Uses host 0.0.0.0 and port 5050 to avoid conflicts on 5000.
    """
    APP.run(host="0.0.0.0", port=5050)


if __name__ == "__main__":
    main()
