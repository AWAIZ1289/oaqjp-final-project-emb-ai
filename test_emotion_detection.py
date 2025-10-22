from EmotionDetection import emotion_detector

def test_joy():
    resp = emotion_detector("I am glad this happened")
    assert resp["dominant_emotion"] == "joy"

def test_anger():
    resp = emotion_detector("I am really mad about this")
    assert resp["dominant_emotion"] == "anger"

def test_disgust():
    resp = emotion_detector("I feel disgusted just hearing about this")
    assert resp["dominant_emotion"] == "disgust"

def test_sadness():
    resp = emotion_detector("I am so sad about this")
    assert resp["dominant_emotion"] == "sadness"

def test_fear():
    resp = emotion_detector("I am really afraid that this will happen")
    assert resp["dominant_emotion"] == "fear"
