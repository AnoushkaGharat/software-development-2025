# Emotion Detection (Combo Model)
# CNN + DeepFace

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from deepface import DeepFace

# Configuration

CNN_MODEL_PATH = r"C:\software-development-2025\face_model.h5"

CNN_WEIGHT = 0.6
DEEPFACE_WEIGHT = 0.4

COMMON_EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# ------------------------------
# Load models
# ------------------------------

print("[INFO] Loading CNN emotion model...")
cnn_model = load_model(CNN_MODEL_PATH)

print("[INFO] Loading face detector...")
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Functions

def predict_cnn_emotion(face_bgr):
    """Return emotion probability dictionary from CNN"""
    face_gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    face_gray = cv2.resize(face_gray, (48, 48))
    face_gray = image.img_to_array(face_gray)
    face_gray = np.expand_dims(face_gray, axis=0)

    preds = cnn_model.predict(face_gray, verbose=0)[0]
    return dict(zip(COMMON_EMOTIONS, preds))


def predict_deepface_emotion(face_bgr):
    """Return emotion probability dictionary from DeepFace"""
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_rgb = cv2.resize(face_rgb, (224, 224))

    try:
        result = DeepFace.analyze(
            face_rgb,
            actions=['emotion'],
            enforce_detection=False
        )

        if isinstance(result, list):
            result = result[0]

        scores = result['emotion']
        return {k.lower(): v / 100 for k, v in scores.items()}

    except Exception as e:
        print("[WARNING] DeepFace error:", e)
        return None


def ensemble_emotion(cnn_probs, deepface_probs):
    """Combine predictions using weighted voting"""
    combined = {}

    for emotion in COMMON_EMOTIONS:
        combined[emotion] = CNN_WEIGHT * cnn_probs.get(emotion, 0)

        if deepface_probs:
            combined[emotion] += DEEPFACE_WEIGHT * deepface_probs.get(emotion, 0)

    final_emotion = max(combined, key=combined.get)
    confidence = combined[final_emotion]

    return final_emotion, confidence

# Webcam loop

cap = cv2.VideoCapture(0)
print("Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]

        # CNN prediction
        cnn_probs = predict_cnn_emotion(face_roi)

        # DeepFace prediction
        deepface_probs = predict_deepface_emotion(face_roi)

        # Ensemble
        emotion, confidence = ensemble_emotion(cnn_probs, deepface_probs)

        # Draw results
        label = f"{emotion.capitalize()} ({confidence:.2f})"

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow("Emotion Detection (CNN + DeepFace)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()