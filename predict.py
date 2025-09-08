import cv2
import mediapipe as mp
import joblib
import numpy as np

# Load trained model
clf = joblib.load("hand_sign_rf_model.pkl")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,  # one hand at a time
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    prediction_text = ""

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks on screen
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmark features
            row = []
            for lm in hand_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z])  # flatten x,y,z

            # Convert to numpy and reshape for model
            row = np.array(row).reshape(1, -1)

            # Predict with trained model
            pred = clf.predict(row)[0]

            # Get confidence (highest probability)
            proba = clf.predict_proba(row)[0]
            confidence = np.max(proba) * 100

            prediction_text = f"Predicted: {pred} ({confidence:.2f}%)"

            if confidence>50:
                cv2.putText(frame, prediction_text, (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Real-time Hand Sign Recognition", frame)

    # ESC to quit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
